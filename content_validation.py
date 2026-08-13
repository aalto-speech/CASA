"""Few-shot content validation: does the answer address the question?

Uses CASA's LLM backbone (base Qwen3.5-2B, LoRA OFF) as a few-shot judge. It sees only the exam
TASK and the ASR answer and replies with one word (good / average / bad); it is explicitly told not
to judge grammar, vocabulary, pronunciation or fluency, so no other assessment dimension leaks in.

This reproduces the content-validation experiment in the paper: run it on the clean test master and
on the attacked masters from build_content_attacks.py (the "unrelated question" variant pairs every
answer with a question about nuclear reactors), then compare the label distributions.

    python build_content_attacks.py                       # -> csv/attacks/*.csv
    python content_validation.py --master_csv csv/master_test_asr.csv                    # clean
    python content_validation.py --master_csv csv/attacks/master_test_asr_unrelated_question.csv

Generation is constrained to the three label tokens, so the judge cannot answer anything else.
"""

import argparse
import json
import os

import pandas as pd

HF_CACHE = os.environ.get("HF_CACHE_DIR") or None
LABELS = ("good", "average", "bad")


def clean_task(llm_input: str) -> str:
    return str(llm_input).replace("\\n", "\n").strip()


SYSTEM = (
    "You are an examiner for a spoken language exam. You are given the exam TASK (the question / "
    "prompt) and the candidate's spoken answer, transcribed by ASR (which may contain recognition "
    "errors). Judge ONLY how well the answer ADDRESSES the task — its topical relevance and task "
    "fulfilment. Do NOT judge grammar, vocabulary, pronunciation or fluency. Reply with exactly one "
    "word:\n"
    "  good = the answer clearly addresses the question / task and stays on topic;\n"
    "  average = on the general topic but incomplete, drifts, or only loosely addresses the task;\n"
    "  bad = does not address the question, is off-topic, or is non-responsive.")

# few-shot: (task_block, label) — ASR-style, generic but SANDI-shaped
FEWSHOT = [
    ("<TASK part=P1>\nQ1: Tell me about your hometown.\n"
     "A1: my home town is quite small it is near the sea and quiet i like it because everyone knows "
     "each other and there are nice beaches and cafes", "good"),
    ("<TASK part=P3>\nQ1: Should all university students do part-time jobs? What are the advantages "
     "and disadvantages, and your opinion?\n"
     "A1: yes part time jobs are good because you get money and experience but also i really like "
     "playing football on weekends with my friends and we watch movies", "average"),
    ("<TASK part=P4>\nQ1: Look at the diagram and explain how pollution causes acid rain.\n"
     "A1: on the weekend i usually meet my friends we go to the cinema or a restaurant and sometimes "
     "we play video games at home it is very fun", "bad"),
    ("<TASK part=P5>\nQ1: A reporter asks your opinion about healthy eating habits.\n"
     "A1: i think eating habits are important you should eat vegetables and fruit every day and not "
     "too much sugar and drink water it helps your health and energy", "good"),
]


def messages(task_block: str) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM}]
    for t, lab in FEWSHOT:
        msgs.append({"role": "user", "content": f"{t}\n\nRelevance:"})
        msgs.append({"role": "assistant", "content": lab})
    msgs.append({"role": "user", "content": f"{task_block}\n\nRelevance:"})
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master_csv", required=True,
                    help="master CSV with an llm_input column (clean or attacked)")
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B", help="judge LLM (base weights, no LoRA)")
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--out_csv", default="", help="optional per-row label dump")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    df = (pd.read_csv(args.master_csv)[["speaker_id", "part", "llm_input"]]
          .dropna().drop_duplicates(["speaker_id", "part"]).reset_index(drop=True))
    df["task_block"] = df["llm_input"].astype(str).map(clean_task)
    print(f"{len(df)} parts | {df.speaker_id.nunique()} speakers | judge={args.model} (base, LoRA off)",
          flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, cache_dir=HF_CACHE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, cache_dir=HF_CACHE, dtype=torch.bfloat16, attn_implementation="sdpa").eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    # constrain generation to exactly the label vocabulary
    seq = {v: tuple(tok(v, add_special_tokens=False)["input_ids"]) for v in LABELS}
    rev = {ids: v for v, ids in seq.items()}
    root: dict = {}
    for ids in seq.values():
        node = root
        for t in ids:
            node = node.setdefault(t, {})
        node[None] = {}
    eos = int(tok.eos_token_id)
    max_new = max(len(i) for i in seq.values()) + 1

    def allowed(_batch_id, generated):
        node = root
        for t in generated.tolist():
            t = int(t)
            if t not in node:
                return [eos]
            node = node[t]
        return [int(t) for t in node if t is not None] + ([eos] if None in node else []) or [eos]

    @torch.inference_mode()
    def judge(convs):
        res = []
        for s in range(0, len(convs), args.batch_size):
            chunk = convs[s:s + args.batch_size]
            txt = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False) for c in chunk]
            enc = tok(txt, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id, eos_token_id=eos,
                                 prefix_allowed_tokens_fn=allowed)
            prompt_len = enc["input_ids"].shape[1]      # left-padded: same for every row
            for row in gen:
                ids = [int(t) for t in row.tolist()[prompt_len:]]
                while ids and ids[-1] in (tok.pad_token_id, eos):
                    ids.pop()
                res.append(rev.get(tuple(ids), "good"))
            print(f"  ...{min(s + args.batch_size, len(convs))}/{len(convs)}", flush=True)
        return res

    df["relevance"] = judge([messages(t) for t in df["task_block"].tolist()])
    counts = df["relevance"].value_counts().to_dict()
    n = len(df)
    print("\n=== content validation ===")
    print(f"source: {args.master_csv}  (n={n})")
    for lab in LABELS:
        c = counts.get(lab, 0)
        print(f"  {lab:8} {c:5d}  ({100 * c / n:.1f}%)")
    print("  bad = off-topic, average = partially addresses, good = on-topic")
    if args.out_csv:
        df.drop(columns=["task_block"]).to_csv(args.out_csv, index=False)
        print("wrote", args.out_csv)


if __name__ == "__main__":
    main()
