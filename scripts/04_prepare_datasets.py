"""
Phase 1 — Merge data tiers, apply chat template, decontaminate, and split.

Reads:
  - Tier 1: filtered machine-translated JSONL files (glob pattern)
  - Tier 3: hand-authored native data JSONL
  - Frozen eval:  data/eval/test.jsonl  (decontamination check only — NOT used for training)

Writes:
  - data/processed/train.jsonl
  - data/processed/val.jsonl
  - data/processed/decontam_report.json

Usage:
    python scripts/04_prepare_datasets.py \
        --tier1_glob "data/raw/translated/*_filtered.jsonl" \
        --tier3      data/raw/native/native_train.jsonl \
        --base_model google/gemma-2-2b
"""

import argparse
import glob
import hashlib
import json
import random
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

VAL_FRACTION = 0.05
DECONTAM_THRESHOLD = 0.8   # Jaccard similarity above which a train example is flagged
SEED = 42


def load_jsonl(path: str) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def ngram_shingles(text: str, n: int = 5) -> set[str]:
    return {text[i : i + n] for i in range(max(0, len(text) - n + 1))}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    denom = len(a | b)
    return len(a & b) / denom if denom else 0.0


def decontaminate(train_records: list[dict], test_records: list[dict]) -> tuple[list[dict], list[dict]]:
    test_shingles = [ngram_shingles(r["instruction"].lower()) for r in test_records]
    clean, flagged = [], []
    for r in tqdm(train_records, desc="Decontaminating"):
        q = ngram_shingles(r["instruction"].lower())
        max_sim = max((jaccard(q, ts) for ts in test_shingles), default=0.0)
        if max_sim >= DECONTAM_THRESHOLD:
            flagged.append({**r, "_max_test_sim": max_sim})
        else:
            clean.append(r)
    return clean, flagged


def apply_chat_template(records: list[dict], tokenizer) -> list[dict]:
    out = []
    for r in records:
        msgs = [
            {"role": "user",      "content": r["instruction"]},
            {"role": "assistant", "content": r["response"]},
        ]
        text = tokenizer.apply_chat_template(msgs, tokenize=False)
        out.append({**r, "text": text})
    return out


def main(args):
    random.seed(SEED)

    # --- Load tiers ---
    all_records = []
    for path in sorted(glob.glob(args.tier1_glob)):
        rows = load_jsonl(path)
        for r in rows:
            r["tier"] = "translated"
        all_records.extend(rows)
        print(f"  Tier 1 — {path}: {len(rows)} rows")

    if args.tier3 and Path(args.tier3).exists():
        native = load_jsonl(args.tier3)
        for r in native:
            r["tier"] = "native"
        all_records.extend(native)
        print(f"  Tier 3 — {args.tier3}: {len(native)} rows (native)")
    else:
        print("  Tier 3 not found — only translated data will be used.")

    print(f"\nTotal before decontam: {len(all_records)}")

    # --- Decontaminate against frozen test ---
    test_records = load_jsonl("data/eval/test.jsonl") if Path("data/eval/test.jsonl").exists() else []
    if test_records:
        clean, flagged = decontaminate(all_records, test_records)
        report = {"n_train_before": len(all_records), "n_flagged": len(flagged), "flagged": flagged}
        Path("data/processed/decontam_report.json").parent.mkdir(parents=True, exist_ok=True)
        Path("data/processed/decontam_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2)
        )
        print(f"Decontam: {len(flagged)} flagged and removed (see data/processed/decontam_report.json)")
        all_records = clean
    else:
        print("WARNING: data/eval/test.jsonl not found — skipping decontamination!")

    # --- Apply chat template ---
    print(f"\nLoading tokenizer ({args.base_model})…")
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    all_records = apply_chat_template(all_records, tok)

    # --- Train / val split ---
    random.shuffle(all_records)
    n_val = max(50, int(len(all_records) * VAL_FRACTION))
    val_records = all_records[:n_val]
    train_records = all_records[n_val:]

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    for name, rows in [("train", train_records), ("val", val_records)]:
        out_path = f"data/processed/{name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(rows)} records → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier1_glob", default="data/raw/translated/*_filtered.jsonl")
    parser.add_argument("--tier3", default="data/raw/native/native_train.jsonl")
    parser.add_argument("--base_model", default="google/gemma-2-2b")
    main(parser.parse_args())
