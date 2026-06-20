"""
Phase 2 — Tokenizer fertility analysis.

Measures tokens-per-word for candidate base models on the same Azerbaijani text
sample and contrasts with English. Run before committing to a base model.

Usage:
    python scripts/01_fertility_analysis.py

Outputs a bar chart to results/fertility.png and prints a summary table.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

CANDIDATE_MODELS = [
    "google/gemma-2-2b",
    "Qwen/Qwen2.5-3B",
    "microsoft/Phi-3-mini-4k-instruct",
    "mistralai/Mistral-7B-v0.3",
]

AZ_SAMPLE_SIZE = 500
EN_SAMPLE_SIZE = 500


def fertility(tokenizer, texts: list[str]) -> float:
    ratios = []
    for t in texts:
        words = t.split()
        if not words:
            continue
        n_tok = len(tokenizer(t, add_special_tokens=False)["input_ids"])
        ratios.append(n_tok / len(words))
    return float(np.mean(ratios))


def load_az_texts(n: int) -> list[str]:
    print("Loading Azerbaijani Wikipedia sample…")
    ds = load_dataset("wikipedia", "20231101.az", split="train", streaming=True, trust_remote_code=True)
    texts = []
    for row in ds:
        texts.append(row["text"][:500])  # first 500 chars per article
        if len(texts) >= n:
            break
    return texts


def load_en_texts(n: int) -> list[str]:
    print("Loading English Wikipedia sample…")
    ds = load_dataset("wikipedia", "20231101.en", split="train", streaming=True, trust_remote_code=True)
    texts = []
    for row in ds:
        texts.append(row["text"][:500])
        if len(texts) >= n:
            break
    return texts


def main(args):
    az_texts = load_az_texts(AZ_SAMPLE_SIZE)
    en_texts = load_en_texts(EN_SAMPLE_SIZE)

    results = {}
    for model_id in args.models:
        print(f"Tokenizing with {model_id}…")
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        az_f = fertility(tok, az_texts)
        en_f = fertility(tok, en_texts)
        results[model_id] = {"az": az_f, "en": en_f, "ratio": az_f / en_f}
        print(f"  az={az_f:.3f}  en={en_f:.3f}  az/en={az_f/en_f:.2f}x")

    # Table
    print("\n--- Fertility Summary ---")
    print(f"{'Model':<45} {'Az':>6} {'En':>6} {'Az/En':>7}")
    print("-" * 70)
    best = min(results, key=lambda m: results[m]["ratio"])
    for m, v in sorted(results.items(), key=lambda kv: kv[1]["ratio"]):
        marker = " ← recommended" if m == best else ""
        print(f"{m:<45} {v['az']:>6.3f} {v['en']:>6.3f} {v['ratio']:>6.2f}x{marker}")

    # Plot
    Path("results").mkdir(exist_ok=True)
    short_names = [m.split("/")[-1] for m in results]
    az_vals = [results[m]["az"] for m in results]
    en_vals = [results[m]["en"] for m in results]
    x = np.arange(len(short_names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, az_vals, w, label="Azerbaijani", color="#E63946")
    ax.bar(x + w / 2, en_vals, w, label="English", color="#457B9D")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=20, ha="right")
    ax.set_ylabel("Tokens per word (lower = more efficient)")
    ax.set_title("Tokenizer fertility: Azerbaijani vs. English")
    ax.legend()
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    out = Path("results/fertility.png")
    plt.savefig(out, dpi=150)
    print(f"\nChart saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=CANDIDATE_MODELS)
    main(parser.parse_args())
