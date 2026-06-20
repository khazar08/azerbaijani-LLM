"""
Phase 4 - Reference-based pairwise evaluation using chrF++ (sacrebleu).

For each eval item, computes chrF++ of system A and system B against the gold
reference, then calls the winner. Also reports per-category breakdown.

No API key required — fully local.

Usage:
    python scripts/07_eval_judge.py \
        --outputs_a results/outputs_base_few_shot.jsonl \
        --outputs_b results/outputs_finetune.jsonl \
        --system_a  base_few_shot \
        --system_b  finetune \
        --output    results/judge_eval.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from sacrebleu.metrics import CHRF


def load_jsonl(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def chrf_score(hyp: str, ref: str) -> float:
    metric = CHRF(word_order=2)
    return metric.sentence_score(hyp, [ref]).score


def verdict(sa: float, sb: float, margin: float = 1.0) -> str:
    d = sa - sb
    if d > margin:
        return "A_wins"
    elif d < -margin:
        return "B_wins"
    return "tie"


def main(args):
    oa = {r["id"]: r for r in load_jsonl(args.outputs_a)}
    ob = {r["id"]: r for r in load_jsonl(args.outputs_b)}

    ids = sorted(set(oa) & set(ob))
    wa, wb, ties = 0, 0, 0
    items = []
    by_cat: dict[str, dict] = defaultdict(lambda: {"A_wins": 0, "B_wins": 0, "tie": 0})

    chrf_a_scores, chrf_b_scores = [], []

    for rid in ids:
        ra = oa[rid]
        rb = ob[rid]
        ref = ra.get("reference", "") or rb.get("reference", "")
        if not ref:
            continue

        sa = chrf_score(ra["response"], ref)
        sb = chrf_score(rb["response"], ref)
        v = verdict(sa, sb, margin=args.margin)

        chrf_a_scores.append(sa)
        chrf_b_scores.append(sb)

        cat = ra.get("category", "unknown")
        by_cat[cat][v] += 1

        if v == "A_wins":
            wa += 1
        elif v == "B_wins":
            wb += 1
        else:
            ties += 1

        items.append({
            "id": rid,
            "category": cat,
            "verdict": v,
            "chrf_a": round(sa, 2),
            "chrf_b": round(sb, 2),
            "delta": round(sa - sb, 2),
        })
        print(f"  {rid:<30} A={sa:5.1f}  B={sb:5.1f}  => {v}")

    n = wa + wb + ties
    avg_a = sum(chrf_a_scores) / max(len(chrf_a_scores), 1)
    avg_b = sum(chrf_b_scores) / max(len(chrf_b_scores), 1)

    print(f"\n{'='*55}")
    print(f"Margin threshold: {args.margin} chrF++ points")
    print(f"{'System':<25} {'avg chrF++':>10}  {'wins':>6}  {'%':>6}")
    print(f"{args.system_a:<25} {avg_a:>10.2f}  {wa:>6}  {wa/n:.1%}")
    print(f"{args.system_b:<25} {avg_b:>10.2f}  {wb:>6}  {wb/n:.1%}")
    print(f"{'Ties':<25} {'':>10}  {ties:>6}  {ties/n:.1%}")

    print(f"\nPer-category breakdown:")
    for cat, counts in sorted(by_cat.items()):
        total = sum(counts.values())
        print(f"  {cat:<25} A={counts['A_wins']:>2}  B={counts['B_wins']:>2}  tie={counts['tie']:>2}  (n={total})")

    output = {
        "system_a": args.system_a,
        "system_b": args.system_b,
        "metric": "chrF++ (word_order=2)",
        "margin": args.margin,
        "n": n,
        "win_a": wa,
        "win_b": wb,
        "ties": ties,
        "avg_chrf_a": round(avg_a, 4),
        "avg_chrf_b": round(avg_b, 4),
        "by_category": {k: dict(v) for k, v in by_cat.items()},
        "items": items,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--outputs_a", required=True)
    p.add_argument("--outputs_b", required=True)
    p.add_argument("--system_a",  default="A")
    p.add_argument("--system_b",  default="B")
    p.add_argument("--margin",    type=float, default=1.0,
                   help="chrF++ point gap to call a win vs tie (default 1.0)")
    p.add_argument("--output",    default="results/judge_eval.json")
    main(p.parse_args())
