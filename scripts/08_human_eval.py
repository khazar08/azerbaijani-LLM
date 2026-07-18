import argparse
import json
import random
from pathlib import Path
from sklearn.metrics import cohen_kappa_score

DIMS = ["instruction_adherence", "fluency", "factuality", "completeness"]
RUBRIC = {
    "instruction_adherence": "Did it do what was asked? (1=ignored, 5=perfect)",
    "fluency":               "Natural Azerbaijani? (1=broken, 5=native-quality)",
    "factuality":            "Facts correct? (1=wrong, 5=all correct / N/A)",
    "completeness":          "Complete? (1=empty, 5=appropriately detailed)",
}
FAILURE_MODES = [
    "turkish_drift", "calque", "wrong_register", "code_switch",
    "factual_error", "ignored_instruction", "incomplete", "other",
]


def load_jsonl(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def score_response(label: str, text: str) -> dict:
    print(f"\n--- Response {label} ---")
    print(text[:800])
    scores = {}
    for dim, desc in RUBRIC.items():
        while True:
            v = input(f"  {desc} [1-5]: ").strip()
            if v in ("1", "2", "3", "4", "5"):
                scores[dim] = int(v)
                break
            print("  Enter 1, 2, 3, 4, or 5.")
    return scores


def avg(scores: dict) -> float:
    return sum(scores[d] for d in DIMS) / len(DIMS)


def verdict(sl: dict, sr: dict, left_is_a: bool) -> str:
    d = avg(sl) - avg(sr)
    if abs(d) <= 0.3:
        return "tie"
    left_wins = d > 0
    if left_is_a:
        return "A_wins" if left_wins else "B_wins"
    return "B_wins" if left_wins else "A_wins"


def main(args):
    random.seed(42)
    rows = load_jsonl(args.eval_file)
    oa   = {r["id"]: r["response"] for r in load_jsonl(args.outputs_a)}
    ob   = {r["id"]: r["response"] for r in load_jsonl(args.outputs_b)}

    judge_v: dict[str, str] = {}
    if args.judge_file and Path(args.judge_file).exists():
        jd = json.loads(Path(args.judge_file).read_text())
        judge_v = {it["id"]: it["verdict"] for it in jd.get("items", [])}

    candidates = [r for r in rows if r["id"] in oa and r["id"] in ob]
    random.shuffle(candidates)
    candidates = candidates[: args.n]

    print(f"\nStarting BLIND human eval on {len(candidates)} items.")
    
    results = []
    wa, wb, ties = 0, 0, 0
    ftally: dict[str, int] = {m: 0 for m in FAILURE_MODES}
    hv_list: list[str] = []
    jv_list: list[str] = []

    for i, row in enumerate(candidates, 1):
        rid = row["id"]
        left_is_a = random.random() < 0.5
        left  = oa[rid] if left_is_a else ob[rid]
        right = ob[rid] if left_is_a else oa[rid]

        print(f"\n{'='*60}")
        print(f"Item {i}/{len(candidates)}  |  category={row.get('category','?')}  |  id={rid}")
        print(f"INSTRUCTION: {row['instruction']}\n")

        sl = score_response("LEFT",  left)
        sr = score_response("RIGHT", right)

        v = verdict(sl, sr, left_is_a)
        print(f"\n  => Verdict: {v}")

        print("  Which side had failure modes? (left / right / both / none):")
        side = input("  > ").strip().lower()
        fm: list[str] = []
        sides_to_ask = []
        if side in ("left", "both"):
            sides_to_ask.append("left")
        if side in ("right", "both"):
            sides_to_ask.append("right")
        for s in sides_to_ask:
            raw = input(f"  Failure modes for {s} ({', '.join(FAILURE_MODES)}): ").strip()
            fm += [m.strip() for m in raw.split(",") if m.strip() in FAILURE_MODES]
        for m in fm:
            ftally[m] += 1

        results.append({
            "id": rid, "category": row.get("category", ""),
            "left_is_a": left_is_a,
            "scores_left": sl, "scores_right": sr,
            "verdict": v, "failure_modes": fm,
        })

        if v == "A_wins":
            wa += 1
        elif v == "B_wins":
            wb += 1
        else:
            ties += 1

        hv_list.append(v)
        if rid in judge_v:
            jv_list.append(judge_v[rid])

    n = len(results)
    print(f"\n{'='*60}")
    print(f"FINAL  n={n}")
    print(f"  {args.system_a}: {wa} wins ({wa/n:.1%})")
    print(f"  {args.system_b}: {wb} wins ({wb/n:.1%})")
    print(f"  Ties:          {ties} ({ties/n:.1%})")
    print("\nFailure mode tally:")
    for m, c in sorted(ftally.items(), key=lambda x: -x[1]):
        if c:
            print(f"  {m:<25} {c}")

    kappa = float("nan")
    if len(jv_list) >= 10:
        lmap = {"A_wins": 0, "tie": 1, "B_wins": 2}
        paired_h = [lmap[hv_list[i]] for i, r in enumerate(results) if r["id"] in judge_v]
        paired_j = [lmap[judge_v[r["id"]]] for r in results if r["id"] in judge_v]
        if len(set(paired_h)) > 1 and len(set(paired_j)) > 1:
            kappa = cohen_kappa_score(paired_h, paired_j)
            print(f"\nJudge-human Cohen's kappa: {kappa:.3f}")

    out = {
        "system_a": args.system_a, "system_b": args.system_b,
        "n": n, "win_a": wa, "win_b": wb, "ties": ties,
        "judge_human_kappa": kappa,
        "failure_tally": ftally, "items": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--eval_file",   required=True)
    p.add_argument("--outputs_a",   required=True)
    p.add_argument("--outputs_b",   required=True)
    p.add_argument("--system_a",    default="A")
    p.add_argument("--system_b",    default="B")
    p.add_argument("--judge_file",  default=None)
    p.add_argument("--output",      default="results/human_eval.json")
    p.add_argument("--n",           type=int, default=50)
    main(p.parse_args())
