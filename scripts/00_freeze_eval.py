import hashlib
import json
import sys
from pathlib import Path

EVAL_FILE = Path("data/eval/test.jsonl")
CHECKSUM_FILE = Path("data/eval/checksum.sha256")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_records(path: Path) -> list[dict]:
    records = []
    required = {"id", "category", "instruction", "response"}
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            sys.exit(f"Line {i}: invalid JSON — {e}")
        missing = required - rec.keys()
        if missing:
            sys.exit(f"Line {i}: missing fields {missing}")
        records.append(rec)
    return records


def main():
    if not EVAL_FILE.exists():
        sys.exit(
            f"{EVAL_FILE} not found.\n"
            "Write your native eval examples there first (see data/eval/taxonomy.json).\n"
            "Each line: {\"id\": \"...\", \"category\": \"...\", \"instruction\": \"...\", \"response\": \"...\"}"
        )

    if CHECKSUM_FILE.exists():
        stored = CHECKSUM_FILE.read_text().strip()
        current = sha256_file(EVAL_FILE)
        if stored != current:
            sys.exit(
                "CHECKSUM MISMATCH — test set has been modified since it was frozen!\n"
                f"Stored:  {stored}\n"
                f"Current: {current}\n"
                "If this was intentional, delete data/eval/checksum.sha256 and re-run."
            )
        print(f"Eval set already frozen and unchanged. SHA-256: {stored}")
        return

    records = validate_records(EVAL_FILE)

    from collections import Counter
    cats = Counter(r["category"] for r in records)
    print(f"Validated {len(records)} eval records across {len(cats)} categories:")
    for cat, n in sorted(cats.items()):
        print(f"  {cat:25s} {n:3d}")

    checksum = sha256_file(EVAL_FILE)
    CHECKSUM_FILE.write_text(checksum + "\n")
    print(f"\nFrozen. SHA-256 written to {CHECKSUM_FILE}: {checksum}")

if __name__ == "__main__":
    main()
