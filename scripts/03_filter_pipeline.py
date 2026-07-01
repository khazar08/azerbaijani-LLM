import argparse
import json
from pathlib import Path

import torch
from datasketch import MinHash, MinHashLSH
from langdetect import detect, LangDetectException
from sacrebleu.metrics import CHRF
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

NLLB_MODEL        = "facebook/nllb-200-distilled-600M"
AZ_LANG           = "azj_Latn"
EN_LANG           = "eng_Latn"
MIN_LENGTH_RATIO  = 0.4
MAX_LENGTH_RATIO  = 2.5
CHRF_THRESHOLD    = 30.0
MINHASH_THRESHOLD = 0.85
MINHASH_PERMS     = 128
BATCH_SIZE        = 32


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]

def word_count(text):
    return len(text.split())

def minhash_for(text):
    m = MinHash(num_perm=MINHASH_PERMS)
    for s in {text[i:i+3] for i in range(len(text)-2)}:
        m.update(s.encode("utf-8"))
    return m

def back_translate(model, tok, device, texts):
    tok.src_lang = AZ_LANG
    forced_bos = tok.convert_tokens_to_ids(EN_LANG)
    inp = tok(texts, return_tensors="pt", padding=True,
               truncation=True, max_length=256).to(device)
    with torch.no_grad():
        out = model.generate(**inp, forced_bos_token_id=forced_bos, max_new_tokens=256)
    return tok.batch_decode(out, skip_special_tokens=True)


AZ_SPECIFIC_CHARS = set("əğ")  # ə/ğ appear in Az but not standard Turkish Latin

def is_azerbaijani(text: str) -> bool:
    """langdetect can't tell Az from Turkish; use detected lang + script heuristic."""
    try:
        lang = detect(text)
    except LangDetectException:
        lang = "unknown"
    # Accept if detected as az or tr (langdetect conflates them), AND text
    # contains at least one Azerbaijani-specific character OR is very short.
    if lang not in ("az", "tr"):
        return False
    if len(text.split()) <= 3:
        return True  # too short to judge by chars alone
    return any(c in AZ_SPECIFIC_CHARS for c in text)


def main(args):
    src = Path(args.input)
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(src)
    n_start = len(records)
    print(f"Starting with {n_start} records.\n")

    chrf_metric = CHRF()
    stats = {"lang_id": 0, "length_ratio": 0, "back_trans": 0, "dedup": 0}

    # 1. Language-ID (az-specific heuristic — langdetect conflates Az with tr)
    print("Filter 1: Language-ID (az/tr + ə/ğ heuristic)...")
    passed = []
    for r in tqdm(records):
        text = r["instruction"] + " " + r["response"]
        if is_azerbaijani(text):
            passed.append(r)
        else:
            stats["lang_id"] += 1
    records = passed
    print(f"  Dropped {stats['lang_id']} ({stats['lang_id']/n_start:.1%})")

    # 2. Length-ratio
    print("Filter 2: Length ratio...")
    passed = []
    for r in records:
        en_wc = max(1, word_count(r.get("_en_instruction","") + " " + r.get("_en_response","")))
        az_wc = max(1, word_count(r["instruction"] + " " + r["response"]))
        ratio = az_wc / en_wc
        if MIN_LENGTH_RATIO <= ratio <= MAX_LENGTH_RATIO:
            passed.append(r)
        else:
            stats["length_ratio"] += 1
    records = passed
    print(f"  Dropped {stats['length_ratio']} ({stats['length_ratio']/n_start:.1%})")

    # 3. Back-translation
    print(f"Filter 3: Back-translation (chrF >= {CHRF_THRESHOLD})...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  Loading {NLLB_MODEL} on {device}...")
    btok   = AutoTokenizer.from_pretrained(NLLB_MODEL)
    bmodel = AutoModelForSeq2SeqLM.from_pretrained(
        NLLB_MODEL,
        dtype=torch.float16 if device == "mps" else torch.float32,
    ).to(device)
    bmodel.eval()

    az_insts    = [r["instruction"] for r in records]
    en_originals = [r.get("_en_instruction","") for r in records]
    back_trans  = []
    for i in tqdm(range(0, len(az_insts), BATCH_SIZE)):
        back_trans.extend(back_translate(bmodel, btok, device, az_insts[i:i+BATCH_SIZE]))
    del bmodel

    passed = []
    for r, bt, ref in zip(records, back_trans, en_originals):
        if chrf_metric.sentence_score(bt, [ref]).score >= CHRF_THRESHOLD:
            passed.append(r)
        else:
            stats["back_trans"] += 1
    records = passed
    print(f"  Dropped {stats['back_trans']} ({stats['back_trans']/n_start:.1%})")

    # 4. Near-dedup
    print("Filter 4: Near-dedup (MinHash)...")
    lsh = MinHashLSH(threshold=MINHASH_THRESHOLD, num_perm=MINHASH_PERMS)
    passed = []
    for i, r in enumerate(tqdm(records)):
        m = minhash_for(r["instruction"])
        if not lsh.query(m):
            lsh.insert(f"r{i}", m)
            passed.append(r)
        else:
            stats["dedup"] += 1
    records = passed
    print(f"  Dropped {stats['dedup']} ({stats['dedup']/n_start:.1%})")

    # Report
    n_end = len(records)
    print(f"\n{'='*50}")
    print(f"Final: {n_end}/{n_start} kept ({n_end/n_start:.1%})")
    for stage, n in stats.items():
        print(f"  {stage:<20} {n:5d} dropped ({n/n_start:.1%})")
    print(f"{'='*50}\n")

    with open(dst, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({k:v for k,v in r.items() if not k.startswith("_")},
                               ensure_ascii=False) + "\n")
    print(f"Wrote {n_end} filtered records to {dst}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True)
    p.add_argument("--output", required=True)
    main(p.parse_args())
