import argparse
import json
import time
from pathlib import Path
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

NLLB_MODEL = "facebook/nllb-200-distilled-600M"
SRC_LANG = "eng_Latn"
TGT_LANG = "azj_Latn"
BATCH_SIZE = 16
MAX_SRC_LEN = 256
MAX_NEW_TOKENS = 300


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def translate_batch(model, tok, texts, device):
    tok.src_lang = SRC_LANG
    inputs = tok(texts, return_tensors="pt", padding=True,
                 truncation=True, max_length=MAX_SRC_LEN).to(device)
    forced_bos = tok.convert_tokens_to_ids(TGT_LANG)
    with torch.no_grad():
        out = model.generate(**inputs, forced_bos_token_id=forced_bos,
                             max_new_tokens=MAX_NEW_TOKENS)
    return tok.batch_decode(out, skip_special_tokens=True)


def main(args):
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(args.input)
    # prefer shorter rows (cleaner translation)
    records = [r for r in records if r.get("instruction") and r.get("response")]
    records.sort(key=lambda r: len(r["instruction"]) + len(r.get("response", "")))
    if args.max_rows:
        records = records[: args.max_rows]
    print(f"Translating {len(records)} records...")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading {NLLB_MODEL}...")
    tok = AutoTokenizer.from_pretrained(NLLB_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        NLLB_MODEL, dtype=torch.float16 if device == "mps" else torch.float32
    ).to(device)
    model.eval()

    instructions = [r["instruction"] for r in records]
    responses    = [r.get("response") or r.get("output") or "" for r in records]

    az_instructions = []
    for i in tqdm(range(0, len(instructions), BATCH_SIZE)):
        az_instructions.extend(
            translate_batch(model, tok, instructions[i : i + BATCH_SIZE], device)
        )

    az_responses = []
    for i in tqdm(range(0, len(responses), BATCH_SIZE)):
        az_responses.extend(
            translate_batch(model, tok, responses[i : i + BATCH_SIZE], device)
        )

    with open(dst, "w", encoding="utf-8") as f:
        for rec, az_inst, az_resp, en_inst, en_resp in zip(
            records, az_instructions, az_responses, instructions, responses
        ):
            out = {
                "id": rec.get("id", ""),
                "source": Path(args.input).stem,
                "instruction": az_inst,
                "response": az_resp,
                "_en_instruction": en_inst,
                "_en_response": en_resp,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} translated pairs to {dst}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input",    required=True)
    p.add_argument("--output",   required=True)
    p.add_argument("--max_rows", type=int, default=None)
    main(p.parse_args())
