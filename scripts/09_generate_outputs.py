import argparse
import json
from pathlib import Path
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

FEW_SHOT_PAIRS = [
    ("Azərbaycanın paytaxtı hansı şəhərdir?", "Azərbaycanın paytaxtı Bakıdır."),
    ("2 + 2 neçədir?", "2 + 2 = 4-dür."),
    ("Ən böyük planet hansıdır?", "Günəş sistemindəki ən böyük planet Yupiterdir."),
]
EVAL_FILE = "data/eval/test.jsonl"
OUT_DIR   = Path("results")


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_eval() -> list[dict]:
    return [json.loads(l) for l in Path(EVAL_FILE).read_text().splitlines() if l.strip()]


def make_model(model_id: str, adapter: str | None = None, device: str = "mps"):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tok.pad_token = tok.eos_token
    dtype = torch.float16 if device != "cpu" else torch.float32
    m = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map={"": device},
        attn_implementation="eager",
    )
    if adapter and Path(adapter).exists():
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
    return m.eval(), tok


def gen(model, tok, instruction: str, few_shot: bool = False) -> str:
    if few_shot:
        messages = []
        for q, a in FEW_SHOT_PAIRS:
            messages.append({"role": "user",      "content": q})
            messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": instruction})
    else:
        messages = [{"role": "user", "content": instruction}]

    enc = tok.apply_chat_template(
        messages, return_tensors="pt", tokenize=True, add_generation_prompt=True
    )
    inp = (enc.input_ids if hasattr(enc, "input_ids") else enc).to(model.device)
    mask = torch.ones_like(inp)
    with torch.no_grad():
        out = model.generate(
            inp,
            attention_mask=mask,
            max_new_tokens=400,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][inp.shape[-1]:], skip_special_tokens=True).strip()


def generate_all(model, tok, rows: list[dict], few_shot: bool = False) -> list[dict]:
    out = []
    for row in tqdm(rows):
        out.append({
            "id": row["id"],
            "category": row.get("category", ""),
            "instruction": row["instruction"],
            "response": gen(model, tok, row["instruction"], few_shot=few_shot),
            "reference": row.get("response", ""),
        })
    return out


def save_outputs(rows: list[dict], name: str):
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / f"outputs_{name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(rows)} -> {path}")


def clear_cache(device: str):
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def main(args):
    device = detect_device()
    print(f"Device: {device}")
    eval_rows = load_eval()
    print(f"Generating outputs for {len(eval_rows)} eval items.\n")

    model, tok = make_model(args.base_model, device=device)
    save_outputs(generate_all(model, tok, eval_rows, few_shot=False), "base_zero_shot")

    print("--- base_few_shot ---")
    save_outputs(generate_all(model, tok, eval_rows, few_shot=True), "base_few_shot")
    del model
    clear_cache(device)

    if args.adapter_dir and Path(args.adapter_dir).exists():
        print("\n--- finetune ---")
        model, tok = make_model(args.base_model, args.adapter_dir, device=device)
        save_outputs(generate_all(model, tok, eval_rows, few_shot=False), "finetune")
        del model
        clear_cache(device)
    else:
        print(f"\nAdapter dir not found ({args.adapter_dir}) — skipping finetune outputs.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base_model",  default="Qwen/Qwen2.5-3B")
    p.add_argument("--adapter_dir", default="adapters/az-instruct-lora")
    main(p.parse_args())
