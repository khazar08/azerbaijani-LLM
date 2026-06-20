"""
Phase 4 — Automatic evaluation: Belebele (azj_Latn), SIB-200 (azj_Latn),
and in-house translation chrF++ on the frozen eval set's translation category.

Systems: base_zero_shot | base_few_shot | finetune

Usage:
    python scripts/06_eval_automatic.py \
        --base_model Qwen/Qwen2.5-3B \
        --adapter_dir adapters/az-instruct-lora \
        --output results/auto_eval.json
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from sacrebleu.metrics import CHRF
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

SIB_LABELS = [
    "science/technology", "travel", "politics", "sports",
    "health", "entertainment", "geography",
]
FEW_SHOT_PAIRS = [
    ("Azərbaycanın paytaxtı hansı şəhərdir?", "Azərbaycanın paytaxtı Bakıdır."),
    ("2 + 2 neçədir?", "2 + 2 = 4-dür."),
    ("Ən böyük planet hansıdır?", "Yupiterdir."),
]


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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


def gen(model, tok, messages: list[dict], max_new: int = 128) -> str:
    enc = tok.apply_chat_template(
        messages, return_tensors="pt", tokenize=True, add_generation_prompt=True
    )
    inp = (enc.input_ids if hasattr(enc, "input_ids") else enc).to(model.device)
    mask = torch.ones_like(inp)
    with torch.no_grad():
        out = model.generate(
            inp,
            attention_mask=mask,
            max_new_tokens=max_new,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][inp.shape[-1]:], skip_special_tokens=True).strip()


def build_messages(instruction: str, few_shot: bool = False) -> list[dict]:
    msgs = []
    if few_shot:
        for q, a in FEW_SHOT_PAIRS:
            msgs.append({"role": "user",      "content": q})
            msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": instruction})
    return msgs


def eval_belebele(model, tok, few_shot: bool = False) -> float:
    try:
        ds = load_dataset("facebook/belebele", "azj_Latn", split="test")
    except Exception as e:
        print(f"  Belebele load failed: {e}")
        return 0.0
    correct = 0
    for row in tqdm(ds, desc="Belebele"):
        choices = [row[f"mc_answer{i}"] for i in range(1, 5)]
        gold = int(row["correct_answer_num"]) - 1
        instruction = (
            f"Mətn: {row['flores_passage']}\nSual: {row['question']}\n"
            + "\n".join(f"{i+1}. {c}" for i, c in enumerate(choices))
            + "\nDüzgün cavabın nömrəsini yazın (1, 2, 3 və ya 4):"
        )
        msgs = build_messages(instruction, few_shot)
        resp = gen(model, tok, msgs, 10)
        pred = next((str(i+1) for i in range(4) if str(i+1) in resp[:10]), "-")
        if pred == str(gold + 1):
            correct += 1
    acc = correct / max(len(ds), 1)
    print(f"  Belebele acc={acc:.4f} ({correct}/{len(ds)})")
    return acc


def eval_sib(model, tok) -> dict:
    try:
        ds = load_dataset("Davlan/sib200", "azj_Latn", split="test")
    except Exception as e:
        print(f"  SIB-200 load failed: {e}")
        return {"accuracy": 0.0, "macro_f1": 0.0}
    # dataset uses string category labels, not integer indices
    label_to_idx = {l: str(i) for i, l in enumerate(SIB_LABELS)}
    yt, yp = [], []
    for row in tqdm(ds, desc="SIB-200"):
        choices_str = "\n".join(f"{i}: {l}" for i, l in enumerate(SIB_LABELS))
        instruction = (
            f"Mətn: {row['text'][:300]}\nMövzular:\n{choices_str}\n"
            "Nömrəni yazın (0-6):"
        )
        msgs = build_messages(instruction)
        resp = gen(model, tok, msgs, 5)
        gold_idx = label_to_idx.get(row["category"], "0")
        yt.append(gold_idx)
        yp.append(next((str(i) for i in range(len(SIB_LABELS)) if str(i) in resp[:10]), "0"))
    acc = accuracy_score(yt, yp)
    f1 = f1_score(yt, yp, average="macro", zero_division=0)
    print(f"  SIB-200 acc={acc:.4f} f1={f1:.4f}")
    return {"accuracy": acc, "macro_f1": f1}


def eval_translation_chrf(outputs_file: str) -> float:
    """chrF++ on the translation category of the frozen eval set, using pre-generated outputs."""
    p = Path(outputs_file)
    if not p.exists():
        print(f"  Translation chrF++: outputs file not found ({p})")
        return 0.0
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    metric = CHRF(word_order=2)
    hyps, refs = [], []
    for r in rows:
        if r.get("category") == "translation" and r.get("reference"):
            hyps.append(r["response"])
            refs.append(r["reference"])
    if not hyps:
        print("  Translation chrF++: no translation items found in outputs")
        return 0.0
    score = metric.corpus_score(hyps, [refs]).score
    print(f"  Translation chrF++ (n={len(hyps)}) = {score:.2f}")
    return score


OUTPUTS_MAP = {
    "base_zero_shot": "results/outputs_base_zero_shot.jsonl",
    "base_few_shot":  "results/outputs_base_few_shot.jsonl",
    "finetune":       "results/outputs_finetune.jsonl",
}


def run_system(name: str, model, tok, few_shot: bool = False) -> dict:
    print(f"\n=== {name} ===")
    return {
        "belebele_acc":    eval_belebele(model, tok, few_shot),
        "sib200":          eval_sib(model, tok),
        "translation_chrf": eval_translation_chrf(OUTPUTS_MAP[name]),
    }


def clear_cache(device: str):
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def main(args):
    device = detect_device()
    print(f"Device: {device}")
    results = {}
    model, tok = make_model(args.base_model, device=device)
    results["base_zero_shot"] = run_system("base_zero_shot", model, tok, False)
    results["base_few_shot"]  = run_system("base_few_shot",  model, tok, True)
    del model
    clear_cache(device)

    if args.adapter_dir and Path(args.adapter_dir).exists():
        model, tok = make_model(args.base_model, args.adapter_dir, device=device)
        results["finetune"] = run_system("finetune", model, tok)
        del model
        clear_cache(device)

    print("\n=== RESULTS MATRIX ===")
    print(f"{'System':<22}{'Belebele':>10}{'SIB-Acc':>10}{'SIB-F1':>10}{'Trans-chrF':>12}")
    print("-" * 66)
    for s, r in results.items():
        sib = r.get("sib200", {})
        print(f"{s:<22}{r.get('belebele_acc',0):>10.4f}{sib.get('accuracy',0):>10.4f}"
              f"{sib.get('macro_f1',0):>10.4f}{r.get('translation_chrf',0):>12.2f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base_model",  default="Qwen/Qwen2.5-3B")
    p.add_argument("--adapter_dir", default="adapters/az-instruct-lora")
    p.add_argument("--output",      default="results/auto_eval.json")
    main(p.parse_args())
