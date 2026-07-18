import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from trl import SFTConfig, SFTTrainer


def load_config(path: str = "configs/train_config.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    for ov in overrides:
        key_path, _, val_str = ov.partition("=")
        keys = key_path.strip().split(".")
        node = cfg
        for k in keys[:-1]:
            node = node[k]
        try:
            val = int(val_str)
        except ValueError:
            try:
                val = float(val_str)
            except ValueError:
                val = val_str.lower() == "true" if val_str.lower() in ("true", "false") else val_str
        node[keys[-1]] = val
    return cfg


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_ds(path: str) -> Dataset:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    return Dataset.from_list(rows)


def qualitative_check(model, tokenizer, prompts: list[str], device: str):
    model.eval()
    print("\n--- Qualitative spot-check ---")
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        enc = tokenizer.apply_chat_template(
            msgs, return_tensors="pt", tokenize=True, add_generation_prompt=True
        )
        inp = (enc.input_ids if hasattr(enc, "input_ids") else enc).to(device)
        with torch.no_grad():
            out = model.generate(inp, max_new_tokens=200, temperature=0.7, do_sample=True)
        decoded = tokenizer.decode(out[0][inp.shape[-1]:], skip_special_tokens=True)
        print(f"PROMPT: {p}")
        print(f"RESPONSE: {decoded}\n")
    model.train()


SPOT_CHECK_PROMPTS = [
    "Azərbaycan paytaxtı hansı şəhərdir?",
    "Novruz bayramı nə vaxt keçirilir?",
    "Günəş sistemindəki ən böyük planet hansıdır?",
]


def main(args):
    cfg = load_config(args.config)
    if args.overrides:
        cfg = apply_overrides(cfg, args.overrides)

    run_name = args.run_name or "default"
    cfg["training"]["output_dir"] = str(Path(cfg["training"]["output_dir"]) / run_name)
    Path(cfg["training"]["output_dir"]).mkdir(parents=True, exist_ok=True)

    seed = cfg["training"]["seed"]
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    device = detect_device()
    print(f"Run: {run_name}")
    print(f"Base model: {cfg['base_model']}")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if device == "cuda":
        # Full QLoRA: 4-bit NF4 quantization
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=cfg["bnb"]["load_in_4bit"],
            bnb_4bit_quant_type=cfg["bnb"]["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=cfg["bnb"]["bnb_4bit_use_double_quant"],
            bnb_4bit_compute_dtype=getattr(torch, cfg["bnb"]["bnb_4bit_compute_dtype"]),
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg["base_model"],
            quantization_config=bnb_cfg,
            device_map="auto",
            attn_implementation="eager",
        )
        use_bf16 = cfg["training"].get("bf16", False)
        use_fp16 = False
    else:
        # MPS / CPU: float16 LoRA (no 4-bit quantization support)
        dtype = torch.float16 if device == "mps" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            cfg["base_model"],
            dtype=dtype,
            device_map={"": device},
            attn_implementation="eager",
        )
        use_bf16 = False
        use_fp16 = device == "mps"  # fp16 on MPS

    peft_cfg = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        lora_dropout=cfg["lora"]["lora_dropout"],
        bias=cfg["lora"]["bias"],
        task_type=cfg["lora"]["task_type"],
        target_modules=cfg["lora"]["target_modules"],
    )

    t_cfg = cfg["training"]
    # Reduce batch size on MPS to fit in shared memory
    train_bs = t_cfg["per_device_train_batch_size"] if device == "cuda" else 1
    grad_accum = t_cfg["gradient_accumulation_steps"] if device == "cuda" else (
        t_cfg["per_device_train_batch_size"] * t_cfg["gradient_accumulation_steps"]
    )

    sft_args = SFTConfig(
        output_dir=t_cfg["output_dir"],
        num_train_epochs=t_cfg["num_train_epochs"],
        per_device_train_batch_size=train_bs,
        gradient_accumulation_steps=grad_accum,
        learning_rate=t_cfg["learning_rate"],
        lr_scheduler_type=t_cfg["lr_scheduler_type"],
        warmup_ratio=t_cfg["warmup_ratio"],
        logging_steps=t_cfg["logging_steps"],
        eval_strategy=t_cfg["eval_strategy"],
        eval_steps=t_cfg["eval_steps"],
        save_steps=t_cfg["save_steps"],
        bf16=use_bf16,
        fp16=use_fp16,
        max_length=t_cfg["max_seq_length"],
        packing=t_cfg["packing"],
        seed=seed,
        dataset_text_field="text",
        report_to=t_cfg.get("report_to", "none"),
        # MPS workaround: disable dataloader multiprocessing
        dataloader_num_workers=0 if device != "cuda" else t_cfg.get("dataloader_num_workers", 0),
    )

    train_ds = load_ds(cfg["paths"]["train_jsonl"])
    val_ds = load_ds(cfg["paths"]["val_jsonl"])

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        peft_config=peft_cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    trainer.train()

    adapter_out = cfg["paths"]["adapter_out"]
    trainer.save_model(adapter_out)
    tokenizer.save_pretrained(adapter_out)
    print(f"Adapter saved to {adapter_out}")

    qualitative_check(trainer.model, tokenizer, SPOT_CHECK_PROMPTS, device)

    summary = {
        "run_name": run_name,
        "base_model": cfg["base_model"],
        "device": device,
        "lora": cfg["lora"],
        "training": {k: v for k, v in t_cfg.items() if k not in ("output_dir",)},
        "overrides": args.overrides or [],
    }
    summary_path = Path(t_cfg["output_dir"]) / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Run summary → {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--overrides", nargs="*", default=None,
                        help="e.g. lora.r=32 training.num_train_epochs=3")
    main(parser.parse_args())
