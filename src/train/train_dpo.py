#!/usr/bin/env python3
"""DPO post-training for Gemma 4 e4b LoRA adapters (supports RSLoRA, LoRA+, and Multi-LoRA).

Reads `{prompt, chosen, rejected}` JSONL produced by `src/collect/build_dpo_pairs.py`,
trains with `trl.DPOTrainer` (adapter-off reference policy, so no duplicate base model in VRAM),
and saves a vLLM-compatible `.safetensors` adapter targeting `gemma-4-e4b-it-qat-w4a16-ct`.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import get_peft_model
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.train_lora import (  # noqa: E402
    build_lora_config,
    build_optimizer,
    load_base,
    render_message,
    resolve_model_path,
    save_adapter,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DPO fine-tuning for Gemma 4 e4b LoRA")
    p.add_argument("--model", default=None, help="base model id or local dir")
    p.add_argument("--data", default="data/dpo_pairs.jsonl")
    p.add_argument("--out_dir", default="submission/adapters/main_lora")
    p.add_argument("--serve_model", default="gemma-4-e4b-it-qat-w4a16-ct")
    p.add_argument("--resume_from", default=None, help="optional SFT/RFT adapter dir to start DPO from")

    # LoRA / RSLoRA / LoRA+
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--target_modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    p.add_argument("--use_rslora", action="store_true", help="Rank-Stabilized LoRA (alpha / sqrt(r))")
    p.add_argument("--lora_plus_ratio", type=float, default=1.0,
                   help="LoRA+ lr multiplier for lora_B params (e.g. 16.0)")

    # DPO & optimisation
    p.add_argument("--beta", type=float, default=0.1, help="DPO KL penalty coefficient")
    p.add_argument("--max_len", type=int, default=4096)
    p.add_argument("--max_prompt_len", type=int, default=1536)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    p.add_argument("--use_4bit", action="store_true", help="QLoRA 4-bit base")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--max_examples", type=int, default=0)
    p.add_argument("--dry_run", action="store_true", help="render 1 pair, save adapter, exit")
    return p.parse_args()


def _render_turns(tokenizer: AutoTokenizer, messages: list[dict[str, Any]]) -> str:
    return "".join(render_message(tokenizer, msg) for msg in messages)


def load_dpo_dataset(path: str, tokenizer: AutoTokenizer, max_examples: int) -> Dataset:
    data_path = Path(path)
    if not data_path.is_file():
        raise SystemExit(f"[error] DPO data not found: {path} (run src/collect/build_dpo_pairs.py first)")

    rows: list[dict[str, str]] = []
    for line in data_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        prompt = _render_turns(tokenizer, item.get("prompt") or [])
        chosen = _render_turns(tokenizer, item.get("chosen") or [])
        rejected = _render_turns(tokenizer, item.get("rejected") or [])
        if prompt and chosen and rejected:
            rows.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
        if max_examples and len(rows) >= max_examples:
            break

    if not rows:
        raise SystemExit(f"[error] no usable preference pairs in {path}")
    print(f"[dpo] loaded {len(rows)} preference pairs from {path}")
    return Dataset.from_list(rows)


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model_path = resolve_model_path(args.model)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dpo_dataset(args.data, tokenizer, args.max_examples)
    model = load_base(args, model_path)
    if args.resume_from is None:
        model = get_peft_model(model, build_lora_config(args, model))
    model.print_trainable_parameters()

    if args.dry_run:
        sample = dataset[0]
        print(f"[dry-run] prompt={len(sample['prompt'])} chars, "
              f"chosen={len(sample['chosen'])} chars, rejected={len(sample['rejected'])} chars")
        save_adapter(model, args)
        return 0

    try:
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:
        raise SystemExit("[error] `trl` is required for DPO (`pip install -q trl`)") from exc

    dpo_args = DPOConfig(
        output_dir=str(Path(args.out_dir) / "_checkpoints"),
        beta=args.beta,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_len,
        max_prompt_length=args.max_prompt_len,
        logging_steps=5,
        save_strategy="no",
        bf16=(not args.use_4bit and torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        fp16=(not args.use_4bit and torch.cuda.is_available() and not torch.cuda.is_bf16_supported()),
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
    )
    optimizer = build_optimizer(model, args)
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # PEFT adapter-off reference saves VRAM
        args=dpo_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        optimizers=(optimizer, None),
    )
    trainer.train()
    save_adapter(model, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
