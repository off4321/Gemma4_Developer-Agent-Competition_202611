"""
PEFT LoRA で gemma-4-e4b を fine-tune する（assistant turn だけ supervised）。

入力は src/collect/collect_trajectories.py の出力、`{"messages": [...]}` の JSONL。

loss masking がこのスクリプトの要点: system 指示と tool 観測は -100（学習しない）にして、
assistant が出す思考 text と tool call 直列化だけを誤差にする。そうしないと、
「観測を再現する」方向に学習して tool call が壊れる。

もう一つの要点が会話丸ごと render すること（`_render_conversation`）: message を 1 件ずつ
isolated で chat template に当てると Gemma 4 は tool call を直列化せず、`<|tool_call>` の代わりに
`<|turn>model{...}` のような文字列を再現する adapter が育つ（`main_lora` 崩れの本体）。
`encode_conversation` は `norm[:i+1]` と `norm[:i]` の token 差分で turn を切り出す。

学習 base と提出後に実際に動く model は違う: ここは bf16 の google/gemma-4-e4b-it で学習し、
adapter_config.json の base_model_name_or_path を提供モデル
gemma-4-e4b-it-qat-w4a16-ct に書き換える（--serve_model）。公式サンプル adapter も同じことをする。
だから adapter は **必ず eval kernel（ADAPTER_MODE=use / VARIANT=lora）で検証してから**提出する。

VRAM が足りないとき: 16 GB GPU 1 枚なら --use_4bit（QLoRA）、batch 1 + grad accum、
max_len を 4096 に落とす。Kaggle の T4x2（bf16 非対応）では fp16 で動く。
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

DEFAULT_MODEL_IDS = ("google/gemma-4-e4b-it",)
KAGGLE_MODEL_GLOBS = (
    "/kaggle/input/models/google/gemma-4/transformers/gemma-4-e4b-it/*",
    "/kaggle/input/models/google/gemma-4/other/gemma-4-e4b-it/*",
    "/kaggle/input/models/google/gemma-4/transformers/gemma-4-e4b-it-qat-w4a16-ct/*",
)
ASSISTANT_ROLES = {"assistant", "model"}
DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
IGNORE = -100  # label the loss skips: the whole turn (system prompt, tool output) stays unsupervised

Message = dict[str, Any]
Conversation = list[Message]
Example = dict[str, list[int]]


# ------------------------------------------------------------------------------ run config
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune Gemma 4 e4b with LoRA")

    # what to train and where it lands
    p.add_argument("--model", default=None, help="model id or local dir. Default: Kaggle input mount, else google/gemma-4-e4b-it")
    p.add_argument("--data", default="data/trajectories.jsonl")
    p.add_argument("--out_dir", default="submission/adapters/main_lora")
    p.add_argument("--serve_model", default="gemma-4-e4b-it-qat-w4a16-ct",
                   help="name written to adapter_config.json base_model_name_or_path (the model vLLM serves)")

    # LoRA shape
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--alpha", type=int, default=16,
                   help="16 (= r) keeps alpha/r=1.0 so LoRA guides format without crushing e4b's native reasoning")
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--target_modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    p.add_argument("--use_rslora", action="store_true", help="Rank-Stabilized LoRA (alpha / sqrt(r))")
    p.add_argument("--lora_plus_ratio", type=float, default=1.0,
                   help="LoRA+ lr multiplier for lora_B params (1.0 = standard AdamW; high ratios cause 0.5s reflex loops)")

    # optimisation
    p.add_argument("--max_len", type=int, default=10240,
                   help="per-example token cap (10240 fits A100 40GB bf16; step-SFT middle-trims older turns)")
    p.add_argument("--epochs", type=float, default=1.0,
                   help="1.0 epoch over ~1000 step-SFT examples (~120 updates) avoids memorising short-thought reflexes")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-5,
                   help="3e-5 gentle LR preserves base gemma-4-e4b-it multi-step error recovery")
    p.add_argument("--lr_scheduler", default="cosine", choices=["cosine", "linear"],
                   help="cosine (default) stabilises special tool_call tokens late in training")
    p.add_argument("--warmup_steps", type=int, default=15)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--neftune_alpha", type=float, default=5.0,
                   help="NEFTune uniform embedding noise scale during train() (0 disables; zero inference cost)")
    p.add_argument("--action_loss_weight", type=float, default=1.25,
                   help="loss multiplier on edit_file/write_file/submit_patch steps to counter read_file skew")

    # which tokens are supervised & train/inference context alignment
    p.add_argument("--step_sft", action="store_true", default=True,
                   help="encode one example per assistant turn with middle-turn trimming")
    p.add_argument("--no_step_sft", dest="step_sft", action="store_false")
    p.add_argument("--keep_prior_thoughts", action="store_true", default=True,
                   help="keep prior assistant reasoning_content in history (matches sampling.yaml include_thoughts=true)")
    p.add_argument("--strip_prior_thoughts", dest="keep_prior_thoughts", action="store_false",
                   help="strip prior assistant reasoning_content (only use when sampling.yaml sets include_thoughts=false)")
    p.add_argument("--mask_system", action="store_true", default=True)
    p.add_argument("--train_on_system", dest="mask_system", action="store_false")
    p.add_argument("--train_on_observations", action="store_true",
                   help="also supervise role=tool observations (NOT recommended: the harness, not the model, emits them)")

    # data / precision / run control
    p.add_argument("--max_examples", type=int, default=0)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"],
                   help="auto: bfloat16 on Ampere+ (the 4x L4 scorer), float16 on Kaggle T4 x2")
    p.add_argument("--use_4bit", action="store_true", help="QLoRA (fit on a single 16 GB T4)")
    p.add_argument("--no_4bit", dest="use_4bit", action="store_false")
    p.add_argument("--log_every", type=int, default=2)
    p.add_argument("--dry_run", action="store_true", help="load model + tokenize 1 example, save an empty adapter, exit")
    p.add_argument("--resume_from", default=None, help="existing adapter dir to continue from")
    return p.parse_args()


def resolve_model_path(explicit: str | None) -> str:
    """Prefer the weights mounted by `kaggle kernels push`, else fall back to the Hub id."""
    if explicit:
        return explicit
    for pattern in KAGGLE_MODEL_GLOBS:
        for hit in sorted(p for p in glob.glob(pattern) if Path(p).is_dir()):
            if (Path(hit) / "config.json").is_file():
                print(f"[model] using mounted Kaggle model weights: {hit}")
                return hit
    print(f"[model] no Kaggle model mount found, falling back to HF id {DEFAULT_MODEL_IDS[0]}")
    return DEFAULT_MODEL_IDS[0]


# -------------------------------------------------------------------------------------- data
def load_examples(path: str, max_examples: int) -> list[Conversation]:
    """Read the JSONL, keeping one message list per line and stopping at the example cap."""
    if not Path(path).is_file():
        raise SystemExit(f"[error] data file not found: {path} (run src/collect/collect_trajectories.py first)")

    conversations: list[Conversation] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        messages = json.loads(line).get("messages") if line.strip() else None
        if not messages:
            continue
        conversations.append(messages)
        if max_examples and len(conversations) >= max_examples:
            break

    if not conversations:
        raise SystemExit(f"[error] no usable conversations in {path}")
    return conversations


def _normalize_tool_calls(message: Message) -> Message:
    """Deserialize string `function.arguments` into dicts and move `content` thought to `reasoning_content`.

    In Gemma 4's Jinja chat_template, `message['content']` on an assistant tool-call turn is emitted
    AFTER `<|tool_call>...<tool_call|>` (and even after `<|tool_response>`), which breaks prefix
    stability and trains reversed output ordering. Placing the thought in `reasoning_content` renders
    `<|channel>thought\\n...\\n<channel|><|tool_call>call:...<tool_call|>` in the native Gemma 4 order.
    """
    calls = message.get("tool_calls")
    if not calls:
        return message
    normalized = []
    for call in calls:
        fn = call.get("function") or {}
        raw_args = fn.get("arguments")
        if isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args or "{}")
            except json.JSONDecodeError:
                parsed = {}
            fn = {**fn, "arguments": parsed if isinstance(parsed, dict) else {"raw": parsed}}
            call = {**call, "function": fn}
        normalized.append(call)
    updated: Message = {**message, "tool_calls": normalized}
    if (
        updated.get("role") in ASSISTANT_ROLES
        and not (updated.get("reasoning_content") or updated.get("reasoning"))
        and isinstance(updated.get("content"), str)
        and updated["content"].strip()
    ):
        updated["reasoning_content"] = updated["content"].strip()
        updated["content"] = ""
    return updated


def render_message(tokenizer: AutoTokenizer, message: Message) -> str:
    """Render one turn with the model's chat template.

    Falls back to embedding the tool call as text when the installed template does not
    serialize `tool_calls` (checked once per assistant turn).
    """
    msg = _normalize_tool_calls(message)
    text = tokenizer.apply_chat_template([msg], add_generation_prompt=False, tokenize=False)
    calls = msg.get("tool_calls") or []
    if not calls or msg.get("role") not in ASSISTANT_ROLES:
        return text
    names = [c.get("function", {}).get("name", "") for c in calls]
    if any(name and name in text for name in names):
        return text  # the template already serialised the call
    payload = [
        {"name": c.get("function", {}).get("name"),
         "arguments": c.get("function", {}).get("arguments") or {}}
        for c in calls
    ]
    return text + "<tool_call>\n" + json.dumps(payload, ensure_ascii=False) + "\n</tool_call>\n"


def _strip_thought_fields(message: Message) -> Message:
    """Remove `reasoning_content` / `reasoning` from prior assistant turns to match `include_thoughts: false`."""
    if message.get("role") not in ASSISTANT_ROLES:
        return message
    return {k: v for k, v in message.items() if k not in ("reasoning_content", "reasoning")}


def _render_conversation(tokenizer: AutoTokenizer, messages: Conversation,
                         preserve_thinking: bool = True,
                         add_generation_prompt: bool = False) -> str:
    """Render a whole conversation with the model's chat template.

    Gemma 4 serialises `tool_calls` only when the template sees the turns around the call. Handing
    it `[message]` takes a template branch without the call header, and the adapter then answers
    `<|tool_call>` with literal `<|turn>model{...}` garbage.
    """
    try:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
            enable_thinking=True,
            preserve_thinking=preserve_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=add_generation_prompt, tokenize=False
        )


def is_supervised(role: str | None, args: argparse.Namespace) -> bool:
    """The whole loss policy in one place: assistant always, system/tool only when asked for."""
    if role in ASSISTANT_ROLES:
        return True
    if role == "system":
        return not args.mask_system
    return role == "tool" and args.train_on_observations


def _step_loss_weight(message: Message, args: argparse.Namespace) -> float:
    """Upweight decisive editing / submission turns so `read_file` frequency does not dominate SFT."""
    boost = float(getattr(args, "action_loss_weight", 1.0) or 1.0)
    if boost == 1.0:
        return 1.0
    calls = message.get("tool_calls") or []
    names = {(c.get("function") or {}).get("name") or c.get("name") for c in calls}
    if names & {"edit_file", "write_file", "submit_patch"}:
        return boost
    return 1.0


def encode_step_examples(tokenizer: AutoTokenizer, messages: Conversation,
                         args: argparse.Namespace) -> list[dict[str, Any]]:
    """Encode one training example per assistant turn, matching vLLM/ADK inference context.

    Why step-level SFT fixes both infinite loops and `submit_patch` omission:
      1. In Gemma 4's Jinja template, `loop.index0 > last_user_idx` keeps thoughts on all prior
         assistant turns unless `reasoning_content` is stripped. Stripping prior thoughts here
         matches `sampling.yaml` (`include_thoughts: false`) token-for-token, including the
         `<|channel>thought\\n` prompt prefix emitted after `<tool_response|>`.
      2. Trajectory-level right-truncation (`input_ids[:max_len]`) silently chopped off the tail of
         long multi-turn episodes — the exact turns containing `edit_file`, `pytest`, and
         `submit_patch`. Here we middle-trim oldest `(assistant, tool)` turns after `[system, user]`
         so terminal `edit_file` and `submit_patch` turns are 100% preserved.
    """
    norm = [_normalize_tool_calls(m) for m in messages]
    keep_prior = bool(getattr(args, "keep_prior_thoughts", False))
    bos_id = tokenizer.bos_token_id
    examples: list[dict[str, Any]] = []

    for step_idx, target_msg in enumerate(norm):
        if target_msg.get("role") not in ASSISTANT_ROLES or step_idx == 0:
            continue
        hist = [
            m if (keep_prior or m.get("role") not in ASSISTANT_ROLES) else _strip_thought_fields(m)
            for m in norm[:step_idx]
        ]
        # Middle-trim oldest `(assistant, tool...)` groups after `[system, user]` to fit `args.max_len`
        trimmed = list(hist)
        prompt_ids: list[int] = []
        full_ids: list[int] = []
        while True:
            try:
                prompt_text = _render_conversation(
                    tokenizer, trimmed, preserve_thinking=keep_prior, add_generation_prompt=True
                )
                full_text = _render_conversation(
                    tokenizer, trimmed + [target_msg], preserve_thinking=keep_prior, add_generation_prompt=False
                )
                if not full_text.startswith(prompt_text):
                    prompt_text = _render_conversation(
                        tokenizer, trimmed, preserve_thinking=keep_prior, add_generation_prompt=False
                    )
            except Exception:
                prompt_text = "".join(render_message(tokenizer, m) for m in trimmed)
                full_text = prompt_text + render_message(tokenizer, target_msg)

            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
            if len(full_ids) <= args.max_len - 1 or len(trimmed) <= 2:
                break
            drop_end = 3
            while drop_end < len(trimmed) and trimmed[drop_end].get("role") == "tool":
                drop_end += 1
            if drop_end >= len(trimmed):
                break
            trimmed = trimmed[:2] + trimmed[drop_end:]

        if not full_ids or len(full_ids) <= len(prompt_ids) or full_ids[:len(prompt_ids)] != prompt_ids:
            continue
        comp_ids = full_ids[len(prompt_ids):]
        input_ids = prompt_ids + comp_ids
        labels = [IGNORE] * len(prompt_ids) + comp_ids
        if bos_id is not None and input_ids and input_ids[0] != bos_id:
            input_ids = [bos_id] + input_ids
            labels = [IGNORE] + labels
        input_ids = input_ids[:args.max_len]
        labels = labels[:args.max_len]
        if not any(lbl != IGNORE for lbl in labels):
            continue
        examples.append({
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
            "weight": _step_loss_weight(target_msg, args),
        })
    return examples


def encode_conversation(tokenizer: AutoTokenizer, messages: Conversation,
                        args: argparse.Namespace) -> Example | None:
    """Tokenize a conversation turn by turn, labelling unsupervised tokens IGNORE.

    A turn is the token delta between the full renders of `norm[:i + 1]` and `norm[:i]`, so the
    template keeps the multi-turn context it needs to serialise `tool_calls` (rendering each
    message alone is what corrupted `main_lora`). Fall back to `render_message` per turn when a
    template rejects a partial conversation.

    Returns None when masking leaves nothing to learn, so the caller can drop the example.
    """
    norm = [_normalize_tool_calls(message) for message in messages]
    input_ids: list[int] = []
    labels: list[int] = []
    bos_id = tokenizer.bos_token_id
    unstable = False

    for index, message in enumerate(norm):
        try:
            prefix = tokenizer(_render_conversation(tokenizer, norm[:index + 1]),
                               add_special_tokens=False)["input_ids"]
        except Exception as exc:  # a template that refuses a partial conversation
            print(f"[warn] apply_chat_template(messages[:{index + 1}]) failed ({exc}); "
                  f"falling back to render_message for this turn")
            ids = tokenizer(render_message(tokenizer, message), add_special_tokens=False)["input_ids"]
            if index and bos_id is not None and ids and ids[0] == bos_id:
                ids = ids[1:]  # the template re-adds BOS every turn; keep only the leading one
        else:
            # a template that re-renders from scratch (rather than appending a turn) would shift the
            # labels, so say so once instead of silently mislabelling every turn after it
            if not unstable and prefix[:len(input_ids)] != input_ids:
                unstable = True
                print("[warn] chat template is not prefix-stable: labels may shift at turn edges")
            ids = prefix[len(input_ids):]  # delta against the prefix already accumulated
        supervised = is_supervised(message.get("role"), args)
        input_ids.extend(ids)
        labels.extend(ids if supervised else [IGNORE] * len(ids))

    if bos_id is not None and input_ids and input_ids[0] != bos_id:
        input_ids, labels = [bos_id] + input_ids, [IGNORE] + labels

    labels = labels[:args.max_len]
    if not any(label != IGNORE for label in labels):
        return None  # nothing left for the loss after truncation
    return {"input_ids": input_ids[:args.max_len], "labels": labels,
            "attention_mask": [1] * min(len(input_ids), args.max_len)}


def build_dataset(tokenizer: AutoTokenizer, conversations: list[Conversation], args: argparse.Namespace) -> Dataset:
    rows: list[dict[str, Any]] = []
    use_step_sft = bool(getattr(args, "step_sft", True)) and not getattr(args, "train_on_observations", False)
    dropped = 0
    for messages in conversations:
        if use_step_sft:
            step_rows = encode_step_examples(tokenizer, messages, args)
            if step_rows:
                rows.extend(step_rows)
            else:
                dropped += 1
        else:
            encoded = encode_conversation(tokenizer, messages, args)
            if encoded is not None:
                rows.append(encoded)
            else:
                dropped += 1

    tokens = [sum(1 for label in row["labels"] if label != IGNORE) for row in rows]
    mode_tag = f"step-SFT (keep_prior_thoughts={getattr(args, 'keep_prior_thoughts', False)})" if use_step_sft else "trajectory-SFT"
    print(f"[data] {mode_tag}: encoded {len(rows)} examples from {len(conversations)} conversations "
          f"({dropped} conversations dropped), {sum(tokens) / max(1, len(rows)):.0f} supervised tokens/example")
    if not rows:
        raise SystemExit("[error] every example collapsed to zero supervised tokens")

    # print the supervised text so a broken tool-call serialisation is visible before training starts
    supervised_ids = [t for t, label in zip(rows[0]["input_ids"], rows[0]["labels"]) if label != IGNORE]
    print("[data] sample supervised text:\n" + tokenizer.decode(supervised_ids))
    return Dataset.from_list(rows)


class Collator:
    """Pad a batch to its own width: labels pad with IGNORE so padding never enters the loss."""

    def __init__(self, tokenizer: AutoTokenizer):
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def __call__(self, features: list[Example]) -> dict[str, torch.Tensor]:
        width = max(len(f["input_ids"]) for f in features)
        rows = {"input_ids": [], "labels": [], "attention_mask": []}
        for feature in features:
            pad = width - len(feature["input_ids"])
            rows["input_ids"].append(feature["input_ids"] + [self.pad_id] * pad)
            rows["labels"].append(feature["labels"] + [IGNORE] * pad)
            rows["attention_mask"].append(feature["attention_mask"] + [0] * pad)
        return {key: torch.tensor(values, dtype=torch.long) for key, values in rows.items()}


# --------------------------------------------------------------------------------- base model
def auto_dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16


def quantization_config():
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=auto_dtype(), bnb_4bit_use_double_quant=True)


def prepare_for_kbit_training(model):
    """Freeze the 4-bit base and enable gradient checkpointing without fp32-upcasting 262k-vocab lm_head."""
    for param in model.parameters():
        param.requires_grad = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    return model


def load_base(args: argparse.Namespace, model_path: str):
    """Load the base weights, wrap a `--resume_from` adapter, and enable the memory savers."""
    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if args.use_4bit and torch.cuda.is_available():
        kwargs["quantization_config"] = quantization_config()
        kwargs["torch_dtype"] = auto_dtype()
        kwargs["device_map"] = "auto"
        print(f"[model] QLoRA: 4-bit NF4 base weights (non-quantized={kwargs['torch_dtype']})")
    else:
        kwargs["torch_dtype"] = auto_dtype() if args.dtype == "auto" else DTYPES[args.dtype]
        print(f"[model] torch_dtype={kwargs['torch_dtype']} (T4 x2 has no bf16 matmul support -> float16 there)")
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"  # ~8B params need both T4s in half precision

    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    if args.resume_from:
        model = PeftModel.from_pretrained(model, args.resume_from, is_trainable=True)
        print(f"[model] resumed adapter from {args.resume_from}")
    elif args.use_4bit:
        model = prepare_for_kbit_training(model)
        gc.collect()
        torch.cuda.empty_cache()
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.config.use_cache = False
    return model


def resolve_target_modules(model, raw_targets: str) -> list[str]:
    """Resolve target module names to actual `nn.Linear` paths, avoiding `Gemma4ClippableLinear`."""
    wanted = [m.strip() for m in raw_targets.split(",") if m.strip()]
    if model is None:
        return wanted
    wanted_set = set(wanted)
    skip_prefixes = ("vision_tower", "audio_tower", "vision_model", "audio_model")
    matched: list[str] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        parts = name.split(".")
        if any(p in skip_prefixes for p in parts):
            continue
        leaf = parts[-1]
        parent_leaf = parts[-2] if len(parts) >= 2 else ""
        if leaf in wanted_set or (leaf == "linear" and parent_leaf in wanted_set):
            matched.append(name)
    if matched:
        print(f"[model] resolved {len(matched)} linear LoRA targets (skipped vision/audio & wrappers)")
        return sorted(matched)
    return wanted


def build_lora_config(args: argparse.Namespace, model=None) -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        use_rslora=getattr(args, "use_rslora", False),
        target_modules=resolve_target_modules(model, args.target_modules),
    )


def build_optimizer(model, args: argparse.Namespace) -> torch.optim.AdamW:
    """Create AdamW with optional LoRA+ asymmetric learning rate for `lora_B`."""
    ratio = getattr(args, "lora_plus_ratio", 1.0)
    if ratio <= 1.0:
        trainable = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    lora_b = [p for n, p in model.named_parameters() if p.requires_grad and "lora_B" in n]
    other = [p for n, p in model.named_parameters() if p.requires_grad and "lora_B" not in n]
    groups = [{"params": other, "lr": args.lr}, {"params": lora_b, "lr": args.lr * ratio}]
    print(f"[train] LoRA+ enabled: lr={args.lr:.2e}, lora_B lr={args.lr * ratio:.2e} (x{ratio:g})")
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


# ---------------------------------------------------------------------------------- training
def chunked_causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor, chunk_size: int = 512) -> torch.Tensor:
    """Compute shifted cross-entropy in sequence chunks to avoid a 4+ GiB fp32 logits spike on 262k vocab."""
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    valid_tokens = (shift_labels != IGNORE).sum()
    if valid_tokens == 0:
        return logits.sum() * 0.0
    total_loss = logits.new_zeros((), dtype=torch.float32)
    seq_len = shift_labels.size(1)
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        sub_labels = shift_labels[:, start:end].reshape(-1)
        if not (sub_labels != IGNORE).any():
            continue
        sub_logits = shift_logits[:, start:end, :].reshape(-1, shift_logits.size(-1)).float()
        total_loss = total_loss + torch.nn.functional.cross_entropy(
            sub_logits, sub_labels, ignore_index=IGNORE, reduction="sum"
        )
    return total_loss / valid_tokens


def register_neftune_hook(model, alpha: float):
    """Attach NEFTune uniform noise hook to input embeddings during `model.train()` (Jain et al., 2024)."""
    if alpha <= 0:
        return None
    emb = model.get_input_embeddings()
    if emb is None:
        return None

    def _hook(module, _inputs, output):
        if not module.training or not torch.is_tensor(output):
            return output
        dims = float(output.size(1) * output.size(2)) if output.dim() >= 3 else float(output.numel())
        mag = alpha / math.sqrt(max(1.0, dims))
        noise = torch.empty_like(output).uniform_(-mag, mag)
        return output + noise

    print(f"[train] NEFTune enabled: alpha={alpha:g}")
    return emb.register_forward_hook(_hook)


def training_step(model, example: dict[str, Any], index: int, device: torch.device,
                  collator: Collator, args: argparse.Namespace) -> float | None:
    """Forward one example and accumulate its gradient; return the raw loss, or None on OOM.

    A trajectory that does not fit is dropped instead of killing a whole afternoon run.
    """
    weight = float(example.get("weight", 1.0))
    clean_ex = {k: example[k] for k in ("input_ids", "labels", "attention_mask")}
    batch = {key: value.to(device) for key, value in collator([clean_ex]).items()}
    labels = batch.pop("labels")
    try:
        logits = model(**batch).logits
        loss = chunked_causal_lm_loss(logits, labels)
        (loss * weight / args.grad_accum).backward()
    except torch.cuda.OutOfMemoryError:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        print(f"[warn] OOM at sample {index} (len={len(example['input_ids'])}), skipping")
        return None
    return float(loss.detach())


def train(model, dataset: Dataset, tokenizer: AutoTokenizer, args: argparse.Namespace) -> None:
    """Manual epoch loop: shuffle, accumulate `grad_accum` single-example micro-steps, then step.

    One example at a time (not `batch_size`) keeps peak memory predictable at long max_len.
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(model, args)
    total_steps = int(max(1, len(dataset) // args.grad_accum) * args.epochs)
    warmup = min(args.warmup_steps, max(1, total_steps // 5))
    sched_name = getattr(args, "lr_scheduler", "cosine")
    if sched_name == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)
    else:
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)
    print(f"[train] scheduler={sched_name}, warmup_steps={warmup}, total_steps={total_steps}")
    device = torch.device(next(model.parameters()).device)
    collator = Collator(tokenizer)
    neftune_handle = register_neftune_hook(model, float(getattr(args, "neftune_alpha", 0.0) or 0.0))

    model.train()
    global_step, running_loss = 0, 0.0
    try:
        for epoch in range(math.ceil(args.epochs)):
            order = list(range(len(dataset)))
            random.shuffle(order)
            optimizer.zero_grad(set_to_none=True)

            for step, index in enumerate(order):
                loss = training_step(model, dataset[index], index, device, collator, args)
                if loss is None:
                    continue  # OOM: the accumulation window simply stays short
                running_loss += loss
                if (step + 1) % args.grad_accum:
                    continue

                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % args.log_every:
                    continue
                avg = running_loss / (args.log_every * args.grad_accum)
                running_loss = 0.0
                print(f"[train] epoch {epoch} step {global_step}/{total_steps} "
                      f"loss {avg:.4f} lr {scheduler.get_last_lr()[0]:.2e} "
                      f"mem {torch.cuda.max_memory_allocated() / 1024**3:.1f} GiB")
    finally:
        if neftune_handle is not None:
            neftune_handle.remove()


# ------------------------------------------------------------------------------------ saving
def rewrite_adapter_config(args: argparse.Namespace) -> None:
    """Point the adapter at the model the harness actually serves (W4A16 quantized e4b).

    Mirrors the official sample adapters, whose base_model_name_or_path is
    `google/gemma-4-31b-it-qat-w4a16-ct`.
    """
    cfg_path = Path(args.out_dir) / "adapter_config.json"
    if not cfg_path.is_file():
        return
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["base_model_name_or_path"] = f"google/{args.serve_model}"
    cfg["inference_mode"] = True
    if int(cfg.get("r") or 0) > 128:
        print(f"[warn] adapter rank {cfg['r']} > vLLM max_lora_rank 128 -> shrink --r")
    cfg_path.write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[adapter] base_model_name_or_path = {cfg['base_model_name_or_path']}, r={cfg['r']}")


def save_adapter(model, args: argparse.Namespace) -> None:
    """Write safetensors only and strip DPO artifacts (`ref/`, `_checkpoints/`)."""
    import shutil

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        model.save_pretrained(args.out_dir, safe_serialization=True, selected_adapters=["default"])
    except TypeError:
        model.save_pretrained(args.out_dir, safe_serialization=True)
    for leftover_dir in ("ref", "_checkpoints"):
        shutil.rmtree(out_dir / leftover_dir, ignore_errors=True)
    rewrite_adapter_config(args)
    for junk in out_dir.glob("*.bin"):
        junk.unlink()
    print(f"[train] saved adapter -> {args.out_dir}")
    files = {name: str(size) for name, size in sorted(
        (p.name, p.stat().st_size) for p in out_dir.rglob("*") if p.is_file())}
    print(json.dumps(files, indent=1))


# --------------------------------------------------------------------------------------- run
def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model_path = resolve_model_path(args.model)

    print(f"[train] base model : {model_path}")
    print(f"[train] data       : {args.data}")
    print(f"[train] out_dir    : {args.out_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[train] chat template mentions tools: {'tool' in (tokenizer.chat_template or '').lower()}")

    dataset = build_dataset(tokenizer, load_examples(args.data, args.max_examples), args)
    model = load_base(args, model_path)
    if args.resume_from is None:
        model = get_peft_model(model, build_lora_config(args, model))
    model.print_trainable_parameters()

    if args.dry_run:
        sample = dataset[0]
        supervised = sum(1 for label in sample["labels"] if label != IGNORE)
        print(f"[dry-run] example length={len(sample['input_ids'])} supervised={supervised}")
        save_adapter(model, args)
        print(f"[dry-run] wrote placeholder adapter to {args.out_dir}")
        return 0

    train(model, dataset, tokenizer, args)
    save_adapter(model, args)
    return 0

if __name__ == "__main__":
    sys.exit(main())
