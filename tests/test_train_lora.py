"""Unit tests for `src/train/train_lora.py` tokenisation & delimiter cleanup."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "train" / "train_lora.py"

_spec = importlib.util.spec_from_file_location("train_lora", MODULE_PATH)
assert _spec is not None and _spec.loader is not None, f"cannot import {MODULE_PATH}"
train_lora = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_lora)


def test_strip_trailing_tool_response_removes_delimiter_and_whitespace() -> None:
    prefix = "<|turn>model\n<|channel>thought\nread\n<channel|><|tool_call>call:read_file{}<tool_call|>"
    assert train_lora._strip_trailing_tool_response(prefix + "<|tool_response>") == prefix
    assert train_lora._strip_trailing_tool_response(prefix + "\n<|tool_response>\n") == prefix
    assert train_lora._strip_trailing_tool_response(prefix + "  \n  <|tool_response>  \n") == prefix
    assert train_lora._strip_trailing_tool_response(prefix + "\n") == prefix


def test_strip_trailing_tool_response_preserves_normal_text_turns() -> None:
    normal_nl = "<|turn>model\n<|channel>thought\ndone\n<channel|>All fixed.<turn|>\n"
    normal_bare = "<|turn>model\nAll fixed.<turn|>"
    prior_tool = (
        "<|turn>model\n<|tool_call>call:read_file{}<tool_call|>"
        "<|tool_response>{\"status\":\"ok\"}<tool_response|>\n"
        "<|turn>model\nAll fixed.<turn|>\n"
    )
    assert train_lora._strip_trailing_tool_response(normal_nl) == normal_nl
    assert train_lora._strip_trailing_tool_response(normal_bare) == normal_bare
    assert train_lora._strip_trailing_tool_response(prior_tool) == prior_tool
    assert train_lora._strip_trailing_tool_response("") == ""


class _FakeGemma4Tokenizer:
    """Minimal pure-Python tokenizer mimicking Gemma 4's `<|tool_response>` emission."""

    bos_token_id = 2
    pad_token_id = 0
    eos_token_id = 1
    _vocab: dict[str, int] = {}

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        add_generation_prompt: bool = False,
        tokenize: bool = False,
        **_kwargs: Any,
    ) -> str:
        assert tokenize is False
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role")
            if role in ("system", "user"):
                parts.append(f"<|turn>{role}\n{msg.get('content', '')}<turn|>\n")
            elif role == "assistant":
                thought = msg.get("reasoning_content") or ""
                thought_blk = f"<|channel>thought\n{thought}\n<channel|>" if thought else ""
                calls = msg.get("tool_calls") or []
                if calls:
                    call_str = "".join(
                        f"<|tool_call>call:{c['function']['name']}{{}}<tool_call|>" for c in calls
                    )
                    # Gemma 4 emits <|tool_response> even with add_generation_prompt=False
                    parts.append(f"<|turn>model\n{thought_blk}{call_str}\n<|tool_response>\n")
                else:
                    parts.append(f"<|turn>model\n{thought_blk}{msg.get('content', '')}<turn|>\n")
            elif role == "tool":
                # Strip trailing `<|tool_response>\n` from prior assistant turn if we're continuing
                if parts and parts[-1].endswith("\n<|tool_response>\n"):
                    parts[-1] = parts[-1][: -len("\n<|tool_response>\n")]
                parts.append(f"\n<|tool_response>{msg.get('content', '')}<tool_response|>\n")
        if add_generation_prompt:
            parts.append("<|turn>model\n<|channel>thought\n")
        return "".join(parts)

    def _encode_str(self, text: str) -> list[int]:
        tokens: list[int] = []
        i = 0
        specials = (
            "<|turn>",
            "<turn|>",
            "<|channel>",
            "<channel|>",
            "<|tool_call>",
            "<tool_call|>",
            "<|tool_response>",
            "<tool_response|>",
        )
        while i < len(text):
            matched = next((sp for sp in specials if text.startswith(sp, i)), None)
            tok = matched if matched is not None else text[i]
            if tok not in self._vocab:
                self._vocab[tok] = len(self._vocab) + 10
            tokens.append(self._vocab[tok])
            i += len(tok)
        return tokens

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        return {"input_ids": self._encode_str(text)}

    def decode(self, ids: list[int]) -> str:
        inv = {v: k for k, v in self._vocab.items()}
        return "".join(inv.get(idx, "") for idx in ids)


def test_encode_step_examples_and_conversation_exclude_tool_response_from_labels() -> None:
    tok = _FakeGemma4Tokenizer()
    args = argparse.Namespace(
        max_len=2048,
        keep_prior_thoughts=True,
        action_loss_weight=1.0,
        mask_system=True,
        train_on_observations=False,
    )
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "fix bug"},
        {
            "role": "assistant",
            "content": "read file",
            "tool_calls": [{"id": "c1", "function": {"name": "read_file", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": '{"status":"ok"}'},
        {"role": "assistant", "content": "All fixed."},
    ]

    steps = train_lora.encode_step_examples(tok, messages, args)
    assert len(steps) == 2
    first_comp = tok.decode([lbl for lbl in steps[0]["labels"] if lbl != train_lora.IGNORE])
    assert "<|tool_response>" not in first_comp
    assert first_comp.endswith("<tool_call|>")

    second_comp = tok.decode([lbl for lbl in steps[1]["labels"] if lbl != train_lora.IGNORE])
    assert second_comp.endswith("<turn|>\n")

    conv = train_lora.encode_conversation(tok, messages, args)
    assert conv is not None
    conv_sup = tok.decode([lbl for lbl in conv["labels"] if lbl != train_lora.IGNORE])
    assert "<|tool_response>" not in conv_sup
    assert conv_sup.endswith("<turn|>\n")


class _RealGemma4MultiTurnTokenizer(_FakeGemma4Tokenizer):
    """Mimics Gemma 4's actual template: multiple tool_call/tool_response cycles inside ONE <|turn>model block."""

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        add_generation_prompt: bool = False,
        tokenize: bool = False,
        **_kwargs: Any,
    ) -> str:
        assert tokenize is False
        parts: list[str] = []
        in_model_turn = False
        for msg in messages:
            role = msg.get("role")
            if role in ("system", "user"):
                if in_model_turn:
                    parts.append("<turn|>\n")
                    in_model_turn = False
                parts.append(f"<|turn>{role}\n{msg.get('content', '')}<turn|>\n")
            elif role == "assistant":
                if not in_model_turn:
                    parts.append("<|turn>model\n")
                    in_model_turn = True
                thought = msg.get("reasoning_content") or ""
                thought_blk = f"<|channel>thought\n{thought}\n<channel|>" if thought else ""
                calls = msg.get("tool_calls") or []
                if calls:
                    call_str = "".join(
                        f"<|tool_call>call:{c['function']['name']}{{}}<tool_call|>" for c in calls
                    )
                    parts.append(f"{thought_blk}{call_str}")
                else:
                    parts.append(f"{thought_blk}{msg.get('content', '')}<turn|>\n")
                    in_model_turn = False
            elif role == "tool":
                parts.append(f"<|tool_response>{msg.get('content', '')}<tool_response|>")
        return "".join(parts)


def test_encode_conversation_supervises_all_turns_inside_single_gemma4_model_turn() -> None:
    tok = _RealGemma4MultiTurnTokenizer()
    args = argparse.Namespace(
        max_len=4096,
        keep_prior_thoughts=True,
        action_loss_weight=1.0,
        mask_system=True,
        train_on_observations=False,
    )
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "fix bug"},
        {
            "role": "assistant",
            "content": "turn 1 read",
            "tool_calls": [{"id": "c1", "function": {"name": "read_file", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "SECRET_OBS_1"},
        {
            "role": "assistant",
            "content": "turn 2 edit",
            "tool_calls": [{"id": "c2", "function": {"name": "edit_file", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c2", "content": "SECRET_OBS_2"},
        {
            "role": "assistant",
            "content": "turn 3 submit",
            "tool_calls": [{"id": "c3", "function": {"name": "submit_patch", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c3", "content": "SECRET_OBS_3"},
        {"role": "assistant", "content": "Fixed cleanly."},
    ]
    conv = train_lora.encode_conversation(tok, messages, args)
    assert conv is not None
    sup = tok.decode([lbl for lbl in conv["labels"] if lbl != train_lora.IGNORE])
    # ALL 4 assistant turns MUST be supervised!
    assert "turn 1 read" in sup and "call:read_file" in sup
    assert "turn 2 edit" in sup and "call:edit_file" in sup
    assert "turn 3 submit" in sup and "call:submit_patch" in sup
    assert "Fixed cleanly.<turn|>\n" in sup
    # ZERO tool observations or `<|tool_response>` delimiters may be supervised!
    assert "SECRET_OBS_1" not in sup
    assert "SECRET_OBS_2" not in sup
    assert "SECRET_OBS_3" not in sup
    assert "<|tool_response>" not in sup
    assert "<tool_response|>" not in sup
