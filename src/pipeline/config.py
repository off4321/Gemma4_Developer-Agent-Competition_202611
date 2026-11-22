"""
実行設定。比較（doc/tasks.md T5）を 1 ファイルで駆動する。

src/pipeline/variants.yaml に `defaults:` と名前付き `variants:` を置き、variant は **変えた key だけ**
列挙する。それを defaults に deep merge するので、スイープはフラグ 1 つで済む:

    python3 src/pipeline/run_pipeline.py --variant think_2048 --stages gpu

src/ は dataset <you>/gemma4-dev-agent-code として Kaggle に上がるため、**同じ variants.yaml を
手元と kernel の両方**が読む。kernel が渡すのは variant 名だけ、というのが設計の要点。

なぜ variant が大事か: agent の挙動は model / budget / sampling の 3 組で決まる。
2 回の run が「比較可能」だと言えるのは、そのうち 1 つだけ違って他が同一だと
ファイルに記録されているときだけだ。だから variant 名 = run_name にしてある。
"""

from __future__ import annotations

import dataclasses
import re
import sys
from pathlib import Path
from typing import Any

import yaml

from pipeline.common import log

DEFAULTS: dict[str, Any] = {
    "run_name": "run01",
    "model_name": "gemma-4-e4b-it-qat-w4a16-ct",
    "task_limit": 12,
    "task_ids": [],
    "shard_index": 0,
    "num_shards": 1,
    "adapter_mode": "off",          # off = adapters/ を外してベースライン、use = adapter を使う
    "sandbox": "subprocess",        # Kaggle notebook には docker daemon が無い
    "graph_smoke": True,
    "smoke_test": True,
    "rewrite_model": True,
    "verbose": False,
    "pip_fallback": True,          # setup: vllm が入らないとき import error 名だけ PyPI から補う（測定 kernel 限り）
    "vllm_version": "",            # 空なら wheelhouse の pin。入れたら測定だけその vLLM に载せ替える（deviation）
    "budget": {                     # 実行 budget（agent ループの長さ）
        "timeout_seconds": 300,
        "verification_timeout_seconds": 300,
        "max_tool_calls": 80,
        "max_time_minutes": 45,
        "max_turns": 150,
    },
    "sampling": {                   # 生成条件（サンプリング）
        "temperature": 0.2,
        "max_output_tokens": 16384,
        "thinking_budget": 4096,
        "include_thoughts": True,
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """dict を深い所でマージする。variant が `sampling:` だけ書いたとき budget を消さないため。"""
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


@dataclasses.dataclass
class RunConfig:
    """1 run の全条件。ここに有った値が pipeline_*.json（manifest）にそのまま残る。"""
    variant: str
    run_name: str
    model_name: str
    task_limit: int
    task_ids: list[str]
    shard_index: int
    num_shards: int
    adapter_mode: str
    sandbox: str
    graph_smoke: bool
    smoke_test: bool
    rewrite_model: bool
    verbose: bool
    pip_fallback: bool
    vllm_version: str
    budget: dict[str, Any]
    sampling: dict[str, Any]
    raw: dict[str, Any] = dataclasses.field(default_factory=dict)

    def describe(self) -> None:
        log("config", f"variant={self.variant} run={self.run_name} model={self.model_name} "
                      f"adapter_mode={self.adapter_mode} sandbox={self.sandbox}")
        log("config", f"tasks: limit={self.task_limit} ids={self.task_ids or '-'} "
                      f"shard={self.shard_index}/{self.num_shards}")
        log("config", f"budget: {self.budget}")
        log("config", f"sampling: {self.sampling}")

    def warnings(self) -> list[str]:
        """「その設定では壊れやすい」と分かっている点を先に警告する（経験則ではなく HARNESS_README 由来）。"""
        notes: list[str] = []
        budget, sampling = self.budget, self.sampling
        if int(budget.get("max_tool_calls", 0)) < 40:
            notes.append(f"max_tool_calls={budget.get('max_tool_calls')} は複数ファイルの修正には少ない")
        if int(budget.get("max_turns", 0)) < 50:
            notes.append(f"max_turns={budget.get('max_turns')} だと nudge 込みで submit_patch 前に終わる")
        if int(sampling.get("max_output_tokens", 0)) < 8192:
            notes.append(f"max_output_tokens={sampling.get('max_output_tokens')} だと "
                         f"<|tool_call> が生成途中で切れる危険がある")
        if int(sampling.get("thinking_budget", 0)) > 8192:
            notes.append(f"thinking_budget={sampling.get('thinking_budget')} は 32k window 上で "
                         f"tool call と token を奪い合う")
        if self.adapter_mode not in {"off", "use"}:
            notes.append(f"adapter_mode={self.adapter_mode!r} は 'off' として扱う")
        return notes


def variants_file() -> Path:
    """variants.yaml の場所（src/pipeline/ に置く: src/ ごと Kaggle に上がるため）。"""
    return Path(__file__).resolve().parent / "variants.yaml"


def load(variant: str = "baseline", overrides: dict[str, Any] | None = None) -> RunConfig:
    """defaults <- variant <- CLI 上書き の順で 1 run の設定を作る（後者ほど強い）。"""
    path = variants_file()
    doc: dict[str, Any] = {}
    if path.is_file():
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    variants = doc.get("variants") or {}
    if variant not in variants and variant != "baseline":
        log("config", f"[warn] variant {variant!r} は {path} に無い（{sorted(variants)}）。"
                      f"defaults だけで実行する")
    merged = deep_merge(DEFAULTS, doc.get("defaults") or {})
    variant_doc = variants.get(variant) or {}
    merged = deep_merge(merged, variant_doc)
    merged = deep_merge(merged, {k: v for k, v in (overrides or {}).items() if v is not None})
    # variant 名がそのまま run_name になる（variant / CLI が明示で名前を付けない限り）。
    # これで `make pipeline-eval VARIANT=calls_40` と `make analyze RUN=calls_40` が揃う。
    if variant != "baseline" and "run_name" not in variant_doc and "run_name" not in (overrides or {}):
        merged["run_name"] = re.sub(r"[^A-Za-z0-9._-]+", "-", variant)

    fields = {f.name for f in dataclasses.fields(RunConfig) if f.name not in {"raw"}}
    kwargs = {k: v for k, v in merged.items() if k in fields}
    cfg = RunConfig(variant=variant, raw=merged, **kwargs)
    for note in cfg.warnings():
        log("config", f"[warn] {note}")
    return cfg


def usage() -> None:
    print("variants defined in " + str(variants_file()), file=sys.stderr)
