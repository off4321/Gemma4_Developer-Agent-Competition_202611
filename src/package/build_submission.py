"""
提出物を **zip 化ではなく契約バリデーション**としてビルドする。

HARNESS_README 2.2 / 2.4 / 3.2 / 3.4 と swegemma.config を写した検査:
  layout      ルート直下に agent.yaml が 1 個 / 拡張子 whitelist / symlink 禁止 / 展開後 3 GiB
  models      宣言 model が 1 種類で ALLOWED_MODEL_NAMES の内側
  generation  generate_content_config の key、max_output_tokens 1..32768、
              thinking_* は thinking_config の内側（トップレベル配置は違反）
  adapters    adapters/<name>/adapter_config.json + adapter_model*.safetensors、pickle 禁止、
              rank <= 128（vLLM の max_lora_rank）、agent.yaml の `adapter:` と実体の対応
  eval_config ルートにあって `evaluation:` key があるか
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

import yaml

# ------------------------------------------------------------------ 契約の定数
MAX_SUBMISSION_SIZE_BYTES = 3 * 1024 ** 3          # 3,221,225,472
ALLOWED_SUBMISSION_EXTENSIONS = {
    ".yaml", ".yml", ".md", ".txt", ".py", ".json", ".safetensors",
}
ALLOWED_ADAPTER_EXTENSIONS = {".safetensors"}
ROOT_CONFIG_NAMES = ("agent.yaml", "agent.yml", "root_agent.yaml", "root_agent.yml")
EVAL_CONFIG_NAMES = ("eval_config.yaml", "eval_config.yml")
ALLOWED_MODEL_NAMES = frozenset({
    "gemma-4-31b-it-qat-w4a16-ct",
    "gemma-4-12b-it-qat-w4a16-ct",
    "gemma-4-e4b-it-qat-w4a16-ct",
    "gemma-4-e2b-it-qat-w4a16-ct",
})
ALLOWED_GENERATION_FIELDS = {
    "temperature", "top_p", "top_k", "max_output_tokens", "presence_penalty",
    "frequency_penalty", "stop_sequences", "response_mime_type", "seed", "thinking_config",
}
ALLOWED_THINKING_FIELDS = {"thinking_budget", "thinking_level", "include_thoughts"}
STRAY_THINKING_FIELDS = ("thinking_budget", "include_thoughts", "thinking_level")
MODEL_PROVIDER_PREFIX = re.compile(r"^(openai|google|hosted_vllm|custom)/")
SAFE_PATH_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
INCLUDE_RE = re.compile(r"!include\s+([^\n]+)")

MAX_LORA_RANK = 128
TOKEN_LIMIT = 32768
MAX_INSTRUCTION_CHARS = 1_000_000
PICKLE_SUFFIXES = {".bin", ".pt", ".pth", ".ckpt"}
EVAL_KNOWN_KEYS = {"timeout_seconds", "max_tool_calls", "max_time_minutes", "max_turns"}


class Checker:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def ok(self, msg: str) -> None:
        print(f"  [ok] {msg}")

    def report(self) -> None:
        for warning in self.warnings:
            print(f"  [warn] {warning}")
        for error in self.errors:
            print(f"  [ERROR] {error}")


def _mebibytes(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.2f} MiB"


def _find_config(source: Path, names: tuple[str, ...]) -> list[Path]:
    return [source / name for name in names if (source / name).is_file()]


def _is_vcs_artifact(rel: Path) -> bool:
    return any(part.startswith(".git") for part in rel.parts)


def _iter_package_files(source: Path) -> list[Path]:
    return [path for path in sorted(source.rglob("*")) if not path.is_dir()]


def load_yaml(path: Path) -> Any:
    """Parse a submission YAML, resolving `!include <rel>` relative to `path.parent`."""
    def include(match: re.Match[str]) -> str:
        target = (path.parent / match.group(1).strip()).resolve()
        if not target.is_file():
            raise FileNotFoundError(f"{path}: !include target missing -> {target}")
        text = target.read_text(encoding="utf-8")
        value = text if target.suffix.lower() in {".md", ".txt"} else yaml.safe_load(text)
        return json.dumps(value)

    return yaml.safe_load(INCLUDE_RE.sub(include, path.read_text(encoding="utf-8")))


def check_layout(source: Path, ck: Checker) -> Path:
    roots = _find_config(source, ROOT_CONFIG_NAMES)
    if len(roots) != 1:
        ck.error(f"expected exactly 1 root config among {ROOT_CONFIG_NAMES}, found {[str(r) for r in roots]}")
        return source / "agent.yaml"
    ck.ok(f"root config: {roots[0].name}")

    total = 0
    for path in _iter_package_files(source):
        rel = path.relative_to(source)
        if path.is_symlink():
            ck.error(f"symlink not permitted: {rel}")
            continue
        if _is_vcs_artifact(rel):
            ck.warn(f"skipping VCS artifact from packaging: {rel}")
            continue
        ext = path.suffix.lower()
        if ext not in ALLOWED_SUBMISSION_EXTENSIONS:
            ck.error(f"disallowed extension {ext or '(none)'}: {rel}")
        total += path.stat().st_size

    if total > MAX_SUBMISSION_SIZE_BYTES:
        ck.error(f"unpacked size {total:,} bytes exceeds 3 GiB limit")
    else:
        ck.ok(f"unpacked size {_mebibytes(total)} / 3 GiB")
    return roots[0]


def _referenced_configs(data: dict) -> list[Any]:
    refs: list[Any] = []
    for key in ("sub_agents", "tools"):
        entries = data.get(key)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            agent_tool = entry.get("agent_tool")
            if isinstance(agent_tool, dict) and agent_tool.get("config_path"):
                refs.append(agent_tool["config_path"])
            if entry.get("config_path"):
                refs.append(entry["config_path"])
    return refs


def iter_agent_docs(root_cfg: Path, source: Path) -> list[tuple[Path, dict]]:
    docs: list[tuple[Path, dict]] = []
    seen: set[Path] = set()
    failures: list[str] = []

    def walk(cfg_path: Path) -> None:
        cfg_path = cfg_path.resolve()
        if cfg_path in seen or not cfg_path.is_file():
            return
        seen.add(cfg_path)
        try:
            data = load_yaml(cfg_path)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{cfg_path.relative_to(source)}: YAML load failed: {exc}")
            return
        if not isinstance(data, dict):
            failures.append(f"{cfg_path.relative_to(source)}: not a mapping")
            return
        docs.append((cfg_path, data))
        for ref in _referenced_configs(data):
            walk(cfg_path.parent / ref)

    walk(root_cfg)
    if failures:
        for failure in failures:
            print(f"  [yaml-error] {failure}")
        sys.exit(2)
    return docs


def _check_thinking_config(name: str, think: dict, ck: Checker) -> None:
    if illegal := set(think) - ALLOWED_THINKING_FIELDS:
        ck.error(f"{name}: thinking_config has illegal fields {sorted(illegal)}")
    budget = think.get("thinking_budget")
    if isinstance(budget, int) and not 0 <= budget <= TOKEN_LIMIT:
        ck.error(f"{name}: thinking_budget {budget} out of 0..{TOKEN_LIMIT}")
    if "thinking_level" in think:
        ck.warn(f"{name}: thinking_level set -> maps to OpenAI reasoning_effort "
                f"and is dropped by litellm; prefer thinking_budget + include_thoughts")


def _check_generation_config(name: str, gen: dict, ck: Checker) -> None:
    if illegal := set(gen) - ALLOWED_GENERATION_FIELDS:
        ck.error(f"{name}: generate_content_config has illegal fields {sorted(illegal)}")
    tokens = gen.get("max_output_tokens")
    if isinstance(tokens, int) and not 1 <= tokens <= TOKEN_LIMIT:
        ck.error(f"{name}: max_output_tokens {tokens} out of 1..{TOKEN_LIMIT}")
    if isinstance(gen.get("thinking_config"), dict):
        _check_thinking_config(name, gen["thinking_config"], ck)
    for stray in STRAY_THINKING_FIELDS:
        if stray in gen:
            ck.error(f"{name}: `{stray}` must live under thinking_config, not top level")


def check_models_and_generation(docs: list[tuple[Path, dict]], ck: Checker) -> None:
    models: set[str] = set()
    for path, data in docs:
        name = path.name
        if data.get("model"):
            models.add(MODEL_PROVIDER_PREFIX.sub("", str(data["model"])))
        if isinstance(data.get("generate_content_config"), dict):
            _check_generation_config(name, data["generate_content_config"], ck)
        instruction = data.get("instruction")
        if isinstance(instruction, str) and len(instruction) > MAX_INSTRUCTION_CHARS:
            ck.error(f"{name}: instruction is {len(instruction)} chars (> {MAX_INSTRUCTION_CHARS:,})")

    if not models:
        ck.error("no `model:` field found in any agent YAML")
        return
    if len(models) > 1:
        ck.error(f"single-model rule violated: declared models {sorted(models)}")
    for model in sorted(models):
        if model not in ALLOWED_MODEL_NAMES:
            ck.error(f"model {model!r} not in ALLOWED_MODEL_NAMES {sorted(ALLOWED_MODEL_NAMES)}")
    ck.ok(f"declared model(s): {sorted(models)}")


def _check_adapter_dir(adapter_dir: Path, ck: Checker) -> int | None:
    name = adapter_dir.name
    if not SAFE_PATH_COMPONENT.fullmatch(name):
        ck.error(f"unsafe adapter name {name!r} (must match {SAFE_PATH_COMPONENT.pattern})")
    config_path = adapter_dir / "adapter_config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        ck.error(f"{config_path}: invalid JSON: {exc}")
        return None

    rank = int(config.get("r") or 0)
    if rank > MAX_LORA_RANK:
        ck.error(f"adapter {name}: rank {rank} > max_lora_rank {MAX_LORA_RANK}")
    if not list(adapter_dir.glob("adapter_model*.safetensors")):
        ck.error(f"adapter {name}: missing adapter_model.safetensors")
    for extra in adapter_dir.iterdir():
        if extra.suffix.lower() in PICKLE_SUFFIXES:
            ck.error(f"adapter {name}: pickle-based weights not allowed: {extra.name}")
    return rank


def _scan_adapter_payloads(source: Path, ck: Checker) -> tuple[set[str], int]:
    adapters_dir = source / "adapters"
    if not adapters_dir.is_dir():
        return set(), 0

    names: set[str] = set()
    max_rank = 0
    for config in sorted(adapters_dir.glob("*/adapter_config.json")):
        names.add(config.parent.name)
        if (rank := _check_adapter_dir(config.parent, ck)) is not None:
            max_rank = max(max_rank, rank)

    for loose in adapters_dir.glob("*.safetensors"):
        names.add(loose.stem)
        ck.warn(f"loose adapter weight {loose.name}: prefer adapters/<name>/adapter_model.safetensors")
    for stray in adapters_dir.rglob("*"):
        if stray.is_file() and stray.suffix.lower() not in ALLOWED_SUBMISSION_EXTENSIONS:
            ck.error(f"disallowed file under adapters/: {stray.relative_to(source)}")
    return names, max_rank


def check_adapters(source: Path, docs: list[tuple[Path, dict]], ck: Checker) -> None:
    names, max_rank = _scan_adapter_payloads(source, ck)
    referenced = {str(d.get("adapter")) for _, d in docs if d.get("adapter")}

    if missing := referenced - names:
        ck.error(f"agent.yaml references adapter(s) {sorted(missing)} that do not exist under adapters/")
    if unused := names - referenced:
        ck.warn(f"adapters present but not referenced by any agent (still loaded by vLLM): {sorted(unused)}")
    if names:
        ck.ok(f"adapters: {sorted(names)} (max rank {max_rank})")
    else:
        ck.warn("no adapters/ found -> submitting base model with prompt engineering only")


def check_eval_config(source: Path, ck: Checker) -> None:
    candidates = _find_config(source, EVAL_CONFIG_NAMES)
    if not candidates:
        ck.warn("no eval_config.yaml: scorer defaults apply (60 min / 100 tool calls / 500 turns)")
        return
    data = yaml.safe_load(candidates[0].read_text(encoding="utf-8")) or {}
    section = data.get("evaluation") if isinstance(data, dict) else None
    if not isinstance(section, dict):
        ck.error("eval_config.yaml must wrap budgets under an `evaluation:` key")
        return

    if unknown := set(section) - EVAL_KNOWN_KEYS:
        ck.warn(f"eval_config.yaml keys ignored by scorer: {sorted(unknown)}")
    turns = section.get("max_turns")
    if isinstance(turns, int) and turns < 50:
        ck.warn(f"max_turns={turns} is low: turn cap + 3 nudges can end the task before submit_patch")
    tools = section.get("max_tool_calls")
    if isinstance(tools, int) and tools < 40:
        ck.warn(f"max_tool_calls={tools} is low for multi-file fixes")
    minutes = section.get("max_time_minutes")
    if isinstance(minutes, (int, float)) and minutes > 60:
        ck.error("max_time_minutes > 60 exceeds the harness allowance")
    ck.ok(f"eval_config.yaml: {json.dumps(section, sort_keys=True)}")


def build_zip(source: Path, out: Path) -> None:
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in _iter_package_files(source):
            rel = path.relative_to(source)
            if path.is_symlink() or _is_vcs_artifact(rel):
                continue
            zf.write(path, arcname=str(rel))
    print(f"[zip] wrote {out} ({_mebibytes(out.stat().st_size)})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate + package submission.zip")
    parser.add_argument("--source", default="submission", help="agent directory (agent.yaml at its root)")
    parser.add_argument("--output", default="submission.zip")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    source = Path(args.source).resolve()
    if not source.is_dir():
        print(f"[error] source dir not found: {source}")
        return 2

    print(f"[validate] {source}")
    ck = Checker()
    root_cfg = check_layout(source, ck)
    docs = iter_agent_docs(root_cfg, source)
    print(f"  [ok] agent docs discovered: {len(docs)}")
    check_models_and_generation(docs, ck)
    check_adapters(source, docs, ck)
    check_eval_config(source, ck)

    ck.report()
    if ck.errors:
        print(f"[validate] FAILED ({len(ck.errors)} error(s))")
        return 1
    print("[validate] PASSED")
    if not args.check_only:
        build_zip(source, Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
