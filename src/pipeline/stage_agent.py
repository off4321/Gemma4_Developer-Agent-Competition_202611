"""
ステージ `agent` = ノートブック Cell 2 の後半 + `make validate`。

ノートブックは sample_submission を /kaggle/working にコピーして書き換える。このパイプラインは
**自分の submission/**（手元）または提出 dataset（Kaggle）を <working>/agent_dir にコピーし、
variant をそのコピーにだけ適用する:

  model 書換え     agent.yaml と sub_agents/*.yaml の `model:`（Cell 2 と同じ挙動）
  adapter_mode=off adapters/ とすべての `adapter:` 行を消す（ベースライン計測）
  budget           eval_config.yaml の `evaluation:` ブロックを書き換える
  sampling         configs/sampling.yaml を書き換える（thinking_* は thinking_config の内側だけ）

submission/ は無変更なので、レビューできるソースと計測された提出物がズレない。
最後に competition 契約（src/package/build_submission.py）を再検査する: ここで落とすと、
40 分の GPU 実行の後に「そもそも提出物として読めない」が発覚する。

初学者向けの注意: YAML の値の後ろの `#` は、前 **に空白が無いとコメントにならない**
（`model: x# comment` は値の一部）。だからこの stage は行内を分割してコメント位置を保つ。
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any

from pipeline.common import ensure_src_on_path, log

ensure_src_on_path()
from package.build_submission import (  # noqa: E402
    Checker, check_adapters, check_eval_config, check_layout, check_models_and_generation,
    iter_agent_docs,
)

ROOT_CONFIG_NAMES = ("agent.yaml", "agent.yml", "root_agent.yaml", "root_agent.yml")
SAFE_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
NESTED_SAMPLING_KEYS = {"thinking_budget", "include_thoughts", "thinking_level"}


# ---------------------------------------------------------------- 提出物の探索
def candidate_submission_dirs(ctx: Any) -> list[Path]:
    """agent.yaml を置きそうな場所を順に並べる（手元 <repo>/submission が最優先）。"""
    explicit = (ctx.cfg.raw or {}).get("submission_dir")
    if explicit:
        return [Path(explicit)]
    slug = os.environ.get("PIPELINE_SUBMISSION_SLUG") or (ctx.cfg.raw or {}).get("submission_slug") \
        or "gemma4-dev-agent-submission"
    user = os.environ.get("KAGGLE_USERNAME", "<username>")
    candidates = [
        ctx.root / "submission",
        Path(f"/kaggle/input/datasets/{user}/{slug}"),
        Path(f"/kaggle/input/{slug}"),
        ctx.paths.working_dir / slug,
        ctx.paths.working_dir / "submission",
        Path("submission"),
    ]
    return candidates


def resolve_submission_dir(ctx: Any) -> Path:
    """探して 1 つ決める。見つからない場合は「どこを試したか」を出す（推測で終わらせない）。"""
    for candidate in candidate_submission_dirs(ctx):
        if not candidate.is_dir():
            continue
        for probe in (candidate, candidate / "submission"):
            if any((probe / name).is_file() for name in ROOT_CONFIG_NAMES):
                log("agent", f"submission dir: {probe}")
                return probe
    tried = ", ".join(str(c) for c in candidate_submission_dirs(ctx))
    raise SystemExit(
        "[agent] agent.yaml が見つからない。試した場所: " + tried +
        "\n        手元: <repo>/submission/agent.yaml に置くこと"
        "\n        Kaggle: 提出 dataset を attach し（make push-submission）、"
        "kernels/eval/kernel-metadata.json の dataset_sources に入れること"
    )


# ---------------------------------------------------------------- YAML の部分書き換え
def _scalar(value: Any) -> str:
    """YAML の scalar 文字列にする（bool は true/false にする: YAML 1.1 は on/off を bool に読む）。"""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def _replace_in(text: str, pattern: re.Pattern, value: Any) -> tuple[str, bool]:
    """1 行の `key: value   # comment` の値部分だけを書き換える。

    pattern は `key:` の位置だけを特定し、行の残りは手作業で分割する。
    こうすると末尾のコメントと桁位置が保たれる（この repo の YAML はコメントで設計判断を
    書いているので、失うと読めなくなる）。
    """
    match = pattern.search(text)
    if not match:
        return text, False
    line_end = text.find("\n", match.end())
    if line_end == -1:
        line_end = len(text)
    tail = text[match.end():line_end]
    annotation = re.search(r"([ \t]+)(#.*)$", tail)
    old_value, gap, comment = tail, "", ""
    if annotation:
        old_value, gap, comment = tail[: annotation.start()], annotation.group(1), annotation.group(2)
    new_value = _scalar(value)
    pad = max(1, len(gap) + len(old_value.strip()) - len(new_value)) if comment else 0
    replacement = f"{match.group(1)} {new_value}{' ' * pad}{comment}".rstrip()
    return text[: match.start(1)] + replacement + text[line_end:], True


def _block_range(text: str, parent: str) -> tuple[int, int] | None:
    """トップレベル key の直下（インデンドされた block）の開始/終端位置。"""
    match = re.search(rf"^{re.escape(parent)}:[ \t]*(?:#[^\n]*)?\n((?:[ \t]+\S[^\n]*\n?)+)", text, re.M)
    if not match:
        return None
    return match.start(1), match.end(1)


def set_yaml_key(path: Path, key: str, value: Any, *, under: str | None = None) -> str:
    """scalar key 1 個をその場で設定する（コメントは保持）。key が無ければ追加する。

    under を与えると、その parent の **内側だけ** を探して書く。
    `thinking_budget` のように `thinking_config:` 直下でなければならない key はこれで強制する。
    """
    if not SAFE_KEY.match(key):
        raise SystemExit(f"[agent] 安全でない YAML key は触らない: {key!r}")
    text = path.read_text(encoding="utf-8")

    if under is None:
        pattern = re.compile(rf"^({re.escape(key)}:)", re.M)
        new_text, done = _replace_in(text, pattern, value)
        if done:
            path.write_text(new_text, encoding="utf-8")
            return "replaced"
        separator = "" if text.endswith("\n") or not text else "\n"
        path.write_text(text + f"{separator}{key}: {_scalar(value)}\n", encoding="utf-8")
        return "appended"

    span = _block_range(text, under)
    if span is None:
        separator = "" if text.endswith("\n") or not text else "\n"
        path.write_text(text + f"{separator}{under}:\n  {key}: {_scalar(value)}\n", encoding="utf-8")
        return "created-parent"
    head, block, tail = text[: span[0]], text[span[0]: span[1]], text[span[1]:]
    pattern = re.compile(rf"^[ \t]+({re.escape(key)}:)", re.M)
    new_block, done = _replace_in(block, pattern, value)
    if done:
        path.write_text(head + new_block + tail, encoding="utf-8")
        return "replaced"
    if not block.endswith("\n"):
        block += "\n"
    path.write_text(head + block + f"  {key}: {_scalar(value)}\n" + tail, encoding="utf-8")
    return "appended-in-block"


def agent_yaml_files(agent_dir: Path) -> list[Path]:
    files = [agent_dir / name for name in ROOT_CONFIG_NAMES if (agent_dir / name).is_file()]
    files += sorted((agent_dir / "sub_agents").glob("*.yaml"))
    return files


# ---------------------------------------------------------------- variant 適用
def apply_model(ctx: Any, agent_dir: Path) -> None:
    """全 agent YAML の `model:` を variant の model に揃える（単一宣言モデル規則のため）。

    `model:` key が **既に有る** ファイルだけ触る。SequentialAgent / LoopAgent は model を
    持たないので、無い key を発明すると提出物として壊れる。
    書き終えたら YAML を再び parse して全 agent の model が一致したか検証する。
    """
    if not ctx.cfg.rewrite_model:
        return
    ensure_src_on_path()
    from package.build_submission import load_yaml

    touched: list[str] = []
    for path in agent_yaml_files(agent_dir):
        text = path.read_text(encoding="utf-8")
        if not re.search(r"^[ \t]*model:", text, re.M):
            continue  # SequentialAgent / LoopAgent には model field が無い。無い key を作らない
        action = set_yaml_key(path, "model", ctx.cfg.model_name)
        touched.append(f"{path.name}({action})")
    log("agent", f"model: -> {ctx.cfg.model_name} in {', '.join(touched) or 'no file'}")

    # 安全検証: 値に接着した `#` は YAML ではコメントにならない（`model: x# note` は値の一部）
    problems = []
    for path in agent_yaml_files(agent_dir):
        try:
            data = load_yaml(path)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{path.name}: {exc}")
            continue
        if not isinstance(data, dict):
            continue
        model = re.sub(r"^(openai|google|hosted_vllm|custom)/", "", str(data.get("model") or ""))
        if model and model != ctx.cfg.model_name:
            problems.append(f"{path.name}: model={model!r}")
    if problems:
        raise SystemExit("[agent] model 書き換え後に model field が不整合: " + "; ".join(problems))


def apply_adapters(ctx: Any, agent_dir: Path) -> None:
    """adapter_mode に応じて adapter を残す / 外す。

    off にすると adapters/ と `adapter:` 行を消す。これは「LoRA 抜きでベースラインを測る」
    ための操作で、後で lora variant と比べるための 1 列目になる（doc/lora_plan.md）。
    """
    adapters_dir = agent_dir / "adapters"
    if ctx.cfg.adapter_mode == "use":
        found = sorted(p.parent.name for p in adapters_dir.glob("*/adapter_config.json"))
        log("agent", f"adapter_mode=use -> adapters 保持: {found or 'adapters 無し'}")
        return
    shutil.rmtree(adapters_dir, ignore_errors=True)
    for path in agent_yaml_files(agent_dir):
        text = re.sub(r"^\s*adapter:\s*[^\n#]*\n?", "", path.read_text(encoding="utf-8"), flags=re.M)
        path.write_text(text, encoding="utf-8")
    log("agent", "adapter_mode=off -> adapters/ と `adapter:` 行を削除した")


def apply_budget(ctx: Any, agent_dir: Path) -> None:
    """variant の budget を staged eval_config.yaml の `evaluation:` block に書く。

    `verification_timeout_seconds` だけはスコア側が 300 s 固定なので書かない
    （HARNESS_README 7.1）。kernel / 手元のループにだけ効く値だと誤解されると困るのでログに出す。
    """
    target = agent_dir / "eval_config.yaml"
    if not target.is_file():
        log("agent", "[warn] staged agent dir に eval_config.yaml が無い。スコア側は"
                     " default（60 min / 100 tool calls / 500 turns）を使う")
        return
    notes = []
    for key, value in ctx.cfg.budget.items():
        if key == "verification_timeout_seconds":
            notes.append("verification_timeout_seconds はスコア側で 300 s 固定"
                         "（HARNESS_README 7.1）。kernel / 手元のループにだけ効く")
            continue
        action = set_yaml_key(target, key, value, under="evaluation")
        log("agent", f"eval_config.yaml evaluation.{key} = {value} ({action})")
    for note in notes:
        log("agent", "[info] " + note)


def apply_sampling(ctx: Any, agent_dir: Path) -> None:
    """variant の sampling を configs/sampling.yaml に書く。

    `thinking_budget` / `include_thoughts` / `thinking_level` は `thinking_config:` の
    **内側**に置かないと契約違反になる（トップレベルに置くと build_submission が reject する）。
    """
    target = agent_dir / "configs" / "sampling.yaml"
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# generated by src/pipeline/stage_agent.py\n", encoding="utf-8")
        log("agent", f"[warn] {target.relative_to(agent_dir)} を生成した")
    for key, value in ctx.cfg.sampling.items():
        under = "thinking_config" if key in NESTED_SAMPLING_KEYS else None
        action = set_yaml_key(target, key, value, under=under)
        log("agent", f"sampling.yaml {under + '.' if under else ''}{key} = {value} ({action})")


def run_contract_check(agent_dir: Path) -> None:
    """提出契約（src/package/build_submission.py）を staged dir に対して通す。ここを通らなければ次へ行かない。"""
    checker = Checker()
    root_cfg = check_layout(agent_dir, checker)
    docs = iter_agent_docs(root_cfg, agent_dir)
    check_models_and_generation(docs, checker)
    check_adapters(agent_dir, docs, checker)
    check_eval_config(agent_dir, checker)
    for warn in checker.warnings:
        log("agent", f"[warn] {warn}")
    if checker.errors:
        for err in checker.errors:
            log("agent", f"[ERROR] {err}")
        raise SystemExit(f"[agent] 契約検査が FAILED（{len(checker.errors)} 件）"
                         f" on staged dir {agent_dir}")
    log("agent", f"契約検査 PASSED（agent doc {len(docs)} 個）")


# ---------------------------------------------------------------- stage
def run(ctx: Any) -> None:
    source = resolve_submission_dir(ctx)
    target = ctx.paths.working_dir / "agent_dir"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", ".git"))
    log("agent", f"staged {source} -> {target}")

    apply_model(ctx, target)
    apply_adapters(ctx, target)
    apply_budget(ctx, target)
    apply_sampling(ctx, target)
    run_contract_check(target)

    ctx.agent_dir = target
    ctx.notes["submission_source"] = str(source)
