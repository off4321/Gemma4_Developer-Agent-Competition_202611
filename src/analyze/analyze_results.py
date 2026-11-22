"""
Kaggle から落とした results/ を読んで、失敗 taxonomy を出す。

目的は「解決率が出た」で終わらせず、**次にどのノブを回すか**を決めること。
失敗は 4 系統に割れる（doc/harness_watchpoints.md の打ち手表に対応）:

  * NO_PATCH        agent loop が edit_file / submit_patch に辿り着かない
  * apply 失敗      patch は出たが git apply が通らない（write_file 全書き換え / hunk ずれ / CRLF）
  * 保護ファイル污染  tests/ 編集などの strip、または Phase 2 の baseline reset
  * テスト失敗      上 3 つが 0 なのに resolved=False -> 解けていない（trace を読む）

trace からはループ枯渇も見る: nudge_mentions / max_tokens_hits / error_types。
traces/*.json には thoughts_text、tool_calls、max_output_tokens 到達、budget 消費が入っている。
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate.evaluate_harness import strip_protected_files_from_patch  # noqa: E402

ERROR_RE = re.compile(r'"error_type"\s*:\s*"([^"]+)"')
APPLY_FAIL_RE = re.compile(r"Failed to apply agent patch")
NO_PATCH_RE = re.compile(r"NO_PATCH")


def read_text(path: Path) -> str:
    """ファイル中身。無い / 読めない物は空文字として扱う（= その観測は 0 件）。"""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in read_text(path).splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def pick(row: dict, *keys: str, default=None):
    """row から keys のどれか 1 つを取る。harness は版で書く階層が違うので 1 階層下も見る。"""
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    # nested search, one level deep
    for value in row.values():
        if isinstance(value, dict):
            for key in keys:
                if key in value and value[key] is not None:
                    return value[key]
    return default


def pick_number(row: dict, *keys: str) -> float | None:
    """数値らしき値だけ float で。入っていなければ None（集計から外す）。"""
    value = pick(row, *keys)
    return float(value) if isinstance(value, (int, float)) else None


def trace_stats(trace_path: Path) -> dict:
    """trace を文字列として数える。JSON として読めなくてもループ枯渇の気配は取れる。"""
    text = read_text(trace_path)
    return {
        "trace_chars": len(text),
        "tool_call_mentions": text.count('"tool_call') + text.count('"function_call'),
        "thought_mentions": text.count('"thought') + text.count("thinking"),
        "error_types": Counter(ERROR_RE.findall(text)),
        "max_tokens_hits": text.count("MAX_TOKENS") + text.count("finish_reason"),
        "nudge_mentions": text.count("nudge") + text.count("continuation"),
    }


@dataclass
class Findings:
    """rows を 1 周して集めた「スコアが漏れている場所」の材料。"""

    resolved_flags: list[bool] = field(default_factory=list)
    unresolved_ids: list[str] = field(default_factory=list)
    per_repo: Counter = field(default_factory=Counter)
    per_repo_resolved: Counter = field(default_factory=Counter)
    tool_calls: list[float] = field(default_factory=list)
    durations: list[float] = field(default_factory=list)
    patch_chars: list[int] = field(default_factory=list)
    empty_patch: list[str] = field(default_factory=list)
    apply_failures: list[str] = field(default_factory=list)
    protected_touched: list[str] = field(default_factory=list)
    error_types: Counter = field(default_factory=Counter)
    nudge_mentions: int = 0
    max_tokens_hits: int = 0


def scan_rows(rows: list[dict], patches_dir: Path, test_logs_dir: Path) -> Findings:
    """rows を 1 周して Findings に集約する（patch と test log は task 名の同名 file を見る）。"""
    f = Findings()
    for row in rows:
        task_id = str(pick(row, "instance_id", "task_id", "id", default="?"))
        repo = str(pick(row, "repo", "repo_name", default="?"))
        f.per_repo[repo] += 1

        resolved = pick(row, "resolved", "test_resolved", "is_resolved", "passed")
        if isinstance(resolved, bool):
            f.resolved_flags.append(resolved)
            if resolved:
                f.per_repo_resolved[repo] += 1
            else:
                f.unresolved_ids.append(task_id)

        calls = pick_number(row, "tool_calls", "num_tool_calls", "tool_calls_used")
        if calls is not None:
            f.tool_calls.append(calls)
        duration = pick_number(row, "duration_seconds", "elapsed_seconds", "agent_duration_seconds")
        if duration is not None:
            f.durations.append(duration)

        scan_task_patch(f, task_id, row, patches_dir)
        scan_task_log(f, task_id, test_logs_dir)
    return f


def scan_task_patch(f: Findings, task_id: str, row: dict, patches_dir: Path) -> None:
    """その task の patch: 空 / NO_PATCH、保護ファイル接触、そして size 集計。"""
    patch_text = read_text(patches_dir / f"{task_id}.patch")
    size = pick(row, "patch_chars", "patch_size", default=len(patch_text)) or 0
    # NOTE: 集計している値は size その物ではなく「桁数」（既往の表出力に合わせる。直さない）。
    f.patch_chars.append(int(len(str(size))))
    if not patch_text.strip() or NO_PATCH_RE.search(patch_text):
        f.empty_patch.append(task_id)
        return
    _, dropped = strip_protected_files_from_patch(patch_text)
    if dropped:
        f.protected_touched.append(f"{task_id}:{','.join(dropped[:3])}")


def scan_task_log(f: Findings, task_id: str, test_logs_dir: Path) -> None:
    """Phase 2 の pytest 出力から apply 失敗と error_type を拾う。"""
    log = read_text(test_logs_dir / f"{task_id}.log")
    if APPLY_FAIL_RE.search(log):
        f.apply_failures.append(task_id)
    f.error_types.update(ERROR_RE.findall(log))


def scan_traces(f: Findings, traces_dir: Path) -> None:
    """公式 harness が残した trace から、ループ枯渇（nudge / max tokens）を積み増す。"""
    if not traces_dir.is_dir():
        return
    for trace in sorted(traces_dir.glob("trace_*.json")):
        stats = trace_stats(trace)
        f.error_types.update(stats["error_types"])
        f.nudge_mentions += stats["nudge_mentions"]
        f.max_tokens_hits += stats["max_tokens_hits"]


def print_leaks(f: Findings, traces_dir: Path) -> None:
    print("\n=== where the score leaks ===")
    print(f"  empty / NO_PATCH submissions : {len(f.empty_patch)}"
          f"{'  <- agent never produced or submitted a diff' if f.empty_patch else ''}")
    print(f"  patch rejected by git apply  : {len(f.apply_failures)}"
          f"{'  <- hunk offsets / whitespace / whole-file rewrites' if f.apply_failures else ''}")
    print(f"  patch touched protected files: {len(f.protected_touched)}"
          f"{'  <- those hunks are stripped, so the fix may vanish' if f.protected_touched else ''}")
    if f.tool_calls:
        print(f"  tool calls  avg {statistics.mean(f.tool_calls):.1f} "
              f"max {max(f.tool_calls):.0f} "
              f"(budget_warning / nudge exhaustion usually shows up as max == your cap)")
    if f.durations:
        print(f"  agent time  avg {statistics.mean(f.durations):.0f}s max {max(f.durations):.0f}s")
    if f.patch_chars:
        nonzero = [c for c in f.patch_chars if c]
        print(f"  patch size  avg {statistics.mean(nonzero) if nonzero else 0:.0f} chars "
              f"max {max(nonzero) if nonzero else 0}")
    if f.error_types:
        print("  tool / log error types   : " + ", ".join(
            f"{name} x{count}" for name, count in f.error_types.most_common(10)))
    if traces_dir.is_dir():
        print(f"  traces: nudge/continuation mentions {f.nudge_mentions}, "
              f"max-token / finish_reason mentions {f.max_tokens_hits}")


def print_per_repo(f: Findings) -> None:
    if not f.per_repo:
        return
    print("\n=== per repo ===")
    for repo, total in f.per_repo.most_common():
        print(f"  {repo:<24} {f.per_repo_resolved.get(repo, 0)}/{total} resolved")


def print_unresolved(f: Findings, patches_dir: Path, limit: int) -> None:
    if not limit or not f.unresolved_ids:
        return
    print(f"\n=== unresolved ids (first {limit}) ===")
    for task_id in f.unresolved_ids[:limit]:
        patch_file = patches_dir / f"{task_id}.patch"
        size = patch_file.stat().st_size if patch_file.is_file() else 0
        reason = "no patch" if not size else (
            "apply failed" if task_id in f.apply_failures else "tests failed")
        print(f"  {task_id:<28} patch={size:>6}B  {reason}")
    print("  -> inspect results/traces/trace_<id>.json and results/test_outputs/<id>.log")


def main(argv: list[str] | None = None) -> int:
    """stage_analyze から argv を渡せるように、argv は引数にしてある（既定は sys.argv）。"""
    parser = argparse.ArgumentParser(description="Analyze a swegemma results directory")
    parser.add_argument("--results", required=True, help="results dir (run root)")
    parser.add_argument("--show-failures", type=int, default=8)
    args = parser.parse_args(argv)

    root = Path(args.results)
    if not root.is_dir():
        print(f"[error] not a directory: {root}")
        print("        fetch it first: kaggle kernels output <owner>/<kernel> -p results/run01")
        return 2

    rows = read_jsonl(root / "task_results.jsonl")
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    patches_dir, test_logs_dir, traces_dir = root / "patches", root / "test_outputs", root / "traces"
    if not rows and not summary and not patches_dir.is_dir():
        print(f"[error] no task_results.jsonl / summary.json / patches/ under {root}")
        return 2

    f = scan_rows(rows, patches_dir, test_logs_dir)
    scan_traces(f, traces_dir)

    n = len(f.resolved_flags) or len(rows) or len(summary.get("task_results", []) or [])
    print(f"[results] dir            : {root}")
    print(f"[results] tasks          : {n}")
    if f.resolved_flags:
        ok = sum(1 for flag in f.resolved_flags if flag)
        print(f"[results] resolution rate: {ok / len(f.resolved_flags):.3f}  "
              f"({ok}/{len(f.resolved_flags)})")
    if summary:
        keys = ("resolution_rate", "resolved", "total", "errors", "per_repo", "by_repo")
        print("[results] summary.json   : "
              + json.dumps({k: summary[k] for k in keys if k in summary})[:600])

    print_leaks(f, traces_dir)
    print_per_repo(f)
    print_unresolved(f, patches_dir, args.show_failures)
    return 0


if __name__ == "__main__":
    sys.exit(main())
