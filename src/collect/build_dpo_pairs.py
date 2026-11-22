#!/usr/bin/env python3
"""Build DPO preference pairs (`prompt`, `chosen`, `rejected`) from eval traces and/or gold tasks.

Two pairing sources are supported so DPO works even when 4B solves few tasks at first:
  1. Self-play pairs: for the same task, `resolved=True` trace (chosen) vs `resolved=False` trace (rejected).
  2. Gold-fallback pairs (--tasks): when no resolved trace exists for a task, pair the gold-patch
     synthetic trajectory (chosen) against a failed/looping/NO_PATCH trace (rejected).

Role filtering (--role main | analyzer) supports Multi-LoRA:
  - main     : full coding/editing/testing/submit_patch trajectories (default)
  - analyzer : read-only exploration + root-cause report trajectories (no edit_file/submit_patch)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collect.collect_trajectories import build_trajectory, normalize_trace  # noqa: E402

ANALYZER_TOOLS = {"read_file", "run_command", "search_similar_code", "get_code_neighbors", "get_code_subgraph"}
DEFAULT_BUDGET = {"max_tool_calls": 80, "max_turns": 150, "max_time_minutes": 45.0}


def _task_key(path: Path, traj: dict[str, Any]) -> str:
    iid = str(traj.get("meta", {}).get("instance_id") or path.stem)
    return iid.removeprefix("trace_")


def _matches_role(messages: list[dict[str, Any]], role: str) -> bool:
    if role == "main":
        return True
    tool_names = {
        tc.get("function", {}).get("name")
        for m in messages
        for tc in (m.get("tool_calls") or [])
    }
    # Analyzer should only use read-only exploration tools (never edit_file / submit_patch).
    return bool(tool_names) and tool_names.issubset(ANALYZER_TOOLS)


def _split_prompt_and_response(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split conversation into (prompt=[system, user], completion=[assistant/tool turns...])."""
    for idx, msg in enumerate(messages):
        if msg.get("role") == "assistant":
            return messages[:idx], messages[idx:]
    return messages[:2], messages[2:]


def _score_trace(traj: dict[str, Any]) -> tuple[int, int]:
    """Rank traces so shorter successful trajectories are preferred as `chosen`."""
    resolved = 1 if traj.get("meta", {}).get("resolved") is True else 0
    return (resolved, -len(traj.get("messages") or []))


def _load_traces(results_dirs: list[Path], role: str) -> dict[str, dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: {"ok": [], "fail": []})
    for root in results_dirs:
        if not root.exists():
            continue
        files = sorted(root.rglob("trace_*.json")) or sorted(root.rglob("*.json"))
        for path in files:
            traj = normalize_trace(path)
            if not traj or not _matches_role(traj["messages"], role):
                continue
            bucket = "ok" if traj["meta"].get("resolved") is True else "fail"
            grouped[_task_key(path, traj)][bucket].append(traj)
    return grouped


def _load_gold_trajectories(tasks_path: Path, system_prompt_path: Path, role: str) -> dict[str, dict[str, Any]]:
    if role == "analyzer" or not tasks_path.is_file() or not system_prompt_path.is_file():
        return {}
    sys_prompt = system_prompt_path.read_text(encoding="utf-8")
    gold: dict[str, dict[str, Any]] = {}
    for line in tasks_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        task = json.loads(line)
        traj = build_trajectory(task, sys_prompt, DEFAULT_BUDGET, native_tools=True, max_edits=6, max_hunk_lines=80)
        if traj:
            gold[str(task["instance_id"])] = traj
    return gold


def build_pairs(
    grouped: dict[str, dict[str, list[dict[str, Any]]]],
    gold_by_id: dict[str, dict[str, Any]],
    max_pairs_per_task: int = 2,
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for task_id, buckets in sorted(grouped.items()):
        failed = sorted(buckets["fail"], key=lambda t: len(t["messages"]), reverse=True)
        if not failed:
            continue
        chosen_pool = sorted(buckets["ok"], key=_score_trace, reverse=True)
        source = "self_play"
        if not chosen_pool and task_id in gold_by_id:
            chosen_pool = [gold_by_id[task_id]]
            source = "gold_fallback"
        if not chosen_pool:
            continue

        chosen_traj = chosen_pool[0]
        prompt, chosen_resp = _split_prompt_and_response(chosen_traj["messages"])
        for rej_traj in failed[:max_pairs_per_task]:
            _, rejected_resp = _split_prompt_and_response(rej_traj["messages"])
            if not chosen_resp or not rejected_resp:
                continue
            pairs.append({
                "prompt": prompt,
                "chosen": chosen_resp,
                "rejected": rejected_resp,
                "meta": {"instance_id": task_id, "source": source},
            })
    return pairs


def main() -> int:
    p = argparse.ArgumentParser(description="Build DPO preference pairs from eval traces")
    p.add_argument("--results", nargs="+", required=True, help="one or more results/ or traces/ directories")
    p.add_argument("--tasks", default="data/tasks.jsonl", help="optional tasks.jsonl for gold-fallback chosen")
    p.add_argument("--system-prompt", default="submission/prompts/system.md")
    p.add_argument("--role", choices=["main", "analyzer"], default="main")
    p.add_argument("--max-pairs-per-task", type=int, default=2)
    p.add_argument("--output", default="data/dpo_pairs.jsonl")
    args = p.parse_args()

    grouped = _load_traces([Path(r) for r in args.results], args.role)
    gold = _load_gold_trajectories(Path(args.tasks), Path(args.system_prompt), args.role)
    pairs = build_pairs(grouped, gold, max_pairs_per_task=args.max_pairs_per_task)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in pairs:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    self_play = sum(1 for r in pairs if r["meta"]["source"] == "self_play")
    print(f"[dpo-data] role={args.role} tasks_with_traces={len(grouped)} pairs={len(pairs)} "
          f"(self_play={self_play}, gold_fallback={len(pairs) - self_play}) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
