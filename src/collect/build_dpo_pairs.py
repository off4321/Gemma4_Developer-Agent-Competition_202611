#!/usr/bin/env python3
"""Build DPO preference pairs (`prompt`, `chosen`, `rejected`) from eval traces and/or gold tasks.

Two pairing sources are supported so DPO works even when 4B solves few tasks at first:
  1. Self-play pairs: for the same task, `resolved=True` trace (chosen) vs `resolved=False` trace (rejected).
  2. Gold-fallback pairs (--tasks): when no resolved trace exists for a task, pair the gold-patch
     synthetic trajectory (chosen) against a failed/looping/NO_PATCH trace (rejected).

Role filtering (--role main | analyzer | verifier) supports Multi-LoRA:
  - main     : full coding/editing/testing/submit_patch trajectories (default)
  - analyzer : read-only exploration + root-cause report trajectories (no edit_file/submit_patch)
  - verifier : run the targeted tests and report VERDICT (no edit_file / submit_patch)

The gold fallback (`--tasks`) works for every role: `build_trajectory(..., role=...)` synthesizes
the matching chosen-side trajectory, with `submission/prompts/{system,analyzer,verifier}.md` as the
system prompt of that role.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collect.collect_trajectories import (  # noqa: E402
    DEFAULT_MAX_HUNK_LINES,
    _load_tool_names,
    build_trajectory,
    normalize_trace,
    resolve_system_prompt,
)

ANALYZER_TOOLS = {"read_file", "run_command", "search_similar_code", "get_code_neighbors", "get_code_subgraph"}
VERIFIER_TOOLS = {"run_command", "read_file", "get_status"}
ROLE_TOOLS: dict[str, set[str]] = {"analyzer": ANALYZER_TOOLS, "verifier": VERIFIER_TOOLS}
# DEFAULT_BUDGET は submission/eval_config.yaml と揃える（教材 user prompt に書き写す値なので
# 80/150 の旧既定は使わない）。
DEFAULT_BUDGET = {"max_tool_calls": 45, "max_turns": 60, "max_time_minutes": 45.0}
# loop 認定: 同じ (tool, args) をこれだけ繰り返す/错误を繰り返す trace は「進まない agent」の rejected にする。
LOOP_MIN_REPEAT_CALLS = 3
LOOP_MIN_FAILED_STEPS = 4


def _task_key(path: Path, traj: dict[str, Any]) -> str:
    iid = str(traj.get("meta", {}).get("instance_id") or path.stem)
    return iid.removeprefix("trace_")


def _role_matches(result: dict[str, Any] | list[dict[str, Any]], role: str) -> bool:
    """Trajectory-level tool filter: `main` takes everything, sub-agents only their own tool set.

    `result` is a normalized trace (`{"messages": [...]}`), a bare message list, or raw trace steps,
    so ATIF traces are matched by the same code as ChatML ones. The check spans the whole
    trajectory: the closing report turn of an analyzer / verifier carries no tool call at all, so a
    per-message check would drop the answer we want to train on.
    """
    if role == "main":
        return True
    messages = result.get("messages", result) if isinstance(result, dict) else result
    tool_names = _load_tool_names(messages)
    # analyzer / verifier must stay inside their own tool set: a trace that edited files or
    # called submit_patch belongs to the main agent, never to these sub-agents.
    return bool(tool_names) and set(tool_names) <= ROLE_TOOLS[role]


def _matches_role(messages: list[dict[str, Any]], role: str) -> bool:
    """Messages-level wrapper, kept for callers that already unpacked `messages`."""
    return _role_matches({"messages": messages}, role)


def _split_prompt_and_response(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split conversation into (prompt=[system, user], completion=[assistant/tool turns...]).

    assistant / tool メッセージは `tool_calls` / `tool_call_id` / `name` をそのまま残す（train_lora.py
    が supervised なしに読む必須項目なので、ここで落とすと DPO が tool 呼び出しを学べない）。
    """
    for idx, msg in enumerate(messages):
        if msg.get("role") == "assistant":
            return messages[:idx], messages[idx:]
    return messages[:2], messages[2:]


def _call_signature(call: dict[str, Any]) -> str:
    """assistant の tool_call 1 件を「同じ呼び出し」判定用のシグネチャにする。"""
    fn = call.get("function") or {}
    name = str(fn.get("name") or call.get("name") or "")
    args = fn.get("arguments", call.get("arguments"))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return f"{name}({args.strip()[:200]})"
    return f"{name}({json.dumps(args, sort_keys=True, ensure_ascii=False)[:200]})"


def _response_signals(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """1つの completion（assistant/tool の並び）から loop / 失敗の信号を抽出する。

    messages に tool 応答が混在する形式（`role="tool"` + `tool_call_id`）でも、assistant メッセージ
    内に call と result が同居する形式でも動く。observation を入力の 1 つとして loop を認定する
    ので、tool 応答が無いと偽の loop は認定できない。
    """
    signatures: list[str] = []
    failures = 0
    for message in messages:
        for call in message.get("tool_calls") or []:
            signatures.append(_call_signature(call))
        content = message.get("content")
        if isinstance(content, str):
            lowered = content.lower()
            if message.get("role") in ("tool", "user") and (
                    '"status": "error"' in lowered or "traceback" in lowered or "error" in lowered):
                failures += 1
        if isinstance(message.get("observations"), list):
            for obs in message["observations"]:
                if str(obs.get("outcome", "")).lower() in ("fail", "error", "failure"):
                    failures += 1
    repeated = {}
    for signature in signatures:
        repeated[signature] = repeated.get(signature, 0) + 1
    worst_name, worst_count = "", 0
    for signature, count in repeated.items():
        if count > worst_count:
            worst_name, worst_count = signature.split("(", 1)[0], count
    edit_repeats = max([c for s, c in repeated.items() if s.startswith(("edit_file", "write_file"))] or [0])
    return {
        "tool_calls": len(signatures),
        "repeated_calls": worst_count,
        "repeated_tool": worst_name,
        "edit_file_repeats": edit_repeats,
        "failed_steps": failures,
    }


def _rejection_reason(trace: dict[str, Any], signals: dict[str, Any]) -> str | None:
    """この trace を rejected に使っていい理由。None なら「失敗の証拠が無い」ので pair にしない。"""
    resolved = trace.get("meta", {}).get("resolved")
    if resolved is True:
        return None
    if signals["repeated_calls"] >= LOOP_MIN_REPEAT_CALLS:
        return "repeat_loop"
    if signals["edit_file_repeats"] >= 2:
        return "edit_file_repeat"
    if signals["failed_steps"] >= LOOP_MIN_FAILED_STEPS:
        return "error_storm"
    if resolved is False:
        return "unresolved"
    return None


def _last_assistant_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant" and isinstance(message.get("content"), str):
            return message["content"].strip()
    return ""


def _ends_with_no_patch(messages: list[dict[str, Any]]) -> bool:
    return _last_assistant_text(messages) == "NO_PATCH"


def _requires_code_change(messages: list[dict[str, Any]]) -> bool:
    """chosen 側が edit_file / write_file を 1 回でも含めば「本当に code 変更が要る task」。"""
    return any(str((call.get("function") or {}).get("name") or call.get("name") or "").startswith(("edit_", "write_"))
               for message in messages for call in message.get("tool_calls") or [])



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
            if not traj or not _role_matches(traj, role):
                continue
            bucket = "ok" if traj["meta"].get("resolved") is True else "fail"
            grouped[_task_key(path, traj)][bucket].append(traj)
    return grouped


def _load_gold_trajectories(tasks_path: Path, role: str, system_prompt: str | None = None,
                            ) -> dict[str, dict[str, Any]]:
    """tasks.jsonl から role 用の gold chosen を作る（analyzer / verifier も編集はしない）."""
    if not tasks_path.is_file():
        return {}
    sys_prompt, _ = resolve_system_prompt(role, system_prompt)
    gold: dict[str, dict[str, Any]] = {}
    for line in tasks_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        task = json.loads(line)
        traj = build_trajectory(task, sys_prompt, DEFAULT_BUDGET, native_tools=True, max_edits=6,
                                max_hunk_lines=DEFAULT_MAX_HUNK_LINES, role=role)
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
        # rejected 候補は「失敗の証拠」がある trace だけ（signals が無い trace は pair にしない）。
        scored = []
        for trace in buckets["fail"]:
            signals = _response_signals(trace["messages"])
            reason = _rejection_reason(trace, signals)
            if reason:
                scored.append((signals["repeated_calls"], signals["failed_steps"], reason, trace, signals))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        if not scored:
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
        if not chosen_resp:
            continue
        chosen_has_action = _requires_code_change(chosen_resp)
        for _, _, reason, rej_traj, signals in scored[:max_pairs_per_task]:
            _, rejected_resp = _split_prompt_and_response(rej_traj["messages"])
            # rejected は assistant で終わらせる。末尾が tool 応答なら、その観測を待たずにやめた
            # 地点より後の tool 応答を落として使う（後半は学習対象にしない、という意味づけ）。
            while rejected_resp and rejected_resp[-1].get("role") != "assistant":
                rejected_resp = rejected_resp[:-1]
            if not rejected_resp:
                continue
            if _ends_with_no_patch(rejected_resp) and not chosen_has_action:
                # NO_PATCH が正解だった task に「NO_PATCH は悪い」と教える pair は作らない。
                continue
            if _last_assistant_text(chosen_resp) and _last_assistant_text(chosen_resp) == _last_assistant_text(rejected_resp):
                continue  # 最后の応答が同一なら preference を教える対象が無い
            pairs.append({
                "prompt": prompt,
                "chosen": chosen_resp,
                "rejected": rejected_resp,
                "meta": {"instance_id": task_id, "source": source, "rejection_reason": reason, **signals},
            })
    return pairs


def main() -> int:
    p = argparse.ArgumentParser(description="Build DPO preference pairs from eval traces")
    p.add_argument("--results", nargs="+", required=True, help="one or more results/ or traces/ directories")
    p.add_argument("--tasks", default="data/tasks.jsonl", help="optional tasks.jsonl for gold-fallback chosen")
    p.add_argument("--system-prompt", default=None,
                   help="override the system prompt used for the gold fallback (default: per --role)")
    p.add_argument("--role", choices=["main", "analyzer", "verifier"], default="main")
    p.add_argument("--max-pairs-per-task", type=int, default=2)
    p.add_argument("--output", default=None,
                   help="output jsonl (default: data/dpo_pairs.jsonl, data/dpo_pairs_<role>.jsonl "
                        "for analyzer / verifier)")
    args = p.parse_args()

    grouped = _load_traces([Path(r) for r in args.results], args.role)
    gold = _load_gold_trajectories(Path(args.tasks), args.role, args.system_prompt)
    pairs = build_pairs(grouped, gold, max_pairs_per_task=args.max_pairs_per_task)

    default_output = "data/dpo_pairs.jsonl" if args.role == "main" else f"data/dpo_pairs_{args.role}.jsonl"
    out = Path(args.output or default_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in pairs:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    self_play = sum(1 for r in pairs if r["meta"]["source"] == "self_play")
    reasons: dict[str, int] = defaultdict(int)
    for row in pairs:
        reasons[str(row["meta"].get("rejection_reason"))] += 1
    print(f"[dpo-data] role={args.role} tasks_with_traces={len(grouped)} pairs={len(pairs)} "
          f"(self_play={self_play}, gold_fallback={len(pairs) - self_play}) -> {out}")
    print("[dpo-data] rejection_reason:", dict(reasons) or "{}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
