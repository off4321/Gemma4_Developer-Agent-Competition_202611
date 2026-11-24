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

Two joins keep the labels honest, because the harness splits the truth over two files:
  * `resolved` lives in `task_results.jsonl`, never in the ATIF `trace_*.json`, so `_load_traces`
    joins it back in (`--task-results`). Without it every episode of a run - including the two
    that passed - would be trained on as a `rejected` example.
  * the failure this run actually produced is a per-tool one (`edit_file` -> `FileEditError`), so
    `_response_signals` attributes every observation to the call that produced it and ranks the
    edit-spiral traces first on the `rejected` side.
"""

from __future__ import annotations

import argparse
import json
import shlex
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
MAIN_TOOLS = {
    "run_command", "get_status", "submit_patch", "read_file", "edit_file", "write_file",
    "get_code_neighbors", "search_similar_code", "get_code_subgraph",
    "code_analyzer", "code_analyzer_agent", "agent_tool",
    "load_skill", "list_skills", "load_skill_resource", "run_skill_script",
}
SKILL_META_TOOLS = {"load_skill", "list_skills", "load_skill_resource", "run_skill_script"}
ROLE_TOOLS: dict[str, set[str]] = {"analyzer": ANALYZER_TOOLS, "verifier": VERIFIER_TOOLS}
# DEFAULT_BUDGET は submission/eval_config.yaml と揃える（教材 user prompt に書き写す値なので
# 80/150 の旧既定は使わない）。
DEFAULT_BUDGET = {"max_tool_calls": 45, "max_turns": 60, "max_time_minutes": 45.0}
# loop 認定: 同じ (tool, args) をこれだけ繰り返す/错误を繰り返す trace は「進まない agent」の rejected にする。
LOOP_MIN_REPEAT_CALLS = 3
LOOP_MIN_FAILED_STEPS = 4
# 2 failed `edit_file` calls in one episode is the spiral this run died on: 189 of its 230
# `edit_file` responses were errors (125 "old_string not found", 23 "no changes were requested",
# 18 "Source path ..." / "Target string occurs N times"). It is the most teachable `rejected`
# signal we have, so it outranks a generic failure storm.
LOOP_MIN_EDIT_ERRORS = 2
TASK_RESULTS_NAME = "task_results.jsonl"
EDIT_CALLS = ("edit_file", "write_file")


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


def _call_name(call: dict[str, Any]) -> str:
    fn = call.get("function") or {}
    return str(fn.get("name") or call.get("name") or "")


def _observation_failed(text: str) -> bool:
    """1 件の tool 応答テキストが「呼び出し自体の失敗」を返したか。

    JSON なら `status` を見る（harness は `{"status": "error", "error_type": "FileEditError", ...}`
    を返す）。`status: ok` の応答は、中のコマンドが exit 1 で終わっても（grep が空、pytest が落ち
    た）「呼び出しは成功・結果は要修正」なので失敗に数えない。さもないと water を読んだだで
    error_storm になる。
    """
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            status = str(payload.get("status") or "").strip().lower()
            if status:
                return status not in ("ok", "okay", "success", "succeeded", "true")
            return any(key in payload for key in ("error", "error_type", "error_message"))
    lowered = text.lower()
    return '"status": "error"' in lowered or "traceback (most recent call last)" in lowered


def _response_signals(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """1つの completion（assistant/tool の並び）から loop / 失敗の信号を抽出する。

    messages に tool 応答が混在する形式（`role="tool"` + `tool_call_id`）でも、assistant メッセージ
    内に call と result が同居する形式でも動く。observation を入力の 1 つとして loop を認定する
    ので、tool 応答が無いと偽の loop は認定できない。

    応答がどの tool のものかは FIFO で対応づける: ATIF の step は `tool_calls` と `observation` が
    同じ step に居て、`normalize_trace` は assistant(tool_calls) の直後に tool を 1 件出すため、
    開いていない一番古い call が次の応答の持ち主になる（これで `FileEditError` を edit_file の
    失敗として数えられる）。
    """
    signatures: list[str] = []
    pending: list[str] = []
    for message in messages:
        for call in message.get("tool_calls") or []:
            signatures.append(_call_signature(call))
            pending.append(_call_name(call))
    failures = 0
    edit_errors = 0
    tool_errors: dict[str, int] = {}

    def consume(text: str) -> None:
        nonlocal failures, edit_errors
        name = pending.pop(0) if pending else ""
        if not _observation_failed(text):
            return
        failures += 1
        key = name or "unknown"
        tool_errors[key] = tool_errors.get(key, 0) + 1
        if key in EDIT_CALLS:
            edit_errors += 1

    for message in messages:
        content = message.get("content")
        if message.get("role") == "tool" and isinstance(content, str):
            consume(content)
        for obs in message.get("observations") or []:
            if isinstance(obs, dict):
                consume(str(obs.get("content") or ""))
            elif isinstance(obs, str):
                consume(obs)

    repeated = {}
    for signature in signatures:
        repeated[signature] = repeated.get(signature, 0) + 1
    worst_name, worst_count = "", 0
    for signature, count in repeated.items():
        if count > worst_count:
            worst_name, worst_count = signature.split("(", 1)[0], count
    edit_repeats = max([c for s, c in repeated.items() if s.startswith(EDIT_CALLS)] or [0])
    return {
        "tool_calls": len(signatures),
        "repeated_calls": worst_count,
        "repeated_tool": worst_name,
        "edit_file_repeats": edit_repeats,
        "failed_steps": failures,
        "edit_errors": edit_errors,
        "tool_errors": tool_errors,
    }


def _rejection_reason(trace: dict[str, Any], signals: dict[str, Any]) -> str | None:
    """この trace を rejected に使っていい理由。None なら「失敗の証拠が無い」ので pair にしない。"""
    resolved = trace.get("meta", {}).get("resolved")
    if resolved is True:
        return None
    if signals["edit_errors"] >= LOOP_MIN_EDIT_ERRORS:
        return "edit_file_repeat"
    if signals["repeated_calls"] >= LOOP_MIN_REPEAT_CALLS:
        return "repeat_loop"
    if signals["failed_steps"] >= LOOP_MIN_FAILED_STEPS:
        return "error_storm"
    if signals["edit_file_repeats"] >= 2:
        return "edit_file_repeat"
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


def _task_results_file(root: Path, explicit: Path | None = None) -> Path | None:
    """The `task_results.jsonl` that belongs to `root` (the harness writes it next to `traces/`)."""
    if explicit is not None:
        return explicit if explicit.is_file() else None
    for candidate in (root / TASK_RESULTS_NAME, root.parent / TASK_RESULTS_NAME):
        if candidate.is_file():
            return candidate
    return None


def _load_task_results(paths: list[Path]) -> dict[str, dict[str, Any]]:
    """1 個以上の `task_results.jsonl` を {instance_id: row} に読む（ATIF trace に無い resolved /
    patch_size の出所）。無いなら {} = 証拠無しなので resolved は推定しない。

    run をまたぐ index を 1 つに混ぜると、runB の `resolved: true` が runA の失敗 trace を解決済み
    にしてしまうので、trace を読む側は `_task_results_for` で dir ごとに引くこと。
    """
    index: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            iid = str(row.get("instance_id") or row.get("task_id") or "").strip()
            if not iid:
                continue
            previous = index.get(iid)
            # 同一 id の行が複数あっても「解決済み」を信じる側（= chosen に使って良い側）を残す。
            if previous is not None and previous.get("resolved") is True and row.get("resolved") is not True:
                continue
            index[iid] = row
    return index


def _task_results_for(root: Path, explicit: Path | None = None) -> dict[str, dict[str, Any]]:
    """`root` 専用の index（run をまたいで resolved を混ぜないための入口）。"""
    path = _task_results_file(root, explicit)
    return _load_task_results([path]) if path else {}


def _apply_task_result(traj: dict[str, Any], iid: str,
                       task_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """`task_results.jsonl` の resolved / patch size を trace meta に写す。"""
    row = task_results.get(iid)
    if not row:
        return traj
    meta = traj.setdefault("meta", {})
    if isinstance(row.get("resolved"), bool):
        meta["resolved"] = row["resolved"]
    for key in ("agent_patch_size", "test_exit_code", "tool_calls", "duration_seconds"):
        if key in row and row[key] is not None:
            meta[key] = row[key]
    if row.get("error"):
        meta["task_error"] = str(row["error"])[:300]
    meta["resolved_source"] = TASK_RESULTS_NAME
    return traj


def _usable_as_chosen(traj: dict[str, Any], role: str) -> bool:
    """resolved=True でも「chosen にして良い trace」と限る。

    `agent_patch_size == 0` の resolved episode は main においては空 diff で通った回（実際
    fastapi_14873: resolved / patch 0 B）で、それを chosen にすると「編集せず submit して良い」と
    教える。サブエージェントは元々 diff を出さないのでこの条件は当てない。
    """
    if role != "main":
        return True
    patch_size = traj.get("meta", {}).get("agent_patch_size")
    return not (isinstance(patch_size, int) and patch_size == 0)


def _load_traces(results_dirs: list[Path], role: str,
                 task_results_file: Path | None = None,
                 ) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Traces -> {instance_id: {"ok": [...], "fail": [...]}}、`task_results.jsonl` を join 済みで。

    `task_results_file` を渡すと全 dir に同じ index を当てる（古い trace に最新の task_results を
    合わせる等）。既定は dir ごとに探すので、runB の resolved が runA の失敗 trace を chosen に
    化する事故が無い。
    """
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: {"ok": [], "fail": []})
    tr_by_dir: dict[Path, dict[str, dict[str, Any]]] = {}
    for root in results_dirs:
        if not root.exists():
            continue
        root_results = _task_results_for(root, task_results_file)
        files = sorted(root.rglob("trace_*.json")) or sorted(root.rglob("*.json"))
        for path in files:
            if path.name in (TASK_RESULTS_NAME, "summary.json"):
                continue
            traj = normalize_trace(path)
            if not traj or not _role_matches(traj, role):
                continue
            iid = _task_key(path, traj)
            local_results = root_results
            if not local_results and task_results_file is None:
                if path.parent not in tr_by_dir:
                    tr_by_dir[path.parent] = _task_results_for(path.parent, None)
                local_results = tr_by_dir[path.parent]
            if local_results:
                _apply_task_result(traj, iid, local_results)
            bucket = "ok" if traj["meta"].get("resolved") is True else "fail"
            grouped[iid][bucket].append(traj)
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
    role: str = "main",
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for task_id, buckets in sorted(grouped.items()):
        # rejected 候補は「失敗の証拠」がある trace だけ（signals が無い trace は pair にしない）。
        # 順序は teachable な順 = 未編集で終わる `edit_file` 失敗 > 同一 call 重複 > tool 失敗数。
        scored = []
        for trace in buckets["fail"]:
            signals = _response_signals(trace["messages"])
            reason = _rejection_reason(trace, signals)
            if reason:
                scored.append((signals["edit_errors"], signals["repeated_calls"],
                               signals["failed_steps"], reason, trace, signals))
        scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        if not scored:
            continue
        chosen_pool = [traj for traj in sorted(buckets["ok"], key=_score_trace, reverse=True)
                       if _usable_as_chosen(traj, role)]
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
        for _, _, _, reason, rej_traj, signals in scored[:max_pairs_per_task]:
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


def _parse_call_args(call: dict[str, Any]) -> dict[str, Any]:
    fn = call.get("function") or {}
    raw = fn.get("arguments", call.get("arguments"))
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _step_call_msg(thought: str, name: str, arguments: dict[str, Any], cid: str) -> dict[str, Any]:
    args_json = json.dumps(arguments, ensure_ascii=False)
    return {
        "role": "assistant",
        "content": thought,
        "tool_calls": [{
            "id": cid,
            "type": "function",
            "name": name,
            "arguments": args_json,
            "function": {"name": name, "arguments": args_json},
        }],
    }


def _is_full_suite_pytest(command: str) -> bool:
    cmd = command.strip()
    if "pytest" not in cmd or "run_pytest_compact.py" in cmd:
        return False
    if ".py" in cmd or " -k " in cmd or " -k=" in cmd:
        return False
    return True


def _trim_prefix_chars(prefix: list[dict[str, Any]], max_chars: int = 36000) -> list[dict[str, Any]]:
    """Keep `[system, user]` + most recent complete `(assistant, tool...)` groups within `max_chars`."""
    if len(prefix) <= 2:
        return prefix
    trimmed = list(prefix)
    while len(trimmed) > 2 and sum(len(json.dumps(m, ensure_ascii=False)) for m in trimmed) > max_chars:
        drop_end = 3
        while drop_end < len(trimmed) and trimmed[drop_end].get("role") == "tool":
            drop_end += 1
        if drop_end >= len(trimmed):
            break
        trimmed = trimmed[:2] + trimmed[drop_end:]
    return trimmed


def _extract_chosen_targets(chosen_messages: list[dict[str, Any]]) -> tuple[list[dict[str, str]], list[str], str]:
    edits: list[dict[str, str]] = []
    files: list[str] = []
    pytest_cmd = "git diff --stat"
    for msg in chosen_messages:
        for call in msg.get("tool_calls") or []:
            cname = _call_name(call)
            cargs = _parse_call_args(call)
            fp = str(cargs.get("filepath") or "").strip()
            if fp and fp not in files and cname in ("edit_file", "write_file", "read_file"):
                files.append(fp)
            if cname == "edit_file" and fp and cargs.get("old_string") and cargs.get("new_string"):
                edits.append({
                    "filepath": fp,
                    "old_string": str(cargs["old_string"]),
                    "new_string": str(cargs["new_string"]),
                })
            elif cname == "run_command":
                cmd = str(cargs.get("command") or "")
                if "pytest" in cmd and ".py" in cmd:
                    for tok in cmd.split():
                        if ".py" in tok and not tok.startswith("-") and "skills/" not in tok:
                            pytest_cmd = f"pytest {shlex.quote(tok)} -q --tb=short"
                            break
    return edits, files, pytest_cmd


def build_step_pairs(
    grouped: dict[str, dict[str, list[dict[str, Any]]]],
    gold_by_id: dict[str, dict[str, Any]],
    max_pairs_per_task: int = 4,
    role: str = "main",
) -> list[dict[str, Any]]:
    """Build single-turn Step-DPO pairs (`chosen` and `rejected` each contain 1 assistant turn, 0 tool turns)."""
    allowed_tools = MAIN_TOOLS if role == "main" else ROLE_TOOLS[role]
    pairs: list[dict[str, Any]] = []

    for task_id, buckets in sorted(grouped.items()):
        scored = []
        for trace in buckets["fail"]:
            signals = _response_signals(trace["messages"])
            reason = _rejection_reason(trace, signals)
            if reason:
                scored.append((signals["edit_errors"], signals["repeated_calls"],
                               signals["failed_steps"], reason, trace, signals))
        scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        if not scored:
            continue

        chosen_pool = [traj for traj in sorted(buckets["ok"], key=_score_trace, reverse=True)
                       if _usable_as_chosen(traj, role)]
        source = "self_play"
        if not chosen_pool and task_id in gold_by_id:
            chosen_pool = [gold_by_id[task_id]]
            source = "gold_fallback"
        if not chosen_pool:
            continue

        chosen_traj = chosen_pool[0]
        base_prompt, chosen_resp = _split_prompt_and_response(chosen_traj["messages"])
        if not chosen_resp:
            continue
        chosen_has_action = _requires_code_change(chosen_resp)
        gold_edits, gold_files, gold_pytest_cmd = _extract_chosen_targets(chosen_resp)

        task_pairs: list[dict[str, Any]] = []
        seen_sigs: set[str] = set()

        for _, _, _, trace_reason, rej_traj, signals in scored:
            if len(task_pairs) >= max_pairs_per_task:
                break
            msgs = list(rej_traj["messages"])
            # Ensure `msgs` starts with `[system, user]` from `base_prompt` even if an ATIF trace omitted them
            first_asst = next((i for i, m in enumerate(msgs) if m.get("role") == "assistant"), len(msgs))
            prefix_roles = [m.get("role") for m in msgs[:first_asst]]
            if base_prompt:
                if "system" not in prefix_roles and "user" not in prefix_roles:
                    msgs = list(base_prompt) + msgs[first_asst:]
                elif msgs and msgs[0].get("role") == "system":
                    msgs = [base_prompt[0]] + msgs[1:]
                elif "system" not in prefix_roles and base_prompt[0].get("role") == "system":
                    msgs = [base_prompt[0]] + msgs

            assistant_indices = [i for i, m in enumerate(msgs) if i >= 1 and m.get("role") == "assistant"]
            last_asst_idx = assistant_indices[-1] if assistant_indices else -1
            any_edit_ok = False
            any_submit = False
            read_by_file: dict[str, list[str]] = defaultdict(list)

            for idx, msg in enumerate(msgs):
                if msg.get("role") != "assistant":
                    continue
                calls = msg.get("tool_calls") or []
                call0 = calls[0] if calls else {}
                cname = _call_name(call0) if calls else ""
                cargs = _parse_call_args(call0) if calls else {}
                content = str(msg.get("content") or "")
                next_msg = msgs[idx + 1] if idx + 1 < len(msgs) else None
                next_obs = str(next_msg.get("content") or "") if (next_msg and next_msg.get("role") == "tool") else ""
                obs_failed = bool(next_obs) and _observation_failed(next_obs)

                if idx >= 1 and msgs[idx - 1].get("role") in ("user", "tool") and len(task_pairs) < max_pairs_per_task:
                    chosen_step: dict[str, Any] | None = None
                    step_reason = ""

                    # 1. Failed edit_file -> teach exact copy from read_file or extract_block.py
                    if role == "main" and cname == "edit_file" and obs_failed:
                        fp = str(cargs.get("filepath") or "").strip()
                        bad_old = str(cargs.get("old_string") or "")
                        matched_gold = next(
                            (ge for ge in gold_edits
                             if ge["filepath"] == fp and ge["old_string"] != bad_old
                             and any(ge["old_string"] in body for body in read_by_file.get(fp, []))),
                            None,
                        )
                        if matched_gold is not None:
                            chosen_step = _step_call_msg(
                                f"Copy the exact verbatim block shown by read_file in {fp} and apply the minimal fix.",
                                "edit_file",
                                {"filepath": fp, "old_string": matched_gold["old_string"],
                                 "new_string": matched_gold["new_string"]},
                                f"step_{idx}",
                            )
                        else:
                            not_found_path = "source path" in next_obs.lower() or "no such file" in next_obs.lower()
                            target_fp = (gold_files[0] if (not_found_path or not fp) and gold_files else fp) or "README.md"
                            ref_old = next((ge["old_string"] for ge in gold_edits if ge["filepath"] == target_fp), bad_old)
                            query_lines = [ln.strip() for ln in ref_old.splitlines() if ln.strip()]
                            query = query_lines[0][:60] if query_lines else ""
                            if query:
                                cmd = f"grep -n -C 4 {shlex.quote(query)} {shlex.quote(target_fp)} | head -n 30"
                                chosen_step = _step_call_msg(
                                    f"Locate the exact line numbers and verbatim text in {target_fp} before calling edit_file.",
                                    "run_command", {"command": cmd}, f"step_{idx}",
                                )
                            else:
                                chosen_step = _step_call_msg(
                                    f"Read {target_fp} first so old_string matches exact file bytes.",
                                    "read_file", {"filepath": target_fp, "start_line": 1, "end_line": 120}, f"step_{idx}",
                                )
                        step_reason = "step_edit_error"

                    # 2. Invalid/disabled tool (e.g. apply_patch, verifier, failed skill lookup) or truncated <|tool_call>
                    elif (
                        (calls and cname not in allowed_tools)
                        or (calls and cname in SKILL_META_TOOLS and obs_failed)
                        or (not calls and "<|tool_call>" in content)
                    ):
                        target_fp = str(cargs.get("filepath") or (gold_files[0] if gold_files else ""))
                        if target_fp:
                            chosen_step = _step_call_msg(
                                f"Keep reasoning short and inspect {target_fp} using a declared tool.",
                                "read_file", {"filepath": target_fp, "start_line": 1, "end_line": 120}, f"step_{idx}",
                            )
                        elif chosen_resp and chosen_resp[0].get("role") == "assistant":
                            chosen_step = chosen_resp[0]
                        step_reason = "step_invalid_tool"

                    # 3. Empty-diff submit_patch when no edit succeeded yet
                    elif role == "main" and cname == "submit_patch" and chosen_has_action and not any_edit_ok:
                        target_fp = gold_files[0] if gold_files else ""
                        if target_fp:
                            chosen_step = _step_call_msg(
                                f"No edit_file succeeded yet so git diff is empty; read {target_fp} and apply the fix before submitting.",
                                "read_file", {"filepath": target_fp, "start_line": 1, "end_line": 120}, f"step_{idx}",
                            )
                        else:
                            chosen_step = _step_call_msg(
                                "Verify non-empty diff with git diff --stat before calling submit_patch.",
                                "run_command", {"command": "git diff --check && git diff --stat"}, f"step_{idx}",
                            )
                        step_reason = "step_empty_submit"

                    # 4. Full-suite pytest (`pytest tests/`) instead of targeted test
                    elif role == "main" and cname == "run_command" and _is_full_suite_pytest(str(cargs.get("command") or "")):
                        chosen_step = _step_call_msg(
                            "Run only the targeted test file instead of the entire test suite.",
                            "run_command", {"command": gold_pytest_cmd}, f"step_{idx}",
                        )
                        step_reason = "step_full_pytest"

                    # 5. Premature text stop / wrong NO_PATCH without submit_patch
                    elif role == "main" and idx == last_asst_idx and not calls and not any_submit:
                        if chosen_has_action or _ends_with_no_patch([msg]):
                            if any_edit_ok:
                                chosen_step = _step_call_msg(
                                    "Verify the diff and call submit_patch instead of stopping without submitting.",
                                    "run_command", {"command": "git diff --check && git diff --stat"}, f"step_{idx}",
                                )
                            elif gold_files:
                                chosen_step = _step_call_msg(
                                    f"Inspect {gold_files[0]} with read_file instead of stopping without a patch.",
                                    "read_file", {"filepath": gold_files[0], "start_line": 1, "end_line": 120}, f"step_{idx}",
                                )
                            step_reason = "step_premature_stop"

                    if chosen_step is not None:
                        sig = f"{step_reason}:{_call_signature(call0) if calls else content[:120]}"
                        if sig not in seen_sigs:
                            seen_sigs.add(sig)
                            prefix = _trim_prefix_chars(msgs[:idx])
                            task_pairs.append({
                                "prompt": prefix,
                                "chosen": [chosen_step],
                                "rejected": [msg],
                                "meta": {
                                    "instance_id": task_id,
                                    "source": source,
                                    "mode": "step",
                                    "rejection_reason": step_reason,
                                    "trace_reason": trace_reason,
                                    "step_index": idx,
                                    **signals,
                                },
                            })

                # Update state after inspecting turn `idx`
                if cname == "read_file" and not obs_failed and next_obs:
                    try:
                        payload = json.loads(next_obs)
                        fp_read = str(cargs.get("filepath") or payload.get("filepath") or "")
                        if fp_read:
                            read_by_file[fp_read].append(str(payload.get("content") or ""))
                    except json.JSONDecodeError:
                        pass
                elif cname in EDIT_CALLS and next_obs and not obs_failed:
                    any_edit_ok = True
                elif cname == "submit_patch":
                    any_submit = True

            # Fallback: if no specific bad step matched on this failed trace, pair first divergent turn
            if not task_pairs and assistant_indices and chosen_resp and chosen_resp[0].get("role") == "assistant":
                first_rej = msgs[assistant_indices[0]]
                first_chosen = chosen_resp[0]
                rej_calls = first_rej.get("tool_calls") or []
                cho_calls = first_chosen.get("tool_calls") or []
                rej_sig = _call_signature(rej_calls[0]) if rej_calls else str(first_rej.get("content") or "")[:120]
                cho_sig = _call_signature(cho_calls[0]) if cho_calls else str(first_chosen.get("content") or "")[:120]
                if rej_sig != cho_sig:
                    task_pairs.append({
                        "prompt": _trim_prefix_chars(msgs[:assistant_indices[0]]),
                        "chosen": [first_chosen],
                        "rejected": [first_rej],
                        "meta": {
                            "instance_id": task_id,
                            "source": source,
                            "mode": "step",
                            "rejection_reason": f"step_first_turn_{trace_reason}",
                            "trace_reason": trace_reason,
                            "step_index": assistant_indices[0],
                            **signals,
                        },
                    })

        pairs.extend(task_pairs[:max_pairs_per_task])
    return pairs


def main() -> int:
    p = argparse.ArgumentParser(description="Build DPO preference pairs from eval traces")
    p.add_argument("--results", nargs="+", required=True, help="one or more results/ or traces/ directories")
    p.add_argument("--tasks", default="data/tasks.jsonl", help="optional tasks.jsonl for gold-fallback chosen")
    p.add_argument("--system-prompt", default=None,
                   help="override the system prompt used for the gold fallback (default: per --role)")
    p.add_argument("--role", choices=["main", "analyzer", "verifier"], default="main")
    p.add_argument("--task-results", default=None,
                   help=f"override the {TASK_RESULTS_NAME} used for resolved / patch size "
                        "(default: look for it in --results and one level up)")
    p.add_argument("--max-pairs-per-task", type=int, default=4)
    p.add_argument("--mode", choices=["step", "trajectory", "both"], default="step",
                   help="step (default): 1-turn Step-DPO pairs without tool observation noise; "
                        "trajectory: full multi-turn completion pairs; both: combine both")
    p.add_argument("--output", default=None,
                   help="output jsonl (default: data/dpo_pairs.jsonl, data/dpo_pairs_<role>.jsonl "
                        "for analyzer / verifier)")
    args = p.parse_args()

    results_dirs = [Path(r) for r in args.results]
    task_results_file = Path(args.task_results) if args.task_results else None
    grouped = _load_traces(results_dirs, args.role, task_results_file)
    gold = _load_gold_trajectories(Path(args.tasks), args.role, args.system_prompt)
    pairs: list[dict[str, Any]] = []
    if args.mode in ("step", "both"):
        pairs.extend(build_step_pairs(grouped, gold, max_pairs_per_task=args.max_pairs_per_task, role=args.role))
    if args.mode in ("trajectory", "both"):
        pairs.extend(build_pairs(grouped, gold, max_pairs_per_task=min(2, args.max_pairs_per_task), role=args.role))

    roots = [root for root in results_dirs if root.exists()]
    resolved = sum(1 for bucket in grouped.values() for traj in bucket["ok"])
    joined = sum(
        1 for root in roots
        if _task_results_file(root, task_results_file)
        or (task_results_file is None and any(root.rglob(TASK_RESULTS_NAME)))
    )
    print(f"[dpo-data] {TASK_RESULTS_NAME}: joined for {joined}/{len(roots)} result dir(s), "
          f"traces_with_resolved_true={resolved}")
    if roots and not joined:
        print(f"[dpo-data] WARNING: no {TASK_RESULTS_NAME} next to these traces -> resolved stays "
              "unknown, every trace counts as a failure, and no self_play pair can be built.")

    default_output = "data/dpo_pairs.jsonl" if args.role == "main" else f"data/dpo_pairs_{args.role}.jsonl"
    out = Path(args.output or default_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in pairs:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    self_play = sum(1 for r in pairs if r["meta"]["source"] == "self_play")
    reasons: dict[str, int] = defaultdict(int)
    edit_errors = 0
    for row in pairs:
        reasons[str(row["meta"].get("rejection_reason"))] += 1
        edit_errors += int(row["meta"].get("edit_errors") or 0)
    print(f"[dpo-data] role={args.role} tasks_with_traces={len(grouped)} pairs={len(pairs)} "
          f"(self_play={self_play}, gold_fallback={len(pairs) - self_play}) -> {out}")
    print("[dpo-data] rejection_reason:", dict(reasons) or "{}")
    print(f"[dpo-data] rejected-side edit_file/write_file errors={edit_errors}, "
          f"file={out.stat().st_size:,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
