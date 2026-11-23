"""
tasks.jsonl（gold patch + test_patch）から SFT 用の軌跡を合成する。

なぜ合成するか: LoRA で直しやすいのは「出力の癖」なので、教材として
**短轨跡で submit_patch まで到達する形** と **edit_file が初回で通る形** を見せるのが最も効く。
実差分に無い文字列を教材にすると、モデルは実在しない old_string を生成する癖を覚える。
だからここでは次を必ず実データから取る:

  * edit_file の old_string / new_string  -> gold `patch` を unified diff から解析した実文字列
  * run_command の pytest target          -> test_patch の `def test_*` + diff のファイルパスから
                                             決まる `path::test_name`（実在する test だけ）

`--role` で 3 LoRA 分の教材を切り替える（既定は `main` = 従来の動作のまま）:

  * main     : edit_file -> run_command(pytest) -> submit_patch まで、親エージェントの軌跡。
  * analyzer : read-only 局所化のみ。run_command(grep) / search_similar_code / read_file を使い、
               最後に `**Target files**` … の報告（analyzer.md の output format）で終える。
               edit_file / write_file / submit_patch は **1 回も出さない**（analyzer に無い tool）。
  * verifier : 検証のみ。get_status -> run_command(ピンポイント pytest + `--tb=line`) ->
               run_command(git status --porcelain) -> `VERDICT: GO` + 5 key の報告で終える。
               同じく edit_file / write_file / submit_patch は使わない。

出力レコードは {"messages": [...], "meta": {...}} の OpenAI / ADK chat 形式。
meta は人間向き（instance_id / repo / base_commit / files_changed / pytest_targets）で、
train_lora.py は messages だけを使う。

--import-traces で results/*/traces/trace_*.json（実行して得た実 trace）も合流できる。
合成データの弱点は tool 観測が理想化されていること（exit_code: 0 前提）なので、
resolved=True の実 trace を混ぜると「失敗観測 -> 修正」の型が入る（doc/lora_plan.md 参照）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

TEST_FUNC_RE = re.compile(r"^\+\s*(?:async\s+)?def (test_[A-Za-z0-9_]+)")
PROTECTED_HINTS = ("conftest.py", "pytest.ini", "pyproject.toml", "setup.cfg", "tox.ini")
SECTION_STARTS = ("diff --git ", "--- ", "Index: ")
HUNK_HEADER_RE = re.compile(r"^@@+ -(\d+)(?:,\d+)? [+](\d+)(?:,\d+)? @@")
SYMBOL_RE = re.compile(r"^\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)")

# hunk 1 個を edit_file 1 発と見なす上限行数。80 line だと 1 turn でファイル単位リライト相当の
# 教材になり、実 harness の max_output_tokens=1024 / old_string 復唱つき edit_file では再現できない。
# 40 line（前後コンテキスト込みで edit_file 1 発が運べる実質上限）超の hunk は task ごと捨てる。
# 教材数を優先するなら `--max-hunk-lines 60` で緩められる（80 に戻すのは避ける）。
DEFAULT_MAX_HUNK_LINES = 40

# submission/eval_config.yaml と揃える budget。教材の user prompt にこの値を書き写すので、
# 80 calls / 150 turns の旧既定は使わない（実運用より甘い前提を覚えてしまう）。
DEFAULT_MAX_TOOL_CALLS = 45
DEFAULT_MAX_TURNS = 60
DEFAULT_MAX_TIME_MINUTES = 45.0

# role ごとに system prompt / tool 集合。main 側の挙動は従来のまま変えない。
ROLE_SYSTEM_PROMPTS = {
    "main": "submission/prompts/system.md",
    "analyzer": "submission/prompts/analyzer.md",
    "verifier": "submission/prompts/verifier.md",
}
ROLE_DEFAULT_OUTPUTS = {
    "main": "data/trajectories.jsonl",
    "analyzer": "data/trajectories_analyzer.jsonl",
    "verifier": "data/trajectories_verifier.jsonl",
}
ANALYZER_ALLOWED_CALLS = {"read_file", "run_command", "search_similar_code",
                          "get_code_neighbors", "get_code_subgraph"}
VERIFIER_ALLOWED_CALLS = {"read_file", "run_command", "get_status"}
FORBIDDEN_SUBAGENT_CALLS = {"edit_file", "write_file", "submit_patch"}
ROLE_ALLOWED_CALLS = {"main": None, "analyzer": ANALYZER_ALLOWED_CALLS, "verifier": VERIFIER_ALLOWED_CALLS}

# --- harness trace (ATIF) reader ---------------------------------------------------------------
# `result*/results/traces/trace_*.json` is written by the harness (`schema_version: ATIF-v1.7`),
# not by this repo:
#
#   {"schema_version": "ATIF-v1.7",
#    "final_metrics": {"total_steps": 14, ...},
#    "steps": [{"step_id": 3, "source": "agent", "message": "...", "extra": {...}},
#              {"step_id": 5, "source": "agent",
#               "tool_calls": [{"tool_call_id": "call_6", "function_name": "run_command",
#                               "arguments": {"command": "..."}}],
#               "observation": {"content": "...", "extra": {...}}}]}
#
# `source` is not a ChatML role, calls are spelled `function_name`, and the tool result is a
# sibling `observation`, so one step expands into an assistant message *and* a tool message.
ATIF_SOURCES = ("system", "user", "agent", "tool")
SOURCE_ROLES = {"system": "system", "user": "user", "agent": "assistant", "tool": "tool"}
CHATML_ROLES = ("system", "user", "assistant", "tool")
ROLE_ALIASES = {"model": "assistant", "ai": "assistant", "function": "tool", "human": "user",
                **SOURCE_ROLES}
STEP_LIST_KEYS = ("messages", "steps", "events", "trajectory")
CALL_LIST_KEYS = ("tool_calls", "toolCalls")
SINGLE_CALL_KEYS = ("tool_call", "function_call")
CALL_NAME_KEYS = ("function_name", "name", "tool_name", "tool_call_name", "functionName")
OBSERVATION_KEYS = ("observations", "observation", "tool_response", "observations_above")
# Call keys worth keeping: everything else a harness attaches to a call (elapsed, timestamps,
# token usage, model version) is bookkeeping, not training signal.
CALL_KEEP_KEYS = ("id", "tool_call_id", "call_id", "type", "name", "arguments", "function")
RESOLVED_KEYS = ("resolved", "test_resolved", "passed")
# Message text keeps arriving with serving-layer artifacts (`<|tool_call>call:` where a structured
# call belonged, `<|"` / `|>` around quoted values); they are not part of any tool grammar we train
# on, so the reader strips them.
MARKER_GARBAGE = ("<|tool_call>call:", "<tool_call>", "</tool_call>", "<|tool_calls|>",
                  "<tool_calls>", "</tool_calls>", "<|tool_call>", "</tool_call>", '<|"', '"|>',
                  "<|", "|>")


def _clean_message_text(text: str) -> str:
    """Message text with serving-layer marker artifacts removed (may come back empty)."""
    cleaned = text
    for garbage in MARKER_GARBAGE:
        cleaned = cleaned.replace(garbage, "")
    return cleaned.strip()


def check_role_calls(role: str, messages: list[dict[str, Any]]) -> None:
    """synthetic trajectory が sub-agent の tool list を超えていないかの自己チェック。

    analyzer / verifier に edit_file / submit_patch が入った教材は、実運用で存在しない tool を
    呼ばせる LoRA になる（HARNESS_README §2.2）ので、ここでビルド側を落とす。
    """
    allowed = ROLE_ALLOWED_CALLS[role]
    if allowed is None:      # main は edit_file / submit_patch を呼んで良い（子専用の検査）
        return
    for message in messages:
        for call in message.get("tool_calls") or []:
            name = call["function"]["name"]
            if name in FORBIDDEN_SUBAGENT_CALLS or (allowed is not None and name not in allowed):
                raise ValueError(f"role={role!r} must not call {name!r}")


# ------------------------------------------------------------------------------ diff parsing
def parse_diff(patch: str) -> list[dict[str, Any]]:
    """Split a unified diff into per-file sections with hunks (context/-/+ lines kept).

    A section opens at `diff --git`, at a bare `--- ` header (the dialect the published gold
    patches use), or at `Index:`. Because `--- /dev/null` is itself a section opener it never
    reaches the body, so `is_new` stays False for every section returned here; `edits_for_file`
    keeps its write_file branch for callers that set the flag themselves.
    """
    files: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    for line in patch.splitlines():
        if line.startswith(SECTION_STARTS):
            match = re.search(r" b/(\S+)", line) if line.startswith("diff --git ") else None
            cur = {"path": match.group(1) if match else None,
                   "hunks": [], "is_new": False, "is_delete": False}
            files.append(cur)
        elif line.startswith("+++ "):
            target = line[4:].strip()
            if cur is not None:
                if target.startswith("b/"):
                    cur["path"] = target[2:]
                if target == "/dev/null":
                    cur["is_delete"] = True
        elif cur is not None:
            if line.startswith("@@"):
                header = HUNK_HEADER_RE.match(line)
                cur["hunks"].append({
                    "body": [],
                    "old_start": int(header.group(1)) if header else 1,
                    "new_start": int(header.group(2)) if header else 0,
                })
            elif cur["hunks"] and line[:1] in {" ", "+", "-", ""}:
                cur["hunks"][-1]["body"].append(line)
    return [f for f in files if f["path"] and f["hunks"] and not f["is_delete"]]


def hunk_to_edit(path: str, body: list[str]) -> tuple[str, str] | None:
    """Turn one hunk body into an (old_string, new_string) pair usable by `edit_file`."""
    old_lines, new_lines = [], []
    for line in body:
        if line.startswith("-"):
            old_lines.append(line[1:])
        elif line.startswith("+"):
            new_lines.append(line[1:])
        elif line.startswith("\\"):
            continue
        else:
            old_lines.append(line[1:] if line else "")
            new_lines.append(line[1:] if line else "")
    old_string = "\n".join(old_lines).strip("\n")
    new_string = "\n".join(new_lines).strip("\n")
    if not new_string and not old_string:
        return None
    return old_string, new_string


def edits_for_file(section: dict[str, Any], max_hunk_lines: int) -> list[dict[str, str]]:
    """1 ファイルの hunk を edit_file / write_file の payload 列に直す。"""
    edits: list[dict[str, str]] = []
    for hunk in section["hunks"]:
        body = hunk["body"]
        if section["is_new"]:
            content = "\n".join(line[1:] for line in body if line.startswith("+"))
            edits.append({"kind": "write_file", "filepath": section["path"], "content": content})
            continue
        if len(body) > max_hunk_lines:
            continue  # giant rewrite: not a realistic single edit_file payload, skip the task
        pair = hunk_to_edit(section["path"], body)
        if pair is None:
            continue
        old_string, new_string = pair
        start_line, end_line = hunk_line_range_pair(hunk)
        edits.append({"kind": "edit_file", "filepath": section["path"],
                      "old_string": old_string, "new_string": new_string,
                      "start_line": start_line, "end_line": end_line})
    return edits


def protected_file(path: str) -> bool:
    """patch から strip される（= 教材にできない）test / runner config 側の file か。"""
    p = path.lower()
    name = p.rsplit("/", 1)[-1]
    return (name.startswith("test_") or name.endswith("_test.py")
            or any(seg in p for seg in ("/tests/", "/test/", "/testing/"))
            or any(h in p for h in PROTECTED_HINTS))


def pytest_targets(test_patch: str, limit: int = 3) -> list[str]:
    """Derive `path::test_name` targets from the graded test patch (order kept, duplicates dropped)."""
    targets: list[str] = []
    cur_path = None
    for line in test_patch.splitlines():
        if line.startswith("+++ "):
            cur_path = line[6:].strip() if line.startswith("+++ b/") else line[4:].strip()
        elif cur_path and (match := TEST_FUNC_RE.match(line)):
            targets.append(f"{cur_path}::{match.group(1)}")
    if not targets:
        # test 名が読めない差分は、触られる test file 自体を target にする
        targets = [line[6:].strip() for line in test_patch.splitlines()
                   if line.startswith("+++ b/") and line[6:].endswith(".py")]
    return list(dict.fromkeys(targets))[:limit]


# ------------------------------------------------------------------------------ trajectory
def assistant_msg(thought: str, call: dict[str, Any], native_tools: bool, call_index: int) -> dict[str, Any]:
    if native_tools:
        return {
            "role": "assistant",
            "content": thought,
            "tool_calls": [{
                "id": f"call_{call_index}",
                "type": "function",
                "function": {"name": call["name"],
                             "arguments": json.dumps(call["arguments"], ensure_ascii=False)},
            }],
        }
    payload = {"name": call["name"], "arguments": call["arguments"]}
    return {"role": "assistant", "content": f"{thought}\n<tool_call>\n{json.dumps(payload, ensure_ascii=False)}\n</tool_call>"}


def tool_msg(call_index: int, observation: dict[str, Any]) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": f"call_{call_index}", "content": json.dumps(observation, ensure_ascii=False)}


def append_call(messages: list[dict[str, Any]], call_index: int, thought: str, call: dict[str, Any],
               observation: dict[str, Any], native_tools: bool) -> None:
    """assistant の発話（tool_call 付き）と tool 応答を 1 組ぶら下げる。"""
    messages.append(assistant_msg(thought, call, native_tools, call_index))
    messages.append(tool_msg(call_index, observation))


def initial_user_prompt(problem_statement: str, repo: str, budget: dict[str, Any]) -> str:
    return (
        f"You are evaluating a software engineering task for repository {repo}.\n\n"
        f"Problem Statement:\n{problem_statement.strip()}\n\n"
        "## Task Budget (Session terminates when any budget is exhausted)\n"
        f"- Time allowance: {budget['max_time_minutes']} minutes\n"
        f"- Tool calls allowance: {budget['max_tool_calls']} calls\n"
        f"- Max loop iterations: {budget['max_turns']} turns\n\n"
        "## Execution Environment Rules\n"
        "- Single command timeout: 300 seconds\n- Command output limit: 5000 characters\n"
        "- File view limit: 150 lines per read_file call\n"
        "- File character limit: 10000 characters per read_file call\n"
        "- Environment is offline (no network/PyPI access). All repository and test "
        "dependencies are ALREADY pre-installed. Do NOT attempt to run pip install.\n"
    )


def read_observation(filepath: str, start_line: int, end_line: int, body: str,
                     limit: int = 1200) -> dict[str, Any]:
    """`read_file` の応答相当（実 harness は全文でなく頼んだ範囲の raw text を返す）。"""
    content = body.strip("\n")
    truncated = len(content) > limit
    return {"status": "ok", "filepath": filepath, "start_line": start_line, "end_line": end_line,
            "content": content[:limit] + ("\n..." if truncated else ""), "truncated": truncated}


def build_trajectory(task: dict[str, Any], system_prompt: str, budget: dict[str, Any],
                     native_tools: bool, max_edits: int, max_hunk_lines: int,
                     role: str = "main") -> dict[str, Any] | None:
    """gold patch 1 件から SFT 用 trajectory を 1 本作る。教材に出来なければ None。

    `role="main"` が従来の挙動（既定）。`analyzer` / `verifier` は sub_agents/*.yaml の tools に
    入っていない edit_file / write_file / submit_patch を絶対に出力しない。
    """
    if role == "analyzer":
        return build_analyzer_trajectory(task, system_prompt, budget, native_tools,
                                         max_edits, max_hunk_lines)
    if role == "verifier":
        return build_verifier_trajectory(task, system_prompt, budget, native_tools,
                                         max_edits, max_hunk_lines)
    sections = parse_diff(task.get("patch") or "")
    sections = [s for s in sections if not protected_file(s["path"])]
    if not sections:
        return None
    edits: list[dict[str, Any]] = []
    for section in sections:
        edits.extend(edits_for_file(section, max_hunk_lines))
    if not edits or len(edits) > max_edits:
        return None

    targets = pytest_targets(task.get("test_patch") or "")
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": initial_user_prompt(
            task.get("problem_statement") or task.get("hints_text") or "", task["repo"], budget)},
    ]

    call_index = 0
    files_changed = {e["filepath"] for e in edits}
    # 各 file の最初の edit_file の前に read_file を 1 発入れる。eval trace は 10 task 0 resolved
    # で、その内訳が「見ていない行を old_string に書いて edit_file が一致しない」巡りだった
    # = edit_file する前に target 範囲を読む、という挙動そのものが未学習だった。
    read_targets: set[str] = set()
    n_reads = 0
    for i, edit in enumerate(edits):
        filepath = edit["filepath"]
        if edit["kind"] == "edit_file" and filepath not in read_targets:
            read_targets.add(filepath)
            n_reads += 1
            start_line = int(edit.get("start_line") or 1)
            end_line = max(int(edit.get("end_line") or start_line), start_line)
            thought = (f"Read the block the issue points at in {filepath}:{start_line}-{end_line}. "
                       "edit_file only matches text I have actually seen."
                       if not i else
                       f"Read {filepath}:{start_line}-{end_line} before editing it there.")
            call_index += 1
            append_call(messages, call_index, thought,
                        {"name": "read_file", "arguments": {
                            "filepath": filepath, "start_line": start_line, "end_line": end_line}},
                        read_observation(filepath, start_line, end_line, edit["old_string"]),
                        native_tools)
        if i:
            thought = f"Apply the remaining change in {filepath}."
        else:
            thought = ("Locate the defect from the issue text. The change belongs in "
                       f"{filepath}. Apply the minimal patch there, then run the targeted test.")
        call_index += 1
        if edit["kind"] == "write_file":
            append_call(messages, call_index, thought,
                        {"name": "write_file", "arguments": {"filepath": filepath,
                                                             "content": edit["content"]}},
                        {"status": "ok", "filepath": filepath, "size": len(edit["content"])},
                        native_tools)
        else:
            append_call(messages, call_index, thought,
                        {"name": "edit_file", "arguments": {
                            "filepath": filepath, "old_string": edit["old_string"],
                            "new_string": edit["new_string"]}},
                        {"status": "ok", "filepath": filepath, "occurrences": 1,
                         "strategy": "exact",
                         "diff": f"--- a/{filepath}\n+++ b/{filepath}\n",
                         "is_truncated": False},
                        native_tools)

    test_cmd = (f"python3 -m pytest {' '.join(targets)} -q --tb=short"
                if targets else "python3 -m pytest -q --tb=short -k changed")
    call_index += 1
    append_call(messages, call_index,
                "Verify with the single test that covers this behaviour, "
                "in /tmp-safe form (no new test files).",
                {"name": "run_command", "arguments": {"command": test_cmd}},
                {"status": "ok", "stdout": "all tests finished without failures",
                 "stderr": "", "exit_code": 0},
                native_tools)

    call_index += 1
    append_call(messages, call_index,
                "Fix verified. No scratch files were left in /workspace, so submit now.",
                {"name": "submit_patch", "arguments": {}},
                {"status": "ok", "patch_size": len(task.get("patch") or ""),
                 "files_changed": len(files_changed)},
                native_tools)
    messages.append({"role": "assistant",
                     "content": "Minimal fix applied and verified with the targeted tests; patch submitted."})

    return {
        "messages": messages,
        "meta": {
            "instance_id": task["instance_id"], "repo": task["repo"],
            "base_commit": task.get("base_commit"), "n_edits": len(edits),
            "n_reads": n_reads,
            "files_changed": sorted(files_changed), "pytest_targets": targets,
            # files_read / files_written / test_targets / test_status は tool 呼び出しから導く
            **_io_summary(messages),
            "test_status": "passed",  # gold 教材の observation は「targeted test が通った」設定
            "source": "gold_patch",
        },
    }


def hunk_line_range_pair(hunk: dict[str, Any]) -> tuple[int, int]:
    """gold hunk 1 件を (start, end) の 1-indexed 行 range にする（`read_file` の target 範囲用）。"""
    body = hunk["body"]
    old_lines = sum(1 for line in body if line[:1] in {" ", "-"}) or 1
    new_lines = sum(1 for line in body if line[:1] in {" ", "+"})
    start = int(hunk.get("old_start") or 1)
    return start, max(start, start + max(old_lines, new_lines) - 1)


def hunk_line_range(hunk: dict[str, Any]) -> str:
    """gold hunk 1 件を `start-end` 行 range にする（analyzer 教材の `path:lines` 用）。"""
    start, end = hunk_line_range_pair(hunk)
    return f"{start}-{end}"


def hunk_symbols(body: list[str]) -> list[str]:
    """hunk 内で触られている def / class 名（root cause 行の語彙にする）。"""
    names: list[str] = []
    for line in body:
        if match := SYMBOL_RE.match(line[1:] if line[:1] in {" ", "+", "-"} else line):
            names.append(match.group(1))
    return list(dict.fromkeys(names))


def gold_hunks(task: dict[str, Any], max_hunk_lines: int) -> list[dict[str, Any]]:
    """gold patch から「教材にできる hunk」だけを取り出す（test / config 変更は除外）。"""
    out: list[dict[str, Any]] = []
    for section in parse_diff(task.get("patch") or ""):
        if protected_file(section["path"]):
            continue
        for hunk in section["hunks"]:
            if len(hunk["body"]) > max_hunk_lines:
                continue
            pair = hunk_to_edit(section["path"], hunk["body"])
            if pair is None:
                continue
            old_string, new_string = pair
            out.append({"path": section["path"], "range": hunk_line_range(hunk),
                        "symbols": hunk_symbols(hunk["body"]),
                        "old_string": old_string, "new_string": new_string})
    return out


def _grep_target(path: str) -> str:
    """grep の検索先。`pkg/mod.py` -> `pkg/`（リポジトリ直下なら `.`）。"""
    parent = path.rsplit("/", 1)[0] if "/" in path else ""
    return parent or "."


def analyzer_user_prompt(problem_statement: str, repo: str, budget: dict[str, Any]) -> str:
    """analyzer に親が投げる依頼文（code_analyzer の AgentTool 相当）。"""
    return (
        f"Repository: {repo} (checked out at the buggy commit, read-only).\n\n"
        f"Task / issue:\n{problem_statement.strip()[:4000]}\n\n"
        "Localize the defect. Report the exact files and line ranges to change, the root cause, "
        "the minimal fix, and the single test that proves it. You cannot edit and cannot submit.\n\n"
        f"Budget handed to you: {budget.get('max_tool_calls', DEFAULT_MAX_TOOL_CALLS)} tool calls, "
        f"{budget.get('max_time_minutes', 45)} minutes.\n"
    )


def verifier_user_prompt(repo: str, budget: dict[str, Any], targets: list[str],
                         files: list[str]) -> str:
    """verifier に親が投げる依頼文（patch は適用済み・test 実行と判定だけ頼む）。"""
    return (
        f"Repo: {repo}. The fix is already applied to the working tree.\n"
        f"Files changed: {', '.join(files) if files else 'unknown'}\n"
        f"Test targets: {', '.join(targets) if targets else 'choose the tests for the changed files'}\n\n"
        "Verify this patch and answer in the fixed report format. Do not edit anything and do not "
        "submit: the parent calls submit_patch after reading your verdict.\n"
    )


def _quote_block(value: str, limit: int = 900) -> str:
    text = value.strip("\n")
    return text if len(text) <= limit else text[:limit] + "\n..."


def analyzer_report(hunks: list[dict[str, Any]], targets: list[str]) -> str:
    """gold hunk を「読み終わった後の局所化報告」の形に起こす（analyzer 教材の final answer）。"""
    paths = list(dict.fromkeys(h["path"] for h in hunks))
    symbols = [s for h in hunks for s in h["symbols"][:2]][:4]
    lines = ["**Target files**"]
    for i, h in enumerate(hunks[:4], 1):
        where = f"{h['path']}:{h['range']}"
        what = f"`{h['symbols'][0]}` must change" if h["symbols"] else "this block must change"
        lines.append(f"{i}. `{where}` — {what}")
    lines += [
        "",
        "2. **Root cause**",
        f"`{symbols[0]}` in `{paths[0]}` is where the reported behaviour is produced; " if symbols
        else f"`{paths[0]}` is where the reported behaviour is defined; ",
        "nothing else in the repo reads those lines, so the defect is local to them.",
        "",
        "3. **Minimal change** (a plan for the parent: you never apply it)",
    ]
    for h in hunks[:2]:
        lines += [f"`{h['path']}:{h['range']}`", "```diff",
                  _quote_block(f"- {h['old_string']}"), _quote_block(f"+ {h['new_string']}"), "```"]
    lines += [
        "4. **Test to run**",
        f"`python3 -m pytest {' '.join(targets)} -q --tb=short`" if targets
        else "grep the tests that import those files, then run that file only",
    ]
    if len(paths) > 1:
        lines += ["", "5. **Risk**", f"{len(paths)} files share these symbols "
                                    f"({', '.join(paths)}); change them together."]
    return "\n".join(lines)


def verifier_report(targets: list[str], files: list[str], ran: list[str]) -> str:
    """verifier 教材の final answer（verifier.md の 6 行フォーマットそのもの）."""
    pending = [t for t in targets if t not in ran]
    return "\n".join([
        "VERDICT: GO",
        f'PENDING_TESTS: {", ".join(pending) if pending else "none"}',
        'FAILURE_SUMMARY: none',
        f'FILES_CHANGED: {", ".join(files) if files else "none"}',
        'CONFIDENCE: 0.9',
        'NEXT_ACTION: call submit_patch now; the targeted tests above all pass',
    ])


def build_analyzer_trajectory(task: dict[str, Any], system_prompt: str, budget: dict[str, Any],
                              native_tools: bool, max_edits: int, max_hunk_lines: int) -> dict[str, Any] | None:
    """analyzer LoRA 用: read-only 局所化だけを行う短軌跡（edit / submit は絶対に呼ばない）。"""
    hunks = gold_hunks(task, max_hunk_lines)
    if not hunks or len(hunks) > max_edits:
        return None
    targets = pytest_targets(task.get("test_patch") or "")
    paths = list(dict.fromkeys(h["path"] for h in hunks))
    first = hunks[0]
    symbol = (first["symbols"] or [Path(first["path"]).stem])[-1]

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": analyzer_user_prompt(
            task.get("problem_statement") or task.get("hints_text") or "", task["repo"], budget)},
    ]

    call_index = 0
    # grep の検索語は gold の生コードより記号名にする（教材が「読んだコードをそのまま grep した」
    # 不自然な形になると、実運用で存在しない行を探しに行く癖が出る）。
    needle = symbol or (first["old_string"].splitlines() or [""])[-1].strip().strip(":") \
        or Path(first["path"]).stem
    call_index += 1
    append_call(messages, call_index,
                "Start where the prompt says to start: grep the reported behaviour, no reading yet.",
                {"name": "run_command", "arguments": {
                    "command": f'grep -rn "{needle}" {_grep_target(first["path"])} | head -40'}},
                {"status": "ok", "exit_code": 0,
                 "stdout": "\n".join(f"{h['path']}:{h['range'].split('-')[0]}: {needle}" for h in hunks[:5]),
                 "stderr": ""},
                native_tools)

    call_index += 1
    append_call(messages, call_index,
                "Confirm the owning symbol through the graph before reading the file.",
                {"name": "search_similar_code", "arguments": {
                    "query": f"{symbol} implementation in {first['path']}"}},
                {"status": "ok", "matches": [
                    {"node": f"{Path(h['path']).stem}.{s}" if (s := (h["symbols"] or [None])[0]) else h["path"],
                     "path": h["path"]} for h in hunks[:3]]},
                native_tools)

    for h in hunks[:2]:
        start_line = max(1, int(h["range"].split("-")[0]) - 5)
        call_index += 1
        append_call(messages, call_index,
                    f"Confirm the exact code around {h['path']}:{h['range']} before reporting it.",
                    {"name": "read_file", "arguments": {
                        "filepath": h["path"], "start_line": start_line,
                        "end_line": start_line + 40}},
                    read_observation(h["path"], start_line, start_line + 40, h["old_string"]),
                    native_tools)

    report = analyzer_report(hunks, targets)
    messages.append({"role": "assistant", "content": report})
    check_role_calls("analyzer", messages)
    return {
        "messages": messages,
        "meta": {
            "instance_id": task["instance_id"], "repo": task["repo"],
            "base_commit": task.get("base_commit"), "role": "analyzer",
            "n_edits": len(hunks), "files_changed": paths,
            "pytest_targets": targets, "source": "gold_patch",
        },
    }


def build_verifier_trajectory(task: dict[str, Any], system_prompt: str, budget: dict[str, Any],
                              native_tools: bool, max_edits: int, max_hunk_lines: int) -> dict[str, Any] | None:
    """verifier LoRA 用: 差分の確認 -> budget 確認 -> 限定した pytest -> 報告。

    1 発目の tool call は必ず `run_command` の差分確認（`git status --short`）。submission/prompts/
    verifier.md の rule 2 / Procedure と同じ順番で、差分を見ていない GO を教材にしない。
    edit_file / write_file / submit_patch は絶対に呼ばない。
    """
    hunks = gold_hunks(task, max_hunk_lines)
    if not hunks:
        return None
    targets = pytest_targets(task.get("test_patch") or "")
    files = sorted({h["path"] for h in hunks})
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": verifier_user_prompt(task["repo"], budget, targets, files)},
    ]

    call_index = 0
    max_calls = int(budget.get("max_tool_calls") or DEFAULT_MAX_TOOL_CALLS)
    # 1) 差分の実在確認（verifier はこれより先に何も実行しない）
    call_index += 1
    append_call(messages, call_index,
                "Look at the patch before anything else: an empty tree means there is nothing to "
                "verify, and a GO without a diff would be a lie.",
                {"name": "run_command", "arguments": {"command": "git status --short"}},
                {"status": "ok", "exit_code": 0,
                 "stdout": "\n".join(f" M {f}" for f in files), "stderr": ""},
                native_tools)
    # 2) budget 確認（残りが僅かなら tests を走らせず TOO_SLOW を返す）
    call_index += 1
    append_call(messages, call_index,
                "Budget check before running tests: with almost nothing left I report at once.",
                {"name": "get_status", "arguments": {}},
                {"status": "ok", "tool_calls_used": 6, "tool_calls_remaining": max_calls - 6,
                 "time_used_seconds": 300, "time_remaining_seconds": 2400},
                native_tools)

    called = targets[:2] if targets else []
    if called:
        test_cmd = f"python3 -m pytest {' '.join(called)} -q --tb=line"
    else:
        # targets が無い task でも full suite は走らせない（proven の変更先ディレクトリだけ）。
        test_cmd = f"python3 -m pytest {_grep_target(files[0])} -q --tb=line"
    call_index += 1
    append_call(messages, call_index,
                "Run only the tests named in the request; a bare pytest collects the whole repo.",
                {"name": "run_command", "arguments": {"command": test_cmd}},
                {"status": "ok", "stdout": f"{max(1, len(called))} passed in 1.20s\n",
                 "stderr": "", "exit_code": 0},
                native_tools)

    messages.append({"role": "assistant", "content": verifier_report(targets, files, called)})
    check_role_calls("verifier", messages)
    return {
        "messages": messages,
        "meta": {
            "instance_id": task["instance_id"], "repo": task["repo"],
            "base_commit": task.get("base_commit"), "role": "verifier",
            "n_edits": 0, "files_changed": files,
            "pytest_targets": targets, "source": "gold_patch",
        },
    }


# ------------------------------------------------------------------------------ trace import
def build_no_patch_trajectory(task: dict[str, Any], system_prompt: str, budget: dict[str, Any],
                              native_tools: bool, max_hunk_lines: int) -> dict[str, Any] | None:
    """「設定が repo の外にある」failure を、最終回答 `NO_PATCH` で終わる main 教材にする。

    tasks.jsonl の gold patch は全て repo の code を変えるため、この型の task は実在しない。そこで
    実在する file / line range / graded test をそのまま使い、「code も test も既に正しい」という設定の
    scenario を組み立てる。観測は他の gold 教材と同じく理想化された合成で、最終回答は `NO_PATCH` の
    1 行だけ（submission/prompts/system.md の rule 5 に対応）。DPO の chosen には使わない。
    """
    hunks = gold_hunks(task, max_hunk_lines)
    targets = pytest_targets(task.get("test_patch") or "")
    if not hunks or not targets:
        return None
    hunk, target = hunks[0], targets[0]
    filepath = hunk["path"]
    start_line, end_line = (int(x) for x in hunk["range"].split("-")[:2])
    symbol = (hunk["symbols"] or [Path(filepath).stem])[0]
    env_var = re.sub(r"[^A-Z0-9]+", "_", Path(filepath).stem.upper()).strip("_")[:24] or "APP"
    issue = (f"Deployment job fails in CI: the runner image sets {env_var}_ENDPOINT to a host this "
             f"repository does not control. `{target}` passes locally against the current tree, and "
             f"the failure appeared right after the runner image was bumped.")

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": (
            f"You are evaluating a software engineering task for repository {task['repo']}.\n\n"
            f"Problem Statement:\n{issue}\n\n"
            "Decide whether any file inside this repository can fix it. If the value that has to "
            "change lives only in config outside the repository (runner image, deployment env, "
            "infrastructure), change nothing and answer with exactly NO_PATCH.")},
    ]

    call_index = 0
    call_index += 1
    append_call(messages, call_index,
                f"Find where {env_var}_ENDPOINT is consumed before saying anything about the config.",
                {"name": "run_command",
                 "arguments": {"command": f'grep -rn "{env_var}_ENDPOINT" . | head -40'}},
                {"status": "ok", "exit_code": 0,
                 "stdout": f"./{filepath}:{start_line}:endpoint = "
                           f"os.environ.get(\"{env_var}_ENDPOINT\", DEFAULT_ENDPOINT)\n",
                 "stderr": ""},
                native_tools)
    call_index += 1
    append_call(messages, call_index,
                f"Read the block that consumes it: {filepath}:{start_line}-{end_line}.",
                {"name": "read_file", "arguments": {"filepath": filepath,
                                                    "start_line": start_line, "end_line": end_line}},
                read_observation(filepath, start_line, end_line, hunk["old_string"]),
                native_tools)
    call_index += 1
    append_call(messages, call_index,
                "The code already falls back correctly, so prove it on the graded test.",
                {"name": "run_command",
                 "arguments": {"command": f"python3 -m pytest {target} -q --tb=line | tail -c 3000"}},
                {"status": "ok", "exit_code": 0, "stdout": "1 passed in 0.42s\n", "stderr": ""},
                native_tools)
    call_index += 1
    append_call(messages, call_index,
                "Nothing in the tree needs to change; confirm the working tree stays clean.",
                {"name": "run_command", "arguments": {"command": "git status --short"}},
                {"status": "ok", "exit_code": 0, "stdout": "", "stderr": ""},
                native_tools)

    messages.append({"role": "assistant", "content": "NO_PATCH"})
    return {
        "messages": messages,
        "meta": {
            "instance_id": task["instance_id"], "repo": task["repo"],
            "base_commit": task.get("base_commit"), "role": "main",
            "n_edits": 0, "n_reads": 1,
            "files_changed": [], "files_read": [f"{filepath}:{start_line}-{end_line}"],
            "files_written": [], "test_targets": [target], "test_status": "passed",
            "outcome": "NO_PATCH", "source": "config_only_no_patch",
            "config_scope": "outside-repo (runner image / deployment env)",
        },
    }


def _coerce_payload(payload: Any) -> Any:
    """dict | list | JSON text | file path (str | Path) -> decoded JSON (None when unreadable)."""
    if isinstance(payload, (dict, list)):
        return payload
    if isinstance(payload, Path):
        try:
            return json.loads(payload.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
    if isinstance(payload, str):
        if payload.lstrip()[:1] in ("{", "["):
            try:
                return json.loads(payload)
            except json.JSONDecodeError:
                return None
        return _coerce_payload(Path(payload))
    return None


def _normalize_role(value: Any) -> str:
    """ChatML role spelled by the harness (`agent` -> `assistant`), or "" when not a role."""
    if not isinstance(value, str):
        return ""
    role = ROLE_ALIASES.get(value.strip().lower(), value.strip().lower())
    return role if role in CHATML_ROLES else ""


def _step_text(value: Any, json_fallback: bool = True) -> str:
    """Step / observation text from `message` | `content` | `text` | `output`, never a raw dict.

    Text never keeps the tokenizer marker garbage the serving layer emitted (`<|tool_call>call:`,
    `<|"`, `|>`), and a dict with no text carrier is JSON-encoded only when `json_fallback` is on
    (that is how `observation` payloads are read, while a text-less step stays empty).
    """
    raw = value
    if isinstance(value, dict):
        for key in ("message", "content", "text", "output"):
            if value.get(key) is not None:
                raw = value[key]
                break
        else:
            raw = value
    if isinstance(raw, str):
        return _clean_message_text(raw)
    if raw is None:
        return ""
    if isinstance(raw, dict):
        return json.dumps(raw, ensure_ascii=False) if json_fallback else ""
    if isinstance(raw, (list, tuple)):
        return json.dumps(list(raw), ensure_ascii=False)
    return str(raw)


def _call_name(call: dict[str, Any]) -> str:
    """Tool name of a call dict, whatever key the producer used to spell it."""
    for key in CALL_NAME_KEYS:
        value = call.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key in ("function", "tool"):
        value = call.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            inner = value.get("name")
            if isinstance(inner, str) and inner.strip():
                return inner.strip()
    return ""


def _coerce_arguments(value: Any) -> dict[str, Any]:
    """Call arguments as a dict (never drop a call): objects parse, other values get wrapped."""
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"raw": value}
        return parsed if isinstance(parsed, dict) else {"raw": value}
    if isinstance(value, (list, tuple)):
        return {"raw": json.dumps(list(value), ensure_ascii=False)}
    return {"raw": value}


def _call_arguments(call: dict[str, Any]) -> Any:
    """Raw `arguments` of a call, wherever this producer put them."""
    for key in ("arguments", "args", "parameters"):
        if call.get(key) is not None:
            return call[key]
    for key in ("function", "tool"):
        nested = call.get(key)
        if isinstance(nested, dict):
            for inner in ("arguments", "args", "parameters"):
                if nested.get(inner) is not None:
                    return nested[inner]
    return None


def _normalize_call(call: Any, index: int = 0) -> dict[str, Any] | None:
    """One call as `{..., "id", "name", "arguments"}` (arguments auto-parsed when stored as text).

    The producer's own keys in `CALL_KEEP_KEYS` survive, so an OpenAI-style
    `{"type", "function": {...}}` call keeps its `function` mirror for the Gemma 4 chat template
    while the flat `name` / `arguments` keys are added for the readers that look there. Timing /
    token-usage metadata the harness attaches to a call is dropped: it is not training signal.
    """
    if not isinstance(call, dict):
        return None
    name = _call_name(call)
    if not name:
        return None
    call_id = ""
    for key in ("id", "tool_call_id", "call_id", "toolCallId"):
        value = call.get(key)
        if isinstance(value, (str, int)) and str(value).strip():
            call_id = str(value).strip()
            break
    keep = {key: value for key, value in call.items() if key in CALL_KEEP_KEYS}
    return {**keep, "id": call_id or f"call_{index}", "name": name,
            "arguments": _coerce_arguments(_call_arguments(call))}


def _extract_calls(obj: Any) -> list[dict[str, Any]]:
    """Calls carried by one message / step, whatever key spelled them.

    Reads `tool_calls` / `toolCalls` (list, nested list, or the `{"0": {...}}` mapping form), the
    single-call spellings `tool_call` / `function_call`, and the flat `{"function": name,
    "arguments": {...}}` / `{"tool": name, "arguments": {...}}` shapes, including the ATIF spelling
    `steps[].tool_calls[] = [{"function_name", "arguments", "tool_call_id"}]`. A bare
    `{"name", "arguments"}` dict is a call, not a message, so it carries no call list and yields [].
    """
    if not isinstance(obj, dict):
        return []
    raw: list[Any] = []
    for key in (*CALL_LIST_KEYS, *SINGLE_CALL_KEYS):
        value = obj.get(key)
        if value is None:
            continue
        if isinstance(value, dict):
            keys = list(value)
            if _call_name(value) or any(k in value for k in ("arguments", "args", "parameters")):
                raw.append(value)
            elif keys and all(str(k).isdigit() for k in keys):
                raw.extend(value[k] for k in sorted(keys, key=lambda k: int(k)))
            else:
                raw.extend(item for item in value.values() if isinstance(item, dict))
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, dict):
                    raw.append(item)
                elif isinstance(item, (list, tuple)):
                    raw.extend(inner for inner in item if isinstance(inner, dict))
    if not raw and any(key in obj for key in ("function", "tool")) \
            and _call_arguments(obj) is not None and _call_name(obj):
        raw = [obj]              # flat {"function": "edit_file", "arguments": {...}] call shape
    calls: list[dict[str, Any]] = []
    for entry in raw:
        call = _normalize_call(entry, len(calls))
        if call:
            calls.append(call)
    return calls


def _extract_observations(obj: Any) -> list[str]:
    """Tool results of one step as plain texts (empty ones dropped)."""
    if not isinstance(obj, dict):
        return []
    for key in OBSERVATION_KEYS:
        if key not in obj:
            continue
        value = obj[key]
        items = list(value) if isinstance(value, (list, tuple)) else [value]
        texts = [text for text in (_step_text(item) for item in items) if text]
        if texts:
            return texts
    return []


def _call_to_message(call: Any, *, role: str = "assistant", content: str = "") -> dict[str, Any] | None:
    """ChatML message for one call.

    assistant -> `{"role", "content", "tool_calls": [{"id", "name", "arguments",
    "function": {"name", "arguments"}}]}`; tool -> `{"role", "tool_call_id", "content"}`. The
    `function` mirror is what the Gemma 4 chat template and `train_lora.py` read, the flat
    `name` / `arguments` is what the role filters read, so both stay cheap.
    """
    normalized = _normalize_call(call)
    if normalized is None:
        return None
    if role == "tool":
        return {"role": "tool", "tool_call_id": normalized["id"], "content": content}
    shaped = {**normalized,
              "function": {"name": normalized["name"], "arguments": normalized["arguments"]}}
    return {"role": role, "content": content, "tool_calls": [shaped]}


def _call_to_message_list(calls: Any, *, role: str = "assistant",
                          content: str = "") -> list[dict[str, Any]]:
    """ChatML messages for `calls`: assistant calls share one message, tool results get one each."""
    items = [calls] if isinstance(calls, dict) else list(calls) if isinstance(calls, (list, tuple)) else []
    normalized = [call for call in (_normalize_call(call, i) for i, call in enumerate(items)) if call]
    if not normalized:
        return []
    if role == "tool":
        return [{"role": "tool", "tool_call_id": call["id"], "content": content} for call in normalized]
    shaped = [{**call, "function": {"name": call["name"], "arguments": call["arguments"]}}
              for call in normalized]
    return [{"role": role, "content": content, "tool_calls": shaped}]


def _is_clean_chatml_message(step: dict[str, Any]) -> bool:
    """True when `step` is already a ChatML message that must be preserved as-is.

    That covers the trajectories this module writes itself (`tool_calls` with an OpenAI-style
    `function` sub-dict) so a round trip through `normalize_trace` cannot reformat them.
    """
    if "source" in step or _normalize_role(step.get("role")) != step.get("role"):
        return False
    if any(key in step for key in (*OBSERVATION_KEYS, "message", "tool_calls_flat")):
        return False
    calls = step.get("tool_calls")
    if calls is None:
        return isinstance(step.get("content", ""), str)
    if not isinstance(calls, list) or not calls:
        return False
    return all(isinstance(call, dict) and isinstance(call.get("function"), dict)
               and isinstance(call["function"].get("name"), str) and call["function"]["name"]
               for call in calls)


def _step_messages(step: Any) -> list[dict[str, Any]]:
    """One trace step (ATIF or ChatML) -> ChatML messages. [] when the step carries nothing.

    A step with calls becomes an assistant message holding those calls, followed by one
    `{"role": "tool"}` message per observation (the ATIF `observation` sits next to the call). A
    step with only an observation is a tool result. A step with only `source` / `message` becomes a
    plain `{role, content}` message with `source` mapped through `SOURCE_ROLES`.
    """
    if not isinstance(step, dict):
        return []
    if _is_clean_chatml_message(step):
        return [dict(step)]
    calls = _extract_calls(step)
    observations = _extract_observations(step)
    raw_content = step.get("message")
    if raw_content is None:
        for key in ("content", "text", "output"):
            if step.get(key) is not None:
                raw_content = step[key]
                break
    content = _step_text(raw_content, json_fallback=False)

    role = (_normalize_role(step.get("role")) or _normalize_role(step.get("type"))
            or _normalize_role(step.get("event_type"))
            or _normalize_role(step.get("source")))
    if not role:
        role = "assistant" if (calls or observations) else ""
    if observations and not calls and role in ("", "system") and step.get("source") != "system":
        role = "tool"          # a bare observation step is a tool result, not a system turn

    messages: list[dict[str, Any]] = []
    if calls and role != "tool":
        messages.extend(_call_to_message_list(calls, role=role, content=content))
        messages.extend({"role": "tool", "content": text} for text in observations)
    elif calls:
        messages.extend(_call_to_message_list(calls, role="tool", content=content))
    elif observations:
        if content:
            messages.append({"role": role or "tool", "content": content})
        messages.extend({"role": "tool", "content": text} for text in observations)
    elif content:
        message: dict[str, Any] = {"role": role or "system", "content": content}
        tool_call_id = step.get("tool_call_id")
        if isinstance(tool_call_id, (str, int)) and str(tool_call_id).strip():
            message["tool_call_id"] = str(tool_call_id)
        messages.append(message)
    return messages


def _load_steps(payload: Any) -> list[dict[str, Any]]:
    """Ordered trace steps in `payload` (`messages` / `steps` / `events` / `trajectory`)."""
    data = _coerce_payload(payload)
    if isinstance(data, list):
        return [step for step in data if isinstance(step, dict)]
    if not isinstance(data, dict):
        return []
    for key in STEP_LIST_KEYS:
        value = data.get(key)
        if isinstance(value, list):
            steps = [step for step in value if isinstance(step, dict)]
            if steps:
                return steps
    if _step_messages(data):
        return [data]
    return []


def _load_messages(payload: Any) -> list[dict[str, Any]]:
    """ChatML messages for `payload` (JSON text, path, trace mapping, step list, or one step)."""
    messages: list[dict[str, Any]] = []
    for step in _load_steps(payload):
        messages.extend(_step_messages(step))
    return messages


def _load_tool_names(payload: Any) -> list[str]:
    """Tool names called in `payload`, whatever key spelled them (order kept, duplicates kept).

    Accepts a messages list, a trace mapping (`messages` / `steps` / ATIF `steps[].tool_calls`), a
    single step / message, or a single call dict.
    """
    names: list[str] = []
    for step in _load_steps(payload):
        names.extend(call["name"] for call in _extract_calls(step))
    if not names and isinstance(payload, dict):
        names = [call["name"] for call in _extract_calls(payload)]
    return names


def _role_matches(payload: Any, role: str) -> bool:
    """Trajectory-level role filter: `main` takes everything, sub-agents only their own tool set.

    The check is per trajectory on purpose: the closing turn of an analyzer / verifier report
    carries no tool call at all, so a per-message check would drop the answer to train on.
    """
    if role == "main":
        return True
    allowed = ROLE_ALLOWED_CALLS.get(role)
    if allowed is None:
        return True
    names = _load_tool_names(payload)
    return bool(names) and set(names) <= set(allowed)


def _resolve_resolved(data: Any) -> bool | None:
    """resolved flag of a trace: `steps[].extra.resolved` first, then any bool in the document."""
    def walk(node: Any) -> bool | None:
        if isinstance(node, dict):
            for key in RESOLVED_KEYS:
                if isinstance(node.get(key), bool):
                    return node[key]
            for value in node.values():
                found = walk(value)
                if found is not None:
                    return found
        elif isinstance(node, (list, tuple)):
            for value in node:
                found = walk(value)
                if found is not None:
                    return found
        return None

    if isinstance(data, dict):
        for step in data.get("steps") or []:
            extra = step.get("extra") if isinstance(step, dict) else None
            if isinstance(extra, dict):
                for key in RESOLVED_KEYS:
                    if isinstance(extra.get(key), bool):
                        return extra[key]
    return walk(data)


def _io_summary(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """messages から入出力の要約（meta に入れる files_read / files_written / test_*）を作る。

    record の meta は训练データの追跡用で、`files_read` は `path:start-end`、`files_written` /
    `files_changed` は path、`test_targets` は pytest の対象、`test_status` は観測された test の
    結果（unknown / passed / failed）。tool 応答が後ろに続く前提で読む。
    """
    files_read: list[str] = []
    files_written: list[str] = []
    test_targets: list[str] = []
    test_status = "unknown"
    pending_test = False
    for message in messages:
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            name = str(fn.get("name") or "")
            raw = fn.get("arguments")
            try:
                args = json.loads(raw) if isinstance(raw, str) else (raw or {})
            except json.JSONDecodeError:
                args = {}
            if name == "read_file" and args.get("filepath"):
                entry = f"{args['filepath']}:{args.get('start_line', '?')}-{args.get('end_line', '?')}"
                if entry not in files_read:
                    files_read.append(entry)
            elif name in ("edit_file", "write_file") and args.get("filepath"):
                if args["filepath"] not in files_written:
                    files_written.append(args["filepath"])
            elif name == "run_command":
                command = str(args.get("command") or "")
                match = re.search(r"python3 -m pytest\s+(\S+)", command)
                if match:
                    test_targets.append(match.group(1))
                    pending_test = True
        if message.get("role") != "tool" or not pending_test:
            continue
        body = message.get("content")
        if isinstance(body, list):  # OpenAI parts form -> plain text に落としてから読む
            body = "".join(part.get("text", "") for part in body if isinstance(part, dict))
        text = str(body).lower()
        if "passed" in text and "failed" not in text:
            test_status = "passed"
        elif "failed" in text or "error" in text:
            test_status = "failed"
        pending_test = False
    return {
        "files_read": files_read,
        "files_written": files_written,
        "files_changed": list(files_written),
        "test_targets": list(dict.fromkeys(test_targets)),
        "test_status": test_status,
    }


def normalize_trace(payload: Any) -> dict[str, Any] | None:
    """Best-effort import of one harness trace into ChatML messages.

    `payload` is a decoded dict, JSON text, or a path (`str` | `Path`) to a trace file. Returns
    `{"messages": [...], "meta": {...}}` where meta keeps `instance_id` / `source` / `resolved`, or
    None when nothing usable came out. A trace is only `resolved=True` when it says so
    (`resolved` / `is_resolved` / `resolved_on_last_commit` ...); a trace that has steps but no
    usable success parameter (every `ATIF-*` trace the harness writes today, which keeps the flag
    in `task_results.jsonl`) comes back `resolved=False`, so `--only-resolved` and the DPO rejected
    side treat it as a failure instead of silently trusting it.
    """
    data = _coerce_payload(payload)
    if not isinstance(data, dict):
        return None
    steps = _load_steps(data)
    messages = _load_messages(data)
    if not messages:
        return None
    if not any(message.get("tool_calls") or message.get("role") == "assistant"
               for message in messages):
        return None
    instance_id = data.get("instance_id") or data.get("task_id")
    if not instance_id and isinstance(payload, (Path, str)):
        instance_id = Path(str(payload)).stem
    resolved = _resolve_resolved(data)
    if resolved is None and (steps or messages):
        # step が有るのに成功パラメータが無い trace は「解決した証拠が無い」= 失敗として扱う。
        resolved = False
    return {
        "messages": messages,
        "meta": {
            "instance_id": str(instance_id or "unknown"),
            "source": "trace",
            "resolved": resolved,
            "n_steps": len(steps),
            **_io_summary(messages),
        },
    }


def skip_reason(task: dict[str, Any], max_hunk_lines: int, role: str = "main") -> str:
    """build_trajectory が None を返した理由（stats key）。role で使える hunk の条件が違う。"""
    if role in ("analyzer", "verifier"):
        if not gold_hunks(task, max_hunk_lines):
            return "skipped_no_usable_hunks"
        return "skipped_too_many_edits"
    sections = parse_diff(task.get("patch") or "")
    if not sections:
        return "skipped_no_edits"
    if all(protected_file(s["path"]) for s in sections):
        return "protected_only"
    if any(len(h["body"]) > max_hunk_lines for s in sections for h in s["hunks"]):
        return "skipped_big_hunk"
    return "skipped_too_many_edits"


def resolve_system_prompt(role: str, path: str | None = None) -> tuple[str, Path]:
    """role 既定の system prompt を repo root 基準で読む（`--system-prompt` が優先）."""
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [Path(p) for p in (([path] if path else []) + [ROLE_SYSTEM_PROMPTS[role]])]
    for candidate in candidates:
        for probe in ([candidate, repo_root / candidate] if not candidate.is_absolute() else [candidate]):
            if probe.is_file():
                return probe.read_text(encoding="utf-8"), probe
    raise FileNotFoundError(f"system prompt not found for role={role}: {candidates}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect / synthesize SFT trajectories")
    parser.add_argument("--tasks", default="data/tasks.jsonl", help="published tasks.jsonl")
    parser.add_argument("--role", choices=sorted(ROLE_SYSTEM_PROMPTS), default="main",
                        help="main: edit/submit 軌跡（既定）/ analyzer: read-only 局所化のみ / "
                             "verifier: test 実行と報告のみ（edit / submit なし）")
    parser.add_argument("--system-prompt", default=None,
                        help=f"override the system prompt (default: per role: "
                             f"{ROLE_SYSTEM_PROMPTS['main']} / "
                             f"{ROLE_SYSTEM_PROMPTS['analyzer']} / {ROLE_SYSTEM_PROMPTS['verifier']})")
    parser.add_argument("--output", default=None,
                        help="output jsonl (default: per role, see ROLE_DEFAULT_OUTPUTS)")
    parser.add_argument("--import-traces", default=None, help="dir with results/traces/trace_*.json")
    parser.add_argument("--only-resolved", action="store_true",
                        help="keep imported traces whose task resolved=True")
    parser.add_argument("--max-edits", type=int, default=6)
    parser.add_argument("--max-hunk-lines", type=int, default=DEFAULT_MAX_HUNK_LINES,
                        help="drop tasks with hunks longer than this (default: %(default)s): a 40 "
                             "line hunk is already at the edge of what one edit_file turn should carry")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--native-tools", action="store_true", default=True,
                        help="emit OpenAI-style assistant.tool_calls (default). Use --text-tools "
                             "if the serving chat template cannot serialize tool calls.")
    parser.add_argument("--text-tools", dest="native_tools", action="store_false")
    parser.add_argument("--max-tool-calls", type=int, default=DEFAULT_MAX_TOOL_CALLS,
                        help="written into the synthetic user prompt (default: %(default)s, "
                             "submission/eval_config.yaml の max_tool_calls)")
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS,
                        help="written into the synthetic user prompt (default: %(default)s, "
                             "submission/eval_config.yaml の max_turns)")
    parser.add_argument("--max-time-minutes", type=float, default=DEFAULT_MAX_TIME_MINUTES)
    parser.add_argument("--no-patch-examples", type=int, default=8,
                        help="role=main のとき、原因が repo の外の設定にある failure を NO_PATCH で "
                             "終わる教材をこの件数だけ足す（0 で無効）")
    args = parser.parse_args()

    tasks_path = Path(args.tasks)
    if not tasks_path.is_file():
        print(f"[error] tasks file not found: {tasks_path}\n"
              f"        fetch with: kaggle competitions download -c gemma-4-developer-agent "
              f"-f tasks.jsonl -p data")
        return 2

    system_prompt, prompt_path = resolve_system_prompt(args.role, args.system_prompt)
    budget = {"max_tool_calls": args.max_tool_calls, "max_turns": args.max_turns,
              "max_time_minutes": args.max_time_minutes}
    print(f"[collect] role={args.role} system_prompt={prompt_path}")

    records: list[dict[str, Any]] = []
    task_cache: list[dict[str, Any]] = []
    stats = {"tasks": 0, "skipped_no_edits": 0, "skipped_too_many_edits": 0,
             "skipped_big_hunk": 0, "protected_only": 0, "skipped_no_usable_hunks": 0}
    for line in tasks_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        task = json.loads(line)
        stats["tasks"] += 1
        traj = build_trajectory(task, system_prompt, budget, args.native_tools,
                                args.max_edits, args.max_hunk_lines, role=args.role)
        if traj is None:
            stats[skip_reason(task, args.max_hunk_lines, args.role)] += 1
            continue
        records.append(traj)
        task_cache.append(task)

    # 原因が repo の外の設定にある型の正解（NO_PATCH）を少量混ぜる。build_dpo_pairs 側で
    # NO_PATCH を失敗として教えないルールと対になっている。
    if args.role == "main" and args.no_patch_examples > 0:
        added = 0
        for task in task_cache:
            if added >= args.no_patch_examples:
                break
            cand = build_no_patch_trajectory(task, system_prompt, budget, args.native_tools,
                                             args.max_hunk_lines)
            if cand:
                records.append(cand)
                added += 1
        print(f"[collect] NO_PATCH 教材を {added} 件追加（final answer = NO_PATCH の設定系）")

    if args.import_traces:
        traces_dir = Path(args.import_traces)
        candidates = list(traces_dir.rglob("trace_*.json")) or list(traces_dir.rglob("*.json"))
        imported = skipped_role = 0
        for path in candidates:
            traj = normalize_trace(path)
            if not traj:
                continue
            if not _role_matches(traj, args.role):
                skipped_role += 1        # an edit_file / submit_patch trace is not analyzer material
                continue
            if args.only_resolved and traj["meta"].get("resolved") is not True:
                continue
            records.append(traj)
            imported += 1
        print(f"[collect] imported {imported} harness traces from {traces_dir}"
              + (f" (skipped {skipped_role} other-role traces)" if skipped_role else ""))

    if args.limit:
        records = records[: args.limit]

    out = Path(args.output or ROLE_DEFAULT_OUTPUTS[args.role])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    n_calls = [sum(1 for m in r["messages"] if m.get("tool_calls")) for r in records]
    avg_len = sum(len(json.dumps(r["messages"])) for r in records) / max(1, len(records))
    print(f"[collect] tasks read     : {stats['tasks']}")
    print(f"[collect] trajectories   : {len(records)} -> {out}")
    print(f"[collect] skipped        : {stats}")
    print(f"[collect] tool calls/ traj: min {min(n_calls) if n_calls else 0} "
          f"max {max(n_calls) if n_calls else 0} avg {sum(n_calls) / max(1, len(n_calls)):.1f}")
    print(f"[collect] avg chars/ traj : {avg_len:.0f} (~{avg_len / 3.2:.0f} tokens)")
    return 0 if records else 1


if __name__ == "__main__":
    sys.exit(main())
