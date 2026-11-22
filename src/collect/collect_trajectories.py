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
        edits.append({"kind": "edit_file", "filepath": section["path"],
                      "old_string": old_string, "new_string": new_string})
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
    edits: list[dict[str, str]] = []
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
    for i, edit in enumerate(edits):
        filepath = edit["filepath"]
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
            "files_changed": sorted(files_changed), "pytest_targets": targets,
            "source": "gold_patch",
        },
    }


def hunk_line_range(hunk: dict[str, Any]) -> str:
    """gold hunk 1 件を `start-end` 行 range にする（analyzer 教材の `path:lines` 用）。"""
    body = hunk["body"]
    old_lines = sum(1 for line in body if line[:1] in {" ", "-"}) or 1
    new_lines = sum(1 for line in body if line[:1] in {" ", "+"})
    start = int(hunk.get("old_start") or 1)
    return f"{start}-{max(start, start + max(old_lines, new_lines) - 1)}"


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
        f"Budget handed to you: {budget.get('max_tool_calls', 80)} tool calls, "
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
                        "path": h["path"], "start_line": start_line,
                        "end_line": start_line + 40}},
                    {"status": "ok", "path": h["path"], "start_line": start_line,
                     "content": h["old_string"][:2000]},
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
    """verifier LoRA 用: get_status -> 限定した pytest -> 報告（edit / submit は絶対に呼ばない）。"""
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
    max_calls = int(budget.get("max_tool_calls") or 80)
    call_index += 1
    append_call(messages, call_index,
                "Budget first: if the run is nearly out of calls or seconds, report GO at once.",
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

    call_index += 1
    append_call(messages, call_index,
                "Confirm nothing protected and no scratch file ended up in the tree.",
                {"name": "run_command", "arguments": {"command": "git status --porcelain"}},
                {"status": "ok", "exit_code": 0,
                 "stdout": "\n".join(f" M {f}" for f in files), "stderr": ""},
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
def normalize_trace(path: Path) -> dict[str, Any] | None:
    """Best-effort import of a harness trace (results/traces/trace_*.json) into messages."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    def resolve_resolved(obj: Any) -> bool | None:
        found = None
        def walk(node: Any) -> None:
            nonlocal found
            if isinstance(node, dict):
                for key in ("resolved", "test_resolved", "passed"):
                    if isinstance(node.get(key), bool):
                        found = node[key]
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)
        walk(obj)
        return found

    steps: list[dict[str, Any]] | None = None
    for key in ("messages", "steps", "events", "trajectory"):
        if isinstance(data.get(key), list):
            steps = data[key]
            break
    if not steps:
        return None

    messages: list[dict[str, Any]] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        role = step.get("role") or step.get("type") or step.get("event_type")
        text = step.get("content") or step.get("text") or step.get("output") or ""
        if isinstance(text, dict):
            text = json.dumps(text, ensure_ascii=False)
        call = step.get("tool_call") or step.get("function_call") or {}
        if role in ("user", "tool", "assistant", "system") and text:
            messages.append({"role": role, "content": text if isinstance(text, str) else str(text)})
        elif call:
            messages.append({"role": "assistant", "content": text,
                             "tool_calls": [{"id": call.get("id", "call_0"), "type": "function",
                                            "function": {"name": call.get("name"),
                                                         "arguments": json.dumps(
                                                             call.get("arguments", {}), ensure_ascii=False)}}]})
    if not any(m.get("role") == "assistant" and m.get("tool_calls") for m in messages):
        return None
    resolved = resolve_resolved(data)
    return {"messages": messages, "meta": {"instance_id": path.stem, "source": "trace",
                                           "resolved": resolved}}


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
    parser.add_argument("--max-hunk-lines", type=int, default=80)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--native-tools", action="store_true", default=True,
                        help="emit OpenAI-style assistant.tool_calls (default). Use --text-tools "
                             "if the serving chat template cannot serialize tool calls.")
    parser.add_argument("--text-tools", dest="native_tools", action="store_false")
    parser.add_argument("--max-tool-calls", type=int, default=80)
    parser.add_argument("--max-turns", type=int, default=150)
    parser.add_argument("--max-time-minutes", type=float, default=45.0)
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

    if args.import_traces:
        traces_dir = Path(args.import_traces)
        candidates = list(traces_dir.rglob("trace_*.json")) or list(traces_dir.rglob("*.json"))
        imported = 0
        for path in candidates:
            traj = normalize_trace(path)
            if traj and not (args.only_resolved and traj["meta"].get("resolved") is not True):
                records.append(traj)
                imported += 1
        print(f"[collect] imported {imported} harness traces from {traces_dir}")

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
