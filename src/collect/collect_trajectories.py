"""
tasks.jsonl（gold patch + test_patch）から SFT 用の軌跡を合成する。

なぜ合成するか: LoRA で直しやすいのは「出力の癖」なので、教材として
**短轨跡で submit_patch まで到達する形** と **edit_file が初回で通る形** を見せるのが最も効く。
実差分に無い文字列を教材にすると、モデルは実在しない old_string を生成する癖を覚える。
だからここでは次を必ず実データから取る:

  * edit_file の old_string / new_string  -> gold `patch` を unified diff から解析した実文字列
  * run_command の pytest target          -> test_patch の `def test_*` + diff のファイルパスから
                                             決まる `path::test_name`（実在する test だけ）

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
                cur["hunks"].append({"body": []})
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
                     native_tools: bool, max_edits: int, max_hunk_lines: int) -> dict[str, Any] | None:
    """gold patch 1 件から SFT 用 trajectory を 1 本作る。教材に出来なければ None。"""
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


def skip_reason(task: dict[str, Any], max_hunk_lines: int) -> str:
    """build_trajectory が None を返した理由（main の stats key）."""
    sections = parse_diff(task.get("patch") or "")
    if not sections:
        return "skipped_no_edits"
    if all(protected_file(s["path"]) for s in sections):
        return "protected_only"
    if any(len(h["body"]) > max_hunk_lines for s in sections for h in s["hunks"]):
        return "skipped_big_hunk"
    return "skipped_too_many_edits"


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect / synthesize SFT trajectories")
    parser.add_argument("--tasks", default="data/tasks.jsonl", help="published tasks.jsonl")
    parser.add_argument("--system-prompt", default="submission/prompts/system.md")
    parser.add_argument("--output", default="data/trajectories.jsonl")
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

    system_prompt = Path(args.system_prompt).read_text(encoding="utf-8")
    budget = {"max_tool_calls": args.max_tool_calls, "max_turns": args.max_turns,
              "max_time_minutes": args.max_time_minutes}

    records: list[dict[str, Any]] = []
    stats = {"tasks": 0, "skipped_no_edits": 0, "skipped_too_many_edits": 0,
             "skipped_big_hunk": 0, "protected_only": 0}
    for line in tasks_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        task = json.loads(line)
        stats["tasks"] += 1
        traj = build_trajectory(task, system_prompt, budget, args.native_tools,
                                args.max_edits, args.max_hunk_lines)
        if traj is None:
            stats[skip_reason(task, args.max_hunk_lines)] += 1
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

    out = Path(args.output)
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
