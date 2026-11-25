"""`build_trajectory(role="main")` の read-before-edit を検査する。

eval trace は 10 task 0 resolved で、その内訳が「見ていない行を old_string に書いて edit_file が
一致しない」巡りだった。合成教材が次を守れているかをここで確かめる。

  1. edit_file より前に、同じ file の read_file 応答がある
  2. その read_file 応答の content に、その edit_file の old_string がそのまま含まれる

1 だけでは足りず、2 を外すと未読の old_string を書き写す教材になる。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "collect" / "collect_trajectories.py"
TASKS_PATH = ROOT / "data" / "tasks.jsonl"
SYSTEM_PROMPT = (ROOT / "submission" / "prompts" / "system.md").read_text(encoding="utf-8")
BUDGET = {"max_tool_calls": 30, "max_turns": 25, "max_time_minutes": 30.0}

_spec = importlib.util.spec_from_file_location("collect_trajectories", MODULE_PATH)
assert _spec is not None and _spec.loader is not None, f"cannot import {MODULE_PATH}"
collect = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(collect)


def load_tasks() -> list[dict[str, Any]]:
    """data/tasks.jsonl の gold task（無い checkout では skip）。"""
    if not TASKS_PATH.is_file():
        pytest.skip("data/tasks.jsonl is missing: download it from the Kaggle competition first")
    rows: list[dict[str, Any]] = []
    for line in TASKS_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def tool_steps(messages: list[dict[str, Any]]) -> list[list[Any]]:
    """[[tool 名, arguments, 返ってきた content], ...] を出現順に返す。

    `append_call` は assistant(tool_calls) の直後に tool 応答を 1 組ぶら下げるので、tool 応答は
    直前の tool call の応答として対応づけられる。
    """
    steps: list[list[Any]] = []
    for message in messages:
        for call in message.get("tool_calls") or []:
            function = call["function"]
            arguments = json.loads(function["arguments"] or "{}")
            steps.append([function["name"], arguments, ""])
        if message.get("role") == "tool" and steps:
            payload = json.loads(message["content"])
            steps[-1][2] = str(payload.get("content", ""))
    return steps


def edits_without_covering_read(messages: list[dict[str, Any]]) -> list[str]:
    """old_string を read_file で確認していない edit_file の説明（空なら OK）。"""
    steps = tool_steps(messages)
    problems: list[str] = []
    for index in range(len(steps)):
        name, arguments = steps[index][0], steps[index][1]
        if name != "edit_file":
            continue
        old_string = str(arguments.get("old_string") or "")
        covered = False
        for previous in steps[:index]:
            same_file = previous[1].get("filepath") == arguments.get("filepath")
            shown = previous[2].find(old_string) >= 0
            if previous[0] == "read_file" and same_file and shown:
                covered = True
        if not covered:
            problems.append(f"step {index}: edit_file on {arguments.get('filepath')} used "
                            f"old_string {arguments.get('old_string', '')[:60]!r} that no "
                            f"earlier read_file showed")
    return problems


TASKS = load_tasks()
TASK_IDS = [str(task.get("instance_id") or f"task-{n}") for n, task in enumerate(TASKS)]


def build_main(task: dict[str, Any]) -> dict[str, Any] | None:
    return collect.build_trajectory(task, SYSTEM_PROMPT, BUDGET, True, 6,
                                    collect.DEFAULT_MAX_HUNK_LINES)


@pytest.mark.parametrize("task", TASKS, ids=TASK_IDS)
def test_every_edit_file_shows_its_old_string_first(task: dict[str, Any]) -> None:
    """main 教材の edit_file は、先に読んだ range から old_string を書き写している。"""
    record = build_main(task)
    if record is None:
        pytest.skip("no usable hunk in this task")
    steps = tool_steps(record["messages"])
    assert any(step[0] == "edit_file" for step in steps), "no edit_file in the material"
    assert edits_without_covering_read(record["messages"]) == []
    assert record["meta"]["n_reads"] >= 1


TWO_HUNK_PATCH = """\
--- a/pkg/client.py
+++ b/pkg/client.py
@@ -10,3 +10,3 @@
 DEFAULT_TIMEOUT = 5
-TIMEOUT_MS = DEFAULT_TIMEOUT
+TIMEOUT_MS = DEFAULT_TIMEOUT * 1000
@@ -88,3 +88,3 @@
     def load(self, url):
-        return self.session.get(url)
+        return self.session.get(url, timeout=DEFAULT_TIMEOUT)
"""

TWO_HUNK_TEST_PATCH = """\
--- a/tests/test_client.py
+++ b/tests/test_client.py
@@ -10,2 +10,4 @@
 def test_existing_client():
     assert True
+def test_timeout_ms():
+    assert client.TIMEOUT_MS == 5000
"""


def two_hunk_task() -> dict[str, Any]:
    return {"instance_id": "synthetic:two-hunks", "repo": "example/project",
            "base_commit": "0" * 40, "hints_text": "",
            "problem_statement": "TIMEOUT_MS must be milliseconds and load() must pass a timeout.",
            "patch": TWO_HUNK_PATCH, "test_patch": TWO_HUNK_TEST_PATCH}


def test_second_hunk_of_the_same_file_gets_its_own_read() -> None:
    """1 ファイル 1 read では 2 つ目の hunk を未読のまま edit する（10 task 0 resolved の原因）。"""
    record = build_main(two_hunk_task())
    assert record is not None, "the synthetic two-hunk patch must build a trajectory"
    steps = tool_steps(record["messages"])
    kinds = [step[0] for step in steps]
    assert kinds.count("edit_file") == 2
    assert kinds.count("read_file") == 2
    assert edits_without_covering_read(record["messages"]) == []
    assert record["meta"]["n_reads"] == 2


def test_read_observation_keeps_the_string_the_edit_needs() -> None:
    """include を渡した read_file は、old_string を切り捨てない（切り詰めると教材が嘘をつく）。"""
    body = "\n".join(f"value_{n} = {n}  # padding so the default limit cuts the body"
                     for n in range(200))
    tail = "value_199 = 199  # padding so the default limit cuts the body"
    cut = collect.read_observation("pkg/a.py", 1, 200, body)
    assert cut["truncated"] is True and cut["content"].endswith("...")
    kept = collect.read_observation("pkg/a.py", 1, 200, body, include=tail)
    assert kept["truncated"] is False
    assert tail in kept["content"]


def test_recovery_trajectory_teaches_reread_and_native_hygiene() -> None:
    record = collect.build_recovery_trajectory(
        two_hunk_task(), SYSTEM_PROMPT, BUDGET, True, 6, collect.DEFAULT_MAX_HUNK_LINES
    )
    assert record is not None
    assert record["meta"]["source"] == "gold_recovery"
    calls = [
        (c["function"]["name"], json.loads(c["function"]["arguments"]))
        for m in record["messages"] for c in (m.get("tool_calls") or [])
    ]
    names = [n for n, _ in calls]
    # First edit fails, then read_file re-reads exact range, then exact edit_file succeeds
    assert names.count("edit_file") == 3
    assert names.count("read_file") >= 2
    run_cmds = [a["command"] for n, a in calls if n == "run_command"]
    assert all("skills/swe-guard" not in cmd for cmd in run_cmds)
    assert any("pytest" in cmd for cmd in run_cmds)
    assert any("py_compile" in cmd and "git diff --stat" in cmd for cmd in run_cmds)


def test_pytest_targets_excludes_newly_added_test_functions_and_new_test_files() -> None:
    existing_file_patch = """\
--- a/tests/test_sse.py
+++ b/tests/test_sse.py
@@ -220,3 +220,8 @@
+def test_brand_new_hidden_case():
+    assert True
"""
    assert collect.pytest_targets(existing_file_patch) == ["tests/test_sse.py"]

    brand_new_test_file = """\
--- a/tests/test_prepare_release.py
+++ b/tests/test_prepare_release.py
@@ -0,0 +1,10 @@
+def test_prepare():
+    assert True
"""
    assert collect.pytest_targets(brand_new_test_file) == []


def test_deoracled_turn1_grep_and_three_recovery_variants() -> None:
    rec = build_main(two_hunk_task())
    assert rec is not None
    steps = tool_steps(rec["messages"])
    assert steps[0][0] == "run_command" and "grep -rn" in steps[0][1]["command"]

    for variant in (0, 1, 2):
        r = collect.build_recovery_trajectory(
            two_hunk_task(), SYSTEM_PROMPT, BUDGET, True, 6, collect.DEFAULT_MAX_HUNK_LINES, variant=variant
        )
        assert r is not None

    new_file_task = {
        "instance_id": "synthetic:new-file",
        "repo": "fastapi/fastapi",
        "base_commit": "0" * 40,
        "problem_statement": "Add scripts/prepare_release.py CLI.",
        "patch": "--- a/scripts/prepare_release.py\n+++ b/scripts/prepare_release.py\n@@ -0,0 +1,3 @@\n+def main():\n+    return 0\n+\n",
        "test_patch": "",
    }
    nf = collect.build_new_file_trajectory(new_file_task, SYSTEM_PROMPT, BUDGET, True)
    assert nf is not None
    nf_kinds = [s[0] for s in tool_steps(nf["messages"])]
    assert "write_file" in nf_kinds
