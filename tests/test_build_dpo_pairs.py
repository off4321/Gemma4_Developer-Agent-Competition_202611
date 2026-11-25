"""`build_dpo_pairs.py` の解決度ジャインと rejected 選定の検査（2026-11-24 の 90-task run 対策）。

押さえる 3 点:
  1. `resolved` / `agent_patch_size` は ATIF trace に無いので `task_results.jsonl` から join する。
     join しないと解決済み trace が rejected（教材の「悪手」側）に流れ込む。
  2. `agent_patch_size == 0` の解決済み episode を main の chosen にしない（空 diff 提出を褒める）。
  3. tool 失敗は tool 単位で数え、`edit_file` を 2 回以上失敗した trace を rejected の先頭に置く。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "collect" / "build_dpo_pairs.py"

_spec = importlib.util.spec_from_file_location("build_dpo_pairs", MODULE_PATH)
assert _spec is not None and _spec.loader is not None
dpo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dpo)

EDIT_ARGS = {"filepath": "pkg/a.py", "old_string": "    x = 1", "new_string": "    x = 2"}


def call(name: str, arguments: dict[str, Any], cid: str) -> dict[str, Any]:
    return {"tool_call_id": cid, "function_name": name, "arguments": arguments}


def step(text: str, tool_calls: list[dict[str, Any]] | None = None,
         result: dict[str, Any] | None = None) -> dict[str, Any]:
    """One ATIF step: agent message, its tool calls, and the tool observation next to them."""
    step_data: dict[str, Any] = {"step_id": 1, "source": "agent", "message": text}
    if tool_calls:
        step_data["tool_calls"] = tool_calls
    if result is not None:
        step_data["observation"] = {"content": json.dumps(result)}
    return step_data


OK = {"status": "ok"}
EDIT_MISS = {"status": "error", "error_type": "FileEditError",
             "error_message": "Failed to replace: old_string not found."}
EDIT_TWICE = {"status": "error", "error_type": "FileEditError",
              "error_message": "Target string occurs 2 times in pkg/a.py."}
CMD_FAIL = {"status": "error", "error_type": "CommandError", "error_message": "",
            "details": {"exit_code": 1}}


def write_trace(run: Path, instance_id: str, name: str, steps: list[dict[str, Any]]) -> None:
    traces = run / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    body = {"schema_version": "ATIF-1.0", "instance_id": instance_id, "steps": steps}
    (traces / f"trace_{instance_id}__{name}.json").write_text(json.dumps(body), encoding="utf-8")


def write_task_results(run: Path, rows: list[dict[str, Any]]) -> None:
    run.mkdir(parents=True, exist_ok=True)
    (run / "task_results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


SOLVED_STEPS = [
    step("locate", [call("read_file", {"filepath": "pkg/a.py"}, "s1")], OK),
    step("fix", [call("edit_file", EDIT_ARGS, "s2")], OK),
    step("submit", [call("submit_patch", {}, "s3")], OK),
    step("x is now 2 and the targeted test passes"),
]
GUESSY_STEPS = [
    step("guess 1", [call("edit_file", {**EDIT_ARGS, "old_string": "x = 1"}, "b1")], EDIT_MISS),
    step("guess 2", [call("edit_file", {**EDIT_ARGS, "old_string": "x=1"}, "b2")], EDIT_TWICE),
    step("submit", [call("submit_patch", {}, "b3")], OK),
    step("submitted my best guess"),
]
NOISY_STEPS = [
    step("grep", [call("run_command", {"command": "grep -r nope ."}, "n1")], CMD_FAIL),
    step("submit", [call("submit_patch", {}, "n2")], OK),
    step("submitted without any edit"),
]


def normalized(function_name: str, arguments: dict[str, Any], cid: str) -> dict[str, Any]:
    """A call message in the shape `normalize_trace` produces (both flat and `function` views)."""
    return {"id": cid, "name": function_name, "arguments": json.dumps(arguments),
            "function": {"name": function_name, "arguments": json.dumps(arguments)}}


def gold(fix_needed: bool = True) -> dict[str, Any]:
    """A synthesised gold `chosen`: prompt then read/edit/test/answer."""
    messages: list[dict[str, Any]] = [{"role": "system", "content": "sys prompt"},
                                      {"role": "user", "content": "issue: x must be 2"}]
    if fix_needed:
        messages.extend([
            {"role": "assistant", "content": "read first",
             "tool_calls": [normalized("read_file", {"filepath": "pkg/a.py"}, "g1")]},
            {"role": "tool", "tool_call_id": "g1", "content": json.dumps(OK)},
            {"role": "assistant", "content": "apply fix",
             "tool_calls": [normalized("edit_file", EDIT_ARGS, "g2")]},
            {"role": "tool", "tool_call_id": "g2", "content": json.dumps(OK)},
        ])
    messages.append({"role": "assistant", "content": "fixed: x = 2 in pkg/a.py"})
    return {"messages": messages}


# --------------------------------------------------------------------------- task_results join
def test_task_results_is_found_next_to_the_traces_dir(tmp_path: Path) -> None:
    write_trace(tmp_path, "rich_1", "a", SOLVED_STEPS)
    write_task_results(tmp_path, [{"instance_id": "rich_1", "resolved": True,
                                   "agent_patch_size": 710}])
    expected = tmp_path / "task_results.jsonl"
    assert dpo._task_results_file(tmp_path) == expected
    assert dpo._task_results_file(tmp_path / "traces") == expected     # `--results .../traces`
    index = dpo._task_results_for(tmp_path / "traces")
    assert index["rich_1"]["agent_patch_size"] == 710


def test_missing_task_results_leaves_resolved_unknown(tmp_path: Path) -> None:
    write_trace(tmp_path, "rich_1", "a", SOLVED_STEPS)
    assert dpo._task_results_for(tmp_path) == {}
    grouped = dpo._load_traces([tmp_path], "main")
    assert grouped["rich_1"]["fail"] and not grouped["rich_1"]["ok"]


def test_resolved_trace_is_never_the_rejected_example(tmp_path: Path) -> None:
    write_trace(tmp_path, "rich_ok", "a", SOLVED_STEPS)
    write_trace(tmp_path, "rich_guessy", "a", GUESSY_STEPS)
    write_task_results(tmp_path, [{"instance_id": "rich_ok", "resolved": True,
                                   "agent_patch_size": 710},
                                  {"instance_id": "rich_guessy", "resolved": False,
                                   "agent_patch_size": 0}])
    grouped = dpo._load_traces([tmp_path], "main")
    assert [t["meta"]["resolved"] for t in grouped["rich_ok"]["ok"]] == [True]
    assert not grouped["rich_ok"]["fail"], "a solved task must not leak into the rejected side"
    # Without the index (an explicit file that does not exist) resolved is never guessed, so the
    # same trace lands on the rejected side -> that is the bug the join fixes.
    no_index = dpo._load_traces([tmp_path], "main", tmp_path / "absent.jsonl")
    assert no_index["rich_ok"]["fail"] and not no_index["rich_ok"]["ok"]

    pairs = dpo.build_pairs(grouped, {"rich_guessy": gold()}, max_pairs_per_task=1)
    assert [p["meta"]["instance_id"] for p in pairs] == ["rich_guessy"]
    assert pairs[0]["meta"]["source"] == "gold_fallback"


def test_empty_patch_resolved_episode_is_never_chosen(tmp_path: Path) -> None:
    """Same task solved twice: run A failed, run B passed -> self_play pair only with a real patch."""
    run_fail = tmp_path / "run_a"
    run_win = tmp_path / "run_b"
    write_trace(run_fail, "fastapi_14873", "a", GUESSY_STEPS)
    write_trace(run_win, "fastapi_14873", "a", SOLVED_STEPS)
    write_task_results(run_fail, [{"instance_id": "fastapi_14873", "resolved": False,
                                   "agent_patch_size": 0}])
    write_task_results(run_win, [{"instance_id": "fastapi_14873", "resolved": True,
                                  "agent_patch_size": 710}])
    grouped = dpo._load_traces([run_fail, run_win], "main")
    pairs = dpo.build_pairs(grouped, {}, max_pairs_per_task=1)
    assert [p["meta"]["source"] for p in pairs] == ["self_play"]
    assert dpo._load_traces([run_fail], "main")["fastapi_14873"]["fail"], \
        "run B must not mark run A's trace solved"

    write_task_results(run_win, [{"instance_id": "fastapi_14873", "resolved": True,
                                  "agent_patch_size": 0}])
    empty_patch = dpo._load_traces([run_fail, run_win], "main")
    assert dpo.build_pairs(empty_patch, {}, max_pairs_per_task=1) == []
    assert dpo.build_pairs(empty_patch, {"fastapi_14873": gold()}, max_pairs_per_task=1) != [], \
        "an empty-patch episode may still be graded against a gold chosen"
    solved = empty_patch["fastapi_14873"]["ok"][0]
    assert dpo._usable_as_chosen(solved, "main") is False
    assert dpo._usable_as_chosen(solved, "analyzer") is True


# ------------------------------------------------------------------------- per-tool error tally
def test_tool_errors_are_attributed_to_the_tool_that_returned_them(tmp_path: Path) -> None:
    write_trace(tmp_path, "rich_guessy", "a", GUESSY_STEPS)
    write_trace(tmp_path, "rich_noisy", "a", NOISY_STEPS)
    write_task_results(tmp_path, [{"instance_id": "rich_guessy", "resolved": False},
                                  {"instance_id": "rich_noisy", "resolved": False}])
    traces = dpo._load_traces([tmp_path], "main")
    guessy = dpo._response_signals(traces["rich_guessy"]["fail"][0]["messages"])
    noisy = dpo._response_signals(traces["rich_noisy"]["fail"][0]["messages"])

    assert guessy["tool_errors"] == {"edit_file": 2}
    assert guessy["edit_errors"] == 2 and guessy["failed_steps"] == 2
    assert dpo._rejection_reason(traces["rich_guessy"]["fail"][0], guessy) == "edit_file_repeat"
    # a failed `grep` is a tool error, not an edit_file error
    assert noisy["tool_errors"] == {"run_command": 1} and noisy["edit_errors"] == 0
    assert dpo._rejection_reason(traces["rich_noisy"]["fail"][0], noisy) == "unresolved"


def test_edit_heavy_trace_is_preferred_as_the_rejected_example(tmp_path: Path) -> None:
    write_trace(tmp_path, "rich_1", "guessy", GUESSY_STEPS)
    write_trace(tmp_path, "rich_1", "noisy", NOISY_STEPS)
    write_task_results(tmp_path, [{"instance_id": "rich_1", "resolved": False}])
    grouped = dpo._load_traces([tmp_path], "main")
    assert len(grouped["rich_1"]["fail"]) == 2

    pairs = dpo.build_pairs(grouped, {"rich_1": gold()}, max_pairs_per_task=1)
    assert len(pairs) == 1, "one pair per task at max_pairs_per_task=1"
    meta = pairs[0]["meta"]
    assert meta["rejection_reason"] == "edit_file_repeat"
    assert meta["edit_errors"] == 2, "the 2x edit_file failure outranks the grep-only failure"


def test_step_pairs_contain_single_assistant_turn_without_tool_observations(tmp_path: Path) -> None:
    bad_steps = [
        step("read", [call("read_file", {"filepath": "pkg/a.py"}, "r1")],
             {"status": "ok", "filepath": "pkg/a.py", "content": "def f():\n    x = 1\n"}),
        step("hallucinated edit", [call("edit_file", {**EDIT_ARGS, "old_string": "self.x = 1"}, "b1")], EDIT_MISS),
        step("invented tool", [call("apply_patch", {"filepath": "pkg/a.py"}, "b2")], OK),
        step("full pytest", [call("run_command", {"command": "python3 -m pytest tests/ -q"}, "b3")], OK),
        step("empty submit", [call("submit_patch", {}, "b4")], {"status": "ok", "patch_size": 0, "files_changed": 0}),
    ]
    write_trace(tmp_path, "rich_step", "a", bad_steps)
    write_task_results(tmp_path, [{"instance_id": "rich_step", "resolved": False, "agent_patch_size": 0}])
    grouped = dpo._load_traces([tmp_path], "main")
    step_pairs = dpo.build_step_pairs(grouped, {"rich_step": gold()}, max_pairs_per_task=6)
    assert len(step_pairs) == 4
    reasons = [p["meta"]["rejection_reason"] for p in step_pairs]
    assert reasons == ["step_edit_error", "step_invalid_tool", "step_full_pytest", "step_empty_submit"]
    for p in step_pairs:
        assert len(p["chosen"]) == 1 and p["chosen"][0]["role"] == "assistant"
        assert len(p["rejected"]) == 1 and p["rejected"][0]["role"] == "assistant"
        assert p["prompt"][-1]["role"] in ("user", "tool")
    # Because read_file already contained EDIT_ARGS["old_string"], step_edit_error teaches the exact gold edit_file
    assert step_pairs[0]["chosen"][0]["tool_calls"][0]["function"]["name"] == "edit_file"


def test_step_pairs_teach_submit_patch_after_verified_edit_and_real_diff_on_noop_edit(tmp_path: Path) -> None:
    noop_err = {"status": "error", "error_type": "FileEditError",
                "error_message": "No changes to apply. old_string and new_string are identical."}
    loop_steps = [
        step("noop edit", [call("edit_file", {**EDIT_ARGS, "new_string": EDIT_ARGS["old_string"]}, "e0")], noop_err),
        step("real edit", [call("edit_file", EDIT_ARGS, "e1")], OK),
        step("verify", [call("run_command", {"command": "python3 -m py_compile pkg/a.py"}, "c1")], OK),
        step("redundant reread", [call("read_file", {"filepath": "pkg/a.py"}, "r2")], OK),
    ]
    write_trace(tmp_path, "fastapi_loop", "a", loop_steps)
    write_task_results(tmp_path, [{"instance_id": "fastapi_loop", "resolved": False, "agent_patch_size": 120}])
    grouped = dpo._load_traces([tmp_path], "main")
    step_pairs = dpo.build_step_pairs(grouped, {"fastapi_loop": gold()}, max_pairs_per_task=6)
    by_reason = {p["meta"]["rejection_reason"]: p for p in step_pairs}
    assert "step_edit_error" in by_reason
    assert by_reason["step_edit_error"]["chosen"][0]["tool_calls"][0]["function"]["name"] == "edit_file"
    assert "step_missing_submit" in by_reason
    assert by_reason["step_missing_submit"]["chosen"][0]["tool_calls"][0]["function"]["name"] == "submit_patch"
