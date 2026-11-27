"""Unit tests for `src/collect/import_hf_trajectories.py`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.collect import import_hf_trajectories as imp
from src.train import train_lora

ROOT = Path(__file__).resolve().parents[1]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _openai_resolved_trajectory(
    *,
    instance_id: str = "fastapi__fastapi-1001",
    repo: str = "fastapi/fastapi",
    resolved: Any = True,
    with_finish: bool = False,
    long_thought: str | None = None,
) -> dict[str, Any]:
    thought1 = long_thought or (
        "Line 1 of teacher thought: inspect routing.py first to check path parameter parsing.\n"
        "Line 2 of teacher thought: make sure we read enough context around line 120."
    )
    msgs: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a helpful coding agent."},
        {"role": "user", "content": "Fix query parameter parsing bug in fastapi/routing.py."},
        {
            "role": "assistant",
            "content": thought1,
            "tool_calls": [
                {
                    "id": "call_a",
                    "type": "function",
                    "function": {
                        "name": "str_replace_editor",
                        "arguments": json.dumps({
                            "command": "view",
                            "path": "/testbed/fastapi/routing.py",
                            "view_range": [100, 140],
                        }),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_a",
            "content": "def get_route():\n    return old_value\n",
        },
        {
            "role": "assistant",
            "content": "I see `old_value` at line 101; replace it with `new_value` cleanly.",
            "tool_calls": [
                {
                    "id": "call_b",
                    "type": "function",
                    "function": {
                        "name": "str_replace_editor",
                        "arguments": json.dumps({
                            "command": "str_replace",
                            "path": "/testbed/fastapi/routing.py",
                            "old_str": "return old_value",
                            "new_str": "return new_value",
                        }),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_b",
            "content": "Replaced successfully.",
        },
        {
            "role": "assistant",
            "content": "Run the targeted pytest suite to verify the routing fix.",
            "tool_calls": [
                {
                    "id": "call_c",
                    "type": "function",
                    "function": {
                        "name": "execute_bash",
                        "arguments": json.dumps({"command": "pytest tests/test_routing.py -q"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_c",
            "content": "3 passed in 0.25s",
        },
    ]
    if with_finish:
        msgs.append({
            "role": "assistant",
            "content": "All tests pass; call finish now.",
            "tool_calls": [
                {
                    "id": "call_d",
                    "type": "function",
                    "function": {"name": "finish", "arguments": "{}"},
                }
            ],
        })
        msgs.append({"role": "tool", "tool_call_id": "call_d", "content": "Finished."})

    rec: dict[str, Any] = {
        "instance_id": instance_id,
        "repo": repo,
        "problem_statement": "Fix query parameter parsing bug in fastapi/routing.py.",
        "messages": msgs,
    }
    if resolved is not None:
        rec["resolved"] = resolved
    return rec


def _xml_swe_agent_trajectory(instance_id: str = "psf__requests-42") -> dict[str, Any]:
    return {
        "instance_id": instance_id,
        "repo": "psf/requests",
        "exit_status": "submitted",
        "problem_statement": "Fix cookie header parsing in requests/cookies.py.",
        "trajectory": [
            {"role": "user", "content": "Fix cookie header parsing."},
            {
                "role": "assistant",
                "content": (
                    "Let's locate cookie parsing first.\n"
                    "We should run grep across src/requests.\n"
                    "<function=execute_bash>\n"
                    "<parameter=command>grep -rn 'extract_cookies' src/requests/</parameter>\n"
                    "</function>"
                ),
            },
            {"role": "user", "content": "OBSERVATION:\nsrc/requests/cookies.py:45:def extract_cookies():"},
            {
                "role": "assistant",
                "content": (
                    "Read lines 40 to 70 of cookies.py.\n"
                    "<function=str_replace_editor>\n"
                    "<parameter=command>view</parameter>\n"
                    "<parameter=path>/workspace/src/requests/cookies.py</parameter>\n"
                    "<parameter=view_range>[40, 70]</parameter>\n"
                    "</function>"
                ),
            },
            {"role": "user", "content": "def extract_cookies():\n    return None"},
            {
                "role": "assistant",
                "content": (
                    "Update extract_cookies in cookies.py.\n"
                    "<function=str_replace_editor>\n"
                    "<parameter=command>str_replace</parameter>\n"
                    "<parameter=path>/testbed/src/requests/cookies.py</parameter>\n"
                    "<parameter=old_str>return None</parameter>\n"
                    "<parameter=new_str>return {}</parameter>\n"
                    "</function>"
                ),
            },
            {"role": "user", "content": "File edited."},
            {
                "role": "assistant",
                "content": (
                    "Create helper module if needed.\n"
                    "<function=str_replace_editor>\n"
                    "<parameter=command>create</parameter>\n"
                    "<parameter=path>/testbed/src/requests/_cookie_helpers.py</parameter>\n"
                    "<parameter=file_text>def clean(v):\n    return v.strip()\n</parameter>\n"
                    "</function>"
                ),
            },
            {"role": "user", "content": "File created."},
            {
                "role": "assistant",
                "content": (
                    "Run tests to confirm cookie handling works.\n"
                    "<function=execute_bash>\n"
                    "<parameter=command>pytest tests/test_cookies.py -q</parameter>\n"
                    "</function>"
                ),
            },
            {"role": "user", "content": "5 passed in 0.30s"},
            {
                "role": "assistant",
                "content": (
                    "Submit the verified fix.\n"
                    "<function=finish>\n"
                    "</function>"
                ),
            },
        ],
    }


def test_openai_tool_calls_conversion_and_testbed_stripping(tmp_path: Path) -> None:
    in_jsonl = tmp_path / "openai.jsonl"
    out_coder = tmp_path / "coder.jsonl"
    long_thought = "First paragraph of reasoning.\n\nSecond paragraph explaining edge cases in detail."
    rec = _openai_resolved_trajectory(with_finish=False, long_thought=long_thought)
    in_jsonl.write_text(json.dumps(rec) + "\n", encoding="utf-8")

    rc = imp.main(["--local-jsonl", str(in_jsonl), "--out-coder", str(out_coder)])
    assert rc == 0

    rows = _read_jsonl(out_coder)
    assert len(rows) == 1
    row = rows[0]
    msgs = row["messages"]

    # System prompt matches submission/prompts/system.md
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == (ROOT / "submission" / "prompts" / "system.md").read_text(encoding="utf-8").strip()

    # Check tool mappings and /testbed/ stripping
    tool_calls = [
        (m["content"], c["function"]["name"], json.loads(c["function"]["arguments"]))
        for m in msgs
        for c in (m.get("tool_calls") or [])
    ]
    assert [name for _, name, _ in tool_calls] == ["read_file", "edit_file", "run_command", "submit_patch"]

    assert tool_calls[0][0] == long_thought  # Full CoT preserved without 1-line truncation
    assert tool_calls[0][2] == {"filepath": "fastapi/routing.py", "start_line": 100, "end_line": 140}
    assert tool_calls[1][2] == {
        "filepath": "fastapi/routing.py",
        "old_string": "return old_value",
        "new_string": "return new_value",
    }
    assert tool_calls[2][2] == {"command": "pytest tests/test_routing.py -q"}
    assert tool_calls[3][2] == {}

    # Ends with submit_patch() -> tool ok -> concise 1-sentence assistant summary
    assert msgs[-3]["tool_calls"][0]["function"]["name"] == "submit_patch"
    assert msgs[-2]["role"] == "tool"
    assert json.loads(msgs[-2]["content"])["status"] == "ok"
    assert msgs[-1]["role"] == "assistant"
    assert not msgs[-1].get("tool_calls")
    assert msgs[-1]["content"].endswith(".")

    # Compatible with train_lora._normalize_tool_calls
    norm = [train_lora._normalize_tool_calls(m) for m in msgs]
    assert norm[2]["reasoning_content"] == long_thought
    assert norm[2]["content"] == ""
    assert isinstance(norm[2]["tool_calls"][0]["function"]["arguments"], dict)


def test_xml_function_call_conversion(tmp_path: Path) -> None:
    in_jsonl = tmp_path / "xml.jsonl"
    out_coder = tmp_path / "coder.jsonl"
    in_jsonl.write_text(json.dumps(_xml_swe_agent_trajectory()) + "\n", encoding="utf-8")

    rc = imp.main(["--local-jsonl", str(in_jsonl), "--out-coder", str(out_coder)])
    assert rc == 0

    rows = _read_jsonl(out_coder)
    assert len(rows) == 1
    msgs = rows[0]["messages"]
    calls = [
        (m["content"], c["function"]["name"], json.loads(c["function"]["arguments"]))
        for m in msgs
        for c in (m.get("tool_calls") or [])
    ]
    assert [name for _, name, _ in calls] == [
        "run_command",
        "read_file",
        "edit_file",
        "write_file",
        "run_command",
        "submit_patch",
    ]
    assert calls[0][2] == {"command": "grep -rn 'extract_cookies' src/requests/"}
    assert calls[1][2] == {"filepath": "src/requests/cookies.py", "start_line": 40, "end_line": 70}
    assert calls[2][2] == {
        "filepath": "src/requests/cookies.py",
        "old_string": "return None",
        "new_string": "return {}",
    }
    assert calls[3][2]["filepath"] == "src/requests/_cookie_helpers.py"
    assert "def clean(v):" in calls[3][2]["content"]
    assert "<function=" not in calls[0][0]
    assert "<parameter=" not in calls[2][0]


def test_filtering_resolved_reasoning_py_edits_and_exclude_repos(tmp_path: Path) -> None:
    in_jsonl = tmp_path / "mixed.jsonl"
    out_coder = tmp_path / "coder.jsonl"

    unresolved = _openai_resolved_trajectory(instance_id="bad-1", resolved=False)
    bad_exit = _openai_resolved_trajectory(instance_id="bad-2", resolved=None)
    bad_exit["exit_status"] = "max_step"
    excluded_repo = _openai_resolved_trajectory(instance_id="bad-3", repo="django/django", resolved=True)

    # Empty assistant reasoning
    empty_thought = _openai_resolved_trajectory(instance_id="bad-4", resolved=True)
    empty_thought["messages"][2]["content"] = "   "

    # Non-.py edit only
    md_only = _openai_resolved_trajectory(instance_id="bad-5", resolved=True)
    md_only["messages"][4]["tool_calls"][0]["function"]["arguments"] = json.dumps({
        "command": "str_replace",
        "path": "/testbed/README.md",
        "old_str": "old",
        "new_str": "new",
    })

    good = _openai_resolved_trajectory(instance_id="good-1", resolved=True)

    lines = [json.dumps(r) for r in (unresolved, bad_exit, excluded_repo, empty_thought, md_only, good)]
    in_jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")

    rc = imp.main([
        "--local-jsonl", str(in_jsonl),
        "--exclude-repos", "django/django",
        "--out-coder", str(out_coder),
    ])
    assert rc == 0

    rows = _read_jsonl(out_coder)
    assert [r["meta"]["instance_id"] for r in rows] == ["good-1"]


def test_locator_prefix_extraction_capped_at_6_calls(tmp_path: Path) -> None:
    in_jsonl = tmp_path / "long_explore.jsonl"
    out_locator = tmp_path / "locator.jsonl"

    msgs: list[dict[str, Any]] = [{"role": "user", "content": "Find and fix bug in pkg/core.py."}]
    for idx in range(1, 9):
        msgs.append({
            "role": "assistant",
            "content": f"Exploration step {idx}: inspect source.",
            "tool_calls": [{
                "id": f"c_{idx}",
                "type": "function",
                "function": {
                    "name": "str_replace_editor",
                    "arguments": json.dumps({
                        "command": "view",
                        "path": "/testbed/pkg/core.py",
                        "view_range": [idx * 10, idx * 10 + 9],
                    }),
                },
            }],
        })
        msgs.append({
            "role": "tool",
            "tool_call_id": f"c_{idx}",
            "content": "def target_fn():\n    return 1\n",
        })

    # Edit comes after 8 exploration calls
    msgs.append({
        "role": "assistant",
        "content": "Apply edit in pkg/core.py.",
        "tool_calls": [{
            "id": "c_edit",
            "type": "function",
            "function": {
                "name": "str_replace_editor",
                "arguments": json.dumps({
                    "command": "str_replace",
                    "path": "/testbed/pkg/core.py",
                    "old_str": "return 1",
                    "new_str": "return 2",
                }),
            },
        }],
    })
    msgs.append({"role": "tool", "tool_call_id": "c_edit", "content": "ok"})
    msgs.append({
        "role": "assistant",
        "content": "Run test file.",
        "tool_calls": [{
            "id": "c_test",
            "type": "function",
            "function": {
                "name": "execute_bash",
                "arguments": json.dumps({"command": "pytest tests/test_core.py -q"}),
            },
        }],
    })
    msgs.append({"role": "tool", "tool_call_id": "c_test", "content": "1 passed"})

    rec = {
        "instance_id": "pkg__repo-99",
        "repo": "pkg/repo",
        "resolved": True,
        "problem_statement": "Fix target_fn in pkg/core.py.",
        "messages": msgs,
    }
    in_jsonl.write_text(json.dumps(rec) + "\n", encoding="utf-8")

    rc = imp.main(["--local-jsonl", str(in_jsonl), "--out-locator", str(out_locator)])
    assert rc == 0

    rows = _read_jsonl(out_locator)
    assert len(rows) == 1
    loc_msgs = rows[0]["messages"]
    assert loc_msgs[0]["content"] == (ROOT / "submission" / "prompts" / "analyzer.md").read_text(encoding="utf-8").strip()

    tool_names = [
        c["function"]["name"]
        for m in loc_msgs
        for c in (m.get("tool_calls") or [])
    ]
    assert len(tool_names) == 6
    assert "edit_file" not in tool_names
    assert "write_file" not in tool_names
    assert "submit_patch" not in tool_names

    final_report = loc_msgs[-1]["content"]
    assert not loc_msgs[-1].get("tool_calls")
    assert "pkg/core.py:11-11" in final_report
    assert "tests/test_core.py" in final_report


def test_verifier_go_and_block_from_synthetic_results_dir(tmp_path: Path) -> None:
    res_dir = tmp_path / "results_run"
    patches_dir = res_dir / "patches"
    test_outputs_dir = res_dir / "test_outputs"
    patches_dir.mkdir(parents=True)
    test_outputs_dir.mkdir(parents=True)

    # 1) Empty diff -> BLOCK
    (patches_dir / "task_empty.patch").write_text("", encoding="utf-8")

    # 2) SyntaxError -> BLOCK
    (patches_dir / "task_syntax.patch").write_text(
        "diff --git a/pkg/mod.py b/pkg/mod.py\n"
        "--- a/pkg/mod.py\n+++ b/pkg/mod.py\n"
        "@@ -1,2 +1,2 @@\n-return 1\n+return (\n",
        encoding="utf-8",
    )
    (test_outputs_dir / "task_syntax.log").write_text(
        "tests/test_mod.py:1: in <module>\n"
        "E     File \"pkg/mod.py\", line 2\n"
        "E       return (\n"
        "E              ^\n"
        "E   SyntaxError: '(' was never closed\n",
        encoding="utf-8",
    )

    # 3) Failing test output -> BLOCK
    (patches_dir / "task_fail.patch").write_text(
        "diff --git a/pkg/calc.py b/pkg/calc.py\n"
        "--- a/pkg/calc.py\n+++ b/pkg/calc.py\n"
        "@@ -1,2 +1,2 @@\n-return a + b\n+return a - b\n",
        encoding="utf-8",
    )
    (test_outputs_dir / "task_fail.log").write_text(
        "FAILED tests/test_calc.py::test_add - AssertionError: assert -1 == 5\n",
        encoding="utf-8",
    )

    # 4) Resolved local result -> GO
    (patches_dir / "task_go.patch").write_text(
        "diff --git a/pkg/ok.py b/pkg/ok.py\n"
        "--- a/pkg/ok.py\n+++ b/pkg/ok.py\n"
        "@@ -1,2 +1,2 @@\n-return False\n+return True\n",
        encoding="utf-8",
    )
    (test_outputs_dir / "task_go.log").write_text(
        "tests/test_ok.py .. [100%]\n2 passed in 0.12s\n",
        encoding="utf-8",
    )

    task_rows = [
        {"instance_id": "task_empty", "repo": "org/pkg", "resolved": False, "agent_patch_size": 0, "test_exit_code": 1},
        {"instance_id": "task_syntax", "repo": "org/pkg", "resolved": False, "agent_patch_size": 90, "test_exit_code": 2},
        {"instance_id": "task_fail", "repo": "org/pkg", "resolved": False, "agent_patch_size": 95, "test_exit_code": 1},
        {"instance_id": "task_go", "repo": "org/pkg", "resolved": True, "agent_patch_size": 92, "test_exit_code": 0},
    ]
    (res_dir / "task_results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in task_rows) + "\n",
        encoding="utf-8",
    )

    # Also pass 2 resolved teacher trajectories so GO and BLOCK are balanced (3 GO, 3 BLOCK -> 3 each)
    in_jsonl = tmp_path / "teacher.jsonl"
    in_jsonl.write_text(
        json.dumps(_openai_resolved_trajectory(instance_id="teacher_1", resolved=True)) + "\n"
        + json.dumps(_openai_resolved_trajectory(instance_id="teacher_2", resolved=True)) + "\n",
        encoding="utf-8",
    )

    out_verifier = tmp_path / "verifier.jsonl"
    rc = imp.main([
        "--local-jsonl", str(in_jsonl),
        "--results-dirs", str(res_dir),
        "--out-verifier", str(out_verifier),
    ])
    assert rc == 0

    rows = _read_jsonl(out_verifier)
    verdicts = [r["meta"]["verdict"] for r in rows]
    assert verdicts.count("GO") == 3
    assert verdicts.count("BLOCK") == 3

    verifier_prompt = (ROOT / "submission" / "prompts" / "verifier.md").read_text(encoding="utf-8").strip()
    block_reasons = {r["meta"].get("block_reason") for r in rows if r["meta"]["verdict"] == "BLOCK"}
    assert block_reasons == {"empty_diff", "syntax_error", "test_failure"}

    for row in rows:
        msgs = row["messages"]
        assert msgs[0]["content"] == verifier_prompt
        final_text = msgs[-1]["content"]
        assert f"VERDICT: {row['meta']['verdict']}" in final_text
        for m in msgs:
            for c in m.get("tool_calls") or []:
                assert c["function"]["name"] in {"run_command", "get_status", "read_file"}


def test_hf_dataset_lazy_streaming_bash_patch_and_max_limits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    bash_patch_rec = {
        "instance_id": "bash__repo-7",
        "repo": "bash/repo",
        "is_resolved": True,
        "problem_statement": "Patch app.py via bash.",
        "messages": [
            {"role": "user", "content": "Patch app.py."},
            {
                "role": "assistant",
                "content": "Inspect and patch app.py using python/sed.",
                "tool_calls": [{
                    "id": "b1",
                    "type": "function",
                    "function": {
                        "name": "run_bash",
                        "arguments": json.dumps({"command": "sed -i 's/old/new/' src/app.py"}),
                    },
                }],
            },
            {"role": "tool", "tool_call_id": "b1", "content": ""},
        ],
    }
    too_long_rec = _openai_resolved_trajectory(instance_id="too-long", resolved=True)
    streamed = [too_long_rec, bash_patch_rec, _openai_resolved_trajectory(instance_id="kept-2", resolved=True)]

    fake_datasets = types.ModuleType("datasets")
    calls_seen: list[tuple[str, str, bool]] = []

    def fake_load_dataset(name: str, split: str = "train", streaming: bool = False):
        calls_seen.append((name, split, streaming))
        return iter(streamed)

    fake_datasets.load_dataset = fake_load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    out_coder = tmp_path / "hf_coder.jsonl"
    rc = imp.main([
        "--hf-dataset", "SWE-bench/SWE-smith-trajectories:train",
        "--max-turns", "2",
        "--max-examples", "1",
        "--out-coder", str(out_coder),
    ])
    assert rc == 0
    assert calls_seen == [("SWE-bench/SWE-smith-trajectories", "train", True)]

    rows = _read_jsonl(out_coder)
    assert [r["meta"]["instance_id"] for r in rows] == ["bash__repo-7"]
    assert rows[0]["meta"]["files_changed"] == ["src/app.py"]
    assert rows[0]["messages"][-3]["tool_calls"][0]["function"]["name"] == "submit_patch"
