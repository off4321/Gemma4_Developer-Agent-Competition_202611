---
name: swe-guard
description: edit_file-safe block reader, targeted pytest runner, and pre-submit patch hygiene / import check.
---

# swe-guard

**IMPORTANT**: Skill reference scripts (`extract_block.py`, `run_pytest_compact.py`, `check_patch.py`,
`safe_edit.py`) live inside the host skill bundle, **NOT** inside `/workspace`.
**NEVER invoke `python3 skills/swe-guard/scripts/...` or `python3 scripts/...` via `run_command`**.
Apply the equivalent workspace-native checks directly with `read_file`, `run_command`, and `verifier`:

1. **Exact block inspection (`extract_block.py` discipline)** - before calling `edit_file`:
   Read exact lines with `read_file(filepath, start_line, end_line)` (`1 <= start_line <= end_line`, <= 150 lines).
   - `old_string not found`: copy 3-6 verbatim lines from `read_file` output (never anchor inside a docstring).
   - `Target string occurs N times`: widen context to 5-8 surrounding code lines until unique.
   - If `edit_file` fails twice on the same block (`safe_edit.py` discipline), run
     `nl -ba <filepath> | sed -n 'X,Yp'` via `run_command` to see exact indentation; if corrupted, run `git checkout -- <filepath>`.
2. **Targeted test execution (`run_pytest_compact.py` discipline)** - bounded pytest with `src/` priority:
   Always prefix with `PYTHONPATH=src:.` and run `PYTHONPATH=src:. python3 -m pytest <single_test_file.py> -q --tb=short`
   (never bare `pytest` or `pytest tests/`).
3. **Pre-submit hygiene (`check_patch.py` discipline)** - mandatory before `submit_patch`:
   Verify runtime import + non-empty clean diff via
   `PYTHONPATH=src:. python3 -c "import <package>" && git diff --check && git diff --stat`, confirm `VERIFY_OK`,
   and pass the `verifier` sub-agent audit (`VERDICT: GO`) before calling `submit_patch()`.

`code_analyzer_agent` and `verifier` are sub-agent tools (`agent_tool`), not skills.
