---
name: swe-guard
description: edit_file-safe block reader, targeted pytest runner, and pre-submit patch hygiene / syntax check.
---

# swe-guard

Run these with `run_command` from the repo root, so paths like `fastapi/routing.py` resolve.
`run_skill_script` is not a shell: it takes a file inside a skill, and passing it
`file_path="verifier"` returned `SKILL_NOT_FOUND` in the 2026-11-24 run. Scripts live in
`skills/swe-guard/scripts/` and each one accepts `--selftest`.

1. **`extract_block.py`** - get the exact bytes to edit *before* calling `edit_file`:
   `python3 skills/swe-guard/scripts/extract_block.py --filepath fastapi/routing.py --query "on_startup = on_startup" --context 3`
   (`--line N --line-count K` and `--start N --end M` also work.) It prints a numbered view, a
   `>>> BLOCK ... <<<` verbatim block, `LEADING_WHITESPACE` and `UNIQUE: yes|no`. It covers every
   way `edit_file` failed in the 2026-11-24 run (189 of 230 calls):
   - `old_string not found` (125x): copy `>>> BLOCK` instead of retyping it - indentation, blank
     lines and trailing spaces are bytes.
   - `Target string occurs N times` (14x): widen `--context` until it says `UNIQUE: yes`.
   - `No changes to apply` (23x): `new_string` was identical to `old_string`; `new_string` is the
     replacement body, so actually change something inside the block or do not call the tool.
   - `Source path ... not found` (15x): wrong file; the script prints `NEARBY_FILES_WITH_THIS_NAME`.
   - New branch/function: there is nothing new to match, so anchor `old_string` on an existing line
     and make `new_string` = that line + your new code.
2. **`run_pytest_compact.py`** - targeted tests with bounded output:
   `python3 skills/swe-guard/scripts/run_pytest_compact.py --query <keyword>` (or
   `--target tests/test_foo.py::test_bar`; `--list-only` shows the files it picked without running
   them). Prints `CMD:`, `EXIT:` and <= 1500 chars that always keep pytest's summary line.
3. **`check_patch.py`** - run immediately before `submit_patch`:
   `python3 skills/swe-guard/scripts/check_patch.py`. Prints `KEPT_FILES` / `PROTECTED_FILES` /
   `ARTIFACTS_IN_PATCH` / `SYNTAX_ERRORS`, then `STATUS: READY_FOR_SUBMIT` or `STATUS: BLOCKED`.
   BLOCKED means only protected paths (`tests/`, `test_*.py`, `conftest.py`, `pyproject.toml`, ...)
   were touched - the patch would strip to 0 B - or a modified `.py` does not parse, which fails
   every test with `exit 2`.

This skill is a set of scripts, not a sub-agent: `code_analyzer` is a sub-agent and is called as a
tool, and neither `code_analyzer` nor `verifier` can be loaded with `load_skill`.
