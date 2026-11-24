---
name: swe-guard
description: edit_file-safe block reader, targeted pytest runner, and pre-submit patch hygiene / syntax check.
---

# swe-guard

**IMPORTANT**: Skill scripts (`extract_block.py`, `run_pytest_compact.py`, `check_patch.py`,
`safe_edit.py`) live inside the ADK skill bundle on the host, **NOT** inside `/workspace`.
**NEVER invoke `python3 skills/swe-guard/scripts/...` or `python3 scripts/...` via `run_command`**
(doing so fails with `No such file or directory`). Instead, apply the equivalent workspace-native
practices directly with `read_file` and `run_command` (or ADK's `run_skill_script` tool):

1. **Exact block inspection (`extract_block.py` discipline)** - before calling `edit_file`:
   Read the exact target lines with `read_file(filepath, start_line, end_line)` (or
   `run_command: nl -ba <filepath> | sed -n 'X,Yp'`).
   - `old_string not found`: copy 3-6 lines verbatim from `read_file` output (including leading spaces).
   - `Target string occurs N times`: widen surrounding context to 5-8 lines until unique.
   - `No changes to apply`: `new_string` must differ from `old_string`.
   - `Source path ... not found`: locate the real path via `find . -maxdepth 4 -name "*<name>*"`.
2. **Targeted test execution (`run_pytest_compact.py` discipline)** - bounded pytest output:
   Run `pytest <single_test_file.py> -k <keyword> -q --tb=short` via `run_command` (never bare
   `pytest` or `pytest tests/`).
3. **Pre-submit hygiene (`check_patch.py` discipline)** - run immediately before `submit_patch`:
   Run `python3 -m py_compile <modified_file.py> && git diff --check && git diff --stat` via
   `run_command` to ensure non-empty library diff, zero syntax errors, and no stray files in `/workspace`.
4. **Line-verified replacement (`safe_edit.py` discipline)**:
   When `edit_file` fails twice on a region, inspect exact numbered lines with `read_file` first
   rather than guessing indentation.

`code_analyzer` is a sub-agent tool, not a skill (`load_skill` cannot load sub-agents).
