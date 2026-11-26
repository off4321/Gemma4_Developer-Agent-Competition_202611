---
name: swe-guard
description: AST locator, edit_file-safe block reader/editor, targeted pytest runner, pre-submit patch hygiene, and conventions resource.
---

# swe-guard

ADK's `run_skill_script` tool executes scripts from `swe-guard` directly inside the sandbox container
(sharing `/workspace`), and `load_skill_resource` loads `resources/conventions.md`.
**NEVER run `python3 skills/swe-guard/scripts/...` or `python3 scripts/...` via `run_command`** —
invoke them ONLY via `run_skill_script` (NOT `run_command`).

## Skill Scripts (`run_skill_script`)
1. `ast_locate.py` — AST symbol locator: finds definitions, callers, and matching tests across `/workspace` in one compact call.
2. `extract_block.py` — Exact block reader: prints verbatim lines with indentation info and uniqueness check before `edit_file` (fallback: `read_file` with explicit `1 <= start_line <= end_line <= 150`).
3. `safe_edit.py` — Fallback 1-indexed line-range editor (`--filepath`, `--start`, `--end`, `--replace-file` / `--stdin`) with protected-path guard and `ast.parse` rollback if `edit_file` fails twice.
4. `run_pytest_compact.py` — Bounded targeted pytest runner (`--target` / `--query`, `src/` priority, output capped to keep 32k context clean).
5. `check_patch.py` — Pre-submit hygiene & syntax check: verifies non-empty non-protected diff, no untracked/binary artifacts, and valid `ast.parse` syntax before `verifier` and `submit_patch()`.

## Skill Resource (`load_skill_resource`)
- `resources/conventions.md` — Repo conventions checklist (sibling exception messages, warning classes, and caller propagation).

`code_analyzer_agent` and `verifier` are sub-agent tools (`agent_tool`), not skills.
