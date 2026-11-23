---
name: swe_guard
description: Targeted pytest runner with compact output and pre-submit patch hygiene / syntax checker.
---

# swe_guard

Use these two scripts via `run_skill_script` to avoid context bloat and `exit 2` / `NO_PATCH` failures:

1. **`scripts/run_pytest_compact.py`** — Locate real `tests/` files matching a keyword or changed file, run `python3 -m pytest --tb=line -q`, and return only exit code, summary, and up to 1,500 chars of failure lines.
   - Example args: `--query <keyword>` or `--target tests/test_foo.py::test_bar`
2. **`scripts/check_patch.py`** — Run before `submit_patch`. Verifies that:
   - At least one non-protected file is modified in `git diff --name-only` (warns if only `pyproject.toml`, `conftest.py`, `tests/`, etc. were touched).
   - Every modified `.py` file passes `ast.parse` / `py_compile` (catches `exit 2` syntax/import errors before submitting).
