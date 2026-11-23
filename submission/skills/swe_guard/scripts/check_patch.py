#!/usr/bin/env python3
"""Pre-submit hygiene & Python syntax check to prevent 0B stripped patches and exit-2 collection errors."""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

PROTECTED_NAMES = {
    "conftest.py", "pytest.ini", ".pytest.ini", "pyproject.toml",
    "setup.cfg", "tox.ini", "sitecustomize.py", "usercustomize.py", "_swegemma_stubs.py",
}


def _is_protected(rel: str) -> bool:
    low = rel.lower()
    name = low.rsplit("/", 1)[-1]
    return (
        name.startswith("test_")
        or name.endswith("_test.py")
        or any(seg in f"/{low}" for seg in ("/tests/", "/test/", "/testing/"))
        or name in PROTECTED_NAMES
    )


def main() -> int:
    diff_out = subprocess.run(
        ["git", "diff", "--name-only", "HEAD"], capture_output=True, text=True
    ).stdout.splitlines()
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"], capture_output=True, text=True
    ).stdout.splitlines()

    changed = [f.strip() for f in diff_out if f.strip()]
    kept = [f for f in changed if not _is_protected(f)]
    protected = [f for f in changed if _is_protected(f)]

    syntax_errors: list[str] = []
    for rel in kept:
        path = Path(rel)
        if path.suffix == ".py" and path.is_file():
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=rel)
            except SyntaxError as exc:
                syntax_errors.append(f"{rel}:{exc.lineno}: {exc.msg}")

    print(f"KEPT_FILES: {kept or 'none'}")
    print(f"PROTECTED_FILES: {protected or 'none'}")
    if untracked:
        print(f"UNTRACKED_SCRATCH: {untracked[:10]}")
    if syntax_errors:
        print(f"SYNTAX_ERRORS: {syntax_errors}")
        print("STATUS: BLOCKED (fix syntax errors before submit_patch)")
        return 1
    if not kept:
        print("STATUS: BLOCKED (no non-protected source files modified; patch would strip to 0B)")
        return 1
    print("STATUS: READY_FOR_SUBMIT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
