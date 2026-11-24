#!/usr/bin/env python3
"""Line-range file replacement with syntax check and automatic rollback.

Fallback for when `edit_file` fails with `old_string not found` due to whitespace
or indentation mismatches. Replaces 1-indexed inclusive lines `[--start, --end]`
with content read from `--replace-file` or `--stdin`, checks protected paths, and
verifies `.py` files with `ast.parse` (rolling back on SyntaxError).
"""

from __future__ import annotations

import argparse
import ast
import difflib
import sys
import tempfile
from pathlib import Path

PROTECTED_NAMES = {"conftest.py", "pytest.ini", "pyproject.toml", "setup.cfg", "tox.ini"}
PROTECTED_DIRS = {"tests", "test", "testing"}


def is_protected(path: Path) -> bool:
    parts = path.parts
    name = path.name
    if name in PROTECTED_NAMES or name.startswith("test_") or name.endswith("_test.py"):
        return True
    return any(part in PROTECTED_DIRS for part in parts)


def apply_line_edit(filepath: Path, start: int, end: int, replacement: str,
                    allow_protected: bool = False) -> tuple[bool, str]:
    if not filepath.is_file():
        return False, f"ERROR: file not found: {filepath}"
    if not allow_protected and is_protected(filepath):
        return False, f"ERROR: refusing to edit protected test/config file: {filepath}"
    if start < 1 or end < start:
        return False, f"ERROR: invalid 1-indexed range start={start} end={end}"

    original_bytes = filepath.read_bytes()
    original_text = original_bytes.decode("utf-8")
    lines = original_text.splitlines(keepends=True)
    if start > len(lines):
        return False, f"ERROR: start={start} exceeds file length ({len(lines)} lines)"
    end_clamped = min(end, len(lines))

    rep_lines = replacement.splitlines(keepends=True)
    if replacement and not replacement.endswith("\n") and (
        end_clamped < len(lines) or (lines and lines[-1].endswith("\n"))
    ):
        rep_lines[-1] = rep_lines[-1] + "\n"

    updated_lines = lines[: start - 1] + rep_lines + lines[end_clamped:]
    updated_text = "".join(updated_lines)
    if updated_text == original_text:
        return False, "ERROR: no changes to apply (replacement matches existing lines)"

    if filepath.suffix == ".py":
        try:
            ast.parse(updated_text, filename=str(filepath))
        except SyntaxError as exc:
            return False, (
                f"ROLLED_BACK: SyntaxError at line {exc.lineno}: {exc.msg}\n"
                f"  {(exc.text or '').rstrip()}"
            )

    filepath.write_bytes(updated_text.encode("utf-8"))
    diff = "".join(
        difflib.unified_diff(
            original_text.splitlines(keepends=True),
            updated_text.splitlines(keepends=True),
            fromfile=f"a/{filepath}",
            tofile=f"b/{filepath}",
            n=3,
        )
    )
    return True, f"STATUS: OK ({filepath}:{start}-{end_clamped} updated)\n{diff[:2000]}"


def _selftest() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "mod.py"
        target.write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
        ok, msg = apply_line_edit(target, 2, 2, "    return a + b\n")
        assert ok and "STATUS: OK" in msg, msg
        assert "return a + b" in target.read_text(encoding="utf-8")

        ok_bad, msg_bad = apply_line_edit(target, 2, 2, "   def broken(:\n")
        assert not ok_bad and "ROLLED_BACK" in msg_bad, msg_bad
        assert "return a + b" in target.read_text(encoding="utf-8")

        prot = Path(tmp) / "test_mod.py"
        prot.write_text("x = 1\n", encoding="utf-8")
        ok_p, _ = apply_line_edit(prot, 1, 1, "x = 2\n")
        assert not ok_p
    print("SELFTEST: OK")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Safe 1-indexed line-range editor with syntax rollback")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--filepath", default="")
    p.add_argument("--start", type=int, default=0, help="1-indexed inclusive start line")
    p.add_argument("--end", type=int, default=0, help="1-indexed inclusive end line")
    p.add_argument("--replace-file", default="", help="path to file containing replacement lines (e.g. /tmp/new.txt)")
    p.add_argument("--stdin", action="store_true", help="read replacement text from stdin")
    p.add_argument("--allow-protected", action="store_true")
    args = p.parse_args()

    if args.selftest:
        return _selftest()
    if not args.filepath or args.start < 1 or args.end < args.start:
        p.error("--filepath and valid 1-indexed --start <= --end are required")
    if args.replace_file:
        replacement = Path(args.replace_file).read_text(encoding="utf-8")
    elif args.stdin:
        replacement = sys.stdin.read()
    else:
        p.error("provide --replace-file /tmp/chunk.py or --stdin")
        return 2

    ok, report = apply_line_edit(Path(args.filepath), args.start, args.end, replacement, args.allow_protected)
    print(report)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
