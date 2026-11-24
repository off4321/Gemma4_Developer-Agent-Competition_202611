#!/usr/bin/env python3
"""Print the exact bytes of the block an `edit_file` call is about to touch.

Why this exists: in the 2026-11-24 90-task run 189 of 230 `edit_file` calls failed, and every
reason was a guessing error - 125x "old_string not found", 23x "No changes to apply"
(old_string == new_string), 18x "Source path ... not found", 14x "Target string occurs N times".
Reading the file with `read_file` does not fix that: the body is indented 8 spaces, the model
retypes it with 4, and the call fails again. This script hands back a copy-pasteable block plus
the two facts `edit_file` enforces - does it appear, and does it appear exactly once.

Usage:
    python3 extract_block.py --filepath fastapi/routing.py --line 323 --context 4
    python3 extract_block.py --filepath fastapi/routing.py --query "on_startup = on_startup"
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

DEFAULT_CONTEXT = 3
MAX_LINES = 120           # the block is for one edit_file, not a code review
MAX_MATCHES = 8           # --all still stops here; more hits means the anchor is hopeless
# Output is sized to fit one `run_command` result (stdout is truncated at 5000 chars).


def _read_text(path: Path) -> str:
    """File text, utf-8 first then latin-1 (repos contain stray cp1252 docstrings)."""
    data = path.read_bytes()
    for encoding in ("utf-8", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _ws_probe(line: str) -> str:
    """Describe the leading whitespace in words the model cannot get wrong."""
    stripped = line.lstrip(" \t")
    indent = line[: len(line) - len(stripped)]
    parts = []
    if " " in indent:
        parts.append(f"{indent.count(' ')} spaces")
    if "\t" in indent:
        parts.append(f"{indent.count(chr(9))} tabs")
    if not parts:
        return "column 0 (no indent)"
    mixed = " AND TABS" if (" " in indent and "\t" in indent) else ""
    return f"{len(indent)} chars: " + " + ".join(parts) + mixed


def _show(lines: list[str], start: int, end: int) -> str:
    """Numbered listing + a verbatim block for lines[start:end] (0-based, end exclusive)."""
    out: list[str] = []
    for number in range(start, min(end, start + MAX_LINES)):
        out.append(f"{number + 1:>5} | {lines[number].rstrip(chr(13))}")
    if end - start > MAX_LINES:
        out.append(f"... truncated after {MAX_LINES} lines (pass --start/--end for a narrower range)")
    return "\n".join(out)


def _trim_blank(lines: list[str], start: int, end: int) -> tuple[int, int]:
    """Drop padding lines that are empty, so the verbatim block never ends on a blank line.

    `edit_file` matches bytes: a trailing "\n" the model cannot see is an 'old_string not found'.
    """
    while end - start > 1 and not lines[end - 1].strip():
        end -= 1
    while end - start > 1 and not lines[start].strip():
        start += 1
    return start, end


def _emit_block(text: str, lines: list[str], start: int, end: int, label: str) -> None:
    start, end = _trim_blank(lines, start, end)
    block = "\n".join(lines[start:end])
    occurrences = text.count(block) if block else 0
    print(f"BLOCK_LINES: {start + 1}-{end} ({end - start} line(s))")
    print(f"LEADING_WHITESPACE(first line of block): {_ws_probe(lines[start])}")
    print(f"OCCURRENCES_OF_THIS_BLOCK_IN_FILE: {occurrences}")
    if occurrences == 1:
        print("UNIQUE: yes -> safe as edit_file old_string")
    elif occurrences == 0:
        print("UNIQUE: no (block not found - did the file change under you?)")
    else:
        print(f"UNIQUE: no ({occurrences} matches) -> widen the range with --context, "
              "otherwise edit_file fails with 'Target string occurs N times'")
    print(f"NEXT_STEP: paste the block below as edit_file old_string VERBATIM (keep the exact "
          f"indentation), write new_string as a full replacement of the same block, and never "
          f"call edit_file with old_string == new_string -> that returns 'No changes to apply'.")
    print(f">>> {label} (raw bytes between the markers, nothing else belongs in old_string)")
    print(block)
    print(f"<<< {label}")


def _run(args: argparse.Namespace) -> int:
    path = Path(args.filepath)
    if not path.is_file():
        candidates = sorted(Path(".").rglob(path.name))[:5]
        print(f"ERROR FILE_NOT_FOUND: {path}")
        print("HINT: the path is wrong; do not retry edit_file with it. Find the real one with "
              "`run_command`: grep -rln --include='*.py' '<symbol from the issue>' . | head -20")
        if candidates:
            print("NEARBY_FILES_WITH_THIS_NAME:")
            for candidate in candidates:
                print(f"  - {candidate}")
        return 1
    if args.line is None and args.query is None and (args.start is None or args.end is None):
        print("ERROR: pass one of --query '<substring>' | --line N | --start N --end M")
        return 2

    text = _read_text(path)
    lines = text.split("\n")
    total = len(lines)
    print(f"FILE: {path} ({total} lines, {len(text)} chars)")

    if args.query is not None:
        def hit(line: str) -> bool:
            return (args.query in line) if args.case_sensitive else (args.query.lower() in line.lower())

        matches = [n for n, line in enumerate(lines) if hit(line)]
        hay = text if args.case_sensitive else text.lower()
        occurrences = hay.count(args.query if args.case_sensitive else args.query.lower())
        print(f"QUERY: {args.query!r} -> {len(matches)} matching line(s), "
              f"{occurrences} occurrence(s) of that string in the file")
        if not matches:
            print("ERROR QUERY_NOT_FOUND: the text you want to edit is not in this file. "
                  "Either the file is wrong (grep -rln it) or the fix is a NEW branch/line: "
                  "then old_string must be an existing anchor line and new_string = anchor + new code.")
            return 1
        selected = (matches if args.all else matches[: max(1, args.top)])[:MAX_MATCHES]
        for position, number in enumerate(selected):
            start = max(0, number - args.context)
            end = min(total, number + args.context + 1)
            if args.line_count > 1:
                end = min(total, end + args.line_count - 1)
            print(f"\n=== MATCH {position + 1}/{len(selected)} at line {number + 1} "
                  f"(showing {start + 1}-{end}) ===")
            print(_show(lines, start, end))
            _emit_block(text, lines, start, end, "BLOCK")
        if len(matches) > len(selected):
            print(f"\n... {len(matches) - len(selected)} more matching line(s) not shown "
                  "(--top N / --all). An ambiguous anchor is the #1 cause of 'Target string occurs N times'.")
        return 0

    if args.line is not None:
        start = max(0, min(total - 1, args.line - 1))
        end = min(total, start + max(1, args.line_count))
        print(f"=== LINES {start + 1}-{end} (context {args.context} requested, line-based range) ===")
        shown_start = max(0, start - args.context)
        print(_show(lines, shown_start, min(total, end + args.context)))
        _emit_block(text, lines, start, end, "BLOCK")
        return 0

    start = max(0, min(total - 1, int(args.start) - 1))
    end = min(total, max(start + 1, int(args.end)))
    print(f"=== LINES {start + 1}-{end} ===")
    print(_show(lines, start, end))
    _emit_block(text, lines, start, end, "BLOCK")
    return 0


def _selftest() -> int:
    """Check the three things the edit_file contract depends on: exact bytes, uniqueness, indent."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pkg.py"
        path.write_text(
            "class Router:\n"
            "    def __init__(self):\n"
            "        self.on_startup = on_startup or []\n"
            "        self.on_startup = on_startup or []\n"
            "\n"
            "    def serve(self):\n"
            "        return self\n",
            encoding="utf-8",
        )
        args = argparse.Namespace(filepath=str(path), query="on_startup", context=0, top=1,
                                 all=False, line=None, line_count=1, start=None, end=None,
                                 case_sensitive=False)
        import io
        import contextlib
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            assert _run(args) == 0
        out = buffer.getvalue()
        assert "2 matching line(s)" in out, out
        assert "UNIQUE: no (2 matches)" in out, "duplicate body must not look unique"
        assert "        self.on_startup = on_startup or []" in out, "verbatim block must keep its 8 spaces"
        assert "8 chars: 8 spaces" in out, out
        assert "BLOCK_LINES: 3-3" in out, out
        buffer.seek(0)
        args = argparse.Namespace(filepath=str(path), query=None, context=0, top=1, all=False,
                                  line=6, line_count=1, start=None, end=None, case_sensitive=False)
        with contextlib.redirect_stdout(buffer):
            assert _run(args) == 0
        assert "UNIQUE: yes" in buffer.getvalue(), buffer.getvalue()
        buffer.seek(0)
        args = argparse.Namespace(filepath=str(path / "missing.py"), query="x", context=0, top=1,
                                  all=False, line=None, line_count=1, start=None, end=None,
                                  case_sensitive=False)
        with contextlib.redirect_stdout(buffer):
            assert _run(args) == 1
        assert "ERROR FILE_NOT_FOUND" in buffer.getvalue(), buffer.getvalue()
    print("SELFTEST: ok (verbatim block, uniqueness flag, indent probe, missing-file hint)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print the exact, uniqueness-checked text block for one edit_file call.")
    parser.add_argument("--filepath", required=False, help="file to read (repo-relative is fine)")
    parser.add_argument("--query", default=None, help="substring/identifier on the line to edit")
    parser.add_argument("--line", type=int, default=None, help="1-based line number to edit")
    parser.add_argument("--line-count", type=int, default=1, help="how many lines the block spans")
    parser.add_argument("--start", type=int, default=None, help="1-based first line of the block")
    parser.add_argument("--end", type=int, default=None, help="1-based last line of the block")
    parser.add_argument("--context", type=int, default=DEFAULT_CONTEXT,
                        help="surrounding lines to include (default 3) - widen it until UNIQUE: yes")
    parser.add_argument("--top", type=int, default=2, help="show at most N matches (default 2)")
    parser.add_argument("--all", action="store_true", help="show every match (still capped by --top)")
    parser.add_argument("--case-sensitive", action="store_true")
    parser.add_argument("--selftest", action="store_true", help="run built-in checks and exit")
    args = parser.parse_args(argv)
    if args.selftest:
        return _selftest()
    if not args.filepath:
        parser.error("--filepath is required (or pass --selftest)")
    return _run(args)


if __name__ == "__main__":
    sys.exit(main())
