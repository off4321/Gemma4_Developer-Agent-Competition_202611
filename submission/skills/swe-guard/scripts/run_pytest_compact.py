#!/usr/bin/env python3
"""Run targeted pytest with bounded output (Observation Masking) to keep the 32k context clean."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

MAX_CHARS = 1500


def _discover_tests(query: str, limit: int = 5) -> list[str]:
    tests_dir = Path("tests")
    if not tests_dir.is_dir():
        return []
    q = query.lower()
    hits = [str(p) for p in sorted(tests_dir.rglob("test_*.py")) if q in str(p).lower()]
    return hits[:limit]


def _compact_output(stdout: str, stderr: str) -> str:
    lines = [line for line in (stdout + "\n" + stderr).splitlines() if line.strip()]
    if not lines:
        return "(no output)"
    # Keep the first failure lines AND the final summary inside MAX_CHARS. The summary ("2 failed
    # in 0.4s") is what the next action depends on, so it gets the tail of the budget even when
    # the head lines are long (a plain combined[:MAX_CHARS] used to drop it).
    if len(lines) <= 25:
        short = "\n".join(lines)
        return short if len(short) <= MAX_CHARS else short[-MAX_CHARS:]
    head_budget = MAX_CHARS // 2
    head_lines: list[str] = []
    used = 0
    for line in lines[:15]:
        if used + len(line) + 1 > head_budget:
            break
        head_lines.append(line)
        used += len(line) + 1
    marker = "\n... [truncated] ...\n"
    tail_budget = max(0, MAX_CHARS - used - len(marker))
    tail = "\n".join(lines[-15:])[-tail_budget:]
    return "".join(head_lines) + marker + tail


def _selftest() -> int:
    """Discovery must find nested real test files, and output must stay bounded."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "tests" / "nest").mkdir(parents=True)
        (root / "tests" / "test_startup.py").write_text("def test_a():\n    assert True\n", encoding="utf-8")
        (root / "tests" / "nest" / "test_deep.py").write_text("def test_b():\n    assert True\n", encoding="utf-8")
        (root / "tests" / "notes.txt").write_text("not a test\n", encoding="utf-8")
        cwd = Path.cwd()
        os.chdir(root)
        try:
            assert _discover_tests("nest") == [str(Path("tests") / "nest" / "test_deep.py")]
            assert _discover_tests("startup") == [str(Path("tests") / "test_startup.py")]
            assert _discover_tests("notes") == [], "non-test files are not targets"
        finally:
            os.chdir(cwd)
    noisy = "\n".join(f"line {n} " + "x" * 80 for n in range(400))
    compacted = _compact_output(noisy, "")
    assert len(compacted) <= MAX_CHARS, len(compacted)
    assert compacted.startswith("line 0 "), compacted[:40]
    assert "line 399 " in compacted, "the final summary lines must survive truncation"
    assert "line 200 " not in compacted, "the middle must be truncated"
    assert _compact_output("", "") == "(no output)"
    assert _compact_output("2 failed in 0.42s\n", "") == "2 failed in 0.42s"
    print("SELFTEST_OK (nested discovery, non-test files ignored, output bounded, summary kept)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Compact targeted pytest runner")
    ap.add_argument("--target", nargs="*", default=[], help="Explicit pytest file or nodeid targets")
    ap.add_argument("--query", default="", help="Substring to find real test_*.py files under tests/")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--list-only", action="store_true",
                    help="only print the discovered targets (never runs pytest)")
    ap.add_argument("--selftest", action="store_true", help="check discovery/output bounding, then exit")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()

    targets = list(args.target)
    if not targets and args.query:
        targets = _discover_tests(args.query)
    if not targets:
        print("NO_TARGETS: pass --target <path::test> or --query <keyword> matching tests/test_*.py")
        return 2
    if args.list_only:
        print("TARGETS: " + " ".join(targets))
        return 0

    cmd = [sys.executable, "-m", "pytest", *targets, "--tb=line", "-q"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT ({args.timeout}s): {' '.join(cmd)}")
        return 124

    print(f"CMD: {' '.join(cmd)}")
    print(f"EXIT: {proc.returncode}")
    print(_compact_output(proc.stdout, proc.stderr))
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
