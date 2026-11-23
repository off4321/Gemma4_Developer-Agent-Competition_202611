#!/usr/bin/env python3
"""Run targeted pytest with bounded output (Observation Masking) to keep the 32k context clean."""

from __future__ import annotations

import argparse
import subprocess
import sys
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
    # Keep first error context + final summary lines within MAX_CHARS.
    head = "\n".join(lines[:15])
    tail = "\n".join(lines[-15:])
    combined = head if len(lines) <= 25 else f"{head}\n...\n{tail}"
    return combined[:MAX_CHARS]


def main() -> int:
    ap = argparse.ArgumentParser(description="Compact targeted pytest runner")
    ap.add_argument("--target", nargs="*", default=[], help="Explicit pytest file or nodeid targets")
    ap.add_argument("--query", default="", help="Substring to find real test_*.py files under tests/")
    ap.add_argument("--timeout", type=int, default=60)
    args = ap.parse_args()

    targets = list(args.target)
    if not targets and args.query:
        targets = _discover_tests(args.query)
    if not targets:
        print("NO_TARGETS: pass --target <path::test> or --query <keyword> matching tests/test_*.py")
        return 2

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
