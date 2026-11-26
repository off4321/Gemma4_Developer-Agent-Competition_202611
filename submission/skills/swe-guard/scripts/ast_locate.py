#!/usr/bin/env python3
"""Locate symbol definitions, cross-file callers/refs, and test files with bounded output."""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

MAX_CHARS = 2500
MAX_TEST_FILES = 10
MAX_DEFS = 20
MAX_LINE_CHARS = 120
SKIP_DIRS = {
    ".git",
    ".venv",
    "docs",
    "docs_src",
    "build",
    "dist",
    "__pycache__",
    "tests",
    "test",
    "testing",
}


def _read_text(path: Path) -> str:
    """Read file text safely without crashing on stray encodings."""
    data = path.read_bytes()
    for enc in ("utf-8", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _iter_source_files(root: Path) -> list[tuple[str, Path]]:
    """Walk .py source files under root while pruning skipped directories."""
    results: list[tuple[str, Path]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames if d not in SKIP_DIRS and not d.endswith(".egg-info")
        )
        base = Path(dirpath)
        for fname in sorted(filenames):
            if fname.endswith(".py"):
                full = base / fname
                rel = full.relative_to(root).as_posix()
                results.append((rel, full))
    return results


def _extract_signature(lines: list[str], start_lineno: int) -> str:
    """Extract a collapsed single-line class/def signature starting at start_lineno."""
    chunks: list[str] = []
    depth = 0
    in_quote: str | None = None
    escaped = False
    max_idx = min(len(lines), start_lineno - 1 + 8)
    for idx in range(start_lineno - 1, max_idx):
        raw = lines[idx].strip()
        chunks.append(raw)
        i = 0
        while i < len(raw):
            ch = raw[i]
            if escaped:
                escaped = False
                i += 1
                continue
            if ch == "\\" and in_quote:
                escaped = True
                i += 1
                continue
            if in_quote:
                if ch == in_quote:
                    in_quote = None
            else:
                if ch in ("'", '"'):
                    in_quote = ch
                elif ch == "#":
                    break
                elif ch in "([{":
                    depth += 1
                elif ch in ")]}":
                    depth = max(0, depth - 1)
                elif ch == ":" and depth == 0:
                    sig = re.sub(r"\s+", " ", " ".join(chunks)).strip()
                    sig = sig.split("#", 1)[0].rstrip()
                    if len(sig) > 140:
                        sig = sig[:137] + "..."
                    return sig
            i += 1
    sig = re.sub(r"\s+", " ", " ".join(chunks)).strip()
    return sig[:137] + "..." if len(sig) > 140 else sig


def _collect_defs(
    tree: ast.AST, lines: list[str], rel: str, symbol: str
) -> tuple[list[tuple[int, str, int, int, str]], set[tuple[str, int]]]:
    """Find matching class/def/async def nodes, prioritizing exact over substring matches."""
    sym_low = symbol.lower()
    defs: list[tuple[int, str, int, int, str]] = []
    def_lines: set[tuple[str, int]] = set()

    def visit(node: ast.AST, parents: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                qual = ".".join([*parents, child.name])
                name_low = child.name.lower()
                qual_low = qual.lower()
                prio: int | None = None
                if child.name == symbol or qual == symbol:
                    prio = 0
                elif name_low == sym_low or qual_low == sym_low:
                    prio = 1
                elif sym_low in name_low or sym_low in qual_low:
                    prio = 2
                if prio is not None:
                    start = child.lineno
                    end = getattr(child, "end_lineno", start)
                    sig = _extract_signature(lines, start)
                    defs.append((prio, rel, start, end, sig))
                    def_lines.add((rel, start))
                visit(child, [*parents, child.name])
            else:
                visit(child, parents)

    visit(tree, [])
    return defs, def_lines


def _select_round_robin(
    refs_by_file: dict[str, list[tuple[int, str]]], max_callers: int
) -> list[tuple[str, int, str]]:
    """Select up to max_callers references distributed across distinct files."""
    if max_callers <= 0 or not refs_by_file:
        return []
    files = sorted(refs_by_file.keys())
    indices = {f: 0 for f in files}
    chosen_by_file: dict[str, list[tuple[int, str]]] = defaultdict(list)
    total = 0
    while total < max_callers:
        progressed = False
        for f in files:
            idx = indices[f]
            items = refs_by_file[f]
            if idx < len(items):
                chosen_by_file[f].append(items[idx])
                indices[f] = idx + 1
                total += 1
                progressed = True
                if total >= max_callers:
                    break
        if not progressed:
            break
    grouped: list[tuple[str, int, str]] = []
    for f in files:
        for lineno, text in chosen_by_file.get(f, ()):
            grouped.append((f, lineno, text))
    return grouped


def _find_test_files(root: Path, symbol: str, pat: re.Pattern[str]) -> list[str]:
    """Return up to MAX_TEST_FILES tests/**/test_*.py matching symbol by name or content."""
    tests_dir = root / "tests"
    if not tests_dir.is_dir():
        return []
    sym_low = symbol.lower()
    name_hits: list[str] = []
    content_hits: list[str] = []
    for path in sorted(tests_dir.rglob("test_*.py")):
        rel = path.relative_to(root).as_posix()
        if sym_low in path.name.lower():
            name_hits.append(rel)
        elif pat.search(_read_text(path)):
            content_hits.append(rel)
    combined = name_hits + [h for h in content_hits if h not in name_hits]
    return combined[:MAX_TEST_FILES]


def locate_symbol(root: Path, symbol: str, max_callers: int = 20) -> str:
    """Locate definitions, cross-file refs, and tests for symbol under root."""
    if re.fullmatch(r"[A-Za-z_]\w*", symbol):
        pat = re.compile(rf"\b{re.escape(symbol)}\b")
    else:
        pat = re.compile(re.escape(symbol))

    all_defs: list[tuple[int, str, int, int, str]] = []
    def_lines: set[tuple[str, int]] = set()
    refs_by_file: dict[str, list[tuple[int, str]]] = defaultdict(list)

    for rel, full in _iter_source_files(root):
        text = _read_text(full)
        lines = text.splitlines()
        try:
            tree = ast.parse(text, filename=rel)
        except (SyntaxError, ValueError):
            tree = None
        if tree is not None:
            file_defs, file_def_lines = _collect_defs(tree, lines, rel, symbol)
            all_defs.extend(file_defs)
            def_lines.update(file_def_lines)

        for idx, line in enumerate(lines, start=1):
            if (rel, idx) in def_lines:
                continue
            if pat.search(line):
                trimmed = line.strip()
                if len(trimmed) > MAX_LINE_CHARS:
                    trimmed = trimmed[: MAX_LINE_CHARS - 3] + "..."
                refs_by_file[rel].append((idx, trimmed))

    all_defs.sort(key=lambda item: (item[0], item[1], item[2]))
    def_entries = [
        f"  {rel}:{start}-{end} {sig}" for _, rel, start, end, sig in all_defs[:MAX_DEFS]
    ]
    if len(all_defs) > MAX_DEFS:
        def_entries.append(f"  ... (+{len(all_defs) - MAX_DEFS} more)")

    selected_refs = _select_round_robin(refs_by_file, max_callers)
    ref_entries = [f"  {rel}:{lineno}: {txt}" for rel, lineno, txt in selected_refs]

    test_files = _find_test_files(root, symbol, pat)
    test_entries = [f"  {tf}" for tf in test_files]

    return _format_bounded(def_entries, ref_entries, test_entries)


def _format_bounded(
    def_entries: list[str], ref_entries: list[str], test_entries: list[str]
) -> str:
    """Format the 3 output sections while strictly keeping total length <= MAX_CHARS."""
    test_block = "TEST_FILES:\n" + ("\n".join(test_entries) if test_entries else "  (none)")

    # Reserve space for TEST_FILES and section headers; fit DEFINITIONS then CALLERS/REFS.
    def_header = "DEFINITIONS:\n"
    ref_header = "CALLERS/REFS:\n"
    fixed_overhead = len(def_header) + len(ref_header) + len(test_block) + 4

    def_budget = max(200, min(1000, MAX_CHARS - fixed_overhead - 200))
    kept_defs: list[str] = []
    used_defs = 0
    for entry in def_entries:
        cost = len(entry) + 1
        if used_defs + cost > def_budget and kept_defs:
            kept_defs.append(f"  ... (+{len(def_entries) - len(kept_defs)} more)")
            break
        kept_defs.append(entry)
        used_defs += cost
    def_block = def_header + ("\n".join(kept_defs) if kept_defs else "  (none)")

    ref_budget = MAX_CHARS - len(def_block) - len(test_block) - len(ref_header) - 4
    kept_refs: list[str] = []
    used_refs = 0
    for entry in ref_entries:
        cost = len(entry) + 1
        if used_refs + cost + 24 > ref_budget and kept_refs:
            kept_refs.append(f"  ... (+{len(ref_entries) - len(kept_refs)} more)")
            break
        kept_refs.append(entry)
        used_refs += cost
    ref_block = ref_header + ("\n".join(kept_refs) if kept_refs else "  (none)")

    output = f"{def_block}\n{ref_block}\n{test_block}"
    if len(output) > MAX_CHARS:
        output = output[: MAX_CHARS - 16] + "\n... [truncated]"
    return output


def _selftest() -> int:
    """Test symbol location, multi-file caller distribution, skip dirs, and char cap in /tmp."""
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        root = Path(tmp)
        (root / "pkg" / "openapi").mkdir(parents=True)
        (root / "tests" / "nest").mkdir(parents=True)
        (root / "docs_src").mkdir()
        (root / ".venv" / "lib").mkdir(parents=True)

        (root / "pkg" / "routing.py").write_text(
            "class APIRoute:\n    def __init__(self, swagger_ui_parameters=None):\n"
            "        self.swagger_ui_parameters = swagger_ui_parameters\n",
            encoding="utf-8",
        )
        # Many occurrences in applications.py must not starve routing.py or openapi/docs.py
        app_refs = "\n".join(
            f"        cfg_{i} = swagger_ui_parameters" for i in range(25)
        )
        (root / "pkg" / "applications.py").write_text(
            "def build_swagger_ui_parameters():\n    return {}\n\n"
            "class FastAPI:\n    def setup(self, swagger_ui_parameters=None):\n"
            f"{app_refs}\n",
            encoding="utf-8",
        )
        (root / "pkg" / "openapi" / "docs.py").write_text(
            "def get_swagger_ui_html(swagger_ui_parameters=None):\n"
            "    return swagger_ui_parameters\n",
            encoding="utf-8",
        )
        (root / "docs_src" / "ignore_me.py").write_text(
            "def swagger_ui_parameters():\n    pass\n", encoding="utf-8"
        )
        (root / ".venv" / "lib" / "ignore_venv.py").write_text(
            "swagger_ui_parameters = 1\n", encoding="utf-8"
        )
        (root / "tests" / "test_swagger_ui_parameters.py").write_text(
            "def test_ok():\n    pass\n", encoding="utf-8"
        )
        (root / "tests" / "nest" / "test_app.py").write_text(
            "def test_ref():\n    x = swagger_ui_parameters\n", encoding="utf-8"
        )

        out = locate_symbol(root, "swagger_ui_parameters", max_callers=9)
        assert len(out) <= MAX_CHARS, len(out)
        assert "DEFINITIONS:" in out and "CALLERS/REFS:" in out and "TEST_FILES:" in out
        assert "pkg/applications.py:1-2 def build_swagger_ui_parameters()" in out
        assert "pkg/applications.py:" in out
        assert "pkg/routing.py:" in out
        assert "pkg/openapi/docs.py:" in out
        assert "docs_src/ignore_me.py" not in out
        assert "ignore_venv.py" not in out
        assert "tests/test_swagger_ui_parameters.py" in out
        assert "tests/nest/test_app.py" in out

        # Verify exact def match sorts before substring match
        (root / "pkg" / "exact.py").write_text(
            "def get_swagger_ui_html_extra():\n    pass\n", encoding="utf-8"
        )
        exact_out = locate_symbol(root, "get_swagger_ui_html", max_callers=5)
        def_section = exact_out.split("CALLERS/REFS:")[0]
        assert def_section.index("pkg/openapi/docs.py") < def_section.index("pkg/exact.py")

    print("SELFTEST_OK (defs, multi-file round-robin callers, skip-dirs, test discovery, 2500-char cap)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Compact AST symbol & cross-file reference locator")
    ap.add_argument("--symbol", help="Symbol name to locate (required unless --selftest)")
    ap.add_argument("--root", default=".", help="Repository root directory (default: .)")
    ap.add_argument("--max-callers", type=int, default=20, help="Max caller/ref lines (default: 20)")
    ap.add_argument("--selftest", action="store_true", help="Run self-test in /tmp and exit")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()
    if not args.symbol:
        ap.error("--symbol is required unless --selftest is passed")
    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"ERROR: root directory not found: {args.root}", file=sys.stderr)
        return 2
    print(locate_symbol(root, args.symbol, max_callers=args.max_callers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
