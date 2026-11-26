#!/usr/bin/env python3
"""Pre-submit hygiene & Python syntax check to prevent 0B stripped patches and exit-2 collection errors."""

from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

PROTECTED_NAMES = {
    "conftest.py", "pytest.ini", ".pytest.ini", "pyproject.toml",
    "setup.cfg", "tox.ini", "sitecustomize.py", "usercustomize.py", "_swegemma_stubs.py",
}
# A binary/compute artifact inside `git add -N . && git diff HEAD` either fails `git apply` on the
# grader side or is stripped from the patch, and `__pycache__` shows up for free as soon as the
# agent runs `python3 -m py_compile something.py` in /workspace.
ARTIFACT_PARTS = {"__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
                  ".ipynb_checkpoints", ".idea", ".vscode"}
ARTIFACT_SUFFIXES = {".pyc", ".pyo", ".pyd", ".so", ".o", ".a", ".egg-info",
                     ".pt", ".bin", ".ckpt", ".safetensors", ".zip", ".gz", ".tgz"}
IMPORT_ERROR_TYPES = (
    "SyntaxError",
    "NameError",
    "ImportError",
    "ModuleNotFoundError",
    "IndentationError",
)


def _is_protected(rel: str) -> bool:
    """True when the grader resets/strips this path, so editing it cannot earn a score."""
    low = rel.lower()
    name = low.rsplit("/", 1)[-1]
    return (
        name.startswith("test_")
        or name.endswith("_test.py")
        or any(seg in f"/{low}" for seg in ("/tests/", "/test/", "/testing/"))
        or name in PROTECTED_NAMES
    )


def _is_artifact(rel: str) -> bool:
    parts = PurePosixPath(rel).parts
    name = parts[-1] if parts else rel
    if any(part in ARTIFACT_PARTS for part in parts):
        return True
    return PurePosixPath(name).suffix.lower() in ARTIFACT_SUFFIXES


def _derive_module_name(rel: str) -> str | None:
    """Convert a repo-relative .py path into an importable dotted module name."""
    norm = rel[2:] if rel.startswith("./") else rel
    if not norm.endswith(".py"):
        return None
    if norm.startswith("src/"):
        norm = norm[4:]
    stem = norm[:-3]
    if stem == "__init__":
        return None
    if stem.endswith("/__init__"):
        stem = stem[: -len("/__init__")]
    parts = [p for p in stem.split("/") if p]
    if not parts or not all(p.isidentifier() for p in parts):
        return None
    return ".".join(parts)


def _check_import_errors(
    kept: list[str], cwd: Path | None = None, timeout: int = 15
) -> list[str]:
    """Run a runtime import smoke check on kept .py library files."""
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    base_pp = f"src{os.pathsep}."
    env["PYTHONPATH"] = f"{base_pp}{os.pathsep}{existing}" if existing else base_pp

    seen_mods: set[str] = set()
    errors: list[str] = []
    for rel in kept:
        mod = _derive_module_name(rel)
        if not mod or mod in seen_mods:
            continue
        seen_mods.add(mod)
        try:
            proc = subprocess.run(
                [sys.executable, "-c", f"import {mod}"],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                cwd=cwd,
            )
        except subprocess.TimeoutExpired:
            continue
        if proc.returncode != 0:
            combined = f"{proc.stderr}\n{proc.stdout}"
            if any(err in combined for err in IMPORT_ERROR_TYPES):
                lines = [ln.strip() for ln in combined.splitlines() if ln.strip()]
                detail = lines[-1] if lines else f"exit {proc.returncode}"
                errors.append(f"{rel} ({mod}): {detail}")
    return errors


def _selftest() -> int:
    protected = ["tests/test_x.py", "docs/test_docs.py", "tests/conftest.py", "pkg/test_api.py",
                 "pyproject.toml", "pkg/tests/test_y.py", "test_z.py", "tox.ini",
                 "pkg/test_helpers.py", "src/testing/runner.py"]
    fixable = ["pkg/app.py", "fastapi/routing.py", "rich/text.py", "src/pkg/testing_utils.py",
               "app_test_helpers.py", "pkg/tests_util.py"]
    artifacts = ["pkg/__pycache__/app.cpython-312.pyc", "app.pyc", "model.ckpt", "out/solution.zip",
                 "__pycache__/x.pyc", "lib/_c.so"]
    clean = ["pkg/app.py", "tests/../pkg/app.py", "rich/text.py", "data/notes.md"]
    bad = [p for p in protected if not _is_protected(p)]
    over = [p for p in fixable if _is_protected(p)]
    miss = [p for p in artifacts if not _is_artifact(p)]
    over_art = [p for p in clean if _is_artifact(p)]
    if bad or over or miss or over_art:
        print(f"SELFTEST_FAILED protected_missed={bad} protected_overreach={over} "
              f"artifact_missed={miss} artifact_overreach={over_art}")
        return 1

    assert _derive_module_name("src/requests/models.py") == "requests.models"
    assert _derive_module_name("fastapi/routing.py") == "fastapi.routing"
    assert _derive_module_name("src/pkg/__init__.py") == "pkg"
    assert _derive_module_name("docs_src/01-intro/app.py") is None
    assert _derive_module_name("README.md") is None

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "src").mkdir()
        (root / "src" / "_selftest_sg_ok.py").write_text("VALUE = 1\n", encoding="utf-8")
        (root / "src" / "_selftest_sg_bad.py").write_text("VALUE = undefined_sym\n", encoding="utf-8")
        assert _check_import_errors(["src/_selftest_sg_ok.py"], cwd=root) == []
        bad_errs = _check_import_errors(["src/_selftest_sg_bad.py"], cwd=root)
        assert len(bad_errs) == 1 and "NameError" in bad_errs[0], bad_errs

    print("SELFTEST_OK (protected-path, artifact, module-name, and import-smoke checks)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-submit patch hygiene check")
    parser.add_argument("--no-import-check", action="store_true",
                        help="skip runtime import smoke check on kept .py files")
    parser.add_argument("--selftest", action="store_true",
                        help="check the classifiers and exit (never touches git)")
    args = parser.parse_args()
    if args.selftest:
        return _selftest()

    diff_out = subprocess.run(
        ["git", "diff", "--name-only", "HEAD"], capture_output=True, text=True
    ).stdout.splitlines()
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"], capture_output=True, text=True
    ).stdout.splitlines()

    changed = [f.strip() for f in diff_out if f.strip()]
    extras = [f.strip() for f in untracked if f.strip()]
    kept = [f for f in changed if not _is_protected(f)]
    protected = [f for f in changed if _is_protected(f)]
    artifacts = sorted({f for f in changed + extras if _is_artifact(f)})

    syntax_errors: list[str] = []
    for rel in kept:
        path = Path(rel)
        if path.suffix == ".py" and path.is_file():
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=rel)
            except (SyntaxError, ValueError, UnicodeDecodeError) as exc:
                syntax_errors.append(f"{rel}: {getattr(exc, 'lineno', '?')}: {getattr(exc, 'msg', exc)}")

    print(f"KEPT_FILES: {kept or 'none'}")
    print(f"PROTECTED_FILES: {protected or 'none'}")
    if artifacts:
        print(f"ARTIFACTS_IN_PATCH: {artifacts[:10]}")
        print("  -> delete them (`git clean -fdX` / rm the files); a binary file in the diff is "
              "stripped or breaks `git apply`, which scores 0")
    if extras:
        print(f"UNTRACKED_SCRATCH: {extras[:10]}")
    if syntax_errors:
        print(f"SYNTAX_ERRORS: {syntax_errors}")
        print("STATUS: BLOCKED (fix syntax errors before submit_patch)")
        return 1
    if artifacts:
        print("STATUS: BLOCKED (artifacts would ride along in the patch)")
        return 1
    if not kept:
        print("STATUS: BLOCKED (no non-protected source files modified; patch would strip to 0B)")
        return 1
    if not args.no_import_check:
        import_errors = _check_import_errors(kept)
        if import_errors:
            print(f"IMPORT_ERRORS: {import_errors}")
            print("STATUS: BLOCKED (import error)")
            return 1
    print("STATUS: READY_FOR_SUBMIT")
    return 0


if __name__ == "__main__":
    sys.exit(main())
