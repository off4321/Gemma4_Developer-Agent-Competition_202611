"""
Kaggle を回す前にできる局所検査。mode は 3 つ:

  contract  submission/ が提出契約を満たすか（package.build_submission と同じ検査）
  patch     patch が Phase 2 を通過しそうか: 保護ファイルの除去 + `git apply --check`
            gold patch に対して走らせれば、**自前の検査が正しいかのテスト**にもなる
            （gold が通らないなら task が全ファイル書き換えか、検査側の誤り）
  harness   `swegemma eval` CLI（Docker sandbox）を正しい引数で組み立てて呼ぶ。
            swegemma が無い環境では実行しない（組み立てだけ表示して exit 0）

なぜ patch 検査が大事か: スコアは Phase 1（patch を出せる）と Phase 2（patch が適用される）の
積で決まる。この mode は後者だけを、GPU も model も無しで測れる。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from package.build_submission import (  # noqa: E402
    Checker, check_adapters, check_eval_config, check_layout, check_models_and_generation,
    iter_agent_docs,
)

PROTECTED_NAMES = ("conftest.py", "pytest.ini", ".pytest.ini", "pyproject.toml", "setup.cfg",
                   "tox.ini", "sitecustomize.py", "usercustomize.py", "_swegemma_stubs.py")
APPLY_PASSES = (
    ["git", "apply", "--unsafe-paths", "-p1", "--check"],
    ["git", "apply", "--unsafe-paths", "-p1", "-3", "--check"],
    ["git", "apply", "--unsafe-paths", "-p1", "--ignore-space-change", "--ignore-whitespace", "--check"],
    ["git", "apply", "--unsafe-paths", "-p1", "--recount", "--check"],
    ["git", "apply", "--unsafe-paths", "-p0", "--check"],
)


def is_protected(path: str) -> bool:
    p = path.lower()
    name = p.rsplit("/", 1)[-1]
    return (name.startswith("test_") or name.endswith("_test.py")
            or any(seg in p for seg in ("/tests/", "/test/", "/testing/"))
            or name in PROTECTED_NAMES)


def _clean_target(target: str) -> str | None:
    target = target.strip()
    if target == "/dev/null":
        return None
    if target.startswith("a/") or target.startswith("b/"):
        return target[2:]
    return target


@dataclass
class _Section:
    """unified diff の 1 file 分。path はヘッダ行（`--- ` / `+++ `）から後追いする。"""

    path: str | None = None
    lines: list[str] = field(default_factory=list)


def split_sections(patch: str) -> list[tuple[str, list[str]]]:
    """Return (path, section_lines) for each file section of a unified diff.

    Handles both `diff --git a/x b/x`-headed diffs and bare `--- a/x / +++ b/x` diffs
    (the published gold patches use the latter form).
    """
    sections: list[_Section] = []
    cur: _Section | None = None
    for line in patch.splitlines(keepends=True):
        if line.startswith("diff --git "):
            match = re.search(r" b/(\S+)", line)
            cur = _Section(match.group(1) if match else None, [line])
            sections.append(cur)
        elif line.startswith("--- "):
            path = _clean_target(line[4:])
            if cur is not None and not cur.path:
                cur.path = path
            else:
                cur = _Section(path, [line])
                sections.append(cur)
        elif line.startswith("+++ ") and cur is not None and not cur.path:
            cur.path = _clean_target(line[4:])
        elif cur is not None:
            cur.lines.append(line)
    return [(sec.path or "", sec.lines) for sec in sections]


def strip_protected_files_from_patch(patch: str) -> tuple[str, list[str]]:
    """Mirror harness `strip_protected_files_from_patch` (HARNESS_README §8.1.3)."""
    kept: list[str] = []
    dropped: list[str] = []
    for path, lines in split_sections(patch):
        if path and is_protected(path):
            dropped.append(path)
            continue
        kept.append("".join(lines))
    return "".join(kept), dropped


def apply_check(patch: str, repo_dir: Path) -> tuple[str, list[str]]:
    """`git apply --check` を passes 順に試す。(通ったコマンド, その間に出たエラー) を返す。

    harness 自身の適用も複数の flag 組み合わせで試すので、1 つでも通れば patch は使える。
    """
    tmp = Path("/tmp/_swegemma_check.patch")
    tmp.write_text(patch, encoding="utf-8")
    errors: list[str] = []
    for cmd in APPLY_PASSES:
        res = subprocess.run(cmd + [str(tmp)], cwd=repo_dir, capture_output=True)
        if res.returncode == 0:
            return " ".join(cmd), errors
        errors.append(f"{' '.join(cmd)} -> {res.stderr.decode('utf-8', 'ignore')[:300]}")
    return "FAILED all git apply passes", errors


def check_patch(patch: str, repo_dir: Path | None) -> dict:
    report = {"sections": 0, "protected_sections": [], "empty_after_strip": False,
              "apply_check": "skipped (no repo checkout)", "apply_errors": []}
    sections = split_sections(patch)
    report["sections"] = len(sections)
    report["paths"] = [p for p, _ in sections]
    stripped, dropped = strip_protected_files_from_patch(patch)
    report["protected_sections"] = dropped
    report["empty_after_strip"] = not stripped.strip()
    if any(len("".join(lines)) > 60_000 for _, lines in sections):
        report["apply_errors"].append("section larger than 60 KB (suspicious whole-file rewrite)")
    if repo_dir is None or not stripped.strip():
        return report  # repo checkout 無し（または tests のみの patch）では apply を測れない
    report["apply_check"], apply_errors = apply_check(stripped, repo_dir)
    report["apply_errors"].extend(apply_errors)
    return report


def patch_inputs(args: argparse.Namespace) -> list[tuple[str, str]] | None:
    """検査対象を [(id, patch_text)] で作る。入力が揃っていなければ None（使い方を表示して exit 2）。

    `--patches-dir` は実行後に落とした agent patch、それを省略すれば tasks.jsonl の gold patch。
    gold patch は上界なので、ここで引っかかるなら自前の検査が間違っている
    （task が全ファイル書き換えの場合は別）。
    """
    if args.patches_dir:
        files = sorted(Path(args.patches_dir).glob("*.patch"))
        if not files:
            print(f"[patch] no *.patch files under {args.patches_dir} (fetch them with: "
                  f"kaggle kernels output <owner>/<kernel> -p results)")
            return None
        return [(path.stem, path.read_text(encoding="utf-8")) for path in files]

    tasks_path = Path(args.tasks)
    if not tasks_path.is_file():
        print(f"[patch] tasks file not found: {tasks_path}")
        return None
    rows = [json.loads(line) for line in tasks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [(row["instance_id"], row.get("patch") or "") for row in rows]


def mode_contract(args: argparse.Namespace) -> int:
    source = Path(args.submission).resolve()
    ck = Checker()
    root_cfg = check_layout(source, ck)
    docs = iter_agent_docs(root_cfg, source)
    check_models_and_generation(docs, ck)
    check_adapters(source, docs, ck)
    check_eval_config(source, ck)
    for warn in ck.warnings:
        print(f"  [warn] {warn}")
    for err in ck.errors:
        print(f"  [ERROR] {err}")
    print("[contract] " + ("PASSED" if not ck.errors else f"FAILED ({len(ck.errors)})"))
    return 0 if not ck.errors else 1


def mode_patch(args: argparse.Namespace) -> int:
    inputs = patch_inputs(args)
    if inputs is None:
        return 2
    repo_dir = Path(args.repo_dir) if args.repo_dir else None
    reports = []
    for identifier, patch_text in inputs[: args.limit or None]:
        report = check_patch(patch_text, repo_dir)
        report["id"] = identifier
        reports.append(report)

    total = len(reports)
    empty = sum(1 for r in reports if r["empty_after_strip"])
    protected = sum(1 for r in reports if r["protected_sections"])
    failed = [r for r in reports if r["apply_check"] == "FAILED all git apply passes"]
    print(f"[patch] reports        : {total}")
    print(f"[patch] empty after strip (tests-only gold patch -> expected in some tasks): {empty}")
    print(f"[patch] sections touching protected test/runner files: {protected}")
    print(f"[patch] apply check    : {reports[0]['apply_check'] if total else 'n/a'}")
    if failed:
        print(f"[patch] apply failures : {len(failed)}")
        for report in failed[:5]:
            print(f"  - {report['id']}: {report['apply_errors'][:1]}")
    if args.verbose:
        for report in reports[: args.limit or 5]:
            print(json.dumps(report, indent=1, ensure_ascii=False)[:1200])
    return 0


def swegemma_installed() -> bool:
    """`swegemma` CLI がこの環境に有るか（wheelhouse を入れた kernel / コンテナだけ True）。"""
    try:
        subprocess.run(["swegemma", "--help"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return True


def mode_harness(args: argparse.Namespace) -> int:
    cmd = [
        "swegemma", "eval",
        "--tasks", args.tasks,
        "--snapshots-dir", args.snapshots_dir,
        "--submission-dir", args.submission,
        "--results-dir", args.results_dir,
        "--sandbox", args.sandbox,
        "--max-tool-calls", str(args.max_tool_calls),
        "--max-time-minutes", str(args.max_time_minutes),
        "--concurrency", str(args.concurrency),
        "--timeout-seconds", str(args.timeout_seconds),
    ]
    if args.task_id:
        cmd += ["--task-ids", *args.task_id]
    print("[harness] " + " ".join(cmd))
    if not swegemma_installed():
        print("[harness] `swegemma` is not installed locally. Run this inside the Kaggle kernel "
              "(kernels/eval/) where the wheelhouse installs it; see doc/kaggle_cli.md")
        return 0
    if args.dry_run:
        return 0
    return subprocess.run(cmd).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Local pre-flight evaluation checks")
    parser.add_argument("--mode", choices=["contract", "patch", "harness"], default="contract")
    parser.add_argument("--submission", default="submission")
    parser.add_argument("--tasks", default="data/tasks.jsonl")
    parser.add_argument("--patches-dir", default=None)
    parser.add_argument("--repo-dir", default=None, help="checkout of the task repo at base_commit")
    parser.add_argument("--snapshots-dir", default="data/snapshots")
    parser.add_argument("--results-dir", default="results/run_local")
    parser.add_argument("--sandbox", default="docker", choices=["docker", "subprocess"])
    parser.add_argument("--max-tool-calls", type=int, default=60)
    parser.add_argument("--max-time-minutes", type=float, default=30)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--task-id", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return {"contract": mode_contract, "patch": mode_patch, "harness": mode_harness}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
