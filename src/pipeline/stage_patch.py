"""
ステージ `patch` = モデルを使わない patch の静的検査。

使い道は 2 つで、その区別が doc/harness_watchpoints.md の要点そのもの:
  * 実行前: tasks.jsonl の **gold patch** を解析する。上位の基準であり、同時に自前の
    保護ファイル strip / apply 検査が正しく働いているかのテストでもある。
  * 実行後: results/<run>/patches/*.patch、つまり agent が実際に出した diff を解析する。

Phase 1「patch を出せる力」と Phase 2「patch が適用される力」は別の失敗で、この stage は後者だけを
測る。実処理は src/evaluate/evaluate_harness.py に委譲する。
"""

from __future__ import annotations

import argparse
from typing import Any

from pipeline.common import ensure_src_on_path, find_results_dir, log

ensure_src_on_path()
from evaluate.evaluate_harness import mode_patch  # noqa: E402


def run(ctx: Any) -> None:
    results_dir = find_results_dir(ctx)
    patches_dir = results_dir / "patches" if results_dir else None
    limit = int((ctx.cfg.raw or {}).get("patch_limit", 20))

    if patches_dir and any(patches_dir.glob("*.patch")):
        log("patch", f"agent の patch を解析する: {patches_dir}")
        args = argparse.Namespace(patches_dir=str(patches_dir), tasks=str(ctx.paths.tasks_path),
                                  repo_dir=None, limit=limit, verbose=ctx.cfg.verbose)
    else:
        if not ctx.paths.tasks_path.is_file():
            log("patch", "[skip] tasks.jsonl も results の patch も無い")
            return
        log("patch", f"agent patch が見つからない（{results_dir or 'results/'} を見た） -> "
                     f"{ctx.paths.tasks_path} の gold patch にフォールバックする")
        args = argparse.Namespace(patches_dir=None, tasks=str(ctx.paths.tasks_path),
                                  repo_dir=None, limit=limit, verbose=ctx.cfg.verbose)

    code = mode_patch(args)
    if code != 0:
        log("patch", f"[warn] mode_patch の return code = {code}")
    ctx.notes["patch_source"] = str(patches_dir or ctx.paths.tasks_path)
