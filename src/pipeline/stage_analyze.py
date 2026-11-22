"""
ステージ `analyze` = ループを閉じる。results/<run> を失敗 taxonomy に変える。

`make out-eval RUN=<variant>` が Kaggle から落とした results dir を
src/analyze/analyze_results.py に渡す。ここでの出力が、variant スイープで次にどのノブを
回すかを決定する: NO_PATCH / apply 失敗 / 保護ファイル污染 / 本物のテスト失敗。

results dir が見つからない場合は失敗ではなく skip する（実行前にも post preset を打てるようにする）。
"""

from __future__ import annotations

from typing import Any

from pipeline.common import ensure_src_on_path, find_results_dir, log

ensure_src_on_path()
from analyze import analyze_results  # noqa: E402


def run(ctx: Any) -> None:
    results_dir = find_results_dir(ctx)
    if results_dir is None:
        log("analyze", f"[skip] run {ctx.cfg.run_name!r} の results dir が見つからない。先に落とす:")
        log("analyze", "        make out-eval RUN=" + ctx.cfg.run_name)
        return
    show = int((ctx.cfg.raw or {}).get("show_failures", 8))
    log("analyze", f"解析対象: {results_dir}")

    # task 別の不出来をファイルに落とす（Colab notebook の cell 7 に注入している物と同じ実装）
    from pipeline.failures_report import write_failures  # noqa: E402

    ids = list(getattr(ctx.cfg, "task_ids", None) or [])
    if not ids:
        ids = [t.get("instance_id") for t in (getattr(ctx, "tasks", None) or []) if t.get("instance_id")]
    try:
        report = write_failures(results_dir, ids or None)
    except SystemExit as exc:
        log("analyze", f"[warn] failures_report: {exc}")
    else:
        official = any((results_dir / name).exists() for name in ("task_results.jsonl", "summary.json", "patches"))
        if official:
            code = analyze_results.main(["--results", str(results_dir), "--show-failures", str(show)])
            if code != 0:
                log("analyze", f"[warn] analyze_results の return code = {code}")
        else:
            log("analyze", "[skip] analyze_results.py は公式 harness 形式（task_results.jsonl / summary.json / "
                           "patches/）専用。この dir は自分の pipeline 形式なので failures.csv / failures.md で読む")

        counts = report["counts"]
        log("analyze", f"form={report['form']} 行={report['rows']} 内訳="
                       + ", ".join(f"{k}:{v}" for k, v in sorted(counts.items())))
        log("analyze", f"task 別の不出来: {report['csv']} / {report['md']}")
        ctx.notes["failures_csv"] = str(report["csv"])
        ctx.notes["failures_md"] = str(report["md"])
    ctx.notes["analyzed_results"] = str(results_dir)
