#!/usr/bin/env python3
"""
実行 -> 評価のループ全体の入口（1 コマンド）。

    # 1) 手元（GPU なし、swegemma なし）: 提出物が契約を通るか、variant が正しく適用されるか
    python3 src/pipeline/run_pipeline.py --variant baseline --stages preflight

    # 2) Kaggle GPU: ノートブックと同じ手順（setup..package）を 1 variant 分走らせる
    python3 src/pipeline/run_pipeline.py --variant baseline --stages gpu        # kernel 内

    # 3) `make out-eval RUN=<variant>` の後: 何がなぜ失敗したかを taxonomy に
    python3 src/pipeline/run_pipeline.py --variant baseline --stages post

ステージ名とプリセットは pipeline.common 由来（cell との対応表は doc/pipeline.md）。
run が依存するものはすべて variants.yaml に固定する: ここが崩れると 2 run を比較できない。

strict モード（既定: Kaggle 内=True / 手元=False）:
  GPU stage のライブラリが無いとき、手元は「理由付きで skip」、kernel は「即失敗」。
  kernel 内で skip されるのは設定ミス（dataset 未添付など）の意味なので、そこは失敗として出す。

import bootstrap: src/ を sys.path に足して pipeline.* と兄弟 package（package.* 等）を
トップレベルで import する。src/evaluate/*.py と同じ。
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path
from typing import Any

_SRC = str(Path(__file__).resolve().parents[1])
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from pipeline import config  # noqa: E402
from pipeline.common import (  # noqa: E402
    GPU_STAGES, PRESETS, STAGE_MODULES, STAGE_ORDER, Context, Paths, StageUnavailable,
    configure_logging, log, resolve_stages, write_manifest,
)


def build_parser() -> Any:
    parser = argparse.ArgumentParser(
        description="段階区切りの notebook パイプラインを手元でも Kaggle kernel 内でも実行する",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="stage 順: " + " -> ".join(STAGE_ORDER) + "\nプリセット: " +
               ", ".join(f"{name}={PRESETS[name]}" for name in sorted(PRESETS)),
    )
    parser.add_argument("--variant", default="baseline", help="src/pipeline/variants.yaml の名前")
    parser.add_argument("--stages", default="preflight", help="プリセット名 または stage のカンマ区切り")
    parser.add_argument("--list-variants", action="store_true", help="variant 一覧を表示して終了")
    parser.add_argument("--plan", action="store_true", help="解決した実行計画を表示だけする")
    parser.add_argument("--no-pip-fallback", dest="pip_fallback", action="store_false", default=None,
                        help="setup の PyPI 補完を止めると、環境の事実だけ見る診断に使える")
    parser.add_argument("--pip-fallback", dest="pip_fallback", action="store_true", default=None)
    parser.add_argument("--strict", action="store_true",
                        help="ライブラリ欠落で skip せず失敗にする（kernel 内は既定 True）")
    parser.add_argument("--no-graph-smoke", dest="graph_smoke", action="store_false", default=None)
    identity = parser.add_argument_group("識別 / task 選別")
    identity.add_argument("--run-name", default=None)
    identity.add_argument("--model", dest="model_name", default=None)
    identity.add_argument("--limit", dest="task_limit", type=int, default=None)
    identity.add_argument("--task-id", dest="task_ids", nargs="*", default=None)
    identity.add_argument("--shard-index", type=int, default=None)
    identity.add_argument("--num-shards", type=int, default=None)
    identity.add_argument("--adapter-mode", choices=["off", "use"], default=None)
    identity.add_argument("--sandbox", choices=["docker", "subprocess"], default=None)

    knobs = parser.add_argument_group("budget / sampling の上書き（variant に勝つ）")
    knobs.add_argument("--max-tool-calls", dest="max_tool_calls", type=int, default=None)
    knobs.add_argument("--max-time-minutes", dest="max_time_minutes", type=float, default=None)
    knobs.add_argument("--max-turns", dest="max_turns", type=int, default=None)
    knobs.add_argument("--timeout-seconds", dest="timeout_seconds", type=int, default=None)
    knobs.add_argument("--thinking-budget", dest="thinking_budget", type=int, default=None)
    knobs.add_argument("--max-output-tokens", dest="max_output_tokens", type=int, default=None)
    knobs.add_argument("--temperature", type=float, default=None)
    knobs.add_argument("--no-smoke-test", dest="smoke_test", action="store_false", default=None)
    knobs.add_argument("--vllm-version", dest="vllm_version", default=None,
                       help="測定だけ vLLM をこの version に载せ替える（wheelhouse の pin から外れる。"
                            "model は 4bit QAT のまま。deviation として manifest に残る）")

    paths = parser.add_argument_group("路径")
    paths.add_argument("--data-dir", default=None, help="competition data dir（Kaggle mount は保つ）")
    paths.add_argument("--working-dir", default=None, help="既定: /kaggle/working または <repo>/work")
    paths.add_argument("--submission-dir", default=None)
    paths.add_argument("--results-dir", default=None, help="post ステージが読む results dir を明示")
    paths.add_argument("--patch-limit", type=int, default=None,
                       help="patch ステージが解析する patch 数（既定 20）")
    paths.add_argument("--show-failures", type=int, default=None,
                       help="analyze が表示する unresolved task id 数（既定 8）")
    paths.add_argument("--verbose", action="store_true", default=None)
    return parser


def overrides(args: Any) -> dict[str, Any]:
    """CLI 引数を config の形に畳む。None は「上書きしない」なので落とす。"""
    flat = {
        "run_name": args.run_name, "model_name": args.model_name, "task_limit": args.task_limit,
        "task_ids": args.task_ids, "shard_index": args.shard_index, "num_shards": args.num_shards,
        "adapter_mode": args.adapter_mode, "sandbox": args.sandbox, "graph_smoke": args.graph_smoke,
        "smoke_test": args.smoke_test, "verbose": args.verbose, "pip_fallback": args.pip_fallback,
        "vllm_version": args.vllm_version,
        "submission_dir": args.submission_dir, "results_dir": args.results_dir,
        "patch_limit": args.patch_limit, "show_failures": args.show_failures,
    }
    budget = {
        "max_tool_calls": args.max_tool_calls, "max_time_minutes": args.max_time_minutes,
        "max_turns": args.max_turns, "timeout_seconds": args.timeout_seconds,
    }
    sampling = {
        "thinking_budget": args.thinking_budget, "max_output_tokens": args.max_output_tokens,
        "temperature": args.temperature,
    }
    out = {k: v for k, v in flat.items() if v is not None}
    budget = {k: v for k, v in budget.items() if v is not None}
    sampling = {k: v for k, v in sampling.items() if v is not None}
    if budget:
        out["budget"] = budget
    if sampling:
        out["sampling"] = sampling
    return out


def list_variants() -> int:
    """variants.yaml の各 variant を、実際に config に展開して表示する（推測でなく load 結果）。"""
    import yaml

    if not config.variants_file().is_file():
        print(f"no variants file at {config.variants_file()}")
        return 2
    doc = yaml.safe_load(config.variants_file().read_text(encoding="utf-8")) or {}
    print(f"# {config.variants_file()}")
    for name in (doc.get("variants") or {}):
        cfg = config.load(name)
        print(f"\n{name}: run_name={cfg.run_name} adapter_mode={cfg.adapter_mode} "
              f"model={cfg.model_name} task_limit={cfg.task_limit} "
              f"shard={cfg.shard_index}/{cfg.num_shards}")
        print(f"  budget   {cfg.budget}")
        print(f"  sampling {cfg.sampling}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_variants:
        return list_variants()

    cfg = config.load(args.variant, overrides(args))
    configure_logging(cfg.verbose)
    paths = Paths.detect(args.data_dir, args.working_dir)
    stages = resolve_stages(args.stages)
    # kernel 内で skip が出るのは設定ミス（dataset 未添付など）の意味なので、そこは失敗扱いにする
    strict = bool(args.strict or paths.kaggle)

    log("pipeline", f"variant={cfg.variant} stages={' -> '.join(stages)} strict={strict}")
    cfg.describe()
    paths.describe()
    if args.plan:
        for stage in stages:
            kind = "GPU/ライブラリ要" if stage in GPU_STAGES else "手元で可"
            print(f"  plan: {stage:<8} ({kind})")
        return 0

    ctx = Context(cfg=cfg, paths=paths)
    ran: list[str] = []
    skipped: list[tuple[str, str]] = []
    timings: dict[str, float] = {}

    for stage in stages:
        module = importlib.import_module(STAGE_MODULES[stage])
        started = time.perf_counter()
        try:
            module.run(ctx)
        except StageUnavailable as exc:
            if strict:
                log(stage, f"[ERROR] {exc}")
                return 1
            skipped.append((stage, str(exc).split(".")[0]))
            log(stage, f"[skip] {exc}")
            continue
        except SystemExit:
            timings[stage] = time.perf_counter() - started
            raise
        except Exception as exc:  # noqa: BLE001
            timings[stage] = time.perf_counter() - started
            log(stage, f"[ERROR] {type(exc).__name__}: {str(exc)[:500]}")
            log(stage, f"[ERROR] stage '{stage}' が {timings[stage]:.1f}s で失敗。"
                       f"ここまで完了: {ran}")
            write_manifest(ctx, ran)
            return 1
        timings[stage] = time.perf_counter() - started
        ran.append(stage)
        log(stage, f"done in {timings[stage]:.1f}s")

    write_manifest(ctx, ran)
    log("pipeline", "ran    : " + (", ".join(f"{s}({timings[s]:.1f}s)" for s in ran) or "nothing"))
    if skipped:
        log("pipeline", "skipped: " + ", ".join(stage for stage, _ in skipped))
        log("pipeline", "        GPU stage は Kaggle kernel が必要（wheelhouse が swegemma/"
                        "adk_submission/vllm を入れる）。手元では --stages preflight / post。")
    log("pipeline", f"variant={cfg.variant} run={cfg.run_name} "
                    f"zip={ctx.zip_path or '-'} results={ctx.notes.get('results_dir') or '-'}")
    if ctx.notes.get("n_resolved") is not None:
        log("pipeline", f"resolved={ctx.notes['n_resolved']}/{ctx.notes.get('n_evaluated')} "
                        f"patches={ctx.notes.get('n_patches')}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("[pipeline] interrupted", flush=True)
        sys.exit(130)
