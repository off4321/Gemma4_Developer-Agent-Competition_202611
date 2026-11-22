"""
ステージ `eval` = ノートブック Cell 5。task ごとに Phase 1（agent loop）と Phase 2（検証）を回す。

budget は staged された eval_config.yaml から **読み戻す**。スコア側が
load_submission_eval_config で読むのと同じ経路なので、ここで測るのは提出物が実際に要求した設定である。
値そのものは variant（-> stage `agent`）が決める。

逐次実行にしてある: `swegemma eval --concurrency N` は CLI の機能で、2x T4 の kernel 内で
agent loop を 2 本走らせると、1 本の vLLM と 30 GB RAM を奪い合って多くの場合お互いを遅くする。
task 1 件の例外は catch して残りを続ける: quota（週 30 h）の中で 1 run を無駄に捨てないため。

失敗の読み分け（詳細 doc/harness_watchpoints.md）:
  * NO_PATCH        agent が edit_file / submit_patch まで辿り着かなかった（Phase 1 の敗北）
  * apply 失敗      patch は出たが git apply が通らない（Phase 2 の敗北）
  * 保護ファイル污染 tests/ や conftest.py 等を触って strip / reset された
  * テスト失敗      上 3 つが 0 なのに resolved=False -> 本当に解けていない
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from pipeline.common import log, require_module

BUDGET_DEFAULTS = {
    "timeout_seconds": 300,
    "verification_timeout_seconds": 300,
    "max_tool_calls": 100,
    "max_time_minutes": 60.0,
    "max_turns": None,
}


def read_staged_budgets(agent_dir: Path) -> dict[str, Any]:
    """staged された eval_config.yaml から budget を読み戻す。

    スコア側が `load_submission_eval_config` で読むのと同じ経路にする。
    ルート直下にあって `evaluation:` key でくるまれたものだけが効く。
    """
    import yaml

    for name in ("eval_config.yaml", "eval_config.yml"):
        path = agent_dir / name
        if not path.is_file():
            continue
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        section = raw.get("evaluation", raw)
        budgets = dict(BUDGET_DEFAULTS)
        for key in budgets:
            if key in section:
                budgets[key] = section[key]
        turns = section.get("max_turns", section.get("max_llm_calls"))
        budgets["max_turns"] = int(turns) if turns is not None else None
        budgets["source"] = str(path.relative_to(agent_dir))
        return budgets
    log("eval", "[warn] staged agent dir に eval_config.yaml が無い -> harness default が適用される")
    return dict(BUDGET_DEFAULTS)


def run_coro(factory: Any) -> Any:
    """swegemma の async coroutine を、notebook でも CLI でも動く形で走らせる。

    Kaggle kernel は ipykernel なので **main thread に event loop が既に走っている**。そこへ
    `asyncio.run()` を呼ぶと `RuntimeError: asyncio.run() cannot be called from a running event loop`
    で eval が 0.1 s で全 task 捨てる（実測でそうなった）。CLI 実行では loop が無いので素直に run する。
    loop が有るときだけ自分专用 loop の thread に逃がす（nest_asyncio を依存に足さない）。
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())
    import concurrent.futures

    log("eval", "running event loop が検えた -> task を dedicated thread の own loop で走らせる")
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, factory()).result()


def run(ctx: Any) -> None:
    require_module("swegemma", why="swegemma.evaluate.Evaluator が必要")
    require_module("google.adk", why="ADK の context cache / events compaction config が必要")
    from google.adk.agents.context_cache_config import ContextCacheConfig
    from google.adk.apps._configs import EventsCompactionConfig
    from swegemma.config import EvalConfig, build_submission_limits
    from swegemma.evaluate import Evaluator

    if ctx.models is None:
        raise SystemExit("[eval] stage 'server' を 'eval' より先に実行すること（model registry が無い）")

    agent_dir = ctx.require_agent_dir()
    budgets = read_staged_budgets(agent_dir)
    log("eval", f"budgets（{budgets.pop('source', 'defaults')} から）: {budgets}")
    if int(budgets["max_time_minutes"]) > 60:
        raise SystemExit("[eval] max_time_minutes > 60 は harness が拒否する")

    limits, generation_constraints = build_submission_limits()
    ctx.limits, ctx.generation_constraints = limits, generation_constraints
    results_dir = ctx.paths.results_dir(ctx.cfg.run_name)
    log("eval", f"results dir: {results_dir}")

    config = EvalConfig(
        tasks_path=ctx.paths.tasks_path,
        snapshots_dir=ctx.paths.snapshots_dir,
        results_dir=results_dir,
        submission_dir=agent_dir,
        models=ctx.models,
        sandbox=ctx.cfg.sandbox,
        timeout_seconds=int(budgets["timeout_seconds"]),
        verification_timeout_seconds=int(budgets["verification_timeout_seconds"]),
        max_time_minutes=float(budgets["max_time_minutes"]),
        max_tool_calls=int(budgets["max_tool_calls"]),
        max_turns=budgets["max_turns"],
        limits=limits,
        generation_constraints=generation_constraints,
        adapter_manifest=ctx.adapters,
        # HARNESS_README 7.2: スコア側が入れるの同じ compaction / prefix caching を入れる
        context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
        events_compaction_config=EventsCompactionConfig(
            compaction_interval=15, overlap_size=2, token_threshold=14336, event_retention_size=5),
        graph_dir=str(ctx.paths.graphs_dir),
        embeddings_dir=str(ctx.paths.embeddings_dir),
        wheels_dir=str(ctx.paths.wheels_dir),
        # 失敗原因が agent loop の中で消えると、次 run まで推測を繰り返すことになる（ quota を食う）。
        # `make pipeline-run EXTRA_ARGS=--verbose` で立てられる。通常 run では静かのまま。
        verbose=bool(getattr(ctx.cfg, "verbose", False)),
    )

    evaluator = Evaluator(config)
    tasks = ctx.require_tasks()
    registry = getattr(ctx.models, "models", None) or getattr(ctx.models, "_models", None) or {}
    if isinstance(registry, dict) and registry:
        first = next(iter(registry.items()))
        log("eval", f"model registry: {len(registry)} alias, e.g. {first[0]} -> {getattr(first[1], 'model', first[1])}")
    predictions: list[dict[str, Any]] = []
    failures: list[str] = []

    for index, task in enumerate(tasks, start=1):
        task_id = getattr(task, "instance_id", f"task_{index}")
        log("eval", f"[{index}/{len(tasks)}] {task_id} ({getattr(task, 'repo', '?')})")
        try:
            result = run_coro(lambda: evaluator.evaluate_task(task=task, task_index=index,
                                                              total_tasks=len(tasks)))
        except Exception as exc:  # noqa: BLE001
            failures.append(task_id)
            log("eval", f"    evaluate_task 失敗 {type(exc).__name__}: {str(exc)[:300]}")
            predictions.append({"id": task_id, "prediction": "NO_PATCH", "resolved": False})
            continue
        patch = (getattr(result, "agent_patch", "") or "").strip()
        resolved = bool(getattr(result, "resolved", False))
        predictions.append({"id": task_id, "prediction": patch if patch else "NO_PATCH",
                            "resolved": resolved})
        tool_calls = getattr(result, "tool_calls", None)
        log("eval", f"    resolved={resolved} exit={getattr(result, 'test_exit_code', None)} "
                    f"patch_chars={len(patch)} tool_calls={tool_calls} "
                    f"duration={getattr(result, 'duration_seconds', 0.0):.1f}s")
        if not patch and not tool_calls:
            # tool call が 1 本も出ていない = agent loop が LLM 応答を得られていない可能性が高い。
            # trace の total_completion_tokens=0 なら生成が一度も走っていない（server / litellm 側）。
            log("eval", f"    [diag] {task_id}: tool call 0 本。trace を見る: "
                        f"{results_dir}/traces/trace_{task_id}.json の final_metrics.total_completion_tokens、"
                        f" logs/{task_id}.log。server 側は vllm_server.log")

    patch_count = sum(1 for p in predictions if p["prediction"] != "NO_PATCH")
    resolved_count = sum(1 for p in predictions if p["resolved"])
    log("eval", f"tasks={len(predictions)} patches={patch_count} resolved={resolved_count} "
                f"crashed={len(failures)}")
    if failures:
        log("eval", f"[warn] 例外で落ちた task id: {failures[:12]}")

    _write_predictions(ctx, predictions)
    # agent が 0 tool call で終わった run は、ほぼ server 側の生成失敗である。log を必ず引き出す。
    from pipeline.stage_server import snapshot_server_log
    snapshot_server_log(ctx, "after_eval")
    ctx.notes.update({"n_evaluated": len(predictions), "n_patches": patch_count,
                      "n_resolved": resolved_count, "crashed_ids": failures[:50],
                      "results_dir": str(results_dir)})
    log("eval", f"next: make out-eval RUN={ctx.cfg.run_name} && make analyze RUN={ctx.cfg.run_name}")


def _write_predictions(ctx: Any, predictions: list[dict[str, Any]]) -> None:
    """submission.parquet はスコア側の形（id, prediction=patch）。無いときは JSON に逃がす。"""
    out_dir = ctx.paths.working_dir
    try:
        import pandas as pd

        frame = pd.DataFrame([{"id": p["id"], "prediction": p["prediction"]} for p in predictions])
        path = out_dir / "submission.parquet"
        frame.to_parquet(path)
        log("eval", f"wrote {path} ({len(frame)} rows)")
    except Exception as exc:  # noqa: BLE001
        log("eval", f"[warn] parquet 書き込みを skip（{type(exc).__name__}: {str(exc)[:120]}）")
    import json

    path = out_dir / f"predictions_{ctx.cfg.run_name}.json"
    path.write_text(json.dumps(predictions, ensure_ascii=False), encoding="utf-8")
    log("eval", f"wrote {path}")
