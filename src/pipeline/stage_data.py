"""
ステージ `data` = ノートブック Cell 2 の前半（task 読み込み）。

Kaggle では harness 本体の loader（swegemma.models.load_tasks）を使い、Evaluator が期待する
本物の Task dataclass を得る。手元（wheelhouse 無し）では同じ属性を持つ軽量な代替オブジェクトに
読み替える。これで preflight / patch / analyze ステージが GPU 無しで動く。

task 選別は `swegemma eval` CLI の意味に揃えた:
  --task-ids  -> 指定 instance_id だけ残す（デバッグ用）
  shard       -> index % num_shards == shard_index の task だけ残す（quota 分割用）
  task_limit  -> その残りの先頭 N 個（0 = 全 task）

shard が limit より先なのは、CLI と同じ順序だから。逆にした瞬間「同じ task 群を 2 回」実行して
quota を無駄に消費するので、ここは崩さない。
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pipeline.common import log, require_module

TASK_FIELDS = ("instance_id", "repo", "base_commit", "patch", "test_patch",
               "problem_statement", "hints_text", "created_at")


def load_jsonl_tasks(path: Path) -> list[Any]:
    """tasks.jsonl を手元用の轻量オブジェクトに読む（swegemma が無いとき用のフォールダ）。"""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        rows.append(SimpleNamespace(**{k: data.get(k) for k in TASK_FIELDS if k in data}))
    return rows


def select(tasks: list[Any], cfg: Any) -> list[Any]:
    """CLI と同じ順序で task を絞る: ids filter -> shard -> limit。"""
    total = len(tasks)
    if cfg.task_ids:
        wanted = set(cfg.task_ids)
        tasks = [t for t in tasks if getattr(t, "instance_id", None) in wanted]
        missing = sorted(wanted - {getattr(t, "instance_id", None) for t in tasks})
        if missing:
            log("data", f"[warn] 公開 task に見えない task id: {missing}")
    if cfg.num_shards and cfg.num_shards > 1:
        tasks = [t for index, t in enumerate(tasks) if index % cfg.num_shards == cfg.shard_index]
        log("data", f"shard {cfg.shard_index}/{cfg.num_shards} -> {len(tasks)} task(s)")
    if cfg.task_limit and cfg.task_limit > 0:
        tasks = tasks[: cfg.task_limit]
    log("data", f"公開 {total} task のうち {len(tasks)} 個を選んだ")
    return tasks


def _swegemma_available() -> bool:
    """swegemma が import できるか。スコア側と同じ task object で読むべきかの判断に使う。
    Kaggle mount の有無ではなく **module が有るか**で選ぶ（Colab でも swegemma は入る）。"""
    try:
        import importlib
        importlib.import_module("swegemma.models")
        return True
    except Exception:
        return False


def run(ctx: Any) -> None:
    tasks_path = ctx.paths.tasks_path
    if not tasks_path.is_file():
        raise SystemExit(f"[data] tasks ファイルが無い: {tasks_path}\n"
                         f"       kaggle competitions download -c gemma-4-developer-agent "
                         f"-f tasks.jsonl -p data")

    if ctx.paths.kaggle or _swegemma_available():
        require_module("swegemma", why="swegemma.models.load_tasks が必要")
        from swegemma.models import load_tasks

        tasks = list(load_tasks(tasks_path))
        log("data", f"swegemma.models.load_tasks で {len(tasks)} task 読んだ（スコア側と同じ Task object）")
    else:
        tasks = load_jsonl_tasks(tasks_path)
        log("data", "[warn] 手元の jsonl reader (SimpleNamespace) を使った。swegemma の verification は "
                    "`task.test_files` を直接見るので Phase 2 が "
                    "AttributeError: 'SimpleNamespace' object has no attribute 'test_files' で落ちる "
                    "（実測 2026-11-21）。measure には swegemma が import できる環境を使うこと")

    ctx.tasks = select(tasks, ctx.cfg)
    if not ctx.tasks:
        raise SystemExit("[data] task 選別で 0 件。task_limit / task_ids / shard を確認すること")

    repos = Counter(getattr(t, "repo", "?") for t in ctx.tasks)
    log("data", "repos: " + ", ".join(f"{repo}={n}" for repo, n in repos.most_common()))
    for task in ctx.tasks[:5]:
        log("data", f"  {getattr(task, 'instance_id', '?')} "
                    f"({getattr(task, 'repo', '?')} @ {str(getattr(task, 'base_commit', ''))[:8]})")
    ctx.notes["n_published_tasks"] = len(tasks)
