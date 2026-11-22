"""
全ステージが共有する実行基盤。

ノートブックが暗黙にやっていた 3 つを、ここで明示する:
  1. 重い import（vllm / litellm / transformers）の前にやるべき環境変数設定とログ静默化、
  2. Kaggle（/kaggle/input/...、/kaggle/working）でも手元（<repo>/data、<repo>/work）でも
     同じように通る路径解決、
  3. run_pipeline.py が使うステージ一覧（STAGE_ORDER / PRESETS）。

用語の補足（初学者向け）:
  * 「GPU stage」= swegemma / adk_submission / vLLM が必要なので Kaggle kernel でしか走らない段階。
  * StageUnavailable = そのライブラリが入っていないときに出す「この機械ではできない」という合図。
    手元では skip、Kaggle 内では即失敗（strict）に使う。
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------- ステージ一覧
STAGE_ORDER = ["setup", "data", "agent", "graph", "server", "eval", "package", "patch", "analyze"]

STAGE_MODULES = {
    "setup": "pipeline.stage_setup",
    "data": "pipeline.stage_data",
    "agent": "pipeline.stage_agent",
    "graph": "pipeline.stage_graph",
    "server": "pipeline.stage_server",
    "eval": "pipeline.stage_eval",
    "package": "pipeline.stage_package",
    "patch": "pipeline.stage_patch",
    "analyze": "pipeline.stage_analyze",
}

# swegemma / adk_submission / vLLM （または GPU）が無いと走れないステージ
GPU_STAGES = {"graph", "server", "eval"}

PRESETS = {
    # swegemma / vLLM が無い機械（この箱は CUDA なし）でも意味のある組み合わせ
    "preflight": ["setup", "data", "agent"],
    "local": ["setup", "data", "agent", "patch", "analyze"],
    # ノートブックの手順そのもの。Kaggle kernel が走るのはこれ
    "gpu": ["setup", "data", "agent", "graph", "server", "eval", "package"],
    # 実行 -> artifacts 取得 -> 学ぶ
    "post": ["patch", "analyze"],
    "all": STAGE_ORDER,
}


def resolve_stages(spec: str) -> list[str]:
    """`--stages` を展開する（プリセット名 / カンマ区切りのステージ名 / all）。"""
    spec = (spec or "preflight").strip().lower()
    if spec in PRESETS:
        return list(PRESETS[spec])
    wanted = {s.strip() for s in spec.split(",") if s.strip()}
    unknown = sorted(wanted - set(STAGE_ORDER))
    if unknown:
        raise SystemExit(f"[pipeline] unknown stage(s) {unknown}; known: {STAGE_ORDER} "
                         f"or presets {sorted(PRESETS)}")
    return [s for s in STAGE_ORDER if s in wanted]


# ---------------------------------------------------------------- sys.path 用意
def ensure_src_on_path() -> Path:
    """`src/` を sys.path に足す。これで `package.*` / `evaluate.*` がトップレベル名で import できる。"""
    src_dir = Path(__file__).resolve().parents[1]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return src_dir


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def is_kaggle() -> bool:
    return Path("/kaggle/working").is_dir()


class StageUnavailable(RuntimeError):
    """そのステージが必要とするライブラリ / GPU が、この機械に無いことを表す合図。"""


def require_module(name: str, *, why: str) -> None:
    if not has_module(name):
        raise StageUnavailable(
            f"{name} を import できない（{why}）。これは competition の wheelhouse 由来なので、"
            f"このステージは Kaggle kernel 内で実行すること（make pipeline-eval）。"
            f"手元では --stages preflight。詳細 doc/kaggle_cli.md"
        )


# ---------------------------------------------------------------- ロギング
QUIET_LOGGERS = (
    "LiteLLM", "LiteLLM Router", "LiteLLM Proxy", "litellm", "httpx", "httpcore",
    "openai", "urllib3", "opentelemetry", "opentelemetry.context",
)


def configure_logging(verbose: bool = False) -> None:
    """外部ライブラリの冗長ログを黙らせる（vLLM / LiteLLM は INFO 以上を出しがち）。"""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    for name in QUIET_LOGGERS:
        logger = logging.getLogger(name)
        logger.setLevel(logging.WARNING)
        if "litellm" in name.lower():
            logger.propagate = False
            logger.handlers.clear()


def set_runtime_env() -> None:
    """ノートブック Cell 1 のオフライン推論 / テレメトリ switch。

    **vllm / litellm / transformers を import する前に** 呼ぶこと。これらの多くは
    import の時点で設定を見るので、後から立てても効かない。
    """
    defaults = {
        "TRANSFORMERS_NO_TF": "1",
        "TRANSFORMERS_NO_FLAX": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "VLLM_NO_USAGE_STATS": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_ENGINE_READY_TIMEOUT_S": "1800",
        "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "1",
        "NCCL_DEBUG": "WARN",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "LITELLM_TELEMETRY": "False",
        "LITELLM_LOG": "WARNING",
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "EMPTY"),
        "LOCAL_API_KEY": os.environ.get("LOCAL_API_KEY", "EMPTY"),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "OTEL_SDK_DISABLED": "true",
        "OTEL_PYTHON_DISABLED": "true",
        # Kaggle は internet を有効にすると proxy env を立てる。httpx/openai はそれを素直に使うので、
        # **127.0.0.1 の vLLM へのリクエストが proxy 経由になって connection error になる**事故がある
        # （実測: litellm.InternalServerError ... Connection error）。localhost は直结にする。
        "NO_PROXY": "127.0.0.1,localhost,0.0.0.0,::1",
        "no_proxy": "127.0.0.1,localhost,0.0.0.0,::1",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    # NO_PROXY は Kaggle が先に立てていることがあるので setdefault だと効かない。localhost が必ず入るようにする。
    for key in ("NO_PROXY", "no_proxy"):
        current = os.environ.get(key, "")
        parts = [p.strip() for p in current.split(",") if p.strip()]
        for host in ("127.0.0.1", "localhost"):
            if host not in parts:
                parts.append(host)
        os.environ[key] = ",".join(parts)


# ---------------------------------------------------------------- 路径
@dataclass
class Paths:
    """ステージが読み書きするものを、Kaggle か手元かを 1 回だけ解決して保持する。"""

    root: Path
    data_dir: Path
    working_dir: Path
    wheelhouse: Path
    models_root: Path
    kaggle: bool

    @classmethod
    def detect(cls, data_dir: str | None = None, working_dir: str | None = None) -> "Paths":
        kaggle = is_kaggle()
        root = repo_root()
        if kaggle:
            default_data = Path("/kaggle/input/competitions/gemma-4-developer-agent")
            default_working = Path("/kaggle/working")
            wheelhouse = Path("/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse")
            models_root = Path("/kaggle/input/models")
        else:
            default_data = root / "data"
            default_working = root / "work"
            wheelhouse = root / "data" / "wheelhouse"
            models_root = root / "models"
        resolved_data = Path(data_dir) if data_dir else default_data
        if kaggle and data_dir and not resolved_data.is_dir():
            # a local --data-dir is meaningless inside a kernel; keep the mounted competition data
            resolved_data = default_data
        resolved_working = Path(working_dir) if working_dir else default_working
        resolved_working.mkdir(parents=True, exist_ok=True)
        return cls(root=root, data_dir=resolved_data, working_dir=resolved_working,
                   wheelhouse=wheelhouse, models_root=models_root, kaggle=kaggle)

    # competition 側の入力（kaggle competitions download か kernel 添付で入る）
    @property
    def tasks_path(self) -> Path:
        return self.data_dir / "tasks.jsonl"

    @property
    def graphs_dir(self) -> Path:
        return self.data_dir / "graphs"

    @property
    def embeddings_dir(self) -> Path:
        return self.data_dir / "embeddings"

    @property
    def wheels_dir(self) -> Path:
        return self.data_dir / "wheels"

    @property
    def snapshots_dir(self) -> Path:
        return self.data_dir / "snapshots"

    @property
    def sample_submission(self) -> Path:
        return self.data_dir / "sample_submission"

    def model_mount(self, model_name: str) -> Path:
        # Kaggle Models の mount 場所: .../models/google/gemma-4/other/<name>/<version>
        return self.models_root / "google" / "gemma-4" / "other" / model_name / "2"

    def results_dir(self, run_name: str) -> Path:
        return self.working_dir / "results" / run_name

    def describe(self) -> None:
        log("paths", f"kaggle={self.kaggle} data={self.data_dir} working={self.working_dir}")
        for label, path in (("tasks", self.tasks_path), ("snapshots", self.snapshots_dir),
                            ("graphs", self.graphs_dir), ("embeddings", self.embeddings_dir)):
            log("paths", f"{label}: {'present' if path.exists() else 'MISSING'} ({path})")


# ---------------------------------------------------------------- 共有 context
@dataclass
class Context:
    """ステージ順に渡していく可変 state（前ステージの結果が後ステージの入力）。"""

    cfg: Any
    paths: Paths
    tasks: list[Any] = field(default_factory=list)
    agent_dir: Path | None = None
    server: Any = None
    models: Any = None
    adapters: Any = None
    adapter_names: list[str] = field(default_factory=list)
    declared_model: str | None = None
    limits: Any = None
    generation_constraints: Any = None
    zip_path: Path | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def root(self) -> Path:
        """repo 直下。`ctx.paths.root` の短縮名（ステージ側はここを使う）。"""
        return self.paths.root

    def require_agent_dir(self) -> Path:
        if self.agent_dir is None:
            raise SystemExit("[pipeline] stage 'agent' did not run; add it to --stages")
        return self.agent_dir

    def require_tasks(self) -> list[Any]:
        if not self.tasks:
            raise SystemExit("[pipeline] stage 'data' did not produce tasks; add it to --stages")
        return self.tasks


def log(stage: str, message: str) -> None:
    print(f"[{stage}] {message}", flush=True)


def find_results_dir(ctx: Any) -> Path | None:
    """post ステージ用の results dir を探す。

    優先順: 明示の --results-dir、`make out-eval` が落とす <repo>/results/<run>、
    最後に手元 pipeline 実行が <working>/results/<run> に作ったもの。
    """
    candidates: list[Path] = []
    explicit = (ctx.cfg.raw or {}).get("results_dir")
    if explicit:
        candidates.append(Path(explicit))
    candidates += [
        ctx.root / "results" / ctx.cfg.run_name,
        ctx.paths.results_dir(ctx.cfg.run_name),
        ctx.root / "results",
    ]
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        if any((candidate / name).exists() for name in ("task_results.jsonl", "summary.json", "patches")):
            return candidate
        if list(candidate.glob("predictions_*.json")) or list(candidate.glob("*.parquet")):
            return candidate        # 自分の pipeline / Kaggle kernel 出力の dir
    return None


def write_manifest(ctx: Context, stages: list[str]) -> Path:
    """この run が「何だったか」を記録する。後で比較表に並べるための証跡（doc/tasks.md T5）。"""
    import json

    manifest = {
        "variant": ctx.cfg.variant,
        "run_name": ctx.cfg.run_name,
        "stages": stages,
        "model": ctx.declared_model or ctx.cfg.model_name,
        "adapters": ctx.adapter_names,
        "adapter_mode": ctx.cfg.adapter_mode,
        "budget": ctx.cfg.budget,
        "sampling": ctx.cfg.sampling,
        "task_limit": ctx.cfg.task_limit,
        "task_ids": ctx.cfg.task_ids,
        "shard": [ctx.cfg.shard_index, ctx.cfg.num_shards],
        "n_tasks": len(ctx.tasks),
        "task_instances": [getattr(t, "instance_id", str(t)) for t in ctx.tasks][:200],
        "kaggle": ctx.paths.kaggle,
        "notes": ctx.notes,
    }
    path = ctx.paths.working_dir / f"pipeline_{ctx.cfg.run_name}.json"
    path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    log("manifest", f"wrote {path}")
    return path
