#!/usr/bin/env python3
"""Kaggle で **セル単位で手で回す**測定ノートブックを作る（Getting Started と同じ資材構成）。

`make_kernel_notebook.py` が作る「1 cell で pipeline を走らせるだけ」の kernel とちがい、
このノートブックは人間が 1 個ずつ実行する:

| cell | 内容 | 公式 Getting Started の cell 対応 |
|---|---|---|
| 1 | 本番資材の mount 検査（competition / wheelhouse / model / code / submission）+ HW | cell 1 |
| 2 | variant 選択 -> RunConfig + Context（以降の cell が共有する state） | cell 1 |
| 3 | setup: wheelhouse pin 適用・torch pin 修理 | cell 1 |
| 4 | data（harness の loader）+ agent（staging・契約検査） | cell 2 |
| 5 | graph + embeddings（シンボル graph / embedding retrieval） | cell 3 |
| 6 | server: adk_submission.VllmServer を起こして 1 文生成 | cell 4 |
| 7 | eval: Phase 1 agent loop + Phase 2 verification | cell 5 |
| 8 | package + validate（提出 zip） | cell 6 |
| 9 | patch 解析 + analyze（failure taxonomy） | 追加 |
| 10 | manifest / deviation 報告 / 提出コマンド | 追加 |

`ctx`（Context）は kernel に保持されるので、**cell 6 で起こした server を cell 7 がそのまま使う**。
コード本体は dataset <you>/gemma4-dev-agent-code 側の `src/pipeline/` にある（ノートブックは薄い）。

    python3 src/pipeline/make_measure_notebook.py
    kaggle kernels push -p kernels/measure --no-run     # あなたの Kaggle 画面で 1 個ずつ回す
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text.strip()))


def code(text: str) -> None:
    CELLS.append(("code", text.strip()))


# ---------------------------------------------------------------- 1. 資材とハードウェア
md("""## 使い方（この notebook を回す順番・いじる定数・詰まったときの切り分け）

### 1) 準備（手元で 1 回）

```bash
# 0) src/ と submission/ を dataset に更新（-r tar が必須: dir-mode だと folder が飛ぶ）
kaggle datasets version -p .kaggle/dataset-code -m "sync src/" -r tar -q \
  || kaggle datasets create -p .kaggle/dataset-code -r tar -q
kaggle datasets version -p .kaggle/dataset-submission -m "sync submission/" -r tar -q \
  || kaggle datasets create -p .kaggle/dataset-submission -r tar -q

# 1) variant を kernel に焼き込む（kernel は実行時に CLI 引数を渡せない）
python3 src/pipeline/set_kernel_variant.py kaggle_nb

# 2) notebook を生成して push（--no-run で queue に入れずに実行だけ自分で開く）
python3 src/pipeline/make_measure_notebook.py
kaggle kernels push -p kernels/measure --no-run
```

Kaggle 側: `<username>/gemma4-l4-measure` を開いて **GPU On** で 1 個ずつ Run。

### 2) 実行順と注意点

`cell 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9 -> 10`。**cell 6 で起動した server を cell 7 が使うので、
6 と 7 の間で kernel を restart / Restart Run and Clear Outputs しない**こと（`ctx.server` が kernel の記憶に居ます）。

### 3) いじる定数

| cell | 定数 | 既定 | 意味 |
|---|---|---|---|
| 2 | `VARIANT` | `kaggle_nb` | `src/pipeline/variants.yaml` の比較単位（task_limit 6 / budget 20 min / `vllm_version` 0.28.0） |
| 2 | `TASK_LIMIT` / `TASK_IDS` | variant に従う | 絞るほど quota を食う。まず 1-6 個で通す |
| 6 | gpu_mem / tp | 2x T4 なので tp=2, fp16, 0.85 | 公式は 4x L4 / tp=4 / bfloat16 / 0.90。`max_model_len=32768` は公式と同じ |

### 4) 物を取り出して台帳に載せる

```bash
kaggle kernels output <username>/gemma4-l4-measure -p results/kaggle_nb -o -q
python3 src/pipeline/failures_report.py --results results/kaggle_nb       # 出ていなければここで出す
make scoreboard NAME=kaggle_nb HARNESS=official-upgraded ENV=kaggle-t4 RESULTS=results/kaggle_nb
python3 src/pipeline/parity_check.py --a results/kaggle_nb --b results/colab12 --label T4測定 L4測定
```

不出来の読み方は `results/kaggle_nb/failures.md`（kernel 側で `analyze` を走らせていれば既に有ります）

### 5) 詰まったとき（実測で踏んだ場所）

| 症状 | 対処 |
|---|---|
| cell 1 で mount が無い / `MISS` が出る | kernel の Inputs に competition / wheelhouse dataset / model instance / code+submission dataset が付いているか見る。dataset を更新したのに反映されない時は kernel を開き直す |
| `out of resource: shared memory, Required: 98304, Hardware limit: 65536` | T4 (sm75) で wheelhouse pin の vLLM 0.19.1 が Gemma 4 attention を出せない為。**`kaggle_nb` は `vllm_version: 0.28.0` を測定専用で使う**（公式の pin とは違うので絶対値は信用しない）。L4 と同じ条件で測りたいなら Colab 版（`kernels/colab/`）を使う |
| `asyncio.run() cannot be called from a running event loop` | 既に pipeline 側で専用 thread bridge を使っています（`src/pipeline/stage_eval.py`）。kernel を自分で書き換えた場合は bridge を保つ |
| smoke completion が 404 | `VllmServer.base_url` は既に `/v1` を含む。二重に足さない |
| GPU quota が無い | `kaggle kernels status` を繰り返さず `kaggle kernels logs <username>/gemma4-l4-measure --follow` で見る。CPU probe kernel でも quota を食うので `make probe` は必要なときだけ |

### 6) 使い方の筋

この kernel は **score 提出ではありません**。公式 harness と同じ資材で「同じ条件の近似」を測る場です。
絶対値（公式 Resolution Rate）は `cell 8` で作った `submission.zip` を `kaggle competitions submit` して取ります。
測った結果は `scoreboard.py` に載せて公式スコアと較正してください（`doc/verification.md`）。

""")

md("""# Gemma 4 Developer Agent — 測定ノートブック（Kaggle 準拠 / セル実行版）

公式 Getting Started と **同じ資材**で、`src/pipeline/` のステージを 1 個ずつ回すノートブックです。
ハーネス（`swegemma` / `adk_submission` / `adk_eval_core` / wheelhouse の pin）と
本番資材（`tasks.jsonl`・`graphs/`・`embeddings/`・`snapshots/`・model instance）を mount して使います。

- **計測対象**: Phase 1 で agent が patch を作り、Phase 2 で apply → `pytest` → resolved（Resolution Rate）
- **この kernel は score 提出ではない**。採点は `submission.zip` を `kaggle competitions submit` した後、
  公式側（4×L4, tp=4, bfloat16）で走ります。ここでは**同じ条件の近似**を測ります。
- **quota**: GPU 実行は週 30 h、1 run 2 h 目安。task 数は variant で絞る（`kaggle_nb` = 6 task × 20 min）。
""")

code('''import json
import os
import subprocess
import sys
from pathlib import Path

USER = os.environ.get("KAGGLE_USERNAME", "<username>")
CODE_SLUG = "gemma4-dev-agent-code"              # src/pipeline 本体（この notebook の中身の正体）
SUBMISSION_SLUG = "gemma4-dev-agent-submission"  # 提出物 submission/
MODEL_REF = "google/gemma-4/other/gemma-4-e4b-it-qat-w4a16-ct/2"


def mount(*parts: str) -> Path | None:
    """Kaggle の mount 路径は dataset/kernel 添付で形が変わるので、複数試す。"""
    tail = "/".join(parts)
    for base in (f"/kaggle/input/datasets/{USER}", "/kaggle/input/datasets", "/kaggle/input",
                 "/kaggle/input/competitions", "/kaggle/input/models"):
        cand = Path(f"{base}/{tail}")
        if cand.exists():
            return cand
    return None


def locate_pipeline_dir() -> Path:
    """mount された src/（pipeline/run_pipeline.py を含む）を探す。dataset は tar 展開で
    `src/` が外れて root 直下に pipeline/ として見えることがある（-r tar の挙動）。"""
    for base in (Path(f"/kaggle/input/datasets/{USER}/{CODE_SLUG}"), Path(f"/kaggle/input/{CODE_SLUG}"),
                 Path("/kaggle/working") / CODE_SLUG, Path.cwd()):
        for probe in (base / "src", base):
            if (probe / "pipeline" / "run_pipeline.py").is_file():
                return probe
    for root in (Path("/kaggle/input"), Path("/kaggle/working")):
        if root.is_dir():
            for hit in root.rglob("run_pipeline.py"):
                if hit.parent.name == "pipeline":
                    return hit.parent.parent
    raise SystemExit("pipeline/run_pipeline.py が見つからない。kernels/measure/kernel-metadata.json の "
                     f"dataset_sources に {USER}/{CODE_SLUG} を入れ、先に push すること")


CODE_DIR = locate_pipeline_dir()
sys.path.insert(0, str(CODE_DIR))
os.environ["PIPELINE_SUBMISSION_SLUG"] = SUBMISSION_SLUG   # stage 'agent' が提出物を探す

MATERIALS = {
    "competition (tasks/graphs/embeddings/snapshots)": mount("gemma-4-developer-agent")
        or Path("/kaggle/input/competitions/gemma-4-developer-agent"),
    "wheelhouse (harness wheels)": mount(f"{USER}/gemma-4-developer-agent-wheelhouse")
        or mount("metric/gemma-4-developer-agent-wheelhouse")
        or Path("/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse"),
    "model weights": Path(f"/kaggle/input/models/{MODEL_REF}"),
    "code (src/pipeline)": CODE_DIR,
    "submission (agent tree)": mount(f"{USER}/{SUBMISSION_SLUG}") or Path(f"/kaggle/input/{SUBMISSION_SLUG}"),
}
print(f"code package: {CODE_DIR}")
for name, path in MATERIALS.items():
    kind = "DIR " if path.is_dir() else ("FILE " if path.is_file() else "MISSING")
    print(f"  {kind} {name:46s} {path}")

print("\\n--- python / GPU ---")
print(subprocess.run("python3 -V", shell=True, capture_output=True, text=True).stdout.strip())
print(subprocess.run("nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader "
                     "2>&1 | head -4", shell=True, capture_output=True, text=True).stdout.strip())
print(subprocess.run('python3 -c "import torch;print(torch.__version__, torch.cuda.device_count())" 2>&1 | tail -1',
                     shell=True, capture_output=True, text=True).stdout.strip())
import importlib.metadata as md

WANT = {"vllm", "transformers", "tokenizers", "swegemma", "adk_submission", "adk_eval_core",
        "google-adk", "litellm", "torch", "torchvision", "torchaudio", "triton", "safetensors"}
pkgs = sorted(f"{d.metadata['Name']}={d.version}" for d in md.distributions()
              if (d.metadata.get("Name") or "").lower() in WANT)
print("base image pkgs:", ", ".join(pkgs) or "-")
''')

# ---------------------------------------------------------------- 2. variant / context
md("""## cell 2 — variant を選ぶ（測る条件はここで固定）

variant は `src/pipeline/variants.yaml`（dataset 経由で mount される同じファイル）にあります。

| variant | 内容 |
|---|---|
| `kaggle_nb` | **この notebook 向け**。6 task × 20 min、vLLM は測定用に 0.28.0（T4 で生成を通すため） |
| `baseline` | 12 task、公式 budget（45 min × 12 は 1 run を超えるので shard に割る） |
| `smoke` | 3 task の配線確認 |
| `colab12` / `colab` | wheelhouse pin のまま測る variant（**L4 / sm89 向け**、Colab で実測済み） |
| `t4fix` | 1 task の生成 probe |

`VLLM_VERSION` を手で入れると測定だけ载せ替えられます（manifest の `hardware` に deviation として残る）。
公式採点は wheelhouse の 0.19.1 + 4×L4 で走る、という差は消えません。
""")

code('''from pipeline import config as pcfg
from pipeline.common import Context, Paths, configure_logging, set_runtime_env

VARIANT = "kaggle_nb"
VLLM_VERSION = None          # None = variant の値。"0.28.0" / ""（= 公式 pin）に上書きできる
OVERRIDES = {} if VLLM_VERSION is None else {"vllm_version": VLLM_VERSION}

pcfg.usage()                 # variant 一覧（GPU を使わない）
configure_logging(True)
set_runtime_env()            # localhost の proxy を外す（server への HTTP が素通りする）

cfg = pcfg.load(VARIANT, OVERRIDES)
cfg.describe()
for warn in cfg.warnings():
    print("[warn]", warn)

paths = Paths.detect(data_dir=None, working_dir="/kaggle/working")
paths.describe()
ctx = Context(cfg=cfg, paths=paths)
print("run_name:", cfg.run_name, "| results:", ctx.paths.results_dir(cfg.run_name))
''')

# ---------------------------------------------------------------- 3. setup
md("""## cell 3 — setup（wheelhouse の pin を公式の組み合わせに戻す）

Kaggle base image は python 3.13 / torch 2.11+cu130 で、wheelhouse は cp312 寄りの pin です。
`setup` は **PEP 427 tag を読んで** incompatible を避け、torch stack を vLLM の pin に合わせ、
最後に wheelhouse の組み合わせを `--no-deps --force-reinstall` で戻します（約 5-10 分）。

T4 (sm75) で wheelhouse の vLLM 0.19.1 は Gemma 4 を生成できません（attention kernel が共有 memory 96 KB > 64 KB）。
`doc/environment_limits.md` §5。だから `kaggle_nb` は測定用に vLLM 0.28.0 を使います。
""")

code('''import pipeline.stage_setup as stage_setup

stage_setup.run(ctx)
print("\\n[setup] import 状態:")
for name in ("vllm", "swegemma", "adk_submission", "adk_eval_core", "google.adk", "litellm", "transformers"):
    try:
        mod = __import__(name, fromlist=["_"])
        print(f"  OK   {name:16s} {getattr(mod, '__version__', '')}")
    except Exception as exc:
        print(f"  FAIL {name:16s} {type(exc).__name__}: {exc}")
''')

# ---------------------------------------------------------------- 4. data + agent
md("""## cell 4 — data（harness の loader）+ agent（staging と契約検査）

- `data` は **スコア側と同じ `swegemma.models.load_tasks`** で読みます（`task.test_files` を harness が素で読むため、
  手元フォールダの `SimpleNamespace` を渡すと Phase 2 が `AttributeError` で落ちる — 実測で踏んだ箇所）
- `agent` は `submission/` を `work/agent_dir` に**コピーして** variant の budget / sampling / model を書き換える。
  `submission/` 自体は変わりません（測った agent と元の agent を突き合わせられる）
""")

code('''import pipeline.stage_agent as stage_agent
import pipeline.stage_data as stage_data

stage_data.run(ctx)
stage_agent.run(ctx)

print("\\nselected tasks:")
for task in ctx.tasks:
    print(f"  {task.instance_id:24s} {getattr(task, 'repo', '?')}")
print("agent dir:", ctx.agent_dir)
print("eval_config.yaml:", (ctx.agent_dir / "eval_config.yaml").read_text())
''')

# ---------------------------------------------------------------- 5. graph
md("""## cell 5 — graph（シンボルグラフ）と embeddings

本番資材の `graphs/`（repo のシンボル graph）と `embeddings/` を `SwegemmaContext` 経由で開き、
1 task で induced subgraph と embedding similar を出します。ここで 0 nodes なら mount か task 選別が壊れています。
""")

code('''import pipeline.stage_graph as stage_graph

stage_graph.run(ctx)
print("\\n[graph] ここが 0 nodes なら graphs/ mount か task ids を疑うこと")
''')

# ---------------------------------------------------------------- 6. server
md("""## cell 6 — vLLM server を起こす（`adk_submission.VllmServer`）

スコア側と同じ **server を別に process として起こす**やり方です。GPU 枚数で tp と dtype を決めます:
4 枚以上なら tp=4 / bfloat16 / gpu_mem 0.90（= 公式採点の形）、2 枚なら tp=2、sm75 は float16。

`ctx.server` に保持するので **cell 7 が同じ server を使います**（kernel を restart しないこと）。
""")

code('''import pipeline.stage_server as stage_server

stage_server.run(ctx)
server = ctx.server
print("\\nserver:", getattr(server, "base_url", None))
print("hardware:", json.dumps(ctx.notes.get("hardware", {}), ensure_ascii=False, indent=2))
print("attempts:", ctx.notes.get("server_attempts"))
''')

code('''# 同じ server に 1 文投げて、token が出るかを肉眼で確認する（Phase 1 の生死はここ）
from openai import OpenAI

models = OpenAI(base_url=server.base_url, api_key=getattr(server, "api_key", "EMPTY")).models.list()
served = models.data[0].id
print("served model id:", served)

resp = OpenAI(base_url=server.base_url, api_key=getattr(server, "api_key", "EMPTY")).chat.completions.create(
    model=served,
    messages=[{"role": "user", "content": "Reply with exactly: OK"}],
    max_tokens=64,
)
print("completion:", repr(resp.choices[0].message.content))
print("tokens:", resp.usage.completion_tokens, "/ prompt", resp.usage.prompt_tokens)
''')

# ---------------------------------------------------------------- 7. eval
md("""## cell 7 — Phase 1（agent loop）+ Phase 2（verification）

`swegemma.evaluate.Evaluator` を使い、staged `eval_config.yaml` の budget を**読み直して**適用します
（スコア側が読むものと 同じもの で測る）。Phase 2 は sandbox で patch apply → test patch → `pytest` + JUnit XML。

Kaggle notebook には docker daemon が無いので `variant.sandbox = subprocess`（既定）で動きます。
1 task 失敗しても次の task には進みます（quota を守る）。
""")

code('''import pipeline.stage_eval as stage_eval

stage_eval.run(ctx)
print("\\n[eval] predictions / results dir:")
for path in sorted(ctx.paths.working_dir.rglob("predictions_*.json")):
    print(" ", path)
print("results dir:", ctx.notes.get("results_dir"))
''')

code('''# 1 task ずつの数字（resolved / patch の長さ / tool 回数 / 時間）
import json

preds = sorted(ctx.paths.working_dir.rglob("predictions_*.json"))
if preds:
    rows = json.loads(preds[-1].read_text(encoding="utf-8"))
    for row in rows if isinstance(rows, list) else [rows]:
        print(f"  {row.get('instance_id'):24s} resolved={row.get('resolved')} "
              f"patch_chars={len(row.get('patch') or '')} duration={row.get('duration')}")
''')

# ---------------------------------------------------------------- 8. package
md("""## cell 8 — 提出 zip を作って契約検査する（**計測した agent** を zip する）

`package` は variant を適用済みの `agent_dir` を zip し、`Checker` + `adk_submission.validate_directory`
（使える model 名 / 3 GiB / 拡張子 / agent.yaml が root）を通します。ここを通った zip が提出物です。
""")

code('''import pipeline.stage_package as stage_package

stage_package.run(ctx)
zip_path = ctx.zip_path or (ctx.paths.working_dir / "submission.zip")
print("\\nzip:", zip_path, zip_path.stat().st_size if zip_path.exists() else "-")
''')

# ---------------------------------------------------------------- 9. analysis
md("""## cell 9 — patch 解析と失敗分類（スコアを上げるための解析）

`patch` は生成された patch を apply 検査 + protected path 違反 + 変更ファイル種別で分ける。
`analyze` は task 横断の failure taxonomy（NO_PATCH / apply 失敗 / protected 汚染 / test 失敗 / 時間切れ）を出します。

`analyze` は **task 別の不出来ファイル 2 個も書きます**（Colab 版 cell 7 と同じ `src/pipeline/failures_report.py`）:

```
/kaggle/working/results/<run>/failures.csv   instance_id / status / 理由 / test exit / patch size / tool calls / duration / error / pytest 末尾
/kaggle/working/results/<run>/failures.md    status 集計 + task 表 + 代表エラー + protected path list
```

status は `resolved` / `patch_fail` / `no_patch` / `crashed` / `not_run`（= その task に eval が到達していない）。
手元でも読めます: `make failures RESULTS=results/kaggle_nb`
""")

code('''import pipeline.stage_analyze as stage_analyze
import pipeline.stage_patch as stage_patch

stage_patch.run(ctx)
stage_analyze.run(ctx)
''')

# ---------------------------------------------------------------- 10. manifest / submit
md("""## cell 10 — 測った条件を記録して、提出する

`write_manifest` で variant / hardware / deviations / task 数を JSON に残します。
**提出は cell を run した kernel ではなく `submission.zip` を `kaggle competitions submit` すること**です。
採点は公式側（4×L4, tp=4, bfloat16, wheelhouse pin）で走ります。

```bash
kaggle kernels output <you>/gemma4-l4-measure -p results/kaggle_nb -o -q
kaggle competitions submit -c gemma-4-developer-agent -f submission.zip -m "e4b agent (prompt+config)"
kaggle competitions submissions -c gemma-4-developer-agent
```
""")

code('''from pipeline.common import write_manifest

manifest_path = write_manifest(ctx, ["setup", "data", "agent", "graph", "server", "eval", "package"])
print("manifest:", manifest_path)
print(json.dumps({"variant": ctx.cfg.variant, "run": ctx.cfg.run_name,
                  "model": ctx.cfg.model_name, "tasks": len(ctx.tasks),
                  "hardware": ctx.notes.get("hardware"),
                  "deviations": ctx.cfg.warnings()}, ensure_ascii=False, indent=2))

working = ctx.paths.working_dir
files = sorted(p for p in working.rglob("*") if p.is_file())
print(f"\\n/kaggle/working: {len(files)} files")
for path in files[:40]:
    print(f"  {path.relative_to(working)}  {path.stat().st_size:,} B")
''')

md("""## quota を分割で使う（12 task を 1 run に詰めないとき）

`variants.yaml` の `shard_index` / `num_shards` は `swegemma eval --shard-index/--num-shards` と同じ意味です
（`task_index % num_shards == shard_index` の task だけ測る）。cell 2 で:

```python
cfg = pcfg.load("baseline", {"num_shards": 3, "shard_index": 0, "run_name": "base_s0"})
```

3 本に分けて測り、`results/<run>` を揃えてから `make analyze RUN=base_s0` でまとめます。
""")


def notebook_bytes(source: Path, title: str, code_slug: str) -> str:
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12] if source.exists() else "-"
    header = (f"出所: `{source.name}`（hash {digest}）／生成 {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC\n\n"
              f"code dataset: `{code_slug}`（`src/pipeline/` が本体。ノートブックは薄い）\n\n"
              "この notebook は **1 cell ずつ実行する**。cell 6 の server を cell 7 が使うので kernel を restart しない。")
    cells = []
    header_merged = False
    for kind, text in CELLS:
        if kind == "markdown" and not header_merged:      # 冒頭 cell に出所・生成時刻を混ぜる
            text = header + "\n\n" + text
            header_merged = True
        lines = text.splitlines(keepends=True)
        cell = {"cell_type": kind, "metadata": {}, "source": lines}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.11"}},
          "nbformat": 4, "nbformat_minor": 4}
    return json.dumps(nb, ensure_ascii=False, indent=1) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="kernels/measure/gemma4_l4_measure.ipynb")
    ap.add_argument("--title", default="gemma4-l4-measure")
    ap.add_argument("--code-slug", default="gemma4-dev-agent-code")
    ap.add_argument("--metadata", default="kernels/measure/kernel-metadata.json")
    ap.add_argument("--model-ref", default="google/gemma-4/other/gemma-4-e4b-it-qat-w4a16-ct/2")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    src = Path(__file__).resolve()
    out.write_text(notebook_bytes(src, args.title, args.code_slug), encoding="utf-8")
    print(f"[measure-ipynb] wrote {out}（cells={len(CELLS) + 1}）")

    meta_path = Path(args.metadata)
    meta = {
        "id": None, "title": args.title, "code_file": out.name, "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": True, "dataset_sources": ["metric/gemma-4-developer-agent-wheelhouse"],
        "competition_sources": ["gemma-4-developer-agent"], "model_sources": [args.model_ref],
        "kernel_sources": [],
    }
    if meta_path.is_file():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["id"] = old.get("id") or meta["id"]
        meta["dataset_sources"] = sorted(set(meta["dataset_sources"]) | set(old.get("dataset_sources") or []))
    user = "<username>"
    meta["id"] = f"{user}/{args.title}"
    meta["dataset_sources"] = sorted(set(meta["dataset_sources"]) |
                                     {f"{user}/{args.code_slug}", f"{user}/gemma4-dev-agent-submission"})
    meta["model_sources"] = [args.model_ref]
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[measure-ipynb] wrote {meta_path}（gpu=True, internet=True, "
          f"datasets={meta['dataset_sources']}, model={args.model_ref}）")
    try:
        import nbformat

        nbformat.validate(nbformat.reads(out.read_text(encoding="utf-8"), as_version=4))
        print(f"[measure-ipynb] nbformat validate OK（{len(nbformat.reads(out.read_text(encoding='utf-8'), as_version=4).cells)} cells）")
    except ImportError:
        print("[measure-ipynb] nbformat が無いので validate は飛ばした（手元 pip install nbformat）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
