# 実行→評価パイプライン（`src/pipeline/`）

公式 Getting Started ノートブック（`notebooks/getting-started-gemma-4-developer-agent.ipynb`）の
6 セルを **1 ステージ 1 モジュール**に切り出したもの。同じコードが

* このマシン（GPU なし / swegemma 未インストール）で **preflight として**走り、
* Kaggle kernel では **GPU 実行として**そのまま走る。

`kernels/eval/gemma4_e4b_eval.py` は薄くなった: 変数名を読んで `src/pipeline` を import するだけ。
`src/` は dataset `<you>/gemma4-dev-agent-code` として Kaggle に上がる（`make pipeline-eval` が push-code 込み）。

## stage 対応表

| stage | notebook cell | 内容 | ローカルで走る？ |
|---|---|---|---|
| `setup` | Cell 1 | オフライン環境変数、ログ静默、wheelhouse install（cutlass `.pth` 除去・`+cu128` 復元） | 部分的（wheel が無いなら報告だけ） |
| `data` | Cell 2 | `load_tasks` + task 選別（ids / shard / limit） | ○（jsonl フォールダで129件読める） |
| `agent` | Cell 2 | `submission/` を `work/agent_dir` に stage → variant 適用 → **契約チェック** | ○ |
| `graph` | Cell 3 | 事前計算シンボルグラフ / 埋め込みの smoke | ✗（swegemma 要） |
| `server` | Cell 4 | 単一モデル検証、adapter 発見、vLLM 起動、model registry、smoke completion | ✗（GPU + adk_submission 要） |
| `eval` | Cell 5 | `EvalConfig` + `Evaluator`：Phase 1 agent loop + Phase 2 検証、`submission.parquet` | ✗ |
| `package` | Cell 6 | symlink / 拡張子 / 3GiB / ルート `agent.yaml` 検証 → zip | ○ |
| `patch` | – | patch 静的検査（agent patch または gold patch） | ○ |
| `analyze` | – | `results/<run>` を失敗 taxonomy に + **`failures.csv` / `failures.md`**（task 別の不出来） | ○ |

プリセット: `preflight` = setup,data,agent ／ `local` = preflight+patch,analyze ／
`gpu` = setup..package（kernel が走る順） ／ `post` = patch,analyze ／ `all`。

## 使い方（**生の Kaggle コマンドが主語**。make は同じ事を並べた省略可能な便利版）

### Kaggle 側で打つ 6 本

```bash
# 0) variant を kernel に焼き込む（kernel は実行時に CLI 引数を渡せないから CONFIG に書く）
python3 src/pipeline/set_kernel_variant.py smoke

# 1) src/ と submission/ を dataset に更新（-r tar が必須: dir-mode だと folder が飛ぶ）
kaggle datasets version -p .kaggle/dataset-code -m "sync src/ $(date -u +%F\ %H:%M)" -r tar -q \
  || kaggle datasets create  -p .kaggle/dataset-code                           -r tar -q
kaggle datasets version -p .kaggle/dataset-submission -m "sync submission/" -r tar -q \
  || kaggle datasets create  -p .kaggle/dataset-submission                     -r tar -q

# 2) source of truth の .py から notebook を生成（script kernel は nbconvert で壊れる）
python3 src/pipeline/make_kernel_notebook.py --source kernels/eval/gemma4_e4b_eval.py \
  --title gemma4-e4b-agent-eval --code-slug gemma4-dev-agent-code \
  --metadata kernels/eval/kernel-metadata.json

# 3) push（この瞬間に実行が始まる。GPU quota を消費）
kaggle kernels push -p kernels/eval

# 4) ログ（**実行中にも**見られる。status を繰り返さないこと）
kaggle kernels logs <username>/gemma4-e4b-agent-eval --follow
kaggle kernels logs <username>/gemma4-e4b-agent-eval > results/logs/smoke.json
python3 src/pipeline/dump_kernel_log.py results/logs/smoke.json --stage server eval

# 5) artifacts（完了後。log / predictions / submission.zip / results dir）
kaggle kernels output <username>/gemma4-e4b-agent-eval -p results/smoke -o -q

# 6) 提出と公式スコア、quota
kaggle competitions submit      -c gemma-4-developer-agent -f submission.zip -m "baseline e4b"
kaggle competitions submissions -c gemma-4-developer-agent
kaggle quota
```

### 同じ事を make でやる（右辺が本体なので、左辺を打たなくても良い）

| make | 打たれているコマンド |
|---|---|
| `make pipeline-run VARIANT=smoke` | set_kernel_variant + datasets version ×2 + make_kernel_notebook + `kaggle kernels push -p kernels/eval` |
| `make logs-follow` / `logs-dump RUN=smoke` | `kaggle kernels logs <slug> --follow` / `... > results/logs/smoke.json` + dump_kernel_log |
| `make out-eval RUN=smoke` | `kaggle kernels output <slug> -p results/smoke -o -q` |
| `make push-code` / `push-submission` | `kaggle datasets version -p .kaggle/dataset-code -r tar -q` |
| `make submit` / `submissions` / `quota` | `kaggle competitions submit` / `submissions` / `kaggle quota` |
| `make probe` / `probe-out` | `kaggle kernels push -p kernels/probe` / `kaggle kernels output <probe slug>` |

### 手元（GPU 無し）でできるのは Python だけ

```bash
python3 src/pipeline/run_pipeline.py --list-variants
python3 src/pipeline/run_pipeline.py --variant baseline --stages local     # 契約+variant 適用+patch/analyze
python3 src/pipeline/failures_report.py --results results/<run>          # task 別の不出来だけ出す（make failures）
python3 src/pipeline/run_pipeline.py --variant baseline --stages gpu --limit 4 \
    --thinking-budget 0 --max-tool-calls 120 --strict
python3 src/pipeline/run_pipeline.py --variant baseline --stages agent     # 1 stage だけ
```

## variant（比較の単位）

`src/pipeline/variants.yaml` の `defaults:` + `variants:` を deep merge する。
**variant 名がそのまま run_name** になる（`make out-eval RUN=<variant>` / `make analyze RUN=<variant>` が揃う）。
`baseline` だけ `defaults.run_name`（`baseline`）を使う。

variant は stage `agent` によって **stage されたコピーにだけ**書き込まれる。
`submission/` は無変更なので、レビューできるソースと計測された提出物が一致する。
書き込み先:

| 対象 | 書き方 |
|---|---|
| `eval_config.yaml` の `evaluation:` | 行内置換（コメントと桁位置を維持） |
| `configs/sampling.yaml` | 上段はトップレベル、`thinking_*` は `thinking_config:` の **内側**だけを書く |
| `model:` | 既に key が存在するファイルだけ置換（`SequentialAgent` に key を発明しない）。置換後 YAML を再パースして全 agent の model 一致を検証 |
| `adapter_mode: off` | `adapters/` を削除 + `adapter:` 行を削除（ベースライン計測） |

`strict` 既定値: Kaggle 内 = `True`（ライブラリ欠落は即失敗）、ローカル = `False`（GPU stage は skip 報告）。

## ローカルでできることの境界

* `swegemma` / `adk_submission` / `vllm` は **PyPI に無い**（`pip download swegemma` → 404）。
  competition の wheelhouse dataset からしか入らない → **Phase 1/2 は Kaggle でしか走らない**。
* このマシンに GPU も docker も無い（`nvidia-smi` 無し）→ `server`/`eval` は設計上 Kaggle 専用。
* `data/tasks.jsonl`（129件）と `HARNESS_README.md` は手元にある。`snapshots/` `graphs/` `embeddings/`
  は数十 GB なので落さない（`paths.describe()` が MISSING と報告する）。

## 実装上の判断

* **task 選別は CLI 意味に合わせた**: `--task-ids` → filter、shard は `index % num_shards == shard_index`
  （`swegemma eval --shard-index/--num-shards` と同じ）、`limit` はその後。
* **eval は逐次実行**。`--concurrency` は CLI の機能で、2x T4 kernel で agent loop を並べても
  1 本の vLLM と RAM を奪い合うだけ。1 task の例外は catch して続きを実行する（quota 保護）。
* budget は staged `eval_config.yaml` から**読み戻す**（スコア側の `load_submission_eval_config` と同じ経路）。
* `verification_timeout_seconds` はスコア側で 300 s 固定（HARNESS_README 7.1）なので、
  kernel/ローカルループにだけ効く旨をログに出す。
* `package` は **stage された agent_dir を zip する**＝計測したもの＝提出する、を強制する。
  ローカルでは `<repo>/submission.zip` にもミラーするので `make submit` がそのまま繋がる。

## 次の一手（doc/tasks.md に対応）

1. `make quota` → 残り 30h/週（2026-11-28 00:00 UTC リセット）は確認済み。kernel 2 本は **まだ未作成**。
2. `make pipeline-eval VARIANT=smoke` で T3 の足場確認 → `make pipeline-post VARIANT=smoke`。
3. baseline 12 task（`VARIANT=baseline`）を 1 列として記録 → `think_0` / `think_2048` /
   `calls_40` / `calls_120` / `out_8192` を 1 変えずつ（T5）。
4. LoRA は `VARIANT=lora`（`adapter_mode: use`）で同じ baseline と比較（`doc/lora_plan.md`）。

## `analyze` が出す不出来ファイル（2026-11-22 追加）

`stage_analyze` は `src/pipeline/failures_report.py` を呼んで、results dir に次を書きます（Colab notebook の
cell 7 には**同じファイルのソースを注入している**ので、Colab 側と Kaggle 側で列が揃います）。

| file | 中身 |
|---|---|
| `results/<run>/failures.csv` | `instance_id, status, reason, test_exit_code, patch_chars, tool_calls, duration_s, error, test_output_tail` |
| `results/<run>/failures.md` | 読みた形式 / 測れた task 数 / 局所 resolution rate / status 集計 / task 表 / 代表エラー / protected path に触れた patch list |

`status` は `resolved` / `patch_fail`（patch は出たが pytest が落ちた） / `no_patch`（`submit_patch` 無し） /
`crashed`（harness 側 error） / `not_run`（**eval がその task に到達していない** = 測れていない）。
読み取りは自動判別（公式 harness の `task_results.jsonl` / 自分の pipeline の `predictions_*.json` / `*.parquet`）。

```bash
make failures RESULTS=results/kaggle_nb
python3 src/pipeline/failures_report.py --results results/colab12 --task-ids fastapi_15661,pandas_10001  # 測る予定だった id を足すと not_run が出る
```
