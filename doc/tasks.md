# タスクリスト（優先順）

状態: `[ ]` 未着手 / `[~]` 進行中 / `[x]` 完了。
各タスクに **完了条件** を書いてあるので、満たせないものは「やったことにしない」。

---

## フェーズ 0 — 足場（0 GPU で終わる）

### [x] T0 提出仕様に submission/ を合わせる
- 完了条件: `make validate` が PASSED。`agent.yaml` が `submission/` ルート、
  `eval_config.yaml` は `evaluation:` key 付きでルート、`sampling.yaml` は `thinking_config` ネスティング済み。
- 済み: `submission/{agent.yaml,eval_config.yaml,configs/sampling.yaml,prompts/*,sub_agents/code_analyzer.yaml}`

### [ ] T1 Kaggle 側の入力（dataset 2 本）を最初に作って固定する
```bash
kaggle datasets create -p .kaggle/dataset-code        -q   # 1 回目だけ
kaggle datasets create -p .kaggle/dataset-submission  -q
make push-code && make push-submission
kaggle datasets files <username>/gemma4-dev-agent-submission | head
```
- 完了条件: 両 dataset に `agent.yaml` / `src/train/train_lora.py` が **正しい階層で**並んでいる。
- 落とし穴: 既存 dataset が無いと `datasets version` は失敗する（Makefile は `version || create`）。

### [ ] T2 GPU quota と model mount を確認
```bash
kaggle quota
kaggle models instances versions files google/gemma-4/other/gemma-4-e4b-it-qat-w4a16-ct/2
```
- 完了条件: GPU 残り ≥ 3h。`kernel-metadata.json` の `model_sources` が v2 を指している。

---

## フェーズ 1 — 基準線（LoRA なし）

### [ ] T3 e4b・adapter なしの baseline を 12 task で測る
```bash
# kernels/eval/gemma4_e4b_eval.py CONFIG:
#   TASK_LIMIT=12, TASK_IDS=[], ADAPTER_MODE="off", MAX_TOOL_CALLS=80, MAX_TIME_MINUTES=45
make push-eval && make logs-follow
make status
make out-eval RUN=run01 && make analyze RUN=run01
```
- 完了条件: `[results] tasks: 12` と per-repo 分解が出す。`submission.zip` が output に入っている。
- 記録するもの: 解決数、NO_PATCH 数、apply 失敗数、tool_calls 平均、duration 平均。
  → 以降の比較表の 1 列目になる。

### [ ] T4 1 run の実測時間を出して、1 日に何 task 回せるか決める
- 完了条件: 1 task あたりの平均 duration が分かり、`TASK_LIMIT` の上限方針が決まった
  （quota 30h/週に対し 1 run ≤ 2h を目安）。

---

## フェーズ 2 — prompt / 設定スイープ（最も費用対効果が高い）

### [ ] T5 3 変数だけ揃えて比較する（1 変え / 1 run）
| 変数 | 候補 |
|---|---|
| `thinking_budget` | 0 / 2048 / 4096 |
| `max_tool_calls` | 40 / 80 / 120 |
| `max_turns` | 60 / 150 |
- 完了条件: 同じ `TASK_IDS` で run 名を変えて 3–5 本。analyze の NO_PATCH / nudge 数が比較表になっている。
- 注意: `eval_config.yaml` は **提出物本体**。best を書いてから zip を作る。

### [ ] T6 verifier / critic sub-agent を追加して 1 run
- 手順: `submission/sub_agents/verifier.yaml`（`run_command`, `read_file`, `get_status` のみ）を
  `agent_tool` で追加 → `make validate` → run。
- 完了条件: apply 失敗数または NO_PATCH 数が T3 より減るか、tool_calls 平均が減る。
  減らなければ削除する（残すと budget を食う）。
- 詳細: `doc/multi_agent.md` §2

### [ ] T7 patch 衛生の専用プロンプト断片を効かせる
- 手順: system.md の「scratch は /tmp」「`conftest.py`/`pytest.ini` を触るな」を
  上位に移動／再掲し、`/workspace` に残った未追跡ファイルを `git status` で確認する手順を追記。
- 完了条件: analyze の `patch touched protected test/runner files` が 0、
  かつ「scratch が patch に混入している task」が 0（`patches/*.patch` を 5 本目視）。

---

## フェーズ 3 — LoRA（基準線と prompt が固まってから）

### [ ] T8 自己蒸留データを混ぜて trajectories を作る
```bash
python3 src/collect/collect_trajectories.py --tasks data/tasks.jsonl \
  --import-traces results/run01 --only-resolved --output data/trajectories.jsonl
```
- 完了条件: gold 由来 + trace 由来が混在し、trace 由来に **失敗観測（exit_code!=0）を含む**ものが入る。
  1 件あたり tool call 数が 3–12 に散る。

### [ ] T9 学習 kernel を乾かす（まず dry_run）
```bash
# kernels/train CONFIG: DRY_RUN=True → make push-train → status → output
# OK なら DRY_RUN=False, EPOCHS=3, R=16, ALPHA=32, MAX_LEN=4096
make push-train && make out-train
```
- 完了条件: `adapter_config.json` の `base_model_name_or_path` が
  `google/gemma-4-e4b-it-qat-w4a16-ct`、`adapter_model.safetensors` のみ、r ≤ 128、
  `[train] chat template mentions tools: True`。OOM なら `USE_4BIT=True` か MAX_LEN 縮小。

### [ ] T10 A/B 判定（必須）
- 手順: `doc/lora_plan.md` §3 の通り `ADAPTER_MODE=off` と `use` を **同じ TASK_IDS** で。
- 完了条件: 解決数が上がり、NO_PATCH と apply 失敗が減り、nudge 数が減る。
  **1 task でも解決数が減ったら adapter は却下**（`submission/adapters/` を消して T5 に戻る）。

### [ ] T11 scale: task 数を増やして確認 → 提出
```bash
make zip && make submit
kaggle competitions submissions -c gemma-4-developer-agent
```
- 完了条件: 提出物の zip が validate 済み（kernel 内 `validate_directory` を通過したもの）。
  提出コメントに「どの run のどの設定か」を書く（後で比較できるように）。

---

## フェーズ 4 — 追い込み（締切 2026-12-02 まで）

### [ ] T12 task 側への働きかけ（モデル側ではなく足回り）
- `TASK_IDS` に「公開 129 のうち解決できない 20 本」を固定して **毎回同じ 20 本で測る**
  （比較のコストが 1/6 になる）。
- `swegemma eval --task-ids ... --concurrency 2`（CLI 使用可なら）で並列化。
- shard 分割（`--shard-index k --num-shards M`）で quota 内に全 task 評価。

### [ ] T13 repo 偏りへの保険
- 公開 129 は fastapi 67 / rich 48 / requests 13 / httpx 1。評価 split は非公開。
- LoRA を入れるなら外部 SWE データを混ぜるか、adapter の rank を下げて汎用性を残す
  （r=8/16 比較を 1 本ずつ）。

### [ ] T14 提出戦略
- 締切まで毎日数本提出可（上限は Kaggle UI / Meta Kaggle で確認）。
- 「prompt のみ」「prompt+LoRA」「sub-agent あり/なし」を **1 変更 1 提出**で本番スコアを追う
  （ローカル↔本番のズレ幅度を把握する）。

---

## 今の私の推奨順序（一言で）

**T1 → T3（基準線）→ T5/T7（prompt & budget）→ T8/T10（LoRA を A/B で判定）→ T11（提出）**
LoRA は最後に、しかも「NO_PATCH と apply 失敗を減らす道具」として使う。先にやると比較基準が消える。
