# LoRA プラン（何を改善するための LoRA か）

## 0. 前提: LoRA で直るもの／直らないもの

| 直りうる（行動の癖） | LoRA が効く率 | 先に打つ手 |
|---|---|---|
| tool call の JSON 生成が壊れる／`old_string` が実在してない | 高 | prompt + `max_output_tokens` |
| 思考が長すぎて `<\|tool_call\|>` が途中で切れる | 高 | `thinking_budget` / `max_output_tokens` |
| 文章だけ返して `submit_patch` しない（NO_PATCH） | 高 | prompt + turn budget |
| 探索 Tool を使い過ぎて budget 枯渇 | 中 | sub-agent 分離 |
| 大きな `write_file` 全書き換え → Phase 2 apply 失敗 | 中 | prompt + `edit_file` 強制 |
| リポジトリの API 知識／ファイル名の当たり | **低** | prompt・graph tool・context |
| 仕様文字列（例外メッセージ等）の完全一致 | 低（データ次第） | prompt 強調 |

→ **まず prompt/config で上限を作り、残った「癖」を LoRA で潰す**（doc/tasks.md の順序はこれ）。

---

## 1. 学習データ

### 1.1 gold patch から合成（`src/collect/collect_trajectories.py`）
`data/tasks.jsonl`（129 task）から **実在する `old_string`/`new_string`** と
**実在する pytest target**（`test_patch` の `def test_*`）でツール呼び系列を組み立てる:

```
system(prompts/system.md) → user(harness 相当の初期 prompt)
 → assistant[thought + edit_file(実差分)] → tool(ok)      × 1..6
 → assistant[thought + run_command(pytest <target>)] → tool(ok)
 → assistant[thought + submit_patch()] → tool(ok, patch_size=…files_changed=…)
 → assistant(final text)
```

- 生成数: **106 / 129**（skipped: edits 数 >6 = 15、巨大 hunk = 8）
- 平均 **4.0 tool calls / trajectory**、約 3k tokens
- 狙いは「**短軌跡で submit まで到達**」と「**edit_file が初回で成功する形**」の刷り込み

⚠ 弱点: 観測（tool 応答）が理想化されている（`exit_code: 0` 前提）。
実 trace を混ぜて「失敗観測 → 修正」の型も入れないと、1 発でテストが通る前提のモデルになる。

### 1.2 自己蒸留（本命、T8）
`kaggle kernels output` で得た `results/runXX/traces/trace_*.json` のうち
**resolved=True のものだけ**を `--import-traces --only-resolved` で合流させる。
失敗観測を含む実分布が入るので 1.1 より効く。ループ防止の教材にもなる。

### 1.3 外部データ（任意）
SWE-bench 系・SWE-Gym 等の open 数据を追加可。ただし公開 129 task は
`fastapi/fastapi` 67 + `Textualize/rich` 48 + `psf/requests` 13 + `encode/httpx` 1 に偏る。
**評価 split は非公開**なので repo 偏りに過学習すると本番で落ちる → 外部数据 or trace 蒸留で均す。

---

## 2. 学習設定（`src/train/train_lora.py`）

| 項目 | 既定 | 根拠 |
|---|---|---|
| base | `google/gemma-4-e4b-it`（bf16, Kaggle mount `transformers/gemma-4-e4b-it/1`） | 学習は bf16、推論は `-qat-w4a16-ct` 上。公式 sample adapter と同じ手順 |
| target modules | q,k,v,o + gate,up,down | tool 出力・コード生成の両方に効く定番 |
| `r` / `alpha` | 16 / 32（alpha=2r） | e4b 級で adapter 数十〜数百 MB。3GiB 上限に余裕。`r>128` は vLLM 拒否 |
| mask | assistant のみ（system・tool 観測は -100） | 観測はハーネスが発行する。 model に生成させてはいけない |
| seq | `--max_len 4096` | trajectory 平均 3k token。context 32k すべてを使わせる必要はない |
| epochs / lr | 3 / 1e-4（linear, warmup 20） | 106 例だと overfit 早い。loss 低下より **format 安定**を見る |
| batch | 1 × grad_accum 8 | T4×2・fp16 の現実解 |
| 4bit | `--use_4bit` は fp16 で OOM した場合のみ | 4bit base 上で学習した adapter は vLLM W4A16 上でわずかにズレる |

Loss masking の要点（`encode_conversation`）: chat template を 1 message ずつ render して連結し、
assistant turn の token だけ label を残す。tool call が template 上でシリアライズされない場合は
`<tool_call>{json}</tool_call>` を text で補う（実行時に自動判定）。**推論側が native gemma4 parser
なので、native 形式で出るかを 1 度必ず確認する**（`[train] chat template mentions tools: True`）。

---

## 3. 検証プロトコル（A/B は必須）

```bash
# 1) adapter を submission に入れる
cp -r adapters_out/adapters/main_lora submission/adapters/main_lora
# submission/agent.yaml の `# adapter: main_lora` をコメント解除（analyzer にも tool_lora 任意）
make validate

# 2) 同じ TASK_IDS で 2 回走る（off → use）
#    kernels/eval CONFIG: ADAPTER_MODE="off" → push → results/run_off
#    kernels/eval CONFIG: ADAPTER_MODE="use" → push → results/run_lora
make out-eval RUN=run_off
make out-eval RUN=run_lora
make analyze RUN=run_off
make analyze RUN=run_lora
```

見る数字（`doc/harness_watchpoints.md` §1 の表に対応）:
- resolution rate（主目的）
- **NO_PATCH 数**（submit 到達率） ← LoRA の主効果はここ
- **apply 失敗数** ← edit 形状の改善はここ
- tool_calls 平均（↓ なら loop 改善）、trace の nudge / MAX_TOKENS 数（↓ なら loop 改善）

**LoRA を入れて resolution が 1 task 以上下がったら却下**（smoke test で adapter 読み込み成功
しても、解決率が下がる adapter はある）。

---

## 4. adapter の提出ルール（ここを外すと提出エラー）

- `adapters/main_lora/adapter_config.json` + `adapter_model.safetensors`（**`.bin/.pt/.pth` 不可**）
- adapter 名は `[A-Za-z0-9][A-Za-z0-9._-]*`
- `r <= 128`（vLLM `max_lora_rank`）。`train_lora.py` が自動警告する
- `adapter_config.json` の `base_model_name_or_path` は **`google/gemma-4-e4b-it-qat-w4a16-ct`**
  （`rewrite_adapter_config()` が自動で書き換え。bf16 で学習しても đây に合わせる）
- adapters 込みで展開後 **3 GiB 未満**
- agent.yaml の `adapter:` は **存在する adapter 名**を指すこと（`build_submission.py` が検査する）。
  使う場合は `# adapter: main_lora` のコメントを解除する

## 5. 2 本目の adapter（任意）

analyzer 用に `tool_lora` を別に訓練する価値はあるか？ 探索特化なら「検索→報告」の
書式が安定して効果が出ることがある。ただし `max_loras=len(adapters)` で LoRA スロットが増え、
vLLM の VRAM と初回レイテンリが増える。**まず 1 本で取り切る**こと。
