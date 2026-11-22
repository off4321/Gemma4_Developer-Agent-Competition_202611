# Kaggle CLI / Kernel の使い方

このリポジトリで検証済み: **Kaggle CLI 2.2.5**（`kaggle --version`）。2.2.5 が pip 上の最新。

---

## 0. 認証

```bash
# ~/.kaggle/kaggle.json （この環境では ~/.kaggle/credentials.json に既に存在）
chmod 600 ~/.kaggle/kaggle.json
kaggle config view                      # username / env vars 確認
kaggle whoami 2>/dev/null || kaggle config view   # 念押し
kaggle quota                          # 認証が通れば quota 表が出る
```

無ければ Kaggle → Account → New API Key から `kaggle.json` をダウンロードして
`~/.kaggle/kaggle.json` に置く。

---

## 1. Kernel の 2 種類（このリポジトリは **notebook kernel** で通した）

| 型 | metadata の `kernel_type` | 実行 | 実測 |
|---|---|---|---|
| Notebook | `notebook` | Web UI か `kernels push` | ✅ 通る（`code_file` は生成した `.ipynb`） |
| Script | `script` + `language: python` | CLI から push | ❌ Kaggle 側 nbconvert が `.py` を JSON として読めず失敗: `nbformat.reader.NotJSONError: Notebook does not appear to be JSON: '#!/usr/bin/env python3...'` |

だからこの repo は「**source of truth は `.py`、push するのは生成された `.ipynb`**」にしている:

```bash
python3 src/pipeline/make_kernel_notebook.py --source kernels/eval/gemma4_e4b_eval.py \
  --title gemma4-e4b-agent-eval --code-slug gemma4-dev-agent-code \
  --metadata kernels/eval/kernel-metadata.json      # ここが kernel_type/code_file を書き換える
```
`.ipynb` は生成物なので `.gitignore` 済み（`kernels/**/*.ipynb`）。

### kernel-metadata.json の必須/重要フィールド

```jsonc
{
  "id": "<username>/gemma4-e4b-agent-eval",   // owner/slug（owner は Kaggle username）
  "title": "gemma4-e4b-agent-eval",
  "code_file": "gemma4_e4b_eval.py",           // このディレクトリ内の相対パス
  "language": "python",
  "kernel_type": "script",
  "is_private": "true",                        // 提出物を出すなら true
  "enable_gpu": "true",                        // T4 x2 を使う
  "enable_tpu": "false",
  "enable_internet": "true",                   // true でも推論は offline 設計
  "machine_shape": "",
  "dataset_sources": [                          // /kaggle/input/datasets/<owner>/<slug>/ に mount
    "metric/gemma-4-developer-agent-wheelhouse",
    "<username>/gemma4-dev-agent-submission"
  ],
  "competition_sources": ["gemma-4-developer-agent"],  // /kaggle/input/competitions/<ref>/ に mount
  "kernel_sources": [],                         // 他の kernel の output を input にできる
  "model_sources": [                            // /kaggle/input/models/<owner>/<slug>/<framework>/<variant>/<version>/
    "google/gemma-4/other/gemma-4-e4b-it-qat-w4a16-ct/2"
  ]
}
```

> `kaggle kernels push -p DIR` は **DIR 直下だけを**アップロードします。
> `kernel-metadata.json` は push 対象ディレクトリの直下に必要（この repo では `kernels/eval/`, `kernels/train/` に置いた）。
> repo 直下に置くと `notebooks/` まで含めて 1 kernel になって紛れるので置いていない。

---

## 2. push（実行）

```bash
kaggle kernels push -p kernels/eval                     # 保存 + 実行開始
kaggle kernels push -p kernels/eval --no-run            # 保存だけ（実行しない）← 編集中はこれ
kaggle kernels push -p kernels/eval --accelerator NvidiaTeslaT4   # GPU 明示（T4 x2）
kaggle kernels push -p kernels/eval -t 7200            # この run だけ 2h で打ち切り（暴走防止）
```

- `--accelerator NvidiaTeslaT4` = **GPU T4 × 2（16GB×2 = 32GB）**。`NvidiaTeslaP100` は
  2026-09-15 に廃止済み（渡すと警告 → 代替機で走る）。
- `enable_gpu: true` かつ quota が尽きていれば **CPU で走る**（vLLM は動かない）。必ず `kaggle quota` を見る。
- 実行時間を切りたければ `-t <seconds>`。**後から止める CLI サブコマンドは 2.2.5 に無い**（下記の「停止」参照）。

### この repo での push 前に必要なもの（生のコマンド）

```bash
# src/（stage 本体）と submission/ を dataset に同期。**-r tar が必須**:
# dir-mode だと "Skipping folder: src; use '--dir-mode' to upload folders" で空になり
# "Dataset creation error: Please upload at least one file" になる。
kaggle datasets version -p .kaggle/dataset-code       -m "sync src/ $(date -u +%F\ %H:%M)" -r tar -q \
  || kaggle datasets create  -p .kaggle/dataset-code                                          -r tar -q
kaggle datasets version -p .kaggle/dataset-submission -m "sync submission/ $(date -u +%F\ %H:%M)" -r tar -q \
  || kaggle datasets create  -p .kaggle/dataset-submission                                    -r tar -q
```

メタデータは `.kaggle/dataset-code/dataset-metadata.json` / `.kaggle/dataset-submission/dataset-metadata.json`。
`-r tar` で上げると **受け取り側は folder が平坦になる**（`.../gemma4-dev-agent-code/pipeline/run_pipeline.py` のように
`src/` 抜きで_MOUNTされる_）。stage 側は両方の形を許容する。

> Kaggle dataset のアタッチ先は `/kaggle/input/datasets/<owner>/<slug>/...`。
> 2 回目以降は `datasets version`（新規作成は最初の 1 回だけ）。Makefile は同じ事を `version || create` でやっている。

---

## 3. 状況確認（**status より logs**。status は完了後も RUNNING を返すことが実測である）

```bash
kaggle kernels logs <username>/gemma4-e4b-agent-eval --follow # ライブ tail（実行中から見られる）
kaggle kernels logs <username>/gemma4-e4b-agent-eval          # 今の run の全ログ（JSON array）
  > results/logs/smoke.json
python3 src/pipeline/dump_kernel_log.py results/logs/smoke.json --stage server eval   # 読みやすい行に
```

`status` は「終わったか」の最終確認にだけ使う（`QUEUED` の間は logs が空になる。これは失敗ではない）:

```bash
kaggle kernels status <username>/gemma4-e4b-agent-eval
# <username>/gemma4-e4b-agent-eval has status "KernelWorkerStatus.QUEUED" / RUNNING / COMPLETE / ERROR
kaggle kernels files <username>/gemma4-e4b-agent-eval         # output 一覧（落とす前に中身確認）
kaggle kernels list --mine                                     # 自分の kernel 一覧 + lastRunTime
kaggle quota                                                   # GPU/TPU 残りと refreshAt
```

推論ランのログで見るべき行（`src/pipeline/stage_*.py` が print する実物）:

```
[pipeline] variant=smoke stages=setup -> data -> agent -> graph -> server -> eval -> package strict=True
[setup] [heal] torch stack を vllm の pin に合わせる: pip install torch==...
[setup] [measure-only] vllm 0.19.1 -> 0.28.0 を PyPI から（依存込み）
[server] gpus=2 [Tesla T4, Tesla T4] sm75 tp=2 dtype=float16 gpu_mem=0.85 adapters=[] ranks=-
[server] [attempt base] starting vLLM: extra_args=['--enforce-eager'] env=-
[smoke] completion OK (200): {"id":"chatcmpl-..."}
[eval] budgets（eval_config.yaml から）: {'max_tool_calls': 40, 'max_time_minutes': 12, 'max_turns': 80}
[eval] [1/1] fastapi_15661 (fastapi/fastapi)
    resolved=False exit=1 patch_chars=1423 tool_calls=9 duration=96.4s
[eval] tasks=1 patches=1 resolved=0 crashed=0
[pipeline] ran    : setup(525.1s), data(0.0s), agent(2.2s), graph(1.6s), server(...), eval(...), package(0.0s)
```

`tool_calls=0` かつ `patch_chars=0` なら **agent の前に生成が壊れている**（`vllm_server.log` を見る）。
`vLLM up` まで行かないなら GPU quota / model mount / VRAM を疑う（`doc/environment_limits.md`）。

**Web UI で見るとき**: `https://www.kaggle.com/code/<username>/<slug>` → Session メニューに
実行中セッションとログ、`.../logs` で実行ログ一覧。

---

## 4. 停止 methods

CLI 2.2.5 に `kaggle kernels stop` は**存在しない**（`kernels -h` = list/files/get/init/push/pull/output/status/logs/update/delete/topics）。

| 手段 | 手順 |
|---|---|
| **Web UI** | kernel 画面右上のセッション表示 → `...` → **Restart session**（現セッション中断） / セッションの停止 |
| **push 時の上限** | `kaggle kernels push -p DIR -t 3600`（この run だけ 1h で kill） |
| **暴走防止（推奨）** | `CONFIG["TASK_LIMIT"]` を小さくする / `submission/eval_config.yaml` の `max_time_minutes` を下げる |
| **うっかり実行避け** | 編集は `--no-run`、実行は明示的に 1 回 |

GPU quota は session 単位で消費するので、失敗が確実な run を放置すると **週 30h** を無駄に食う。
`logs --follow` で `[server] vLLM up` 失敗を早期検知したら UI で止めるのが最も効率的。

---

## 5. output 取得（artifacts / adapter / zip）

```bash
kaggle kernels output <username>/gemma4-e4b-agent-eval -p results/run01 -o -q
kaggle kernels output <username>/gemma4-e4b-lora-train -p adapters_out -o -q
kaggle kernels output <owner>/<slug> -p out --file-pattern "*.patch"   # 特定パターンだけ
```

eval kernel の output（`/kaggle/working` 配下）:

```
results/run01/
├── summary.json                 # resolution_rate 等の集合
├── task_results.jsonl           # 1 task 1 行のメトリクス
├── patches/<instance_id>.patch  # Phase 1 が抽出した diff
├── test_outputs/<id>.log        # Phase 2 の pytest 全体出力
├── traces/trace_<id>.json       # 思考・tool call・token の全 trace
└── logs/<id>.log                # Phase 1 セッションの transcript
submission.zip                   # validate 済み＝そのまま提出できる
submission.parquet               # id, prediction（=patch or NO_PATCH）
```

学習 kernel の output:

```
adapters/main_lora/adapter_config.json
adapters/main_lora/adapter_model.safetensors
trajectories.jsonl
```

---

## 6. 提出と確認

```bash
kaggle competitions submit -c gemma-4-developer-agent -f submission.zip -m "e4b prompt-only baseline"
kaggle competitions submissions -c gemma-4-developer-agent       # score 一覧（反映に time）
kaggle competitions files -c gemma-4-developer-agent --page-size 200   # 公開データ一覧
kaggle competitions download -c gemma-4-developer-agent -f tasks.jsonl -p data
kaggle models instances versions files google/gemma-4/other/gemma-4-e4b-it-qat-w4a16-ct/2
```

`submission.zip` の **ルート直下に agent.yaml**。 organiser の inference script がそれを展開して
`/kaggle/working/submission.parquet` を作り、metric が Resolution Rate を返す。

---

## 7. よくある落とし穴

1. **`kernel-metadata.json` が push ディレクトリ直下にない** → push されるが実行されない／新規 kernel が作られる。
2. **dataset_sources を足すのを忘れる** → wheelhouse 安装が走らず `import swegemma` で ImportError。
3. **model_sources 忘れる** → `/kaggle/input/models/google/gemma-4/other/<model>/2` が無く assert で即死。
4. **quota 切れで CPU 実行** → vLLM が起動せず 25 分を無駄にする。push 前に `kaggle quota`。
5. **同一 slug に push しまくると version が積まれる** → 状況確認は `<owner>/<slug>`（最新版）か `<owner>/<slug>/<version>` 明示で。
6. **`kernel_type: script` で `if __name__ == "__main__"` がない** → 何も実行されない。
7. **stdout が flush されない** → `logs --follow` が止まって見える。print には `flush=True`（この repo の eval kernel は対応済み）。
8. **大量 output** → Kaggle の output ディスク上限（约 20GB）に注意。traces/logs は task 数に比例する。
