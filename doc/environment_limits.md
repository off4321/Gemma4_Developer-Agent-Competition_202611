# 実行環境: 採点側（4×L4）と Kaggle notebook（T4×2）

## 結論（質問への直接回答）

> **Kaggle notebook で 4×L4（採点環境と同じ GPU）は選べない。**
> 2026-09-15 以降、Kaggle の無料 GPU は **Nvidia Tesla T4 × 2（16GB × 2 = 32GB）** のみ。
> P100 は廃止済み（`--accelerator NvidiaTeslaP100` は警告して代替機で走る）。
> `machine_shape` も自由指定は効かない（`kernels push --accelerator` の選択肢に限る）。

つまり **推論サーバーの構成は採点側と完全には揃えられない**。揃えられるのは
「モデル・ツール・プロンプト・budget・patch 抽出/検証ロジック」で、そこを揃えれば
ローカル↔Kaggle↔採点のスコアは十分比較可能になる。

---

## 1. 採点環境（HARNESS_README §3.1）

| 項目 | 値 |
|---|---|
| GPU | **4 × NVIDIA L4**（24GB each / 96GB total） |
| vLLM | `tensor_parallel_size=4`, `gpu_memory_utilization=0.90`, `max_model_len=32768` |
| tool/reasoning parser | `gemma4` / `gemma4`（reasoning 区切りは `<\|channel>` … `<channel\|>`） |
| LoRA | `enable_lora=bool(adapters)`, `max_loras=len(adapters)`, `max_lora_rank=max(rank) or 128` |
| sandbox | 1 コンテナ **4 GiB RAM / 2 vCPU / network none**, workdir `/workspace` |
| 公認モデル | `gemma-4-{31b,12b,e4b,e2b}-it-qat-w4a16-ct`（1 提出につき 1 モデル） |
| 提出上限 | 展開後 **3 GiB**（adapters 込み）, 拡張子は `.yaml .yml .md .txt .py .json .safetensors` |

### 公認モデルの weight footprint（Harness README 記載）

| Model | 公称 footprint（4 GPU 合計） |
|---|---|
| `gemma-4-31b-it-qat-w4a16-ct` | ~16–18 GB |
| `gemma-4-12b-it-qat-w4a16-ct` | ~6–8 GB |
| `gemma-4-e4b-it-qat-w4a16-ct` | ~3–4 GB |
| `gemma-4-e2b-it-qat-w4a16-ct` | ~2–3 GB |

一方 **Kaggle Models に実際に置いてあるファイル**（`kaggle models instances versions files` で実測）:

| Variation (`google/gemma-4/other/...` v2) | model.safetensors |
|---|---|
| `gemma-4-31b-it-qat-w4a16-ct` | 23.3 GB |
| `gemma-4-12b-it-qat-w4a16-ct` | 10.3 GB |
| `gemma-4-e4b-it-qat-w4a16-ct` | **11.5 GB** |
| `gemma-4-e2b-it-qat-w4a16-ct` | 8.3 GB |
| `transformers/gemma-4-e4b-it` v1（bf16・学習用） | 16.0 GB |
| `transformers/gemma-4-e2b-it` v1 | 10.2 GB |

→ 「e4b = 4B」ではなく **総パラメータは 8B 級（effective 4B）**。README の footprint 値とは差があるので、
実際の VRAM は kernel 内で vLLM 起動ログと `nvidia-smi` で確認する（下記 §4）。

---

## 2. Kaggle notebook のセッション実測上限

| 項目 | 値 |
|---|---|
| GPU | **T4 × 2 = 32 GB**（bf16 matmul 非対応 → fp16 で動かす） |
| RAM / CPU | 約 30 GB / 4 cores |
| Session 時間 | **CPU/GPU: 最大 12 時間**（TPU: 9 時間）。GPU quota とは別に切る |
| GPU quota | **週 30 時間**（`kaggle quota` の `refreshAt` = 土曜 00:00 UTC） |
| TPU quota | 週 20 時間 |
| Output disk | 约 20 GB（traces/logs は task 数に比例して膨れる） |
| Internet | 有効化可能（推論は offline 設計。学習時に pip が入る） |

### 1 run の目安（TASK_LIMIT=12, e4b）

- vLLM 起動（e4b, tp=2）: 3–8 分
- 1 task: 90–400 秒（agent loop 時間。コンテナ準備は別途）
- 12 task: 25–80 分 → **GPU quota の 1/15 前後**
- 129 task 全部: 概算 5–12 時間 → **quota を使い切る风险**。`TASK_LIMIT` で切るか、
  `--shard-index/--num-shards`（`swegemma eval` CLI）で分割して数日に分ける

---

## 3. 採点環境との差を縮めるための設定（`kernels/eval/gemma4_e4b_eval.py` に実装済み）

| 項目 | 採点側 | Kaggle kernel 側（この repo） |
|---|---|---|
| `tensor_parallel_size` | 4 | `4 if gpus>=4 else 2 if gpus>=2 else 1`（T4×2 → **2**） |
| `dtype` | `bfloat16` | `bfloat16 if is_bf16_supported() else float16`（T4 → **float16**） |
| `gpu_memory_utilization` | 0.90 | gpus<4 なら **0.85**（KV cache 確保の余裕） |
| `max_model_len` | 32768 | **32768 を維持**（変えると agent の挙動が採点とズレる。VRAM 不足時のみ 16384 に落とす＝差異として記録） |
| `tool_call_parser` / `reasoning_parser` | `gemma4` | 同一 |
| sandbox | `docker` | **`subprocess`**（Kaggle notebook に docker daemon が無い） |
| budget | `eval_config.yaml` | 同一ファイルを読む |
| LoRA | rank 自動 | 同一（`max_lora_rank=max(rank) or 128`） |

### 残る既知の差異（スコア解读時に考慮）

1. **fp16 vs bf16**: 生成がわずかにズレる（temperature 0.2 でも差分は出る）。傾向を見るのに十分。
2. **tp=2 vs tp=4**: 数値の丸め順が違う。同上。
3. **docker vs subprocess sandbox**: subprocess backend は `/workspace` を一時ディレクトリに写し、
   `pip install -e` 相当が効かない場合がある → 「import 失敗で解決できない task」がローカルだけ増える。
   → **ローカルの解決率 ≦ 採点の解決率** と考えてよい。

---

## 4. VRAM を確かめるコマンド（kernel 内 / push 前に）

```bash
nvidia-smi                                   # GPU 型号と memory
python3 -c "import torch;print(torch.cuda.device_count(), [torch.cuda.get_device_properties(i).name for i in range(torch.cuda.device_count())], torch.cuda.is_bf16_supported())"
```

vLLM 側は起動ログに `GPU memory usage` / `Maximum memory for state` が出る。
OOM なら `gpu_memory_utilization` を 0.8 に、`max_model_len` を 16384 に、
adapters を 1 本に減らす。T4×2 で e4b + LoRA + 32k ctx は「入る見込みだが余裕は薄い」。
31b は T4×2 では無理、12b は ctx を削らないときつい（→ e4b を選ぶ理由）。

---

## 5. **T4 では Gemma 4 が生成できない**（2026-11-21 実測で確定）

kernel version 19 のログ（`make logs-dump RUN=smoke LOG_STAGE="server eval"` で取った）:

```
[server] [attempt base] 生きてるのに生成が失敗（health は OK でも attention kernel で死ぬ）
  RuntimeError: Worker failed with error 'out of resource: shared memory,
               Required: 98304, Hardware limit: 65536. Reducing block sizes or `num_stages` may help.'
  -> triton_attn.py -> ops/triton_unified_attention.py -> triton.runtime.errors.OutOfResources
```

5 config（base / `--block-size 64` / `--block-size 32` / FLASH_ATTN / FLASHINFER）全部で
`Required: 98304` が**同じ数字のまま**出た。理由は上流が報告している通り:

- vLLM issue #38918: *Gemma4 on Turing GPUs (SM 7.5): all attention backends fail* —
  Gemma4 は `head_dim: 256`（sliding layers）と `global_head_dim: 512`（full attention）で、
  Triton attention kernel が **96 KB** shared memory を要求、Turing の上限は **64 KB / SM**。
- vLLM issue #57712 + fix PR #60942 / #39018（T4 で動くようになった修正は**より新しい vLLM 側**に入っている）。
- 失敗は **Triton JIT の compile 時**。だから `--block-size`（KV cache の分割サイズ）や
  `--max-num-seqs` のような実行時 knob は効かない（実測で 5/5 が同値で失敗）。

つまり wheelhouse の pin（vLLM 0.19.1）をそのまま使う限り、**Kaggle free の T4×2 では Phase 1 が 1 token も生成できない**。
`setup` / `graph` / `server` 起動 / `package` まで通っても `[eval]` は `tool_calls=0` になる（v17, v18 でも同じ）。

### どう付き合うか（selected: 未決定）
| 選択肢 | 内容 | 代償 |
|---|---|---|
| A. 公式採点で測る | `make submit` で `submission.zip` を提出すると採点は **4×L4 (sm89)** で走る。T4 の壁に影響されない | 提出回数制限、結果が返るまで時間、1 回ずつしか試せない |
| B. 計測だけ新しい vLLM に载せ替える | upstream fix 入りの vLLM を PyPI から入れて T4 で生成させる | 公式 env（0.19.1, wheelhouse）との**数値差**が出る。`adk_submission` の期望 flag とズレる可能性 |
| C. 生成を要さない解析に寄せる | patch 解析・契約検査・variant 設計・プロンプト改善は GPU 無しで進められる | 解決率そのものは A に依存する |

**公式採点環境（4×L4, sm89, shared memory 100 KB）では同じ kernel が通る**ので、
「agent が解けるか」の答えは A でしか出ない。ここで測れるのは setup / 起動 / 契約 / patch 解析まで。

---

## 6. Colab / 手元 GPU で測る（Kaggle mount が無い環境）

`src/pipeline/common.py` は Kaggle mount を **既定値として持つだけ**で、無いときの代替パスが用意済み:

| 用途 | Kaggle mount（既定） | 無いときの代替（`common.py:183-191, 228`） |
|---|---|---|
| competition data | `/kaggle/input/competitions/gemma-4-developer-agent` | `<repo>/data`（`--data-dir` で変更可） |
| wheelhouse | `/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse` | `<repo>/data/wheelhouse/*.whl` |
| model weights | `/kaggle/input/models` | `<repo>/models/google/gemma-4/other/<model_name>/2` |
| working | `/kaggle/working` | `<repo>/work`（`--working-dir`） |

Colab で足りるもの（全部 public で確認済み 2026-11-21）:

```bash
kaggle competitions download -c gemma-4-developer-agent -p data
kaggle datasets download -d metric/gemma-4-developer-agent-wheelhouse -p data/wheelhouse --unzip
hf download google/gemma-4-e4b-it-qat-w4a16-ct \
  --local-dir models/google/gemma-4/other/gemma-4-e4b-it-qat-w4a16-ct/2      # 11.5 GB
python3 src/pipeline/run_pipeline.py --variant t4fix --stages gpu --strict     # --strict が必須（下記）
```

- **`/kaggle/working` が無いと strict の既定が False** になり、GPU stage が「理由付き skip」になる。
  Colab で GPU stage を本当に走らせるなら `--strict` を明式に付ける。
- `run_coro`（`stage_eval.py`）は **ipykernel の running event loop** を扱うので、Colab でも同じ事故を避ける
  （Kaggle kernel と Colab は同じ落とし穴を持つ）。
- Colab free の GPU は **1×T4 (sm75)** = §5 と同じ壁。`vllm_version: "0.28.0"` が無いと生成 0 token。
  `tensor_parallel_size` は 1 になる（`stage_server.py` は GPU 枚数で決める）。4bit 重み 5.44 GB + 32k ctx
  を 16 GB に積むので `gpu_memory_utilization` 0.85 が入るかどうかは実測確認が必要。
- Colab の GPU 使用量は**週単位の動的上限**（公式に固定値は出ておらず、需要により変動。無操作 ~90 分で
  runtime 切断）→ 長時間 run は心拍（`while True: sleep`）か有料 plan が必要。
- wheelhouse の `cp312-cp312` wheel は Kaggle の cp313 では弾かれたが、Colab の image（3.12 系）では
  そのまま通る可能性が高い → setup の `[wheel]` 行が減る（要実測確認）。
