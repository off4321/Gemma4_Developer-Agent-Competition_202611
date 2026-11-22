# Gemma 4 Developer Agent Competition (2026) — workspace

Kaggle `gemma-4-developer-agent`（Featured, deadline **2026-12-02**, 5966 teams）用ワークスペース。
提出物は **`submission.zip`（ルート直下に `agent.yaml`）** で、主催側の inference script が
それを sandbox で走らせて `submission.parquet`（`id,prediction=patch`）を作り、
**Resolution Rate** で採点される。学習済みモデル単体ではなく **エージェント定義（YAML+プロンプト+任意で LoRA）** が提出物。

## 今どれを使えばいいか

**Colab で測るなら `kernels/colab/gemma4_colab.ipynb` 1 冊**（`make colab-notebook`）。資材 DL / model DL /
提出物の解決（提出 zip -> 自 dataset -> 公式 `sample_submission/`）/ server 構築 / スコア算出 /
**task 別の不出来を `results/failures.csv`・`failures.md` に落とす**所までが 1 冊です（`src/pipeline/` は読みません）。
Kaggle を触らない資材経路は同じ notebook の cell 2 を `MATERIAL_SOURCE = "hf"` にするか `make colab-notebook-hf`。
Kaggle 側で測るなら `kernels/measure/gemma4_l4_measure.ipynb`（`make measure-notebook`）。対応表は `doc/colab.md` §13。

## 構成

```
submission/                 # ← これが提出物。ここだけを zip する
├── agent.yaml              # root agent（単一 model 宣言, tools 9 個 + agent_tool）
├── eval_config.yaml        # budget（evaluation: key 必須）
├── configs/sampling.yaml   # thinking_config ネスティング済み
├── prompts/system.md       # 短軌跡・loop 防止・patch 衛生の命令文
├── prompts/analyzer.md
├── sub_agents/code_analyzer.yaml   # 読み取り専用 explorer（skip_summarization）
└── adapters/               # LoRA が入る場所（README.md あり / 未使用なら空）

src/
├── pipeline/                      # ノートブックの各セル→1ステージ1モジュール（run_pipeline.py が入口）
│   ├── stage_{setup,data,agent,graph,server,eval,package,patch,analyze}.py
│   └── variants.yaml              # 比較の単位。variant 名 = run_name
├── collect/collect_trajectories.py  # tasks.jsonl(gold patch+test_patch) から SFT 軌跡を合成
├── train/train_lora.py              # LoRA（assistant turn のみ mask, safetensors 出力）
├── evaluate/evaluate_harness.py     # contract / patch 静的検査 / swegemma eval ドライバ
├── analyze/analyze_results.py       # results/ から失敗分類（NO_PATCH / apply 失敗 / 保護污染）
└── package/build_submission.py      # 提出仕様のバリデーション + zip 化

kernels/
├── eval/   gemma4_e4b_eval.py + kernel-metadata.json   # Phase1+2 を Kaggle GPU で再現
└── train/  gemma4_e4b_lora_train.py + kernel-metadata.json

doc/                        # 使い方と設計判断（kaggle_cli / limits / watchpoints / multi_agent / lora / tasks）
.kaggle/                    # dataset アップロードの staging（Makefile が src/・submission/ を写す）
data/                       # tasks.jsonl, trajectories.jsonl
```

## 最初の一歩（生の Kaggle コマンドが主語。`make` は同じ事を並べた省略可能な便利版）

```bash
# 手元（GPU 無し）: 提出仕様に通るか / variant が適用されるか
python3 src/pipeline/run_pipeline.py --list-variants
python3 src/pipeline/run_pipeline.py --variant baseline --stages local

# Kaggle GPU で測る（variant を kernel に焼き込んでから push）
python3 src/pipeline/set_kernel_variant.py smoke
kaggle datasets version -p .kaggle/dataset-code       -m "sync src/"       -r tar -q
kaggle datasets version -p .kaggle/dataset-submission -m "sync submission/" -r tar -q
python3 src/pipeline/make_kernel_notebook.py --source kernels/eval/gemma4_e4b_eval.py \
  --title gemma4-e4b-agent-eval --code-slug gemma4-dev-agent-code --metadata kernels/eval/kernel-metadata.json
kaggle kernels push -p kernels/eval

# 実行中はログを直接見る（status を繰り返さない）
kaggle kernels logs <username>/gemma4-e4b-agent-eval --follow
kaggle kernels output <username>/gemma4-e4b-agent-eval -p results/smoke -o -q
python3 src/pipeline/run_pipeline.py --variant smoke --stages post

# 公式提出と quota
kaggle competitions submit      -c gemma-4-developer-agent -f submission.zip -m "baseline e4b"
kaggle competitions submissions -c gemma-4-developer-agent
kaggle quota
```

全コマンドの対応表は **`doc/kaggle_cli.md`**、stage と variant の話は **`doc/pipeline.md`**。

タスク一覧と完了条件は **`doc/tasks.md`**。ハーネス側の見方は **`doc/harness_watchpoints.md`**。

## 手元で済ませるデータ取得

```bash
kaggle competitions download -c gemma-4-developer-agent -f tasks.jsonl -p data
kaggle competitions download -c gemma-4-developer-agent -f HARNESS_README.md -p data
```
（snapshots / graphs / embeddings / wheels は合計で数十 GB。評価は Kaggle kernel で走らせる。）
