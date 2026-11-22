# Gemma 4 Developer Agent Competition (2026)

Kaggle コンペティション **`gemma-4-developer-agent`** 向けのエージェント開発・LoRA 学習・評価用リポジトリです。  
提出物（`submission.zip`）として、Gemma 4 e4b ベースのエージェント定義（YAML・プロンプト・Multi-LoRA アダプタ）を作成し、Colab 上での検証および RFT / DPO による LoRA 学習を行います。

## フォルダ概要

```text
.
├── submission/             # 提出物本体（ここを validation / zip 化して提出）
│   ├── agent.yaml          # ルートエージェント定義（main_lora）
│   ├── eval_config.yaml    # タスクあたりの時間・ターン・tool call 予算
│   ├── configs/            # 生成・思考予算パラメータ（sampling / verifier_sampling）
│   ├── prompts/            # システムプロンプト（system / analyzer / verifier）
│   ├── sub_agents/         # サブエージェント定義（code_analyzer / verifier）
│   └── adapters/           # 学習済み LoRA アダプタ配置先（*.safetensors）
├── src/
│   ├── collect/            # 学習データ生成（gold 合成・RFT 軌跡・DPO 選好ペア）
│   ├── train/              # LoRA 学習（SFT/RFT: train_lora.py、DPO: train_dpo.py）
│   ├── evaluate/           # ローカル事前検査・評価ハーネス連携
│   ├── analyze/            # 評価結果の失敗分類・分析
│   ├── package/            # 提出契約のバリデーション & submission.zip 生成
│   └── pipeline/           # Colab ノートブック生成・結果レポート・計測ユーティリティ
├── kernels/
│   └── colab/              # Colab 実行用ノートブック（評価用・LoRA 学習用）
├── doc/                    # 設計メモ・制限事項・運用ドキュメント
├── scripts/                # 補助スクリプト
├── data/                   # tasks.jsonl 等の入力データ（git 管理外）
└── results/                # 評価ログ・失敗レポート出力先（git 管理外）
```
