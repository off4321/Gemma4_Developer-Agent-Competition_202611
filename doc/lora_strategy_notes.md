# LoRA・事後学習 検討メモ（4B モデルの工夫と限界）

既存の [lora_plan.md](lora_plan.md)（SFT パイプラインの基本手順）を補完する、戦略・手法の検討メモです。

---

## 1. 課題パターン別：LoRA で直せるもの／別の工夫が要るもの

| 観察されたパターン | LoRA の効き目 | 現実的なアプローチ |
|---|---|---|
| **ループしている**（同じエラー・コマンドの反復） | **◎ 効きやすい** | 「エラー後に別手段へ切り替える」「早めに `submit_patch` に向かう」軌跡の学習、または **DPO/ORPO** でループ軌跡を Rejected にする |
| **テストが上手くできていない** | **○〜◎ 効きやすい** | `pytest path::test_name` への絞り込み・編集→テスト→提出の型を軌跡で定着させる |
| **トークンを使いすぎる** | **○（設定併用）** | LoRA で簡潔な思考の軌跡を学ばせつつ、即効性の高い `sampling.yaml` の `thinking_budget` / `max_output_tokens` 調整とプロンプト制約を併用する |
| **表面的しか見ずに実行している** | **△（手順のみ）** | 「編集前に `code_analyzer` や `get_code_neighbors` で関連箇所を確認する」手順は矯正可。深い仕様理解そのものは 4B の基礎能力に依存 |
| **難しいタスクが解けない** | **△〜×（限界あり）** | 推論上限を超える問題は LoRA だけでは難しい。コンテキスト分離（探索をサブエージェントへ逃がし親の 32k を綺麗に保つ）等で取りこぼしを減らす |

**基本方針**: まず実測（`failures.md`）で `no_patch` / `apply_fail` / `patch_fail` の比率を見て、漏れている穴に対してプロンプト・予算調整と LoRA を使い分ける。

---

## 2. 巨大教師モデルに頼らず 4B を伸ばす事後学習・アダプタ手法

### 2.1 RFT（Rejection Sampling Fine-Tuning / 自己成功軌跡の学習）
- `temperature` をやや上げて（例: `0.6〜0.8`）同じタスクを複数回試行（Pass@K）させ、**自力で解けた軌跡（`resolved=True`）だけ**を回収して SFT する。
- 4B 自身の自然な文体・思考パターンのまま Pass@1（1発成功率）を底上げでき、外部の強モデルが不要。

### 2.2 DPO / ORPO（成功 vs 失敗の選好学習）
- 同一タスクで以下をペアにして選好学習を行う：
  - **Chosen**: 少ない手数で原因特定 → `pytest` パス → `submit_patch`
  - **Rejected**: 同じコマンドをループ / 全ファイル書き換え / `NO_PATCH` 終了
- SFT だけでは抑えにくい**「ループ」「無駄な探索」「パッチ未提出」を直接罰して抑制**できる。
- `trl` の `DPOTrainer` / `ORPOTrainer` で学習しても出力は**標準の PEFT LoRA（`.safetensors`）と同一形式**になるため、vLLM でそのまま動作する。

### 2.3 役割別 Multi-LoRA（工程分割）
`submission/agent.yaml` と `submission/sub_agents/code_analyzer.yaml` はそれぞれ独立した `adapter:` を指定できる：
- `analyzer_lora`: `get_code_neighbors` / `search_similar_code` / `read_file` で原因箇所と行番号を特定し、短い構造化レポートだけを返す専門
- `editor_lora`: 報告を受けて正確な `edit_file` とピンポイント `pytest` を行い `submit_patch` する専門

※ ただし vLLM では `max_loras=len(adapters)` となり VRAM とロード時間がわずかに増えるため、まずは 1 本から検証し、効果が確認できてから分割する。

### 2.4 合成データへの「失敗 → リカバリ」注入
`src/collect/collect_trajectories.py` の gold patch 合成に、疑似的な 1 回目の失敗（例: `edit_file` の不一致や `pytest` 失敗）とそれを受けて修正するステップを混ぜることで、教師モデルなしでもリカバリ手順を学習させる余地がある。

---

## 3. PEFT バリアントと本番環境（vLLM + QAT）の相性

本番ハーネスは `gemma-4-e4b-it-qat-w4a16-ct` を vLLM（`0.19.1`）でサーブする前提のため、以下の互換性に注意する：

| 手法 | 特徴 | 本コンペでの扱い |
|---|---|---|
| **LoRA+** | 行列 $A$ と $B$ で学習率を変える（$B$ 側を上げる） | **◎ 安全**：学習時の最適化だけの違いで、保存ファイルは標準 LoRA と完全同一 |
| **RSLoRA** | スケーリングを $1/\sqrt{r}$ にして高ランク（`r=64..128`）を安定化 | **○ 要動作確認**：`r <= 128` の範囲で有効。vLLM `0.19.1` での読み込みを事前確認する |
| **DoRA** | 重みの方向と大きさを分離 | **△ 注意**：追加ベクトルを持つため vLLM / QAT 環境で非対応になる恐れあり |
| **PiSSA** | ベース重みの SVD 初期化（残差重みも変更） | **× 不向き**：固定の運営ベースモデルに差し込む形式と噛み合わない |

---

## 4. 事後学習（DPO 等）のコンペ上の扱いと確認事項

- **技術・提出仕様面**: 提出物は `submission.zip` 内の `adapters/<name>/adapter_config.json` + `adapter_model.safetensors`（`r <= 128`、pickle 禁止）であり、SFT・RFT・DPO・ORPO いずれで作った標準 LoRA でも提出契約上は受け付けられる。
- **規約面（要確認）**: 本リポジトリには Kaggle コンペ公式 Rules 本文は含まれていないため、**外部データセットや外部モデルの出力を学習データに混ぜる場合**は、Kaggle コンペページの Rules（External Data / Model Eligibility）で許可範囲を必ず確認すること（自分の 4B モデルの実行ログだけで RFT / DPO を回す場合は外部モデル依存がない）。
