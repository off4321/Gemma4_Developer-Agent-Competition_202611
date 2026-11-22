# doc/

このディレクトリは **Gemma 4 Developer Agent コンペを回すための運用リファレンス** です。
実装は `submission/`（提出物）と `src/`（パイプライン）、`kernels/`（Kaggle 実行用スクリプト）。

| ファイル | 内容 | いつ読むか |
|---|---|---|
| `kaggle_cli.md` | Kaggle kernel の push / 状況確認 / ログ / 停止 / output 取得、dataset & model 添付、quota | 初めて kernel を回すとき、動かないとき |
| `environment_limits.md` | 採点環境（4×L4）と Kaggle notebook（T4×2）の違い、モデル容量、タイム/ディスク制限 | 「4×L4 と同じ環境にできるの？」に対する回答 |
| `colab.md` | Colab L4 を 2 台目の実測場として立てる手順（認証 / 必要なものだけ入れる / stage 回し / 落とし穴） | Kaggle の T4 壁で詰まったとき、sm89 で同じ条件を確かめたいとき |
| `verification.md` | 検証の三段（公式提出 / 公式 harness 局所測定 / 自分の harness）、`parity_check.py` と `scoreboard.py` の使い方、逸脱の申告先 | 「自分のハーネスの検証はどうする？」・「zip を送った方がいい？」に対する回答 |
| `harness_watchpoints.md` | ハーネス側の観測点：ループ防止、tool_call の正しさ、budget、patch の抽出/適用、保護ファイル | スコアが伸びないときの原因切り分け |
| `multi_agent.md` | sub_agents / agent_tool / Sequential・Loop・Parallel の使い分けと落とし穴 | エージェント構成を変えるとき |
| `lora_plan.md` | LoRA で何を改善するか（tool_call 精度・ループ防止）、データ、VRAM、検証手順、落とし穴 | LoRA に着手するとき |
| `tasks.md` | 優先順位付きタスク一覧（受入条件つき） | 何から着手すべきか迷ったとき |

## 最短経路（コマンド）

```bash
make validate                      # submission/ が提出仕様に合致するか
make pipeline-local VARIANT=baseline  # src/pipeline preflight（GPU 無しで契約+variant）
make push-code                     # src/（pipeline stage 込み）を dataset に上げる
make push-submission               # submission/ を Kaggle dataset に上げる
make pipeline-eval VARIANT=smoke   # kernels/eval を push して実行開始（GPU quota 消費）
make logs-follow                   # ライブログ
make status                        # 実行状況
make out-eval RUN=smoke            # results/smoke に artifacts を落とした解析
make pipeline-post VARIANT=smoke   # 失敗分類（no patch / apply 失敗 / 保護ファイル污染）
make push-train                    # LoRA 学習 kernel
make out-train                     # adapter 取得
# adapter を submission/adapters/main_lora に置いてから
make zip && make submit
```
