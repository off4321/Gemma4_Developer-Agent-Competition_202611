# 検証の三段（自分の harness / 変更をどう信用するか）

「自分で作ったハーネスの検証」は **公式 harness との突き合わせ**で測ります。
`submission.zip` の提出は**公式スコアを取る行為**であって、検証の代替にはなりません
（1 提出 = 数字 1 個、task 単位の内訳は見えません）。

## 三段と答えられる質問

| 段 | 何を使うか | 答えられる | 答えられない |
|---|---|---|---|
| **A. 公式提出** | `kaggle competitions submit -f submission.zip` | 公式 Resolution Rate（非公開 256 task / 4×L4 / 公式 harness / 公式 sandbox） | 内訳。失敗理由。提出回数にも限りがある |
| **B. 公式 harness 局所測定**（既定） | wheelhouse pin の `swegemma` + `adk_submission` を Colab L4 / Kaggle で実行（`kernels/colab/gemma4_colab.ipynb`） | task 単位 resolved・patch size・test exit code・tool calls・失敗 taxonomy・variant 間比較 | 公式スコアそのもの（task 集合と GPU が違う） |
| **C. 自分の harness** | `src/pipeline` の eval / 自作検査（patch 静的検査・log grep） | 速い反復・log を見た解析・低コストの前筛查し | 公式と同じ結果である**保証**。`parity_check` を通すまで信用しない |

**既定は B**です。A は変更が固まった節目だけ。C は B と突き合わせてから使う補助道具です。

## 逸脱は flag と note で申告する（黙ってやらない）

| 逸脱 | 入口 | 申告先 |
|---|---|---|
| vLLM version を測定の為に上げる（T4 で 0.19.1 が shared memory で死ぬ） | `variants.yaml` の `vllm_version` / `--vllm-version` | run manifest + `scoreboard --note` |
| sandbox が `subprocess`（Colab に docker daemon が無い） | `--sandbox subprocess` | manifest + note |
| task 部分集合（12 / 6 / 3 / 1 個） | `task_limit` / `task_ids` / `TASK_IDS` | scoreboard の `tasks` 列 |
| GPU 構成（1×L4 tp=1 / 2×T4 tp=2 fp16 / 公式 4×L4 tp=4 bf16 0.90） | `stage_server` が compute capability で決める | manifest の deviation |
| 自分の pipeline を使う | `--stages ... --variant ...` | scoreboard の `harness=own-pipeline` |

## 検証の手順（1 変更 1 測定）

```bash
# 1) variant を 1 個だけ変えて測る（variants.yaml で比較単位を固定する）
python3 src/pipeline/run_pipeline.py --variant colab12 --stages gpu

# 2) 公式 harness 出力と自分の pipeline 出力を突き合わせる（exit 0 = 乖離なし）
python3 src/pipeline/failures_report.py --results results/<run>   # 先に「どれが出来ていないか」を出す
python3 src/pipeline/parity_check.py \
    --a /content/repo/work/results/colab12 \
    --b results/colab12 \
    --label 公式harness 自分のpipeline \
    --note "sandbox subprocess / task 12 個" --json results/parity_colab12.json

# 3) 台帳に積む
python3 src/pipeline/scoreboard.py --add --name colab12 --harness official-pin --env colab-l4 \
    --results /content/repo/work/results/colab12 --note "公式 pin vllm 0.19.1 / subprocess sandbox"
python3 src/pipeline/scoreboard.py --add --name colab12_mine --harness own-pipeline --env colab-l4 \
    --results results/colab12

# 4) 節目で提出して公式スコアを台帳に還流させる
kaggle competitions submit -c gemma-4-developer-agent -f submission.zip -m "prompt v2 + budget 20min"
kaggle competitions submissions -c gemma-4-developer-agent --csv > results/official_submissions.csv
python3 src/pipeline/scoreboard.py --official-csv results/official_submissions.csv
python3 src/pipeline/scoreboard.py --table
```

`parity_check` の exit code は **0 = 食い違い 0 / 1 = 食い違いあり**なので、CI 的に使えます。

## 台帳で見るもの（校正）

`--table` は measured 行と official 行を同じ表に並べ、`best official` との差を出します。
目的は一致させることではなく **ズレの符号と大きさを知る**こと:

- 同じ `submission.zip` に対する measured rate と公式スコアの差が +0.1 なら、以降の局所測定は「公式 + 0.1 付近」と読んで比較に使う
- measured が上がって公式が下がったら、測っている task 集合が公式の hidden set と違う（overfit の兆候）
- harness を上げた（vLLM 0.28.0 等）測定は**絶対値を信用しない**。同じ harness 内で前後比較だけする

## 測ってはいけない物・気をつける物

- `data/tasks.jsonl` には gold `patch` と `test_patch` が両方入っている。**agent に見せてはいけない**（gold 解析は `stage_patch` だけが見る）。agent 側に出すのは `problem_statement` と workspace だけ
- 公開 129 task で上げたスコアは hidden 256 の予測にならない。variant 比較（A/B）にだけ使い、絶対値は公式に聞く
- protected path（`conftest.py`, `pytest.ini`, `pyproject.toml`, `setup.cfg`, `tox.ini`, `sitecustomize.py`, `usercustomize.py`, `_swegemma_stubs.py`）を patch が触っていたら `stage_patch` が弾く。公式でも同じ理由で落ちる
- 単一 model 制約（`configs/sampling.yaml` に model 1 個）と adapter 名規則 `[A-Za-z0-9][A-Za-z0-9._-]*` は提出前に cell 4 / cell 5 の assert で落とす

## 一言で

**既定は公式 harness（wheelhouse pin）。逸脱は flag で可視化し、`parity_check` で乖離ゼロを確認し、`scoreboard` で公式スコアと較正する。`submission.zip` は「測る道具」ではなく「公式の答えを取る道具」。**

## 不出来を読む file（B / C 共通）

測定後は必ず `results/<run>/failures.md` を読んでください（`analyze` stage が書く。Colab notebook の cell 7 も同じ実装）。
status 集計が `no_patch` 偏重なら budget / tool 側、`patch_fail` 偏重なら patch の質、`crashed` が 1 個でもあると
harness・loader・sandbox 側の bug を先に潰します。`not_run` が有る run は**測れていない task が残っている**ので、
scoreboard に載せるとき note に task 数を書いておくこと。詳細は `doc/pipeline.md` の analyze 節。
