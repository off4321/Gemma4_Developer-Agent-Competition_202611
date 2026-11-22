# Colab で測る（L4 / 公式 env に一番近い測定場）

`doc/kaggle_cli.md` が「公式採点に提出する」側の手順、この文書は **手元実験場として Colab を使う**側の手順。
実測は 2026-11-21（`colab-cli 0.8.0`, `google-colab-cli`）。skill 版は `/root/.pi/agent/skills/colab-runner/`。

## 0. なぜ Colab を 2 台目にするか

| | Kaggle kernel | Colab L4 | 公式採点 |
|---|---|---|---|
| GPU | 2× Tesla T4 (sm75) | **1× NVIDIA L4 (sm89)** | **4× L4 (sm89), tp=4** |
| dtype | float16 強制 | **bfloat16** | bfloat16 |
| Triton attention (Gemma4) | **共有 memory 64 KB < 必要 96 KB で生成 0 token**（doc/environment_limits.md §5） | 通る（実測で生成 OK） | 通る |
| wheelhouse の vLLM pin 0.19.1 | ABI / cp313 tag で载せ替えが必要 | **pin のまま通る**（torch pin 修理のみ） | pin |
| python | 3.13 | 3.13 | ( scoring 側 image ) |

→ **同じ model・同じ harness・同じ vLLM pin で測れる実測環境**として Colab が使える。差は GPU 枚数（tp=1）。

## 1. 認証（初回だけ、1 回人間の認可が要る）

```bash
UV=$(dirname $(readlink -f $(which colab)))/python
uv pip install --python "$UV" keyring keyrings.alt           # headless は keyring backend が無い
tmux new-session -d -s colabauth 'PYTHON_KEYRING_BACKEND=keyrings.alt.file.PlaintextKeyring \
  colab --auth=oauth2 sessions > /tmp/colab_auth.out 2>&1 < /dev/tty; sleep 600'
grep -o 'https://accounts.google.com[^ ]*' /tmp/colab_auth.out | tail -1     # この URL を人間が開く
tmux send-keys -t colabauth -l "<認可コード>"; tmux send-keys -t colabauth Enter
colab sessions                                               # "No active sessions found on server." = 成功
```
**URL はファイルからそのまま渡す**（`code_challenge` / `state` を書き換えると PKCE が壊れて失敗する。実測で一度やった）。
`--auth=adc` が本来は headless に強いが `gcloud` が要る（この環境に無い）。

## 2. session を立てて、必要なものだけ入れる

```bash
colab usage                                  # compute units の残高（借りる前に見る）
colab new -s g4l4b --gpu L4
colab upload /tmp/repo.tgz /content/repo.tgz -s g4l4b                      # src + submission (~250 KB)
colab exec   -s g4l4b -f src/pipeline/colab_stage.py --env STAGES=...      # 前段で mkdir /content/repo
colab upload ~/.kaggle/credentials.json /root/.kaggle/credentials.json -s g4l4b   # mkdir 後でないと 500
colab exec   -s g4l4b -f src/pipeline/colab_prepare_env.py --timeout 900   # kaggle CLI 修理 + 缺口 package
```

### 入れるもの（lean: 合計 ~11 GB）

| 物 | size | 出所 | 何に要る |
|---|---|---|---|
| `data/tasks.jsonl`, `HARNESS_README.md` | 2 MB | `kaggle competitions download -c <comp> -f <file>` | task / 契約 |
| `data/graphs/*.json` (127) + `data/embeddings/*.npz` (127) | 1.3 GB | 同上（1 個ずつ） | graph stage の retrieval |
| `data/snapshots/<instance_id>.tgz` **選んだ task だけ** | 12 個で ~2.4 GB | 同上 | **Phase 2 verification**（`extract_snapshot`） |
| `data/wheelhouse/*.whl` | 867 MB | `kaggle datasets download -d metric/gemma-4-developer-agent-wheelhouse --unzip` | harness pin |
| model | 7.3 GB tar → 11.5 GB | `kaggle models instances versions download <...> --untar` | 推論 |

`kaggle competitions download -c <comp> -p data`（**zip 全部 21.9 GB**）は使わない。
`snapshots/` 全部は 21.3 GB で、測定には要らない。`colab_bootstrap.py` は `TASK_LIMIT` / `TASK_IDS`
で stage の選別（`tasks[:limit]`）と同じ先頭 N 個だけ落とす。

## 3. stage を回す（1 exec に長時間を置かない）

```bash
colab exec -s g4l4b -f src/pipeline/colab_stage.py --env STAGES=setup --env VARIANT=colab --timeout 300
colab exec -s g4l4b -f src/pipeline/colab_stage.py --env STAGES=data,agent,graph,server,eval,package --env VARIANT=colab
# 読み取り（exec が落ちても通る）
colab download /content/stage_setup.log /tmp/stage_setup.log -s g4l4b
colab log -s g4l4b -n 40                       # CLI が叩いた接続の履歴
colab log -s g4l4b -n 60 -o /tmp/session.ipynb # kernel の出力込みで export（真因探索に強い）
```

## 4. 実測で分かった Colab 固有の落とし穴

| 現象 | 対処 |
|---|---|
| `RuntimeError: Connection was lost.`（VM が混む / 短時間に接続を積みると） | **同じコマンドを再試行**（実測で 2-3 回目で通る）。読み取りは `colab download` |
| `TimeoutError: no message from the kernel for 30s` | `--timeout <秒>` を上げる。長い download は detached process + log ファイル |
| preinstall `kaggle` が import 段階で死ぬ | `pip install -U kaggle`（`colab_prepare_env.py`） |
| `Missing key in configuration.` | `~/.kaggle/kaggle.json` を credentials.json から作る + 600 |
| `base_api_transform.py: import jmespath` | `jmespath` 追加（MHCS 0.1.16 の deps が image に入っていない） |
| `[graph] No module named 'docker'` | `docker` 追加 |
| `[server] No module named 'litellm'` | `litellm` 追加 |
| `(EngineCore) No module named 'blake3'` | `blake3`（+ `ninja`, `mcp`, `flashinfer-python==0.6.6`, `flashinfer-cubin==0.6.6`, `nvidia-cudnn-frontend<1.19`, `nvidia-cutlass-dsl`） |
| Kaggle model download が tar.gz 1 個で返す | `--untar`、無ければ手動 `tar xzf`。`model.safetensors` を stage の見る場所に move |
| competition download が `data/<comp>/` に展開する | `data/` 直下に上げる（`flatten_competition`） |
| heartbeat | tmux で 150 s 毎に `colab status`。session は生存した |
| `Status: IDLE` でも課金は進む | `colab usage` の rate（実測 1.94 units/h）。使わないなら `colab stop -s <name>` |

## 5. 実測結果（2026-11-21, variant `colab` = 3 task）

```
[server] gpus=1 [NVIDIA L4] sm89 tp=1 dtype=bfloat16 gpu_mem=0.85
[smoke]  completion OK (200) {"id":"chatcmpl-..."}      <- 生成が動く
[server] vLLM up at http://127.0.0.1:8000/v1 (tp=1)
[pipeline] data(0.0s), agent(0.0s), graph(6.8s), server(16.9s), eval(397.9s), package(0.0s)
[pipeline] resolved=0/3 patches=3       patch_chars=944/956  tool_calls=4/16
```
0/3 の原因は harness ではなく自分のコード: `stage_data.py` が **Kaggle mount の有無**で task loader を
分岐させ、Colab で `SimpleNamespace` フォールダを通した。`swegemma/harness/verification.py:431` は
`task.test_files` を素で読むため `AttributeError`。→ loader を「swegemma が import できるか」で選ぶように変更済み。

副次: `swegemma.sandbox.subprocess WARNING ensurepip unavailable in sandbox ...`（Phase 2 の venv に pip が入らない）。
test が pip を要求する task では効く。未確認。

## 6. Kaggle の credential は 12 h で切れる（実測 2026-11-21）

`~/.kaggle/credentials.json` は OAuth 型で `access_token_expiration` を持つ（この環境では 02:17 発行 -> **14:17 切れ**た）。
切れると CLI はファイルがあっても中身を使わず
`Option B: API token file / Save the token to ~/.kaggle/access_token`
を出して全 API 呼び（datasets / kernels / competitions）が失敗する。Colab 側に upload 済みの file も **同じ時刻で無効化される**（fetch が `tasks.jsonl` 1 個目で死んだ真因はこれ。session の creds 欠落ではなかった）。

```bash
python3 - <<'EOF'
import json, pathlib, datetime
c = json.loads(pathlib.Path.home().joinpath(".kaggle/credentials.json").read_text())
print(c.get("username"), c.get("access_token_expiration"), datetime.datetime.now(datetime.timezone.utc))
EOF
```
長い fetch / run を組む前に期限を見る。`colab_kalive.py` は `CREDS` / `KAGGLE_OK` を出す（期限までは見ない）。
更新は Kaggle 側で再認証（設定画面の API token を `~/.kaggle/access_token` に置くのが一番素直）。

## 7. セル実行ノートブック 2 種（Kaggle 版 / Colab 版）

| | Kaggle 版 | Colab 版 |
|---|---|---|
| ファイル | `kernels/measure/gemma4_l4_measure.ipynb` | `kernels/colab/gemma4_colab.ipynb` |
| 生成元 | `src/pipeline/make_measure_notebook.py` | `src/pipeline/make_colab_notebook.py` |
| GPU | 2×T4 (sm75) -> wheelhouse pin が Gemma4 を生成できない | **1×L4 (sm89) -> pin のまま生成 OK（実測済み）** |
| 資材 | mount（competition / wheelhouse / model / code dataset） | **Kaggle から落として `/content/repo` に同じ配置**（cell 3） |
| 既定 variant | `kaggle_nb`（6 task × 20 min、vLLM 0.28.0 測定用） | `colab12`（12 task × 15 min、**wheelhouse pin**） |
| stage 呼び出し | 同一（`stage_setup / data / agent / graph / server / eval / package / patch / analyze`） | 同一 |

手順（Colab 版）:

```bash
python3 src/pipeline/make_colab_notebook.py                       # notebook を作る
colab new -s g4l4d --gpu L4
colab upload kernels/colab/gemma4_colab.ipynb /content/gemma4_colab.ipynb -s g4l4d
# A) Colab 画面で 1 個ずつ回す（推奨。cell 9 の server を cell 10 が使うので kernel を restart しない）
# B) CLI でまとめて回す（1 kernel で実行するので ctx / server が続く）
colab exec -s g4l4d -f src/pipeline/colab_run_notebook.py --timeout 200
colab download /content/notebook_run.log /tmp/nb_run.log -s g4l4d
colab download /content/gemma4_colab_measure_executed.ipynb /tmp/executed.ipynb -s g4l4d
```

cell 対応（両 notebook 共通）:

| cell | 内容 | 資材 |
|---|---|---|
| 1 | 資材/認証の検査（Colab 版は creds + kaggle CLI 修理） | - |
| 2 | variant / stage 本体（Colab 版は code dataset を download） | code dataset |
| 3 | 資材を必要な分だけ download（Colab 版のみ） | tasks/graphs/embeddings/snapshots/wheelhouse/model |
| 4 | 環境の缺口を補う（Colab 版のみ: jmespath docker litellm blake3 ninja mcp flashinfer…） | wheelhouse |
| 5 | variant 選択 -> `RunConfig` + `Context` | variants.yaml |
| 6 | `setup`: wheelhouse pin + torch pin | wheelhouse |
| 7 | `data`（harness loader）+ `agent`（staging・契約検査） | tasks.jsonl, submission |
| 8 | `graph` + embeddings | graphs/, embeddings/ |
| 9 | `server`（`VllmServer`）+ 1 文生成 | model |
| 10 | `eval`: Phase 1 + Phase 2 | snapshots/, staged eval_config |
| 11 | `package`（validate 込み）+ `patch` + `analyze` | agent_dir |
| 12 | manifest / deviation / 物を下ろす | 全部 |

注意:
- Colab 版 cell 2 は dataset `<you>/gemma4-dev-agent-code` を読む。**dataset が古いと直した `stage_data.py` が見えない**ので先に `kaggle datasets version -p .kaggle/dataset-code -m "sync src/" -r tar -q`。
- 認証 token は §6 のとおり 12 h で切れる。長い job 組む前に `kaggle competitions files -c gemma-4-developer-agent --page-size 3` で通るか見る。
- GPU は 1 枚（tp=1）。公式は 4 枚（tp=4）なので、速度と KV cache の余裕が違う（manifest の `hardware` に残る）。

## 8. Kaggle を**実行中に触らない** Colab 版（`kernels/colab/gemma4_colab_hf.ipynb`）

生成元 `src/pipeline/make_colab_notebook.py`（`make colab-notebook-hf`）。30 cells / code 15。
資材の出所を Kaggle mount から Hugging Face に置き換えた版で、実行中に Kaggle API / CLI を使いません。

| 物 | 出所 | 確かめた値 |
|---|---|---|
| competition 資材 | HF dataset **`HanG/gemma-4-developer-agent`** の `gemma-4-developer-agent.zip` | 21,923,985,083 B（Kaggle の competition download と同じ） |
| harness wheelhouse | 同 repo の `gemma-4-developer-agent-wheelhouse-v30.zip` | 880,849,882 B・42 entries。中央 directory を読んだ一覧: `swegemma-0.2.11`, `adk_submission-0.2.13`, `adk_eval_core-0.1.0`, `google_adk-1.36.1`, `vllm-0.19.1-cp38-abi3-manylinux_2_31_x86_64`, `transformers-5.13.1`, `tokenizers-0.22.2`, `safetensors-0.8.0`, `model_hosting_container_standards-0.1.16`, `compressed_tensors-0.15.0.1`, `flashinfer_python/cubin-0.6.6`, `quack_kernels-0.6.4`, `nvidia_cudnn_frontend-1.18.0`, `xgrammar-0.2.5.post1`, `outlines_core-0.2.11`, `llguidance-1.3.0`, `bitsandbytes-0.46.1` … **公式 pin と名前は一致** |
| model | HF **`google/gemma-4-E4B-it-qat-w4a16-ct`**（gated=False, apache-2.0） | `model.safetensors` の content-length 11,513,861,828 B = Kaggle model instance と同一サイズ。ファイル構成も同一（README / chat_template.jinja / config.json / generation_config.json / model.safetensors / processor_config.json / tokenizer.json / tokenizer_config.json） |
| 追加 package | PyPI（jmespath, docker, litellm, blake3, ninja, mcp, cudnn-frontend, cutlass-dsl, flashinfer） | 実測で必要だった |
| stage 本体 | 手元から `repo.tgz`（`tar czf /tmp/repo.tgz src submission`、~250 KB） | Kaggle 経由しない唯一の手作業 |

重要: **`swegemma` / `adk_submission` は PyPI に無い**（実測 `pip index versions swegemma` -> `No matching distribution found`）。
harness は wheelhouse の wheel から offline install するしかない。なので Kaggle 完全回避は「**ミラーを 1 回 HF から読む**」形で実現している。
`swegemma/submission.py` は Kaggle path を直書きしています（`/kaggle/input/competitions/gemma-4-developer-agent`,
`/kaggle/input/datasets/metric/swegemma-evaluation`）が、こちらの `Paths.detect(data_dir=...)` で上書きするので Colab で動く。

正直な注意:
- HF mirror は第三者のもので、**byte 一致は未検証**（名前は一致）。測定には使えるが、提出物の評価は公式資材で公式側がやる
- zip 全部（21.9 GB）は展開しない。`tasks.jsonl` を先に読んで**測る task の snapshot だけ**取り出す（`snapshots/` 全部で 21.3 GB）
- 2 回目以降は cell 3 の Drive cache（`/content/drive/MyDrive/gemma4`）で download を省ける
- GPU 1 枚（tp=1, bfloat16, 0.85）。公式は 4 枚（tp=4, 0.90）

## 9. Kaggle token を env に入れると Colab でも Kaggle 経路が通る（CLI 2.2.5 の実装を読んだ）

`kaggle/api/kaggle_api_extended.py` の `authenticate()` と `kagglesdk.get_access_token_from_env()` の順序:

| 順 | 出所 | 中身 | 切れ方 |
|---|---|---|---|
| 1 | env **`KAGGLE_API_TOKEN`** | token 文字列そのものでも可、**token を入れた file の path でも可**（`Path(...).exists()` を見る） | settings UI の token は長命 |
| 1 | `~/.kaggle/access_token` / `~/.kaggle/access_token.txt` | file 1 行 | 長命 |
| 2 | `~/.kaggle/kaggle.json`（username + key）/ env `KAGGLE_USERNAME` + `KAGGLE_KEY` | legacy API key。`read_config_environment` は `KAGGLE_` prefix の env を全部 config に入れる | legacy |
| 3 | `~/.kaggle/credentials.json`（`kaggle login` の OAuth） | access + refresh token | **12 h で切れる（§6 で実測）** |
| 4 | 匿名 | 公開 asset だけ | - |

invalid / 期限切れの access token は `_introspect_token` で弾かれて**無視され次の経路に進む**。
Kaggle kernel 内では `KAGGLE_API_V1_TOKEN_PATH` が最優先（Colab には無い）。

Colab で使うときの 3 つの実務:

1. `!export KAGGLE_API_TOKEN=...` は **次の cell に残らない**（`!` は shell 1 回ずつ）。
   `os.environ["KAGGLE_API_TOKEN"] = ...` を同じ kernel 内でセットすると、その後の `subprocess` 経由の kaggle に伝わる
2. Colab の **Secrets**（`google.colab.userdata.get("KAGGLE_API_TOKEN")`）に入れると `.ipynb` に token が残らない
3. colab CLI で流すなら **token を file で上げて env はその path** にする（argv に secret を出さない）:
   ```bash
   colab upload ~/.kaggle/access_token /root/.kaggle/access_token -s <session>
   colab exec -s <session> --env KAGGLE_API_TOKEN=/root/.kaggle/access_token -f src/pipeline/colab_fetch.py --env TASK_LIMIT=12
   ```

ノートブック側の切り替え（`kernels/colab/gemma4_colab_hf.ipynb` cell 1）:

```python
MATERIAL_SOURCE = "hf"       # mirror。token 不要
MATERIAL_SOURCE = "kaggle"   # 公式資材（competition / metric/...-wheelhouse / Kaggle model instance）
```

`"kaggle"` にすると cell 2 は `src/pipeline/colab_bootstrap.py` を呼ぶだけ（実測で Colab L4 に tasks / graphs /
embeddings / 選んだ snapshots / wheelhouse / model が揃う）。配置は `"hf"` と同一（`/content/repo/data`、
`/content/repo/data/wheelhouse`、`/content/repo/models/google/gemma-4/other/<model>/2`）なので cell 4 以降は同じ。
Colab image の preinstall `kaggle` は import 段階で死ぬので、cell 1 が `pip install -q -U kaggle` を自動でやる。

## 10. Colab **Secrets** の `KAGGLE_API_TOKEN` を前提にしたノートブック

| 物 | 場所 |
|---|---|
| ノートブック | `kernels/colab/gemma4_colab.ipynb`（30 cells / code 15 / nbformat OK / 全 code cell 構文検査済み） |
| 生成 | `make colab-notebook`（= `python3 src/pipeline/make_colab_notebook.py`） |
| 認証なしの分岐 | `make colab-notebook-hf`（`--source hf` -> `gemma4_colab_hf.ipynb`） |

### Secrets の入れ方（Colab 画面）

1. https://www.kaggle.com/settings/api で **Create New Token**（long-lived の access token。OAuth の `credentials.json` と違い期限 12 h に縛られない）
2. Colab 左パネルの **Secrets（鍵アイコン）** -> *Add new secret*
   - Name: `KAGGLE_API_TOKEN`
   - Value: token
   - **Notebook access** をこの notebook に対して ON にする（無いと `NotebookAccessError`）
3. 任意で `KAGGLE_USERNAME` / `KAGGLE_KEY`（legacy API key）も入れられる。cell 1 は main token が無いときだけこれを見る

### cell 1 が token を扱う形（値は output に出さない）

```python
from google.colab import userdata
userdata.keys()                     # 秘密の「名前」だけ print
token = userdata.get("KAGGLE_API_TOKEN")     # 無ければ SecretNotFoundError
os.environ["KAGGLE_API_TOKEN"] = token       # 同じ kernel の subprocess に伝わる
(Path.home() / ".kaggle" / "access_token").write_text(token + "\n")   # chmod 600（detached process でも通る）
```

その後 `pip install -q -U kaggle`（Colab image の preinstall kaggle は import で死ぬ）と認証検査
`kaggle competitions files -c gemma-4-developer-agent --page-size 3` まで自動でやる。検査出力に "Option A" /
"Missing key" が混ざったら cell 1 がそこで止める（ダラダラ download を始めない）。

安全メモ:
- `kaggle auth print-access-token` は **実行しない**（token が output に print され `.ipynb` に残る）
- `colab exec --env KAGGLE_API_TOKEN=...` は argv に secret が出る。CLI 経由なら file で上げて env は path にする（§9 の 3 番）
- notebook を共有しても Secrets は付かない。Secrets は Colab の Secrets パネルから削除できる

### Drive cache（download を 1 回にする）

`/content/drive/MyDrive/gemma4/data/` に `wheelhouse / graphs / embeddings / snapshots` + `tasks.jsonl` を置くと
（合計 ~4 GB、Drive free 15 GB に収まる）、次 session 以降は **cell 1 が Drive から戻し、`colab_bootstrap` 側の
「有りなら skip」で download が走らない**。cell 3 が push、cell 1 が restore。

### cell 早見（`gemma4_colab.ipynb`）

| cell | 内容 |
|---|---|
| 1 | `repo.tgz` 展開 + Drive mount/restore + **Secrets から token** + kaggle CLI 修理 + 認証検査 |
| 2 | `colab_bootstrap` で公式資材（tasks / HARNESS / graphs 127 / embeddings 127 / **測る task の snapshot だけ** / wheelhouse / model） |
| 3 | Drive に cache を push |
| 4 | PyPI で缺口補い（`colab_prepare_env.main()`） |
| 5 | variant `colab12` + `Context`（wheelhouse pin のまま） |
| 6 | `setup`（wheelhouse pin + torch pin）+ import 再検査 |
| 7 | `data`（harness loader）+ `agent` |
| 8 | `graph` + embeddings |
| 9 | `server` + 1 文生成 |
| 10 | `eval`（Phase 1 + Phase 2） |
| 11 | `package` + `patch` + `analyze` |
| 12 | manifest + 物リスト + 持ち帰り手順 |

### stage 本体（`src/` と `submission/`）の入手は 4 通り。**Colab upload は必須ではない**

cell 1 が A -> B -> C -> D の順で探し、何を拾っても `/content/repo/src/pipeline/` に正規化します
（`common.repo_root()` は `Path(__file__).parents[2]` なので、ここに寄せることで `Paths.detect` の
`data / models / work` 基準が揃う）。

| | 方法 | token | 毎回 upload |
|---|---|---|---|
| A | `/content/repo.tgz` を `colab upload` か Files panel | 不要 | **要る**（VM は session ごとに消える） |
| B | Drive の `gemma4/repo.tgz`（A を 1 回 cell 3 がそこへコピーする） | 不要 | 不要 |
| C | `kaggle datasets download -d <you>/gemma4-dev-agent-code`（+ `<you>/gemma4-dev-agent-submission` を `/content/repo/submission` に） | 要る | 不要 |
| D | Drive の `gemma4/src/` + `gemma4/submission/` を copy | 不要 | 不要 |

C を使う前提は **dataset の version が新しいこと**です:

```bash
kaggle datasets version -p .kaggle/dataset-code -m "sync src/" -r tar -q
kaggle datasets version -p .kaggle/dataset-submission -m "sync submission/" -r tar -q
```

古いと `stage_data.py` の loader 修正（harness の `Task` を使う方）や variant `colab12` が Colab 側に見えません。
`submission/` が無いとき cell 1 が Kaggle の submission dataset から取るのは、`stage_agent.candidate_submission_dirs`
の先頭が `<repo>/submission` なので確実です。

## 11. notebook 1 冊で完結させる版（`kernels/colab/gemma4_colab.ipynb`）

`make colab-notebook`（生成元 `src/pipeline/make_colab_notebook.py`）。21 cells / code 9 /
nbformat OK / 全 code cell 構文検査済み。**`src/pipeline/` を読まない**ので Colab に上げる物は notebook 1 個 + Secrets の token。
公式 library をそのまま叩きます。

### wheel を展開して確かめた公式 library の局所評価 API（`/tmp/whl` の unpacked 品）

`swegemma` 側に CLI がある（`python3 -m swegemma eval`）:

```
--tasks <tasks.jsonl>            --snapshots-dir <snapshots/>       --results-dir <out/>
--submission-dir <agent tree>    --sandbox docker|subprocess         --image swebench-sandbox:latest
--max-time-minutes 20            --max-tool-calls 80                --max-turns 150
--verification-timeout-seconds   --task-id <instance_id>(複数可)    --concurrency 1
--skip-agent-patch               --models-yaml <models.yaml>        --display auto|dashboard|single|quiet
```

endpoint は env で選ぶ（`MODEL_PROXY_URL` → `LITELLM_API_BASE` → `LOCAL_INFERENCE_URL` → `OPENAI_BASE_URL`、既定 `http://localhost:8000/v1`）、
key は `LOCAL_API_KEY` / `OPENAI_API_KEY`（既定 `EMPTY`）。

スコアは harness が出る: `results/summary.json` = `{total_tasks, resolved, resolution_rate, by_repo, errors}`、
`results/task_results.jsonl`（`instance_id / repo / resolved / agent_patch_size / test_exit_code / duration_seconds / error / tool_calls / total_llm_calls`）、
`results/patches/<id>.patch`、`results/test_outputs/<id>.log`。CLI 終了行は
`Evaluation complete: {resolved}/{total} resolved ({resolution_rate:.2%})`。**公式スコアはこの resolution_rate と同じ意味**。

アダプタ込みの server（`adk_submission`）:

- `discover_adapters(submission_dir)` -> `AdapterManifest.adapters = {name: AdapterInfo(name, path)}`（`adapters/<name>/adapter_config.json` + `adapter_model.safetensors`）
- `VllmServer(config, adapter_manifest=...)` が vLLM に LoRA module として渡す。`VllmConfig` に `enable_lora=True`, `max_loras=1`, `max_lora_rank=128`（`_round_up_vllm_lora_rank` で adapter rank を vLLM の bucket に上げる）
- model alias に adapter 名と `adapter:<name>` が入る（`create_model_registry(aliases=..., model_prefix="hosted_vllm/")`）
- `swegemma` 側は `discover_adapters` で見えた adapter を `openai/<adapter 名>` として register する -> **server が実際に serve していなければ resolable にならない**
- `validate_directory(submission_dir, limits=None)` が competition の制約（3 GiB / 拡張子 / agent.yaml が root / symlink 拒否）を見る

### tag filter は test 済み

`wheel_ok(name, py_tag)` は wheelhouse の実名 23 個で一致（`swegemma / adk_submission / google_adk / vllm-0.19.1-cp38-abi3 /
tokenizers-cp39-abi3 / safetensors-cp310-abi3 / openai_harmony-cp38-abi3` を採用、
`apache_tvm_ffi / cbor2 / ijson / msgspec / outlines_core / pybase64 / setproctitle / xgrammar / nvidia_cudnn_frontend`
の cp312-cp312 を skip、win/macosx を skip）。cp312 image に変えると cp312 wheel が通り cp313 wheel が落ちることも確認。

### 正直な前提（この notebook で変わりうる所・変われない所）

| 項目 | 公式 | この notebook | 変えられる? |
|---|---|---|---|
| sandbox | docker container | `--sandbox subprocess`（Colab に docker daemon が無い） | 変えられない（Colab 側で docker を立てられない） |
| task | 非公開 256 個 | 公開 129 のうち `TASK_LIMIT` 個 | 変えられない（非公開資材は渡らない） |
| GPU / vLLM 設定 | 4×L4, tp=4, bfloat16, 0.90 | 1×L4, tp=1, bfloat16, 0.85（`max_model_len=32768` は同じ） | 一部（gpu_mem は上げられる） |
| adapter | 提出 `adapters/` を load | `discover_adapters` で load。**今の提出 tree は adapter 0 個** | LoRA を train して `adapters/` に入れれば反映される |
| 提出物の評価 | 公式側が公式資材で実行 | 同じ zip を作れるが、評価は公式 | 提出して公式スコアを取る |

### 資材は 4 系統（ここを混同すると詰まる）

| 系統 | Kaggle 上の ID | 中身 | 公式 GS での読み方 |
|---|---|---|---|
| **コンペ資材**（被测 side） | competition `gemma-4-developer-agent` | `tasks.jsonl` / `snapshots/`(129 repo 状態) / `graphs/` 127 / `embeddings/` 127 / `wheels/` 124(task deps) / `docker/` `sandbox/` / `sample_submission/` | mount `/kaggle/input/competitions/gemma-4-developer-agent` |
| **ハーネス一式** | dataset `metric/gemma-4-developer-agent-wheelhouse` | `swegemma` `adk_submission` `adk_eval_core` `google-adk` `vllm 0.19.1` `transformers 5.13.1` pin の wheel 41 個 | mount `/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse` → pip install |
| **model** | instance `google/gemma-4/other/<model>/2` | `model.safetensors` 11.5 GB（4-bit QAT w4a16） | mount `/kaggle/input/models/google/gemma-4/other/<model>/2` |
| **あなたの提出物** | dataset `<you>/gemma4-dev-agent-submission` / `submission.zip` | `agent.yaml` `prompts/` `configs/sampling.yaml` `eval_config.yaml` `sub_agents/` `adapters/` | GS は `sample_submission/` を copy して `AGENT_DIR` にする |

**ハーネスは `submission.zip` からは来ません**。提出物は「agent の宣言 + LoRA」だけで、
harness（eval ループ・sandbox・verification・vLLM 起動）は wheelhouse dataset から install します。
`submission.zip` はcell 8 で再生成して提出する物であり、test の主体ではありません。

### Getting Started との対応

| 公式 GS の cell | している事 | `gemma4_colab.ipynb` | `src/pipeline` stage |
|---|---|---|---|
| Cell 1 Environment Configuration and Package Installation | offline env、wheelhouse を `--no-deps --force-reinstall`（Kaggle が消す `+cu128` 名前の復元含む） | **cell 3**（tag filter 版。Colab は cp313 なので cp312 wheel は skip、torch 三兄弟だけ deps 込み） | `setup` |
| Cell 2 Competition Dataset and Sample Submission | `load_tasks`、`MODEL_NAME` 乗り換え（agent.yaml + sub_agents/*.yaml の `model:` 書換 / 旧 model 用 `adapters/` を削除）、`AGENT_DIR` を working に copy | **cell 2**（download 版）+ **cell 4**（`MODEL_NAME_OVERRIDE` / budget 書換） | `data` + `agent` |
| Cell 3 Code Graph Functions | `swegemma.graph.get_graph` / embeddings を sample task で読む | **cell 2**（graphs/embeddings を揃える）+ **cell 6**（harness が `tasks.jsonl` の sibling を auto 発見、`swegemma/config.py:202-214`、wheels は `harness/container_setup.py:resolve_wheels_dir`） | `graph` |
| Cell 4 Start vLLM Server | `discover_declared_models` / `discover_adapters(adapter_extensions=...)` / `VllmConfig(tool_call_parser="gemma4", reasoning_parser="gemma4", max_model_len=32768, bfloat16, 0.90, enable_lora=bool(adapters), max_lora_rank=max(ranks) or 128, scheduler_reserve_full_isl=False, extra_args=...)` / `VllmServer(..., adapter_manifest=...)` / `create_model_registry(aliases=[model], model_prefix="openai/")` | **cell 5**（同じ引数。tp だけ 1） | `server` |
| Cell 5 Run Phase 1 Inference and Phase 2 Verification | `eval_config.yaml` から budget、`EvalConfig` + `Evaluator`、ThreadPool bridge、`submission.parquet` を kernel output に | **cell 6**（`python3 -m swegemma eval` を subprocess で。同じ `Evaluator` を harness 側が build する）+ **cell 7**（score） | `eval` |
| Cell 6 Package Submission Archive | symlink / size / 拡張子検査 + `validate_directory` + zip | **cell 8** | `package` |
| （GS に無い） | 失敗 taxonomy・patch 静的検査 | **cell 7**（`score.json`） | `patch` + `analyze` |

GS と意図的に違う 2 点:

* **mount ではなく download**（Colab には Kaggle mount が無い）。`snapshots/` は測る task だけ落とす（全部で 21.3 GB）
* **`Evaluator` を直接呼ぶ GS に対し、`swegemma eval` CLI を subprocess で叩く**（Colab kernel 側の event loop と喧嘩しないため。
  `cli.py` は `asyncio.run(evaluator.run())` を呼ぶので、kernel 内から直接叩くと `cannot be called from a running event loop` になる）

## 12. Colab で「公式と同じ資材」で測る / 提出物を測る / 自作 harness を測る

| 質問 | 答え | 根拠（この repo の実測） |
|---|---|---|
| Colab 上で Getting Started と**同じデータセット**で検証できる? | **できる**。公式 competition `gemma-4-developer-agent` を Kaggle CLI で叩けば GS が mount している物と同じ file が揃う（`tasks.jsonl` / `snapshots/` / `graphs/` 127 / `embeddings/` 127 / `wheels/` 124 / `docker/` `sandbox/` / `sample_submission/`）。Colab には mount が無いので **download になる**という差だけ。実測で `tasks.jsonl`・`graphs/`127・`embeddings/`127・snapshot・wheelhouse・model を Colab L4 に揃えた実績あり（`doc/colab.md` §3-§5） | 同一 source（Kaggle 公式）。HF mirror（`HanG/gemma-4-developer-agent`）は byte 検証していないので**公式と同じ**とは言わない |
| Colab 上で**ハーネス**と**最終提出物**を使った検証ができる? | **できる**。harness は wheelhouse dataset の pin 41 wheels で install して `swegemma eval` / `VllmServer` を実行（Colab L4 で vllm 0.19.1 pin の生成成功を実測済み）。提出物は `submission.zip` を展開して測り、測った tree から zip を作り直して `validate_directory` + 開き直し（round-trip）まで cell 8 で検査する | `AGENT_TREE_SOURCE` と `SUBMISSION_ZIP_CANDIDATES`（`/content/submission.zip`、`/content/drive/MyDrive/gemma4/submission.zip`）。zip が有れば zip を優先する |
| **自作ハーネス**（`src/pipeline`）や自作提出物を notebook の検証に入れられる? | **入れられる**。cell 7 の `PREV_RESULTS` に公式 harness の results dir か自作 pipeline の `predictions_*.json` を渡すと resolved の食い違いが出ます（同じ検査の CLI 版が `src/pipeline/parity_check.py`）。提出物側は自分の prompts / configs / adapters を cell 4 で staging して測る | `parity_check.py` は形式自動判別（`task_results.jsonl` / `predictions_*.json` / `*.parquet`）、exit 0 = 乖離なし |

### 公式と一致させられる物・一致させられない物

| 一致させられる | 一致させられない（Colab では真似できない） |
|---|---|
| 資材の **出所**（同じ competition / wheelhouse dataset / model instance） | 公式 scorer が使う **非公開 task 256 個**と hidden tests |
| harness の **version pin**（swegemma 0.2.11 / adk_submission 0.2.13 / adk_eval_core 0.1.0 / vllm 0.19.1 / transformers 5.13.1） | **4×L4 / tp=4 / gpu_mem 0.90**（Colab は 1×L4, tp=1） |
| `VllmConfig` の引数（`gemma4` parsers / `max_model_len=32768` / bfloat16 / LoRA 設定） | **docker container sandbox**（Colab に docker daemon が無いので `--sandbox subprocess`） |
| 提出物の **形**（`agent.yaml` が root、adapters 規則、3 GiB、`validate_directory`） | 公式の実行順そのもの（official harness は `Evaluator` を公式 side の runner 経由で呼ぶ） |

なので Colab の数字は「**公式スコアそのもの**」にはなりません。公式 harness と同じ資材・同じ pin で測った
**同じ条件の A/B 比較**として使い、絶対値は `submission.zip` の提出で取ります（`doc/verification.md`）。

## 13. notebook は 1 冊に集約しました（2026-11-22）

`kernels/colab/` に `colab` 付きが 4 個あって紛らわしかったので、**1 冊 + 資材モード 2 本**に畳みました。
生成元は `src/pipeline/make_colab_notebook.py` だけです（`make_colab_allinone_notebook.py` /
`make_colab_standalone_notebook.py` / `make_colab_measure_notebook.py` と、生成していた 4 つの `.ipynb` は削除済み）。

```bash
make colab-notebook            # -> kernels/colab/gemma4_colab.ipynb      （MATERIAL_SOURCE = "kaggle"）
make colab-notebook-hf         # -> kernels/colab/gemma4_colab_hf.ipynb   （MATERIAL_SOURCE = "hf"）
```

同じ notebook の cell 2 冒頭定数（`MATERIAL_SOURCE`）で資材と model の取り先だけが切り替わります。
Kaggle を触りたくなければ **生成し分けずに cell 2 を `"hf"` に書き換える**だけでも動きます。

### 依頼の 6 項目 -> cell 対応

| 依頼 | cell | 何を使うか |
|---|---|---|
| Kaggle の資材 DL | 2 | competition `gemma-4-developer-agent`（`tasks.jsonl` / `HARNESS_README.md` / `graphs/` / `embeddings/` / 測る task の `snapshots/`）+ dataset `metric/gemma-4-developer-agent-wheelhouse`（harness pin） |
| model DL | 2 | `kaggle models instances versions download google/gemma-4/other/gemma-4-e4b-it-qat-w4a16-ct/2 --untar`（HF モードなら `google/gemma-4-E4B-it-qat-w4a16-ct`） |
| 提出物が有ればそれ、無ければ既定の資材・ハーネス | 2 + 3 + 4 | 解決順 **提出 zip -> 自分の提出 dataset -> 公式 `sample_submission/`**。harness は常に wheelhouse の pin を `--no-deps` で入れる（cell 3） |
| サーバー構築 | 5 | `VllmConfig` + `VllmServer(cfg, adapter_manifest=discover_adapters(...))` + `create_model_registry(model_prefix="openai/")` + completion smoke |
| スコア算出 | 7 | `python3 -m swegemma eval`（`--sandbox subprocess`）の `summary.json` -> `resolution_rate` |
| 問題別に何が出来ていないか（エラー含む）をファイルに | 7 | **`results/failures.csv`** と **`results/failures.md`**（後述） |

### `failures.csv` / `failures.md` の中身（cell 7）

`task_results.jsonl` + `patches/*.patch` + `test_outputs/*.log` を突き合わせて、task 1 行に落とします。

| 列 | 意味 |
|---|---|
| `instance_id` | task id |
| `status` | `resolved` / `patch_fail`（patch は出たが test が落ちた） / `no_patch`（`submit_patch` が無い） / `crashed`（harness 側 error） / `not_run`（eval がその task に到達していない） |
| `reason` | 分類の理由。`protected path に触れている: ['conftest.py']` のような反則もここに足します |
| `test_exit_code` | Phase 2 の pytest exit code |
| `patch_chars` / `tool_calls` / `duration_s` | agent の出力量・tool 呼び出し数・時間 |
| `error` / `test_output_tail` | harness error 文と pytest 出力の末尾 3 行 |

`failures.md` には上の表 + `status` の集計 + 代表エラー + protected path list が入るので、**「どの problem が出来ていないか」はこの 1 ファイルで読めます**。
この実装は 1 個です（`src/pipeline/failures_report.py`）。cell 7 にはソースそのまま注入しているので、**Colab 側と Kaggle / pipeline
側（`stage_analyze`・`make failures`）で CSV の列が揃います**。pipeline 側は `doc/pipeline.md` の analyze 節。
`not_run` が出るのは snapshot 欠けか途中で打ち切った場合で、**測れていない task が残っている**合図です。

### cell 早見（21 cells / code 9）

| cell | 内容 |
|---|---|
| 0 | 使い方（準備 / 実行順 / 定数表 / 時間の目安 / 出てくる物 / 詰まったとき / 筋） |
| 1 | Colab Secrets の `KAGGLE_API_TOKEN` + CLI 修理 + 認証検査（HF モードでは飛ばして可） |
| 2 | 資材 DL + model DL + 提出 tree 解決（zip -> 自 dataset -> 公式 sample_submission）+ Drive cache |
| 3 | wheelhouse pin を PEP 427 tag で選んで install + torch 三兄弟整合 + Colab 固有 gap 補完 |
| 4 | 提出 tree を staging（`eval_config.yaml` の budget 書換 / `MODEL_NAME_OVERRIDE` / 単一 model 検査） |
| 5 | adapter 込み vLLM server + registry + completion smoke |
| 6 | `python3 -m swegemma eval`（Phase 1 + Phase 2） |
| 7 | score 抽出 + **`failures.csv` / `failures.md`** + `PREV_RESULTS` parity |
| 8 | `validate_directory` + zip 化 + zip 往復再検査 |
| 9 | `server.stop()` + 出力一覧 + download / 提出コマンド |
