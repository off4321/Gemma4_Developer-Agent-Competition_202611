# MEMORY

この workspace（Gemma4 Developer Agent 競合作成用 repo）で確定した事実・はめどころ。
作業の都度追記する。長文の設計は `doc/` を読む。

## ハーネス（`data/HARNESS_README.md` が source of truth）

- 宣言できる model は **1 種類だけ**（`validate_single_declared_model`）。`agent.yaml` と
  `sub_agents/*.yaml` を全部舐めた集合が 2 要素だとサーバ起動失敗。adapter 名は agent ごとに
  変えてよい（親 `main_lora` / 子 `analyzer_lora` / `verifier_lora`）。
- sub-agent の生成パラメータは **`generate_content_config`**（`max_output_tokens`,
  `thinking_config.thinking_budget` / `include_thoughts`）。Google ADK / Gemini API 側の
  `generate_config` / `max_tokens` / `max_thoughts` は **通らない**。
  `build_submission.py` が `ALLOWED_GENERATION_FIELDS` で弾く。
- `thinking_config.thinking_budget > 0` + `include_thoughts: false` は「reasoning は使って
  thought は親 context に出さない」。analyzer は `include_thoughts: true` の親（tool-call 連鎖が
  必要）なので verifier だけ `false`。
- `agent.yaml` から `agent_tool: {config_path}` で refer されていない `sub_agents/*.yaml` は
  `build_submission.py --check-only` の docs 探索対象から **外れる**（`agent docs discovered: 2` のまま）。
  つまり verifier は `agent.yaml` の `agent_tool:` をコメントアウトした限り zip には入るが検証されない。
- `agent.yaml` の `instruction` / `tools` に `!include` は使えるが、**`sub_agents/*.yaml` の中身は
  include 禁止**（`_reject_python` / schema 制約）。
- sub-agent の tool list を超える tool を教材にすると、サーバ側は `FunctionNotFoundError` JSON を
  返すだけで停止も tool list 再送もしない（`ToolErrorPlugin`、`max_tool_calls` を溶かす）。
  → `collect_trajectories.check_role_calls()` でビルド時に弾く。
- gold patch が「新規ファイル 1 個」の task は hunk が `max_hunk_lines` を超えて
  `edits_for_file()` が空になり、analyzer / verifier の gold fallback が作られない
  （129 task 中 analyzer 106 / verifier 128 本）。

## データ合成（`src/collect/`）

- `collect_trajectories.py --role {main,analyzer,verifier}`（既定 `main`）。出力は ROLE_DEFAULT_OUTPUTS 既定（main のみ `data/trajectories.jsonl`）。
- **main 教材は edit_file の前に必ず read_file**（104/104 本が 1 発目 `read_file`、meta に `n_reads`）。read_file の args は `{filepath, start_line, end_line}`（int、`end_line >= start_line >= 1`）で、observation は `content` のみに `start_line` / `end_line` を添える（`content_preview` は作らない）。committed 済みの 106 行は旧 form なので md5 リグレッション比較は不可。
- `--no-patch-examples`（既定 8・main のみ）で「原因は repo の外の設定」型の **final answer `NO_PATCH`** 教材を足す（`build_no_patch_trajectory()`: grep → read_file → targeted pytest → `git status --short` → `NO_PATCH`）。全量は 104 + 8 = 112 行。
- record の meta には `_io_summary()` が messages から導いた `files_read`（`path:start-end`）/ `files_written` / `files_changed` / `test_targets` / `test_status` を入れる（`normalize_trace()` の meta にも同じ key が入る）。
  `--limit 0` 相当の全量実行で `cmp` / md5 比較（data/trajectories.jsonl は committed 済み 106 行）。
- `build_dpo_pairs.py` は `--role` を `main` / `analyzer` / `verifier` で受け、`build_trajectory` に
  そのまま渡す。analyzer / verifier にも gold fallback（`tasks.jsonl` だけ）が効く。出力先指定は必須。
- `check_role_calls()` が analyzer / verifier で `edit_file` / `write_file` / `submit_patch` を検出したら
  `ValueError`。生成データの tool list は `read_file` / `run_command` / `search_similar_code`（analyzer）、
  `run_command` / `get_status`（verifier）に制限されている。
- `collect_trajectories.normalize_trace()` は harness の実 trace（ATIF: `steps[].source` / `steps[].message`、`tool_calls[{function_name, arguments}]`）を OpenAI chat form に変換する。**`resolved` は trace に無い**（実データは `task_results.jsonl` 側）ので、steps があって成功 パラメータが無い trace は `resolved=False` 扱い（`--only-resolved` では 0 件になる）。tool 応答は `role: "tool"` + `tool_call_id` に寄せる。`role: "model"` / `output` でも読める。ATIF にしか無い key は message 本文に保存し、round-trip の完全性は要求しない。
- `build_dpo_pairs.py` は rejected を messages + observations の signal で選ぶ（`repeat_loop` / `edit_file_repeat` / `error_storm` / `unresolved`。`LOOP_MIN_REPEAT_CALLS = 3`、`LOOP_MIN_FAILED_STEPS = 4`）。**chosen と rejected の最終 assistant 回答が同一なら pair を作らない**。rejected が `NO_PATCH` で終わり chosen に edit_file / write_file が無い pair も作らない（NO_PATCH 自体を不正解として教えない）。rejected の末尾 tool 応答は落として assistant で終わらせる。実 results dir 2 個で 6 件ずつ作れた（旧 0 件）。`tool_call_id` は保持される。
  （main の挙動を変える変更になるので未修正）。
- `Makefile` の `collect:` は main 固定（`--tasks data/tasks.jsonl --output data/trajectories.jsonl`）。
  analyzer / verifier のターゲットは無い。

## 学習 notebook の 2-LoRA 配線（`src/pipeline/make_train_notebook.py`）

- cell 3: `ADAPTER_NAME #@param ["main_lora", "analyzer_lora", "verifier_lora"]` /
  `ROLE #@param ["main", "analyzer", "verifier"]`。cell 3 末尾の `role_suffix` から
  `TRAJECTORIES_JSONL`（main = `data/trajectories.jsonl`、analyzer / verifier は
  `data/trajectories_<role>.jsonl`）と `DPO_PAIRS = f"data/dpo_pairs_{ROLE}.jsonl"` を組む。
- cell 3 のフォームは `BUILD_MODE #@param ["main", "main+analyzer", "all"]`（既定 `main+analyzer` = 提出に必要な `main_lora` + `analyzer_lora` を 1 周で作る）。`ROLES` から `ADAPTER_BY_ROLE` / `ROLE_BY_DATA` / `OUT_DIR_BY_ROLE` / `DATA_BY_ROLE` / `DPO_BY_ROLE` を導く。`ADAPTER_NAME` は `BUILD_MODE = "main"` のときだけ出力先を上書きする。
- **A100 前提**: `nb.metadata["colab"] = {"gpuType": "A100", "gpu_types": ["A100"], ...}（`gpu_types` は新しい schema 側。片方だけだと Colab が黙って落とす）。`USE_4BIT = False`（bf16 で約 55GB）、`MAX_LEN = 8192`、`MAX_EXAMPLES = 200`、`MAX_TOOL_CALLS = 45` / `MAX_TURNS = 60`（`submission/eval_config.yaml` と collect 側の既定と同じ）。L4 / T4 のときだけ 4bit + `MAX_LEN 6144` 以下に下げる。
- **cell 数を増やすより cell 3 の値で分岐させる**（role ごとに cell を复制すると実行順ミスの温床）。cell 4（collect）/ cell 5（train_lora）/ cell 6（DPO・main のみ）/ cell 7（検証）が `for role in ROLES`。**`--role` だけでは足りず `--output` が必須**（既定出力が main 固定）は不変。
- **ループ内では `!python3 ...` を書かない**（magic 行の `{}` 展開と `\` 継続で生成 notebook が壊れ、`_check_python()` も magic 行を `pass` に置換するのでループ本体を検証できない）。`cmd = (f"...")` + `os.system(cmd)` にする。効果的な SFT 引数は `--lr 5e-5 --max_len 8192 --max_examples 200`（`LEARNING_RATE` / `MAX_LEN` / `MAX_EXAMPLES` の既定）。DPO は `train_dpo.py` が自前の `--max_len`（既定 4096）を持つ。
- **cell ソース内で `\` 行継続を使わない**: cell を囲む `"""` 側で継続が先に解決され、生成 notebook の
  行が潰れて `else ...` が 1 行に繋がる。中間変数（`role_suffix` など）に逃がす。
  shell コマンド側の継続（generator 内 `\\\\`）はそのまま大丈夫。
- `main()` が全 code cell を ast 検査 → `nbf.validate` してから書く。cell 数は 13（md 7 + code 6）。
  `kernels/` は git 管理対象外なので notebook の変更は `git status` に出ない（生成物）。
- cell 1 は `!pip install -q -U peft datasets trl accelerate bitsandbytes pyyaml kaggle`（`kaggle` を入れたのは cell 2 の
  `data/tasks.jsonl` 取得用。Colab 同梱の kaggle CLI は import で死ぬのでここで上書きする = `doc/colab.md` §10）。
- cell 2 の token は `resolve_token()` で **cell の #@param → kernel env → Colab Secrets** の順（`doc/colab.md` §9 / §10）。
  env は `KAGGLE_API_TOKEN` / `KAGGLE_API_KEY` / `KAGGLE_ACCESS_TOKEN`、Secrets は `KAGGLE_API_TOKEN` → `KAGGLE_API_KEY`
  （legacy の `KAGGLE_USERNAME`+`KAGGLE_KEY` は `~/.kaggle/kaggle.json` に書く）。env の値は token 文字列でも
  **file path でも通す**。取れたら `os.environ["KAGGLE_API_TOKEN"]` + `~/.kaggle/access_token`（mode 600）。値は print しない。
  **generator 側の `#@param` 既定は常に空文字**（token を直書きしない。VS Code / Kaggle kernel では env 側で動く）。
- cell 2 は `kaggle competitions download -c gemma-4-developer-agent -f tasks.jsonl -p data` で `data/tasks.jsonl` を取りにいく。
  zip で返ると Python の `zipfile` で展開（Colab image に `unzip` が無いケースを避ける）。**既存の tasks.jsonl は上書きしない**。
  token が無いのは skip（落ちない）。`results/` は competition の file ではないので今まで通り手元で `kaggle kernels output`。
- **cell 2 をローカルで流すテストは temp repo に `chdir` してから**やる（cell の `%cd {REPO_DIR}` を真似しないと
  `kaggle competitions download -p data` が **カレントの `data/tasks.jsonl` を踏み抜く**。実損あり）。`data/` は gitignore なので git から戻せない。

## A100 2-LoRA step で変えた設定

- `submission/configs/sampling.yaml`（main / analyzer 共有）: `max_output_tokens: 1024`、
  `thinking_budget: 2048`。**256 以下に戻さない**（`edit_file` の JSON が切れて `}` 欠落 → retry の
  無限ループ）。`temperature` 0.6（analyzer は 0.2 固定）。verifier 側も `thinking_budget: 2048`。
- `submission/eval_config.yaml`: `max_tool_calls: 45` / `max_turns: 60`。`build_submission.py` は
  `max_turns < 50` と `max_tool_calls < 40` で警告を出す（40 以下に戻すと 警告の復活、80/150 に戻さない）。
- `submission/prompts/system.md`: rule 0（**会話をしない。assistant turn ごとに tool call 1 個**）、
  rule 4（**edit_file / write_file が 1 回も成功するまで submit_patch しない**）、rule 5（**`NO_PATCH` は
  「repo の外の設定が原因」と証明できたときだけ**。追跡可能な設定問題は通常の `edit_file` 修正として扱う）、
  2 回連続失敗で局所化手段を変える、を追加。`analyzer.md` は「read していない範囲の old_string を出さない」、
  `verifier.md` は「**1 発目の tool call は `run_command("git status --short")`**（`get_status` より先）」と
  「GO は差分 + 通った targeted tests がセット。config-only の修正は GO」に更新。
- `submission/sub_agents/code_analyzer.yaml`: `adapter: analyzer_lora` を**有効化**（旧 `# adapter: tool_lora`）。
  → `submission/adapters/analyzer_lora/` を生成するまで `build_submission.py --check-only` は
  `[ERROR] ... references adapter(s) ['analyzer_lora'] that do not exist` で 1 件失敗する（Colab の cell 5 が
  書けば消える。main だけで組み直すときは行をコメントに戻す）。実 trace の `read_file` は `{"filepath": ...}`
  で、`path` ではない。

## 作業用メモ

- このシェル（`/opt/venv/bin/python3`）だと素の `python3` で通る: `build_submission.py --check-only`、
  `collect_trajectories.py`、notebook 生成（nbformat 5.11.1 入り）。transformers / peft / trl は無いので
  `src/train/*.py` はここでは実行しない（notebook / Kaggle kernel 側で）。
- Kaggle code competition の submission は file version 経由のみ（kernel submission は不可）。
  code competition では GPU kernel を複数追加できるが、submit はコンペページから。
