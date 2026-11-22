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

- `collect_trajectories.py --role {main,analyzer,verifier}`（既定 `main`）。出力は
  ROLE_DEFAULT_OUTPUTS 既定で main のみ `data/trajectories.jsonl` がそのまま。リグレッションは
  `--limit 0` 相当の全量実行で `cmp` / md5 比較（data/trajectories.jsonl は committed 済み 106 行）。
- `build_dpo_pairs.py` は `--role` を `main` / `analyzer` / `verifier` で受け、`build_trajectory` に
  そのまま渡す。analyzer / verifier にも gold fallback（`tasks.jsonl` だけ）が効く。出力先指定は必須。
- `check_role_calls()` が analyzer / verifier で `edit_file` / `write_file` / `submit_patch` を検出したら
  `ValueError`。生成データの tool list は `read_file` / `run_command` / `search_similar_code`（analyzer）、
  `run_command` / `get_status`（verifier）に制限されている。
- `collect_trajectories.normalize_trace()` は harness の実 trace 形式（`steps[].source` /
  `steps[].message`、単数 `tool_call`）を解釈できない。results からの self-play は 0 件のまま
  （main の挙動を変える変更になるので未修正）。
- `Makefile` の `collect:` は main 固定（`--tasks data/tasks.jsonl --output data/trajectories.jsonl`）。
  analyzer / verifier のターゲットは無い。

## 学習 notebook の 3-LoRA 配線（`src/pipeline/make_train_notebook.py`）

- cell 3: `ADAPTER_NAME #@param ["main_lora", "analyzer_lora", "verifier_lora"]` /
  `ROLE #@param ["main", "analyzer", "verifier"]`。cell 3 末尾の `role_suffix` から
  `TRAJECTORIES_JSONL`（main = `data/trajectories.jsonl`、analyzer / verifier は
  `data/trajectories_<role>.jsonl`）と `DPO_PAIRS = f"data/dpo_pairs_{ROLE}.jsonl"` を組む。
- cell 4 は `collect_trajectories.py --role {ROLE} --output {TRAJECTORIES_JSONL}` →
  `train_lora.py --data {TRAJECTORIES_JSONL} --out_dir submission/adapters/<ADAPTER_NAME>`。
  **`--role` だけでは足りず `--output` が必須**（既定出力が main 固定なので、role を切り替えると
  前の教材を上書きする）。`import_arg`（`--import-traces` / `--only-resolved`）は
  `Path(RESULTS_DIR).exists()` のときだけ付く。cell 5 は元から `--role {ROLE} --output {DPO_PAIRS}`。
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

## 作業用メモ

- このシェル（`/opt/venv/bin/python3`）だと素の `python3` で通る: `build_submission.py --check-only`、
  `collect_trajectories.py`、notebook 生成（nbformat 5.11.1 入り）。transformers / peft / trl は無いので
  `src/train/*.py` はここでは実行しない（notebook / Kaggle kernel 側で）。
- Kaggle code competition の submission は file version 経由のみ（kernel submission は不可）。
  code competition では GPU kernel を複数追加できるが、submit はコンペページから。
