#!/usr/bin/env python3
"""Generate the Colab RFT + DPO Multi-LoRA training notebook (`kernels/colab/gemma4_lora_train_colab.ipynb`).

Keeps the notebook clean and reproducible from source, mirroring `make_colab_notebook.py`
without touching the evaluation notebook. Runs on H100 / A100 (80GB) with `USE_4BIT = False`
(bf16) by default and builds `main_lora` + `analyzer_lora` in one pass through `BUILD_MODE`.
"""

from __future__ import annotations

import argparse
import ast
import re
import textwrap
from pathlib import Path

import nbformat as nbf

MAGIC_LINE = re.compile(r"^\s*[!%]")


def _body(text: str) -> str:
    """notebook に書く本文。source 側でインデントしているので common indent を落とす。

    `strip()` だけだと `#@title` も Python も 12 spaces 付きで cell に残り、Colab で即 SyntaxError になる。
    """
    return textwrap.dedent(text).strip() + "\n"


def _md(text: str):
    return nbf.v4.new_markdown_cell(_body(text))


def _code(text: str):
    return nbf.v4.new_code_cell(_body(text))


def _check_python(source: str) -> None:
    """magic 行（`!` / `%`）とその `\\` 継続を `pass` に換えて、残りを実際に parse する。

    cell の indent が壊れていても notebook は書けてしまうので、構文は build 時点で保証する。
    """
    lines: list[str] = []
    continuing = False
    for raw in source.splitlines():
        if continuing:  # magic 行の shell 継続行は捨ててよい
            continuing = raw.rstrip().endswith("\\")
            continue
        if MAGIC_LINE.match(raw):
            lines.append(raw[: len(raw) - len(raw.lstrip())] + "pass")
            continuing = raw.rstrip().endswith("\\")
            continue
        lines.append(raw)
    ast.parse("\n".join(lines) or "pass")


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata["accelerator"] = "GPU"
    nb.metadata["colab"] = {"provenance": [], "gpuType": "A100"}
    # newer Colab metadata key; keep both so neither schema drops the accelerator request
    nb.metadata["colab"]["gpu_types"] = ["A100"]

    nb.cells = [
        _md(
            """
            # Gemma 4 e4b — RFT + DPO Multi-LoRA Training (Colab)

            Colab 上でリポジトリを clone し、**RFT（解決済み軌跡＋gold合成 SFT）→ DPO（解決しなかった実軌跡を
            rejected にした選好学習）** で `main_lora` / `analyzer_lora`（そしてお望みなら `verifier_lora`）を
            作るノートブックです。**cell 3 の `BUILD_MODE = "main+analyzer"` が既定で、コンペが必要とする
            「親用 + analyzer 用」の 2 本が 1 周で揃います**（`main` = main のみ、`all` = 3 本）。
            提出構成は `agent.yaml` が `main_lora`、`sub_agents/code_analyzer.yaml` が `analyzer_lora` を指すので、
            この 2 本が最少構成です。

            - **GPU は H100 / A100 (80GB) が既定**（`USE_4BIT = False` = bf16、約 55GB）。L4 / T4 を選んだときだけ
              cell 3 で `USE_4BIT = True`（4bit）に下げ、`MAX_LEN` も 6144 以下にしてください
            - **RSLoRA** (`--use_rslora`): 高ランクでも勾配が潰れにくい（既定 OFF。rank を 64 以上に上げるときだけ ON）
            - **LoRA+** (`--lora_plus_ratio 4.0`): `lora_B` の学習率を上げてツール形式の収束を早める。倍率を稼ぎすぎると
              （`16.0` 以上）tool call がただの文字列として学習され、tool call 崩れの原因になる（学習率の既定は `--lr 5e-5`）
            - 学習ベースは `google/gemma-4-e4b-it`、保存時の `adapter_config.json` は本番の `google/gemma-4-e4b-it-qat-w4a16-ct` に自動書き換えされます

            cell 1 → 2 → 3 → 4 → 5 → 6 → 7 の順に実行します（cell 4 = 教材収集、cell 5 = LoRA、cell 6 = main のみ DPO、
            cell 7 = adapter と提出契約の検証）。cell 1 の `pip install` で Colab が「ランタイムを再起動」と
            出したら、再起動してから cell 2 以降を続けてください（未再起動だと transformers / peft の import が
            混ざって落ちます）。`data/tasks.jsonl` は cell 2 が Kaggle から自動 download します（token が無いときだけ
            Files パネルにアップロード）。eval 軌跡入りの `results/` は相変わらず手動アップロードです。
            """
        ),
        _md(
            """
            ## Cell 1 — GPU確認 & 依存インストール

            `nvidia-smi` で **NVIDIA A100 / H100 (80GB)** が当たっているかを確認し（cell 3 の既定は bf16 = 80GB 前提）、
            `pip install` で `peft` / `datasets` / `trl` / `accelerate` / `bitsandbytes` / `pyyaml` / `kaggle` を導入します。

            `kaggle` を一緒に入れるのは cell 2 が `data/tasks.jsonl` を competition から落とすためです（Colab 同梱の
            `kaggle` は import で死ぬので、ここで上書きしておくと credential も素直に読めます）。

            **インストール後に Colab が「ランタイムを再起動」するダイアログを出したら、必ず再起動してから
            cell 2 以降を実行してください。** 再起動を飛ばすと Colab 既定版と新規版のライブラリが混ざって
            import が壊れます。
            """
        ),
        _code(
            """
            #@title 1. GPU確認 & 依存インストール
            !nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
            !pip install -q -U peft datasets trl accelerate bitsandbytes pyyaml kaggle
            """
        ),
        _md(
            """
            ## Cell 2 — リポジトリ clone / 更新 & Kaggle / HF 認証 + `data/tasks.jsonl` 取得

            `REPO_URL` を自分のリポジトリに向けて clone（2 回目以降は `git pull --ff-only`）し、`/content/repo` に cd します。
            `#@param` の `REPO_URL` だけ自分の GitHub URL に書き換えれば後はそのままで動きます。

            ### token は「cell の #@param → kernel env → Colab Secrets」の順で探す

            | 変数 | env 名 | Colab Secrets 名 | 無いとき |
            | --- | --- | --- | --- |
            | `KAGGLE_API_TOKEN` | `KAGGLE_API_TOKEN` / `KAGGLE_API_KEY` / `KAGGLE_ACCESS_TOKEN` | `KAGGLE_API_TOKEN` → `KAGGLE_API_KEY` | `data/tasks.jsonl` の download を skip（手元で置いた物を使う） |
            | `KAGGLE_USERNAME` + `KAGGLE_KEY` | 同じ名前 | 同じ名前 | legacy key 経路は使わない |
            | `HF_TOKEN` | `HF_TOKEN` | `HF_TOKEN` | cell 5 の gated model 読み込みが 401 になる |

            - Colab の運用は Secrets が定番です（https://www.kaggle.com/settings/api の Create New Token を Secrets に入れる。
              左の **Secrets** -> Notebook access ON を忘れると `NotebookAccessError`）。`.ipynb` に token が残りません
            - **VS Code / Kaggle kernel では `google.colab` が無いので env 側**で動きます（`export KAGGLE_API_TOKEN=...`）。
              環境変数の値は token 文字列でも、token を書いた **file の path** でも通します（`kaggle` CLI と同じ読み方 = `doc/colab.md` §9）
            - 取れた token は `os.environ["KAGGLE_API_TOKEN"]` に戻し、おまけに `~/.kaggle/access_token`（mode 600）へも書きます。
              Colab の `!export` は subprocess に残らないので、この 2 段どりで後続 cell の `kaggle` 呼び出しにも効きます
            - legacy の `KAGGLE_USERNAME` + `KAGGLE_KEY` しかないときは `~/.kaggle/kaggle.json` を作ります
            - **token の値は print しません**（出所と文字数だけ。`kaggle auth print-access-token` は実行しない）
            - token が無いところで cell は落ちません（download を skip して案内を出すだけ。git や Files パネルで置く運用が生きています）

            ### `data/tasks.jsonl` はこの cell が Kaggle から取りにいく

            ```bash
            kaggle competitions download -c gemma-4-developer-agent -f tasks.jsonl -p data
            ```

            zip で返ってきたときは Python の `zipfile` で `data/tasks.jsonl` に展開します（`unzip` の無い環境でも動くように）。
            **既に `data/tasks.jsonl` がある場合は上書きしません**（`git pull` で入れた物 / Files パネルで置いた物をそのまま使う）。

            | パス | 中身 | 無い場合 |
            | --- | --- | --- |
            | `data/tasks.jsonl` | 評価タスク 129 個（`patch` = 正解 gold patch、`test_patch` / `problem_statement` 付き） | gold 合成 SFT が作れず cell 4 が空振りする |
            | `results/` | eval kernel の出力（`kaggle kernels output <kernel> -p results/` で自分の kernel から取る） | cell 4 は gold 合成のみ、cell 6 の DPO は skip |

            `results/` は competition の file ではないので自動取得しません。eval kernel を Kaggle で回して、その出力を
            手元で `kaggle kernels output <自分の kernel> -p results/` と下ろし、`/content/repo/results` に置いて `git pull` する（運用は今まで通り）。
            """
        ),
        _code(
            """
            #@title 2. リポジトリ clone / 更新 & Kaggle / HF 認証 + tasks.jsonl 取得
            import json
            import os
            import subprocess
            import zipfile
            from pathlib import Path

            REPO_URL = "https://github.com/off4321/Gemma4_Developer-Agent-Competition_202611.git"  #@param {type:"string"}
            KAGGLE_API_TOKEN = ""  #@param {type:"string"}
            KAGGLE_USERNAME = ""  #@param {type:"string"}
            KAGGLE_KEY = ""  #@param {type:"string"}
            HF_TOKEN = ""  #@param {type:"string"}
            REPO_DIR = Path("/content/repo")
            COMPETITION = "gemma-4-developer-agent"
            TASKS = REPO_DIR / "data" / "tasks.jsonl"
            KAGGLE_ENVS = ("KAGGLE_API_TOKEN", "KAGGLE_API_KEY", "KAGGLE_ACCESS_TOKEN")
            KAGGLE_SECRETS = ("KAGGLE_API_TOKEN", "KAGGLE_API_KEY")

            if not (REPO_DIR / ".git").exists():
                !git clone "{REPO_URL}" "{REPO_DIR}"
            else:
                !git -C "{REPO_DIR}" pull --ff-only

            %cd {REPO_DIR}


            def resolve_token(cell_value, env_names, secret_names, quiet=False):
                '''token を cell の #@param -> kernel env -> Colab Secrets の順で探す（doc/colab.md §9 §10）。

                Colab の `!export` は subprocess に残らないので、取れたら caller が os.environ に戻す前提。
                env の値は token 文字列でも「token を書いた file の path」でも通す（kaggle CLI と同じ作法）。
                無いところで落とさない（token 無し運用 = 手元で data/tasks.jsonl を置く道を生かす）。
                値は print せず長さと出所だけ出す。
                '''
                value = (cell_value or "").strip()
                if value:
                    print(f"[auth] cell の #@param で受け取った（{len(value)} chars）")
                    return value
                for name in env_names:
                    raw = (os.environ.get(name) or "").strip()
                    if not raw:
                        continue
                    if "/" in raw and Path(raw).expanduser().is_file():
                        raw = Path(raw).expanduser().read_text(encoding="utf-8").strip()
                        print(f"[auth] env {name} は file path なので中身を使う（{len(raw)} chars）")
                    else:
                        print(f"[auth] env {name} を使う（{len(raw)} chars）")
                    return raw
                try:
                    from google.colab import userdata
                except ImportError:                      # VS Code / Kaggle kernel では Secrets 自体が無い
                    if not quiet:
                        print(f"[auth] token 無し（cell / env {env_names[0]} が空、Colab 以外では Secrets 無し）")
                    return ""
                for name in secret_names:
                    try:
                        raw = (userdata.get(name) or "").strip()
                    except Exception:                    # SecretNotFoundError / Secrets の Notebook access OFF
                        raw = ""
                    if raw:
                        print(f"[auth] Colab Secrets の {name} を使う（{len(raw)} chars）")
                        return raw
                if not quiet:
                    print(f"[auth] token 無し（cell / env {env_names[0]} / Secrets {secret_names[0]} が空）")
                return ""


            def run(cmd, tail=300):
                '''kaggle CLI や unzip を回す（+ 行で実行内容、out は末尾だけ print、返し値は (code, out)）。'''
                print("+ " + " ".join(str(part) for part in cmd), flush=True)
                proc = subprocess.run([str(part) for part in cmd], capture_output=True, text=True)
                out = ((proc.stdout or "") + (proc.stderr or "")).strip()
                if out:
                    print(out[-tail:], flush=True)
                return proc.returncode, out


            kdir = Path.home() / ".kaggle"
            kdir.mkdir(parents=True, exist_ok=True)
            kaggle_token = resolve_token(KAGGLE_API_TOKEN, KAGGLE_ENVS, KAGGLE_SECRETS)
            legacy_user = resolve_token(KAGGLE_USERNAME, ("KAGGLE_USERNAME",), ("KAGGLE_USERNAME",), quiet=True)
            legacy_key = resolve_token(KAGGLE_KEY, ("KAGGLE_KEY",), ("KAGGLE_KEY",), quiet=True)

            if kaggle_token:
                os.environ["KAGGLE_API_TOKEN"] = kaggle_token            # 同じ kernel の subprocess 経由 CLI に効く
                with (kdir / "access_token").open("w", encoding="utf-8") as fh:
                    print(kaggle_token, file=fh)                         # 改行だけ付けて file に置く（doc §10）
                os.chmod(kdir / "access_token", 0o600)
                print(f"[auth] ~/.kaggle/access_token に設定（{len(kaggle_token)} chars、値は出さない）")
            if not kaggle_token and legacy_user and legacy_key:
                os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"] = legacy_user, legacy_key
                (kdir / "kaggle.json").write_text(json.dumps({"username": legacy_user, "key": legacy_key}))
                os.chmod(kdir / "kaggle.json", 0o600)
                print(f"[auth] legacy key を ~/.kaggle/kaggle.json に設定（user: {legacy_user}）")

            kaggle_ok = bool(kaggle_token or (legacy_user and legacy_key))
            if not kaggle_ok:
                print("[auth] Kaggle credential 無し -> tasks.jsonl の download は skip（手元から置けばそれで動く）")

            hf_token = resolve_token(HF_TOKEN, ("HF_TOKEN",), ("HF_TOKEN",))
            if hf_token:
                os.environ["HF_TOKEN"] = hf_token
                print("[auth] HF_TOKEN を kernel env に設定（cell 5 の gated model 読み込みに要る）")

            if TASKS.is_file():
                print(f"[tasks] 有り（download は skip）: {TASKS}（{TASKS.stat().st_size} bytes）")
            elif kaggle_ok:
                code, out = run(["kaggle", "competitions", "download", "-c", COMPETITION, "-f", "tasks.jsonl", "-p", "data"])
                if code != 0 and any(word in out for word in ("401", "403", "404", "Forbidden", "Unauthorized")):
                    print("[tasks] Kaggle が download を断った。規約同意（competitions/gemma-4-developer-agent）か token の有効性を確認")
                for zipped in sorted((REPO_DIR / "data").glob("*.zip")):
                    with zipfile.ZipFile(zipped) as zf:
                        hits = [name for name in zf.namelist() if name.endswith("tasks.jsonl")]
                        if hits:
                            TASKS.write_bytes(zf.read(hits[0]))
                            print(f"[tasks] zip から展開: {zipped} -> {TASKS}")
                    zipped.unlink(missing_ok=True)
            if TASKS.is_file():
                rows = len(TASKS.read_text(encoding="utf-8").splitlines())
                print(f"[tasks] OK: {TASKS}（{rows} 行）-> cell 4 の gold 合成 SFT に使う")
            else:
                print("[tasks] 無い。手元で: kaggle competitions download -c gemma-4-developer-agent -f tasks.jsonl -p data")
                print("[tasks] か Files パネルで /content/repo/data/ に置く（無ければ cell 4 の gold 合成が空振り）")
            print("[results]", "有り（cell 5 の DPO に使う）" if (REPO_DIR / "results").is_dir() else "無し（cell 5 は skip）")
            """
        ),
        _md(
            """
            ## Cell 3 — 学習設定（`BUILD_MODE` で LoRA の本数を決めるフォーム）

            **コードを書かずに学習設定を切り替えるフォーム cell** です。`BUILD_MODE` が「何本 LoRA を作るか」を決めます。

            | `BUILD_MODE` | 作る adapter | 使い方 |
            | --- | --- | --- |
            | `main` | `main_lora` だけ | 親子構成の変更なしで main だけ更新するとき |
            | `main+analyzer` | `main_lora` + `analyzer_lora` | **このコンペの既定（parent + analyzer の 2 LoRA）** |
            | `all` | `main_lora` + `analyzer_lora` + `verifier_lora` | verifier も子として付ける予定のとき |

            - `main_lora` は **読み込み専用の analyzer に対して edit / run / DPO を学ぶ adapter**、
              `analyzer_lora` は **read-only での局所化を学ぶ adapter** です。同じ checkpoint から別教材で育てるので
              混ざりません（analyzer 教材に edit_file / submit_patch は一切出てきません）
            - `ADAPTER_NAME`: `BUILD_MODE = "main"` のときだけ出力先名を上書きします（`main_lora` 以外の実験用名OK）。
              それ以外の mode では role 既定名（`ADAPTER_BY_ROLE`）を使います
            - **GPU は H100 / A100 (80GB) 前提**なので `USE_4BIT = False`（bf16）が既定です。
              e4b の bf16 学習 + AdamW で約 56 GB 必要ですが 80 GB に収まります。L4 / T4 (11 / 15GB) を選んだときだけ
              `USE_4BIT = True` にし、`MAX_LEN` も 6144 以下に下げて OOM に備えてください
            - `MAX_LEN = 10240`（1ステップの token 上限）/ `MAX_EXAMPLES = 200`（教材の会話数上限）は cell 5 にそのまま渡ります。
              `train_lora.py` は **Decision-Step SFT（1 assistant turn 単位 + 中間ターン自動トリミング + 過去思考除去）** で学習するため、
              長い軌跡でも末尾の `edit_file` や `submit_patch` が切り捨てられず、本番の `include_thoughts: false` と完全一致します
            - `LEARNING_RATE = 5e-5` / Cosine LR スケジューラ / `NEFTUNE_ALPHA = 5.0`（埋め込みノイズ正則化で反復ループ暗記を防止）
            - `LORA_PLUS_RATIO = 4.0`: `lora_B` の学習率を上げて tool 呼び出しの収束を早める倍率。`16.0` 以上に振ると
              tool call が `<|tool_call>` ではなく `<|turn>model{...}` というただの文字列に固まるので上げない
            - `USE_RSLORA`: 既定 `False`。rank を 64 以上に上げるときだけ検討する
            - `SFT_EPOCHS = 2.0`: 105 件超の教材を 3 周させると tool call の「形式」そのものを暗記し始めるので抑えめ
            - `RUN_DPO` / `DPO_EPOCHS`: Stage 2（DPO）を回すか。**既定は `False`（まずは歪みのない Stage 1 RFT-SFT アダプタで
              評価確認し、解決トレースが増えてから `True` にするのが安全）**。DPO を掛けるのは **main だけ**です
            - `MAX_TOOL_CALLS = 45` / `MAX_TURNS = 60`: `submission/eval_config.yaml` と同じ予算を教材の user プロンプトに書き写す値。
              教材と実運用で budget がズレると「予算 80 前提の行動」を覚えてしまいます

            出力先は `submission/adapters/<adapter>`（cell 3 の末尾で role ごとに組み立てます）。
            """
        ),
        _code(
            """
            #@title 3. 学習設定（BUILD_MODE で LoRA の本数を決める）
            BUILD_MODE = "main+analyzer"      #@param ["main", "main+analyzer", "all"]
            ADAPTER_NAME = "main_lora"        #@param {type:"string"}
            R = 16                            #@param {type:"integer"}
            ALPHA = 16                        #@param {type:"integer"}
            USE_RSLORA = False                #@param {type:"boolean"}
            LORA_PLUS_RATIO = 1.0             #@param {type:"number"}
            NEFTUNE_ALPHA = 5.0               #@param {type:"number"}
            USE_4BIT = False                  #@param {type:"boolean"}
            LEARNING_RATE = "3e-5"            #@param {type:"string"}
            MAX_LEN = 10240                   #@param {type:"integer"}
            MAX_EXAMPLES = 200                #@param {type:"integer"}
            SFT_EPOCHS = 1.0                  #@param {type:"number"}
            RUN_DPO = False                   #@param {type:"boolean"}
            DPO_EPOCHS = 1.0                  #@param {type:"number"}
            MAX_TOOL_CALLS = 45               #@param {type:"integer"}
            MAX_TURNS = 60                    #@param {type:"integer"}
            TASKS_JSONL = "data/tasks.jsonl"  #@param {type:"string"}
            RESULTS_DIR = "results"           #@param {type:"string"}

            ADAPTER_BY_ROLE = {"main": "main_lora", "analyzer": "analyzer_lora", "verifier": "verifier_lora"}
            ROLE_BY_DATA = {"main": "main", "analyzer": "analyzer", "verifier": "verifier"}

            if BUILD_MODE == "all":
                ROLES = ["main", "analyzer", "verifier"]
            elif BUILD_MODE == "main+analyzer":
                ROLES = ["main", "analyzer"]
            else:
                ROLES = ["main"]

            ADAPTERS = [ADAPTER_BY_ROLE[role] for role in ROLES]
            if ROLES == ["main"] and ADAPTER_NAME:
                # 単独 main の実行だけ、出力先名をそのまま変更可（ADAPTER_NAME は main と別名でも試せる）
                ADAPTERS = [ADAPTER_NAME]
            ADAPTER_BY_ROLE = {role: adapter for role, adapter in zip(ROLES, ADAPTERS)}

            OUT_DIR_BY_ROLE = {role: f"submission/adapters/{ADAPTER_BY_ROLE[role]}" for role in ROLES}
            DATA_BY_ROLE = {role: f"data/trajectories_{ROLE_BY_DATA[role]}.jsonl" for role in ROLES}
            DPO_BY_ROLE = {role: f"data/dpo_pairs_{ROLE_BY_DATA[role]}.jsonl" for role in ROLES}

            extra_lora_flags = []
            if USE_RSLORA:
                extra_lora_flags.append("--use_rslora")
            if LORA_PLUS_RATIO > 1.0:
                extra_lora_flags += ["--lora_plus_ratio", str(LORA_PLUS_RATIO)]
            if USE_4BIT:
                extra_lora_flags.append("--use_4bit")
            EXTRA_FLAGS = " ".join(extra_lora_flags)

            print(f"BUILD_MODE={BUILD_MODE} roles={ROLES} adapters={ADAPTERS} RUN_DPO={RUN_DPO}")
            print(f"precision={'4bit' if USE_4BIT else 'bf16 (A100/H100)'} MAX_LEN={MAX_LEN} NEFTUNE={NEFTUNE_ALPHA} EXTRA_FLAGS={EXTRA_FLAGS or '(none)'}")
            for role in ROLES:
                print(f"  {role:8s} {DATA_BY_ROLE[role]} -> {OUT_DIR_BY_ROLE[role]}")
            """
        ),
        _md(
            """
            ## Cell 4 — Stage 1a: role ごとに教材を収集（`collect_trajectories.py`）

            `BUILD_MODE` の role 数だけ実行します。出力は role 別ファイル（`data/trajectories_main.jsonl` /
            `data/trajectories_analyzer.jsonl` / `data/trajectories_verifier.jsonl`）で、role を分けないと
            後から走らせた role が前の教材を上書きしてしまいます。

            - `--role main` ... **read_file → edit_file → pytest → submit_patch** の順の軌跡。
              gold patch の hunk 範囲を読む `read_file` を edit_file の前に必ず入れるので、
              「見ていない行を old_string に書いて一致しない」巡りを教材側から消せます
              （`--max-hunk-lines` 既定 40 超の hunk は 1 発の edit_file として非現実的なのでその task は skip。件数優先なら 60）
            - `--role analyzer` ... read-only の局所化のみ（grep → search_similar_code → 範囲 read → 報告）。
              edit_file / write_file / submit_patch は 1 回も出てきません
            - `--role verifier` ... 対象 test だけ実行して `VERDICT: GO` を返す報告のみ
            - `results/` が有るときは `--import-traces results --only-resolved` を main に付けます。
              実 trace（`steps[].source` / `steps[].tool_calls` の harness 形式）は resolved が
              `task_results.jsonl` 側にあるので、解決フラグを読めない分は gold 合成が埋めます
              （analyzer / verifier には付けない: resolved 済み trace はほぼ main の編集軌跡だから）
            - `--max-tool-calls` / `--max-turns` は `submission/eval_config.yaml`（45 / 60）と揃えてあります

            cell 2 で `data/tasks.jsonl` が取れていないと gold 合成が空振りするので、その場合は cell 2 を先にやり直してください。
            """
        ),
        _code(
            """
            #@title 4. Stage 1a — role ごとに教材を収集
            import subprocess
            from pathlib import Path


            def run_stream(cmd: str) -> int:
                proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                assert proc.stdout is not None
                for line in proc.stdout:
                    print(line, end="", flush=True)
                return proc.wait()


            for role in ROLES:
                # resolved 済み trace を main の RFT に足す（analyzer / verifier は gold 合成のみで十分）
                import_arg = ""
                if Path(RESULTS_DIR).is_dir() and role == "main":
                    import_arg = f"--import-traces {RESULTS_DIR} --only-resolved "
                cmd = (f"python3 src/collect/collect_trajectories.py --role {role} "
                       f"--tasks {TASKS_JSONL} {import_arg}"
                       f"--max-tool-calls {MAX_TOOL_CALLS} --max-turns {MAX_TURNS} "
                       f"--output {DATA_BY_ROLE[role]}")
                print("[collect]", cmd, flush=True)
                if run_stream(cmd) != 0:
                    print(f"[stage1a] collect failed for role={role}（cell 5 はこの role を skip します）")

            for role in ROLES:
                data_file = Path(DATA_BY_ROLE[role])
                count = len(data_file.read_text(encoding="utf-8").splitlines()) if data_file.is_file() else 0
                print(f"[stage1a] {role:8s} {count} 件 -> {data_file}")
            """
        ),
        _md(
            """
            ## Cell 5 — Stage 1b: LoRA 学習（`train_lora.py` / RFT-SFT）

            role 順に `main` → `analyzer` → `verifier` で学習します。出力先は cell 3 の `OUT_DIR_BY_ROLE`
            （`submission/adapters/main_lora` など）で、`train_lora.py` が保存時の `adapter_config.json` を
            本番名（`google/gemma-4-e4b-it-qat-w4a16-ct`）に書き換えてくれます。

            ### cell 5 が必ず渡す値（tool call 形式と推論履歴整合を守る設定）

            - `--lr 5e-5` + `--lr_scheduler cosine` + `--neftune_alpha 5.0` — 特殊 token を壊さず反復暗記を抑える
            - `--max_len 10240` — 1 ステップの token 上限（長い会話は中間ターンを自動トリミングし、末尾の `edit_file` / `submit_patch` を必ず保持）
            - `main` は過去ターンの思考を除去（`include_thoughts: false` 整合）、`analyzer` は `--keep_prior_thoughts`（`include_thoughts: true` 整合）
            """
        ),
        _code(
            """
            #@title 5. Stage 1b — LoRA 学習（role 順に main → analyzer → verifier）
            from pathlib import Path

            for role in ROLES:
                data_file, out_dir = DATA_BY_ROLE[role], OUT_DIR_BY_ROLE[role]
                if not Path(data_file).is_file() or Path(data_file).stat().st_size == 0:
                    print(f"[stage1b] [skip] 教材が無い: {data_file}")
                    continue
                cmd = (f"python3 src/train/train_lora.py --data {data_file} --out_dir {out_dir} "
                       f"--r {R} --alpha {ALPHA} --lr {LEARNING_RATE} --epochs {SFT_EPOCHS} "
                       f"--neftune_alpha {NEFTUNE_ALPHA} --log_every 2 "
                       f"--max_len {MAX_LEN} --max_examples {MAX_EXAMPLES} --keep_prior_thoughts {EXTRA_FLAGS}")
                print("[sft]", cmd, flush=True)
                if run_stream(cmd) != 0:
                    print(f"[stage1b] train_lora failed for role={role}（cell 6 の DPO は main だけ試します）")
            """
        ),
        _md(
            """
            ## Cell 6 — Stage 2: DPO（`build_dpo_pairs.py` → `train_dpo.py`、**main のみ**）

            選好ペアは `--results results` の実 trace を rejected（解決しなかった / ループして打ち切られた軌跡）、
            同じ task の gold 合成 SFT または resolved 済み trace を chosen に作ります（`--role main`）。
            生成先は `data/dpo_pairs_main.jsonl`。

            - **既定は `RUN_DPO = False` です。** まずは Stage 1 の `main_lora` + `analyzer_lora` で評価を通し、
              DPO を試す場合は cell 3 で `RUN_DPO = True` に切り替えてください
            - `build_dpo_pairs.py` は `edit_file` 同士の対比や検証後ループに対する `submit_patch` を chosen に含め、
              `train_dpo.py` は長さ正規化 DPO（`sigmoid_norm`）と過去思考除去で `submit_patch` 抑制を防ぎます
            """
        ),
        _code(
            """
            #@title 6. Stage 2 — DPO（main だけ / RUN_DPO=True のときのみ実行）
            from pathlib import Path

            if not (RUN_DPO and "main" in ROLES):
                print("[stage2] [skip] RUN_DPO=False（Stage 1 RFT-SFT アダプタをそのまま使用します）")
            else:
                out_dir, dpo_pairs = OUT_DIR_BY_ROLE["main"], DPO_BY_ROLE["main"]
                if not Path(out_dir).exists():
                    print(f"[stage2] [skip] Stage 1 adapter が無い: {out_dir}")
                elif not Path(RESULTS_DIR).is_dir():
                    print(f"[stage2] [skip] {RESULTS_DIR}/ が無い（rejected になる eval trace が無い）")
                else:
                    build = (f"python3 src/collect/build_dpo_pairs.py --results {RESULTS_DIR} "
                             f"--tasks {TASKS_JSONL} --role main --output {dpo_pairs}")
                    print("[dpo]", build, flush=True)
                    if run_stream(build) != 0:
                        print("[stage2] build_dpo_pairs failed")
                    if Path(dpo_pairs).is_file() and Path(dpo_pairs).stat().st_size > 0:
                        train = (f"python3 src/train/train_dpo.py --data {dpo_pairs} "
                                 f"--resume_from {out_dir} --out_dir {out_dir} "
                                 f"--r {R} --alpha {ALPHA} --max_len {MAX_LEN} --epochs {DPO_EPOCHS} {EXTRA_FLAGS}")
                        print("[dpo]", train, flush=True)
                        if run_stream(train) != 0:
                            print("[stage2] train_dpo failed")
                    else:
                        print(f"[stage2] [skip] 選好ペアが 0 件: {dpo_pairs}")
            """
        ),
        _md(
            """
            ## Cell 7 — 検証（adapter ファイル & 提出契約チェック）

            学習した role ごとに `submission/adapters/<name>/adapter_config.json` と
            `adapter_model*.safetensors` が書けているかを確認し、`src/package/build_submission.py
            --source submission --check-only` で提出契約（単一モデル宣言 / 単一データセット /
            `agent.yaml` の `adapter:` とファイルの対応）を通します。

            - `agent.yaml` は `adapter: main_lora`、`sub_agents/code_analyzer.yaml` は `adapter: analyzer_lora`
              を指します。`BUILD_MODE = "main+analyzer"` で作れば両方揃うはずです
            - 足りない adapter がある場合は `[check] MISSING` を出します（そのまま提出するとその agent は base モデルで走る）
            - Colab のディスクは消えるので、`submission/` ごと zip でダウンロードしてローカルで最終
              `submission.zip` を組みます（`agent.yaml` / `sub_agents/` / `configs/` も全部必要です）
            """
        ),
        _code(
            """
            #@title 7. 検証 — adapter ファイル & 提出契約チェック
            import glob
            import os

            for role, out_dir in OUT_DIR_BY_ROLE.items():
                weights = sorted(glob.glob(f"{out_dir}/adapter_model*.safetensors"))
                config = os.path.isfile(f"{out_dir}/adapter_config.json")
                state = "OK" if (config and weights) else "MISSING"
                print(f"[check] {role:8s} {out_dir}: config={config} weights={len(weights)} {state}")
                for extra in sorted(glob.glob(f"{out_dir}/*"))[:6]:
                    print(f"          {extra} ({os.path.getsize(extra)} bytes)")

            if os.system("python3 src/package/build_submission.py --source submission --check-only") != 0:
                print("[check] build_submission failed（agent.yaml の adapter 名と cell 3 の ADAPTER_NAME を確認）")
            """
        ),
    ]
    return nb


def main() -> int:
    p = argparse.ArgumentParser(description="Generate Colab LoRA training notebook")
    p.add_argument("--output", default="kernels/colab/gemma4_lora_train_colab.ipynb")
    args = p.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    nb = build_notebook()
    for index, cell in enumerate(nb.cells, start=1):
        if cell.cell_type != "code":
            continue
        try:
            _check_python(cell.source)
        except SyntaxError as exc:
            raise SystemExit(f"[error] cell {index} is not valid python ({exc}):\n{cell.source}") from exc
    nbf.validate(nb)
    nbf.write(nb, out)
    print(f"[notebook] wrote {out} ({len(nb.cells)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
