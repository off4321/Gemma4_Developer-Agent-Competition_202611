#!/usr/bin/env python3
"""Generate the Colab RFT + DPO Multi-LoRA training notebook (`kernels/colab/gemma4_lora_train_colab.ipynb`).

Keeps the notebook clean and reproducible from source, mirroring `make_colab_notebook.py`
without touching the evaluation notebook.
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
    nb.metadata["colab"] = {"provenance": [], "gpuType": "L4"}

    nb.cells = [
        _md(
            """
            # Gemma 4 e4b — RFT + DPO Multi-LoRA Training (Colab)

            Colab 上でリポジトリを clone し、**RFT（自己成功軌跡＋gold合成 SFT）→ DPO（成功/失敗ペア選好学習）** で
            `main_lora` / `analyzer_lora` / `verifier_lora` の 3 本を作成・検証するためのノートブックです。
            1 セット（cell 1 → 6）で 1 本学習できるので、`ADAPTER_NAME` / `ROLE` を切り替えて 3 周してください。

            - **RSLoRA** (`--use_rslora`): 高ランクでも勾配が潰れにくい（既定 OFF。rank を 64 以上に上げるときだけ ON）
            - **LoRA+** (`--lora_plus_ratio 4.0`): `lora_B` の学習率を上げてツール形式の収束を早める。倍率を稼ぎすぎると
              （`16.0` 以上）tool call がただの文字列として学習され、tool call 崩れの原因になる（学習率の既定は `--lr 5e-5`）
            - 学習ベースは `google/gemma-4-e4b-it`、保存時の `adapter_config.json` は本番の `google/gemma-4-e4b-it-qat-w4a16-ct` に自動書き換えされます

            cell 1 → 2 → 3 → 4 → 5 → 6 の順に実行します。cell 1 の `pip install` で Colab が「ランタイムを再起動」と
            出したら、再起動してから cell 2 以降を続けてください（未再起動だと transformers / peft の import が
            混ざって落ちます）。`data/tasks.jsonl` は cell 2 が Kaggle から自動 download します（token が無いときだけ
            Files パネルにアップロード）。eval 軌跡入りの `results/` は相変わらず手動アップロードです。
            """
        ),
        _md(
            """
            ## Cell 1 — GPU確認 & 依存インストール

            `nvidia-smi` で L4 GPU が当たっているかを確認し、`pip install` で `peft` / `datasets` / `trl` /
            `accelerate` / `bitsandbytes` / `pyyaml` / `kaggle` を導入します。

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
            | `HF_TOKEN` | `HF_TOKEN` | `HF_TOKEN` | cell 4 の gated model 読み込みが 401 になる |

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
            | `results/` | eval kernel の出力（`kaggle kernels output <kernel> -p results/` で自分の kernel から取る） | cell 4 は gold 合成のみ、cell 5 の DPO は skip |

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

            REPO_URL = "https://github.com/<you>/Gemma4_Developer-Agent-Competition_202611.git"  #@param {type:"string"}
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
                !git clone {REPO_URL} {REPO_DIR}
            else:
                !git -C {REPO_DIR} pull --ff-only

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
                print("[auth] HF_TOKEN を kernel env に設定（cell 4 の gated model 読み込みに要る）")

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
            ## Cell 3 — 学習設定（`#@param` フォームで Multi-LoRA / RSLoRA / LoRA+ / DPO）

            **コードを書かずに学習設定を切り替えるフォーム cell** です。下の値を変えるだけで、cell 4 / 5 に渡す
            CLI 引数が組み立てられます。

            - `ADAPTER_NAME` / `ROLE`: Multi-LoRA の作成対象。3 LoRA（`main_lora` / `analyzer_lora` / `verifier_lora`）を
              それぞれ 1 本ずつこの notebook で作ります。`ADAPTER_NAME` と `ROLE` は必ず組み合わせ揃えます
              （`main_lora`↔`main` / `analyzer_lora`↔`analyzer` / `verifier_lora`↔`verifier`）。
              `ROLE` で system prompt（`submission/prompts/{system,analyzer,verifier}.md`）と教材データが決まります
            - `TRAJECTORIES_JSONL` / `DPO_PAIRS`: role 単位の学習データパス。cell 3 の末尾で `ROLE` から自動計算する:
              `data/trajectories.jsonl` / `data/trajectories_analyzer.jsonl` / `data/trajectories_verifier.jsonl` と
              `data/dpo_pairs_main.jsonl` / `data/dpo_pairs_analyzer.jsonl` / `data/dpo_pairs_verifier.jsonl`
              （main は従来の `data/trajectories.jsonl` のまま）
            - `R` / `ALPHA`: LoRA ランクとスケーリング。既定 `16` / `32`（`ALPHA = 2 * R` 目安）。105 件程度の軌跡では rank より学習率の
              方が tool call 形式に効くので、まず既定の小さい方から触る
            - `USE_RSLORA`: RSLoRA（`--use_rslora`）。高ランクでも勾配が潰れにくいが、既定は `False`（4bit 基底 + LoRA+ で
              実効 lr が立ちすぎるため）。rank を 64 以上に上げる検証のときだけ `True`
            - `LORA_PLUS_RATIO`: LoRA+ の倍率（`--lora_plus_ratio`）。`lora_B` の学習率を上げてツール呼び出しの収束を早めます。1.0 で無効。
              既定 `4.0`（`16.0` に振ると tool call の直列化が `<|tool_call>` ではなく `<|turn>model{...}` という文字列として
              固まる。`main_lora` の tool call 崩れはこれが本体）
            - 学習率 `--lr` は cell 4 で `5e-5` を明示する（`train_lora.py` の既定も `5e-5`。1e-4 を 4bit + LoRA+ で当てると
              特殊 token の embedding が飛ぶ）。cell 4 は併せて `--max_len 6144`（1 会話の token 上限）と `--max_examples 200`
              （教材の会話数上限）で締める — L4 15GB で QLoRA が落ちず、105 件超の周回で tool call 形式だけ暗記しないため
            - `USE_4BIT`: 4bit 量子化（`--use_4bit`）。既定 `True` — L4 15GB に bf16 の e4b は載らず、cell 4 / 5 は QLoRA 前提。
              bf16 で載る GPU で検証するときだけ `False`
            - `SFT_EPOCHS` / `RUN_DPO` / `DPO_EPOCHS`: Stage 1 のエポック数（既定 `2.0` — 105 件で 3 周すると tool call の
              形式そのものが過学習で崩れる）、Stage 2（DPO）を回すか、そのエポック数
            - `TASKS_JSONL` / `RESULTS_DIR`: cell 2 で置いた `data/tasks.jsonl` と `results/` のパス

            出力先 `OUT_DIR` は `submission/adapters/<ADAPTER_NAME>` 固定です（Kaggle 側で学習した場合も同じ相対パスに
            なるので、cell 4 / 5 のコマンドはそのままで差し替えできます）。
            """
        ),
        _code(
            """
            #@title 3. 学習設定（Multi-LoRA / RSLoRA / LoRA+ / DPO）
            ADAPTER_NAME = "main_lora"          #@param ["main_lora", "analyzer_lora", "verifier_lora"]
            ROLE = "main"                       #@param ["main", "analyzer", "verifier"]
            R = 16                              #@param {type:"integer"}
            ALPHA = 32                          #@param {type:"integer"}
            USE_RSLORA = False                  #@param {type:"boolean"}
            LORA_PLUS_RATIO = 4.0               #@param {type:"number"}
            USE_4BIT = True                     #@param {type:"boolean"}
            SFT_EPOCHS = 2.0                    #@param {type:"number"}
            RUN_DPO = True                      #@param {type:"boolean"}
            DPO_EPOCHS = 1.0                    #@param {type:"number"}

            TASKS_JSONL = "data/tasks.jsonl"    #@param {type:"string"}
            RESULTS_DIR = "results"             #@param {type:"string"}
            OUT_DIR = f"submission/adapters/{ADAPTER_NAME}"

            # role ごとに教材を分ける（main は従来の data/trajectories.jsonl のまま）
            role_suffix = "" if ROLE == "main" else f"_{ROLE}"
            TRAJECTORIES_JSONL = f"data/trajectories{role_suffix}.jsonl"
            DPO_PAIRS = f"data/dpo_pairs_{ROLE}.jsonl"

            extra_lora_flags = []
            if USE_RSLORA:
                extra_lora_flags.append("--use_rslora")
            if LORA_PLUS_RATIO > 1.0:
                extra_lora_flags += ["--lora_plus_ratio", str(LORA_PLUS_RATIO)]
            if USE_4BIT:
                extra_lora_flags.append("--use_4bit")
            EXTRA_FLAGS = " ".join(extra_lora_flags)
            print("OUT_DIR:", OUT_DIR, "| EXTRA_FLAGS:", EXTRA_FLAGS)
            """
        ),
        _md(
            """
            ## Cell 4 — Stage 1: `collect_trajectories.py` で RFT/SFT データ作成 → `train_lora.py` で LoRA 学習

            1. `src/collect/collect_trajectories.py --role {ROLE}` が学習データを 1 本作ります。出力先は cell 3 の `TRAJECTORIES_JSONL`:
               `main` → `data/trajectories.jsonl`、`analyzer` → `data/trajectories_analyzer.jsonl`、
               `verifier` → `data/trajectories_verifier.jsonl`（role で別々にしておかないと、3 本目の学習で前の教材が上書きされます）。
               `--role` で使う system prompt と教材の形が変わります:
               `main` = edit_file → pytest → submit_patch、`analyzer` = read-only の局所化報告、`verifier` = 対象 test のみ実行して
               `VERDICT: GO` の報告（analyzer / verifier は edit_file / write_file / submit_patch を教材にしない）。
               `results/` があるときだけ `--import-traces results --only-resolved` を付けて、**解けた（resolved=True）
               軌跡だけを RFT として import** し、足りない分は tasks.jsonl の gold patch から合成した SFT 軌跡で補います。
            2. `src/train/train_lora.py` が cell 3 の設定（RSLoRA / LoRA+ / 4bit）で LoRA を学習し、`OUT_DIR`
               （`submission/adapters/<ADAPTER_NAME>`）に保存します。

            ### cell 4 が `train_lora.py` に渡す固定引数（tool call を壊さないための値）

            - `--lr 5e-5` — 既定値と同じ。4bit 基底 + LoRA+ に `1e-4` を当てると tool call / turn の特殊 token が飛んで、
              生成が `tool_call` ではなく `<|turn>model{...}` というただの文字列に落ちる
            - `--max_len 6144` — L4 15GB の QLoRA で落ちない長さの上限。main 軌跡は system 指示 + 観測で伸びるが、
              8192 にすると OOM する。超えた末尾 turn は切り捨てる
            - `--max_examples 200` — 教材を 200 会話で打ち切る。105 件を超えて周回させると tool call の「形式」だけを
              暗記して、中身の JSON が壊れ始める

            試走させたいときは `collect_trajectories.py` に `--limit N` を付け、`train_lora.py` の `--max_examples` を下げて
            件数を絞ってください（初回は tokenizer 読み込みだけで数分かかります）。`train_lora.py` は学習開始前に
            `[data] sample supervised text:` として 1 件目の supervised 部分だけを decode した text を print します。
            ここに tool call が正しく直列化されていなければ、学習を止めて教材側を見ることになります。
            """
        ),
        _code(
            """
            #@title 4. Stage 1 — RFT/SFT データ作成 & LoRA 学習
            from pathlib import Path

            import_arg = f"--import-traces {RESULTS_DIR} --only-resolved" if Path(RESULTS_DIR).exists() else ""
            !python3 src/collect/collect_trajectories.py \\
                --role {ROLE} \\
                --tasks {TASKS_JSONL} \\
                {import_arg} \\
                --output {TRAJECTORIES_JSONL}
            print("[stage1] training data:", TRAJECTORIES_JSONL, "-> adapter:", OUT_DIR)

            # train_lora.py の引数は tool call 保全が最優先（上げる方向に振ると形式が壊れる）:
            #   --lr 5e-5          4bit 基底 + LoRA+ で 1e-4 を当てると特殊 token の embedding が飛ぶ（既定も 5e-5）
            #   --max_len 6144     L4 15GB の QLoRA で落ちない長さ。これを超えた末尾 turn は切り捨てる
            #   --max_examples 200 教材を 200 会話で打ち切る（105 件超の周回だと tool call の形式だけ暗記する）
            !python3 src/train/train_lora.py \\
                --data {TRAJECTORIES_JSONL} \\
                --out_dir {OUT_DIR} \\
                --r {R} --alpha {ALPHA} \\
                --epochs {SFT_EPOCHS} \\
                --lr 5e-5 \\
                --max_len 6144 \\
                --max_examples 200 \\
                {EXTRA_FLAGS}
            """
        ),
        _md(
            """
            ## Cell 5 — Stage 2: `build_dpo_pairs.py` で選好ペア生成 → DPO 事後学習

            1. `src/collect/build_dpo_pairs.py --role {ROLE}` が eval 軌跡から `prompt` / `chosen` / `rejected` の選好ペアを作ります。
               出力は cell 3 の `DPO_PAIRS`（= `data/dpo_pairs_{ROLE}.jsonl`）で、Stage 1 と同じ role のペアだけを使います
               （analyzer / verifier のペアに main の edit_file 軌跡が混ざることはない）。
               chosen は `resolved=True` の軌跡（無ければ tasks.jsonl から gold 合成）、rejected は同じ task の
               解決できなかった軌跡（テスト失敗・ERROR/TIMEOUT・NO_PATCH・過剰ターン打ち切り）です。1 task あたり
               `--max-pairs-per-task`（既定 2）まで。
            2. ペアファイルが実際に生成できたときだけ `src/train/train_dpo.py` を走らせ、`--resume_from` で Stage 1 の
               adapter を初期値にして上から DPO を掛けます。cell 3 の `EXTRA_FLAGS`（`--use_rslora` /
               `--lora_plus_ratio` / `--use_4bit`）をそのまま渡すので、Stage 1 と同じ LoRA 構造になります。

            `RUN_DPO = False` または `results/` が無い場合は実行せず `[skip]`、ペアが 0 件なら `[skip]` です。
            DPO は Stage 1 より学習率が小さく（既定 5e-6）、`--beta` は 0.1 固定です。
            LoRA の形は `EXTRA_FLAGS` 経由で Stage 1 と揃う（`--lora_plus_ratio 4.0` と `--use_4bit`、`--use_rslora` は既定 OFF）ので、
            Stage 1 の adapter にそのまま `--resume_from` できます。

            token budget だけ Stage 1 と別です。Stage 1 は cell 4 で `--max_len 6144` / `--max_examples 200` に締めていますが、
            `train_dpo.py` は自前の `--max_len`（既定 4096）で prompt / chosen / rejected を切り捨て、
            `--max_examples`（既定は全件）でペア数を見ます。chosen / rejected が途中で切れているときは
            cell 5 のコマンドに `--max_len 6144` を足してください（L4 で OOM したら下げる）。
            """
        ),
        _code(
            """
            #@title 5. Stage 2 — DPO 事後学習（成功 vs 失敗/ループ軌跡）
            from pathlib import Path

            DPO_PAIRS = f"data/dpo_pairs_{ROLE}.jsonl"

            if RUN_DPO and Path(RESULTS_DIR).exists():
                !python3 src/collect/build_dpo_pairs.py \\
                    --results {RESULTS_DIR} \\
                    --tasks {TASKS_JSONL} \\
                    --role {ROLE} \\
                    --output {DPO_PAIRS}

                # build_dpo_pairs が失敗すると出力が無いので、stat する前に存在を見る
                if Path(DPO_PAIRS).is_file() and Path(DPO_PAIRS).stat().st_size > 0:
                    !python3 src/train/train_dpo.py \\
                        --data {DPO_PAIRS} \\
                        --resume_from {OUT_DIR} \\
                        --out_dir {OUT_DIR} \\
                        --r {R} --alpha {ALPHA} \\
                        --epochs {DPO_EPOCHS} \\
                        {EXTRA_FLAGS}
                else:
                    print("[skip] DPO pairs empty (need eval traces under results/ first)")
            else:
                print("[skip] RUN_DPO=False or results/ not found")
            """
        ),
        _md(
            """
            ## Cell 6 — 提出契約チェック（`build_submission.py --check-only`）

            `src/package/build_submission.py --source submission --check-only` が、`submission/` が提出要件を満たすかを
            zip にせずに検査します（ルート直下の config が 1 個だけか / 禁止拡張や symlink / 解凍後 3 GiB 以内 /
            agent YAML の model・thinking・max_output_tokens / 単一モデルルール / adapter 名と `adapter_config.json`）。
            続けて `ls -lh {OUT_DIR}` で `OUT_DIR` の中身（adapter ファイル）を表示します。

            Colab のディスクは消えるので、`submission/`（`adapters/main_lora` / `adapters/analyzer_lora` /
            `adapters/verifier_lora` の adapter ファイル含む）を
            zip にしてダウンロードし、ローカル側で最終的な `submission.zip` をビルドしてください。
            """
        ),
        _code(
            """
            #@title 6. 提出契約チェック（submission/adapters/<name>）
            !python3 src/package/build_submission.py --source submission --check-only
            !ls -lh {OUT_DIR}
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
