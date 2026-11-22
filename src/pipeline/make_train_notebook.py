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
            `main_lora` / `analyzer_lora` を作成・検証するためのノートブックです。

            - **RSLoRA** (`--use_rslora`): 高ランクでも勾配が潰れにくい
            - **LoRA+** (`--lora_plus_ratio 16.0`): `lora_B` の学習率を上げてコード・ツール形式の収束を高速化
            - 学習ベースは `google/gemma-4-e4b-it`、保存時の `adapter_config.json` は本番の `google/gemma-4-e4b-it-qat-w4a16-ct` に自動書き換えされます

            cell 1 → 2 → 3 → 4 → 5 → 6 の順に実行します。cell 1 の `pip install` で Colab が「ランタイムを再起動」と
            出したら、再起動してから cell 2 以降を続けてください（未再起動だと transformers / peft の import が
            混ざって落ちます）。`data/tasks.jsonl` と eval 軌跡入りの `results/` は cell 2 の clone 後にアップロードします。
            """
        ),
        _md(
            """
            ## Cell 1 — GPU確認 & 依存インストール

            `nvidia-smi` で L4 GPU が当たっているかを確認し、`pip install` で `peft` / `datasets` / `trl` /
            `accelerate` / `bitsandbytes` / `pyyaml` を導入します。

            **インストール後に Colab が「ランタイムを再起動」するダイアログを出したら、必ず再起動してから
            cell 2 以降を実行してください。** 再起動を飛ばすと Colab 既定版と新規版のライブラリが混ざって
            import が壊れます。
            """
        ),
        _code(
            """
            #@title 1. GPU確認 & 依存インストール
            !nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
            !pip install -q -U peft datasets trl accelerate bitsandbytes pyyaml
            """
        ),
        _md(
            """
            ## Cell 2 — リポジトリの clone / 更新 & HF ログイン（Colab Secrets の `HF_TOKEN`）

            `REPO_URL` を自分のリポジトリに向けて clone（2回目以降は `git pull --ff-only`）し、`/content/repo` に cd します。
            `#@param` の `REPO_URL` だけ自分の GitHub URL に書き換えれば後はそのままで動きます。

            HF トークンは**コードに直書きせず Colab Secrets の `HF_TOKEN`** から読みます。
            右側のファイルパネルの 🔑（Secrets / パスワード）タブに `HF_TOKEN` という名前のシークレットを登録しておくと、
            `userdata.get("HF_TOKEN")` で取り出して環境変数にセットします（ノートブック本体にトークンは残りません）。

            ### clone 後にアップロードするファイル（この cell の **後** で OK）

            | パス | 中身 | 無い場合 |
            | --- | --- | --- |
            | `data/tasks.jsonl` | gold patch / test patch 入りの評価タスク定義（JSONL） | gold 合成 SFT が作れず、cell 4 が空振りします |
            | `results/` | eval kernel の出力（`kaggle kernels output <kernel> -p results/` で取得した JSON 一式） | cell 4 は gold 合成 SFT のみ、cell 5 の DPO は skip になります |

            いずれもリポジトリ直下（`/content/repo/data/tasks.jsonl`、`/content/repo/results/`）に置きます。
            ファイルパネルへのドラッグ&ドロップが一番手っ取り早く、`files.upload()` でも置けます。

            ```python
            from google.colab import files

            files.upload()  # tasks.jsonl を /content にアップロード → !mv -f /content/tasks.jsonl data/tasks.jsonl
            ```
            """
        ),
        _code(
            """
            #@title 2. リポジトリの clone / 更新 & HF ログイン
            import os
            from pathlib import Path

            REPO_URL = "https://github.com/<you>/gemma4-dev-agent.git"  #@param {type:"string"}
            REPO_DIR = Path("/content/repo")

            if not (REPO_DIR / ".git").exists():
                !git clone {REPO_URL} {REPO_DIR}
            else:
                !git -C {REPO_DIR} pull --ff-only

            %cd {REPO_DIR}

            # Gemma 4 重み取得用トークン（Colab Secrets の HF_TOKEN を推奨）
            try:
                from google.colab import userdata
                hf_token = userdata.get("HF_TOKEN")
                if hf_token:
                    os.environ["HF_TOKEN"] = hf_token
                    print("[auth] HF_TOKEN loaded from Colab Secrets")
            except Exception:
                pass
            """
        ),
        _md(
            """
            ## Cell 3 — 学習設定（`#@param` フォームで Multi-LoRA / RSLoRA / LoRA+ / DPO）

            **コードを書かずに学習設定を切り替えるフォーム cell** です。下の値を変えるだけで、cell 4 / 5 に渡す
            CLI 引数が組み立てられます。

            - `ADAPTER_NAME` / `ROLE`: Multi-LoRA の作成対象。`main_lora`（`ROLE=main`）と `analyzer_lora`（`ROLE=analyzer`）を
              それぞれこの notebook で 1 本ずつ作ります
            - `R` / `ALPHA`: LoRA ランクとスケーリング。`ALPHA = 2 * R` 目安
            - `USE_RSLORA`: RSLoRA（`--use_rslora`）。高ランクでも勾配が潰れにくく、rank を変えても学習率を再調整しなくて済みます
            - `LORA_PLUS_RATIO`: LoRA+ の倍率（`--lora_plus_ratio`）。`lora_B` の学習率を上げてツール呼び出しの収束を早めます。1.0 で無効
            - `USE_4BIT`: 4bit 量子化（`--use_4bit`）。L4 15GB に収まらないときだけ有効化
            - `SFT_EPOCHS` / `RUN_DPO` / `DPO_EPOCHS`: Stage 1 のエポック数、Stage 2（DPO）を回すか、そのエポック数
            - `TASKS_JSONL` / `RESULTS_DIR`: cell 2 で置いた `data/tasks.jsonl` と `results/` のパス

            出力先 `OUT_DIR` は `submission/adapters/<ADAPTER_NAME>` 固定です（Kaggle 側で学習した場合も同じ相対パスに
            なるので、cell 4 / 5 のコマンドはそのままで差し替えできます）。
            """
        ),
        _code(
            """
            #@title 3. 学習設定（Multi-LoRA / RSLoRA / LoRA+ / DPO）
            ADAPTER_NAME = "main_lora"          #@param ["main_lora", "analyzer_lora"]
            ROLE = "main"                       #@param ["main", "analyzer"]
            R = 32                              #@param {type:"integer"}
            ALPHA = 64                          #@param {type:"integer"}
            USE_RSLORA = True                   #@param {type:"boolean"}
            LORA_PLUS_RATIO = 16.0              #@param {type:"number"}
            USE_4BIT = False                    #@param {type:"boolean"}
            SFT_EPOCHS = 3.0                    #@param {type:"number"}
            RUN_DPO = True                      #@param {type:"boolean"}
            DPO_EPOCHS = 1.0                    #@param {type:"number"}

            TASKS_JSONL = "data/tasks.jsonl"    #@param {type:"string"}
            RESULTS_DIR = "results"             #@param {type:"string"}
            OUT_DIR = f"submission/adapters/{ADAPTER_NAME}"

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

            1. `src/collect/collect_trajectories.py` が学習データ `data/trajectories.jsonl` を作ります。
               `results/` があるときだけ `--import-traces results --only-resolved` を付けて、**解けた（resolved=True）
               軌跡だけを RFT として import** し、足りない分は tasks.jsonl の gold patch から合成した SFT 軌跡で補います。
            2. `src/train/train_lora.py` が cell 3 の設定（RSLoRA / LoRA+ / 4bit）で LoRA を学習し、`OUT_DIR` に保存します。

            試走させたいときは `collect_trajectories.py` に `--limit N`、`train_lora.py` に `--max_examples N` を足して
            件数を絞ってください（初回は tokenizer 読み込みだけで数分かかります）。
            """
        ),
        _code(
            """
            #@title 4. Stage 1 — RFT/SFT データ作成 & LoRA 学習
            from pathlib import Path

            import_arg = f"--import-traces {RESULTS_DIR} --only-resolved" if Path(RESULTS_DIR).exists() else ""
            !python3 src/collect/collect_trajectories.py \\
                --tasks {TASKS_JSONL} \\
                {import_arg} \\
                --output data/trajectories.jsonl

            !python3 src/train/train_lora.py \\
                --data data/trajectories.jsonl \\
                --out_dir {OUT_DIR} \\
                --r {R} --alpha {ALPHA} \\
                --epochs {SFT_EPOCHS} \\
                {EXTRA_FLAGS}
            """
        ),
        _md(
            """
            ## Cell 5 — Stage 2: `build_dpo_pairs.py` で選好ペア生成 → DPO 事後学習

            1. `src/collect/build_dpo_pairs.py` が eval 軌跡から `prompt` / `chosen` / `rejected` の選好ペアを作ります。
               chosen は `resolved=True` の軌跡（無ければ tasks.jsonl から gold 合成）、rejected は同じ task の
               解決できなかった軌跡（テスト失敗・ERROR/TIMEOUT・NO_PATCH・過剰ターン打ち切り）です。1 task あたり
               `--max-pairs-per-task`（既定 2）まで。
            2. ペアファイルが実際に生成できたときだけ `src/train/train_dpo.py` を走らせ、`--resume_from` で Stage 1 の
               adapter を初期値にして上から DPO を掛けます。cell 3 の `EXTRA_FLAGS`（`--use_rslora` /
               `--lora_plus_ratio` / `--use_4bit`）をそのまま渡すので、Stage 1 と同じ LoRA 構造になります。

            `RUN_DPO = False` または `results/` が無い場合は実行せず `[skip]`、ペアが 0 件なら `[skip]` です。
            DPO は Stage 1 より学習率が小さく（既定 5e-6）、`--beta` は 0.1 固定です。
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

            Colab のディスクは消えるので、`submission/`（`adapters/main_lora` または `adapters/analyzer_lora` 含む）を
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
