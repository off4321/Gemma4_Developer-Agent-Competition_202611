#!/usr/bin/env python3
"""Colab 用の **1 冊で完結する**測定ノートブック生成器（これが既定の notebook）。

この notebook だけで次を順にやります（依頼の 6 項目に対応）:

| 依頼 | cell | 使うもの |
|---|---|---|
| Kaggle の資材 DL | cell 2 | competition `gemma-4-developer-agent`（tasks / graphs / embeddings / snapshots / sample_submission）+ dataset `metric/gemma-4-developer-agent-wheelhouse`（harness pin） |
| model DL | cell 2 | `kaggle models instances versions download google/gemma-4/other/<model>/2 --untar`（`MATERIAL_SOURCE = "hf"` なら HF の同一 model） |
| 提出物が有ればそれを使う / 無ければ既定の資材・ハーネス | cell 2 + 4 | 解決順: **`submission.zip`** -> 自分の提出 dataset -> **公式 `sample_submission/`**（既定）。harness は常に wheelhouse の pin（cell 3） |
| サーバー構築 | cell 5 | `adk_submission.VllmServer(VllmConfig(...), adapter_manifest=discover_adapters(...))`（Getting Started と同じ引数） |
| スコア算出 | cell 7 | harness が書く `summary.json` の `resolution_rate`（= 公式 Resolution Rate） |
| 問題別に何が出来ていないか（エラー含め）ファイルに落とす | cell 7 | `results/failures.csv` と `results/failures.md`（task 単位 status / 理由 / test exit / tool calls / patch size / protected path 接触 / pytest 末尾 / 未測測 task） |

資材の取り方は 2 モードです（notebook は 1 冊のまま cell 2 の定数で切り替わります）:
`MATERIAL_SOURCE = "kaggle"`（既定、公式資材）/ `"hf"`（Kaggle を触らない。HF mirror + PyPI）

生成と実行:

```bash
python3 src/pipeline/make_colab_notebook.py                 # -> kernels/colab/gemma4_colab.ipynb（Kaggle 既定）
python3 src/pipeline/make_colab_notebook.py --source hf     # -> kernels/colab/gemma4_colab_hf.ipynb（認証不要）

colab new -s g4 --gpu L4
colab upload kernels/colab/gemma4_colab.ipynb /content/gemma4_colab.ipynb -s g4
# Colab で Secrets に KAGGLE_API_TOKEN -> cell 1 -> 2 -> ... -> 10 を順に run
```
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import re
from pathlib import Path

CELLS: list[tuple[str, str]] = []


def injectable(path: Path) -> str:
    """src/pipeline の 1 実装を、notebook cell にそのまま流し込める形にする。

    shebang / module docstring / `from __future__ import annotations` / `if __name__ == "__main__":` block だけ落とす。
    comment と format は保持するので、cell 側の実装と pipeline 側の実装が常に同じになる
    （`failures_report.py` を直すと notebook にも次生成で反映される）。
    """
    src = Path(path).read_text(encoding="utf-8")
    lines = src.splitlines()
    drop: set[int] = set()
    tree = ast.parse(src)
    head = tree.body[0] if tree.body else None
    if isinstance(head, ast.Expr) and isinstance(getattr(head, "value", None), ast.Constant) and isinstance(head.value.value, str):
        drop.update(range(head.lineno, (head.end_lineno or head.lineno) + 1))
    for node in tree.body:
        if isinstance(node, ast.If):
            test = node.test
            if getattr(getattr(test, "left", None), "id", "") == "__name__":
                drop.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    keep = []
    for idx, line in enumerate(lines, start=1):
        if idx in drop or line.startswith("#!") or line == "from __future__ import annotations":
            continue
        keep.append(line)
    return "\n".join(keep).strip() + "\n"


FAILURES_SRC = injectable(Path(__file__).resolve().with_name("failures_report.py"))


def md(text: str) -> None:
    CELLS.append(("markdown", text.strip()))


def code(text: str) -> None:
    CELLS.append(("code", text.strip()))


# notebook 側にも入れて、手元でも test できる tag filter（PEP 427）
HF_SRC = '\ndef pick(namelist, suffix):\n    """zip 内で `<...>/tasks.jsonl` のように suffix に末尾一致する entry を 1 個選ぶ（mirror の中の階層が未知なので末尾で寄せる）。"""\n    hits = [n for n in namelist if n.endswith(suffix) and not n.endswith("/")]\n    if not hits:\n        return None\n    depth = suffix.count("/")\n    hits.sort(key=lambda n: (abs(n.count("/") - depth), len(n)))\n    return hits[0]\n\n\ndef extract_one(zf, namelist, suffix, out_path):\n    entry = pick(namelist, suffix)\n    if entry is None:\n        return False\n    out_path.parent.mkdir(parents=True, exist_ok=True)\n    with zf.open(entry) as src, out_path.open("wb") as dst:\n        shutil.copyfileobj(src, dst)\n    return True\n\n\ndef group_from(zf, namelist, marker, dest):\n    """zip 内の `<任何前缀>/marker/...` を dest/相対パス に展開する（graphs / embeddings / sample_submission / wheels）。"""\n    n = 0\n    token = marker + "/"\n    for entry in namelist:\n        if entry.endswith("/"):\n            continue\n        i = entry.find(token)\n        if i < 0:\n            continue\n        rel = entry[i + len(token):]\n        if not rel:\n            continue\n        target = dest / rel\n        target.parent.mkdir(parents=True, exist_ok=True)\n        with zf.open(entry) as src, target.open("wb") as dst:\n            shutil.copyfileobj(src, dst)\n        n += 1\n    return n\n\n'

WHEEL_OK_SRC = '''
def wheel_ok(name: str, py_tag: str = "cp313") -> bool:
    """PEP 427 tag を読んで、この Colab image (cp313 / linux x86_64) で install できるか。

    cp312-cp312 の wheel は cp313 では使えない（実測で apache_tvm_ffi / cbor2 / msgspec が落ちた）。
    cp38-abi3 / cp39-abi3 / cp310-abi3 は abi3 なので cp313 で使える（openai_harmony / tokenizers / safetensors）。
    """
    def scale(tag: str) -> int:
        digits = tag[2:] if tag.startswith("cp") else tag
        num = int(digits)
        return num if num >= 100 else 300 + (num % 10)      # cp38 -> 308, cp312 -> 312, cp313 -> 313

    stem = name[:-4] if name.endswith(".whl") else name
    parts = stem.split("-")
    if len(parts) < 5:                       # <dist>-<ver>-<py>-<abi>-<platform>（build tag は稀なので見ない）
        return True
    python_tag, abi_tag, platform_tag = parts[-3], parts[-2], parts[-1]
    py_need = scale(py_tag)
    for tag in python_tag.split("."):
        if tag.startswith("py"):                      # py3 / py2.py3
            continue
        if tag.startswith("cp"):
            try:
                need = scale(tag)
            except ValueError:
                return False
            if abi_tag == "abi3":
                if need > py_need:
                    return False
                continue
            if need != py_need:
                return False
            continue
        if tag.startswith("pp") or tag.startswith("ip") or tag.startswith("graalpy"):
            return False
    if platform_tag in ("any", "none"):
        return True
    if "win" in platform_tag or "macosx" in platform_tag or platform_tag.startswith("musllinux"):
        return False
    if "linux" in platform_tag and "x86_64" not in platform_tag:
        return False
    return True
'''

exec(compile(WHEEL_OK_SRC, "<wheel_ok>", "exec"), globals())

md("""## 使い方（この notebook を回す順番・いじる定数・詰まったときの切り分け）

### 0) 何を成立させる物か

1 冊で **資材導入 → 公式 harness install → 提出 tree staging → アダプタ込み vLLM → `swegemma eval` → スコア → zip 検査** までやります。
Colab に上げる物は **この notebook 1 個 + Secrets の `KAGGLE_API_TOKEN`**（`src/pipeline/` は読みません）。
cell 7 に出る `resolution_rate` は公式スコアと同じ意味ですが、**公式スコアそのものではありません**（task 集合・GPU・sandbox が違う）。

### 1) 準備（手元で 1 回）

```bash
make colab-notebook                     # -> kernels/colab/gemma4_colab.ipynb（Kaggle 資材の既定モード）
make colab-notebook-hf                  # -> kernels/colab/gemma4_colab_hf.ipynb（Kaggle を触らない HF モード）

colab new -s g4 --gpu L4
colab upload kernels/colab/gemma4_colab.ipynb /content/gemma4_colab.ipynb -s g4
colab upload submission.zip /content/submission.zip -s g4         # 任意: 最終提出物をそのまま測る
```

Colab 側: ランタイムを **L4 GPU** にする（T4 だと cell 5 で vLLM 0.19.1 が shared memory で落ちます）。
左パネルの **Secrets** に `KAGGLE_API_TOKEN`（その notebook の **Notebook access ON**）。ブラウザで開いて 1 個ずつ run でも、
`colab upload` + `colab exec` でも動きます（Files panel からの upload でも可）。

### 2) 実行順と注意点

`cell 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9` の順。**cell 5 で起動した server を cell 6 が使うので、5 と 6 の間で kernel を restart しない**こと。
cell 2 は download だけ（初回 model 11.5 GB + graphs/embeddings、2 回目からは Drive cache が効きます）。
cell 9 で `server.stop()` を忘れないこと（忘れると vLLM が GPU を掴んだまま）。

### 3) いじる定数（該当 cell の先頭にあります）

| cell | 定数 | 既定 | 意味 |
|---|---|---|---|
| 2 | `MATERIAL_SOURCE` | `"kaggle"` | 資材と model の取り先。`"hf"` にすると Kaggle を触らない（mirror は third party で byte 検証無し） |
| 2 | `AGENT_TREE_SOURCE` | `"auto"` | 提出 tree の解決順: **提出 zip -> 自分の提出 dataset -> 公式 `sample_submission/`**。`"official"` で常に公式に固定 |
| 2 | `SUBMISSION_ZIP_CANDIDATES` | `/content/submission.zip`, Drive | **zip が有れば zip を優先**（公式が受け取る形そのものを測る） |
| 2 | `USE_DRIVE` | True | `/content/drive/MyDrive/gemma4` に資材を cache（次回 download を省く） |
| 2 | `WITH_TASK_WHEELS` | False | 公式 docker sandbox が使う task 依存 wheel 124 個も落とすか |
| 2 | `WITH_MODEL` | True | model を落とすか（既に有れば飛ばす） |
| 4 | `BUDGET` | 20 min / 80 tools / 150 turns | `eval_config.yaml` に書き換える予算（scorer 既定は 60 min / 100 tools / 500 turns） |
| 4 | `MODEL_NAME_OVERRIDE` | None | model 乗り換え（YAML の `model:` 全部書換 + 旧 model 用 `adapters/` を捨てる） |
| 5 | `GPU_MEM` / `TP_SIZE` | 0.90 / 1 | OOM なら 0.85 に落とす。`max_model_len=32768` は公式と同じなので触らない |
| 7 | `PREV_RESULTS` | None | 公式 harness 結果と自分の pipeline 結果を突き合わせる先の dir |

### 4) 時間の目安（実測）

cell 2 download 10-25 min（model 11.5 GB / graphs 1.3 GB / snapshot 12 個 2.4 GB）／cell 3 install 5-10 min／
cell 5 server 起動 1-3 min／**cell 6 は 1 task 79-222 s**（前に Colab L4 で測った値）。
Colab は session を止めないと同じ VM を使い続け、実測 1.94 compute units/h 前後（`colab usage` で見られます）。

### 5) 出てくる物と取り出し方

```
/content/repo/work/results/summary.json          resolution_rate（= スコア）
/content/repo/work/results/task_results.jsonl    task 単位 resolved / patch size / test exit / tool calls
/content/repo/work/results/failures.csv          task 別の不出来（status / 理由 / exit / patch / tools / protected path / pytest 末尾）
/content/repo/work/results/failures.md           上の表 + 集計 + 代表エラー 5 件（人が読む用）
/content/repo/work/results/patches/*.patch       agent が出した patch
/content/repo/work/results/test_outputs/*.log    Phase 2 の pytest 出力
/content/repo/work/results/score.json            上の集計 + 失敗分類
/content/repo/work/submission.zip                測った agent tree から作った提出 zip
/content/repo/work/agent_dir/                    staging された提出 tree（実際に測られた物）
```

```bash
colab download /content/repo/work/submission.zip /tmp/submission.zip -s g4one
colab download /content/repo/work/results/summary.json /tmp/summary.json -s g4one
colab download /content/repo/work/results/task_results.jsonl /tmp/task_results.jsonl -s g4
colab download /content/repo/work/results/failures.md /tmp/failures.md -s g4
colab download /content/repo/work/results/failures.csv /tmp/failures.csv -s g4
```

スコアが欲しいとき（提出は手元の Kaggle CLI で）:

```bash
kaggle competitions submit -c gemma-4-developer-agent -f submission.zip -m "e4b agent (prompt+config)"
kaggle competitions submissions -c gemma-4-developer-agent --csv > results/official_submissions.csv
python3 src/pipeline/scoreboard.py --official-csv results/official_submissions.csv
python3 src/pipeline/scoreboard.py --table
```

### 6) 詰まったとき（実測で踏んだ場所）

| 症状 | 対処 |
|---|---|
| cell 1 に `Option A: Environment variable` 或いは `Missing key` | Secrets の名前/値/notebook の Notebook access ON を確認。legacy key なら `KAGGLE_USERNAME` + `KAGGLE_KEY` でも通ります |
| cell 1 の `kaggle --version` が fail | Colab image の preinstall kaggle は import で死にます。cell 1 が `pip install -U kaggle` しますが、失敗したら手で `!python3 -m pip install -q -U kaggle` |
| cell 3 の `vllm import` が `model_hosting_container_standards` で fail | cell 3 が `--no-deps` で入れ直します。治らなければ `!python3 -m pip install -q --no-deps model-hosting-container-standards` |
| cell 5 で `out of resource: shared memory, Required: 98304, Hardware limit: 65536` | T4 (sm75) で vLLM 0.19.1 の Gemma 4 attention が無理。**この notebook は L4 (sm89) で使う物**です。T4 で測るなら Kaggle 側の variant `kaggle_nb`（vllm 0.28.0 測定専用）を使う |
| cell 5 で CUDA OOM | `GPU_MEM = 0.85` に落とす。`max_model_len=32768` は公式条件なので、まず GPU_MEM、task 数、adapter 数を疑う |
| cell 6 で `ensurepip unavailable in sandbox ... creating venv without pip` | Phase 2 の venv に pip が無い警告。test が pip install を要求するとそこで止まります。task を変えるか `WITH_TASK_WHEELS = True` で offline wheels を揃える |
| Kaggle の credential が切れた / quota が無い | `MATERIAL_SOURCE = "hf"` にして HF mirror + PyPI で測る（cell 1 の token 処理はスキップして構いません）。ただし mirror 由来の測定は公式 byte 一致未検証なので scoreboard の `harness` に `hf-mirror` と書いておく |
| cell 2 が重すぎる | `TASK_LIMIT` を 1-3 に、`WITH_TASK_WHEELS=False` のまま、Drive cache を効かせる（`USE_DRIVE=True`）。snapshot は 129 個で 21.3 GB なので全部落とさない |

### 7) この notebook の使い方の筋

**既定は公式 harness（wheelhouse pin）で測る**。自作 pipeline（`src/pipeline`）で測った物は `src/pipeline/parity_check.py` で
公式 harness との乖離ゼロを確認してから信用し、台帳 `src/pipeline/scoreboard.py` に載せて公式スコアと較正します。
絶対値（公式 Resolution Rate）が欲しいときは `submission.zip` を提出する。詳細は `doc/verification.md` と `doc/colab.md` §11-§12。

""")

md("""# Gemma 4 Developer Agent — 1 冊で完結する Colab 測定（資材 DL / model DL / 提出物 / server / スコア / task 別不出来）

**Colab に上げるものはこの notebook 1 個だけ**（既定モードでは + Secrets の `KAGGLE_API_TOKEN`）。
`src/pipeline/` の upload は不要で、公式 library そのまま使います: `swegemma`（eval harness）と `adk_submission`
（`VllmServer` / `discover_adapters` / `validate_directory`）。

| cell | 何を成立させるか |
|---|---|
| 1 | Colab Secrets の token で Kaggle 認証 + CLI 修理 |
| 2 | **資材導入**: tasks / HARNESS / graphs / embeddings / 測る task の snapshots / wheelhouse / model / 提出 tree |
| 3 | wheelhouse の pin で環境を公式の組み合わせに戻す + import 検査 |
| 4 | 提出 tree を staging（budget 書き換え・単一 model 検査） |
| 5 | **アダプタ込み server**: `VllmServer(cfg, adapter_manifest=discover_adapters(...))` + 1 文生成 + model alias 一覧 |
| 6 | **ハーネス実行**: `python3 -m swegemma eval`（Phase 1 agent loop + Phase 2 pytest verification） |
| 7 | **スコア算出**: `summary.json` の `resolution_rate` + per task 表 + 失敗分類 |
| 8 | 提出 zip 生成 + `validate_directory` |
| 9 | server を止めて物を下ろす |

### 3 つの質問への答え（この notebook の中で成立する事）

| 質問 | 成立する? | どの cell で |
|---|---|---|
| Colab 上で **Getting Started と同じデータセット**を使って検証できるか | **できます**。同じ competition `gemma-4-developer-agent` から `tasks.jsonl` / `snapshots/` / `graphs/` / `embeddings/` / `wheels/` / `sample_submission/` を公式 CLI で落とす（GS は mount、Colab は download という違いだけ）。`AGENT_TREE_SOURCE = "official"` にすると GS が使っている `sample_submission/` そのもので測ります | cell 2 |
| Colab 上で **ハーネスと最終提出物**を使った検証ができるか | **できます**。ハーネス一式は wheelhouse dataset から install（cell 3）、`submission.zip` を展開して測る（cell 2 の `SUBMISSION_ZIP_CANDIDATES`）、測った agent tree から zip を作り直して `validate_directory` + 開き直し検査（cell 8） | cell 2, 3, 8 |
| ノートブックに **自作ハーネス**や自作の提出物を使った検証を入れられるか | **入れられます**。公式 harness の結果ディレクトリと自作 pipeline の `predictions_*.json` を cell 7 の `PREV_RESULTS` で突き合わせます（外では `src/pipeline/parity_check.py`）。提出物側は自分の prompts / configs / adapters を cell 4 で staging して測ります | cell 4, 7 |

正直に言っておく制約:
- **これは公式採点ではない**。公式は `submission.zip` を受け取り、4×L4 (tp=4, bfloat16, gpu_mem 0.90) で**非公開 task 256 個**を同じ harness で走らせ、Resolution Rate を出します。ここでは公開 129 task のうち測る分だけ
- Colab は GPU 1 枚（L4, sm89, tp=1, gpu_mem 0.85）。`max_model_len=32768` は公式と同じに保つ
- **アダプタ**: `submission/adapters/` に `adapter_config.json` + `adapter_model.safetensors` が無いと adapter は 0 個で測られます（今の提出 tree は LoRA 無し）。4-bit QAT の base に LoRA を載せるには vLLM の LoRA 対応が必要で、`enable_lora=True / max_loras / max_lora_rank` は harness 側が面倒を見ます（rank は inferred）
- Colab の compute units を使います（実測 1.94/h 前後）。初回 download 22 GB + 11.5 GB、2 回目からは Drive cache
""")

# ---------------------------------------------------------------- 1. Secrets + 認証
md("""## cell 1 — Kaggle 認証（`KAGGLE_API_KEY` 変数 / Colab Secrets の `KAGGLE_API_TOKEN`）

変数 `KAGGLE_API_KEY` が設定されている場合は最優先で使用します。
未設定の場合は Colab Secrets（`KAGGLE_API_KEY` または `KAGGLE_API_TOKEN`、legacy の `KAGGLE_USERNAME`+`KAGGLE_KEY`）から読み込みます。
CLI 2.2.5 の認証順序: `KAGGLE_API_TOKEN` env → `~/.kaggle/access_token` → `kaggle.json`（`KAGGLE_USERNAME`+`KAGGLE_KEY`）→ OAuth。
Colab の `!export` は次の cell に残らないので `os.environ` にいれ、detached にも伝わるよう file にも書きます。
""")

code('''import json
import os
import subprocess
from pathlib import Path

SECRET_NAME = "KAGGLE_API_TOKEN"
COMPETITION = "gemma-4-developer-agent"

# 1. KAGGLE_API_KEY 変数（Python グローバル/ローカル変数、または環境変数）を優先チェック
api_key_var = globals().get("KAGGLE_API_KEY") or locals().get("KAGGLE_API_KEY") or os.environ.get("KAGGLE_API_KEY", "")

token = os.environ.get(SECRET_NAME, "")
user_name = os.environ.get("KAGGLE_USERNAME", "")
api_key_legacy = os.environ.get("KAGGLE_KEY", "")

if api_key_var:
    print(f"[cell 1] 変数/環境変数 KAGGLE_API_KEY が設定されているため、これを使用します（{len(str(api_key_var))} chars）")
    token = str(api_key_var).strip()
    api_key_legacy = token

if not token and not (user_name and api_key_legacy):
    try:
        from google.colab import userdata

        if hasattr(userdata, "keys"):
            print("[cell 1] Secrets の名前:", sorted(userdata.keys()))
        token = userdata.get("KAGGLE_API_KEY") or userdata.get(SECRET_NAME)
    except Exception as exc:                                # SecretNotFoundError / NotebookAccessError
        print(f"[cell 1] {type(exc).__name__}: Secrets に KAGGLE_API_KEY / {SECRET_NAME} が無い"
              "（Secrets -> Add new secret -> Notebook access ON）")
        try:
            from google.colab import userdata as ud

            user_name = ud.get("KAGGLE_USERNAME")
            api_key_legacy = ud.get("KAGGLE_API_KEY") or ud.get("KAGGLE_KEY")
            if api_key_legacy:
                print("[cell 1] Secrets の KAGGLE_API_KEY / KAGGLE_KEY で代用する")
        except Exception:
            pass

kdir = Path.home() / ".kaggle"
kdir.mkdir(parents=True, exist_ok=True)
if token:
    os.environ[SECRET_NAME] = token
    os.environ["KAGGLE_API_KEY"] = token
    os.environ["KAGGLE_KEY"] = token
    (kdir / "access_token").write_text(token + "\\n")
    os.chmod(kdir / "access_token", 0o600)
    kjson = {"key": token}
    if user_name:
        kjson["username"] = user_name
        os.environ["KAGGLE_USERNAME"] = user_name
    (kdir / "kaggle.json").write_text(json.dumps(kjson))
    os.chmod(kdir / "kaggle.json", 0o600)
    print(f"[cell 1] 認証情報を kernel env と ~/.kaggle/access_token, ~/.kaggle/kaggle.json に設定（{len(token)} chars）")
elif user_name and api_key_legacy:
    os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"] = user_name, api_key_legacy
    os.environ["KAGGLE_API_KEY"] = api_key_legacy
    (kdir / "kaggle.json").write_text(json.dumps({"username": user_name, "key": api_key_legacy}))
    os.chmod(kdir / "kaggle.json", 0o600)
    print(f"[cell 1] legacy API key を ~/.kaggle/kaggle.json に設定（user: {user_name}）")
else:
    raise SystemExit("[cell 1] Kaggle 認証が無い。変数 KAGGLE_API_KEY を設定するか、Secrets に登録すること")

proc = subprocess.run("kaggle --version 2>&1 | tail -1", shell=True, capture_output=True, text=True)
if "CLI" not in (proc.stdout or ""):                        # Colab image の preinstall kaggle は import で死ぬ（実測）
    print("[cell 1] kaggle CLI 修理: pip install -q -U kaggle")
    subprocess.run("python3 -m pip install -q -U kaggle", shell=True, capture_output=True, text=True)
check = subprocess.run(f"kaggle competitions files -c {COMPETITION} --page-size 3 -q 2>&1 | tail -4",
                      shell=True, capture_output=True, text=True)
out = ((check.stdout or "") + (check.stderr or "")).strip()
print("[cell 1] 認証検査（token の値は出さない）:\\n" + out[-400:])
if "Option A" in out or "Missing key" in out:
    raise SystemExit("[cell 1] Kaggle 認証が通らない。KAGGLE_API_KEY の値または Secrets の設定を確認")

import sys

print("[cell 1]", sys.version.split()[0], subprocess.run(
    "nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader",
    shell=True, capture_output=True, text=True).stdout.strip())
''')

# ---------------------------------------------------------------- 2. 資材導入
md("""## cell 2 — 資材導入と model DL（`MATERIAL_SOURCE` で 2 経路 / 必要な分だけ）

`MATERIAL_SOURCE = "kaggle"`（既定）は公式資材、`"hf"` は Kaggle を触らず HF mirror + PyPI で揃えます。
落ちる物は両モードで同じ形になります（`DATA/wheelhouse`・`DATA/graphs`・`DATA/embeddings`・測る task の `DATA/snapshots`・同一 model）。

| 物 | Kaggle モード | HF モード | size |
|---|---|---|---|
| `tasks.jsonl`（129 task, `test_patch` と gold `patch` を含む） + `HARNESS_README.md` | competition から 1 個ずつ | zip 内で末尾一致した物だけ抽出 | 2 MB |
| `graphs/` 127 + `embeddings/` 127 | 1 個ずつ（competition zip 全部 21.9 GB は落とさない） | zip 内の marker だけ抽出 | 1.3 GB |
| `snapshots/<instance_id>.tgz` | **測る task だけ**（`TASK_LIMIT`） | 同（zip から selected だけ） | 12 個で 2.4 GB |
| `wheelhouse/*.whl` | dataset `metric/gemma-4-developer-agent-wheelhouse` | zip 内の `.whl` 41 個（pin 名は公式と同じと実測済み） | 867 MB |
| model | `kaggle models instances versions download google/gemma-4/other/gemma-4-e4b-it-qat-w4a16-ct/2 --untar` | `snapshot_download(google/gemma-4-E4B-it-qat-w4a16-ct)` | 11.5 GB |
| 提出 tree | 提出 zip -> 自 dataset -> 公式 `sample_submission/` | 提出 zip -> zip 内の `sample_submission/` | ~10 KB |

提出 tree の解決順は **提出 zip -> 自分の提出 dataset -> 公式 `sample_submission/`（既定）**です。zip が有れば zip を優先します
（公式が受け取る形そのものを測る）。公式 `sample_submission/` には 31B 用の dummy LoRA が入っているので、model を乗り換えるときは
cell 4 の `MODEL_NAME_OVERRIDE` で捨てください（Getting Started と同じ操作）。

`G4 = /content/drive/MyDrive/gemma4` に cache すると、次回以降 download は model 以外走りません
""")

code(f'''import json
import os
import shutil
import subprocess
from pathlib import Path

MATERIAL_SOURCE = "kaggle"             # "hf" にすると Kaggle を触らない（HF mirror + PyPI。mirror は third party で byte 検証無し）
TASK_LIMIT = 6                         # 実際に測る task 数（0 で全 129）
WITH_MODEL = True
WITH_TASK_WHEELS = False               # 公式 docker sandbox が使う task 依存 wheel 124 個も落とすか（Colab は internet があるので既定 off）
USE_DRIVE = True
COMPETITION = "gemma-4-developer-agent"
WHEELHOUSE_SLUG = "metric/gemma-4-developer-agent-wheelhouse"
MODEL_SLUG = "google/gemma-4/other/gemma-4-e4b-it-qat-w4a16-ct/2"
MODEL_NAME = "gemma-4-e4b-it-qat-w4a16-ct"
HF_DATA_ID = "HanG/gemma-4-developer-agent"
HF_DATA_ZIP = "gemma-4-developer-agent.zip"                       # 21,923,985,083 byte
HF_WHEELHOUSE_ZIP = "gemma-4-developer-agent-wheelhouse-v30.zip"  # 880,849,882 byte（41 wheels、pin 名は公式と同じを実測）
HF_MODEL_ID = "google/gemma-4-E4B-it-qat-w4a16-ct"               # HF 側は大文字 E4B。同一 model.safetensors 11,513,861,828 byte
SUBMISSION_SLUG = os.environ.get("KAGGLE_USERNAME", "<username>") + "/gemma4-dev-agent-submission"

REPO = Path("/content/repo")
DATA = REPO / "data"
MODEL_DIR = REPO / "models" / "google" / "gemma-4" / "other" / MODEL_NAME / "2"
G4 = Path("/content/drive/MyDrive/gemma4")
CACHE_SUBS = ("wheelhouse", "graphs", "embeddings", "snapshots", "wheels")
for _d in (DATA, DATA / "snapshots", DATA / "wheelhouse", MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def run(argv, tail=300):
    print("+ " + " ".join(str(a) for a in argv), flush=True)
    proc = subprocess.run([str(a) for a in argv], capture_output=True, text=True)
    out = ((proc.stdout or "") + (proc.stderr or "")).strip()
    if out:
        print(out[-tail:], flush=True)
    if proc.returncode != 0:
        raise SystemExit(f"失敗 (exit {{proc.returncode}}): {{' '.join(str(a) for a in argv)}}")
    return out


def comp_files():
    rows, token = [], None
    for _ in range(12):
        cmd = ["kaggle", "competitions", "files", "-c", COMPETITION, "--page-size", "200", "-q"]
        if token:
            cmd += ["--page-token", token]
        out = run(cmd, tail=0)
        token = None
        for line in out.splitlines():
            if line.startswith("Next Page Token = "):
                token = line.split("= ", 1)[1].strip()
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                rows.append((parts[0], int(parts[1])))
        if not token:
            break
    return rows


def download_single(path, dest):
    dest.mkdir(parents=True, exist_ok=True)
    run(["kaggle", "competitions", "download", "-c", COMPETITION, "-f", path, "-p", str(dest), "-q"], tail=120)
    for z in sorted(dest.glob("*.zip")):
        run(["unzip", "-o", "-q", str(z), "-d", str(dest)], tail=120)
        z.unlink(missing_ok=True)


''' + HF_SRC + f'''

# Drive cache を先に戻す（session が VM を捨てても Drive は残る）
if USE_DRIVE:
    try:
        from google.colab import drive

        drive.mount("/content/drive")
    except Exception as exc:
        print("[cache] Drive mount 失敗（無しで続ける）:", type(exc).__name__)
    if G4.is_dir():
        for name in ("tasks.jsonl", "HARNESS_README.md"):
            if (G4 / "data" / name).is_file() and not (DATA / name).is_file():
                shutil.copy2(str(G4 / "data" / name), str(DATA / name))
        for sub in CACHE_SUBS:
            src_dir = G4 / "data" / sub
            if not src_dir.is_dir():
                continue
            (DATA / sub).mkdir(parents=True, exist_ok=True)
            for f in sorted(src_dir.iterdir()):
                dst = DATA / sub / f.name
                if f.is_file() and not dst.exists():
                    shutil.copy2(str(f), str(dst))
        print("[cache] Drive から戻した:", [s for s in CACHE_SUBS if (DATA / s).is_dir()], flush=True)


def read_tasks():
    if not (DATA / "tasks.jsonl").is_file():
        return []
    return [json.loads(line) for line in (DATA / "tasks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def select_ids(rows):
    picked = rows if TASK_LIMIT <= 0 else rows[:TASK_LIMIT]
    return [r.get("instance_id") for r in picked if r.get("instance_id")]


rows = read_tasks()
ids = select_ids(rows)
print(f"[cell 2] MATERIAL_SOURCE={{MATERIAL_SOURCE}} 測る task {{len(ids)}} 個: {{ids[:4]}}{{'...' if len(ids) > 4 else ''}}", flush=True)

if MATERIAL_SOURCE == "hf":
    import zipfile as _zf

    from huggingface_hub import hf_hub_download, snapshot_download

    tmp = Path("/content/hf")
    tmp.mkdir(parents=True, exist_ok=True)
    data_zip = None
    if not list((DATA / "wheelhouse").glob("*.whl")):
        wh_zip = hf_hub_download(repo_id=HF_DATA_ID, filename=HF_WHEELHOUSE_ZIP, repo_type="dataset", local_dir=str(tmp))
        print(f"[cell 2] HF wheelhouse zip: {{Path(wh_zip).stat().st_size:,}} byte", flush=True)
        with _zf.ZipFile(wh_zip) as zf:
            n = 0
            for entry in zf.namelist():
                if entry.endswith(".whl"):
                    target = DATA / "wheelhouse" / Path(entry).name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(entry) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    n += 1
            print(f"[cell 2] wheelhouse: {{n}} wheels", flush=True)
    if not ids:
        data_zip = hf_hub_download(repo_id=HF_DATA_ID, filename=HF_DATA_ZIP, repo_type="dataset", local_dir=str(tmp))
        with _zf.ZipFile(data_zip) as zf:
            extract_one(zf, zf.namelist(), "tasks.jsonl", DATA / "tasks.jsonl")
        rows, ids = read_tasks(), select_ids(read_tasks())
        print(f"[cell 2] tasks: {{len(rows)}} 件 / 測る task {{len(ids)}} 個", flush=True)
    if not (DATA / "graphs").is_dir() or len(list((DATA / "graphs").glob("*"))) < 100:
        data_zip = data_zip or hf_hub_download(repo_id=HF_DATA_ID, filename=HF_DATA_ZIP, repo_type="dataset", local_dir=str(tmp))
        with _zf.ZipFile(data_zip) as zf:
            names = zf.namelist()
            extract_one(zf, names, "HARNESS_README.md", DATA / "HARNESS_README.md")
            for marker in ("graphs", "embeddings"):
                print(f"[cell 2] {{marker}}: {{group_from(zf, names, marker, DATA / marker)}} file", flush=True)
            if WITH_TASK_WHEELS:
                print(f"[cell 2] wheels: {{group_from(zf, names, 'wheels', DATA / 'wheels')}} file", flush=True)
        print("[cell 2] zip 全体は展開しません（21.9 GB）。必要な marker だけ出しました", flush=True)
else:
    for name in ("tasks.jsonl", "HARNESS_README.md"):
        if not (DATA / name).is_file():
            download_single(name, DATA)
    rows, ids = read_tasks(), select_ids(read_tasks())
    have = {{s: (len(list((DATA / s).glob("*"))) if (DATA / s).is_dir() else 0) for s in ("graphs", "embeddings")}}
    if have["graphs"] < 100 or have["embeddings"] < 100 or (WITH_TASK_WHEELS and not (DATA / "wheels").is_dir()):
        files = comp_files()
        for sub in ["graphs", "embeddings"] + (["wheels"] if WITH_TASK_WHEELS else []):
            want = [q for q in files if q.split("/", 1)[0] == sub]
            done = len(list((DATA / sub).glob("*"))) if (DATA / sub).is_dir() else 0
            for i, path in enumerate(want[done:], start=done + 1):
                download_single(path, DATA / sub)
                if i % 25 == 0:
                    print(f"[{{sub}}] {{i}}/{{len(want)}}", flush=True)
            print(f"[{{sub}}] 合計 {{len(list((DATA / sub).glob('*')))}}", flush=True)
    if not list((DATA / "wheelhouse").glob("*.whl")):
        run(["kaggle", "datasets", "download", "-d", WHEELHOUSE_SLUG, "-p", str(DATA / "wheelhouse"), "--unzip", "-q"], tail=200)
        print(f"[cell 2] wheelhouse: {{len(list((DATA / 'wheelhouse').glob('*.whl')))}} wheels", flush=True)

# snapshots は測る task だけ（全部 129 個で 21.3 GB）
have_snap = len(list((DATA / "snapshots").glob("*.tgz"))) if (DATA / "snapshots").is_dir() else 0
if have_snap < len(ids):
    if MATERIAL_SOURCE == "hf":
        import zipfile as _zf

        data_zip = data_zip or hf_hub_download(repo_id=HF_DATA_ID, filename=HF_DATA_ZIP, repo_type="dataset", local_dir=str(Path("/content/hf")))
        with _zf.ZipFile(data_zip) as zf:
            names = zf.namelist()
            for k, iid in enumerate(ids, start=1):
                extract_one(zf, names, f"snapshots/{{iid}}.tgz", DATA / "snapshots" / f"{{iid}}.tgz")
                if k % 5 == 0:
                    print(f"[snapshots] {{k}}/{{len(ids)}}", flush=True)
    else:
        for k, iid in enumerate(ids, start=1):
            download_single(f"snapshots/{{iid}}.tgz", DATA / "snapshots")
            if k % 5 == 0:
                print(f"[snapshots] {{k}}/{{len(ids)}}", flush=True)
    print(f"[cell 2] snapshots: {{len(list((DATA / 'snapshots').glob('*.tgz')))}}/{{len(ids)}}（Phase 2 が extract_snapshot で展開する）", flush=True)

# 提出 tree は「提出物が有ればそれ」、無ければ公式既定（sample_submission）に寄せる
SUBMISSION_ZIP_CANDIDATES = ("/content/submission.zip", "/content/drive/MyDrive/gemma4/submission.zip")
AGENT_TREE_SOURCE = "auto"             # "auto" = zip -> 自 dataset -> 公式 sample_submission / "official" = 常に公式
sub_dir = REPO / "submission"
AGENT_TREE_KIND = "既に /content/repo/submission に有る（upload 分）"
zip_hit = next((Path(z) for z in SUBMISSION_ZIP_CANDIDATES if Path(z).is_file()), None)
if zip_hit is not None:
    import zipfile as _zf

    with _zf.ZipFile(zip_hit) as zf:
        zf.extractall(str(sub_dir))
    AGENT_TREE_KIND = f"提出 zip（{{zip_hit.name}}）"
elif (sub_dir / "agent.yaml").is_file():
    pass
elif AGENT_TREE_SOURCE == "auto" and MATERIAL_SOURCE == "kaggle":
    tmp = Path("/content/submission")
    tmp.mkdir(parents=True, exist_ok=True)
    subprocess.run(["kaggle", "datasets", "download", "-d", SUBMISSION_SLUG, "-p", str(tmp), "--unzip", "-q"],
                   capture_output=True, text=True)
    found = tmp if (tmp / "agent.yaml").is_file() else next((h.parent for h in tmp.rglob("agent.yaml")), None)
    if found is not None:
        shutil.copytree(str(found), str(sub_dir), dirs_exist_ok=True)
        AGENT_TREE_KIND = f"自分の提出 dataset（{{SUBMISSION_SLUG}}）"
if not (sub_dir / "agent.yaml").is_file():                  # 公式の既定 tree（Getting Started と同じ）
    if MATERIAL_SOURCE == "kaggle":
        files = comp_files()
        want = [q for q in files if q.split("/", 1)[0] == "sample_submission"]
        for path in want:
            rel = Path(path).relative_to(Path(path).parts[0])
            download_single(path, sub_dir / rel.parent if str(rel.parent) != "." else sub_dir)
        AGENT_TREE_KIND = "公式 sample_submission（既定。31B 用 dummy LoRA を含む）"
    else:
        import zipfile as _zf

        from huggingface_hub import hf_hub_download

        data_zip = data_zip or hf_hub_download(repo_id=HF_DATA_ID, filename=HF_DATA_ZIP, repo_type="dataset", local_dir=str(Path("/content/hf")))
        with _zf.ZipFile(data_zip) as zf:
            n = group_from(zf, zf.namelist(), "sample_submission", sub_dir)
        AGENT_TREE_KIND = f"HF mirror の sample_submission（{{n}} file）"
print("[cell 2] 使う提出 tree:", AGENT_TREE_KIND, flush=True)

if WITH_MODEL and not (MODEL_DIR / "model.safetensors").is_file():
    if MATERIAL_SOURCE == "kaggle":
        raw = REPO / "models" / "_raw"
        raw.mkdir(parents=True, exist_ok=True)
        run(["kaggle", "models", "instances", "versions", "download", MODEL_SLUG, "-p", str(raw), "--untar", "-q"], tail=200)
        for tar in sorted(raw.rglob("*.tar.gz")):
            run(["tar", "xzf", str(tar), "-C", str(tar.parent)], tail=120)
        found = list(raw.rglob("model.safetensors"))
        if not found:
            raise SystemExit("[cell 2] model.safetensors が見つからない")
        for f in sorted(found[0].parent.iterdir()):
            if f.is_file() and f.suffix != ".gz":
                shutil.move(str(f), str(MODEL_DIR / f.name))
        shutil.rmtree(raw, ignore_errors=True)
    else:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=HF_MODEL_ID, local_dir=str(MODEL_DIR),
                          allow_patterns=["*.json", "*.safetensors", "*.model*", "*.jinja", "*.txt"])
    print(f"[cell 2] model: {{(MODEL_DIR / 'model.safetensors').stat().st_size:,}} byte -> {{MODEL_DIR}}", flush=True)

if USE_DRIVE and G4.is_dir():                               # cache を溜める（次回 download を省く）
    for name in ("tasks.jsonl", "HARNESS_README.md"):
        if (DATA / name).is_file() and not (G4 / "data" / name).is_file():
            (G4 / "data").mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(DATA / name), str(G4 / "data" / name))
    for sub in CACHE_SUBS:
        src_dir = DATA / sub
        if not src_dir.is_dir():
            continue
        dst_dir = G4 / "data" / sub
        dst_dir.mkdir(parents=True, exist_ok=True)
        for f in sorted(src_dir.iterdir()):
            if f.is_file() and not (dst_dir / f.name).is_file():
                shutil.copy2(str(f), str(dst_dir / f.name))
        print(f"[cache] {{sub}} を Drive に置いた", flush=True)

missing = []
for label, path in (("tasks", DATA / "tasks.jsonl"), ("graphs", DATA / "graphs"), ("embeddings", DATA / "embeddings"),
                    ("snapshots", DATA / "snapshots"), ("wheelhouse", DATA / "wheelhouse"),
                    ("submission", sub_dir / "agent.yaml"), ("model", MODEL_DIR / "model.safetensors")):
    ok = path.is_dir() or path.is_file()
    size = path.stat().st_size if path.is_file() else 0
    print(f"  {{'OK ' if ok else 'MISS'}} {{label:11s}} {{path}} {{size:,}}")
    if not ok:
        missing.append(label)
if missing:
    print("[cell 2] 欠け:", missing, "-> 後の cell で落ちます。TASK_LIMIT / WITH_MODEL / MATERIAL_SOURCE と認証を見ること")
TASK_IDS = ids
''')

# ---------------------------------------------------------------- 3. env（wheelhouse pin）
md("""## cell 3 — wheelhouse の pin で環境を公式の組み合わせに戻す

Colab image は torch 2.14.1+cu130 / transformers 5.22.1 / tokenizers 0.23.3。wheelhouse は
vllm 0.19.1 / transformers 5.13.1 / tokenizers 0.22.2 / torch 2.10.0 の pin。
手順（実測で通った順）:

1. wheelhouse の wheel を **PEP 427 tag で選別**して `--no-deps` で入れる（cp312 の wheel は cp313 で使えないので捨てる）
2. vllm の metadata から torch pin を読んで **torch 三兄弟だけ** deps 込みで入れる（vllm 一式を入れると pip が huggingface_hub を下げて transformers 5.13.1 が壊れる）
3. harness の wheel を `--no-deps --force-reinstall` で戻す
4. Colab image に無い deps を PyPI から（jmespath / docker / litellm / blake3 / ninja / mcp / flashinfer 0.6.6 pins / cudnn-frontend / cutlass-dsl）
5. `import vllm.entrypoints.openai.api_server` を見る。死ぬなら `model-hosting-container-standards` を `--no-deps` で入れ直す
""")

code('''import importlib
import subprocess
import sys
from pathlib import Path

DATA = Path("/content/repo/data")
WHEELHOUSE = DATA / "wheelhouse"
PROTECTED_PREFIX = ("nvidia_", "triton", "torch-", "torchvision-", "torchaudio-")
HARNESS_WHEELS = ("swegemma", "adk_submission", "adk_eval_core", "google_adk", "transformers", "tokenizers",
                  "safetensors", "vllm", "model_hosting_container_standards", "compressed_tensors")


''' + WHEEL_OK_SRC + '''

def pip(args, tail=500):
    print("+ pip " + " ".join(args), flush=True)
    proc = subprocess.run([sys.executable, "-m", "pip", *args], capture_output=True, text=True)
    out = ((proc.stdout or "") + (proc.stderr or "")).strip()
    if out:
        print(out[-tail:], flush=True)
    return proc.returncode


py_tag = "cp" + "".join(str(x) for x in sys.version_info[:2])
wheels = sorted(p for p in WHEELHOUSE.glob("*.whl") if not p.name.startswith(PROTECTED_PREFIX))
skip = [p.name for p in WHEELHOUSE.glob("*.whl") if p.name.startswith(PROTECTED_PREFIX) or not wheel_ok(p.name, py_tag)]
print(f"[cell 3] python {py_tag}: install 候補 {len(wheels)} / skip {len(skip)}")
for name in skip[:12]:
    print("   skip:", name)
if wheels:
    pip(["install", "--no-cache-dir", "--no-deps", "-q", *[str(p) for p in wheels]], tail=900)

import importlib.metadata as md_meta

pins = []
for req in (md_meta.requires("vllm") or []):
    low = req.lower()
    if low.startswith(("torch=", "torchvision=", "torchaudio=", "torch==", "torchvision==", "torchaudio==")) and ";" not in req:
        pins.append(req.split(";")[0].strip())
if pins:
    print("[cell 3] vllm の pin 通り torch stack を deps 込みで:", pins, flush=True)
    pip(["install", "--no-cache-dir", "-q", *pins], tail=900)
importlib.invalidate_caches()

targets = []
for p in WHEELHOUSE.glob("*.whl"):
    if any(p.name.startswith(h + "-") for h in HARNESS_WHEELS):
        targets.append(str(p))
if targets:
    pip(["install", "--no-cache-dir", "--no-deps", "--force-reinstall", "-q", *targets], tail=600)

GAPS = ["jmespath", "docker", "litellm", "blake3", "ninja", "mcp", "flashinfer-python==0.6.6",
        "flashinfer-cubin==0.6.6", "nvidia-cudnn-frontend<1.19.0,>=1.13.0", "nvidia-cutlass-dsl>=4.4.0.dev1"]
import importlib.util as u

gaps = [g for g in GAPS if u.find_spec(g.split("[")[0].split("=")[0].split("<")[0].split(">")[0].replace("-", "_")) is None]
if gaps:
    print("[cell 3] PyPI 補い:", gaps, flush=True)
    pip(["install", "--no-cache-dir", "-q", *gaps], tail=700)

def try_api_import():
    importlib.invalidate_caches()
    try:
        importlib.import_module("vllm.entrypoints.openai.api_server")
        return "api_server import OK"
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


out = try_api_import()
print("[cell 3] vllm import:", out)
if "OK" not in out and "model_hosting_container_standards" in out:
    pip(["install", "--no-cache-dir", "--no-deps", "-q", "model-hosting-container-standards"], tail=300)
    out = try_api_import()
    print("[cell 3] 入れ直し後:", out)

print()
for name in ("vllm", "vllm.entrypoints.openai.api_server", "swegemma", "adk_submission", "adk_eval_core",
             "google.adk", "litellm", "openai", "docker", "jmespath", "blake3", "transformers", "tokenizers", "torch"):
    try:
        mod = __import__(name, fromlist=["_"])
        print(f"  OK   {name:38s} {getattr(mod, '__version__', '')}")
    except Exception as exc:
        print(f"  FAIL {name:38s} {type(exc).__name__}: {exc}")
import torch

print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "| gpu:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
''')

# ---------------------------------------------------------------- 4. 提出 tree staging
md("""## cell 4 — 提出 tree を staging（`submission/` は無変更、書き換えるのはコピー側）

Getting Started の Cell 2 にあたります。公式は `sample_submission/` を `/kaggle/working/sample_submission` にコピーして
`MODEL_NAME` を 31B 以外に乗り換えるとき **agent.yaml と sub_agents/*.yaml の `model:` 行を全部書き換え**、
**前の model 用の `adapters/` を捨てます**（sample_submission の `adapters/main_lora`, `adapters/tool_lora` は
31B 用の dummy LoRA なので、12B/e4B で使うと vLLM が拒否する）。この cell はそれを `MODEL_NAME_OVERRIDE` で同じ事をします。

その上で `eval_config.yaml` の budget（`max_time_minutes` / `max_tool_calls` / `max_turns`）を実測値に書き換えます。
公式の既定は 60 min / 100 tool calls（`eval_config.yaml` に key が無いとき harness 側 default）なので、
近くしたければ `BUDGET` を上げてください。単一 model 制約（`configs/sampling.yaml` に 1 model だけ）も検査します。
""")

code(r'''import re
import shutil
from pathlib import Path

import yaml

REPO = Path("/content/repo")
AGENT_DIR = REPO / "work" / "agent_dir"
BUDGET = {"max_time_minutes": 20.0, "max_tool_calls": 80, "max_turns": 150, "command_timeout_seconds": 300,
          "verification_timeout_seconds": 300}
MODEL_NAME_OVERRIDE = None      # 例 "gemma-4-12b-it-qat-w4a16-ct"。Getting Started と同じ操作（YAML の model 行を書き換え）

if AGENT_DIR.exists():
    shutil.rmtree(str(AGENT_DIR))
AGENT_DIR.mkdir(parents=True, exist_ok=True)
shutil.copytree(str(REPO / "submission"), str(AGENT_DIR), dirs_exist_ok=True)

# Getting Started の Cell 2 と同じ: model を乗り換えるときは agent YAML 全部の model 行を書き換え、
# 前の model 用の adapters は捨てる（sample_submission の adapters/ は 31B 用の dummy LoRA だった）
if MODEL_NAME_OVERRIDE:
    for yaml_path in [AGENT_DIR / "agent.yaml", *sorted((AGENT_DIR / "sub_agents").glob("*.yaml"))]:
        if yaml_path.is_file():
            text = yaml_path.read_text(encoding="utf-8")
            text = re.sub(r"^(\s*model:\s*).*$", rf"\g<1>{MODEL_NAME_OVERRIDE}", text, flags=re.MULTILINE)
            text = re.sub(r"^\s*adapter:\s*.*\n?", "", text, flags=re.MULTILINE)
            yaml_path.write_text(text, encoding="utf-8")
    shutil.rmtree(str(AGENT_DIR / "adapters"), ignore_errors=True)
    print("[cell 4] model を", MODEL_NAME_OVERRIDE, "に乗り換えた（adapters は捨てた）")

# eval_config.yaml の budget だけ書く（公式はここを Evaluator が読む。cell 6 の CLI 引数も同じ値にする）
eval_cfg_path = AGENT_DIR / "eval_config.yaml"
cfg = yaml.safe_load(eval_cfg_path.read_text(encoding="utf-8")) if eval_cfg_path.is_file() else {}
cfg = cfg or {}
cfg.setdefault("agent", {})
for key, value in BUDGET.items():
    if value is not None:
        cfg["agent"][key] = value
eval_cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print("[cell 4] staged agent dir:", AGENT_DIR, "| 出所:", globals().get("AGENT_TREE_KIND", "-"))
print(eval_cfg_path.read_text(encoding="utf-8"))

agent_cfg = yaml.safe_load((AGENT_DIR / "agent.yaml").read_text(encoding="utf-8")) or {}
sampling = yaml.safe_load((AGENT_DIR / "configs" / "sampling.yaml").read_text(encoding="utf-8")) if (AGENT_DIR / "configs" / "sampling.yaml").is_file() else {}
models = list(((sampling or {}).get("models") or {}).keys())
print("[cell 4] agent.yaml model:", agent_cfg.get("model"), "| sampling models:", models)
if len(models) > 1:
    raise SystemExit("[cell 4] 単一 model 制約に反する（configs/sampling.yaml に model が複数ある）")
for child in sorted((AGENT_DIR / "adapters").iterdir()) if (AGENT_DIR / "adapters").is_dir() else []:
    if child.is_dir():
        print("  adapter:", child.name,
              "| config:", (child / "adapter_config.json").is_file(),
              "| weights:", (child / "adapter_model.safetensors").is_file())
print("[cell 4] prompts:", sorted(p.name for p in (AGENT_DIR / "prompts").iterdir()) if (AGENT_DIR / "prompts").is_dir() else "-")
''')

# ---------------------------------------------------------------- 5. アダプタ込み server
md("""## cell 5 — アダプタ込みの server（`adk_submission.VllmServer`）+ 1 文生成

Getting Started の Cell 4 を **同じ引数で**叩きます（GPU 枚数だけ違う）。公式の key となる引数:

| 引数 | 公式（Getting Started） | この notebook |
|---|---|---|
| `tool_call_parser` / `reasoning_parser` | `gemma4` / `gemma4` | 同じ（**既定の `hermes` にしない**。ここが違うと tool call が壊れる） |
| `max_model_len` | 32768 | 同じ |
| `dtype` | `bfloat16` if `is_bf16_supported()` else `auto` | 同じ |
| `gpu_memory_utilization` | 0.90 | 0.90（VRAM が足りないなら `GPU_MEM = 0.85` に落とす） |
| `tensor_parallel_size` | gpu 数で 1 / 2 / 4 | Colab は 1 |
| `enable_lora` / `max_loras` | `bool(adapter_map)` / `max(1, len(adapter_map))` | 同じ（adapter が無いなら `False`） |
| `max_lora_rank` | `max(detected_ranks)`（AdapterInfo.rank）で無ければ 128 | 同じ |
| `scheduler_reserve_full_isl` | False | 同じ |
| `extra_args` | 12B だけ `--hf-overrides {"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}` | 同じ table |
| `startup_timeout` | 1200 s | 同じ |
| model 検証 | `discover_declared_models()` が 1 個 / 許可 4 model の内側 / adapter 名は `[A-Za-z0-9][A-Za-z0-9._-]*` | 同じ asserts |
| registry | `create_model_registry(aliases=[declared_model], model_prefix="openai/", api_key=LOCAL_API_KEY)` | 同じ |

`discover_adapters(agent_dir, adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)` の manifest を
`VllmServer(cfg, adapter_manifest=...)` に渡すと vLLM が LoRA module として load し、served id に adapter 名が入ります。
**adapter が 0 個なら `adapters=[]` と出て `enable_lora=False` になる**（今の提出 tree は LoRA 無し）。

**cell 6 が同じ server を使うので kernel を restart しないこと。**
""")

code('''import os
import re

import litellm
import torch
from adk_submission import VllmConfig, VllmServer, discover_adapters
from adk_submission.discovery import discover_declared_models
from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS

litellm.drop_params = True                        # Getting Started の cell 4 と同じ（ADK 側のパラメータを vLLM に渡さない）

ALLOWED_MODEL_NAMES = frozenset({"gemma-4-31b-it-qat-w4a16-ct", "gemma-4-12b-it-qat-w4a16-ct",
                                 "gemma-4-e4b-it-qat-w4a16-ct", "gemma-4-e2b-it-qat-w4a16-ct"})
MODEL_VLLM_EXTRA_ARGS = {"gemma-4-12b-it-qat-w4a16-ct": ("--hf-overrides",
                           '{"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}')}
_SAFE_PATH_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
INFERENCE_API_KEY = os.environ.get("LOCAL_API_KEY", "EMPTY")
GPU_MEM = 0.90                                    # Getting Started は 0.90 固定。VRAM が足りないなら 0.85 に落とす
TP_SIZE = 1                                       # Colab は GPU 1 枚（公式 4 枚 / Getting Started は tp=4）

REPO = Path("/content/repo")
AGENT_DIR = REPO / "work" / "agent_dir"

declared = discover_declared_models(AGENT_DIR)
assert len(declared) == 1, f"単一 model 制約に反する: {sorted(declared)}"
declared_model = next(iter(declared))
assert declared_model in ALLOWED_MODEL_NAMES, f"許可されていない model: {declared_model}"
MODEL_DIR = REPO / "models" / "google" / "gemma-4" / "other" / declared_model / "2"
assert (MODEL_DIR / "model.safetensors").is_file(), f"model が無い: {MODEL_DIR}（cell 2 を見る）"

adapters = discover_adapters(str(AGENT_DIR), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
adapter_map = getattr(adapters, "adapters", None) or {}
for adapter_name in adapter_map:
    assert _SAFE_PATH_COMPONENT.fullmatch(adapter_name), f"adapter 名が unsafe: {adapter_name!r}"
ranks = [r for info in adapter_map.values() if (r := getattr(info, "rank", None)) is not None and r > 0]

gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
dtype = "bfloat16" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else "auto"
print(f"[cell 5] gpus={gpu_count} tp={TP_SIZE} dtype={dtype} gpu_mem={GPU_MEM} "
      f"adapters={sorted(adapter_map)} ranks={ranks or '-'}")

if globals().get("server") is not None:
    globals()["server"].stop()

cfg = VllmConfig(model=str(MODEL_DIR), port=8000, host="127.0.0.1",
                 tool_call_parser="gemma4", reasoning_parser="gemma4",
                 max_model_len=32768, dtype=dtype, gpu_memory_utilization=GPU_MEM,
                 enable_auto_tool_choice=True, enable_lora=bool(adapter_map),
                 max_loras=max(1, len(adapter_map)), max_lora_rank=max(ranks) if ranks else 128,
                 scheduler_reserve_full_isl=False, tensor_parallel_size=TP_SIZE, startup_timeout=60 * 20,
                 extra_args=list(MODEL_VLLM_EXTRA_ARGS.get(declared_model, ())))
server = VllmServer(cfg, adapter_manifest=adapters)
server.start()
print("[cell 5] vLLM up:", server.base_url, f"(tp={TP_SIZE})")

# ADK / LiteLlm が model 名で呼ぶ registry（Getting Started と同じ openai/ prefix）
models = server.create_model_registry(aliases=[declared_model], model_prefix="openai/", api_key=INFERENCE_API_KEY)
print("[cell 5] registry:", type(models).__name__)

from openai import OpenAI

client = OpenAI(base_url=server.base_url, api_key=INFERENCE_API_KEY)
served = [m.id for m in client.models.list().data]
print("[cell 5] served model ids:", served)
for name in [declared_model, *sorted(adapter_map)]:
    try:
        resp = client.chat.completions.create(model=name, max_tokens=32,
                                              messages=[{"role": "user", "content": "Reply with exactly: OK"}])
        print(f"[cell 5] completion {name!r}:", repr(resp.choices[0].message.content))
    except Exception as exc:
        print(f"[cell 5] completion {name!r} 失敗:", type(exc).__name__, str(exc)[:160])
''')

# ---------------------------------------------------------------- 6. ハーネス実行
md("""## cell 6 — ハーネス実行（`swegemma eval`）: Phase 1 agent loop + Phase 2 verification

公式 CLI をそのまま subprocess で叩きます（Colab の kernel 側に event loop があっても安全な形）。
モデル endpoint は env `LOCAL_INFERENCE_URL`（= cell 5 の `server.base_url`、`/v1` を含む）で渡します。
adapter がある場合 `swegemma` は `discover_adapters` で見つけた adapter 名を `openai/<adapter 名>` として register します。

`--sandbox subprocess` にするのは Colab に docker daemon が無いからです（公式は docker container sandbox）。
Phase 2 は snapshot を展開して patch apply → test patch → `pytest` + JUnit XML。1 task 落ちても次の task には進みます。
""")

code('''import os
import subprocess
from pathlib import Path

REPO = Path("/content/repo")
AGENT_DIR = REPO / "work" / "agent_dir"
DATA = REPO / "data"
RESULTS = REPO / "work" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

TASK_IDS = globals().get("TASK_IDS", [])
BUDGET = globals().get("BUDGET", {"max_time_minutes": 20.0, "max_tool_calls": 80, "max_turns": 150})

os.environ["LOCAL_INFERENCE_URL"] = server.base_url          # "/v1" を含む。足さないこと
os.environ["LOCAL_API_KEY"] = os.environ.get("LOCAL_API_KEY", "EMPTY")
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost,0.0.0.0,::1")
os.environ.setdefault("no_proxy", "127.0.0.1,localhost,0.0.0.0,::1")

cmd = ["python3", "-m", "swegemma", "eval",
       "--tasks", str(DATA / "tasks.jsonl"),
       "--snapshots-dir", str(DATA / "snapshots"),
       "--results-dir", str(RESULTS),
       "--submission-dir", str(AGENT_DIR),
       "--sandbox", "subprocess",
       "--max-time-minutes", str(BUDGET.get("max_time_minutes", 20.0)),
       "--max-tool-calls", str(BUDGET.get("max_tool_calls", 80)),
       "--max-turns", str(BUDGET.get("max_turns", 150)),
       "--display", "single"]
for iid in TASK_IDS:
    cmd += ["--task-id", iid]
print("+ " + " ".join(cmd), flush=True)
proc = subprocess.run(cmd, cwd=str(REPO))                    # output は notebook に live で出る
print("[cell 6] swegemma eval exit:", proc.returncode, flush=True)
print("[cell 6] results:", RESULTS)
''')

# ---------------------------------------------------------------- 7. スコア算出
md("""## cell 7 — スコア算出（`resolution_rate` = 公式の Resolution Rate）

harness が `results/summary.json` に `total_tasks / resolved / resolution_rate / by_repo / errors` を、
`results/task_results.jsonl` に task 単位（`resolved`, `agent_patch_size`, `test_exit_code`, `duration_seconds`,
`tool_calls`, `error`）を書きます（`patches/<id>.patch`, `test_outputs/<id>.log` も）。
公式スコアはこの `resolution_rate` そのものです。

- **`failures.csv` / `failures.md` に task 別の不出来を書き出します**（status = `resolved` / `patch_fail` / `no_patch` / `crashed` / `not_run`、理由、test exit code、patch size、tool calls、duration、protected path 接触、pytest 末尾 3 行、status 集計、代表エラー、protected path list）。`not_run` は「測れていない task」です
- この実装は `src/pipeline/failures_report.py` の物そのもの（cell に注入しています）。pipeline 側の stage `analyze`（`make pipeline-post`）も同じ CSV を出すので、**Colab 側と Kaggle 側で列が揃います**。手元では `python3 src/pipeline/failures_report.py --results results/<run>`
- `server.py` の `/v1` は `base_url` に既に含まれる（二重に足すと 404）
- `test_exit_code != 0` = Phase 2 の pytest が落ちた（patch は出たが直っていない）
- `agent_patch_size = 0` = agent が `submit_patch` を呼ばなかった（budget 切れ / tool 失敗）
""")

code('''import json
from pathlib import Path

REPO = Path("/content/repo")
RESULTS = REPO / "work" / "results"

rows = []
for line in (RESULTS / "task_results.jsonl").read_text(encoding="utf-8").splitlines() if (RESULTS / "task_results.jsonl").is_file() else []:
    if line.strip():
        rows.append(json.loads(line))
summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8")) if (RESULTS / "summary.json").is_file() else {}

print(f"[score] resolution_rate = {summary.get('resolution_rate', 'n/a')} "
      f"({summary.get('resolved', 0)}/{summary.get('total_tasks', len(rows))})  <- 公式スコアと同じ意味")
for row in rows:
    print(f"  {row.get('instance_id'):24s} resolved={str(row.get('resolved')):5s} "
          f"patch={row.get('agent_patch_size', 0):>6} exit={row.get('test_exit_code')} "
          f"tools={row.get('tool_calls', 0):>3} dur={row.get('duration_seconds', 0):>6} "
          f"error={str(row.get('error'))[:40]}")

buckets = {"no_patch": 0, "patch_but_test_fail": 0, "resolved": 0, "crashed": 0}
for row in rows:
    if row.get("resolved"):
        buckets["resolved"] += 1
    elif row.get("error"):
        buckets["crashed"] += 1
    elif not row.get("agent_patch_size"):
        buckets["no_patch"] += 1
    else:
        buckets["patch_but_test_fail"] += 1
print("[score] 分類:", buckets)
score_out = {"summary": summary, "rows": rows, "buckets": buckets,
             "hardware": {"gpus": 1, "note": "Colab L4 tp=1 / 0.85（公式は 4x L4 tp=4 / 0.90）"},
             "sandbox": "subprocess（公式は docker container）", "max_model_len": 32768}
(RESULTS / "score.json").write_text(json.dumps(score_out, ensure_ascii=False, indent=2), encoding="utf-8")
print("[score] 書いた:", RESULTS / "score.json")

''' + FAILURES_SRC + '''
# 同じ task 別 report 実装を pipeline 側（src/pipeline/failures_report.py）から注入している。
_report = write_failures(RESULTS, task_ids=globals().get("TASK_IDS") or None)
print("[score] task 別の不出来: form=" + _report["form"] + " 行=" + str(_report["rows"]) + " 集計=" + str(_report["counts"]))
print("[score] 書いた: " + str(_report["csv"]) + " / " + str(_report["md"]))
_report_md = Path(_report["md"]).read_text(encoding="utf-8")
print(_report_md[:1200])

# 自分の harness（src/pipeline の eval）や別の run と突き合わせる欄。None のままなら何もしない。
PREV_RESULTS = None            # 例 "/content/repo/work/results/colab12" や results/colab12（predictions_*.json でも可）


def load_tree(d):
    d = Path(d)
    hit = sorted(d.glob("**/task_results.jsonl"))
    if hit:
        data = [json.loads(l) for l in hit[0].read_text(encoding="utf-8").splitlines() if l.strip()]
        return {str(r.get("instance_id")): bool(r.get("resolved")) for r in data}, "公式 harness"
    hit = sorted(d.glob("**/predictions_*.json"))
    if hit:
        data = json.loads(hit[0].read_text(encoding="utf-8"))
        return {str(r.get("id")): bool(r.get("resolved")) for r in data}, "自分の pipeline"
    return {}, "?"


if PREV_RESULTS:
    other, form = load_tree(PREV_RESULTS)
    here = {str(r.get("instance_id")): bool(r.get("resolved")) for r in rows}
    shared = sorted(set(other) & set(here))
    diff = [t for t in shared if other[t] != here[t]]
    print(f"[parity] PREV_RESULTS={PREV_RESULTS} ({form}) 共有 {len(shared)} / 食い違い {len(diff)}")
    for t in diff:
        print(f"   - {t}: そちら={other[t]} / 今回の run={here[t]}")
    if not diff and shared:
        print("[parity] resolved は全部一致。harness の乖離は出ていない")
    if not shared:
        print("[parity] task 集合が交差しない。TASK_IDS / task_limit を揃えること")

for name in ("patches", "test_outputs"):
    d = RESULTS / name
    if d.is_dir():
        print(f"  {name}: {len(list(d.iterdir()))} file")
''')

# ---------------------------------------------------------------- 8. zip + validate
md("""## cell 8 — 提出 zip と `validate_directory`（ staging した agent を zip にする）

`agent.yaml` が zip の root に来るようにします。rules（HARNESS_README）: unpacked < 3 GiB（adapters 込み）、
model 名は許可 list、adapter は `adapters/<name>/adapter_config.json` + `adapter_model.safetensors`（`.safetensors` のみ）。
""")

code('''import shutil
import zipfile
from pathlib import Path

from adk_submission import validate_directory

REPO = Path("/content/repo")
AGENT_DIR = REPO / "work" / "agent_dir"
ZIP = REPO / "work" / "submission.zip"

try:
    report = validate_directory(AGENT_DIR)
    print("[cell 8] validate_directory OK:", report)
except Exception as exc:
    print(f"[cell 8] validate_directory 失敗: {type(exc).__name__}: {exc}")
    raise SystemExit("提出物として通らない箇所がある。直してから zip を作ること")

if ZIP.exists():
    ZIP.unlink()
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    for path in sorted(AGENT_DIR.rglob("*")):
        if path.is_file():
            zf.write(path, path.relative_to(AGENT_DIR).as_posix())
print(f"[cell 8] zip: {ZIP} {ZIP.stat().st_size:,} byte")
with zipfile.ZipFile(ZIP) as zf:
    names = zf.namelist()
    print("  root に agent.yaml:", "agent.yaml" in names, "| entries:", len(names))
    print("  sample:", names[:8])

# zip にした物をもう一度開いて、公式が受け取る形のまま通るか確認する（提出物の自己検査）
UNZIPPED = REPO / "work" / "unzipped_submission"
if UNZIPPED.exists():
    shutil.rmtree(str(UNZIPPED))
UNZIPPED.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(ZIP) as zf:
    zf.extractall(str(UNZIPPED))
extracted = {p.relative_to(UNZIPPED).as_posix() for p in UNZIPPED.rglob("*") if p.is_file()}
staged = {p.relative_to(AGENT_DIR).as_posix() for p in AGENT_DIR.rglob("*") if p.is_file()}
try:
    report2 = validate_directory(UNZIPPED)
    ok = type(report2).__name__
except Exception as exc:
    ok = f"失敗 {type(exc).__name__}: {exc}"
print("[cell 8] zip round-trip: staged と zip の集合一致:", extracted == staged,
      "| 差:", sorted(extracted ^ staged)[:6] or "-", "| 展開後に validate_directory:", ok)
print("[cell 8] 提出物は agent tree（宣言 + adapters）だけ。harness 自体は wheelhouse dataset から入れている")
''')

# ---------------------------------------------------------------- 9. 後始末
md("""## cell 9 — server を止えて物を下ろす

`server.stop()` を忘れると vLLM が GPU を掴んだままになります（Colab は session を止めるまで VM を離さない）。
物は Files panel か `colab download` で下ろします。提出だけ Kaggle CLI（手元）です。

```bash
colab download /content/repo/work/submission.zip /tmp/submission.zip -s g4one
colab download /content/repo/work/results/summary.json /tmp/summary.json -s g4one
kaggle competitions submit -c gemma-4-developer-agent -f submission.zip -m "e4b agent (prompt+config)"
kaggle competitions submissions -c gemma-4-developer-agent
```
""")

code('''import json
from pathlib import Path

try:
    server.stop()
    print("[cell 9] server stopped")
except NameError:
    print("[cell 9] server は起動していなかった")
except Exception as exc:
    print("[cell 9] server.stop():", type(exc).__name__, exc)

REPO = Path("/content/repo")
for pat in ("work/submission.zip", "work/results/summary.json", "work/results/task_results.jsonl",
            "work/results/score.json", "work/results/failures.csv", "work/results/failures.md",
            "work/results/patches/*", "work/results/test_outputs/*"):
    hits = sorted(REPO.glob(pat))[:6]
    print(f"  {pat:38s} " + (", ".join(p.name for p in hits) or "-"))
summary = REPO / "work" / "results" / "summary.json"
if summary.is_file():
    print("[cell 9] 記録:", json.dumps(json.loads(summary.read_text(encoding="utf-8")), ensure_ascii=False))
''')

md("""## 公式採点との差（manifest に残す前提）

| 項目 | 公式 | この notebook |
|---|---|---|
| task | 非公開 256 個（+ 公開 129 は development set） | 公開 129 のうち `TASK_LIMIT` 個 |
| GPU | 4×L4, tp=4, bfloat16, gpu_mem 0.90 | 1×L4, tp=1, bfloat16, 0.85 |
| sandbox | docker container（`swebench-sandbox:latest`） | `--sandbox subprocess`（Colab に docker daemon が無い） |
| vLLM | wheelhouse pin 0.19.1 | 同じ（cell 3 で pin に戻す） |
| model | `gemma-4-e4b-it-qat-w4a16-ct` + 提出 adapters | 同じ zip の tree を staging して adapters は `discover_adapters` で load |
| budget | scorer 既定 60 min / 100 tools / 500 turns（`HARNESS_README.md`。`sample_submission/eval_config.yaml` は 1 min / 10 tools / 50 turns） | cell 4 の `BUDGET`（20 min / 80 tools / 150 turns）で上書き |
| score | Resolution Rate | `summary.json` の `resolution_rate` |

公式スコアが欲しいなら `work/submission.zip` を提出してください（提出物は公式資材・公式環境で評価されます）。
""")


def notebook_text(source: Path, mode: str = "kaggle") -> str:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    header = (f"出所: `{source.name}` ／ 生成 {stamp} ／ `MATERIAL_SOURCE = \"{mode}\"`\n\n"
              "**Colab に上げるのはこの notebook 1 個だけ**（Kaggle 資材で測る既定モードは + Secrets の `KAGGLE_API_TOKEN`）。"
              "`src/pipeline/` は読みません。cell 5 の server を cell 6 が使うので kernel を restart しないこと。")
    cells = []
    header_merged = False
    for kind, text in CELLS:
        if kind == "code" and mode != "kaggle":
            text = text.replace('MATERIAL_SOURCE = "kaggle"', f'MATERIAL_SOURCE = "{mode}"')
        if kind == "markdown" and not header_merged:      # 冒頭の markdown cell に出所・生成時刻を混ぜる
            text = header + "\n\n" + text
            header_merged = True
        cell = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.11"}},
          "nbformat": 4, "nbformat_minor": 4}
    return json.dumps(nb, ensure_ascii=False, indent=1) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("kaggle", "hf"), default="kaggle",
                    help="kaggle = 公式資材（既定）/ hf = HF mirror + PyPI（Kaggle を触らない）")
    ap.add_argument("--out", help="既定: kaggle なら kernels/colab/gemma4_colab.ipynb、hf なら kernels/colab/gemma4_colab_hf.ipynb")
    args = ap.parse_args()
    out = Path(args.out or ("kernels/colab/gemma4_colab.ipynb" if args.source == "kaggle" else "kernels/colab/gemma4_colab_hf.ipynb"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(notebook_text(Path(__file__).resolve(), args.source), encoding="utf-8")
    print(f"[colab-ipynb] wrote {out}（cells={len(CELLS)}, code cells={sum(1 for k, _ in CELLS if k == 'code')}）")
    try:
        import nbformat

        nb = nbformat.reads(out.read_text(encoding="utf-8"), as_version=4)
        nbformat.validate(nb)
        print(f"[colab-ipynb] nbformat validate OK（{len(nb.cells)} cells）")
    except ImportError:
        print("[colab-ipynb] nbformat が無いので validate は飛ばした")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
