#!/usr/bin/env python3
"""Colab VM で **notebook を 1 kernel でまとめて実行**する起動スクリプト（cell 間の state を保つ）。

`colab exec` で cell を 1 個ずつ流すのは無理があります（別 process になるので `ctx` / `ctx.server` が続くかない）。
なので `jupyter nbconvert --execute` を **detached** で走らせて、実行済み notebook を出力します。

    colab upload kernels/colab/gemma4_colab.ipynb /content/gemma4_colab.ipynb -s <session>
    colab exec   -s <session> -f src/pipeline/colab_run_notebook.py --timeout 200
    colab download /content/notebook_run.log /tmp/nb_run.log -s <session>
    colab download /content/gemma4_colab_measure_executed.ipynb /tmp/executed.ipynb -s <session>

cell 3（資材 download）と cell 6（setup）が重いので、`SKIP_CELLS` で飛ばして手元で 1 個ずつ回すこともできます。
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

NB = Path(os.environ.get("NOTEBOOK", "/content/gemma4_colab.ipynb"))
OUT = Path(os.environ.get("NOTEBOOK_OUT", "/content/gemma4_colab_measure_executed.ipynb"))
LOG = Path(os.environ.get("NOTEBOOK_LOG", "/content/notebook_run.log"))
TIMEOUT = os.environ.get("CELL_TIMEOUT", "3600")


def main() -> int:
    if not NB.is_file():
        print(f"[run_notebook] notebook が無い: {NB}（colab upload すること）")
        return 1
    if shutil.which("jupyter") is None:
        print("[run_notebook] jupyter が無い -> pip install -q nbclient nbconvert ipykernel を先に")
        return 1
    shutil.copy2(NB, OUT)
    cmd = ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
           "--output", str(OUT), str(NB),
           f"--ExecutePreprocessor.timeout={TIMEOUT}", "--ExecutePreprocessor.kernel_name=python3"]
    proc = subprocess.Popen(cmd, stdout=open(LOG, "a"), stderr=subprocess.STDOUT,
                            start_new_session=True, stdin=subprocess.DEVNULL)
    print(f"[run_notebook] launched pid={proc.pid} notebook={NB} out={OUT} log={LOG}")
    print("[run_notebook] 進捗: colab download /content/notebook_run.log -s <session>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
