#!/usr/bin/env python3
"""薄い .py kernel ラッパーから Kaggle 用の .ipynb を生成する。

なぜ必要か（実測で当たった罠）: kernels/eval を `kernel_type: script` で push すると、
Kaggle 側は notebook 扱いで nbconvert を走らせ、こう落ちる:

    nbformat.reader.NotJSONError: Notebook does not appear to be JSON: '#!/usr/bin/env python3...'

つまり **CLI から push する GPU kernel は .ipynb のほうが安全**だ（GPU quota は消費されない
失敗だったが、実行前に止まる）。なので:

  * ロジックと説明は .py に書く（この repo の source of truth。docstring で日本語で説明）
  * Kaggle に上がるのは **その .py を 1 cell に収めた .ipynb**（生成物。手で触らない）
  * kernel-metadata.json は kernel_type=notebook / code_file=<生成物>

生成した ipynb は `make pipeline-run` の前に必ず作られる（ズレたら push 前に気づくように、
.ipynb の中身に .py のハッシュを markdown へ書く）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

MARKDOWN = """# {title}

このセルは **{source} を 1 cell に収したものだけ** です。本体は `src/pipeline/` にあり、
dataset `{code_slug}` として mount されます。だから同じコードが手元の preflight と
この kernel の GPU 実行の両方で走ります（doc/pipeline.md 参照）。

- kernel 側の設定: `CONFIG["VARIANT"]`（variant 名だけ。他は `src/pipeline/variants.yaml`）
- source hash: `{digest}`（`make kernel-ipynb` で生成。手で編集しないこと）
- quota 注意: ここは GPU を **実行開始** する。1 run <= 2 h 目安 / 週 30 h
"""


def build_notebook(py_path: Path, title: str, code_slug: str) -> dict:
    source = py_path.read_text(encoding="utf-8")
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
    markdown = MARKDOWN.format(title=title, source=py_path.name, code_slug=code_slug, digest=digest)
    return {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": markdown.splitlines(keepends=True)},
            {"cell_type": "code", "execution_count": None, "metadata": {},
             "outputs": [], "source": source.splitlines(keepends=True)},
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.13"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="kernel .py -> Kaggle .ipynb 生成")
    parser.add_argument("--source", default="kernels/eval/gemma4_e4b_eval.py")
    parser.add_argument("--out", default=None, help="既定: <source>.ipynb（同じ dir）")
    parser.add_argument("--title", default="gemma4-e4b-agent-eval")
    parser.add_argument("--code-slug", default="gemma4-dev-agent-code")
    parser.add_argument("--metadata", default=None,
                        help="指定すると kernel-metadata.json の kernel_type / code_file を更新する")
    args = parser.parse_args(argv)

    src = Path(args.source)
    if not src.is_file():
        print(f"[kernel-ipynb] source が無い: {src}")
        return 2
    out = Path(args.out) if args.out else src.with_suffix(".ipynb")
    nb = build_notebook(src, args.title, args.code_slug)
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[kernel-ipynb] wrote {out}（cells={len(nb['cells'])}, source={src.name}）")

    if args.metadata:
        meta_path = Path(args.metadata)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["kernel_type"] = "notebook"
        meta["code_file"] = out.name
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"[kernel-ipynb] {meta_path}: kernel_type=notebook, code_file={out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
