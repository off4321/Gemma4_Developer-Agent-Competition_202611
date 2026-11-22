#!/usr/bin/env python3
"""Kaggle kernel の CONFIG["VARIANT"] を書き換える（`make pipeline-run VARIANT=...` 用）。

Kaggle kernel には実行引数を渡せない。だから variant の選択は **コードの中に書き込んでから push** する。
このスクリプトはその 1 行だけ直す（手で探して書き間違える事故を防ぐ）。

    python3 src/pipeline/set_kernel_variant.py smoke [--kernel kernels/eval/gemma4_e4b_eval.py]

注意: ここが直るのは kernel 側の入口だけ。budget / sampling / task 選別は
`src/pipeline/variants.yaml` 側にあり、それは code dataset として上がる。
 kernel 側と local 側で設定がズレる原因の 9 割はここなので、**push 前に make pipeline-plan
で同じ variant の計画を見てから**実行すること。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

VARIANT_RE = re.compile(r'("VARIANT":\s*")[^"]*(")')


def set_variant(kernel: Path, variant: str) -> int:
    """CONFIG["VARIANT"] を 1 か所だけ差し替える。見つからなければ 2 を返す（黙って成功しない）。"""
    if not kernel.is_file():
        print(f"[set-variant] kernel ファイルが無い: {kernel}")
        return 2
    if not re.fullmatch(r"[A-Za-z0-9._-]+", variant):
        print(f"[set-variant] 使えない variant 名: {variant!r}")
        return 2
    text = kernel.read_text(encoding="utf-8")
    new, count = VARIANT_RE.subn(rf"\g<1>{variant}\g<2>", text, count=1)
    if count != 1:
        print(f"[set-variant] {kernel} に CONFIG[\"VARIANT\"] が見つからない")
        return 2
    kernel.write_text(new, encoding="utf-8")
    line = next((i for i, l in enumerate(new.splitlines(), 1) if '"VARIANT"' in l), 0)
    print(f"[set-variant] {kernel}:{line} -> VARIANT={variant}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="kernel CONFIG の variant を書き換える")
    parser.add_argument("variant", help="src/pipeline/variants.yaml に定義された variant 名")
    parser.add_argument("--kernel", default="kernels/eval/gemma4_e4b_eval.py")
    parser.add_argument("--show", action="store_true", help="現在の値を表示するだけ（変更しない）")
    args = parser.parse_args(argv)

    kernel = Path(args.kernel)
    if args.show:
        text = kernel.read_text(encoding="utf-8")
        match = VARIANT_RE.search(text)
        print(f"[set-variant] {kernel} の現在: {match.group(0) if match else '見つからない'}")
        return 0
    return set_variant(kernel, args.variant)


if __name__ == "__main__":
    sys.exit(main())
