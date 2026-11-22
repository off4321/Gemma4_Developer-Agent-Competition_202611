"""Colab VM の preinstall kaggle CLI が壊れているときの修理（import 失敗の中身を出して pip で上げ直す）。

Colab image の `kaggle` は `/usr/local/lib/python3.13/dist-packages/kaggle/__init__.py` で
import 段階に失敗することがある（実測 2026-11-21: `from kaggle.cli import main` が落ちる）。
直すと competition / dataset / model の download が Colab 側から打てる。

    colab exec -s <session> -f src/pipeline/colab_fix_kaggle_cli.py
"""
from __future__ import annotations

import subprocess
import traceback


def show(cmd: str, limit: int = 900) -> None:
    print(f"$ {cmd}")
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print((proc.stdout or proc.stderr).strip()[:limit] or "(empty)", flush=True)


def try_import(tag: str) -> bool:
    proc = subprocess.run(
        ["python3", "-c", "import kaggle, sys; print('kaggle import OK', "
         "getattr(kaggle, '__version__', 'no __version__'))"],
        capture_output=True, text=True)
    out = (proc.stdout + proc.stderr).strip()
    print(f"[{tag}] {out[-700:]}")
    return proc.returncode == 0


def main() -> int:
    show("pip list 2>/dev/null | grep -i '^kaggle'")
    if try_import("before"):
        show("kaggle --version 2>&1 | head -2")
        return 0
    print("\n--- 完整 traceback ---")
    try:
        import kaggle  # noqa: F401
    except Exception:
        traceback.print_exc()
    show("python3 -m pip install -q -U kaggle 2>&1 | tail -4", 1200)
    if try_import("after"):
        show("kaggle --version 2>&1 | head -2")
        show("kaggle competitions files gemma-4-developer-agent 2>&1 | head -4")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
