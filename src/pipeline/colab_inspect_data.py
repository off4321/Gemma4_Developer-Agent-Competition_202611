"""Colab VM に落とした competition data の中身を見るだけの検査。

`kaggle competitions download -c <comp> -p data` は newer 版だと **`data/<comp>/...` の下に**
展開する（実測 2026-11-21: `/content/repo/data/gemma-4-developer-agent/`）。stage は `data/tasks.jsonl`
/ `data/graphs` / `data/embeddings` を見るので、このままだと見つからない。まず中身と size を見る。

    colab exec -s <session> -f src/pipeline/colab_inspect_data.py
"""
from __future__ import annotations

import subprocess
from pathlib import Path

REPO = "/content/repo"
DATA = Path(REPO) / "data"


def sh(cmd: str, limit: int = 1500) -> str:
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return ((proc.stdout or "") + (proc.stderr or "")).strip()[:limit] or "(empty)"


def main() -> int:
    print("--- du ---")
    print(sh(f"du -sh {DATA}/* 2>/dev/null | sort -h | tail -12"))
    print("--- tree (2 levels,  biggest 20 files) ---")
    print(sh(f"find {DATA} -maxdepth 2 -type d | head -20"))
    print(sh(f"find {DATA} -type f | head -12"))
    print(sh(f"find {DATA} -type f -printf '%s %p\\n' 2>/dev/null | sort -rn | head -8"))
    print("--- tasks.jsonl を探す ---")
    print(sh(f"find {DATA} -name 'tasks.jsonl' -o -name 'HARNESS_README.md' | head -6"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
