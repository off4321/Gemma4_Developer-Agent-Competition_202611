"""Colab VM に repo を展開し、`~/.kaggle` を用意し、環境の要物を 1 回で報告する。

session を作り直した直後に必ずこれを通す（`colab upload` は存在しない dir だと 500 になる、
`kaggle` CLI は `~/.kaggle/kaggle.json` が無いと "Missing key in configuration." を出す）。
`src/pipeline` を `PYTHONPATH` に載せるのもここで行う（stage script が `import pipeline...` を使う）。
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

REPO = Path("/content/repo")
KAGGLE = Path.home() / ".kaggle"


def run(cmd: str, limit: int = 500) -> str:
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    out = ((proc.stdout or "") + (proc.stderr or "")).strip()
    print(out[:limit] or "(empty)", flush=True)
    return out


def main() -> int:
    KAGGLE.mkdir(parents=True, exist_ok=True)
    out = run(f"mkdir -p {REPO} && tar xzf /content/repo.tgz -C {REPO} && ls {REPO}")
    if "src" not in out:
        print("[ensure_repo] src が無い -> upload が通っていない", flush=True)
        return 1
    run("python3 -V; nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader")
    run(f"df -h /content | tail -1; free -g | head -2 | tail -1")
    run(f"ls {REPO}/data 2>/dev/null | head -8; ls {REPO}/models 2>/dev/null | head -3")
    print(f"[ensure_repo] PYTHONPATH に {REPO}/src を足す", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
