"""Colab VM の裏 download（colab_fetch.py）の進行を読むだけの poll。

`colab exec -f` は 1 回 = 1 接続。このスクリプトは数秒で終わるようにして、
長い download の最中に何度でも叩ける（heartbeat にもなる）。

    colab exec -s <session> -f src/pipeline/colab_poll.py
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO = os.environ.get("REPO_DIR", "/content/repo")


def sh(cmd: str, limit: int = 1000) -> str:
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return ((proc.stdout or "") + (proc.stderr or "")).strip()[:limit] or "(empty)"


def running() -> list[str]:
    """`pgrep` が無い環境でも読めるように /proc から拾う。"""
    out = []
    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit():
            continue
        try:
            cmd = (pid_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        except OSError:
            continue
        if "colab_bootstrap" in cmd or "run_pipeline" in cmd or ("kaggle" in cmd and "download" in cmd):
            out.append(f"pid {pid_dir.name}: {cmd[:140]}")
    return out


def main() -> int:
    print(f"[poll] {sh('date -u +%Y-%m-%dT%H:%M:%SZ')}")
    procs = running()
    print("[poll] 実行中:" + ("\n  " + "\n  ".join(procs) if procs else "  (無し)"))
    for log in sorted(Path("/content").glob("stage_*.log")):
        print(f"--- tail {log.name} ---")
        print(sh(f"tail -12 {log}", 1800))
    for log in sorted(Path("/content").glob("bootstrap_*.log")):
        print(f"--- tail {log.name} ---")
        print(sh(f"tail -6 {log}", 900))
    print("--- gpu ---")
    print(sh("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || echo 'no gpu info'"))
    print(sh("ls /content/work 2>/dev/null | head -8 || echo '(work 無し)'"))
    print("--- size ---")
    for path in (f"{REPO}/data", f"{REPO}/data/wheelhouse", f"{REPO}/models"):
        print(f"{path}: " + sh(f"du -sh {path} 2>/dev/null | cut -f1 || echo -"))
    print("--- data 中身 ---")
    print(sh(f"ls {REPO}/data 2>/dev/null | head -12"))
    print(sh(f"ls {REPO}/models 2>/dev/null | head -6"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
