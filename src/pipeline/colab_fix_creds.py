"""Colab VM 側に Kaggle credentials を正しい形に並べ直すだけのスクリプト。

Kaggle CLI の newer 版は `~/.kaggle/credentials.json`、older 版は `~/.kaggle/kaggle.json` を読む。
どちらで認証されるか分からないので **両方置いて 600 にする**。 perms が 600 でないと CLI が拒否する。

    colab exec -s <session> -f src/pipeline/colab_fix_creds.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

KAGGLE_DIR = Path.home() / ".kaggle"


def main() -> int:
    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    have = sorted(p.name for p in KAGGLE_DIR.iterdir()) if KAGGLE_DIR.is_dir() else []
    print(f"[creds] dir に有るもの: {have}")

    src = KAGGLE_DIR / "credentials.json"
    dst = KAGGLE_DIR / "kaggle.json"
    if src.is_file() and not dst.is_file():
        shutil.copy2(src, dst)
        print(f"[creds] {src.name} -> {dst.name} にコピー")
    elif dst.is_file() and not src.is_file():
        shutil.copy2(dst, src)
        print(f"[creds] {dst.name} -> {src.name} にコピー")
    elif not src.is_file() and not dst.is_file():
        print("[creds] どちらも無い -> `colab upload <local> /root/.kaggle/credentials.json` が必要")
        return 1

    for name in ("credentials.json", "kaggle.json"):
        p = KAGGLE_DIR / name
        if p.is_file():
            os.chmod(p, 0o600)

    cmd = ("kaggle --version 2>&1 | head -2; ls -l /root/.kaggle | tail -4; "
           "kaggle competitions files gemma-4-developer-agent 2>&1 | head -4")
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print((proc.stdout or proc.stderr).strip()[:1200])
    return 0 if "name" in (proc.stdout or "") else 1


if __name__ == "__main__":
    raise SystemExit(main())
