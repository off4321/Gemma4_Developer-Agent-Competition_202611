"""Colab VM 上で data / wheelhouse / model の download を **裏で**走らせるだけの起動スクリプト。

理由: `colab exec` は kernel の出力が 30 s 以上無いと `TimeoutError: Timeout waiting for output`
を出す（実測 2026-11-21）。867 MB の wheelhouse と 11.5 GB の model を前面に流すと必ず落ちるので、
**kernel から切り離した detached process** にして log ファイルに出し、この exec は即 return する。
 progress は `colab_poll.py` で読む。

    colab exec -s <session> -f src/pipeline/colab_fetch.py            # data + wheelhouse
    colab exec -s <session> -f src/pipeline/colab_fetch.py --model    # model も足す
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = os.environ.get("REPO_DIR", "/content/repo")
LOG_DIR = "/content"


def main() -> int:
    # `colab exec` は argv を native の様に渡さないので、env でも受け取る（--env WITH_MODEL=1）
    with_model = "--model" in sys.argv[1:] or os.environ.get("WITH_MODEL", "").lower() in ("1", "true", "yes")
    # tarball が有れば展開（upload 後に毎回同期）
    tar = Path("/content/repo.tgz")
    if tar.is_file():
        subprocess.run(["tar", "xzf", str(tar), "-C", REPO], check=False)

    env = dict(os.environ)
    env.update({
        "REPO_DIR": REPO,
        "SKIP_MODEL": "0" if with_model else "1",
        "PYTHONUNBUFFERED": "1",
    })
    log = f"{LOG_DIR}/bootstrap_{'all' if with_model else 'data'}.log"
    cmd = ["python3", "src/pipeline/colab_bootstrap.py"]
    proc = subprocess.Popen(
        cmd, cwd=REPO, env=env,
        stdout=open(log, "w"), stderr=subprocess.STDOUT,
        start_new_session=True, stdin=subprocess.DEVNULL,
    )
    print(f"[fetch] launched pid={proc.pid} log={log} model={'yes' if with_model else 'no'}")
    print(f"[fetch] progress は colab_poll.py で読むこと")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
