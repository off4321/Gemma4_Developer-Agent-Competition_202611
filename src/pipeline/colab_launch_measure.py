#!/usr/bin/env python3
"""Colab VM で `colab_measure.py` を **detached で**起動するだけ（exec 側は 1 行）。

    colab exec -s <session> -f src/pipeline/colab_launch_measure.py --env VARIANT=colab12
読み取り:
    colab download /content/measure.log /tmp/measure.log -s <session>
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO = os.environ.get("REPO_DIR", "/content/repo")
VARIANT = os.environ.get("VARIANT", "colab12")
STAGES = os.environ.get("STAGES", "data,agent,graph,server,eval,package")
LOG = "/content/measure.log"


def main() -> int:
    tar = Path("/content/repo.tgz")
    if tar.is_file():
        subprocess.run(["tar", "xzf", str(tar), "-C", REPO], check=False)
    env = dict(os.environ)
    env.update({"REPO_DIR": REPO, "PYTHONUNBUFFERED": "1", "PYTHONPATH": f"{REPO}/src"})
    cmd = ["python3", "src/pipeline/colab_measure.py", "--variant", VARIANT, "--stages", STAGES]
    proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=open(LOG, "a"),
                            stderr=subprocess.STDOUT, start_new_session=True,
                            stdin=subprocess.DEVNULL)
    print(f"[launch_measure] pid={proc.pid} variant={VARIANT} stages={STAGES} log={LOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
