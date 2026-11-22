"""Colab VM で pipeline の stage を **裏 process で**走らせて、即 return するだけの起動スクリプト。

`colab exec` は kernel が 30 s 以上無出力だと `TimeoutError`、長い exec は `Connection was lost.`
で切れる（both 実測 2026-11-21）。setup の pip install (~10 分) や server/eval は前面に置けないので、
detached process にして log をファイルに出し、 progress は `colab_poll.py` で読む。

    colab exec -s <session> -f src/pipeline/colab_stage.py --env STAGES=setup --env VARIANT=colab
    colab exec -s <session> -f src/pipeline/colab_stage.py --env STAGES=server --env VARIANT=colab
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO = os.environ.get("REPO_DIR", "/content/repo")
STAGES = os.environ.get("STAGES", "setup")
VARIANT = os.environ.get("VARIANT", "colab")
EXTRA = os.environ.get("EXTRA_ARGS", "").split()


def main() -> int:
    tar = Path("/content/repo.tgz")
    if tar.is_file():
        subprocess.run(["tar", "xzf", str(tar), "-C", REPO], check=False)
    # untar 時に model dir へ紛れ込んだ tar.gz（7.3 GB）を消す（stage には不要）
    stray = (Path(REPO) / "models" / "google" / "gemma-4" / "other" /
             "gemma-4-e4b-it-qat-w4a16-ct" / "2" / "gemma-4.tar.gz")
    if stray.is_file():
        stray.unlink()
        print("[stage] model dir の stray tar.gz を削除")
    log = f"/content/stage_{STAGES.replace(',', '_')}.log"
    env = dict(os.environ)
    env.update({"REPO_DIR": REPO, "PYTHONUNBUFFERED": "1", "PYTHONPATH": f"{REPO}/src"})
    cmd = ["python3", "src/pipeline/run_pipeline.py", "--variant", VARIANT,
           "--stages", STAGES, "--strict", "--data-dir", f"{REPO}/data",
           "--working-dir", "/content/work"] + EXTRA
    proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=open(log, "w"),
                            stderr=subprocess.STDOUT, start_new_session=True,
                            stdin=subprocess.DEVNULL)
    print(f"[stage] launched pid={proc.pid} stages={STAGES} variant={VARIANT} log={log}")
    print("[stage] 進捗: colab exec -s <session> -f src/pipeline/colab_poll.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
