#!/usr/bin/env python3
"""Colab VM 側で fetch 完了を待って setup -> 計測 run までを 1 process で走らせる driver.

`colab exec` は 30 s 無出力で切れる / 長時間だと "Connection was lost." になる（実測）。
なので **VM 側で detached で走らせる**。`scripts/colab_watchdog.sh` が生存只看る。

    python3 src/pipeline/colab_launch_measure.py --variant colab12   # exec 側はこれ 1 行
    tail /content/measure.log

手順（resume 安全）:
  1. bootstrap の fetch 完了（summary 行）を待つ        <- 大きい download を exec 内で待たない
  2. setup stage（wheelhouse pin + torch pin 修理）
  3. import 検査（Colab image 固有の缺口が出たら prepare_env で補う）
  4. data,agent,graph,server,eval,package
  5. `[measure] DONE` / `[measure] FAILED` を log に出す（watchdog の完了マーカー）
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/content/repo")
WORK = Path("/content/work")
LOG = Path("/content/measure.log")
BOOT = Path("/content/bootstrap_all.log")
TIMEOUT_FETCH = int(__import__("os").environ.get("WAIT_FETCH_SECONDS", "5400"))
MEASURE_ONLY = {"MEASURE_ONLY_STAGES"}


def stamp(msg: str) -> None:
    line = f"[measure {time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.open("a", encoding="utf-8").write(line + "\n")


def step(argv: list[str], tag: str, timeout: int = 10800) -> int:
    stamp(f"run {tag}: {' '.join(argv)}")
    with LOG.open("a", encoding="utf-8") as fh:
        proc = subprocess.Popen(argv, cwd=str(REPO), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            fh.write(line)
            fh.flush()
        code = proc.wait(timeout=timeout)
    stamp(f"exit {tag} = {code}")
    return code


def wait_fetch() -> bool:
    deadline = time.monotonic() + TIMEOUT_FETCH
    while time.monotonic() < deadline:
        if BOOT.is_file():
            text = BOOT.read_text(encoding="utf-8", errors="replace")
            if '"wheelhouse"' in text:
                tail = [ln for ln in text.splitlines() if ln.startswith("[bootstrap]")]
                stamp(f"fetch 完了: {tail[-1] if tail else '?'}")
                return True
        time.sleep(20)
    stamp("fetch 完了を待てない（WAIT_FETCH_SECONDS 超過）")
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="colab12")
    ap.add_argument("--stages", default="data,agent,graph,server,eval,package")
    ap.add_argument("--skip-setup", action="store_true")
    args = ap.parse_args()

    LOG.write_text("", encoding="utf-8")
    stamp(f"start variant={args.variant} stages={args.stages} cwd={REPO}")

    if not wait_fetch():
        stamp("[measure] FAILED fetch 未完了")
        return 1

    py = sys.executable
    if not args.skip_setup:
        code = step([py, "src/pipeline/run_pipeline.py", "--variant", args.variant,
                     "--stages", "setup", "--strict",
                     "--data-dir", str(REPO / "data"), "--working-dir", str(WORK)], "setup")
        if code != 0:
            stamp("[measure] FAILED setup")
            return 1
        step([py, "src/pipeline/colab_prepare_env.py"], "prepare_env", timeout=1800)

    code = step([py, "src/pipeline/run_pipeline.py", "--variant", args.variant,
                 "--stages", args.stages, "--strict",
                 "--data-dir", str(REPO / "data"), "--working-dir", str(WORK)], "measure")
    stamp(f"[measure] {'DONE' if code == 0 else 'FAILED'} exit={code}")
    for pat in ("submission.zip", "predictions_*.json", "results/*/"):
        stamp(f"artifacts {pat}: " + json.dumps([str(p) for p in sorted(REPO.glob(pat))][:6]))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
