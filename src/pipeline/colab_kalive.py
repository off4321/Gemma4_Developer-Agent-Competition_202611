"""Colab の assignment を生かし続ける heartbeat + job の生存報告（機械が読める行で出す）。

実測（2026-11-21 14:13 UTC）: 15 分ほど CLI が exec を叩かずに `colab status` だけ見ていると、
`/tun/m/assignments` が `{"assignments":[]}` を返して **VM が預けられた**（fetch が途中で死んだ）。
`colab status` は Colab 本体への contact にならない（runtime proxy を叩かない）。
kernel を 1 回走らせる exec が有効。

注意: `pgrep -af 'run_pipeline|colab_bootstrap'` は **自分自身の shell 行に一致する**ので、
watchdog の「動いてないなら起こす」判定が偽阳性になる（実測で fetch が再開されなかった）。
/proc を読んで自己除外する。

watchdog が読む行:
    FETCH_RUNNING=0|1
    PIPE_RUNNING=0|1
    MEASURE_RUNNING=0|1
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

PATTERNS = {"FETCH": ["colab_fetch.py", "colab_bootstrap.py"],
            "PIPE": ["run_pipeline.py"],
            "MEASURE": ["colab_measure.py"]}
SELF_WORDS = ("kaalive", "pgrep", "/bin/sh -c")


def find(patterns: list[str]) -> str:
    for entry in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            cmd = entry.read_text().replace("\0", " ").strip()
        except Exception:
            continue
        if not cmd or any(word in cmd for word in SELF_WORDS):
            continue
        if any(p in cmd for p in patterns):
            return cmd[:110]
    return ""


def main() -> int:
    print("KA", time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    for tag, patterns in PATTERNS.items():
        cmd = find(patterns)
        print(f"{tag}_RUNNING={1 if cmd else 0}{' CMD=' + cmd if cmd else ''}", flush=True)

    # Kaggle CLI がその VM で使えるか。ここで 0 だと fetch は tasks.jsonl 1 個で死ぬ
    # 実測 2026-11-21: session 作り直後に creds upload を忘れると fetch が tasks.jsonl 1 個目で死ぬ。
    kaggle_dir = Path.home() / ".kaggle"
    creds = any((kaggle_dir / n).is_file() for n in ("kaggle.json", "access_token", "credentials.json"))
    cli_ok = False
    try:
        proc = subprocess.run("timeout 25 kaggle --version 2>&1 | head -1", shell=True,
                              capture_output=True, text=True)
        cli_ok = "CLI" in (proc.stdout or "")
    except Exception:
        cli_ok = False
    print(f"CREDS={1 if creds else 0} KAGGLE_OK={1 if cli_ok else 0}", flush=True)

    for log in sorted(Path("/content").glob("*.log")):
        print(f"LOG {log.name} {log.stat().st_size}", flush=True)
    usage = shutil.disk_usage("/content")
    print(f"DISK used={usage.used / 1e9:.2f}GB free={usage.free / 1e9:.2f}GB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
