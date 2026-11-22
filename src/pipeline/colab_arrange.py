"""Colab VM の competition data を stage が見る形に並べ替えるだけのスクリプト。

実測（2026-11-21）: `kaggle competitions download -c gemma-4-developer-agent -p data` は
`data/gemma-4-developer-agent/` の中に全部展開する。stage が expects するのは `data/` 直下:

    data/tasks.jsonl  data/HARNESS_README.md  data/graphs/  data/embeddings/
    data/snapshots/   data/wheels/            data/docker/  data/sandbox/

`snapshots/<instance_id>.tgz` は **task ごとの repo の実体**（fastapi_15661.tgz 319 MB 等）。
これが無いと local の Phase 2（patch apply + pytest）は repo 実体無しになる。zip（21.9 GB）は
展開済みなら消して disk を空ける。

    colab exec -s <session> -f src/pipeline/colab_arrange.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

DATA = Path(os.environ.get("DATA_DIR", "/content/repo/data"))


def counts(d: Path) -> str:
    if not d.is_dir():
        return "MISSING"
    files = [p for p in d.iterdir() if p.is_file()]
    return f"{len(files)} files / {sum(f.stat().st_size for f in files) / 1e9:.2f} GB"


def main() -> int:
    inner = DATA / "gemma-4-developer-agent"
    if inner.is_dir():
        print(f"[arrange] {inner} の中身を {DATA} に上げる")
        for src in sorted(inner.iterdir()):
            dst = DATA / src.name
            if dst.exists():
                print(f"[arrange] skip (既に有る): {dst.name}")
                continue
            shutil.move(str(src), str(dst))
        shutil.rmtree(inner, ignore_errors=True)
        zip_path = DATA / "gemma-4-developer-agent.zip"
        if zip_path.is_file():
            print(f"[arrange] zip を捨てる: {zip_path.name} ({zip_path.stat().st_size / 1e9:.1f} GB)")
            zip_path.unlink()

    print("[arrange] 最終 layout:")
    for name in ("tasks.jsonl", "HARNESS_README.md", "graphs", "embeddings", "snapshots",
                 "wheels", "docker", "sandbox", "sample_submission"):
        p = DATA / name
        if p.is_dir():
            print(f"  DIR  {name:18s} {counts(p)}")
        else:
            size = p.stat().st_size if p.is_file() else 0
            print(f"  {'FILE' if size else 'MISS'} {name:18s} {size / 1e6:.2f} MB")
    out = subprocess.run(["df", "-h", "/content"], capture_output=True, text=True).stdout
    print(out.strip().splitlines()[-1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
