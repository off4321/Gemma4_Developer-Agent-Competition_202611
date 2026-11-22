"""Kaggle model download が落とした tar.gz を展開して、stage が見る場所に重みを置くだけのスクリプト。

実測（2026-11-21, Colab L4）: `kaggle models instances versions download ... -p models/_raw` は
`gemma-4.tar.gz`（6.8 GB）1 個だけを落とし、`model.safetensors` は展開後に姿を見せる。
裏 process にすると colab の websocket が "Connection was lost." で切れることがあるので、
**この展開は同期実行**（1-2 分）して `--timeout` を大きくする。

    colab exec -s <session> -f src/pipeline/colab_untar_model.py --timeout 900
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

REPO = Path(os.environ.get("REPO_DIR", "/content/repo"))
MODEL_NAME = "gemma-4-e4b-it-qat-w4a16-ct"
EXPECTED = REPO / "models" / "google" / "gemma-4" / "other" / MODEL_NAME / "2"
RAW = REPO / "models" / "_raw"


def run(argv: list[str]) -> None:
    print("+ " + " ".join(argv), flush=True)
    proc = subprocess.run(argv, capture_output=True, text=True)
    if proc.returncode != 0:
        print(((proc.stdout or "") + (proc.stderr or "")).strip()[-800:])
        raise SystemExit(f"失敗 (exit {proc.returncode}): {' '.join(argv)}")


def main() -> int:
    EXPECTED.mkdir(parents=True, exist_ok=True)
    if (EXPECTED / "model.safetensors").is_file():
        print("[untar] 既に配置済み。飛ばす")
    else:
        tars = sorted(RAW.rglob("*.tar.gz")) if RAW.is_dir() else []
        if not tars:
            raise SystemExit(f"tar.gz が見つからない: {RAW}")
        for tar in tars:
            print(f"[untar] {tar} ({tar.stat().st_size / 1e9:.2f} GB)", flush=True)
            run(["tar", "xzf", str(tar), "-C", str(tar.parent)])
        found = list(RAW.rglob("model.safetensors"))
        if not found:
            print("[untar] 展開後に model.safetensors が見つからない。ツリー:")
            print("\n".join(str(p) for p in sorted(RAW.rglob("*"))[:20]))
            return 1
        src_dir = found[0].parent
        print(f"[untar] {src_dir} -> {EXPECTED}", flush=True)
        for p in sorted(src_dir.iterdir()):
            if p.is_file():
                shutil.move(str(p), str(EXPECTED / p.name))
        shutil.rmtree(RAW, ignore_errors=True)

    rows = [f"  {p.stat().st_size:>14,}  {p.name}" for p in sorted(EXPECTED.iterdir()) if p.is_file()]
    print("\n".join(rows))
    total = sum(p.stat().st_size for p in EXPECTED.glob("*.safetensors"))
    print(f"[untar] safetensors 合計 {total:,} byte（期待 11,513,861,828）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
