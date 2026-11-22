"""Colab VM の model 重みが stage の見る場所に揃っているか確かめるだけの検査。

期待する場所: `<repo>/models/google/gemma-4/other/gemma-4-e4b-it-qat-w4a16-ct/2/`
期待する物: `model.safetensors`（実測 size 11,513,861,828 byte）+ config.json / tokenizer 類。
Kaggle の model download が一部のファイルだけ落とす / 別名で置くケースをここで捕まえる。

    colab exec -s <session> -f src/pipeline/colab_check_model.py
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

REPO = os.environ.get("REPO_DIR", "/content/repo")
MODEL_NAME = "gemma-4-e4b-it-qat-w4a16-ct"
EXPECTED = Path(REPO) / "models" / "google" / "gemma-4" / "other" / MODEL_NAME / "2"
EXPECTED_BYTES = 11513861828


def sh(cmd: str, limit: int = 1400) -> str:
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return ((proc.stdout or "") + (proc.stderr or "")).strip()[:limit] or "(empty)"


def main() -> int:
    print(f"[model] 期待 dir: {EXPECTED}")
    if EXPECTED.is_dir():
        rows = []
        for p in sorted(EXPECTED.iterdir()):
            if p.is_file():
                rows.append(f"  {p.stat().st_size:>14,}  {p.name}")
        print("\n".join(rows) or "  (空)")
        weights = list(EXPECTED.glob("*.safetensors"))
        total = sum(p.stat().st_size for p in weights)
        print(f"[model] safetensors: {len(weights)} 個 / {total:,} byte "
              f"(期待 {EXPECTED_BYTES:,} に対して {'一致' if abs(total - EXPECTED_BYTES) < 1_000_000 else '足りない'})")
        cfg = EXPECTED / "config.json"
        if cfg.is_file():
            data = json.loads(cfg.read_text())
            q = data.get("quantization_config") or {}
            print(f"[model] model_type={data.get('model_type')} architectures={data.get('architectures')} "
                  f"quant format={q.get('format')} num_bits={(q.get('quantized_weight') or {}).get('num_bits')}")
    else:
        print("[model] 期待 dir が無い")
    print("--- models ツリー全体 ---")
    print(sh(f"find {REPO}/models -maxdepth 6 -type d | head -12"))
    print(sh(f"find {REPO}/models -name '*.safetensors' -printf '%s %p\\n' | sort -rn | head -6"))
    print("--- bootstrap log tail ---")
    for log in sorted(Path("/content").glob("bootstrap_*.log")):
        print(f"{log.name}: " + sh(f"tail -4 {log}", 700))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
