"""Colab VM の stage log から「torch / vLLM の载せ替え判断」だけを拾う検査。

`vllm/_C.abi3.so: undefined symbol: _ZN3c104cuda29c10_cuda_check_implementation...` は
**vLLM がビルドされた torch と今の torch が違う**という意味（ABI 不一致）。
setup が torch を vLLM の pin に合わせに行くので、その判断行を探す。

    colab exec -s <session> -f src/pipeline/colab_log_grep.py --env GREP="torch pin"
"""
from __future__ import annotations

import os
import re
from pathlib import Path

PATTERNS = os.environ.get("GREP", "torch pin|install_torch|measure-only|reassert|heal|deviation|abi3").split("|")
DEFAULT = ["torch", "vllm", "triton", "transformers", "tokenizers", "huggingface"]


def main() -> int:
    logs = sorted(Path("/content").glob("stage_*.log")) + sorted(Path("/content").glob("bootstrap_*.log"))
    if not logs:
        print("log が無い")
        return 1
    rx = re.compile("|".join(PATTERNS), re.IGNORECASE)
    for log in logs:
        print(f"=== {log.name} ===")
        hits = []
        for i, line in enumerate(log.read_text(errors="replace").splitlines(), 1):
            if rx.search(line):
                hits.append(f"{i:5d}  {line[:220]}")
        print("\n".join(hits[-25:]) or "  (該当行なし)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
