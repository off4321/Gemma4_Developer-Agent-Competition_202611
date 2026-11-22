"""vLLM 0.19.1 の api_server が要求する `model_hosting_container_standards` を直して import を通す。

実測（2026-11-21, Colab L4）:
  * 入れたまま: `.../model_hosting_container_standards/common/transforms/base_api_transform.py", line 5`
    で import 失敗（package 自体が壊れている / 他の pin と喧嘩）
  * 外すと: vllm 0.19.1 は `vllm/entrypoints/sagemaker/api_router.py` で
    `import model_hosting_container_standards.sagemaker` を **無条件**で読むので ModuleNotFoundError

→ 外すだけでは解決しない。version を合わせて入れ直す（`--no-deps` で transformers/torch を壊さない）。
まず完全な traceback を出してから判断する。

    colab exec -s <session> -f src/pipeline/colab_fix_mhcs2.py --timeout 900
"""
from __future__ import annotations

import subprocess


def show(cmd: str, limit: int = 1600) -> None:
    print(f"$ {cmd}")
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(((proc.stdout or "") + (proc.stderr or "")).strip()[:limit] or "(empty)", flush=True)


IMPORT = ("python3 -c \"import vllm.entrypoints.openai.api_server; print('api_server import OK')\" 2>&1 | tail -14")


def main() -> int:
    print("### 1) 今の状態")
    show(IMPORT)

    print("\n### 2) 公式 pin (wheelhouse) の version を --no-deps で入れる")
    show("ls /content/repo/data/wheelhouse | grep -i model_hosting || echo '  wheelhouse に無い'", 400)
    show("python3 -m pip install -q --no-deps model-hosting-container-standards 2>&1 | tail -3")
    show(IMPORT)

    print("\n### 3) それでも死ぬなら: sagemaker 側を隠して import できるか確認")
    show(IMPORT.replace("api_server; print", "api_server; print"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
