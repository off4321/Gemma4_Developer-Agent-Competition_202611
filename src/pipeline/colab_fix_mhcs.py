"""Colab image 同梱の `model_hosting_container_standards` が vLLM の api_server import を壊すので外す。

実測（2026-11-21, Colab L4 / torch pin 修理後）:
    deep import vllm.entrypoints.openai.api_server (after torch pin) ->
    File ".../model_hosting_container_standards/common/transforms/base_api_transform.py", line 5, in <module>
torch の ABI 問題は解消したが、Colab image が入れている **別 package** の import で死ぬ。
vLLM は optional integration としてこれを見るため、api_server の import が落ちる。
Kaggle image には無いものなので、Colab 側だけの処理。

    colab exec -s <session> -f src/pipeline/colab_fix_mhcs.py --timeout 600
"""
from __future__ import annotations

import subprocess


def show(cmd: str, limit: int = 1000) -> None:
    print(f"$ {cmd}")
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(((proc.stdout or "") + (proc.stderr or "")).strip()[:limit] or "(empty)", flush=True)


def main() -> int:
    show("pip list 2>/dev/null | grep -iE 'model.hosting|container.standards|vllm|^torch '")
    show("python3 -m pip uninstall -y model_hosting_container_standards 2>&1 | tail -3")
    show("python3 -c \"import vllm; print('vllm', vllm.__version__)\" 2>&1 | tail -3")
    show("python3 -c \"import vllm.entrypoints.openai.api_server as m; print('api_server import OK')\" 2>&1 | tail -6")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
