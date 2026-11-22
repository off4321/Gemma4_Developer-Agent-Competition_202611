"""Colab image に無い python `docker` package を足す（graph stage が import で死ぬ）。

実測（2026-11-21, Colab L4）: `[graph] [ERROR] ModuleNotFoundError: No module named 'docker'`
Kaggle kernel image には有る（swegemma / adk_submission の graph 側が docker SDK を import する）。
**daemon は要らない**（sandbox=subprocess）ので import が通ればよい。

    colab exec -s <session> -f src/pipeline/colab_fix_docker_mod.py --timeout 600
"""
import subprocess
def show(cmd: str, limit: int = 700) -> None:
    print(f"$ {cmd}")
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(((p.stdout or "") + (p.stderr or "")).strip()[:limit] or "(empty)", flush=True)
show("python3 -m pip install -q docker 2>&1 | tail -2")
show("python3 -c \"import docker; print('docker sdk OK', docker.__version__)\" 2>&1 | tail -3")
