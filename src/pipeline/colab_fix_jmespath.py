"""MHCS(model_hosting_container_standards) が要求する `jmespath` を足して vLLM api_server の import を通す。

`colab log -s <session> -o nb.ipynb` の export から真因が出た（2026-11-21）:
    model_hosting_container_standards/common/transforms/base_api_transform.py", line 5, in <module>
        import jmespath
    ModuleNotFoundError: No module named 'jmespath'
Colab image は MHCS を dependency 込みで入れていない（Kaggle image には jmespath が有る）。
`--no-deps` で MHCS を入れ直す setup の後補として jmespath だけ足す。

    colab exec -s <session> -f src/pipeline/colab_fix_jmespath.py --timeout 600
"""
import subprocess

def show(cmd: str, limit: int = 900) -> None:
    print(f"$ {cmd}")
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(((p.stdout or "") + (p.stderr or "")).strip()[:limit] or "(empty)", flush=True)

show("python3 -m pip install -q jmespath 2>&1 | tail -2")
show("python3 -c \"import vllm.entrypoints.openai.api_server; print('api_server import OK')\" 2>&1 | tail -8")
