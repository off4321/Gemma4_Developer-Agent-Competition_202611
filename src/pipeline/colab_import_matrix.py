"""Colab VM で harness が要求する module が全部 import できるかを 1 回で洗い出す。

Kaggle kernel image は litellm / docker / jmespath を preinstall しているが、Colab image は無い
（実測 2026-11-21）。stage が 1 個ずつ壊れるより、ここで全缺口を出す方が安い。
`colab install` で足す -> この matrix で確認 -> stage を再 launch、の順で使う。

    colab exec -s <session> -f src/pipeline/colab_import_matrix.py --timeout 600
"""
import importlib
MODS = ["vllm", "vllm.entrypoints.openai.api_server", "litellm", "openai", "swegemma",
        "adk_submission", "adk_eval_core", "google.adk", "docker", "jmespath", "uvicorn",
        "fastapi", "transformers", "tokenizers", "safetensors", "torch", "yaml", "requests"]
missing = []
for name in MODS:
    try:
        m = importlib.import_module(name)
        ver = getattr(m, "__version__", "")
        print(f"  OK   {name:38s} {ver}")
    except Exception as exc:
        missing.append(name)
        print(f"  FAIL {name:38s} {type(exc).__name__}: {str(exc)[:80]}")
print("missing:", " ".join(missing) or "(none)")
