"""Colab image 固有の缺口を 1 回で直す（Kaggle kernel image には有って Colab に無いもの）。

実測（2026-11-21, Colab L4）で stage を壊した順:
  1. kaggle CLI が import 段階で死ぬ（`from kaggle.cli import main` 失敗）-> `pip install -U kaggle`
  2. `~/.kaggle/kaggle.json` が無いと `_authenticate_with_legacy_apikey` が "Missing key" -> credentials.json から作る
  3. `model_hosting_container_standards` が `jmespath` を要求 -> ModuleNotFoundError
  4. graph stage が `docker` SDK を import -> ModuleNotFoundError
  5. server stage が `litellm` を要求 -> ModuleNotFoundError
  6. EngineCore が `blake3` を要求 -> ModuleNotFoundError
  7. vllm の declares に `ninja / mcp / flashinfer-python==0.6.6 / flashinfer-cubin==0.6.6 /
     nvidia-cudnn-frontend<1.19 / nvidia-cutlass-dsl` が並ぶ（extra のマーカー付きは無視）
import できるものには手を出さない（冪等）。

    colab exec -s <session> -f src/pipeline/colab_prepare_env.py --timeout 900
"""
from __future__ import annotations

import importlib
import importlib.metadata as md
import os
import shutil
import subprocess
from pathlib import Path

KAGGLE_DIR = Path.home() / ".kaggle"
PINNED = {
    "flashinfer-python": "flashinfer-python==0.6.6",
    "flashinfer-cubin": "flashinfer-cubin==0.6.6",
    "nvidia-cudnn-frontend": "nvidia-cudnn-frontend<1.19.0,>=1.13.0",
    "nvidia-cutlass-dsl": "nvidia-cutlass-dsl>=4.4.0.dev1",
}
PLAIN = ["jmespath", "docker", "litellm", "blake3", "ninja", "mcp"]
CHECK = ["vllm", "vllm.entrypoints.openai.api_server", "litellm", "swegemma", "adk_submission",
         "adk_eval_core", "google.adk", "docker", "jmespath", "blake3", "transformers", "torch"]


def show(cmd: str, limit: int = 700) -> str:
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    out = ((p.stdout or "") + (p.stderr or "")).strip()
    print(out[:limit] or "(empty)", flush=True)
    return out


def fix_creds() -> None:
    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    a, b = KAGGLE_DIR / "credentials.json", KAGGLE_DIR / "kaggle.json"
    if a.is_file() and not b.is_file():
        shutil.copy2(a, b)
    elif b.is_file() and not a.is_file():
        shutil.copy2(b, a)
    for p in (a, b):
        if p.is_file():
            os.chmod(p, 0o600)
    print(f"[creds] {[p.name for p in sorted(KAGGLE_DIR.iterdir())]}", flush=True)


def fix_kaggle_cli() -> None:
    proc = subprocess.run(["python3", "-c", "import kaggle.cli"], capture_output=True, text=True)
    if proc.returncode == 0:
        print("[kaggle] cli import OK", flush=True)
        return
    print("[kaggle] cli 壊れている -> pip install -U kaggle", flush=True)
    show("python3 -m pip install -q -U kaggle 2>&1 | tail -3")
    show("kaggle --version 2>&1 | head -1")


def missing_modules() -> list[str]:
    out = []
    for name in CHECK:
        try:
            importlib.import_module(name)
        except Exception:
            out.append(name.split(".")[0])
    have = {d.metadata["Name"].lower().replace("_", "-") for d in md.distributions()}
    need = []
    for name in PLAIN:
        if name.lower().replace("_", "-") not in have or name.split(".")[0] in out:
            need.append(name)
    for mod, spec in PINNED.items():
        if mod not in have:
            need.append(spec)
    return need


def main() -> int:
    fix_creds()
    fix_kaggle_cli()
    need = missing_modules()
    print("[env] 足すもの:", need or "(無し)", flush=True)
    if need:
        show("python3 -m pip install -q " + " ".join(f'"{n}"' for n in need) + " 2>&1 | tail -4", 1200)
    print("[env] import 検査:", flush=True)
    bad = []
    for name in CHECK:
        try:
            m = importlib.import_module(name)
            print(f"  OK   {name:38s} {getattr(m, '__version__', '')}", flush=True)
        except Exception as exc:
            bad.append(name)
            print(f"  FAIL {name:38s} {type(exc).__name__}: {str(exc)[:70]}", flush=True)
    show("python3 -c \"import torch; print('torch', torch.__version__, torch.cuda.is_available(), "
         "torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')\" 2>&1 | tail -2")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
