"""Colab VM の環境事実を出すだけのスクリプト（`colab exec -s <session> -f` で実行する）。

何を確かめるか: GPU 型号 / compute capability（sm89 なら bfloat16 と Triton attention が通る）、
Python version（wheelhouse の wheel tag と合うか）、disk（model 11.5 GB + wheelhouse 867 MB が入るか）、
base image に入っている torch / triton / vllm の version（setup の [heal] がどれだけ働くか）。
"""
import platform
import subprocess
import sys


def sh(cmd: str, limit: int = 1200) -> None:
    print(f"$ {cmd}")
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    out = (proc.stdout or proc.stderr or "").strip()
    print(out[:limit] or "(empty)")


sh("nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader")
print(f"python {sys.version.split()[0]} impl={platform.python_implementation()} machine={platform.machine()}")
sh("df -h /content | tail -2")
sh("free -g | head -2")
sh("pip list 2>/dev/null | grep -iE '^(torch|torchvision|triton|transformers|vllm|flashinfer|"
   "nvidia-cuda-runtime|huggingface-hub|tokenizers|safetensors) ' || echo '(none of them installed)'")
sh("python3 -c \"import torch;print('torch', torch.__version__, 'cuda', torch.version.cuda, "
   "'devs', torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')\"")
sh("ls /content")
