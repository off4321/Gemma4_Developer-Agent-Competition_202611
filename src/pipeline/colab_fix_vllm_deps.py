"""vLLM 0.19.1 が要求するのに Colab image に無い package を、pin のまままとめて入れる。

実測（2026-11-21, Colab L4）: server stage で EngineCore が
    ModuleNotFoundError: No module named 'blake3'
で死ぬ。Kaggle kernel image には preinstall されている（blake3 / litellm / docker / jmespath）。
1 個ずつ出すより、`importlib.metadata.requires("vllm")` を読んで缺口を一度に出す。
pin 付きで入れるので transformers / torch の組み合わせを壊しにくい。

    colab exec -s <session> -f src/pipeline/colab_fix_vllm_deps.py --timeout 900
"""
import importlib.metadata as md
import subprocess

EXTRA = ["blake3", "litellm", "docker", "jmespath"]


def show(cmd: str, limit: int = 900) -> str:
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    out = ((p.stdout or "") + (p.stderr or "")).strip()
    print(out[:limit] or "(empty)", flush=True)
    return out


def main() -> int:
    try:
        reqs = md.requires("vllm") or []
    except Exception as exc:
        print(f"requires(vllm) 読めず: {exc}")
        reqs = []
    have = {d.metadata["Name"].lower().replace("_", "-") for d in md.distributions()}
    missing = []
    for r in reqs:
        if ";" in r:            # extra == "..." のやつは測定に要らない（pip もマーカーを嫌う）
            continue
        name = r.split("[")[0].split("=")[0].split(">")[0].split("<")[0].split("!")[0].strip()
        if name and name.lower().replace("_", "-") not in have and "torch" not in name.lower():
            missing.append(r)
    for r in EXTRA:
        if r.lower() not in have:
            missing.append(r)
    print("缺口:", missing or "(無し)")
    if not missing:
        return 0
    show("python3 -m pip install -q " + " ".join(f'"{m}"' for m in missing) + " 2>&1 | tail -4")
    show('python3 -c "import vllm.entrypoints.openai.api_server, blake3, litellm; print(\'import OK\')" 2>&1 | tail -4')
    show("python3 -c \"import torch; print('torch', torch.__version__, torch.cuda.is_available())\" 2>&1 | tail -2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
