"""
ステージ `setup` = ノートブック Cell 1。

仕事は 3 つ、順番も固定:
  1. vllm / litellm / transformers を import する **前に**、オフライン推論用の環境変数を立てる
  2. competition の wheelhouse を、**今の interpreter で通る wheel だけ**選んで offline install する
  3. install 後に import できるかを素直に出す（後段の skip 原因をここで切り分ける）

なぜ「全部入れる」ではダメなのか（実測で当たった）:
Kaggle base image の Python は 3.13 なのに、wheelhouse には `cp312-cp312` の wheel が混在している。
`pip install <cp312 wheel>` は即失敗する:

    ERROR: apache_tvm_ffi-0.1.13.post3-cp312-cp312-...whl is not a supported wheel on this platform.

だから wheel 名の tag を読んで選別する。規則は wheel spec（PEP 427）そのまま:
  * `py3` / `py37` / `py2.py3`     -> pure python。常に OK
  * `cp38-abi3`                    -> abi3 なので **3.8 以上なら通る**（3.13 でも OK。vllm がこれ）
  * `cp312-cp312`                  -> 3.12 専用。3.13 では skip
  * platform が win / macosx / aarch64 / i686 -> skip（Kaggle は linux x86_64）

torch / triton / nvidia_* は **入れ替えない**。base image が CUDA に合わせて組んだ stack を
壊すと GPU stage が总て壊れる（wheelhouse の torch は cp312 専用なのでそもそも通らない）。

install は `--no-deps`。依存が足りないときは `pip_fallback: true` なら **import エラーが出た
名前だけ** PyPI から補う（測定用 kernel は internet 可。公式採点側は internet 禁止なので、
この補完は採点の仕組みではなく自分の run のためのもの）。

初学者向けの注意: wheelhouse があるのに import 失敗するなら、多くの場合ここが原因である。
ログの `skip` 行と `heal` 行を見ること。
"""

from __future__ import annotations

import glob
import importlib
import os
import re
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Any

from pipeline.common import configure_logging, has_module, log, set_runtime_env

CUTLASS_PTH_PATTERNS = (
    "/usr/lib/python*/dist-packages/*cutlass*.pth",
    "/usr/local/lib/python*/dist-packages/*cutlass*.pth",
    "/usr/local/lib/python*/site-packages/*cutlass*.pth",
)

# base image が CUDA 向けに組んだ stack。wheelhouse で上書きしない（壊れる）。
PROTECTED_DIST_PREFIXES = ("torch", "torchaudio", "torchvision", "triton", "nvidia", "cuda",
                           "flashinfer", "xformers", "nvidia-")
# platform tag から除外する語（Kaggle は linux x86_64）
BAD_PLATFORM_TOKENS = ("win", "macosx", "aarch64", "armv", "i686", "i386", "universal2",
                       "android", "ppc", "s390", "riscv", "power", "aarch")
HEAL_TARGETS = ("vllm", "vllm.config", "vllm.entrypoints.openai.api_server")
HEAL_MAX_ATTEMPTS = 10
# vllm を依存込みで入れると pip が torch / transformers を **自分の pin に合わせて下げる**。
# 実測: vllm==0.19.1 を入れたら transformers 5.13.1 -> 4.57.6 になり、vLLM がこう落ちた:
#   The checkpoint you are trying to load has model type `gemma4` but Transformers does not recognize
# i.e. gemma4 を知るのは transformers 5.x だけである。PyPI の vllm 0.19.1 の requires は
#   torch==2.10.0 / torchvision==0.25.0 / torchaudio==2.10.0 / transformers>=4.56（5.0〜5.5 除外）
#   そして **huggingface_hub の pin は無い**。
# transformers 5.13.1 は huggingface-hub>=1.5,<2.0 を要求する。だから hf_hub を 0.36 に下げると
# transformers が `cannot import name 'is_offline_mode'` で落ちる（実際に落ちた）。
# 正しい repair は「vllm 一式を入れ直す」ことではなく **torch 三兄弟だけ vllm の pin に合わせる**こと。
TORCH_TRIO = ("torch", "torchvision", "torchaudio")

CP_MINOR = sys.version_info[1]     # 3.13 なら 13
CURRENT_KEY = sys.version_info[0] * 100 + sys.version_info[1]   # 3.13 なら 313（tag と同じ尺度で比べる）


def _tag_key(digits: str) -> int | None:
    """wheel tag の数字を (major*100+minor) に直す。cp312 -> 312、cp38 -> 308、py37 -> 307。
    「38」は 3.8 であり 338 ではない。これを怠ると「38 > 13」様な間違った比べ方をして
    abi3 wheel を落とす（実際に落とした）。major は 1 桁、残りが minor だ。"""
    if not digits.isdigit() or len(digits) < 2:
        return None
    major, minor = int(digits[0]), int(digits[1:])
    return major * 100 + minor


def _python_tag_ok(python_tag: str, abi_tag: str) -> bool:
    """python tag が今の interpreter で通るか。abi3 は「その cp 以上なら通る」なので注意。"""
    for tag in python_tag.split("."):
        if tag == "py3":
            return True
        if tag.startswith("cp"):
            key = _tag_key(tag[2:])
            if key is None:
                continue
            if abi_tag == "abi3":
                if key <= CURRENT_KEY:          # cp38-abi3 は 3.13 でも通る
                    return True
            elif abi_tag == "none":
                return True
            elif key == CURRENT_KEY:            # cp312-cp312 は 3.12 でだけ通る
                return True
        elif tag.startswith("py"):
            key = _tag_key(tag[2:])
            if key is not None and key <= CURRENT_KEY:   # py37 -> 3.7 以上なら OK
                return True
    return False


def _wheel_parts(filename: str) -> tuple[str, str, str] | None:
    """wheel 名から (python tag, abi tag, platform tag) を取り出す（PEP 427 の並び）。"""
    if not filename.endswith(".whl"):
        return None
    parts = filename[: -len(".whl")].split("-")
    if len(parts) < 5:
        return None
    return parts[-3], parts[-2], parts[-1]


def _platform_tag_ok(platform_tag: str) -> bool:
    for tag in platform_tag.split("."):
        if tag in {"any", "linux_x86_64"}:
            return True
        if tag.startswith(("manylinux", "musllinux")) and tag.endswith("_x86_64"):
            if not any(bad in tag for bad in BAD_PLATFORM_TOKENS):
                return True
    return False


def wheel_verdict(filename: str) -> tuple[bool, str]:
    """(採用か, 理由)。理由ログを出すことで「なぜ vllm が入った/入らない」を再現可能にする。"""
    parts = _wheel_parts(filename)
    if parts is None:
        return False, "wheel 名が解析できない"
    python_tag, abi_tag, platform_tag = parts
    if not _platform_tag_ok(platform_tag):
        return False, f"platform 外 ({platform_tag})"
    if not _python_tag_ok(python_tag, abi_tag):
        return False, f"interpreter 外 (今の env {CURRENT_KEY} に対して {python_tag}-{abi_tag})"
    return True, f"tag OK ({python_tag}-{abi_tag})"


def dist_of_wheel(filename: str) -> str:
    """wheel 名から distribution 名（pip が使う名前）を取り出す。"""
    parts = filename[: -len(".whl")].split("-")
    return parts[0].replace("_", "-").lower() if parts else ""


def is_protected(dist: str) -> bool:
    """base image が CUDA に合わせて組んだ stack か。PyPI 版でも **入れてはいけない**
    （PyPI の torch は CPU build で、GPU が总て動かなくなる。実際にやりかけた）。"""
    return any(dist.startswith(p.rstrip("-")) for p in PROTECTED_DIST_PREFIXES)


def version_of_wheel(filename: str) -> str:
    """wheel 名から version を取り出す。`+cu128` の様な local version は PyPI に無いので削る。"""
    parts = filename[: -len(".whl")].split("-")
    if len(parts) < 2:
        return ""
    return parts[1].split("+")[0]


def module_of_wheel(filename: str) -> str:
    """distribution 名から import 名を推定する（google-adk -> google.adk の例外だけ）。"""
    dist = dist_of_wheel(filename)
    if dist in {"google-adk", "google_adk"}:
        return "google.adk"
    return dist.replace("-", "_")


def drop_cutlass_pth_hooks() -> int:
    """壊れた cutlass の `.pth` フックを消す。`.pth` は interpreter 起動時に import されるので、
    これが残っていると vLLM が起動前に壊れる。"""
    removed = 0
    for pattern in CUTLASS_PTH_PATTERNS:
        for pth in glob.glob(pattern):
            try:
                Path(pth).unlink()
                removed += 1
            except OSError:
                pass
    return removed


def stage_wheelhouse(wheelhouse: Path, dest: Path = Path("/tmp/wheelhouse")) -> list[str]:
    """wheel を書き込み可能な dir に写す。Kaggle の dataset アップロードが消す
    PEP 440 の `+cu128`（local version）を復元する事もここの仕事（cutlass の .pth は除ける）。"""
    dest.mkdir(parents=True, exist_ok=True)
    for wheel in sorted(wheelhouse.glob("*.whl")):
        if "cutlass" in wheel.name.lower():
            continue
        name = wheel.name
        if "cu128" in name and "+" not in name:
            name = name.replace("cu128", "+cu128")
        target = dest / name
        if target.exists():
            continue
        try:
            target.symlink_to(wheel)
        except OSError:
            shutil.copyfile(wheel, target)
    return sorted(str(w) for w in dest.glob("*.whl"))


def plan_wheels(wheels: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
    """install する wheel と、skip する wheel を (名前, 理由) で分ける。

    選別の方針:
      * tag が今の interpreter に合わない wheel            -> skip（入れても pip が Reject する）
      * torch / triton / nvidia_*                           -> skip（base image の CUDA stack を守る。ここだけ heal で直す）
      * pure python (py3 / abi none)                        -> install（harness 本体はこれ）
      * binary wheel で base image に同じ version が有る     -> skip（触らない）
      * binary wheel で base image の version が **違う**    -> install（wheelhouse の pin を公式の組み合わせとして優先）
    「base に有れば何でも skip」は危ない。実測で落ちた例:
      wheelhouse = transformers 5.13.1 + tokenizers 0.22.2、base image = tokenizers 0.23.3
      -> transformers が `tokenizers>=0.22.0,<=0.23.0 is required ... found 0.23.3` で落ちた。
    """
    install: list[str] = []
    skip: list[tuple[str, str]] = []
    for path in wheels:
        name = Path(path).name
        dist = dist_of_wheel(name)
        ok, reason = wheel_verdict(name)
        if not ok:
            skip.append((name, reason))
            continue
        if is_protected(dist):
            skip.append((name, "base image の torch/CUDA stack を守るため入れない"))
            continue
        module = module_of_wheel(name)
        pure_python = "-none-any" in name or name.split("-")[-3].startswith("py")
        if not pure_python and has_module(module):
            have, want = pkg_version(dist), version_of_wheel(name)
            if have and want and have.split("+")[0] == want:
                skip.append((name, f"{module} {have} は base image に同じ version が有る"))
                continue
            log("setup", f"[wheel] {dist}: wheelhouse={want} / base image={have or '-'} -> "
                         "wheelhouse の pin を入れる（公式 env の組み合わせに合わせる）")
            install.append(path)
            continue
        install.append(path)
    return install, skip


def pip_install(wheels: list[str]) -> tuple[int, str]:
    """1 発の offline install（--no-deps）。失敗時は stderr をそのまま返す（hidden にしない）。"""
    proc = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", *wheels],
                          capture_output=True, text=True)
    return proc.returncode, (proc.stderr or proc.stdout or "").strip()


def import_state(name: str) -> str:
    """import して version を返す。失敗時は例外名だけ短く出す（ログが読みやすいように）。"""
    try:
        module = importlib.import_module(name)
        return f"OK version={getattr(module, '__version__', '?')}"
    except BaseException as exc:  # noqa: BLE001  (import 時の SystemExit/初期化失敗も見る)
        return f"FAIL {type(exc).__name__}: {str(exc)[:180]}"


def heal_missing_dep(module: str, pin: str | None = None) -> bool:
    """internet が通る測定用 kernel で、足りない依存だけを PyPI から補う。
    公式採点側では internet が閉じているので、これは自分の run を動かすための処置だと書く。

    `pin`（例 "msgspec==0.21.1"）を渡すと **wheelhouse の pinned version のまま** cp313 版を入れる。
    latest を拾うと vllm の pin と喧喧するので、pin が分かるものは必ず pin 側を使う。"""
    spec = pin or module.replace("_", "-")
    log("setup", f"[heal] pip install --no-deps --only-binary=:all: {spec}（internet 使用: 測定 kernel 限り）")
    proc = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                           "--only-binary=:all:", spec], capture_output=True, text=True)
    if proc.returncode:
        log("setup", f"[heal] {spec} install 失敗 rc={proc.returncode}: "
                     f"{(proc.stderr or '')[:300]}")
        return False
    importlib.invalidate_caches()
    return True


def replace_incompatible(skipped: list[tuple[str, str]]) -> list[str]:
    """interpreter tag 違いで skip した wheel を、**同じ version の cp313 版**に置き換える。

    wheelhouse は公式 env（cp312）向けに pin されている。cp313 の Kaggle image では
    `pip install <cp312 wheel>` が Reject されるので、名前で PyPI に同じ version を探す。
    これで「公式 env と同じ version 構成」を保ったまま vllm が動くようになる。"""
    specs: list[str] = []
    for name, reason in skipped:
        if "interpreter 外" not in reason:
            continue                      # base 提供 / platform 外 / protected は補わない
        dist, ver = dist_of_wheel(name), version_of_wheel(name)
        if not dist or not ver:
            continue
        if is_protected(dist):
            continue                      # torch/triton/nvidia_* は PyPI 版でも入れない
        specs.append(f"{dist}=={ver}")
    if not specs:
        return []
    log("setup", f"[heal] tag 違いで飛ばした {len(specs)} 個を PyPI の same-version 版で補う: "
                 f"{', '.join(specs)}")
    proc = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                           "--only-binary=:all:", *specs], capture_output=True, text=True)
    if proc.returncode == 0:
        importlib.invalidate_caches()
        return specs
    # まとめて 1 発が失敗したら（1 個でも PyPI に cp313 版が無いと全体落ちる）、1 個ずつ落とす
    log("setup", f"[heal] まとめ install 失敗 rc={proc.returncode} → 1 個ずつ試す")
    done: list[str] = []
    for spec in specs:
        if heal_missing_dep("", pin=spec):
            done.append(spec)
    importlib.invalidate_caches()
    if done:
        log("setup", f"[heal] same-version 補完成功: {', '.join(done)}")
    return done


def nvidia_lib_dirs() -> list[str]:
    """pip が入れる site-packages/nvidia/<pkg>/lib と torch/lib の dir（そこに libcudart.so.12 が居る）。"""
    purelib = Path(sysconfig.get_paths()["purelib"])
    dirs = [str(p) for p in sorted((purelib / "nvidia").glob("*/lib")) if p.is_dir()]
    torch_lib = purelib / "torch" / "lib"
    if torch_lib.is_dir():
        dirs.append(str(torch_lib))
    return dirs


def apply_ld_path() -> None:
    """`nvidia_*` の lib dir を LD_LIBRARY_PATH に足す。
    base image の LD_LIBRARY_PATH は /usr/local/cuda-13 だけを見ており、torch は自前の loader で
    site-packages/nvidia を引きに行く。一方 **vllm の .so は DT_NEEDED で素直に linker を探す**ので、
    ここに足さないと `libcudart.so.12: cannot open shared object file` になる（実際の事故）。

    注意: 同じ process 内の dynamic loader は起動時の env を覚えている。だからここを変えても
    今の process の import は直らない。直るのは **次に起動する child process**（= vllm server）である。"""
    dirs = nvidia_lib_dirs()
    if not dirs:
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    known = set(filter(None, current.split(":")))
    added = [d for d in dirs if d not in known]
    if not added:
        return
    os.environ["LD_LIBRARY_PATH"] = ":".join(([current] if current else []) + added)
    log("setup", f"LD_LIBRARY_PATH に child 向け lib dir を追加: {', '.join(added)}")


def preload_cuda_libs(want_prefixes: tuple[str, ...] = ("libcudart.so.12", "libcublas.so.12",
                                                        "libcublasLt.so.12", "libcudnn.so.9",
                                                        "libnccl.so.2", "libcufft.so.12",
                                                        "libcurand.so.12", "libcusparse.so.12",
                                                        "libcusolver.so.12", "libnvrtc", "libnvjitlink")
                       ) -> list[str]:
    """今の process に cu12 の lib を先に RTLD_GLOBAL で読み込ませる（loader の env は
    起動後に変えても効かないため、import 前のこの手当てが要る）。"""
    import ctypes

    loaded: list[str] = []
    for lib in sorted(Path(sysconfig.get_paths()["purelib"]).glob("nvidia/*/lib/lib*")):
        if not any(lib.name.startswith(w.split(".so")[0]) for w in want_prefixes):
            continue
        try:
            ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
            loaded.append(lib.name)
        except OSError:
            pass
    if loaded:
        log("setup", f"preload CUDA libs in-process: {', '.join(loaded[:10])}")
    return loaded


def deep_probe(target: str) -> tuple[bool, str]:
    """`import vllm` は通っても `import vllm.config` で落ちる（実際の事故）。
    server が走らせる entry point を **この process で import して** 完全な traceback を見る。"""
    try:
        importlib.import_module(target)
        return True, "OK"
    except BaseException as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {str(exc)[:300]}"


def deep_probe_child(target: str) -> tuple[bool, str]:
    """同じ import を **child process で** 試す。これは vllm server が受ける条件そのものなので、
    採点側が実際に通るかどうかの判定はこっちで行う（今の process だけ失敗しても止まらない）。"""
    proc = subprocess.run([sys.executable, "-c", f"import {target}"],
                          capture_output=True, text=True, timeout=600)
    if proc.returncode == 0:
        return True, "OK"
    tail = (proc.stderr or "").strip().splitlines()
    return False, " | ".join(line.strip() for line in tail[-3:])[:400] or f"rc={proc.returncode}"
    try:
        importlib.import_module(target)
        return True, "OK"
    except BaseException as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {str(exc)[:300]}"


def heal_loop(targets: tuple[str, ...] = HEAL_TARGETS, pins: dict[str, str] | None = None) -> None:
    """target が import できるまで、import error が出た名前だけ補い続ける（最大 HEAL_MAX_ATTEMPTS）。
    vllm の様に依存が複数有る場合、1 発では直らないので「エラー名 -> install -> 再 import」で詰める。"""
    pins = pins or {}
    for target in targets:
        attempts = 0
        while attempts < HEAL_MAX_ATTEMPTS:
            ok, message = deep_probe(target)
            if ok:
                log("setup", f"[heal] {target} import OK (補った回数 {attempts})")
                break
            exc = None
            try:
                importlib.import_module(target)
            except ModuleNotFoundError as error:   # 名前が特定できる場合だけ補う
                exc = error
            except BaseException as error:  # noqa: BLE001
                log("setup", f"[heal] {target} は別の理由で失敗: {type(error).__name__}: "
                             f"{str(error)[:240]}")
                break
            missing = getattr(exc, "name", None)
            if not missing:
                log("setup", f"[heal] {target}: 足りない module が特定できない ({message})")
                break
            attempts += 1
            top = missing.split(".")[0]
            if not heal_missing_dep(missing, pin=pins.get(top)):
                break
            importlib.invalidate_caches()
        else:
            log("setup", f"[heal] {target}: 補完を {HEAL_MAX_ATTEMPTS} 回で打ち切った")


def log_environment() -> None:
    """setup の失敗はほぼ「base image の CUDA stack と wheelhouse の相性」なので、
    その事実を先に全部 logs に出す。推測で GPU run を重ねないための関数である。"""
    log("setup", f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '-')}")
    cuda_dirs = sorted(str(p) for p in Path("/").glob("usr/local/cuda*"))
    log("setup", f"/usr/local/cuda* = {cuda_dirs or '無い'}")
    for pattern in ("lib/libcudart.so*", "libcudart.so*"):
        hits: list[str] = []
        for base in cuda_dirs:
            hits += [str(p) for p in Path(base).glob(pattern)]
        if hits:
            log("setup", f"  libcudart candidates ({pattern}): {', '.join(hits[:6])}")
    proc = subprocess.run([sys.executable, "-m", "pip", "list", "--disable-pip-version-check"],
                          capture_output=True, text=True)
    rows = [line.strip() for line in (proc.stdout or "").splitlines()
            if line.split(" ")[0].lower().startswith(("torch", "nvidia", "vllm", "triton",
                                                      "transformers", "flashinfer", "xformers",
                                                      "huggingface", "tokenizers", "safetensors",
                                                      "compressed", "litellm", "google"))]
    log("setup", "pip list (torch/nvidia/vllm/triton/transformers):")
    for row in rows:
        log("setup", f"  {row}")
    try:
        import torch

        log("setup", f"torch {torch.__version__} cuda_built={getattr(torch.version, 'cuda', None)} "
                     f"available={torch.cuda.is_available()} devices={torch.cuda.device_count()}")
    except BaseException as exc:  # noqa: BLE001
        log("setup", f"torch import 失敗: {type(exc).__name__}: {str(exc)[:200]}")
    nvidia_dir = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    if nvidia_dir.is_dir():
        libs = sorted(p.name for p in nvidia_dir.glob("*/lib/lib*.so*"))
        log("setup", f"site-packages/nvidia libs: {', '.join(libs[:24]) or '無し'}")
    log("setup", f"LD paths from nvidia libs: {'yes' if nvidia_dir.is_dir() else 'no'}")


def vllm_version_from(wheels: list[str]) -> str:
    """wheelhouse が pin している vllm の version（これを PyPI から依存込みで入れ直すときの指定にする）。"""
    for wheel in wheels:
        name = Path(wheel).name
        if dist_of_wheel(name) == "vllm":
            return version_of_wheel(name)
    return ""


def pkg_version(name: str) -> str:
    """今入っている package の version（import せずに読む）。無ければ空文字。"""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return ""


HARNESS_REASSERT = ("swegemma", "adk-submission", "adk-eval-core", "google-adk")


# 測定環境で意図的に载せ替えた dist。reassert_harness は wheelhouse の組み合わせを取り戻すが、
# ここに入れた物は元に戻さない（戻すと「T4 で通す」という目的が消える）。
MEASURE_ONLY_REPLACED: set[str] = set()


def upgrade_vllm_for_measurement(requested: str, wheelhouse_version: str) -> None:
    """測定だけ vLLM を PyPI の新しい版に载せ替える（model は同じ 4bit QAT checkpoint のまま）。

    wheelhouse の pin 0.19.1 は Gemma4 の head_dim 256（sliding）/ 512（full attention）を
    Triton attention の shared memory 64 KB (Turing sm75) に収められず、生成が 0 token になる
    （doc/environment_limits.md §5、実測 results/logs/smoke.json）。T4 を通す修正（vLLM PR #60942 /
    #39018）は新しい版に入っているので、**測定の為だけ**上げる。
    公式採点は 0.19.1 + 4x L4 (sm89) で走るから、ここでの数字は deviation として manifest に残す。
    """
    log("setup", f"[measure-only] vllm {wheelhouse_version} -> {requested} を PyPI から（依存込み）")
    log("setup", "[measure-only] これは測定環境の载せ替え。提出物（submission/）の model 宣言は変えない")
    proc = subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"vllm=={requested}"],
                          capture_output=True, text=True, timeout=2400)
    if proc.returncode:
        log("setup", f"[measure-only] vllm upgrade 失敗 rc={proc.returncode}: {(proc.stderr or '')[:400]}")
        return
    importlib.invalidate_caches()
    MEASURE_ONLY_REPLACED.add("vllm")
    for name in ("vllm", "torch", "transformers", "compressed_tensors", "tokenizers"):
        log("setup", f"  [measure-only] {name} -> {import_state(name)}")
    ok, msg = deep_probe_child("vllm.entrypoints.openai.api_server")
    log("setup", f"[measure-only] deep import vllm api_server (child) -> {msg}")
    probe = subprocess.run([sys.executable, "-c",
                            "import torch;print(torch.__version__, torch.version.cuda, "
                            "torch.cuda.is_available(), torch.cuda.device_count())"],
                           capture_output=True, text=True, timeout=300)
    log("setup", f"[measure-only] child torch: {(probe.stdout or probe.stderr).strip()[:200]}")


def reassert_harness(wheels: list[str]) -> None:
    """wheelhouse の組み合わせを **最後に取り戻す**（offline, --no-deps --force-reinstall）。

    実測で分かった事: 公式 flow は wheelhouse を --no-deps で入れるので、vllm が宣言している
    `transformers<5` 様の pin は **効かない**。実際の組み合わせは wheelhouse が決める
    （vllm 0.19.1 + transformers 5.13.1 + adk 1.36.1 ...）。ところが PyPI から vllm を依存込みで
    入れると pip が transformers を 4.57.6 に下げてしまい、gemma4 が読めなくなる。
    CUDA repair の後は wheelhouse の wheel を全部入れ直して、公式の組み合わせに戻す。
    （PyPI に無い名前もある所以外から入れる。spec 名で PyPI を探さないこと。）"""
    targets = [w for w in wheels if Path(w).is_file()
               and (dist_of_wheel(w) or "").replace("-", "_") not in MEASURE_ONLY_REPLACED]
    skipped = [w for w in wheels if Path(w).is_file() and w not in targets]
    if skipped:
        log("setup", f"[heal] 意図的に元に戻さない wheel: {[Path(w).name for w in skipped]}")
    if not targets:
        return
    log("setup", f"[heal] wheelhouse の組み合わせを取り戻す: {len(targets)} 個を "
                 f"--no-deps --force-reinstall（offline）")
    proc = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                           "--force-reinstall", *targets], capture_output=True, text=True, timeout=1200)
    if proc.returncode:
        log("setup", f"[heal] wheelhouse re-assert 失敗 rc={proc.returncode}: {(proc.stderr or '')[:300]}")
    importlib.invalidate_caches()
    for name in ("swegemma", "adk_submission", "adk_eval_core", "google.adk", "vllm", "transformers"):
        log("setup", f"  {name} -> {import_state(name)}")


def vllm_torch_pins() -> list[str]:
    """入っている vllm が要求する torch / torchvision / torchaudio の **正確な pin**を
    installed metadata から読む（ version を推測で書かない）。"""
    from importlib.metadata import requires

    specs: list[str] = []
    for req in (requires("vllm") or []):
        head = re.split(r"[<>=!~;\[]", req, 1)[0].strip().lower()
        if head in TORCH_TRIO and "==" in req:
            specs.append(req.split(";")[0].strip())
    return specs


def install_torch_matching_vllm() -> bool:
    """vllm の pin に合った torch 三兄弟を入れる（ deps 込み → nvidia の cu12x lib も付いて来る）。

    なぜ vllm 一式を入れないか: pip が transformers / huggingface_hub まで vllm の pin に合わせて
    下げる。transformers 5.13.1 は huggingface-hub>=1.5 を要求するので、そこへ触ると gemma4 が
    読めないまま落ちる。torch だけ触るのが一番狭い repair である。
    """
    specs = vllm_torch_pins()
    if not specs:
        log("setup", "[heal] vllm の metadata から torch pin を読めない。"
                     "（pip download 出来ないので pypi.org/pypi/vllm/<ver>/json を見て手で決める）")
        return False
    log("setup", f"[heal] torch stack を vllm の pin に合わせる: pip install {' '.join(specs)} "
                 "(deps 込み。nvidia の cu12x lib もここから入る)")
    try:
        proc = subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", *specs],
                              capture_output=True, text=True, timeout=2400)
    except subprocess.TimeoutExpired:
        log("setup", "[heal] torch install が 40 min で終わらない（quota を守って打ち切る）")
        return False
    importlib.invalidate_caches()
    if proc.returncode:
        log("setup", f"[heal] torch pin install 失敗 rc={proc.returncode}")
        for line in (proc.stderr or "").splitlines()[-8:]:
            log("setup", f"  pip: {line[:220]}")
        return False
    log("setup", f"[heal] torch pin install 完了（{', '.join(specs)}）")
    return True


def reinstall_vllm_stack(version: str) -> bool:
    """最終手段: vllm を **依存込み**で入れ直す。pip が vllm の pin に合う torch（cu12 build）と
    nvidia libs をまとめて入れるので、base image の torch stack は置き換わる。
    これが必要だという事は「wheelhouse と今の Kaggle image の CUDA が合ってない」という意味である。"""
    spec = f"vllm=={version}" if version else "vllm"
    log("setup", f"[heal] 最終手段: pip install {spec}（依存込み。base image の torch まで置き換わる）")
    try:
        proc = subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", spec],
                              capture_output=True, text=True, timeout=2400)
    except subprocess.TimeoutExpired:
        log("setup", "[heal] pip install が 40 min で終わらない（quota を守って打ち切る）")
        return False
    importlib.invalidate_caches()
    if proc.returncode:
        log("setup", f"[heal] {spec} 失敗 rc={proc.returncode}")
        for line in (proc.stderr or "").splitlines()[-8:]:
            log("setup", f"  pip: {line[:220]}")
        return False
    log("setup", f"[heal] {spec} install 完了 -> {import_state('vllm')}")
    return True


def run(ctx: Any) -> None:
    set_runtime_env()
    configure_logging(getattr(ctx.cfg, "verbose", False))
    log("setup", f"kaggle={ctx.paths.kaggle} python={sys.version.split()[0]} "
                 f"(wheel tag 基準 cp{CURRENT_KEY})")

    if not ctx.paths.wheelhouse.is_dir():
        log("setup", f"wheelhouse が {ctx.paths.wheelhouse} に無い -> "
                     f"GPU stage は skip される。手元では --stages preflight、"
                     f"GPU 実行は make pipeline-run VARIANT=smoke")
        return

    removed = drop_cutlass_pth_hooks()
    log("setup", f"cutlass .pth hooks removed: {removed}")
    # 何も入れる前に「base image が元々持っていた version」を控えておく（restore の基準）。
    # 何も入れる前の base image の version を控えておく。repair 後に何が動いたか比べる基準にする。
    base_watch = ("torch", "torchvision", "torchaudio", "transformers", "huggingface_hub",
                  "tokenizers", "safetensors", "triton")
    log("setup", "base image versions: "
                 + ", ".join(f"{pkg}={pkg_version(pkg) or '-'}" for pkg in base_watch))
    wheels = stage_wheelhouse(ctx.paths.wheelhouse)
    if not wheels:
        log("setup", "[warn] wheelhouse dir はあるが .whl が 1 つも無い")
        return

    install, skip = plan_wheels(wheels)
    # install より先に LD_LIBRARY_PATH を足す: これで **child process（vllm server）**が
    # site-packages/nvidia の libcudart.so.12 などを見つけられるようになる。
    apply_ld_path()
    log("setup", f"wheel {len(wheels)} 個のうち install {len(install)} 個 / skip {len(skip)} 個 "
                 f"(--no-deps)")
    for name, reason in skip:
        log("setup", f"  skip {name} :: {reason}")
    if install:
        rc, message = pip_install(install)
        if rc:
            log("setup", f"[ERROR] offline install 失敗 rc={rc}")
            for line in message.splitlines()[-12:]:
                log("setup", f"  pip: {line}")
            raise RuntimeError(f"wheelhouse install が失敗した（cp{CURRENT_KEY} と tag の相性を見ること）")
    importlib.invalidate_caches()
    configure_logging(getattr(ctx.cfg, "verbose", False))

    for name in ("swegemma", "adk_submission", "adk_eval_core", "google.adk", "vllm", "litellm"):
        log("setup", f"importable after install: {name} -> {import_state(name)}")
    log_environment()

    # 判定は child process で行う: vllm server は別の interpreter で起動するからだ。
    # 今の process 側だけ失敗しても eval は動く（in-process 側は preload で立てる）。
    preload_cuda_libs()
    target = "vllm.entrypoints.openai.api_server"
    # 測定だけ vLLM を上げる（variant `t4fix`）。wheelhouse の pin がそのまま通る環境でも上げる:
    # T4 で生成を通す目的は「import できるか」ではなく「attention kernel が shared memory に収まるか」だから。
    requested = str(getattr(ctx.cfg, "vllm_version", "") or "").strip()
    wheelhouse_vllm = vllm_version_from(wheels) or "?"
    ok_child, msg_child = deep_probe_child(target)
    if requested and requested != wheelhouse_vllm:
        upgrade_vllm_for_measurement(requested, wheelhouse_vllm)
        ok_child, msg_child = deep_probe_child(target)
        log("setup", f"[measure-only] 载せ替え後の deep import (child) -> {msg_child}")
    log("setup", f"deep import {target} (in-process) -> {deep_probe(target)[1]}")
    log("setup", f"deep import {target} (child)     -> {msg_child}")
    if ok_child:
        log("setup", "vllm の entry point は child で import できる -> server stage は起動できる")
        return

    if not getattr(ctx.cfg, "pip_fallback", True):
        log("setup", "[hint] pip_fallback=false なので補完しない。このままだと server stage は落ちる")
        return

    # 補完の順番は「安い順」。いきなり torch 一式を入れ直さない。
    pins: dict[str, str] = {}
    for name, reason in skip:
        dist, ver = dist_of_wheel(name), version_of_wheel(name)
        if "interpreter 外" in reason and dist and ver and not is_protected(dist):
            pins[dist.replace("-", "_")] = f"{dist}=={ver}"
    replace_incompatible(skip)
    ok_child, msg_child = deep_probe_child(target)
    log("setup", f"deep import {target} (after same-version heal) -> {msg_child}")

    if not ok_child:
        heal_loop((target,), pins)
        ok_child, msg_child = deep_probe_child(target)
        log("setup", f"deep import {target} (after import-name heal) -> {msg_child}")

    if not ok_child:
        if install_torch_matching_vllm():
            reassert_harness(install)  # pip が動かした物を取り戻し、公式の組み合わせに戻す
            apply_ld_path()
            ok_child, msg_child = deep_probe_child(target)
            log("setup", f"deep import {target} (after torch pin) -> {msg_child}")
            # torch 入れ替えで GPU を見失ってないか child で確かめる（ここで失う事故がある）
            probe = subprocess.run([sys.executable, "-c",
                                    "import torch;print(torch.__version__, torch.version.cuda, "
                                    "torch.cuda.is_available(), torch.cuda.device_count())"],
                                   capture_output=True, text=True, timeout=300)
            log("setup", f"child torch after heal: {(probe.stdout or probe.stderr).strip()[:200]}")
            # transformers / huggingface_hub の相性は server まで行くと高いコストで跳ねるので、ここで見る
            for mod in ("transformers", "vllm.config", "compressed_tensors", "huggingface_hub"):
                log("setup", f"child import {mod} -> {deep_probe_child(mod)[1]}")
        elif reinstall_vllm_stack(vllm_version_from(wheels)):
            reassert_harness(install)
            apply_ld_path()
            ok_child, msg_child = deep_probe_child(target)
            log("setup", f"deep import {target} (after vllm reinstall) -> {msg_child}")

    if ok_child:
        log("setup", "vllm の entry point は child で import できる -> server stage は起動できる")
    else:
        log("setup", "[ERROR] vllm server を起動できない。wheelhouse の vllm は cu12 系で base image の "
                     "torch (cu130) と ABI が合わない。上の [heal] 行がどこで止まったかを見ること")
    # 補完で torch / transformers の version が入れ替わった可能性がある。ここを記録しておかないと
    # 「gemma4 model type を transformers が知らない」様の失敗が追えなくなる。
    log_environment()
