"""
ステージ `server` = ノートブック Cell 4。

「単一宣言モデル」検査、adapter 発見、vLLM 起動、agent が使う model registry の作成。

ここで使う数字は doc/environment_limits.md に書いた Kaggle と採点環境の妥協点:
採点側は 4x L4 / bf16 / tp=4 / gpu_memory_utilization 0.90。Kaggle notebook は 2x T4 で
bf16 非対応なので tp=2 / fp16 / 0.85 に落とす。ただし **max_model_len=32768 は意図的に維持**する
（context 上限を変えると agent の挙動が採点とズレ、計測が意味を失う）。

初学者向けの注意: VRAM = 重み + KV cache。e4b の重みは実測 11.5 GB で、32,768 token の
KV cache が同じ GPU 上の余裕を削る。T4x2(32 GB) で e4b + LoRA + 32k は「入るが余裕が無い」。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pipeline.common import log, require_module, StageUnavailable

MODEL_VLLM_EXTRA_ARGS = {
    "gemma-4-12b-it-qat-w4a16-ct": (
        "--hf-overrides",
        '{"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}',
    ),
}


def find_model_mount(ctx: Any, model_name: str) -> Path:
    """重みの mount 場所を見つける。見つからないときは「何を試したか」と直す場所を出す。"""
    expected = ctx.paths.model_mount(model_name)
    if expected.is_dir():
        return expected
    matches = sorted(ctx.paths.models_root.rglob(model_name)) if ctx.paths.models_root.is_dir() else []
    if matches:
        log("server", f"[warn] 期待した mount {expected} が無い。{matches[0]} を使う")
        return matches[0]
    raise SystemExit(
        f"[server] {model_name} の weight が見つからない: {expected} と {ctx.paths.models_root} を見た\n"
        "        kernels/eval/kernel-metadata.json の model_sources に"
        " google/gemma-4/other/<model>/2 を入れること。doc/kaggle_cli.md"
    )


def dump_server_log(server: Any, exc: BaseException, ctx: Any) -> None:
    """vLLM server の full log を /kaggle/working（= kernel output）に写して出す。
    `kaggle kernels output` で落ちる so 失敗原因が手元で読める。ここが無いと推測で run を重ねることになる。"""
    output = getattr(exc, "output", "") or ""
    log_path = getattr(server, "log_path", None)
    text = output
    if not text and log_path and Path(log_path).is_file():
        text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    if not text:
        log("server", "[warn] server の captured output が無い（log file が空）")
        return
    lines = text.splitlines()
    log("server", f"server log: {len(lines)} 行（{log_path or 'captured output'}）")
    for line in lines[-60:]:
        log("server", f"  vllm: {line[:240]}")
    dest = ctx.paths.working_dir / "vllm_server.log"
    try:
        dest.write_text(text, encoding="utf-8")
        log("server", f"[saved] {dest}（make out-eval RUN=<run> で落とした files に入る）")
    except OSError as write_exc:
        log("server", f"[warn] server log を save できない: {write_exc}")


def snapshot_server_log(ctx: Any, tag: str) -> None:
    """vLLM server の log を working にコピーする（`kaggle kernels output` で落ちるように）。

    server が **起動に成功した**とき、adk_submission は log を /tmp の temp file に書くだけなので
    kernel output には出ない。engine core が eval の途中で死ぬと（実測で死んだ）、その trace が
    毎 run 消えて推測で push を重ねることになる。ここで必ず引き出す。
    """
    src = ctx.notes.get("server_log_path") or ""
    if not src or not Path(src).is_file():
        log("server", f"[warn] server log が見つからない（{tag}）: {src or '未知'}")
        return
    text = Path(src).read_text(encoding="utf-8", errors="replace")
    dest = ctx.paths.working_dir / "vllm_server.log"
    try:
        dest.write_text(text, encoding="utf-8")
        log("server", f"[saved] {dest}（{tag}: {len(text.splitlines())} 行）")
    except OSError as exc:
        log("server", f"[warn] server log を save できない（{tag}）: {exc}")


def server_log_headlines(log_path: str, tag: str, limit: int = 8) -> None:
    """vLLM log から「だから失敗した」行だけ抜いて kernel ログに出す。
    複数の attempt を 1 run で試すと、失敗理由が log file 1 本に上書きされて行く。
    ここに抜いておかないと「どの attempt が何で死んだか」が後から読めない。"""
    if not log_path:
        return
    try:
        text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    keys = ("OutOfResources", "Traceback", "Error", "error", "not support", "No kernels",
            "raise", "assert")
    hits = [ln for ln in text.splitlines() if any(k in ln for k in keys)]
    for line in hits[-limit:]:
        log("server", f"  [{tag}] {line[:220]}")


def list_models(server: Any) -> None:
    """proxy を使わない生 HTTP で /v1/models を見る（served 名はここで見せる）。"""
    import urllib.request

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # ← proxy を無効化
    base = server.base_url  # "/v1" を含む（adk_submission/server.py:204）
    try:
        with opener.open(f"{base}/models", timeout=30) as resp:
            log("smoke", f"GET {base}/models -> {resp.status} {resp.read().decode(errors='replace')[:400]}")
    except Exception as exc:  # noqa: BLE001
        detail = exc.read().decode(errors="replace")[:200] if hasattr(exc, "read") else ""
        log("smoke", f"[ERROR] GET /v1/models 失敗 {type(exc).__name__}: {str(exc)[:200]} {detail}")


def completion_ok(server: Any, served_name: str, api_key: str) -> bool:
    """短い completion を 1 回投げて **生成が実際に走るか**確かめる（health check は通っても
    attention kernel が死んで生成で 500 になる、というのが実測の失敗だったのでここを見る）。"""
    import json
    import urllib.request

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    payload = json.dumps({
        "model": served_name,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "max_tokens": 16, "temperature": 0.0,
    }).encode()
    request = urllib.request.Request(
        f"{server.base_url}/chat/completions", data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key or 'EMPTY'}"})
    try:
        with opener.open(request, timeout=300) as resp:
            log("smoke", f"completion OK ({resp.status}): {resp.read().decode(errors='replace')[:300]}")
            return True
    except Exception as exc:  # noqa: BLE001
        detail = exc.read().decode(errors="replace")[:300] if hasattr(exc, "read") else ""
        log("smoke", f"[ERROR] completion 失敗 {type(exc).__name__}: {str(exc)[:200]} {detail}")
        return False


# 起動に成功しても生成で死ぬ、という失敗の実測原因（results/smoke/vllm_server.log）:
#   v1/attention/backends/triton_attn.py -> ops/triton_unified_attention.py
#   triton.runtime.errors.OutOfResources: out of resource: shared memory, Required: 98304, Hardware limit: 65536
# これは **Triton JIT の compile 時**に起きる（kernel が SM あたりの shared memory を超える）。実測で
# --block-size 64 / 32 を入れても Required: 98304 が 1 文字違わず同じだった -> block size は KV cache の
# 分割サイズで、attention kernel の tile ではない（vLLM issue #38918 / #57712, fix PR #60942）:
#   Gemma4 は head_dim 256（sliding）と global_head_dim 512（full attention）で Triton kernel が 96 KB 要る、
#   Turing (sm75) の上限は 64 KB -> 「利用可能な backend 全てで動かない」と上流が報告している。
# つまり Kaggle free の T4 では vLLM 0.19.1（= wheelhouse の pin）のままでは生成できない。
# ここに残す probe は「backend 自体を変えてみる」安い 1 本だけ（外れたら即 fail なので quota 負担は小）。
ATTENTION_ATTEMPTS = (
    ("base", (), {}),
    ("attn=FLASHINFER", ("--attention-backend", "FLASHINFER"), {}),
    # T4 で Gemma4 が通った報告（vLLM issue #38918 の reply）が使っていた形:
    #   --dtype float16 --gpu-memory-utilization 0.95 --max-num-seqs 1（auto backend = TRITON_ATTN）
    ("max-num-seqs=1", ("--max-num-seqs", "1"), {}),
)


def stop_server(server: Any) -> None:
    """probe に失敗した server は必ず止める（止めないと GPU を掴んだまま次の attempt が起動できない）。"""
    try:
        server.stop()
    except Exception as exc:  # noqa: BLE001
        log("server", f"[warn] server stop 失敗 {type(exc).__name__}: {str(exc)[:120]}")


def run(ctx: Any) -> None:
    require_module("adk_submission", why="VllmServer / VllmConfig が必要")
    require_module("torch", why="tensor_parallel_size を決めるのに必要")
    import litellm
    import torch
    from adk_submission import VllmConfig, VllmServer, discover_adapters
    from adk_submission.discovery import discover_declared_models
    from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS

    from package.build_submission import ALLOWED_MODEL_NAMES

    litellm.drop_params = True
    agent_dir = ctx.require_agent_dir()

    declared = set(discover_declared_models(agent_dir))
    assert len(declared) == 1, f"単一モデル規則に反している: {sorted(declared)}"
    declared_model = next(iter(declared))
    assert declared_model in ALLOWED_MODEL_NAMES, f"使えない model {declared_model!r}"
    if declared_model != ctx.cfg.model_name:
        log("server", f"[warn] agent は {declared_model} を宣言、variant は {ctx.cfg.model_name}")
    model_path = find_model_mount(ctx, declared_model)
    log("server", f"model weights: {model_path}")

    adapters = discover_adapters(str(agent_dir), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
    adapter_map = getattr(adapters, "adapters", None) or {}
    ranks = [r for info in adapter_map.values()
             if (r := getattr(info, "rank", None)) is not None and r > 0]

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if gpu_count < 2:
        log("server", "[warn] GPU が 1 枚以下。e4b は重み 11.5 GB + 32k context の KV cache が必要")
    tp_size = 4 if gpu_count >= 4 else (2 if gpu_count >= 2 else 1)

    # dtype は **compute capability で決める**。`torch.cuda.is_bf16_supported()` は T4 (sm75) で True と
    # 嘘を言う（実測: vLLM が "Bfloat16 is only supported on GPUs with compute capability of at least
    # 8.0. Your Tesla T4 GPU has compute capability 7.5" で worker を落とした）。Kaggle free の GPU は
    # T4 / P100（sm75 / sm70）で、公式採点の 4×L4 (sm89) と違う。ここは計測条件の差なので notes に書く。
    cap = torch.cuda.get_device_capability(0) if gpu_count else (0, 0)
    cc = cap[0] * 10 + cap[1]
    gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
    bf16_ok = cc >= 80
    dtype = "bfloat16" if bf16_ok else "float16"
    gpu_memory_utilization = 0.90 if gpu_count >= 4 else 0.85
    log("server", f"gpus={gpu_count} [{', '.join(gpu_names) or 'none'}] sm{cc} tp={tp_size} "
                  f"dtype={dtype} gpu_mem={gpu_memory_utilization} adapters={sorted(adapter_map)} "
                  f"ranks={ranks or '-'}")
    if not bf16_ok and gpu_count:
        log("server", f"[deviation] sm{cc} は bfloat16 を扱えないので float16 で測る。"
                      "公式採点は 4×L4 (sm89) なので dort では bfloat16。数値の差はここから出る。")
    ctx.notes["hardware"] = {
        "gpu_count": gpu_count,
        "gpu_names": gpu_names,
        "compute_capability": cc,
        "tensor_parallel_size": tp_size,
        "dtype": dtype,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": 32768,
        "official_hw": "4x L4 (sm89), tp=4, gpu_mem=0.90, bfloat16",
    }

    # vllm server は **別の interpreter** で起動する。setup が直した LD_LIBRARY_PATH（site-packages/nvidia の
    # libcudart.so.12 置き場）が child に届いているかを、25 min 待つ前にここで確かめる（quota 節約）。
    from pipeline.stage_setup import deep_probe_child

    ok, message = deep_probe_child("vllm.entrypoints.openai.api_server")
    log("server", f"preflight vllm import (child) -> {message}")
    if not ok:
        raise StageUnavailable(
            f"vllm server が起動できない: {message} "
            "(wheelhouse の vllm は cu12 系, base image の torch は cu130。setup の [heal] 行を見ること)")

    # vLLM の torch.compile / CUDA graph は Turing (sm75) で落ちる。実測:
    #   out of resource: shared memory, Required: 98304, Hardware limit: 65536
    # （Inductor が作る kernel が Turing の 64 KB shared memory を超える）。Kaggle free は T4 なので
    # ここでは eager で動かす。公式採点の 4×L4 (sm89) では compile + cudagraph が生きているので、
    # **ここでの run は生成速度が遅い**（_resolution rate には出ないが task 時間の計測はズレる_）。
    extra_args = list(MODEL_VLLM_EXTRA_ARGS.get(declared_model, ()))
    enforce_eager = bool(gpu_count) and cc < 80
    if enforce_eager:
        extra_args.append("--enforce-eager")
        log("server", f"[deviation] sm{cc} では torch.compile/CUDA graph が shared memory 超過で落ちる -> "
                      "--enforce-eager で測る（生成が遅くなる。公式 L4 では eager ではない）")
    ctx.notes["hardware"]["enforce_eager"] = enforce_eager
    ctx.notes["hardware"]["vllm_extra_args"] = extra_args

    base_kwargs = dict(
        model=str(model_path),
        port=8000,
        host="127.0.0.1",
        tool_call_parser="gemma4",
        reasoning_parser="gemma4",
        max_model_len=32768,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_auto_tool_choice=True,
        enable_lora=bool(adapter_map),
        max_loras=max(1, len(adapter_map)),
        max_lora_rank=max(ranks) if ranks else 128,
        scheduler_reserve_full_isl=False,
        tensor_parallel_size=tp_size,
        # probe ladder で回すので 1 attempt の待ち上限は 12 min（全 try で長引きすぎないための上限）。
        startup_timeout=60 * 12,
    )

    api_key = os.environ.get("LOCAL_API_KEY", "EMPTY")
    server = None
    config = None
    attempts: list[dict[str, str]] = []
    for name, extra, env in ATTENTION_ATTEMPTS:
        for key, value in env.items():
            os.environ[key] = value          # child process に渡すので起動前に立てる
        cfg = VllmConfig(**{**base_kwargs, "extra_args": [*extra_args, *extra]})
        candidate = VllmServer(cfg, adapter_manifest=adapters)
        log("server", f"[attempt {name}] starting vLLM: extra_args={cfg.extra_args} env={env or '-'}")
        try:
            candidate.start()
        except BaseException as exc:  # noqa: BLE001
            log("server", f"[attempt {name}] 起動失敗 {type(exc).__name__}: {str(exc)[:200]}")
            server_log_headlines(getattr(candidate, "log_path", "") or "", name)
            stop_server(candidate)
            attempts.append({"attempt": name, "result": "start_failed"})
            continue
        if not completion_ok(candidate, cfg.model, api_key):
            log("server", f"[attempt {name}] 生きてるのに生成が失敗（health は OK でも attention kernel で死ぬ）")
            server_log_headlines(getattr(candidate, "log_path", "") or "", name)
            snapshot_server_log(ctx, f"attempt_{name}")
            stop_server(candidate)
            attempts.append({"attempt": name, "result": "generation_failed"})
            continue
        server, config = candidate, cfg
        attempts.append({"attempt": name, "result": "ok"})
        log("server", f"[attempt {name}] 生成 OK -> この config で測る")
        break

    if server is None:
        ctx.notes["server_attempts"] = attempts
        raise StageUnavailable(
            "vLLM で token が 1 つも生成できない。試した config: "
            + ", ".join(f"{a['attempt']}={a['result']}" for a in attempts)
            + "  sm75 (T4) の attention kernel が shared memory 64 KB に収まらない問題の可能性が高い"
            "（results/<run>/vllm_server.log と doc/environment_limits.md を見る）")

    log("server", f"vLLM up at {server.base_url} (tp={tp_size})")
    requested_vllm = str(getattr(ctx.cfg, "vllm_version", "") or "").strip()
    if requested_vllm:
        ctx.notes["hardware"]["vllm_version_requested"] = requested_vllm
        log("server", f"[deviation] vLLM は測定だけ {requested_vllm} に载せ替えている。"
                      "公式採点は wheelhouse の 0.19.1 + 4x L4 (sm89) で走る")
    ctx.notes["server_log_path"] = str(getattr(server, "log_path", "") or "")
    ctx.notes["server_attempts"] = attempts
    ctx.notes["hardware"]["vllm_extra_args"] = list(config.extra_args)

    models = server.create_model_registry(aliases=[declared_model], model_prefix="openai/",
                                         api_key=api_key)

    if ctx.cfg.smoke_test:
        # litellm が投げる model id は **vLLM の served name** と一致していなければならない:
        #   base_url は既に "/v1" を含む（足すと 404 detail:Not Found）
        #   base model の served name は model path / LoRA adapter は adapter 名（--lora-modules name=path）
        # create_model_registry が作る LiteLlm も同じ形なので、ここで測るのは agent loop が使う経路そのもの。
        list_models(server)
        served_base = getattr(server, "effective_model_path", None) or str(config.model)
        targets = [(declared_model, served_base)] + [(name, name) for name in sorted(adapter_map)]
        for alias, served_name in targets:
            try:
                response = litellm.completion(
                    model=f"openai/{served_name}", api_base=server.base_url, api_key=api_key,
                    messages=[{"role": "user", "content": "Reply with exactly: OK"}],
                    max_tokens=64, temperature=0.0, num_retries=2,
                )
                log("smoke", f"litellm {alias} (served {served_name}): "
                             f"{response.choices[0].message.content!r}")
            except Exception as exc:  # noqa: BLE001
                log("smoke", f"[warn] litellm {alias} 失敗: {type(exc).__name__}: {str(exc)[:300]}")

    ctx.server, ctx.models, ctx.adapters = server, models, adapters
    ctx.adapter_names, ctx.declared_model = sorted(adapter_map), declared_model
    ctx.notes.update({"tp_size": tp_size, "dtype": dtype, "bf16": bf16_ok})
