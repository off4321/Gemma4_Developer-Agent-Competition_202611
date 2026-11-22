"""Colab VM に **その run に要るものだけ**を落とす（lean 版）。

competition 全部（21.9 GB zip）は落とさない。run に要るのは:

| 物 | size | 何に要る |
|---|---|---|
| `tasks.jsonl` | 2 MB | task 選別 / problem statement / test_patch |
| `HARNESS_README.md` | 56 KB | 契約（budget key 等） |
| `graphs/*.json` + `embeddings/*.npz` | 1.3 GB | graph stage（symbol retrieval / embedding similar） |
| `snapshots/<instance_id>.tgz` | **選んだ task だけ**（12 個で ~2 GB） | Phase 2 verification（`extract_snapshot` が repo 実体を展开） |
| `data/wheelhouse/*.whl` | 867 MB | swegemma / adk_submission / google-adk / transformers 等の pin |
| model tar.gz → safetensors | 7.3 GB → 11.5 GB | 推論 |

`snapshots/` を全部（129 個 21.3 GB）落とさないのが要点。`TASK_LIMIT` / `TASK_IDS` で
stage 側の選別（`tasks[:limit]`）と同じ先頭 N 個を揃える。

    colab exec -s <session> -f src/pipeline/colab_fetch.py --env WITH_DATA=1   # data 系
    colab exec -s <session> -f src/pipeline/colab_fetch.py --env WITH_MODEL=1  # model だけ
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(os.environ.get("REPO_DIR", "/content/repo"))
COMPETITION = "gemma-4-developer-agent"
WHEELHOUSE = "metric/gemma-4-developer-agent-wheelhouse"
MODEL_VERSION = "google/gemma-4/other/gemma-4-e4b-it-qat-w4a16-ct/2"
MODEL_NAME = "gemma-4-e4b-it-qat-w4a16-ct"
TASK_LIMIT = int(os.environ.get("TASK_LIMIT", "12"))
TASK_IDS = [t for t in os.environ.get("TASK_IDS", "").split(",") if t]


def skip(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes")


def run(argv: list[str], tail: int = 600) -> None:
    print("+ " + " ".join(argv), flush=True)
    proc = subprocess.run(argv, capture_output=True, text=True)
    out = ((proc.stdout or "") + (proc.stderr or "")).strip()
    if out:
        print(out[-tail:], flush=True)
    if proc.returncode != 0:
        raise SystemExit(f"失敗 (exit {proc.returncode}): {' '.join(argv)}")


def data_dir() -> Path:
    return REPO / "data"


def wheelhouse_dir() -> Path:
    return data_dir() / "wheelhouse"


def model_dir() -> Path:
    return REPO / "models" / "google" / "gemma-4" / "other" / MODEL_NAME / "2"


def comp_files() -> list[tuple[str, int]]:
    """competition の全ファイル名（page 越しで集める）。"""
    rows: list[tuple[str, int]] = []
    token = None
    for _ in range(12):
        cmd = ["kaggle", "competitions", "files", "-c", COMPETITION, "--page-size", "200", "-q"]
        if token:
            cmd += ["--page-token", token]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        out = proc.stdout or ""
        token = None
        for line in out.splitlines():
            if line.startswith("Next Page Token = "):
                token = line.split("= ", 1)[1].strip()
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit() and "/" in parts[0]:
                rows.append((parts[0], int(parts[1])))
            elif len(parts) >= 2 and parts[1].isdigit():
                rows.append((parts[0], int(parts[1])))
        if not token:
            break
    return rows


def download_single(path: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    run(["kaggle", "competitions", "download", "-c", COMPETITION, "-f", path,
         "-p", str(dest), "-q"], tail=200)
    # 単一 file download は dest に直接置かれる / zip で来る場合がある
    zips = sorted(dest.glob("*.zip"))
    for z in zips:
        run(["unzip", "-o", "-q", str(z), "-d", str(dest)], tail=200)
        z.unlink(missing_ok=True)


def selected_tasks() -> list[dict]:
    tasks_path = data_dir() / "tasks.jsonl"
    if not tasks_path.is_file():
        return []
    rows = []
    for line in tasks_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if TASK_IDS:
        return [r for r in rows if r.get("instance_id") in TASK_IDS]
    return rows[: TASK_LIMIT] if TASK_LIMIT > 0 else rows


def fetch_core_data() -> None:
    for name in ("tasks.jsonl", "HARNESS_README.md"):
        if (data_dir() / name).is_file():
            print(f"[data] 有り: {name}", flush=True)
            continue
        download_single(name, data_dir())
    print(f"[data] {data_dir() / 'tasks.jsonl'}", flush=True)


def fetch_graph_and_embeddings() -> None:
    want = {"graphs": [], "embeddings": []}
    have = {"graphs": 0, "embeddings": 0}
    for sub in ("graphs", "embeddings"):
        d = data_dir() / sub
        if d.is_dir():
            have[sub] = len(list(d.iterdir()))
    if have["graphs"] >= 100 and have["embeddings"] >= 100:
        print(f"[graph/emb] 済み graphs={have['graphs']} embeddings={have['embeddings']}", flush=True)
        return
    for path, _size in comp_files():
        head = path.split("/", 1)[0]
        if head in ("graphs", "embeddings"):
            want[head].append(path)
    for sub, paths in want.items():
        done = len(list((data_dir() / sub).glob("*"))) if (data_dir() / sub).is_dir() else 0
        for i, path in enumerate(paths[done:], start=done + 1):
            download_single(path, data_dir() / sub)
            if i % 25 == 0:
                print(f"[{sub}] {i}/{len(paths)}", flush=True)
        print(f"[{sub}] {len(paths)} 個落とした", flush=True)


def fetch_snapshots() -> None:
    tasks = selected_tasks()
    if not tasks:
        print("[snapshots] task が読めない（先に tasks.jsonl）", flush=True)
        return
    ids = [t.get("instance_id") for t in tasks if t.get("instance_id")]
    print(f"[snapshots] 選ぶ task {len(ids)} 個だけ落とす: {ids}", flush=True)
    all_files = {p: s for p, s in comp_files()}
    total = 0
    for iid in ids:
        path = f"snapshots/{iid}.tgz"
        if path not in all_files:
            print(f"[snapshots] 見当たらない: {path}", flush=True)
            continue
        if (data_dir() / "snapshots" / f"{iid}.tgz").is_file():
            print(f"[snapshots] 有り: {iid}", flush=True)
            continue
        download_single(path, data_dir() / "snapshots")
        total += all_files[path]
    print(f"[snapshots] {total / 1e9:.2f} GB（全部 129 個なら 21.3 GB のところ）", flush=True)


def fetch_wheelhouse() -> None:
    wheelhouse_dir().mkdir(parents=True, exist_ok=True)
    if list(wheelhouse_dir().glob("*.whl")):
        print("[wheelhouse] 済み", flush=True)
        return
    run(["kaggle", "datasets", "download", "-d", WHEELHOUSE, "-p", str(wheelhouse_dir()),
         "--unzip", "-q"], tail=300)


def fetch_model() -> None:
    model_dir().mkdir(parents=True, exist_ok=True)
    if (model_dir() / "model.safetensors").is_file():
        print("[model] 済み", flush=True)
        return
    raw = REPO / "models" / "_raw"
    raw.mkdir(parents=True, exist_ok=True)
    for tar in sorted(raw.rglob("*.tar.gz")):
        print(f"[model] 既有の tar を展開: {tar.name}", flush=True)
        run(["tar", "xzf", str(tar), "-C", str(tar.parent)], tail=200)
    found = list(raw.rglob("model.safetensors"))
    if not found:
        # Kaggle model download は tar.gz 1 個で返す（実測 7.27 GB）
        run(["kaggle", "models", "instances", "versions", "download", MODEL_VERSION,
             "-p", str(raw), "--untar", "-q"], tail=300)
        found = list(raw.rglob("model.safetensors"))
        if not found:
            for tar in sorted(raw.rglob("*.tar.gz")):
                run(["tar", "xzf", str(tar), "-C", str(tar.parent)], tail=200)
            found = list(raw.rglob("model.safetensors"))
    if not found:
        raise SystemExit(f"model.safetensors が見つからない: {raw}")
    src_dir = found[0].parent
    for f in sorted(src_dir.iterdir()):
        if f.is_file() and f.suffix != ".gz":      # tar.gz を model dir に転がさない
            shutil.move(str(f), str(model_dir() / f.name))
    shutil.rmtree(raw, ignore_errors=True)
    print(f"[model] {model_dir() / 'model.safetensors'} "
          f"{(model_dir() / 'model.safetensors').stat().st_size:,} byte", flush=True)


def main() -> int:
    print(f"[bootstrap] REPO={REPO} python={sys.version.split()[0]} "
          f"TASK_LIMIT={TASK_LIMIT} TASK_IDS={TASK_IDS or '-'}", flush=True)
    if not (Path.home() / ".kaggle" / "credentials.json").is_file():
        print("[bootstrap] kaggle creds 無し -> colab upload が必要", flush=True)
        return 1

    fetch_core_data()
    if not skip("SKIP_GRAPH"):
        fetch_graph_and_embeddings()
    if not skip("SKIP_SNAPSHOTS"):
        fetch_snapshots()
    if not skip("SKIP_WHEELHOUSE"):
        fetch_wheelhouse()
    if not skip("SKIP_MODEL"):
        fetch_model()

    summary = {
        "tasks_lines": sum(1 for _ in open(data_dir() / "tasks.jsonl")) if (data_dir() / "tasks.jsonl").is_file() else 0,
        "graphs": len(list((data_dir() / "graphs").glob("*"))) if (data_dir() / "graphs").is_dir() else 0,
        "embeddings": len(list((data_dir() / "embeddings").glob("*"))) if (data_dir() / "embeddings").is_dir() else 0,
        "snapshots": len(list((data_dir() / "snapshots").glob("*.tgz"))) if (data_dir() / "snapshots").is_dir() else 0,
        "wheelhouse": len(list(wheelhouse_dir().glob("*.whl"))),
        "model_bytes": (model_dir() / "model.safetensors").stat().st_size
        if (model_dir() / "model.safetensors").is_file() else 0,
    }
    print("[bootstrap] " + json.dumps(summary, ensure_ascii=False), flush=True)
    print(shutil.disk_usage(str(REPO)).__str__(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
