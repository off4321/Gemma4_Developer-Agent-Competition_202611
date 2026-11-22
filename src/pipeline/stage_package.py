"""
ステージ `package` = ノートブック Cell 6。

**variant を適用した後の staged agent dir** を zip する。つまり「計測した agent」と
「提出する agent」がバイト単位で一致することを強制する。よくある事故は、計測した YAML と
別の YAML を提出してスコアが説明できなくなることだ。

harness のライブラリが import できるなら adk_submission.validate_directory を使い、
いずれにせよ自前の契約検査（src/package/build_submission.py）を通す。だから wheelhouse が無い
手元でもこの stage は動く。

初学者向けの注意: 提出物は「学習済みモデル」ではなく **エージェント定義の zip**。
ルート直下に agent.yaml が無いと読まれない。拡張子は .yaml .yml .md .txt .py .json .safetensors
だけ、展開後 3 GiB まで、symlink 禁止。
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Any

from pipeline.common import ensure_src_on_path, has_module, log

ensure_src_on_path()
from package.build_submission import (  # noqa: E402
    ALLOWED_SUBMISSION_EXTENSIONS, MAX_SUBMISSION_SIZE_BYTES, Checker, build_zip,
    check_adapters, check_eval_config, check_layout, check_models_and_generation, iter_agent_docs,
)


def assert_no_symlinks(agent_dir: Path) -> None:
    """symlink を防ぐ（zip に込むと展開先を外に出せるため提出規則で禁止されている）。"""
    for dirpath, dirnames, filenames in os.walk(agent_dir, followlinks=False):
        parent = Path(dirpath)
        for name in (*dirnames, *filenames):
            path = parent / name
            if path.is_symlink():
                raise SystemExit(f"[package] symlink は禁止: {path.relative_to(agent_dir)}")
    log("package", "symlink scan OK")


def run(ctx: Any) -> None:
    agent_dir = ctx.require_agent_dir()

    checker = Checker()
    root_cfg = check_layout(agent_dir, checker)
    docs = iter_agent_docs(root_cfg, agent_dir)
    check_models_and_generation(docs, checker)
    check_adapters(agent_dir, docs, checker)
    check_eval_config(agent_dir, checker)
    for warn in checker.warnings:
        log("package", f"[warn] {warn}")
    if checker.errors:
        for err in checker.errors:
            log("package", f"[ERROR] {err}")
        raise SystemExit(f"[package] 契約エラー {len(checker.errors)} 件の提出物は zip しない")

    assert_no_symlinks(agent_dir)
    if has_module("adk_submission") and ctx.limits is not None:
        try:
            from adk_submission import validate_directory

            validate_directory(agent_dir, limits=ctx.limits)
            log("package", "adk_submission.validate_directory: OK")
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"[package] validate_directory が agent dir を拒否した: "
                             f"{type(exc).__name__}: {str(exc)[:300]}") from exc

    out = ctx.paths.working_dir / "submission.zip"
    build_zip(agent_dir, out)

    with zipfile.ZipFile(out) as archive:
        entries = [info for info in archive.infolist() if not info.is_dir()]
        unpacked = sum(info.file_size for info in entries)
        if unpacked > MAX_SUBMISSION_SIZE_BYTES:
            raise SystemExit(f"[package] 展開後 {unpacked:,} B は 3 GiB 上限を越える")
        for info in entries:
            ext = Path(info.filename).suffix.lower()
            if ext not in ALLOWED_SUBMISSION_EXTENSIONS:
                raise SystemExit(f"[package] 許可されない拡張子: {info.filename}")
        root_names = [info.filename for info in entries if "/" not in info.filename]
        log("package", f"{len(entries)} file(s), unpacked {unpacked / (1024 * 1024):.2f} MiB, "
                       f"root entries: {sorted(root_names)}")
        if not any(Path(name).stem in {"agent", "root_agent"} for name in root_names):
            raise SystemExit("[package] agent.yaml が zip のルートに無い")

    # make submit は <repo>/submission.zip を読む。計測した成果物を Kaggle 側の場所にも保つ
    if not ctx.paths.kaggle:
        mirror = ctx.root / "submission.zip"
        mirror.write_bytes(out.read_bytes())
        log("package", f"{mirror} に写した（kaggle competitions submit -f submission.zip）")

    ctx.zip_path = out
    ctx.notes["submission_zip"] = str(out)
    ctx.notes["zip_bytes"] = out.stat().st_size
    log("package", "提出は: make submit   （初めて出す前に doc/kaggle_cli.md を読むこと）")
