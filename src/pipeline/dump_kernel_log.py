#!/usr/bin/env python3
"""`kaggle kernels logs` の出力を生ログにする（実行中にも使える）。

 Kaggle の API は kernel のログを **JSON array** で返す:
   [{"stream_name":"stderr","time":123.4,"data":"行\n"}, ...]
 `kaggle kernels output` は完了後の file 一式だけで、実行中のログは取れない。
 `kaggle kernels logs <slug>`（実行中なら `-f`）なら status を繰り返さずに中身が見える:

   kaggle kernels logs <username>/gemma4-e4b-agent-eval > results/logs/smoke.json
   python3 src/pipeline/dump_kernel_log.py results/logs/smoke.json --stage server

 --stage を付けると [server] / [eval] の行だけ見る（500 行 super の setup 一覧を越えるため）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_entries(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    start = text.find("[")
    if start < 0:
        # queued / 起動前だと Kaggle は空を出す。これは失敗ではなく「まだログが無い」。
        raise SystemExit(f"ログがまだ無い: {path}（中身 {text[:60]!r}。status が QUEUED/RUNNING 開始直後は空になる）")
    try:
        return json.loads(text[start:])
    except json.JSONDecodeError as exc:
        # 実行中に -f で切り出した途中切断はここで救う（最後の要素が壊れているだけ）。
        print(f"[warn] JSON 最後まで読めず: {exc}（読めた分だけ出す）", file=sys.stderr)
        cut = text.rfind("},{")
        return json.loads(text[start:cut + 1] if cut > start else "[]")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", help="kaggle kernels logs をリダイレクトした file")
    parser.add_argument("--stage", nargs="*", default=None,
                        help="[server] などの prefix だけ出す（stage 名。'server' と書く）")
    parser.add_argument("--grep", nargs="*", default=None, help="この語を含む行だけ")
    parser.add_argument("--tail", type=int, default=0, help="最後の N 行だけ")
    parser.add_argument("--out", default=None, help="生の全文をここで file に書く（.log）")
    args = parser.parse_args(argv)

    entries = load_entries(Path(args.log))
    lines: list[str] = []
    for entry in entries:
        for raw in (entry.get("data") or "").splitlines():
            if raw.strip():
                lines.append(raw.rstrip())

    if args.out:
        Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[dump] {args.out}（{len(lines)} 行）")

    wanted = lines
    if args.stage:
        prefixes = tuple(f"[{s}]" for s in args.stage)
        wanted = [ln for ln in lines if ln.startswith(prefixes)]
    if args.grep:
        wanted = [ln for ln in wanted if any(g in ln for g in args.grep)]
    if args.tail:
        wanted = wanted[-args.tail:]

    for line in wanted:
        print(line[:400])
    if not wanted:
        print(f"[dump] 該当なし（全 {len(lines)} 行）", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
