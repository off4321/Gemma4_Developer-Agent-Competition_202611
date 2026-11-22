#!/usr/bin/env python3
"""**公式 harness と自分の harness が同じ結果を出しているか**を突き合わせる検査。

測定の信憑性は「公式 harness と同じ物を実行しているか」で決まります。このスクリプトは 2 つの結果ディレクトリを
比べて、**同じ task 集合で resolved が食い違う task を洗い出します**。

読み取る物（どちらの形式でも自動判別）:

| 出所 | file | 内容 |
|---|---|---|
| 公式 harness (`swegemma eval` / `Evaluator`) | `summary.json`, `task_results.jsonl`, `patches/*.patch` | `instance_id / resolved / agent_patch_size / test_exit_code / error / tool_calls` |
| 自分の pipeline (`src/pipeline` の `eval`) | `predictions_<run>.json`, `pipeline_<run>.json` | `{id, prediction, resolved}` |
| Kaggle kernel 出力 (`kaggle kernels output`) | `predictions_*.json` + `results/` の写し | 上 2 つの混在 |

使い方:

```bash
# 公式 harness（Colab の cell 6 出力）と自分の pipeline を突き合わせる
python3 src/pipeline/parity_check.py --a /path/to/official_results --b results/smoke

# 同じ公式 harness の 2 回実行（温度 0 でも出る揺らぎ）を測る
python3 src/pipeline/parity_check.py --a results/run_a --b results/run_b --label run_a run_b

# 表だけ欲しければ JSON を出す
python3 src/pipeline/parity_check.py --a ... --b ... --json results/parity.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DETAIL_KEYS = ("resolved", "patch_chars", "exit", "tool_calls", "error")


def load_source(dirpath: Path) -> tuple[dict[str, dict], dict, str]:
    """loader は failures_report.py の 1 実装に寄せる（同じ results dir を両方で読むため）。

    src/pipeline/ を直接実行したときと、stage 経由で import されたときで親 package が違う。
    """
    try:
        from failures_report import load_source as _load_source
    except ImportError:
        from pipeline.failures_report import load_source as _load_source
    return _load_source(dirpath)


def resolved_count(rows: dict[str, dict], ids: list[str]) -> tuple[int, int]:
    """(ids のうち resolved, ids の数)。"""
    return sum(1 for tid in ids if rows[tid]["resolved"]), len(ids)


def _side(dirpath: Path, label: str, form: str, rows: dict, summary: dict, shared: list[str]) -> dict:
    resolved, total = resolved_count(rows, shared)
    return {"dir": str(dirpath), "label": label, "format": form, "resolved": resolved, "of": total,
            "total_reported": summary.get("resolved"), "summary_rate": summary.get("resolution_rate")}


def compare(a_dir: Path, b_dir: Path, a_label: str, b_label: str) -> dict:
    """2 つの結果 dir を突き合わせ、共通 task で resolved が食い違う物を洗い出す。"""
    a_rows, a_summary, a_form = load_source(a_dir)
    b_rows, b_summary, b_form = load_source(b_dir)

    shared = sorted(set(a_rows) & set(b_rows))
    agree, differ = [], []
    for tid in shared:
        (agree if a_rows[tid]["resolved"] == b_rows[tid]["resolved"] else differ).append(tid)

    return {
        "a": _side(a_dir, a_label, a_form, a_rows, a_summary, shared),
        "b": _side(b_dir, b_label, b_form, b_rows, b_summary, shared),
        "shared": shared,
        "only_a": sorted(set(a_rows) - set(b_rows)),
        "only_b": sorted(set(b_rows) - set(a_rows)),
        "agree": agree, "differ": differ,
        "detail": [{"id": tid,
                    "a": {k: a_rows[tid][k] for k in DETAIL_KEYS},
                    "b": {k: b_rows[tid][k] for k in DETAIL_KEYS}} for tid in differ],
    }


def _print_diff(row: dict) -> None:
    x, y = row["a"], row["b"]
    print(f"  - {row['id']:24s} A resolved={str(x['resolved']):5s} patch={x['patch_chars']:>6} exit={x['exit']} "
          f"tools={x['tool_calls']} | B resolved={str(y['resolved']):5s} patch={y['patch_chars']:>6} "
          f"exit={y['exit']} tools={y['tool_calls']}")


def report(out: dict, notes: list[str]) -> None:
    a, b = out["a"], out["b"]
    print(f"[parity] A = {a['label']}  {a['dir']}")
    print(f"[parity] B = {b['label']}  {b['dir']}")
    print(f"[parity] 共有 task {len(out['shared'])} 件  A 一致 {len(out['agree'])} / 食い違い {len(out['differ'])}")
    print(f"[parity] resolved  A {a['resolved']}/{a['of']}   B {b['resolved']}/{b['of']}   "
          f"差 {b['resolved'] - a['resolved']:+d}")
    if out["only_a"] or out["only_b"]:
        print(f"[parity] 片側だけ測った task: A のみ {len(out['only_a'])} / B のみ {len(out['only_b'])} "
              "（task 集合が同じになるように TASK_IDS / task_limit を揃えること）")
    for row in out["detail"]:
        _print_diff(row)
    if not out["differ"]:
        print("[parity] 同じ task で resolved が全部一致。**harness の乖離は出ていない**（patch size や tool calls は違うことがある）")
    if notes:
        print("[parity] 意図した違いとして申告済み:")
        for note in notes:
            print("   -", note)
    print("[parity] 乖離が「意図した違い」で説明できるなら scoreboard に note を残すこと。"
          "説明できないなら自分の harness を直すまで公式提出はしない")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--label", nargs=2, default=["official", "mine"], metavar=("A", "B"))
    ap.add_argument("--note", action="append", default=[],
                    help="意図した違い（sandbox subprocess / task 部分集合 / vllm version 等）を申告する")
    ap.add_argument("--json", help="結果 JSON を書く先")
    args = ap.parse_args()

    out = compare(Path(args.a), Path(args.b), args.label[0], args.label[1])
    report(out, args.note)
    if args.json:
        target = Path(args.json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[parity] 書いた:", args.json)
    return 0 if not out["differ"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
