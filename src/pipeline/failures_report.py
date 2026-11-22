#!/usr/bin/env python3
"""results dir を **task 別の不出来ファイル 2 個**（`failures.csv` / `failures.md`）に落とす。

「問題別でどれができていないか、エラーも含めて」に応出す物です。公式 harness（Colab / Kaggle の cell 出力）でも
自分の pipeline（`src/pipeline` の `eval`）でも同じ列になるように、読み取りは形式を自動判別します。

| 出所 | file | 内容 |
|---|---|---|
| 公式 harness（`swegemma eval` / `Evaluator`） | `summary.json` / `task_results.jsonl` / `patches/*.patch` / `test_outputs/*.log` | `resolved / agent_patch_size / test_exit_code / error / tool_calls / duration_seconds` |
| 自分の pipeline（`src/pipeline` の `eval`） | `predictions_<run>.json` / `pipeline_<run>.json` | `{id, prediction, resolved, exit_code, error, tool_calls, duration}` |
| Kaggle kernel 出力 | 上の写し（`predictions_*.json` / `task_results.jsonl`）+ `*.parquet` | 混在 |

`status` の意味（`failures.md` の集計はこの内訳です）:

| status | 意味 | 次の手 |
|---|---|---|
| `resolved` | Phase 2 の test が通った | — |
| `patch_fail` | patch は出たが pytest が落ちた（exit != 0） | patch を見る / budget・prompt を変える |
| `no_patch` | `submit_patch` が呼ばれていない（patch 0 個） | budget（turns / tool calls / 時間）と tool 失敗を見る |
| `crashed` | harness 側で error が出た（agent 完走前に死ぬ） | loader / sandbox / env の bug を疑う。まず `error` 文 |
| `not_run` | **その task に eval が到達していない** | snapshot 欠け・server crash・途中で打ち切り。測れていない task が残っている合図 |

`protected path`（`conftest.py` 等）に touch している patch は `reason` に載せます（反則は resolved 0 になる領域です）。

使い方:

```bash
python3 src/pipeline/failures_report.py --results results/smoke
python3 src/pipeline/failures_report.py --results /content/repo/work/results --task-ids fastapi_15661,pandas_10001
python3 src/pipeline/failures_report.py --results results/colab12 --json results/failures_colab12.json
```

stage `analyze`（`make pipeline-post` / `make pipeline-eval`）が自動的に呼びます。Colab / Kaggle の notebook でも
同じ関数ブロックを cell に埋め込んでいます（`src/pipeline/make_colab_notebook.py` がこのファイルを注入する）。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

OFFICIAL_SUMMARY = "summary.json"
OFFICIAL_ROWS = "task_results.jsonl"

PROTECTED = {"conftest.py", "pytest.ini", ".pytest.ini", "pyproject.toml", "setup.cfg", "tox.ini",
             "sitecustomize.py", "usercustomize.py", "_swegemma_stubs.py"}

FIELDS = ["instance_id", "status", "reason", "test_exit_code", "patch_chars", "tool_calls",
          "duration_s", "error", "test_output_tail"]


def _find(dirpath: Path, patterns: tuple[str, ...]) -> Path | None:
    """dir 直下、見つからなければ再帰、で最初に引っかかる file を 1 個返す。"""
    for pat in patterns:
        hits = sorted(dirpath.glob(pat))
        if hits:
            return hits[0]
    return None


def _json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_jsonl(path: Path) -> list[dict]:
    lines = (line for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return [json.loads(line) for line in lines]


def _row(resolved, patch_chars, exit_code, error, tool_calls, duration) -> dict:
    """3 種類の source が出す行を、report が食べる 1 形の dict に寄せる。"""
    return {"resolved": bool(resolved), "patch_chars": int(patch_chars or 0), "exit": exit_code,
            "error": error, "tool_calls": tool_calls, "duration": duration}


def _official_rows(path: Path) -> dict[str, dict]:
    rows = _iter_jsonl(path)
    return {str(r.get("instance_id")): _row(r.get("resolved"), r.get("agent_patch_size"),
                                            r.get("test_exit_code"), r.get("error"),
                                            r.get("tool_calls"), r.get("duration_seconds"))
            for r in rows}


def _prediction_rows(path: Path) -> dict[str, dict]:
    """自分の pipeline の `predictions_<run>.json`（1 task = 1 prediction 文字列）。"""
    rows = {}
    for item in _json(path):
        prediction = item.get("prediction") or ""
        rows[str(item.get("id"))] = _row(item.get("resolved"),
                                         0 if prediction in ("", "NO_PATCH") else len(prediction),
                                         item.get("exit_code"), item.get("error"),
                                         item.get("tool_calls"), item.get("duration"))
    return rows


def _parquet_rows(path: Path) -> dict[str, dict]:
    import pandas as pd

    frame = pd.read_parquet(path)
    return {str(r.get("id")): _row(r.get("resolved"), len(str(r.get("prediction") or "")),
                                   None, None, None, None)
            for _, r in frame.iterrows()}


def load_source(dirpath: Path) -> tuple[dict[str, dict], dict, str]:
    """1 dir を {task_id: {resolved, patch_chars, exit, error, tool_calls, duration}} + summary + 形式名にする。

    公式 harness / 自分の pipeline / kernel 出力（parquet）のどれが有るか順に見るので、
    Colab から落とした dir も自分の pipeline の output dir も同じ入口で読める。
    """
    dirpath = Path(dirpath)
    if not dirpath.is_dir():
        raise SystemExit(f"無い dir: {dirpath}")

    rows_path = _find(dirpath, (OFFICIAL_ROWS, f"**/{OFFICIAL_ROWS}"))
    if rows_path:
        summary_path = rows_path.parent / OFFICIAL_SUMMARY
        summary = _json(summary_path) if summary_path.is_file() else {}
        return _official_rows(rows_path), summary, "公式 harness (swegemma)"

    pred_path = _find(dirpath, ("predictions_*.json", "**/predictions_*.json"))
    if pred_path:
        manifest = _find(dirpath, ("pipeline_*.json", "**/pipeline_*.json"))
        return _prediction_rows(pred_path), (_json(manifest) if manifest else {}), \
            "自分の pipeline (src/pipeline)"

    parquet = _find(dirpath, ("*.parquet", "**/*.parquet"))
    if parquet:
        try:
            return _parquet_rows(parquet), {}, "parquet (kernel 出力)"
        except Exception as exc:  # pandas が無い / 期待した列が無い。他の form も無いので最後で fail する
            print(f"[failures] parquet を読めなかった: {type(exc).__name__}: {exc}")

    raise SystemExit(f"結果が見つからない: {dirpath}（task_results.jsonl / predictions_*.json / *.parquet のどれかが要る）")


def classify(row: dict) -> tuple[str, str]:
    """1 task を (status, 理由) にする。protected path 反則は build_report で理由に足す。"""
    if row.get("resolved"):
        return "resolved", ""
    if row.get("error"):
        return "crashed", str(row["error"])
    if not row.get("patch_chars"):
        return "no_patch", "agent が submit_patch を呼んでいない（budget 切れ / tool 呼び出し失敗 / 応答が tool 途中で切れた）"
    code = row.get("exit")
    if code == 0:
        return "patch_fail", "pytest exit 0 なのに resolved=False（JUnit XML の read や test 選別を見る）"
    return "patch_fail", f"Phase 2 の pytest が exit {code}（patch は出たが直っていない）"


def _result_file(results_dir: Path, subdir: str, iid: str, suffix: str) -> Path | None:
    """harness は `<instance_id>` を sanitise した名前で書く（results.py 実装）。両方の名前を見る。"""
    for name in (str(iid).replace("/", "__"), iid):
        path = results_dir / subdir / f"{name}{suffix}"
        if path.is_file():
            return path
    return None


def protected_hits(results_dir: Path, iid: str) -> list[str]:
    """`patches/<id>.patch` の diff header から protected path 接触を出す。"""
    patch = _result_file(results_dir, "patches", iid, ".patch")
    if patch is None:
        return []
    text = patch.read_text(encoding="utf-8", errors="replace")
    hits = set()
    for line in text.splitlines():
        if not (line.startswith("+++") or line.startswith("---")):
            continue
        parts = line.split()
        if not parts:
            continue
        name = parts[-1].split("/")[-1].split("\t")[0]
        if name in PROTECTED:
            hits.add(name)
    return sorted(hits)


def test_tail(results_dir: Path, iid: str, lines: int = 3) -> str:
    """Phase 2 の pytest 出力末尾（`test_outputs/<id>.log`）。無いと空。"""
    log = _result_file(results_dir, "test_outputs", iid, ".log")
    if log is None:
        return ""
    text = log.read_text(encoding="utf-8", errors="replace")
    keep = [line.strip() for line in text.splitlines() if line.strip()]
    return " | ".join(keep[-lines:])[:400]


def _not_run_row(iid: str) -> dict:
    return {"instance_id": iid, "status": "not_run",
            "reason": "eval がこの task に到達していない（snapshot 欠け / server crash / 途中で打ち切り）",
            "test_exit_code": "", "patch_chars": 0, "tool_calls": 0, "duration_s": "",
            "error": "", "test_output_tail": ""}


def build_report(results_dir: Path, rows: dict[str, dict], task_ids: list[str] | None = None) -> list[dict]:
    """rows（load_source 出力）+ 測る予定だった task_ids から、not_run を含めた report 行を作る。"""
    out = []
    for iid, row in rows.items():
        status, why = classify(row)
        hits = protected_hits(results_dir, iid)
        if hits:
            why = f"protected path に触れている: {hits} / " + why
        out.append({"instance_id": iid, "status": status, "reason": why,
                    "test_exit_code": row.get("exit"), "patch_chars": int(row.get("patch_chars") or 0),
                    "tool_calls": row.get("tool_calls") or 0, "duration_s": row.get("duration"),
                    "error": str(row.get("error") or "")[:200],
                    "test_output_tail": test_tail(results_dir, iid)})

    measured = {r["instance_id"] for r in out}
    out += [_not_run_row(iid) for iid in (task_ids or []) if iid and iid not in measured]
    order = {"crashed": 0, "not_run": 1, "no_patch": 2, "patch_fail": 3, "resolved": 4}
    out.sort(key=lambda r: (order.get(r["status"], 9), r["instance_id"]))
    return out


def _counts(report_rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in report_rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return counts


def markdown_lines(report_rows: list[dict], results_dir: Path, form: str, summary: dict) -> list[str]:
    counts = _counts(report_rows)
    resolved = sum(1 for r in report_rows if r["status"] == "resolved")
    ran = sum(1 for r in report_rows if r["status"] != "not_run")
    rate = (resolved / ran) if ran else 0.0
    lines = [f"# task 別の出来/不出来（{results_dir.name}）", "",
             f"- 読みた form: {form}",
             f"- 測れた task: {ran} / 報告した task: {len(report_rows)}（`not_run` 含む）",
             f"- resolved: {resolved} / resolution rate（この局所集合）: {rate:.4f}"]
    if summary:
        keys = [k for k in ("total_tasks", "resolved", "resolution_rate", "errors") if k in summary]
        if keys:
            lines.append("- harness summary: " + json.dumps({k: summary[k] for k in keys}, ensure_ascii=False)[:300])
    lines += ["", "## status 集計", ""]
    lines += [f"- {key}: {value}" for key, value in
              sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    lines += ["", "## task 表", "", "| task | status | 理由 | exit | patch | tools | dur |",
              "|---|---|---|---|---|---|---|"]
    lines += [f"| {r['instance_id']} | {r['status']} | {r['reason'][:110]} | {r['test_exit_code']} | "
              f"{r['patch_chars']} | {r['tool_calls']} | {r['duration_s']} |" for r in report_rows]

    reps = [r for r in report_rows if r["status"] in ("crashed", "not_run", "no_patch")][:6]
    reps += [r for r in report_rows if r["status"] == "patch_fail"][:6]
    lines += ["", "## 代表エラー（まずここを見る）", ""]
    lines += [f"- **{r['instance_id']}** [{r['status']}] "
              f"{(r['error'] or r['test_output_tail'] or r['reason'])[:300]}" for r in reps]

    protected = [r["instance_id"] for r in report_rows if "protected path" in r["reason"]]
    lines += ["", "## protected path に触れた patch", ""]
    lines += [f"- {iid}" for iid in protected] or ["- （無し）"]
    lines += ["", "NOTE: gold patch は `data/tasks.jsonl` に有ります。**agent に見せたらこの測定は無効**です。", ""]
    return lines


def write_failures(results_dir: Path, task_ids: list[str] | None = None,
                   out_dir: Path | None = None) -> dict:
    """`failures.csv` / `failures.md` を書いて summary dict を返す（notebook と CLI 共通の入口）。"""
    results_dir = Path(results_dir)
    rows, summary, form = load_source(results_dir)
    report_rows = build_report(results_dir, rows, task_ids)
    out_dir = Path(out_dir) if out_dir else results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "failures.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(report_rows)
    md_path = out_dir / "failures.md"
    md_path.write_text("\n".join(markdown_lines(report_rows, results_dir, form, summary)), encoding="utf-8")

    return {"csv": csv_path, "md": md_path, "counts": _counts(report_rows), "rows": len(report_rows),
            "form": form, "summary": summary}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="results dir から task 別の不出来（failures.csv / failures.md）を出す")
    ap.add_argument("--results", required=True, help="task_results.jsonl / predictions_*.json の有る dir")
    ap.add_argument("--task-ids", default="", help="測る予定だった task id（comma 区切り）。無い task は not_run 行になる")
    ap.add_argument("--out-dir", default="", help="出力 dir（既定は results dir の中）")
    ap.add_argument("--json", default="", help="集計 JSON を書く先（任意）")
    args = ap.parse_args(argv)

    ids = [s.strip() for s in args.task_ids.split(",") if s.strip()]
    report = write_failures(Path(args.results), ids, Path(args.out_dir) if args.out_dir else None)
    print(f"[failures] form = {report['form']}")
    print(f"[failures] 行 {report['rows']} / 集計 {report['counts']}")
    for path in (report["csv"], report["md"]):
        print(f"[failures] 書いた: {path}")
    if args.json:
        target = Path(args.json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({k: report[k] for k in ("counts", "form", "summary")},
                                     ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[failures] 書いた: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
