#!/usr/bin/env python3
"""測定の**台帳**を作る道具。「自分の測定値」と「公式スコア」を同じ表に並べて、ズレを把握する。

測定の種類は 3 つあり、それぞれ答えられる質問が違う:

| 型 | 何を使うか | 答えられる | 答えられない |
|---|---|---|---|
| **公式提出** | `submission.zip` を Kaggle に提出 | 公式 Resolution Rate（非公開 256 task、4×L4、公式 harness） | 内訳（task 単位は見えない） |
| **公式 harness 局所測定** | wheelhouse pin の `swegemma` / `adk_submission` を Colab/Kaggle で実行 | task 単位の内訳・失敗 taxonomy・prompt/config の差分比較 | 公式スコアそのもの（task 集合と GPU が違う） |
| **自分の harness** | `src/pipeline` 独自 eval / 自作の検証 | 速い反復・log・patch 検査 | 公式と同じ結果である保証（parity_check で測るまで信用しない） |

台帳は `results/scoreboard.csv`。1 行 = 1 測定。公式提出のスコアも同じ表に載せて
**「局所測定の Resolution Rate は公式スコアとどれくらいズレるか」** を常に見られるようにします。

使い方:

```bash
# 1) 局所測定を 1 行登録する（公式 harness 出力でも自分の pipeline 出力でも自動判別）
python3 src/pipeline/scoreboard.py --add --name colab12 --harness official-pin --env colab-l4 \
    --results /content/repo/work/results/colab12
python3 src/pipeline/scoreboard.py --add --name smoke --harness own-pipeline --env kaggle-t4 \
    --results results/smoke --note "vllm 0.28.0 測定専用 / sandbox subprocess"

# 2) 公式提出の一覧（スコア）を拾って同じ表に載せる
kaggle competitions submissions -c gemma-4-developer-agent --csv > results/official_submissions.csv
python3 src/pipeline/scoreboard.py --official-csv results/official_submissions.csv

# 3) 表を見る（測定値と公式スコアを並べてズレを出す）
python3 src/pipeline/scoreboard.py --table
```
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import subprocess
from pathlib import Path

LEDGER = "results/scoreboard.csv"
FIELDS = ["ts", "name", "kind", "harness", "env", "tasks", "resolved", "rate", "note"]
COMPETITION = "gemma-4-developer-agent"
SCORE_HINTS = ("totalScore", "score", "correctFraction", "resolutionRate", "publicScore", "privateScore")

OFFICIAL_ROWS = "**/task_results.jsonl"
PREDICTIONS = "**/predictions_*.json"
SUMMARY = "**/summary.json"


def _first_hit(dirpath: Path, pattern: str) -> Path | None:
    hits = sorted(dirpath.glob(pattern))
    return hits[0] if hits else None


def _resolved_of(rows: list[dict]) -> int:
    return sum(1 for row in rows if row.get("resolved"))


def read_result_dir(dirpath: Path) -> tuple[int, int, str]:
    """結果 dir から (resolved, tasks, 形式) を読む（parity_check と同じ 3 形式を順に見る）。

    「file が有るが 0 件」は測定済み 0 件として、その形式名を返す（次の形式に落とさない）。
    """
    dirpath = Path(dirpath)

    rows_path = _first_hit(dirpath, OFFICIAL_ROWS)
    if rows_path is not None:
        lines = rows_path.read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines if line.strip()]
        return _resolved_of(rows), len(rows), "official harness"

    preds_path = _first_hit(dirpath, PREDICTIONS)
    if preds_path is not None:
        predictions = json.loads(preds_path.read_text(encoding="utf-8"))
        return _resolved_of(predictions), len(predictions), "own pipeline"

    summary_path = _first_hit(dirpath, SUMMARY)
    if summary_path is not None:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return (int(summary.get("resolved") or 0), int(summary.get("total_tasks") or 0),
                "official harness")

    raise SystemExit(f"結果が無い: {dirpath}（task_results.jsonl / predictions_*.json / summary.json）")


def load_ledger(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def save_ledger(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDS})


def cmd_add(args) -> int:
    """局所測定 1 回分を台帳に 1 行足す（harness / env を書けば後から比較できる）。"""
    resolved, tasks, form = read_result_dir(Path(args.results))
    harness, env = args.harness or form, args.env or "-"
    rate = round(resolved / tasks, 4) if tasks else 0.0
    row = {"ts": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M"),
           "name": args.name, "kind": "measured", "harness": harness, "env": env,
           "tasks": tasks, "resolved": resolved, "rate": rate, "note": args.note or form}
    path = Path(args.ledger)
    rows = load_ledger(path)
    rows.append(row)
    save_ledger(path, rows)
    print(f"[scoreboard] 追記: {row['name']} {resolved}/{tasks} = {rate} "
          f"(harness={harness} env={env}) -> {path}")
    return 0


def official_csv_text(args) -> str | None:
    """Kaggle 提出一覧の CSV text（--official-csv が無ければ kaggle CLI）。認証無しなら None。"""
    if args.official_csv:
        return Path(args.official_csv).read_text(encoding="utf-8")
    proc = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMPETITION, "--csv", "-q"],
                          capture_output=True, text=True)
    text = ((proc.stdout or "") + (proc.stderr or "")).strip()
    if "Option A" in text or "Missing key" in text or not text:
        print("[scoreboard] Kaggle 認証が取れないので CSV を手で落としてから --official-csv を使う:")
        print(f"  kaggle competitions submissions -c {COMPETITION} --csv > results/official_submissions.csv")
        return None
    return text


def score_column(fields: list[str]) -> str | None:
    """列名は版で変わる。公式 CSV の列名 -> "score" を含む列 の順に寄せる。"""
    exact = (f for f in fields if f in SCORE_HINTS)
    loose = (f for f in fields if "score" in f.lower())
    return next(exact, None) or next(loose, None)


def official_row(item: dict, score_col: str | None) -> dict:
    return {"ts": (item.get("submittedDate") or item.get("date") or "").strip(),
            "name": (item.get("submissionSource") or item.get("fileName") or "submission").strip(),
            "kind": "official", "harness": "official (4xL4, hidden 256 tasks)",
            "env": "kaggle-official", "tasks": 256, "resolved": "",
            "rate": (item.get(score_col) or "").strip() if score_col else "",
            "note": (item.get("comment") or "").strip()[:120]}


def cmd_official(args) -> int:
    """公式提出のスコアを台帳に載せ直す（measured 行はそのまま、official 行だけ差替）。"""
    text = official_csv_text(args)
    if text is None:
        return 1
    reader = csv.DictReader(text.splitlines())
    submissions = list(reader)
    if not submissions:
        print("[scoreboard] 公式提出はまだ無い（No submissions found）")
        return 0
    score_col = score_column(reader.fieldnames or [])
    print(f"[scoreboard] 公式提出 {len(submissions)} 件（score column = {score_col or '?'}）")

    path = Path(args.ledger)
    ledger = [row for row in load_ledger(path) if row.get("kind") != "official"]
    ledger += [official_row(item, score_col) for item in submissions]
    save_ledger(path, ledger)
    for row in (r for r in ledger if r["kind"] == "official"):
        print(f"  {row['ts']}  {row['name']:28s} score={row['rate'] or '-'}  {row['note'][:60]}")
    print(f"[scoreboard] {path} に official 行として反映した")
    return 0


def rate_text(value) -> str:
    """rate 列を表示用 4 桁に。数値で無い物はそのまま、空は "-"。"""
    try:
        return f"{float(value or 0):.4f}"
    except (TypeError, ValueError):
        return str(value or "-")


def _print_calibration(measured: list[dict], official: list[dict]) -> None:
    """局所測定の rate と公式スコアの差（同じ提出に対する測定を並べると符号が見える）。"""
    best = max((float(row["rate"]) for row in official if row.get("rate")), default=None)
    print("\n[scoreboard] 校正: 自分の測定 rate と公式スコアの差を見る物差し")
    for row in measured:
        if best is None:
            continue
        try:
            rate = float(row.get("rate") or 0)
        except ValueError:
            continue
        print(f"  {row['name']:22s} measured={rate:.4f}  best official={best:.4f}  "
              f"差={rate - best:+.4f}")
    print("  （task 集合が違うので一致は期待しない。同じ提出に対する測定を並べるとズレの符号が見える）")


def cmd_table(args) -> int:
    """台帳を 1 枚の表にして出し、公式スコアが有ればズレも出す。"""
    rows = load_ledger(Path(args.ledger))
    if not rows:
        print("[scoreboard] 台帳が空。まず --add か --official-csv を使うこと")
        return 0
    print(f"{'ts':17s} {'kind':9s} {'name':22s} {'harness':28s} {'env':13s} {'tasks':>5s} {'res':>4s} {'rate':>7s}")
    for row in rows:
        print(f"{row.get('ts',''):17s} {row.get('kind',''):9s} {row.get('name',''):22s} "
              f"{row.get('harness',''):28s} {row.get('env',''):13s} {str(row.get('tasks','')):>5s} "
              f"{str(row.get('resolved','')):>4s} {rate_text(row.get('rate')):>7s}")

    measured = [row for row in rows if row.get("kind") == "measured"]
    official = [row for row in rows if row.get("kind") == "official"]
    if measured and official:
        _print_calibration(measured, official)
    elif not official:
        print("\n[scoreboard] 公式スコアがまだ無い。節目では submission.zip を提出して --official-csv で取り込むこと")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default=LEDGER)
    ap.add_argument("--add", action="store_true")
    ap.add_argument("--official", action="store_true", help="kaggle CLI で提出一覧を拾う")
    ap.add_argument("--name")
    ap.add_argument("--results")
    ap.add_argument("--harness", help="official-pin / official-upgraded / own-pipeline 等")
    ap.add_argument("--env", help="colab-l4 / kaggle-t4 / local 等")
    ap.add_argument("--note")
    ap.add_argument("--official-csv")
    ap.add_argument("--table", action="store_true")
    args = ap.parse_args()
    if args.add:
        if not args.name or not args.results:
            raise SystemExit("--add には --name と --results が必要")
        return cmd_add(args)
    if args.official_csv or args.official:
        return cmd_official(args)
    return cmd_table(args)


if __name__ == "__main__":
    raise SystemExit(main())
