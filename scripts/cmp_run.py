#!/usr/bin/env python3
"""Compare gold patch (tasks.jsonl) vs agent patch for a results dir."""
import json
import re
import sys
from pathlib import Path

tasks = {json.loads(l)["instance_id"]: json.loads(l)
         for l in open("data/tasks.jsonl", encoding="utf-8")}


def files_of(patch):
    out = []
    for m in re.finditer(r"^\+\+\+ b/(\S+)", patch, re.M):
        out.append(m.group(1))
    added = len(re.findall(r"^\+[^+]", patch, re.M))
    removed = len(re.findall(r"^-[^-]", patch, re.M))
    return out, added, removed


def main(resdir):
    res = Path(resdir) / "results"
    rows = [json.loads(l) for l in (res / "task_results.jsonl").read_text().splitlines() if l.strip()]
    for r in rows:
        iid = r["instance_id"]
        gold = tasks.get(iid, {}).get("patch", "")
        agent_p = res / "patches" / f"{iid}.patch"
        agent = agent_p.read_text(encoding="utf-8") if agent_p.exists() else ""
        fresh = agent_p.stat().st_mtime if agent_p.exists() else 0
        gf, ga, gr = files_of(gold)
        af, aa, ar = files_of(agent)
        to = res / "test_outputs" / f"{iid}.log"
        fails = []
        if to.exists():
            for line in to.read_text(errors="ignore").splitlines():
                if line.startswith(("FAILED ", "ERROR tests")):
                    fails.append(line.split(" ")[1].split(" - ")[0])
        print("=" * 90)
        print(f"{iid}  resolved={r['resolved']} exit={r['test_exit_code']} tools={r['tool_calls']}")
        print(f"  GOLD  ({len(gold)}B, +{ga}/-{gr}): {gf}")
        print(f"  AGENT ({len(agent)}B, +{aa}/-{ar}) mtime_ok: {af}")
        print(f"  overlap: {sorted(set(gf) & set(af))}")
        print(f"  missed:  {sorted(set(gf) - set(af))}")
        print(f"  extra:   {sorted(set(af) - set(gf))}")
        print(f"  failing tests ({len(fails)}): {fails[:12]}")


main(sys.argv[1])
