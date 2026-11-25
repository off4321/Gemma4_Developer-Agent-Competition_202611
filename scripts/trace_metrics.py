#!/usr/bin/env python3
"""Per-trace behaviour metrics for the swegemma ATIF traces."""
import glob
import json
import os
import sys
from collections import Counter

COMPACTION = 14336


def analyse(path):
    d = json.load(open(path))
    name = os.path.basename(path).replace("trace_", "").replace(".json", "")
    steps = d["steps"]
    tools = Counter()
    fails = Counter()
    prompt_max = 0
    over_compaction = 0
    verify_ok = False
    verify_ok_before_submit = False
    submit_step = None
    edits = 0
    edit_fails = 0
    repeated = []
    seen_calls = Counter()
    nudge = 0
    for s in steps:
        if s["source"] == "user":
            pass
        for tc in s.get("tool_calls") or []:
            fn = tc["function_name"]
            tools[fn] += 1
            key = (fn, json.dumps(tc.get("arguments"), sort_keys=True, ensure_ascii=False))
            seen_calls[key] += 1
            if seen_calls[key] > 1:
                repeated.append((s["step_id"], fn, seen_calls[key]))
            if fn == "edit_file":
                edits += 1
            if fn == "submit_patch":
                submit_step = s["step_id"]
        o = s.get("observation")
        if o:
            c = o.get("content") if isinstance(o, dict) else str(o)
            st = None
            try:
                j = json.loads(c)
                st = j.get("status")
            except Exception:
                pass
            if st == "error":
                try:
                    fails[json.loads(c).get("error_type") or "error"] += 1
                except Exception:
                    fails["unparsed"] += 1
            if "VERIFY_OK" in c:
                verify_ok = True
                if submit_step is None:
                    verify_ok_before_submit = True
        m = s.get("metrics") or {}
        p = m.get("prompt_tokens", 0)
        prompt_max = max(prompt_max, p)
        if p > COMPACTION:
            over_compaction += 1
    for s in steps:
        if s["source"] == "user" and "continue your work" in (s.get("message") or ""):
            nudge += 1
    return {
        "task": name,
        "steps": len(steps),
        "prompt_max": prompt_max,
        "steps_over_14k": over_compaction,
        "tools": dict(tools),
        "tool_total": sum(tools.values()),
        "errors": dict(fails),
        "edit_calls": edits,
        "verify_ok": verify_ok,
        "verify_ok_before_submit": verify_ok_before_submit,
        "repeat_calls": len(repeated),
        "nudges": nudge,
        "tokens_total": d.get("final_metrics", {}).get("total_tokens"),
    }


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "result/new_25/results/traces"
    rows = [analyse(p) for p in sorted(glob.glob(d + "/trace_*.json"))]
    for r in rows:
        print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
