#!/usr/bin/env python3
"""Print tool-call / observation timeline of a trace, with full args for edit_file."""
import json
import sys

path = sys.argv[1]
only = sys.argv[2] if len(sys.argv) > 2 else None
wid = int(sys.argv[3]) if len(sys.argv) > 3 else 400
d = json.load(open(path))
for s in d["steps"]:
    who = s.get("extra", {}).get("author", "?")
    tag = f"[{s['step_id']:3}] {who[:12]:12}"
    for tc in s.get("tool_calls") or []:
        a = tc.get("arguments") or {}
        fn = tc["function_name"]
        if only and fn != only:
            continue
        if fn in ("edit_file", "write_file"):
            print(tag + "CALL " + fn + " filepath=" + str(a.get("filepath")))
            print("      OLD = " + json.dumps(a.get("old_string"), ensure_ascii=False)[:wid])
            print("      NEW = " + json.dumps(a.get("new_string"), ensure_ascii=False)[:wid])
        else:
            print(tag + "CALL " + fn + " " + json.dumps(a, ensure_ascii=False)[:wid])
    o = s.get("observation")
    if o:
        c = o.get("content") if isinstance(o, dict) else str(o)
        print("      -> " + c[:wid].replace("\n", " "))
