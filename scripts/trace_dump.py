#!/usr/bin/env python3
"""Compact human-readable dump of an ATIF trace.

usage: python3 scripts/trace_dump.py <trace.json> [--full] [--grep PATTERN]
"""
import json
import sys


def trunc(s, n):
    s = str(s).replace("\r", "")
    if len(s) <= n:
        return s
    return s[:n] + f"...[+{len(s)-n} chars]"


def main():
    path = sys.argv[1]
    full = "--full" in sys.argv
    pat = None
    if "--grep" in sys.argv:
        pat = sys.argv[sys.argv.index("--grep") + 1]
    d = json.load(open(path))
    steps = d["steps"]
    print("== metrics:", d.get("final_metrics"))
    for s in steps:
        src = s["source"]
        sid = s["step_id"]
        who = (s.get("extra") or {}).get("author", src)
        model = s.get("model_name", "")
        if who == "code_analyzer_agent":
            who = "ANALYZER"
        elif who == "verifier_agent":
            who = "VERIFIER"
        elif who in ("agent", "gemma_e4b_developer_agent"):
            who = "MAIN"
        line = f"[{sid:3}] {who:9}"
        msg = s.get("message") or ""
        if src == "system":
            print(line, "SYSTEM:", trunc(msg, 60) if not full else trunc(msg, 100000))
            continue
        if src == "user":
            print(line, "USER:", trunc(msg, 2000) if not full else trunc(msg, 100000))
            continue
        if msg:
            print(line, "msg:", trunc(msg, 700 if not full else 100000).replace("\n", " ⏎ "))
        for tc in s.get("tool_calls") or []:
            args = json.dumps(tc.get("arguments"), ensure_ascii=False)
            print(line, "CALL:", tc.get("function_name"), trunc(args, 900 if not full else 100000))
        obs = s.get("observation")
        if obs:
            c = obs.get("content") if isinstance(obs, dict) else obs
            print(line, "  -> ", trunc(c, 900 if not full else 100000).replace("\n", " ⏎ "))
        if pat and pat not in json.dumps(s, ensure_ascii=False):
            pass


if __name__ == "__main__":
    main()
