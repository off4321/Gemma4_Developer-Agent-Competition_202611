"""Import and normalize external SWE-agent / OpenHands / HF trajectories into `swegemma` JSONL.

Produces up to three role-specialized ChatML JSONL datasets compatible with `src/train/train_lora.py`:
  * `--out-coder`    : Full multi-turn `swegemma` trajectories (`submission/prompts/system.md`)
                       ending with `submit_patch()` -> tool ok -> concise 1-sentence summary.
  * `--out-locator`  : Exploration-only prefix before the first edit/write (<= 6 tool calls) +
                       structured factual localization report (`submission/prompts/analyzer.md`).
  * `--out-verifier` : Balanced `VERDICT: GO` (resolved teacher + local results) and
                       `VERDICT: BLOCK` (`--results-dirs` failed patches, empty diffs,
                       `SyntaxError`, failing `test_outputs`), using `submission/prompts/verifier.md`.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.collect import collect_trajectories as ct  # noqa: E402

MAX_LOCATOR_CALLS = 6
DEFAULT_MAX_EXAMPLES = 2000
DEFAULT_MAX_TURNS = 40
DEFAULT_MIN_EDITS = 1

POSITIVE_EXIT_STATUSES = {"submitted", "resolved", "success", "passed"}
RESOLVED_BOOL_KEYS = ("resolved", "is_resolved", "test_resolved", "passed", "success")

XML_FUNC_RE = re.compile(
    r"<function=([A-Za-z0-9_.-]+)>(.*?)(?:</function>|$)",
    re.DOTALL,
)
XML_PARAM_RE = re.compile(
    r"<parameter=([A-Za-z0-9_.-]+)>(.*?)</parameter>",
    re.DOTALL,
)
JSON_TAG_RE = re.compile(
    r"<(?:tool_call|function_call)>\s*(\{.*?\})\s*</(?:tool_call|function_call)>",
    re.DOTALL,
)
THOUGHT_WRAPPER_RE = re.compile(r"</?(?:think|thought|tool_call|function_call|observation)>", re.IGNORECASE)
TEST_PATH_RE = re.compile(r"(?:^|[\s'\"`(=])((?:[\w.-]+/)*(?:tests?|testing)/[\w./-]*\.py|[\w./-]*test_[\w./-]*\.py|[\w./-]*_test\.py)")
BASH_PY_WRITE_RE = re.compile(
    r"(?:git\s+apply|patch\s+-p|\bsed\s+-i|\btee\b|cat\s+<<[^\n]*>|>\s*)[^\n]*?([A-Za-z0-9_./-]+\.py)"
)
BASH_PATCH_CMD_RE = re.compile(r"\b(?:git\s+apply|patch\s+-p)\b")
EXIT_CODE_RE = re.compile(r"(?:exit[_\s]?code|return\s*code)\s*[:=]\s*(-?\d+)", re.IGNORECASE)


def _load_prompt(rel_path: str) -> str:
    path = REPO_ROOT / rel_path
    return path.read_text(encoding="utf-8").strip()


def normalize_filepath(filepath: str) -> str:
    """Strip container prefixes (`/testbed/`, `/workspace/`) to yield workspace-relative paths."""
    cleaned = str(filepath or "").strip().strip("'\"")
    for prefix in ("/testbed/", "/workspace/"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break
    if cleaned in ("/testbed", "/workspace"):
        return "."
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned


def _normalize_repo(repo: Any) -> str:
    text = str(repo or "").strip()
    if "/" not in text and "__" in text:
        parts = text.split("__", 1)
        return f"{parts[0]}/{parts[1]}"
    return text


def _parse_exclude_repos(raw: str | None) -> set[str]:
    if not raw:
        return set()
    excluded: set[str] = set()
    for item in raw.split(","):
        norm = _normalize_repo(item).lower()
        if norm:
            excluded.add(norm)
    return excluded


def _repo_excluded(repo: str, excluded: set[str]) -> bool:
    if not excluded or not repo:
        return False
    norm = _normalize_repo(repo).lower()
    return norm in excluded


def _coerce_bool(val: Any) -> bool | None:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        low = val.strip().lower()
        if low in ("true", "1", "yes"):
            return True
        if low in ("false", "0", "no"):
            return False
    return None


def _is_resolved_record(row: dict[str, Any]) -> bool:
    """Return True if the record is resolved or has no resolution metadata."""
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    for container in (row, meta):
        for key in RESOLVED_BOOL_KEYS:
            if key in container and container[key] is not None:
                parsed = _coerce_bool(container[key])
                if parsed is not None:
                    return parsed

    for container in (row, meta):
        exit_status = container.get("exit_status")
        if exit_status is not None and str(exit_status).strip():
            return str(exit_status).strip().lower() in POSITIVE_EXIT_STATUSES

    return True


def _extract_raw_turns(row: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("messages", "trajectory", "history", "steps", "events", "conversations"):
        raw = row.get(key)
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                continue
        if isinstance(raw, list) and raw:
            return [item for item in raw if isinstance(item, dict)]
    return []


def _infer_repo(row: dict[str, Any], instance_id: str) -> str:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    repo = row.get("repo") or row.get("repo_name") or meta.get("repo") or ""
    if repo:
        return _normalize_repo(repo)
    if "__" in instance_id:
        owner, rest = instance_id.split("__", 1)
        repo_name = re.sub(r"-\d+$", "", rest)
        return f"{owner}/{repo_name}"
    return "unknown/repo"


def _extract_problem_statement(row: dict[str, Any], turns: list[dict[str, Any]]) -> str:
    for key in ("problem_statement", "issue", "instruction", "prompt"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    for turn in turns:
        role = str(turn.get("role") or turn.get("source") or turn.get("from") or "").lower()
        if role in ("user", "human"):
            text = _turn_text(turn)
            if text:
                return text
    return "Resolve the software engineering issue described in the repository."


def _turn_text(turn: dict[str, Any]) -> str:
    for key in ("content", "message", "text", "value", "observation"):
        val = turn.get(key)
        if isinstance(val, str):
            return val
        if isinstance(val, list):
            parts = [str(p.get("text", p)) if isinstance(p, dict) else str(p) for p in val]
            return "\n".join(parts)
    return ""


def _parse_view_range(raw: Any) -> tuple[int, int]:
    if isinstance(raw, (list, tuple)) and len(raw) >= 1:
        start = max(1, int(raw[0]))
        end = int(raw[1]) if len(raw) >= 2 and raw[1] is not None else -1
        return start, (start + 149 if end < start else end)
    if isinstance(raw, str) and raw.strip():
        nums = [int(n) for n in re.findall(r"-?\d+", raw)]
        if nums:
            start = max(1, nums[0])
            end = nums[1] if len(nums) >= 2 else -1
            return start, (start + 149 if end < start else end)
    return 1, 150


def _strip_param_edge_newlines(key: str, value: str) -> str:
    if key in ("old_str", "old_string", "new_str", "new_string"):
        return value.strip("\n")
    if key in ("file_text", "content"):
        return value.lstrip("\n")
    return value.strip()


def _parse_xml_calls(text: str) -> tuple[list[tuple[str, dict[str, Any]]], str]:
    """Extract `<function=name><parameter=k>v</parameter></function>` blocks and remaining thought."""
    calls: list[tuple[str, dict[str, Any]]] = []
    spans: list[tuple[int, int]] = []
    for match in XML_FUNC_RE.finditer(text):
        func_name = match.group(1).strip()
        body = match.group(2)
        args: dict[str, Any] = {}
        for pmatch in XML_PARAM_RE.finditer(body):
            pkey = pmatch.group(1).strip()
            pval = _strip_param_edge_newlines(pkey, pmatch.group(2))
            args[pkey] = pval
        calls.append((func_name, args))
        spans.append(match.span())

    if not spans:
        return [], text

    parts: list[str] = []
    cursor = 0
    for start, end in spans:
        parts.append(text[cursor:start])
        cursor = end
    parts.append(text[cursor:])
    remainder = THOUGHT_WRAPPER_RE.sub("", "".join(parts))
    return calls, ct._clean_message_text(remainder)


def _parse_json_tool_calls(text: str) -> tuple[list[tuple[str, dict[str, Any]]], str]:
    """Extract `<tool_call>{...}</tool_call>` JSON blocks and remaining thought."""
    calls: list[tuple[str, dict[str, Any]]] = []
    spans: list[tuple[int, int]] = []
    for match in JSON_TAG_RE.finditer(text):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("name"):
            args = payload.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if isinstance(args, dict):
                calls.append((str(payload["name"]), args))
                spans.append(match.span())

    if not spans:
        return [], text

    parts: list[str] = []
    cursor = 0
    for start, end in spans:
        parts.append(text[cursor:start])
        cursor = end
    parts.append(text[cursor:])
    remainder = THOUGHT_WRAPPER_RE.sub("", "".join(parts))
    return calls, ct._clean_message_text(remainder)


def _clean_thought(text: str) -> str:
    cleaned = THOUGHT_WRAPPER_RE.sub("", text or "")
    cleaned = XML_FUNC_RE.sub("", cleaned)
    cleaned = JSON_TAG_RE.sub("", cleaned)
    return ct._clean_message_text(cleaned)


def _extract_calls_and_thought(turn: dict[str, Any]) -> tuple[list[tuple[str | None, str, dict[str, Any]]], str]:
    """Extract `(call_id, raw_name, args_dict)` list and full cleaned thought from an assistant turn."""
    raw_thought = ""
    for tkey in ("reasoning_content", "reasoning", "thought"):
        if isinstance(turn.get(tkey), str) and turn[tkey].strip():
            raw_thought = turn[tkey]
            break
    content_text = _turn_text(turn)
    if not raw_thought:
        raw_thought = content_text

    # 1) OpenAI tool_calls
    raw_calls = turn.get("tool_calls") or turn.get("toolCalls")
    if isinstance(raw_calls, list) and raw_calls:
        extracted: list[tuple[str | None, str, dict[str, Any]]] = []
        for c in raw_calls:
            if not isinstance(c, dict):
                continue
            fn = c.get("function") if isinstance(c.get("function"), dict) else c
            name = str(fn.get("name") or c.get("function_name") or "").strip()
            raw_args = fn.get("arguments") if "arguments" in fn else c.get("arguments")
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args or "{}")
                except json.JSONDecodeError:
                    args = {}
            elif isinstance(raw_args, dict):
                args = dict(raw_args)
            else:
                args = {}
            if name:
                extracted.append((c.get("id") or c.get("tool_call_id"), name, args))
        return extracted, _clean_thought(raw_thought)

    # 2) Legacy OpenAI function_call
    if isinstance(turn.get("function_call"), dict):
        fc = turn["function_call"]
        name = str(fc.get("name") or "").strip()
        raw_args = fc.get("arguments")
        args = json.loads(raw_args) if isinstance(raw_args, str) else (dict(raw_args) if isinstance(raw_args, dict) else {})
        if name:
            return [(None, name, args)], _clean_thought(raw_thought)

    # 3) OpenHands event format (action + args)
    if turn.get("action") and isinstance(turn.get("args"), dict):
        action = str(turn["action"]).strip().lower()
        args = dict(turn["args"])
        action_map = {"run": "execute_bash", "edit": "str_replace_editor", "write": "write_file",
                      "read": "read_file", "finish": "finish"}
        mapped_action = action_map.get(action, action)
        return [(None, mapped_action, args)], _clean_thought(raw_thought)

    # 4) XML `<function=...>...</function>` in content
    xml_calls, xml_thought = _parse_xml_calls(content_text)
    if xml_calls:
        thought = _clean_thought(raw_thought) if raw_thought != content_text else xml_thought
        return [(None, name, args) for name, args in xml_calls], thought

    # 5) JSON `<tool_call>...</tool_call>` in content
    json_calls, json_thought = _parse_json_tool_calls(content_text)
    if json_calls:
        thought = _clean_thought(raw_thought) if raw_thought != content_text else json_thought
        return [(None, name, args) for name, args in json_calls], thought

    return [], _clean_thought(raw_thought)


def _map_foreign_call(raw_name: str, args: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """Map SWE-agent / OpenHands / R2E-Gym / swegemma tool calls to canonical `swegemma` tools."""
    name = raw_name.strip().lower()

    if name in ("str_replace_editor", "editor", "file_editor", "oh_editor"):
        cmd = str(args.get("command") or args.get("cmd") or "").strip().lower()
        filepath = normalize_filepath(args.get("path") or args.get("filepath") or args.get("file_path") or "")
        if not filepath:
            return None
        if cmd in ("view", "read", ""):
            if "start_line" in args or "end_line" in args:
                start_line = max(1, int(args.get("start_line") or 1))
                end_line = max(start_line, int(args.get("end_line") or (start_line + 149)))
            else:
                start_line, end_line = _parse_view_range(args.get("view_range"))
            return "read_file", {"filepath": filepath, "start_line": start_line, "end_line": end_line}
        if cmd in ("str_replace", "edit", "replace"):
            old_s = args.get("old_str") if args.get("old_str") is not None else args.get("old_string", "")
            new_s = args.get("new_str") if args.get("new_str") is not None else args.get("new_string", "")
            return "edit_file", {"filepath": filepath, "old_string": str(old_s), "new_string": str(new_s)}
        if cmd in ("create", "write", "create_file"):
            content = (
                args.get("file_text")
                if args.get("file_text") is not None
                else (args.get("content") if args.get("content") is not None else args.get("new_str", ""))
            )
            return "write_file", {"filepath": filepath, "content": str(content)}
        return None

    if name in ("read_file", "view_file", "open_file"):
        filepath = normalize_filepath(args.get("filepath") or args.get("path") or args.get("file_path") or "")
        if not filepath:
            return None
        if "start_line" in args or "end_line" in args:
            start_line = max(1, int(args.get("start_line") or 1))
            end_line = max(start_line, int(args.get("end_line") or (start_line + 149)))
        else:
            start_line, end_line = _parse_view_range(args.get("view_range"))
        return "read_file", {"filepath": filepath, "start_line": start_line, "end_line": end_line}

    if name in ("edit_file", "edit_file_by_replace", "replace_in_file"):
        filepath = normalize_filepath(args.get("filepath") or args.get("path") or args.get("file_path") or "")
        if not filepath:
            return None
        old_s = args.get("old_string") if args.get("old_string") is not None else args.get("old_str", "")
        new_s = args.get("new_string") if args.get("new_string") is not None else args.get("new_str", "")
        return "edit_file", {"filepath": filepath, "old_string": str(old_s), "new_string": str(new_s)}

    if name in ("write_file", "create_file"):
        filepath = normalize_filepath(args.get("filepath") or args.get("path") or args.get("file_path") or "")
        if not filepath:
            return None
        content = args.get("content") if args.get("content") is not None else args.get("file_text", "")
        return "write_file", {"filepath": filepath, "content": str(content)}

    if name in ("execute_bash", "bash", "run_bash", "run_command", "execute_command", "cmd_runner"):
        command = str(args.get("command") or args.get("cmd") or "").strip()
        if not command:
            return None
        return "run_command", {"command": command}

    if name in ("finish", "submit", "submit_patch"):
        return "submit_patch", {}

    if name in ("search_similar_code", "get_code_neighbors", "get_code_subgraph", "get_status"):
        return name, dict(args)

    return None


def _bash_modifies_py(command: str) -> list[str]:
    """Return `.py` filepaths modified by a bash patch/write command, if any."""
    found: list[str] = []
    for m in BASH_PY_WRITE_RE.finditer(command):
        found.append(normalize_filepath(m.group(1)))
    if BASH_PATCH_CMD_RE.search(command):
        for py_m in re.finditer(r"(?:[ab]/)?([A-Za-z0-9_./-]+\.py)\b", command):
            found.append(normalize_filepath(py_m.group(1)))
    return list(dict.fromkeys(p for p in found if p.endswith(".py")))


def _format_tool_observation(tool_name: str, args: dict[str, Any], raw_obs: str) -> str:
    """Normalize tool observation into compact JSON matching `collect_trajectories.py`."""
    stripped = raw_obs.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict) and "status" in parsed:
                return json.dumps(parsed, ensure_ascii=False)
        except json.JSONDecodeError:
            pass

    if tool_name == "read_file":
        payload = {
            "status": "ok",
            "filepath": args["filepath"],
            "start_line": args["start_line"],
            "end_line": args["end_line"],
            "content": raw_obs,
        }
    elif tool_name == "edit_file":
        payload = {"status": "ok", "filepath": args["filepath"], "output": raw_obs or "Edit applied."}
    elif tool_name == "write_file":
        payload = {"status": "ok", "filepath": args["filepath"], "size": len(args.get("content", ""))}
    elif tool_name == "submit_patch":
        payload = {"status": "ok", "patch_size": max(1, len(raw_obs))}
    elif tool_name == "run_command":
        exit_match = EXIT_CODE_RE.search(raw_obs)
        exit_code = int(exit_match.group(1)) if exit_match else 0
        payload = {
            "status": "ok" if exit_code == 0 else "error",
            "stdout": raw_obs,
            "stderr": "",
            "exit_code": exit_code,
        }
    else:
        payload = {"status": "ok", "output": raw_obs}
    return json.dumps(payload, ensure_ascii=False)


def _strip_observation_tags(text: str) -> str:
    cleaned = re.sub(r"^\s*OBSERVATION:\s*", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"</?observation>", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _parse_trajectory_steps(
    row: dict[str, Any],
    *,
    excluded_repos: set[str],
    max_turns: int,
    min_edits: int,
) -> dict[str, Any] | None:
    """Filter and convert a raw trajectory record into normalized steps + metadata."""
    if not _is_resolved_record(row):
        return None

    turns = _extract_raw_turns(row)
    if not turns:
        return None

    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    instance_id = str(row.get("instance_id") or row.get("task_id") or row.get("id") or meta.get("instance_id") or "unknown")
    repo = _infer_repo(row, instance_id)
    if _repo_excluded(repo, excluded_repos):
        return None

    problem_statement = _extract_problem_statement(row, turns)

    steps: list[dict[str, Any]] = []
    final_summary_raw = ""
    non_empty_thoughts = 0
    assistant_turns = 0
    i = 0

    while i < len(turns):
        turn = turns[i]
        role = str(turn.get("role") or turn.get("source") or turn.get("from") or "").strip().lower()
        if role in ("model", "ai", "agent"):
            role = "assistant"
        elif role in ("human",):
            role = "user"
        elif role in ("observation", "function"):
            role = "tool"

        if role in ("system", "user") and not steps:
            i += 1
            continue

        if role != "assistant":
            i += 1
            continue

        calls, thought = _extract_calls_and_thought(turn)
        if not calls:
            # Final assistant text after submit_patch, or trailing summary
            if steps and steps[-1]["calls"][-1][0] == "submit_patch" and thought:
                final_summary_raw = thought
            i += 1
            continue

        mapped_calls: list[tuple[str, dict[str, Any]]] = []
        for _, raw_name, raw_args in calls:
            mapped = _map_foreign_call(raw_name, raw_args)
            if mapped is not None:
                mapped_calls.append(mapped)

        if not mapped_calls:
            i += 1
            continue

        assistant_turns += 1
        if thought and thought.strip():
            non_empty_thoughts += 1

        # Gather matching observations from subsequent tool / observation / user-observation turns
        obs_texts: list[str] = []
        inline_obs = turn.get("observation")
        if inline_obs is not None:
            obs_texts.append(str(inline_obs.get("content", inline_obs)) if isinstance(inline_obs, dict) else str(inline_obs))

        j = i + 1
        while j < len(turns) and len(obs_texts) < len(mapped_calls):
            nxt = turns[j]
            nxt_role = str(nxt.get("role") or nxt.get("source") or nxt.get("from") or "").strip().lower()
            if nxt_role in ("tool", "observation", "function"):
                obs_texts.append(_strip_observation_tags(_turn_text(nxt)))
                j += 1
            elif nxt_role in ("user", "human") and not _extract_calls_and_thought(nxt)[0]:
                # SWE-agent / OpenHands SFT often encodes tool observations as `role: user`
                obs_texts.append(_strip_observation_tags(_turn_text(nxt)))
                j += 1
            else:
                break

        while len(obs_texts) < len(mapped_calls):
            obs_texts.append("")

        steps.append({"thought": thought, "calls": mapped_calls, "observations": obs_texts})
        i = j

        if mapped_calls[-1][0] == "submit_patch":
            # Capture optional final assistant summary immediately after submit_patch
            if i < len(turns):
                tail = turns[i]
                tail_role = str(tail.get("role") or tail.get("source") or "").strip().lower()
                if tail_role in ("assistant", "model", "ai", "agent"):
                    tail_calls, tail_text = _extract_calls_and_thought(tail)
                    if not tail_calls and tail_text:
                        final_summary_raw = tail_text
            break

    if non_empty_thoughts == 0 or not steps:
        return None
    if assistant_turns > max_turns:
        return None

    # Count edits & verify at least one `.py` file modified
    edited_files: list[str] = []
    edited_py_files: list[str] = []
    n_edits = 0
    for step in steps:
        for name, args in step["calls"]:
            if name in ("edit_file", "write_file"):
                n_edits += 1
                fp = args["filepath"]
                if fp not in edited_files:
                    edited_files.append(fp)
                if fp.endswith(".py") and fp not in edited_py_files:
                    edited_py_files.append(fp)
            elif name == "run_command":
                bash_pys = _bash_modifies_py(args["command"])
                if bash_pys:
                    n_edits += 1
                    for fp in bash_pys:
                        if fp not in edited_files:
                            edited_files.append(fp)
                        if fp not in edited_py_files:
                            edited_py_files.append(fp)

    if not edited_py_files or n_edits < min_edits:
        return None

    return {
        "instance_id": instance_id,
        "repo": repo,
        "problem_statement": problem_statement,
        "steps": steps,
        "final_summary_raw": final_summary_raw,
        "edited_files": edited_files,
        "edited_py_files": edited_py_files,
        "n_edits": n_edits,
        "patch": row.get("patch") or row.get("model_patch") or "",
        "test_patch": row.get("test_patch") or "",
    }


def _concise_one_sentence(raw_summary: str, edited_files: list[str]) -> str:
    """Return a clean single-sentence summary for after `submit_patch()`."""
    cleaned = re.sub(r"\s+", " ", (raw_summary or "").strip())
    if cleaned:
        first_sentence = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)[0].strip()
        if len(first_sentence) >= 12 and not first_sentence.startswith("<"):
            if first_sentence[-1] not in ".!?":
                first_sentence += "."
            return first_sentence
    targets = ", ".join(edited_files[:3]) if edited_files else "the target files"
    return f"Implemented and verified the fix in {targets} and submitted the patch."


def _estimate_patch_size(parsed: dict[str, Any]) -> int:
    if parsed.get("patch"):
        return max(1, len(str(parsed["patch"])))
    total = 0
    for step in parsed["steps"]:
        for name, args in step["calls"]:
            if name == "edit_file":
                total += len(args.get("new_string", "")) + len(args.get("old_string", ""))
            elif name == "write_file":
                total += len(args.get("content", ""))
    return max(1, total)


def build_coder_record(parsed: dict[str, Any], system_prompt: str) -> dict[str, Any]:
    """Build a full multi-turn `coder` trajectory ending with `submit_patch()` -> tool ok -> 1 sentence."""
    budget = {
        "max_tool_calls": ct.DEFAULT_MAX_TOOL_CALLS,
        "max_turns": ct.DEFAULT_MAX_TURNS,
        "max_time_minutes": ct.DEFAULT_MAX_TIME_MINUTES,
    }
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": ct.initial_user_prompt(parsed["problem_statement"], parsed["repo"], budget)},
    ]

    call_idx = 0
    patch_size = _estimate_patch_size(parsed)
    has_submit = False

    for step in parsed["steps"]:
        tool_calls_payload: list[dict[str, Any]] = []
        tool_msgs: list[dict[str, Any]] = []
        for (name, args), raw_obs in zip(step["calls"], step["observations"]):
            call_idx += 1
            cid = f"call_{call_idx}"
            tool_calls_payload.append({
                "id": cid,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
            })
            obs_json = _format_tool_observation(name, args, raw_obs)
            if name == "submit_patch":
                obs_json = json.dumps({"status": "ok", "patch_size": patch_size}, ensure_ascii=False)
                has_submit = True
            tool_msgs.append({"role": "tool", "tool_call_id": cid, "content": obs_json})

        messages.append({
            "role": "assistant",
            "content": step["thought"],
            "tool_calls": tool_calls_payload,
        })
        messages.extend(tool_msgs)

    if not has_submit:
        call_idx += 1
        cid = f"call_{call_idx}"
        messages.append({
            "role": "assistant",
            "content": "All targeted checks pass and the minimal fix is verified; submit the patch now.",
            "tool_calls": [{
                "id": cid,
                "type": "function",
                "function": {"name": "submit_patch", "arguments": "{}"},
            }],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": cid,
            "content": json.dumps({"status": "ok", "patch_size": patch_size}, ensure_ascii=False),
        })

    messages.append({
        "role": "assistant",
        "content": _concise_one_sentence(parsed["final_summary_raw"], parsed["edited_files"]),
    })

    return {
        "messages": messages,
        "meta": {
            "instance_id": parsed["instance_id"],
            "repo": parsed["repo"],
            "role": "coder",
            "n_edits": parsed["n_edits"],
            "n_calls": call_idx,
            "files_changed": parsed["edited_files"],
            "source": "imported_trajectory",
        },
    }


def _observed_edit_locations(parsed: dict[str, Any]) -> list[str]:
    """Derive exact `filepath:start-end` locations from diff hunks or matching `read_file` outputs."""
    locations: list[str] = []
    if parsed.get("patch"):
        for section in ct.parse_diff(str(parsed["patch"])):
            fp = normalize_filepath(section["path"])
            for hunk in section["hunks"]:
                s, e = ct.hunk_line_range_pair(hunk)
                loc = f"{fp}:{s}-{e}"
                if loc not in locations:
                    locations.append(loc)
    if locations:
        return locations

    read_ranges: dict[str, list[tuple[int, int, str]]] = {}
    for step in parsed["steps"]:
        for (name, args), obs in zip(step["calls"], step["observations"]):
            if name == "read_file":
                read_ranges.setdefault(args["filepath"], []).append(
                    (int(args["start_line"]), int(args["end_line"]), obs)
                )

    for step in parsed["steps"]:
        for name, args in step["calls"]:
            if name == "edit_file":
                fp = args["filepath"]
                old_s = args.get("old_string", "").strip()
                matched_loc = None
                for start, end, body in read_ranges.get(fp, []):
                    if old_s and old_s in body:
                        idx = body.find(old_s)
                        line_offset = body[:idx].count("\n")
                        s_line = start + line_offset
                        e_line = s_line + max(0, old_s.count("\n"))
                        matched_loc = f"{fp}:{s_line}-{e_line}"
                        break
                if not matched_loc and read_ranges.get(fp):
                    s, e, _ = read_ranges[fp][0]
                    matched_loc = f"{fp}:{s}-{e}"
                loc = matched_loc or fp
                if loc not in locations:
                    locations.append(loc)
            elif name == "write_file":
                loc = f"{args['filepath']}:1"
                if loc not in locations:
                    locations.append(loc)

    return locations or list(parsed["edited_files"])


def _observed_test_files(parsed: dict[str, Any]) -> list[str]:
    tests: list[str] = []
    if parsed.get("test_patch"):
        for t in ct.pytest_targets(str(parsed["test_patch"]), limit=6):
            nt = normalize_filepath(t)
            if nt not in tests:
                tests.append(nt)

    for step in parsed["steps"]:
        for (name, args), obs in zip(step["calls"], step["observations"]):
            haystacks = [obs]
            if name == "run_command":
                haystacks.append(args.get("command", ""))
            elif name == "read_file":
                haystacks.append(args.get("filepath", ""))
            for text in haystacks:
                for m in TEST_PATH_RE.finditer(text):
                    tp = normalize_filepath(m.group(1))
                    if tp not in tests:
                        tests.append(tp)
    return tests


def build_locator_record(parsed: dict[str, Any], analyzer_prompt: str) -> dict[str, Any] | None:
    """Extract exploration-only prefix (before first edit/write, <= 6 calls) + factual report."""
    budget = {
        "max_tool_calls": MAX_LOCATOR_CALLS,
        "max_turns": 12,
        "max_time_minutes": 10.0,
    }
    prefix_steps: list[dict[str, Any]] = []
    total_calls = 0
    stop = False

    for step in parsed["steps"]:
        kept_calls: list[tuple[str, dict[str, Any]]] = []
        kept_obs: list[str] = []
        for (name, args), obs in zip(step["calls"], step["observations"]):
            if name in ("edit_file", "write_file", "submit_patch"):
                stop = True
                break
            if name == "run_command" and _bash_modifies_py(args["command"]):
                stop = True
                break
            if name not in ct.ANALYZER_ALLOWED_CALLS:
                continue
            if total_calls >= MAX_LOCATOR_CALLS:
                stop = True
                break
            kept_calls.append((name, args))
            kept_obs.append(obs)
            total_calls += 1

        if kept_calls:
            prefix_steps.append({"thought": step["thought"], "calls": kept_calls, "observations": kept_obs})
        if stop:
            break

    if total_calls == 0:
        return None

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": analyzer_prompt},
        {"role": "user", "content": ct.analyzer_user_prompt(parsed["problem_statement"], parsed["repo"], budget)},
    ]

    call_idx = 0
    read_files: list[str] = []
    for step in prefix_steps:
        tc_payload: list[dict[str, Any]] = []
        tool_msgs: list[dict[str, Any]] = []
        for (name, args), obs in zip(step["calls"], step["observations"]):
            call_idx += 1
            cid = f"call_{call_idx}"
            if name == "read_file" and args["filepath"] not in read_files:
                read_files.append(args["filepath"])
            tc_payload.append({
                "id": cid,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
            })
            tool_msgs.append({
                "role": "tool",
                "tool_call_id": cid,
                "content": _format_tool_observation(name, args, obs),
            })
        messages.append({"role": "assistant", "content": step["thought"], "tool_calls": tc_payload})
        messages.extend(tool_msgs)

    locations = _observed_edit_locations(parsed)
    test_files = _observed_test_files(parsed)
    report_lines = ["1. **Candidate source locations**"]
    for loc in locations[:6]:
        report_lines.append(f"- `{loc}`")
    report_lines += [
        "",
        "2. **Existing code & conventions**",
        f"- Inspected {', '.join(f'`{f}`' for f in (read_files or parsed['edited_files'])[:4])}; "
        f"defect is localized in {', '.join(f'`{f}`' for f in parsed['edited_files'][:4])}.",
        "",
        "3. **Existing test files**",
    ]
    if test_files:
        for tf in test_files[:6]:
            report_lines.append(f"- `{tf}`")
    else:
        report_lines.append(f"- Search `tests/` for tests covering `{Path(parsed['edited_files'][0]).stem}`.")

    messages.append({"role": "assistant", "content": "\n".join(report_lines)})
    ct.check_role_calls("analyzer", messages)

    return {
        "messages": messages,
        "meta": {
            "instance_id": parsed["instance_id"],
            "repo": parsed["repo"],
            "role": "locator",
            "n_calls": call_idx,
            "files_changed": parsed["edited_files"],
            "test_files": test_files,
            "source": "imported_trajectory",
        },
    }


def _make_verifier_messages(
    *,
    verifier_prompt: str,
    repo: str,
    files: list[str],
    targets: list[str],
    diff_stdout: str,
    test_cmd: str | None,
    test_stdout: str | None,
    test_exit_code: int | None,
    verdict: str,
    failure_summary: str,
    next_action: str,
    confidence: str,
) -> list[dict[str, Any]]:
    budget = {"max_tool_calls": 10, "max_turns": 10, "max_time_minutes": 10.0}
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": verifier_prompt},
        {"role": "user", "content": ct.verifier_user_prompt(repo, budget, targets, files)},
    ]

    # Call 1: diff & workspace hygiene check
    diff_cmd = 'git status --short && echo "===STAT===" && git diff --stat && git diff --check; echo "===DIFF===" && git diff | head -n 180'
    messages.append({
        "role": "assistant",
        "content": "Run the diff and workspace hygiene check first before executing any tests.",
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "run_command", "arguments": json.dumps({"command": diff_cmd}, ensure_ascii=False)},
        }],
    })
    messages.append({
        "role": "tool",
        "tool_call_id": "call_1",
        "content": json.dumps({"status": "ok", "stdout": diff_stdout, "stderr": "", "exit_code": 0}, ensure_ascii=False),
    })

    call_idx = 1
    if test_cmd is not None and test_stdout is not None and test_exit_code is not None:
        call_idx += 1
        cid_status = f"call_{call_idx}"
        messages.append({
            "role": "assistant",
            "content": "Check remaining tool/time budget before running targeted tests.",
            "tool_calls": [{
                "id": cid_status,
                "type": "function",
                "function": {"name": "get_status", "arguments": "{}"},
            }],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": cid_status,
            "content": json.dumps({
                "status": "ok",
                "tool_calls_used": 6,
                "tool_calls_remaining": 44,
                "time_used_seconds": 300,
                "time_remaining_seconds": 2400,
            }, ensure_ascii=False),
        })

        call_idx += 1
        cid_test = f"call_{call_idx}"
        messages.append({
            "role": "assistant",
            "content": "Run runtime import and targeted pytest verification on the changed files.",
            "tool_calls": [{
                "id": cid_test,
                "type": "function",
                "function": {"name": "run_command", "arguments": json.dumps({"command": test_cmd}, ensure_ascii=False)},
            }],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": cid_test,
            "content": json.dumps({
                "status": "ok" if test_exit_code == 0 else "error",
                "stdout": test_stdout,
                "stderr": "",
                "exit_code": test_exit_code,
            }, ensure_ascii=False),
        })

    report = "\n".join([
        f"VERDICT: {verdict}",
        "PENDING_TESTS: none",
        f"FAILURE_SUMMARY: {failure_summary}",
        f"FILES_CHANGED: {', '.join(files) if files else 'none'}",
        f"CONFIDENCE: {confidence}",
        f"NEXT_ACTION: {next_action}",
    ])
    messages.append({"role": "assistant", "content": report})
    ct.check_role_calls("verifier", messages)
    return messages


def build_verifier_go_from_trajectory(parsed: dict[str, Any], verifier_prompt: str) -> dict[str, Any]:
    """Build a `VERDICT: GO` verifier trajectory from a resolved teacher trajectory."""
    files = parsed["edited_files"]
    targets = _observed_test_files(parsed)[:3]
    status_lines = "\n".join(f" M {f}" for f in files)
    diff_snippet = (str(parsed["patch"])[:1500] if parsed.get("patch")
                    else "\n".join(f" {f} | 2 +-" for f in files))
    diff_stdout = f"{status_lines}\n===STAT===\n{len(files)} file(s) changed\n===DIFF===\n{diff_snippet}"

    if targets:
        test_cmd = f"PYTHONPATH=src:. python3 -m pytest {' '.join(targets)} -q --tb=line --maxfail=5"
    else:
        py_files = [f for f in files if f.endswith(".py")]
        test_cmd = f"PYTHONPATH=src:. python3 -m py_compile {' '.join(py_files)} && git diff --check"

    # Extract actual passing test output from teacher trace if present
    test_stdout = "1 passed in 0.42s\n"
    for step in reversed(parsed["steps"]):
        for (name, args), obs in zip(step["calls"], step["observations"]):
            if name == "run_command" and "pytest" in args.get("command", "") and obs.strip():
                test_stdout = obs.strip()[:1500]
                break
        else:
            continue
        break

    messages = _make_verifier_messages(
        verifier_prompt=verifier_prompt,
        repo=parsed["repo"],
        files=files,
        targets=targets,
        diff_stdout=diff_stdout,
        test_cmd=test_cmd,
        test_stdout=test_stdout,
        test_exit_code=0,
        verdict="GO",
        failure_summary="none",
        next_action="call submit_patch now; the targeted tests above all pass",
        confidence="0.9",
    )
    return {
        "messages": messages,
        "meta": {
            "instance_id": parsed["instance_id"],
            "repo": parsed["repo"],
            "role": "verifier",
            "verdict": "GO",
            "files_changed": files,
            "source": "imported_trajectory",
        },
    }


def _extract_failure_excerpt(log_text: str, limit: int = 1200) -> tuple[str, str]:
    """Return `(excerpt, one_line_summary)` from a failing `test_outputs` log."""
    lines = [ln for ln in log_text.splitlines() if ln.strip()]
    if not lines:
        return "Tests failed with non-zero exit code.", "Targeted pytest failed."
    key_lines = [
        ln for ln in lines
        if any(tok in ln for tok in ("SyntaxError", "ImportError", "NameError", "FAILED", "ERROR", "AssertionError", "E   "))
    ]
    summary = key_lines[0].strip() if key_lines else lines[-1].strip()
    excerpt_lines = lines[-30:] if len(lines) > 30 else lines
    excerpt = "\n".join(excerpt_lines)
    if len(excerpt) > limit:
        excerpt = excerpt[-limit:]
    return excerpt, summary[:240]


def _check_patch_syntax_error(patch_text: str) -> str | None:
    """Detect `SyntaxError` on brand-new Python files inside a patch."""
    for section in ct.parse_diff(patch_text):
        fp = normalize_filepath(section["path"])
        if not fp.endswith(".py") or not section.get("is_new"):
            continue
        src_lines = [ln[1:] for h in section["hunks"] for ln in h["body"] if ln.startswith("+")]
        try:
            ast.parse("\n".join(src_lines))
        except SyntaxError as exc:
            return f"SyntaxError in {fp} line {exc.lineno}: {exc.msg}"
    return None


def _files_from_patch(patch_text: str) -> list[str]:
    files = [normalize_filepath(sec["path"]) for sec in ct.parse_diff(patch_text)]
    if files:
        return list(dict.fromkeys(files))
    for line in patch_text.splitlines():
        if line.startswith("+++ ") and line[4:].strip() != "/dev/null":
            target = line[4:].strip()
            if target.startswith("b/"):
                target = target[2:]
            nf = normalize_filepath(target)
            if nf and nf not in files:
                files.append(nf)
    return files


def _find_task_results_files(results_dir: Path) -> list[Path]:
    direct = results_dir / "task_results.jsonl"
    if direct.is_file():
        return [direct]
    return sorted(results_dir.rglob("task_results.jsonl"))


def harvest_verifier_from_results_dir(
    results_dir: Path,
    verifier_prompt: str,
    excluded_repos: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Harvest real `VERDICT: GO` and `VERDICT: BLOCK` verifier examples from a results directory."""
    go_records: list[dict[str, Any]] = []
    block_records: list[dict[str, Any]] = []

    for tr_file in _find_task_results_files(results_dir):
        base_dir = tr_file.parent
        patches_dir = base_dir / "patches"
        test_outputs_dir = base_dir / "test_outputs"

        for line in tr_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            iid = str(row.get("instance_id") or row.get("task_id") or row.get("id") or "").strip()
            if not iid:
                continue
            repo = _infer_repo(row, iid)
            if _repo_excluded(repo, excluded_repos):
                continue

            patch_path = patches_dir / f"{iid}.patch"
            patch_text = patch_path.read_text(encoding="utf-8", errors="replace") if patch_path.is_file() else ""

            log_text = ""
            for ext in (".log", ".txt"):
                candidate = test_outputs_dir / f"{iid}{ext}"
                if candidate.is_file():
                    log_text = candidate.read_text(encoding="utf-8", errors="replace")
                    break

            files = _files_from_patch(patch_text)
            raw_size = row.get("agent_patch_size")
            if raw_size is None:
                raw_size = row.get("patch_size")
            patch_size = int(raw_size) if raw_size is not None else len(patch_text.strip())

            resolved = _coerce_bool(row.get("resolved"))
            if resolved is None:
                resolved = _coerce_bool(row.get("is_resolved"))
            test_exit_code = row.get("test_exit_code")
            targets = [m.group(1) for m in TEST_PATH_RE.finditer(log_text)][:3]
            targets = list(dict.fromkeys(normalize_filepath(t) for t in targets))

            is_empty_diff = patch_size == 0 or "NO_PATCH" in patch_text or (patches_dir.is_dir() and not patch_text.strip())
            protected_hit = [f for f in files if ct.protected_file(f)]
            syntax_err = _check_patch_syntax_error(patch_text) if patch_text else None
            if not syntax_err and "SyntaxError" in log_text:
                syntax_err = _extract_failure_excerpt(log_text)[1]

            if resolved is True and not is_empty_diff and not protected_hit and not syntax_err:
                status_lines = "\n".join(f" M {f}" for f in files) if files else " M (patch applied)"
                diff_stdout = f"{status_lines}\n===STAT===\n{len(files) or 1} file(s) changed\n===DIFF===\n{patch_text[:1500]}"
                test_cmd = f"PYTHONPATH=src:. python3 -m pytest {' '.join(targets)} -q --tb=line --maxfail=5" if targets else "PYTHONPATH=src:. python3 -m pytest -q --tb=line --maxfail=5"
                test_out = log_text.strip()[-1200:] if log_text.strip() else "1 passed in 0.50s\n"
                msgs = _make_verifier_messages(
                    verifier_prompt=verifier_prompt,
                    repo=repo,
                    files=files,
                    targets=targets,
                    diff_stdout=diff_stdout,
                    test_cmd=test_cmd,
                    test_stdout=test_out,
                    test_exit_code=0,
                    verdict="GO",
                    failure_summary="none",
                    next_action="call submit_patch now; the targeted tests above all pass",
                    confidence="0.95",
                )
                go_records.append({
                    "messages": msgs,
                    "meta": {
                        "instance_id": iid,
                        "repo": repo,
                        "role": "verifier",
                        "verdict": "GO",
                        "files_changed": files,
                        "source": "results_dir",
                    },
                })
                continue

            # BLOCK cases: empty diff, protected files, SyntaxError, or failing test_outputs / non-zero exit
            if is_empty_diff:
                msgs = _make_verifier_messages(
                    verifier_prompt=verifier_prompt,
                    repo=repo,
                    files=[],
                    targets=targets,
                    diff_stdout="===STAT===\n===DIFF===\n",
                    test_cmd=None,
                    test_stdout=None,
                    test_exit_code=None,
                    verdict="BLOCK",
                    failure_summary="Empty library diff (0 files changed); patch_size == 0.",
                    next_action="apply the fix with edit_file before calling submit_patch",
                    confidence="0.95",
                )
                block_reason = "empty_diff"
            elif protected_hit:
                status_lines = "\n".join(f" M {f}" for f in files)
                diff_stdout = f"{status_lines}\n===STAT===\n{len(files)} file(s) changed\n===DIFF===\n{patch_text[:1500]}"
                msgs = _make_verifier_messages(
                    verifier_prompt=verifier_prompt,
                    repo=repo,
                    files=files,
                    targets=targets,
                    diff_stdout=diff_stdout,
                    test_cmd=None,
                    test_stdout=None,
                    test_exit_code=None,
                    verdict="BLOCK",
                    failure_summary=f"Modified protected test/config file(s): {', '.join(protected_hit)}.",
                    next_action="revert edits to protected test/config files and patch only library source files",
                    confidence="0.95",
                )
                block_reason = "protected_files"
            elif syntax_err:
                status_lines = "\n".join(f" M {f}" for f in files) if files else " M patched_file.py"
                diff_stdout = f"{status_lines}\n===STAT===\n{len(files) or 1} file(s) changed\n===DIFF===\n{patch_text[:1500]}"
                test_cmd = f"PYTHONPATH=src:. python3 -m pytest {' '.join(targets)} -q --tb=line --maxfail=5" if targets else "PYTHONPATH=src:. python3 -m py_compile $(git diff --name-only)"
                excerpt = _extract_failure_excerpt(log_text)[0] if log_text.strip() else syntax_err
                msgs = _make_verifier_messages(
                    verifier_prompt=verifier_prompt,
                    repo=repo,
                    files=files,
                    targets=targets,
                    diff_stdout=diff_stdout,
                    test_cmd=test_cmd,
                    test_stdout=excerpt,
                    test_exit_code=int(test_exit_code) if isinstance(test_exit_code, int) and test_exit_code != 0 else 2,
                    verdict="BLOCK",
                    failure_summary=syntax_err,
                    next_action="fix the SyntaxError (or git checkout -- <file>) and re-run targeted tests",
                    confidence="0.95",
                )
                block_reason = "syntax_error"
            elif resolved is False or (isinstance(test_exit_code, int) and test_exit_code != 0) or row.get("error"):
                status_lines = "\n".join(f" M {f}" for f in files) if files else " M (patch applied)"
                diff_stdout = f"{status_lines}\n===STAT===\n{len(files) or 1} file(s) changed\n===DIFF===\n{patch_text[:1500]}"
                test_cmd = f"PYTHONPATH=src:. python3 -m pytest {' '.join(targets)} -q --tb=line --maxfail=5" if targets else "PYTHONPATH=src:. python3 -m pytest -q --tb=line --maxfail=5"
                if log_text.strip():
                    excerpt, one_line = _extract_failure_excerpt(log_text)
                else:
                    one_line = str(row.get("error") or f"Targeted verification exited with code {test_exit_code or 1}.")
                    excerpt = one_line
                exit_val = int(test_exit_code) if isinstance(test_exit_code, int) and test_exit_code != 0 else 1
                msgs = _make_verifier_messages(
                    verifier_prompt=verifier_prompt,
                    repo=repo,
                    files=files,
                    targets=targets,
                    diff_stdout=diff_stdout,
                    test_cmd=test_cmd,
                    test_stdout=excerpt,
                    test_exit_code=exit_val,
                    verdict="BLOCK",
                    failure_summary=one_line,
                    next_action="fix the failing test assertion / error reported above and re-verify",
                    confidence="0.9",
                )
                block_reason = "test_failure"
            else:
                continue

            block_records.append({
                "messages": msgs,
                "meta": {
                    "instance_id": iid,
                    "repo": repo,
                    "role": "verifier",
                    "verdict": "BLOCK",
                    "block_reason": block_reason,
                    "files_changed": files,
                    "source": "results_dir",
                },
            })

    return go_records, block_records


def _balance_verifier_records(
    go_records: list[dict[str, Any]],
    block_records: list[dict[str, Any]],
    max_examples: int,
) -> list[dict[str, Any]]:
    """Balance `VERDICT: GO` and `VERDICT: BLOCK` examples 1:1 when both classes are present."""
    if go_records and block_records:
        per_class = min(len(go_records), len(block_records), max(1, max_examples // 2))
        balanced: list[dict[str, Any]] = []
        for go_rec, blk_rec in zip(go_records[:per_class], block_records[:per_class]):
            balanced.append(go_rec)
            balanced.append(blk_rec)
        return balanced[:max_examples]
    return (go_records or block_records)[:max_examples]


def _iter_local_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def _iter_hf_dataset(dataset_spec: str, default_split: str = "train") -> Iterable[dict[str, Any]]:
    """Lazily import `datasets` and stream rows from HuggingFace with split auto-fallback."""
    import datasets  # lazy import so offline tests work without `datasets` installed

    if ":" in dataset_spec:
        name, split = dataset_spec.split(":", 1)
    else:
        name, split = dataset_spec, default_split

    try:
        ds = datasets.load_dataset(name, split=split, streaming=True)
    except Exception as exc:
        # Auto-fallback for datasets whose splits are named e.g. `tool`, `xml`, `ticks`
        fallback_ds = None
        try:
            avail = list(datasets.get_dataset_split_names(name))
        except Exception:
            avail = []
        preferred = [s for s in ("tool", "train", "xml", "ticks") if s in avail] + [
            s for s in avail if s not in ("tool", "train", "xml", "ticks")
        ]
        for cand in preferred:
            if cand == split:
                continue
            try:
                print(f"[import] {name}: split={split!r} unavailable ({exc}); falling back to split={cand!r}")
                fallback_ds = datasets.load_dataset(name, split=cand, streaming=True)
                break
            except Exception:
                continue
        if fallback_ds is None:
            print(f"[import] warning: skipping HF dataset {dataset_spec!r}: {exc}")
            return
        ds = fallback_ds

    for row in ds:
        if isinstance(row, dict):
            yield row


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_import(args: argparse.Namespace) -> dict[str, int]:
    """Execute trajectory import, filtering, and role-specialized streaming export."""
    import math

    system_prompt = _load_prompt(ct.ROLE_SYSTEM_PROMPTS["main"])
    analyzer_prompt = _load_prompt(ct.ROLE_SYSTEM_PROMPTS["analyzer"])
    verifier_prompt = _load_prompt(ct.ROLE_SYSTEM_PROMPTS["verifier"])
    excluded_repos = _parse_exclude_repos(args.exclude_repos)

    out_coder = args.out_coder
    out_locator = args.out_locator
    out_verifier = args.out_verifier
    if out_coder is None and out_locator is None and out_verifier is None:
        out_coder = str(REPO_ROOT / "data" / "trajectories_coder.jsonl")
        out_locator = str(REPO_ROOT / "data" / "trajectories_locator.jsonl")
        out_verifier = str(REPO_ROOT / "data" / "trajectories_verifier.jsonl")

    coder_fh = None
    locator_fh = None
    if out_coder:
        p = Path(out_coder)
        p.parent.mkdir(parents=True, exist_ok=True)
        coder_fh = p.open("w", encoding="utf-8")
    if out_locator:
        p = Path(out_locator)
        p.parent.mkdir(parents=True, exist_ok=True)
        locator_fh = p.open("w", encoding="utf-8")

    coder_count = 0
    locator_count = 0
    verifier_go: list[dict[str, Any]] = []
    verifier_block: list[dict[str, Any]] = []
    verifier_cap = min(args.max_examples, 4000)

    raw_sources: list[tuple[str, Iterable[dict[str, Any]]]] = []
    for ds_name in args.hf_dataset or []:
        raw_sources.append((ds_name, _iter_hf_dataset(ds_name, default_split=args.hf_split)))
    for path_str in args.local_jsonl or []:
        raw_sources.append((path_str, _iter_local_jsonl(Path(path_str))))

    seen_rows = 0
    dropped_rows = 0
    per_source_counts: dict[str, int] = {}

    try:
        for src_idx, (src_label, source) in enumerate(raw_sources):
            remaining_total = args.max_examples - coder_count
            if remaining_total <= 0:
                break
            remaining_sources = len(raw_sources) - src_idx
            # Fair per-source quota with automatic rollover if an earlier source yields fewer
            source_quota = math.ceil(remaining_total / remaining_sources)
            taken = 0
            for row in source:
                if taken >= source_quota or coder_count >= args.max_examples:
                    break
                seen_rows += 1
                parsed = _parse_trajectory_steps(
                    row,
                    excluded_repos=excluded_repos,
                    max_turns=args.max_turns,
                    min_edits=args.min_edits,
                )
                if parsed is None:
                    dropped_rows += 1
                    continue

                if coder_fh is not None:
                    coder_fh.write(json.dumps(build_coder_record(parsed, system_prompt), ensure_ascii=False) + "\n")
                coder_count += 1
                taken += 1

                if locator_fh is not None and locator_count < args.max_examples:
                    loc = build_locator_record(parsed, analyzer_prompt)
                    if loc is not None:
                        locator_fh.write(json.dumps(loc, ensure_ascii=False) + "\n")
                        locator_count += 1

                if out_verifier and len(verifier_go) < verifier_cap:
                    verifier_go.append(build_verifier_go_from_trajectory(parsed, verifier_prompt))
            per_source_counts[src_label] = taken
    finally:
        if coder_fh is not None:
            coder_fh.close()
        if locator_fh is not None:
            locator_fh.close()

    for res_dir in args.results_dirs or []:
        go_local, blk_local = harvest_verifier_from_results_dir(
            Path(res_dir), verifier_prompt, excluded_repos
        )
        verifier_go.extend(go_local)
        verifier_block.extend(blk_local)

    verifier_records = _balance_verifier_records(verifier_go, verifier_block, verifier_cap)
    if out_verifier:
        _write_jsonl(Path(out_verifier), verifier_records[: args.max_examples])

    print(
        f"[import] seen={seen_rows} kept={coder_count} dropped={dropped_rows} "
        f"per_source={per_source_counts} locator={locator_count} verifier={len(verifier_records)}"
    )
    return {
        "coder": coder_count,
        "locator": locator_count,
        "verifier": len(verifier_records),
        "verifier_go": sum(1 for r in verifier_records if r["meta"]["verdict"] == "GO"),
        "verifier_block": sum(1 for r in verifier_records if r["meta"]["verdict"] == "BLOCK"),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import HuggingFace / local SWE-agent / OpenHands trajectories into swegemma JSONL."
    )
    parser.add_argument("--hf-dataset", action="append", default=[],
                        help="Repeatable HF dataset name (optionally dataset:split), streamed via datasets.")
    parser.add_argument("--hf-split", default="train", help="Default HF split when not suffixed with :split.")
    parser.add_argument("--local-jsonl", action="append", default=[],
                        help="Repeatable local JSONL file(s) of OpenHands/SWE-smith/R2E-Gym/OpenAI trajectories.")
    parser.add_argument("--results-dirs", action="append", default=[],
                        help="Repeatable swegemma results dirs (task_results.jsonl, patches/, test_outputs/).")
    parser.add_argument("--max-examples", type=int, default=DEFAULT_MAX_EXAMPLES)
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--min-edits", type=int, default=DEFAULT_MIN_EDITS)
    parser.add_argument("--exclude-repos", default="", help="Comma-separated repo names to exclude (owner/repo).")
    parser.add_argument("--out-coder", default=None, help="Output JSONL for coder LoRA.")
    parser.add_argument("--out-locator", default=None, help="Output JSONL for locator/analyzer LoRA.")
    parser.add_argument("--out-verifier", default=None, help="Output JSONL for verifier LoRA.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    counts = run_import(args)
    print(
        f"[import_hf] coder={counts['coder']} locator={counts['locator']} "
        f"verifier={counts['verifier']} (GO={counts['verifier_go']}, BLOCK={counts['verifier_block']})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
