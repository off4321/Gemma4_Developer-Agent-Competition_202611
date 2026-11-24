"""`submission/` の YAML 結び合わせ検査（提出物の契約はここで固定する）。

`build_submission.py --check-only` は「読める + model 1 種 + adapter 実体」までしか見ない。
ここでは 2026-11-24 の設定変更を落とさないために次を確かめる:
  * `verifier` sub-agent は無効のまま（`agent.yaml` の agent_tool entry も、
    `sub_agents/verifier.yaml` の `adapter:` もコメント扱い）。
  * `code_analyzer` は自分専用の `configs/analyzer_sampling.yaml` を使う（root と共有しない）。
  * その設定が `build_submission.py` の許可フィールドだけを運ぶ。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submission"
sys.path.insert(0, str(ROOT / "src" / "package"))
import build_submission as build_submission_module  # noqa: E402  (needs the sys.path above)

pytestmark = pytest.mark.skipif(not SUBMISSION.is_dir(), reason="submission/ not present")


def load(relative: str) -> dict:
    return build_submission_module.load_yaml(SUBMISSION / relative)


def strip_comments(relative: str) -> str:
    return "\n".join(line for line in (SUBMISSION / relative).read_text(encoding="utf-8").splitlines()
                     if not line.lstrip().startswith("#"))


def test_verifier_sub_agent_is_not_reachable() -> None:
    agent = load("agent.yaml")
    config_paths = [entry["agent_tool"]["config_path"] for entry in agent["tools"]
                    if isinstance(entry, dict) and "agent_tool" in entry]
    assert config_paths == ["sub_agents/code_analyzer.yaml"]
    assert "sub_agents/verifier.yaml" not in strip_comments("agent.yaml")


def test_verifier_yaml_has_no_active_adapter_line() -> None:
    body = strip_comments("sub_agents/verifier.yaml")
    keys = [line.split(":")[0].strip() for line in body.splitlines() if ":" in line]
    assert "adapter" not in keys, "verifier_lora must stay commented out with the sub-agent off"
    assert "model" in keys, "the file must stay loadable if the agent_tool entry comes back"


def test_analyzer_uses_its_own_generation_config() -> None:
    analyzer = load("sub_agents/code_analyzer.yaml")
    body = (SUBMISSION / "sub_agents/code_analyzer.yaml").read_text(encoding="utf-8")
    assert "../configs/analyzer_sampling.yaml" in body
    assert "generate_content_config" in analyzer
    assert analyzer["adapter"] == "analyzer_lora"
    assert analyzer["model"] == load("agent.yaml")["model"], "single-model rule (HARNESS §2.3)"


def test_analyzer_generation_config_is_legal_and_deterministic() -> None:
    config = load("configs/analyzer_sampling.yaml")
    checker = build_submission_module.Checker()
    build_submission_module._check_generation_config("configs/analyzer_sampling.yaml", config, checker)
    assert checker.errors == []
    assert config["temperature"] == pytest.approx(0.1)
    assert config["top_p"] == pytest.approx(0.9)
    assert config["max_output_tokens"] == 2048
    thinking = config["thinking_config"]
    assert thinking["thinking_budget"] == 1024 and thinking["include_thoughts"] is True


def test_root_still_shares_only_the_skill_scripts() -> None:
    agent = load("agent.yaml")
    assert agent["skills"] == ["skills/swe-guard"]
    skill_doc = (SUBMISSION / "skills/swe-guard/SKILL.md").read_text(encoding="utf-8")
    for script in ("extract_block.py", "run_pytest_compact.py", "check_patch.py"):
        assert script in skill_doc, f"SKILL.md must document {script}"
