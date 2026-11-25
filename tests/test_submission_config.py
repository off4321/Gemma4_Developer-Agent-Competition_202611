"""`submission/` の Hybrid 構成（Root 基底 e4b + `code_analyzer` 専用 LoRA）契約検査。

検査項目:
  * ルート `agent.yaml` はベース `gemma-4-e4b-it-qat-w4a16-ct` で動き（`main_lora` による `<|tool_call>` 破損を回避）、
    `code_analyzer` (`agent_tool`, `skip_summarization: true`) に探索を隔離する。
  * `verifier` sub-agent は無効のまま（`agent.yaml` でも `sub_agents/verifier.yaml` でも未接続）。
  * `code_analyzer` は `analyzer_lora` と `configs/analyzer_sampling.yaml` を使用し、単一モデル規則を満たす。
  * 生成パラメータが `build_submission.py` の許可フィールドのみで構成されている。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submission"
sys.path.insert(0, str(ROOT / "src" / "package"))
import build_submission as build_submission_module  # noqa: E402

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
    assert "adapter" not in agent, "root coder runs on base e4b without main_lora tag corruption"


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
    assert analyzer["model"] == load("agent.yaml")["model"], "single-model rule (HARNESS §3.2)"


def test_generation_configs_are_legal_and_well_sized() -> None:
    for rel in ("configs/analyzer_sampling.yaml", "configs/sampling.yaml"):
        config = load(rel)
        checker = build_submission_module.Checker()
        build_submission_module._check_generation_config(rel, config, checker)
        assert checker.errors == []

    analyzer_cfg = load("configs/analyzer_sampling.yaml")
    assert analyzer_cfg["temperature"] == pytest.approx(0.1)
    assert analyzer_cfg["top_p"] == pytest.approx(0.9)
    assert analyzer_cfg["max_output_tokens"] == 2048
    assert analyzer_cfg["thinking_config"]["thinking_budget"] == 1024
    assert analyzer_cfg["thinking_config"]["include_thoughts"] is True

    root_cfg = load("configs/sampling.yaml")
    assert root_cfg["max_output_tokens"] == 6144
    assert "frequency_penalty" not in root_cfg and "presence_penalty" not in root_cfg
    assert root_cfg["thinking_config"]["thinking_budget"] == 1536
    assert root_cfg["thinking_config"]["include_thoughts"] is True


def test_root_still_shares_only_the_skill_scripts() -> None:
    agent = load("agent.yaml")
    assert agent["skills"] == ["skills/swe-guard"]
    skill_doc = (SUBMISSION / "skills/swe-guard/SKILL.md").read_text(encoding="utf-8")
    for script in ("extract_block.py", "run_pytest_compact.py", "check_patch.py"):
        assert script in skill_doc, f"SKILL.md must document {script}"
