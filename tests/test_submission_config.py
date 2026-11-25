"""`submission/` の 3段 SequentialAgent パイプライン契約検査。

検査項目:
  * ルート `agent.yaml` は `SequentialAgent` で、`code_analyzer` -> `coder` -> `submitter` の 3 段構成。
  * `verifier` sub-agent は無効のまま（`agent.yaml` でも `sub_agents/verifier.yaml` でも未接続）。
  * 全到達可能 `LlmAgent` が単一の `gemma-4-e4b-it-qat-w4a16-ct` を宣言（HARNESS §3.2）。
  * Stage 1 (`code_analyzer`): read-only ツールのみ、`output_key: localization_plan`、`analyzer_lora`。
  * Stage 2 (`coder`): `edit_file` / `write_file` を持ち `submit_patch` を持たない、`output_key: impl_report`、`main_lora`、`skills/swe-guard` 保持。
  * Stage 3 (`submitter`): 最終段のみが `submit_patch` を保持し、決定論的 sampling を使用。
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


def test_root_is_three_stage_sequential_agent_and_verifier_is_unreachable() -> None:
    agent = load("agent.yaml")
    assert agent["agent_class"] == "SequentialAgent"
    config_paths = [entry["config_path"] for entry in agent["sub_agents"]]
    assert config_paths == [
        "sub_agents/code_analyzer.yaml",
        "sub_agents/coder.yaml",
        "sub_agents/submitter.yaml",
    ]
    assert "sub_agents/verifier.yaml" not in strip_comments("agent.yaml")


def test_verifier_yaml_has_no_active_adapter_line() -> None:
    body = strip_comments("sub_agents/verifier.yaml")
    keys = [line.split(":")[0].strip() for line in body.splitlines() if ":" in line]
    assert "adapter" not in keys, "verifier_lora must stay commented out with the sub-agent off"
    assert "model" in keys, "the file must stay loadable if the sub-agent entry comes back"


def test_single_declared_model_and_stage_role_isolation() -> None:
    analyzer = load("sub_agents/code_analyzer.yaml")
    coder = load("sub_agents/coder.yaml")
    submitter = load("sub_agents/submitter.yaml")

    assert {analyzer["model"], coder["model"], submitter["model"]} == {"gemma-4-e4b-it-qat-w4a16-ct"}

    # Stage 1: read-only explorer -> localization_plan
    assert analyzer["adapter"] == "analyzer_lora"
    assert analyzer["output_key"] == "localization_plan"
    assert not (set(analyzer["tools"]) & {"edit_file", "write_file", "submit_patch"})

    # Stage 2: coder -> impl_report (no premature submit_patch)
    assert coder["adapter"] == "main_lora"
    assert coder["output_key"] == "impl_report"
    assert {"read_file", "edit_file", "write_file", "run_command"} <= set(coder["tools"])
    assert "submit_patch" not in coder["tools"]

    # Stage 3: submitter -> holds submit_patch
    assert "submit_patch" in submitter["tools"]
    assert "adapter" not in submitter


def test_all_stage_generation_configs_are_legal() -> None:
    for rel in ("configs/analyzer_sampling.yaml", "configs/sampling.yaml", "configs/submitter_sampling.yaml"):
        config = load(rel)
        checker = build_submission_module.Checker()
        build_submission_module._check_generation_config(rel, config, checker)
        assert checker.errors == []
        assert config["thinking_config"]["include_thoughts"] is True

    analyzer_cfg = load("configs/analyzer_sampling.yaml")
    assert analyzer_cfg["temperature"] == pytest.approx(0.1)
    assert analyzer_cfg["max_output_tokens"] == 2048

    coder_cfg = load("configs/sampling.yaml")
    assert coder_cfg["temperature"] == pytest.approx(0.4)
    assert coder_cfg["max_output_tokens"] == 4096


def test_coder_keeps_swe_guard_skill_docs() -> None:
    coder = load("sub_agents/coder.yaml")
    assert coder["skills"] == ["skills/swe-guard"]
    skill_doc = (SUBMISSION / "skills/swe-guard/SKILL.md").read_text(encoding="utf-8")
    for script in ("extract_block.py", "run_pytest_compact.py", "check_patch.py"):
        assert script in skill_doc, f"SKILL.md must document {script}"
