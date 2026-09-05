import json
from pathlib import Path

from evals.p2j4_freeze_dynamic_e2e_contract import (
    static_materializer_prompt_hash,
    validate_materializer_config,
    validate_task_set,
)


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_dynamic_task_set_has_no_contract_failures() -> None:
    payload = json.loads(
        (ROOT / "evals" / "p2j4_dynamic_e2e_task_set_v1.json").read_text(encoding="utf-8")
    )
    assert validate_task_set(payload) == []


def test_materializer_config_is_bound_to_current_prompt_literals() -> None:
    config_path = ROOT / "evals" / "p2j4_dynamic_e2e_materializer_config_v1.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    failures, prompt_hash = validate_materializer_config(
        payload,
        ROOT / "mico_agent_runtime" / "ports" / "research_planner.py",
    )
    assert failures == []
    assert payload["systemPromptHash"] == prompt_hash
    assert prompt_hash == static_materializer_prompt_hash(
        ROOT / "mico_agent_runtime" / "ports" / "research_planner.py"
    )


def test_task_set_rejects_raw_query_fragments_and_unbounded_relation_oracle() -> None:
    task_set = json.loads(
        (ROOT / "evals" / "p2j4_dynamic_e2e_task_set_v1.json").read_text(encoding="utf-8")
    )
    task_set["tasks"][3]["question"] = "Run SELECT * from the business database."
    task_set["tasks"][3]["oracle"]["queryPlan"]["relationPathMax"] = 3

    failures = validate_task_set(task_set)

    assert any("question" in failure for failure in failures)
    assert any("relationPathMax" in failure for failure in failures)
