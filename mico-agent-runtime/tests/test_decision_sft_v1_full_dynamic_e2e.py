from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from mico_agent_runtime.e2e.decision_sft_v1_full_dynamic import (
    ALL_SCIENTIFIC_ACTIONS,
    DEFAULT_FREEZE_MANIFEST,
    DEFAULT_TASK_SET_PATH,
    TaskSetValidationError,
    load_task_set,
    prepare_task_set,
    validate_task_set,
)
from scripts.run_decision_sft_v1_full_dynamic_e2e import (
    _controlled_finish_record,
    _task_from_spec,
    main as e2e_main,
)


def test_full_dynamic_task_set_is_open_and_provenance_safe() -> None:
    payload = load_task_set(DEFAULT_TASK_SET_PATH)
    audit = validate_task_set(payload)

    assert audit["schema_valid"] is True
    assert audit["task_count"] == 5
    assert audit["fixed_action_set"] == list(ALL_SCIENTIFIC_ACTIONS)
    assert audit["scenario_coverage"] == [
        "controlled_finish_boundary",
        "real_availability_boundary",
        "real_dynamic",
        "real_evidence_optional",
    ]
    assert audit["all_training_ineligible"] is True
    assert payload["execution_policy"]["open_trajectory"] is True
    assert payload["execution_policy"]["fixed_action_sequence"] is False
    assert all(task["max_actions"] == 6 for task in payload["tasks"])


def test_task_set_rejects_policy_oracle_fields() -> None:
    payload = load_task_set(DEFAULT_TASK_SET_PATH)
    invalid = copy.deepcopy(payload)
    invalid["tasks"][0]["required_action"] = "compare_groups"
    with pytest.raises(TaskSetValidationError, match="required_action"):
        validate_task_set(invalid)


def test_task_set_rejects_fake_runtime_payload() -> None:
    payload = load_task_set(DEFAULT_TASK_SET_PATH)
    invalid = copy.deepcopy(payload)
    invalid["tasks"][0]["queryPlan"] = {"root_entity": "sample"}
    with pytest.raises(TaskSetValidationError, match="unknown_fields|sensitive_marker|physical_endpoint"):
        validate_task_set(invalid)


def test_offline_preparation_has_no_remote_contact(tmp_path: Path) -> None:
    manifest = prepare_task_set(
        DEFAULT_TASK_SET_PATH,
        tmp_path,
        DEFAULT_FREEZE_MANIFEST,
    )
    assert manifest["status"] == "PREPARED_NOT_RUN"
    assert manifest["model_calls"] == 0
    assert manifest["no_remote_contact"] is True
    assert manifest["a100_started"] is False
    assert manifest["full_e2e_started"] is False
    assert manifest["dpo_started"] is False
    assert manifest["training_eligible"] is False
    assert manifest["FULL_E2E_READY"] is False
    assert manifest["READY_TO_START_A100"] is False
    assert all(value == "PENDING_LIVE" for key, value in manifest["acceptance_gates"].items() if key in {
        "java_mysql_preflight",
        "qwen_sft_http_endpoint",
        "gemini_task_understanding",
        "gemini_materializer",
        "real_observation_and_state_update",
        "trace_provenance",
    })
    assert (tmp_path / "task_set_snapshot.json").exists()
    assert (tmp_path / "preparation_manifest.json").exists()
    assert (tmp_path / "acceptance_gates.json").exists()
    saved = json.loads((tmp_path / "preparation_manifest.json").read_text(encoding="utf-8"))
    assert saved["task_set_sha256"] == manifest["task_set_sha256"]


def test_runner_task_projection_uses_all_fixed_actions_and_six_budget() -> None:
    task = load_task_set(DEFAULT_TASK_SET_PATH)["tasks"][0]
    request = _task_from_spec(task, 0)
    assert request.maxActions == 6
    assert request.allowedActions == list(ALL_SCIENTIFIC_ACTIONS)
    assert request.question == task["question"]


def test_controlled_finish_is_not_presented_as_real_trace(tmp_path: Path) -> None:
    task = load_task_set(DEFAULT_TASK_SET_PATH)["tasks"][-1]
    result = _controlled_finish_record(task, tmp_path)
    assert result["status"] == "CONTROLLED_BOUNDARY_NOT_EXECUTED"
    assert result["training_eligible"] is False
    assert result["policy_request_count"] == 0
    assert (tmp_path / task["task_id"] / "controlled_boundary.json").exists()


def test_default_e2e_runner_is_offline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import scripts.run_decision_sft_v1_full_dynamic_e2e as runner

    def fail_if_called() -> None:
        raise AssertionError("offline runner must not call preflight")

    monkeypatch.setattr(runner, "run_preflight", fail_if_called)
    assert e2e_main([
        "--output-dir",
        str(tmp_path),
        "--task-set",
        str(DEFAULT_TASK_SET_PATH),
        "--freeze-manifest",
        str(DEFAULT_FREEZE_MANIFEST),
    ]) == 0
