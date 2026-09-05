from __future__ import annotations

import json
from pathlib import Path

import pytest

from mico_agent_runtime.datasets.decision_sft_v1 import (
    STATE_BLOCKS,
    _build_state,
    _load_catalog,
    _record,
    build_decision_sft_v1,
    validate_record,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = REPO_ROOT / "artifacts" / "gemini_dynamic_canary_4d1g_final3" / "semantic_catalog.json"


def _catalog():
    return _load_catalog(CATALOG_PATH)


def _valid_raw(*, action: str = "inspect_cohort") -> dict:
    catalog = _catalog()
    state, profile = _build_state(
        catalog=catalog,
        variant=991,
        mode="empty",
        desired_action=action,
    )
    return _record(
        sample_index=999001,
        trajectory_id="trajectory-test-v1-001",
        turn_index=0,
        state=state,
        source_type="controlled_state",
        state_origin="controlled_state",
        availability_profile=profile,
        action=action,
        hard_case_class=None,
    )


def test_builder_creates_audited_candidate_pool_and_disjoint_splits(tmp_path: Path) -> None:
    audit = build_decision_sft_v1(
        tmp_path,
        catalog_path=CATALOG_PATH,
        manual_review_completed=False,
    )

    assert audit["candidate_count"] == 543
    assert audit["approved_count"] == 480
    assert audit["rejected_count"] == 63
    assert audit["split_distribution"] == {"train": 360, "validation": 60, "test": 60}
    assert audit["duplicate_count"] == 20
    assert audit["near_duplicate_count"] == 0
    assert audit["pair_trajectory_split_leakage"]["overlap_count"] == 0
    assert audit["checks"]["sample_id_unique"] is True
    assert audit["checks"]["selected_action_available_100pct"] is True
    assert audit["checks"]["alternative_actions_available_100pct"] is True
    assert audit["checks"]["finish_contract_100pct"] is True
    assert audit["checks"]["hard_cases_covered"] is True
    assert audit["DECISION_SFT_V1_DATA_READY"] is False
    assert audit["training_started"] is False
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["training_eligible"] is False
    assert manifest["candidate_training_eligible"] is True

    for filename, expected in (
        ("all_candidates.jsonl", 543),
        ("approved.jsonl", 480),
        ("train.jsonl", 360),
        ("validation.jsonl", 60),
        ("test.jsonl", 60),
    ):
        assert len((tmp_path / filename).read_text(encoding="utf-8").splitlines()) == expected

    model_record = json.loads((tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert set(model_record) == {"input", "output"}
    assert model_record["input"]["decision_type"] == "scientific_action"
    assert tuple(model_record["input"]["state"]) == STATE_BLOCKS
    assert set(model_record["output"]) == {
        "selected_action",
        "decision_reason",
        "alternative_actions",
        "stop_reason",
    }


def test_validator_recomputes_availability_and_rejects_unavailable_label() -> None:
    raw = _valid_raw()
    raw["target"]["selected_action"] = "compare_groups"

    record, issues = validate_record(raw, catalog=_catalog())

    assert record is not None
    assert "selected_action_unavailable" in issues


def test_validator_rejects_unavailable_alternative_and_legacy_fields() -> None:
    raw = _valid_raw()
    raw["target"]["alternative_actions"] = ["compare_groups"]
    _, issues = validate_record(raw, catalog=_catalog())
    assert "alternative_action_unavailable" in issues

    legacy = _valid_raw()
    legacy["state"]["goal_code"] = "leak"
    _, legacy_issues = validate_record(legacy, catalog=_catalog())
    assert "legacy_or_leakage_field" in legacy_issues


def test_validator_rejects_objective_action_confusion_and_finish_shape() -> None:
    objective = _valid_raw()
    objective["target"]["selected_action"] = "cross_project_validation"
    _, objective_issues = validate_record(objective, catalog=_catalog())
    assert "objective_action_confusion" in objective_issues
    assert "schema_invalid" in objective_issues

    catalog = _catalog()
    finish_state, finish_profile = _build_state(
        catalog=catalog,
        variant=992,
        mode="project",
        desired_action="finish",
        group_status="completed",
        adjustment_status="completed",
        cross_project_status="completed",
        stratified_status="completed",
        projection_status="completed",
        cross_disease_status="completed",
        evidence_status="completed",
        evidence_consistency="mostly_supportive",
        project_count=3,
        positive_projects=2,
        negative_projects=1,
        covariates=["sample.age"],
        group_effect=0.4,
        adjusted_effect=0.2,
    )
    finish = _record(
        sample_index=999002,
        trajectory_id="trajectory-test-v1-finish",
        turn_index=0,
        state=finish_state,
        source_type="controlled_state",
        state_origin="controlled_state",
        availability_profile=finish_profile,
        action="finish",
        hard_case_class=None,
    )
    finish["target"]["stop_reason"] = None
    _, finish_issues = validate_record(finish, catalog=_catalog())
    assert "finish_contract_violation" in finish_issues
    assert "schema_invalid" in finish_issues


def test_model_jsonl_excludes_provenance_and_raw_fields() -> None:
    raw = _valid_raw()
    record, issues = validate_record(raw, catalog=_catalog())
    assert issues == []
    assert record is not None

    model_record = record.model_jsonl_record()
    assert set(model_record) == {"input", "output"}
    assert "metadata" not in model_record["input"]
    assert tuple(model_record["input"]["state"]) == STATE_BLOCKS
    assert "raw_rows" not in json.dumps(model_record, ensure_ascii=False)


def test_validator_rejects_invalid_enum_and_negative_count() -> None:
    invalid_enum = _valid_raw()
    invalid_enum["state"]["analysis_state"]["cross_project_validation"]["heterogeneity"] = "VERY_HIGH"
    _, enum_issues = validate_record(invalid_enum, catalog=_catalog())
    assert "schema_invalid" in enum_issues

    negative = _valid_raw()
    negative["state"]["data_state"]["row_count"] = -10
    _, negative_issues = validate_record(negative, catalog=_catalog())
    assert "negative_count" in negative_issues
    assert "schema_invalid" in negative_issues


def test_validator_checks_provenance_alignment() -> None:
    raw = _valid_raw()
    raw["metadata"]["state_origin"] = "boundary_state"
    _, issues = validate_record(raw, catalog=_catalog())
    assert "provenance_mismatch" in issues


def test_validator_rejects_template_only_reason() -> None:
    raw = _valid_raw()
    raw["target"]["decision_reason"] = "The next appropriate action is inspect_cohort."
    _, issues = validate_record(raw, catalog=_catalog())
    assert "ungrounded_reason" in issues


@pytest.mark.parametrize(
    "forbidden_path",
    [
        ("state", "rows"),
        ("state", "data_state", "patient_id"),
    ],
)
def test_validator_rejects_raw_rows_or_identity(forbidden_path: tuple[str, ...]) -> None:
    raw = _valid_raw()
    cursor: dict = raw
    for key in forbidden_path[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[forbidden_path[-1]] = [] if forbidden_path[-1] == "rows" else "patient-1"
    _, issues = validate_record(raw, catalog=_catalog())
    assert ("raw_rows" if forbidden_path[-1] == "rows" else "pii_or_identity") in issues
