from __future__ import annotations

import pytest
from pydantic import ValidationError

from mico_agent_runtime.contracts.decision_state import (
    ScientificDecisionState,
    ScientificTaskState,
)


def test_default_state_round_trip() -> None:
    state = ScientificDecisionState(task=ScientificTaskState(query="test question"))

    payload = state.model_dump(mode="json")
    restored = ScientificDecisionState.model_validate(payload)

    assert restored == state
    assert payload["task"]["query"] == "test question"
    assert payload["data_state"]["has_tabular_data"] is False
    assert payload["analysis_state"]["group_comparison"]["status"] == "not_started"
    assert payload["evidence_state"]["status"] == "not_started"
    assert payload["progress"]["action_count"] == 0
    assert payload["action_space"]["available_actions"] == []


def test_full_state_round_trip() -> None:
    state = ScientificDecisionState.model_validate({
        "task": {
            "query": "compare T2D and Healthy",
            "objectives": [
                "group_comparison",
                "confounder_assessment",
                "cross_project_validation",
                "evidence_support",
            ],
            "constraints": {
                "disease_groups": ["T2D", "Healthy"],
                "target_features": ["abundance"],
                "focus_covariates": ["age"],
                "requested_projects": [],
                "requested_stratifiers": [],
            },
        },
        "data_state": {
            "has_tabular_data": True,
            "row_count": 326,
            "available_dimensions": [
                "sample.disease", "metadata.project", "sample.age", "sample.gender",
            ],
            "available_outcomes": ["abundance.value"],
            "group_state": {
                "group_field": "disease",
                "group_count": 2,
                "group_sizes": {"T2D": 172, "Healthy": 154},
            },
            "project_state": {"has_project_field": True, "project_count": 7},
            "covariate_state": {
                "available_covariates": ["sample.age", "sample.gender"],
                "imbalance": {"age": "high", "sex": "low"},
            },
            "data_quality": {
                "missingness_level": "low",
                "sample_size_level": "sufficient",
            },
        },
        "analysis_state": {
            "group_comparison": {
                "status": "completed",
                "effect_size": 0.62,
                "p_value": 0.008,
            },
            "confounder_adjustment": {
                "status": "completed",
                "adjusted_effect_size": 0.18,
                "adjusted_p_value": 0.21,
                "adjusted_covariates": ["sample.age", "sample.gender"],
            },
            "cross_project_validation": {
                "status": "completed",
                "project_count": 7,
                "heterogeneity": "high",
            },
            "stratified_analysis": {
                "status": "not_started",
                "stratify_fields": [],
                "stratum_count": 0,
                "heterogeneity": "unknown",
            },
            "projection_analysis": {"status": "not_started", "result_count": 0},
            "cross_disease_validation": {
                "status": "not_started",
                "disease_count": 0,
                "heterogeneity": "unknown",
            },
        },
        "evidence_state": {
            "status": "completed",
            "evidence_count": 6,
            "support_count": 4,
            "conflict_count": 1,
            "context_count": 1,
            "consistency": "mostly_supportive",
        },
        "progress": {
            "completed_actions": [
                "execute_read_query",
                "compare_groups",
                "adjust_confounders",
            ],
            "action_counts": {
                "execute_read_query": 1,
                "compare_groups": 1,
                "adjust_confounders": 1,
            },
            "last_action": "adjust_confounders",
            "remaining_objectives": ["cross_project_validation", "evidence_support"],
            "action_count": 3,
        },
        "action_space": {
            "available_actions": [
                "execute_read_query",
                "compare_groups",
                "adjust_confounders",
                "cross_project_validate",
                "retrieve_evidence",
                "finish",
            ],
        },
    })

    restored = ScientificDecisionState.model_validate(state.model_dump(mode="json"))
    assert restored == state
    assert state.data_state.project_state.project_count == 7
    assert state.analysis_state.group_comparison.effect_size == 0.62
    assert state.evidence_state.evidence_count == 6


def test_invalid_enum_rejected() -> None:
    with pytest.raises(ValidationError):
        ScientificDecisionState.model_validate({
            "task": {"query": "test"},
            "analysis_state": {
                "cross_project_validation": {"heterogeneity": "VERY_HIGH"},
            },
        })


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("data_state", "row_count"), -10),
        (("data_state", "project_state", "project_count"), -1),
        (("evidence_state", "evidence_count"), -1),
    ],
)
def test_negative_counts_rejected(path: tuple[str, ...], value: int) -> None:
    payload: dict[str, object] = {"task": {"query": "test"}}
    target = payload
    for key in path[:-1]:
        child: dict[str, object] = {}
        target[key] = child
        target = child
    target[path[-1]] = value

    with pytest.raises(ValidationError):
        ScientificDecisionState.model_validate(payload)


def test_extra_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        ScientificDecisionState.model_validate({
            "task": {"query": "test"},
            "should_adjust_confounders": True,
        })


def test_old_numeric_field_name_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ScientificDecisionState.model_validate({
            "task": {"query": "test"},
            "data_state": {"available_numeric_fields": ["sample.age"]},
        })
