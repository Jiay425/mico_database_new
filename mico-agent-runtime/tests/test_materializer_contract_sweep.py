from __future__ import annotations

from types import SimpleNamespace

from mico_agent_runtime.contracts.materializer_contract import (
    MATERIALIZER_CONTRACTS,
    MATERIALIZER_CONTRACT_BY_ACTION,
)
from scripts.audit_materializer_contracts import build_report
from scripts.run_gemini_dynamic_canary import _materializer_call_audit
from scripts.run_materializer_contract_sweep import _validate_result_shape


EXPECTED_ACTIONS = {
    "inspect_cohort",
    "execute_read_query",
    "compare_groups",
    "analyze_projection",
    "stratified_analysis",
    "adjust_confounders",
    "cross_project_validate",
    "cross_disease_validate",
    "retrieve_evidence",
    "finish",
}


def test_materializer_contract_manifest_is_exhaustive_and_unique() -> None:
    assert {item.action for item in MATERIALIZER_CONTRACTS} == EXPECTED_ACTIONS
    assert len(MATERIALIZER_CONTRACTS) == len(EXPECTED_ACTIONS)
    assert set(MATERIALIZER_CONTRACT_BY_ACTION) == EXPECTED_ACTIONS
    assert sum(item.materializer_required for item in MATERIALIZER_CONTRACTS) == 8
    assert MATERIALIZER_CONTRACT_BY_ACTION["finish"].contract_kind == "none"
    assert not MATERIALIZER_CONTRACT_BY_ACTION["inspect_cohort"].materializer_required
    assert all(item.response_parsers for item in MATERIALIZER_CONTRACTS)
    assert all(item.domain_validators for item in MATERIALIZER_CONTRACTS)


def test_analysis_actions_have_one_canonical_v2_discriminator() -> None:
    actual = {
        item.action: item.discriminator
        for item in MATERIALIZER_CONTRACTS
        if item.contract_kind == "analysis_plan_v2"
    }
    assert actual == {
        "compare_groups": "analysis_type=group_comparison",
        "analyze_projection": "analysis_type=projection",
        "stratified_analysis": "analysis_type=stratified_comparison",
        "adjust_confounders": "analysis_type=confounder_adjustment",
        "cross_project_validate": "analysis_type=cross_project_validation",
        "cross_disease_validate": "analysis_type=cross_disease_validation",
    }


def test_static_audit_is_no_api_and_captures_projection_root_cause() -> None:
    report = build_report()
    assert report["status"] == "PASS"
    assert report["api_calls_made"] == 0
    assert report["summary"]["action_count"] == 10
    root_cause = report["analyze_projection_root_cause"]
    assert root_cause["parser_contract"]["model"] == "AnalyzeProjectionArguments"
    assert root_cause["parser_contract"]["observation_cardinality"] == "one singular observationId"
    assert root_cause["projection_prompt_has_singular_rule"] is True


def test_repaired_final_contract_is_not_mislabeled_as_first_pass_failure() -> None:
    contract = MATERIALIZER_CONTRACT_BY_ACTION["compare_groups"]
    audit = {
        "model_response_count": 2,
        "first_model_response_valid": False,
        "first_model_response_failure_reason": "schema_validation",
        "first_model_contract_kind": "unknown",
        "final_model_response_valid": True,
        "final_model_response_failure_reason": None,
        "final_model_contract_kind": "typed_analysis_plan",
        "expected_analysis_type": "group_comparison",
        "contract_repair_count": 1,
    }
    result = SimpleNamespace(mode="model")

    first_valid, first_reason = _validate_result_shape(
        contract, result, audit, final=False
    )
    final_valid, final_reason = _validate_result_shape(
        contract, result, audit, final=True
    )

    assert first_valid is False
    assert first_reason == "schema_validation"
    assert final_valid is True
    assert final_reason is None


def test_deterministic_materializer_fallback_is_rejected_by_api_sweep() -> None:
    contract = MATERIALIZER_CONTRACT_BY_ACTION["analyze_projection"]
    audit = {
        "model_response_count": 1,
        "first_model_response_valid": True,
        "first_model_response_failure_reason": None,
        "first_model_contract_kind": "typed_analysis_plan",
        "final_model_response_valid": True,
        "final_model_response_failure_reason": None,
        "final_model_contract_kind": "typed_analysis_plan",
        "expected_analysis_type": "projection",
    }
    result = SimpleNamespace(mode="deterministic")

    valid, reason = _validate_result_shape(contract, result, audit, final=True)

    assert valid is False
    assert reason == "deterministic_materializer_fallback_not_allowed"


def test_transport_attempts_are_not_scored_as_contract_failures() -> None:
    records = [
        {
            "materializer_call": 1,
            "action_name": "execute_read_query",
            "contract_kind": "unknown",
            "planner_method": None,
            "expected_analysis_type": None,
            "approved_actions": ["execute_read_query"],
            "failure_reason": "http_failure",
            "model_response": False,
            "transport_error": "RemoteProtocolError",
            "repair_request": False,
        },
        {
            "materializer_call": 1,
            "action_name": "execute_read_query",
            "contract_kind": "scientific_action",
            "planner_method": "plan_action",
            "expected_analysis_type": None,
            "approved_actions": ["execute_read_query"],
            "failure_reason": None,
            "model_response": True,
            "transport_error": None,
            "repair_request": False,
        },
    ]

    audit = _materializer_call_audit(records)[0]

    assert audit["transport_attempts"] == 2
    assert audit["model_response_count"] == 1
    assert audit["contract_repair_count"] == 0
    assert audit["first_model_response_valid"] is True
    assert audit["first_model_response_failure_reason"] is None
    assert audit["transport_failure_reasons"] == ["RemoteProtocolError"]


def test_sweep_report_shape_can_represent_all_ten_actions() -> None:
    # The API sweep may stop at the bounded request budget, but the report
    # must still contain an explicit status for every closed Action.
    from scripts.run_materializer_contract_sweep import MATERIALIZER_CONTRACTS as sweep_contracts

    assert len(sweep_contracts) == 10
