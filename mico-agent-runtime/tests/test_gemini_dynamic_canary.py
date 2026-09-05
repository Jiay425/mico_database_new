from __future__ import annotations

from datetime import datetime, timezone

from mico_agent_runtime.contracts.schema_catalog import (
    SchemaEntitySemantics,
    SchemaFieldSemantics,
    SchemaSemanticCatalog,
)
from mico_agent_runtime.contracts.generated_analysis import GeneratedAnalysisPlan
from mico_agent_runtime.contracts.materialization import QueryPlan
from scripts.run_gemini_dynamic_canary import (
    _materializer_failure_reason,
    _materializer_call_audit,
    _normalise_materializer_response,
    _query_semantics_audit,
    _query_result_profile,
    _query_plan_fingerprint,
    run_failure_canaries,
)


def _catalog() -> SchemaSemanticCatalog:
    return SchemaSemanticCatalog(
        schemaVersion="schema-catalog-v1",
        source="java_schema_contract",
        generatedAt=datetime.now(timezone.utc),
        entities=[SchemaEntitySemantics(
            entityId="sample",
            entityName="sample",
            sourceTable="sample",
            fields=[SchemaFieldSemantics(
                fieldId="sample.disease",
                name="disease",
                dataType="string",
                nullable=True,
                semanticStatus="verified",
                filterable=True,
                groupable=True,
                displayable=True,
                scientificCapabilities=["dimension"],
                description="group",
            )],
        )],
        queryRules=[
            "select_or_with_only",
            "explicit_columns_only",
            "no_cross_database_reference",
            "bounded_limit_required",
            "java_final_validation",
        ],
    )


def test_gemini_dynamic_failure_canaries_are_fail_closed(tmp_path) -> None:
    failures = run_failure_canaries(tmp_path, _catalog())

    assert set(failures) == {
        "F1_invalid_policy_action",
        "F2_unavailable_policy_action",
        "F3_policy_http_500",
        "F4_materializer_action_mismatch",
        "F5_invalid_query_plan",
        "F6_typed_insufficient_data",
    }
    assert failures["F1_invalid_policy_action"]["outcome"] == {
        "status": "raised",
        "code": "POLICY_DECISION_FAILED",
    }
    assert failures["F2_unavailable_policy_action"]["outcome"] == {
        "status": "returned",
        "selected_action": "execute_read_query",
    }
    assert failures["F3_policy_http_500"]["outcome"] == {
        "status": "raised",
        "code": "POLICY_DECISION_FAILED",
    }
    assert failures["F4_materializer_action_mismatch"]["code"] == (
        "MATERIALIZATION_ACTION_MISMATCH"
    )
    assert failures["F5_invalid_query_plan"]["status"] == "validator_rejected"
    assert failures["F6_typed_insufficient_data"]["status"] == "operator_rejected"


def test_query_plan_fingerprint_includes_default_contract_version() -> None:
    without_version = {
        "root_entity": "sample",
        "relation_path": [],
        "select_fields": ["sample.disease"],
        "aggregations": [],
        "filters": [],
        "group_by": [],
        "limit": 10,
    }
    with_version = {**without_version, "schemaVersion": "query-plan-v1"}

    assert _query_plan_fingerprint(without_version) == _query_plan_fingerprint(with_version)
    assert _query_plan_fingerprint({"root_entity": "not a plan"}) is None


def test_query_result_profile_keeps_result_rows_separate_from_samples() -> None:
    plan = QueryPlan(
        root_entity="sample",
        relation_path=["sample_to_abundance"],
        select_fields=["sample.disease", "abundance.feature", "abundance.value"],
        limit=10,
    )
    profile = _query_result_profile(
        {
            "columns": [
                "a_sample_disease",
                "a_abundance_feature",
                "a_abundance_value",
            ],
            "rows": [
                {
                    "a_sample_disease": "healthy",
                    "a_abundance_feature": "f1",
                    "a_abundance_value": 0.1,
                },
                {
                    "a_sample_disease": "T2D",
                    "a_abundance_feature": "f2",
                    "a_abundance_value": 0.2,
                },
            ],
        },
        plan,
        authoritative_row_count=10,
        query_hash="sha256:" + "a" * 64,
    )
    assert profile["row_count"] == 10
    assert profile["returned_row_count"] == 2
    assert profile["disease_value_counts"] == {"T2D": 1, "healthy": 1}
    assert profile["distinct_abundance_feature_count"] == 2
    assert profile["distinct_sample_count"] is None
    assert profile["limit_semantics"] == "result_rows"
    assert profile["sql_parameters"] == [10]


def test_query_semantics_audit_flags_feature_pooling_aggregate() -> None:
    plan = {
        "schemaVersion": "query-plan-v1",
        "root_entity": "sample",
        "relation_path": ["sample_to_abundance"],
        "select_fields": ["sample.disease"],
        "aggregations": [{"field": "abundance.value", "op": "mean"}],
        "filters": [],
        "group_by": ["sample.disease"],
        "limit": 100,
    }
    profile = _query_result_profile(
        {
            "columns": ["a_sample_disease", "a_abundance_value_mean"],
            "rows": [
                {"a_sample_disease": "T2D", "a_abundance_value_mean": 0.1},
                {"a_sample_disease": "healthy", "a_abundance_value_mean": 0.2},
            ],
        },
        QueryPlan.model_validate(plan),
        authoritative_row_count=2,
        query_hash="sha256:" + "b" * 64,
    )
    audit = _query_semantics_audit(
        [{
            "java_call_index": 1,
            "selected_action": "execute_read_query",
            "query_plan": plan,
            "compiled_query_fingerprint": "sha256:" + "b" * 64,
            "query_result_profile": profile,
        }],
        [{
            "analysis_plan": {
                "analysis_type": "group_comparison",
                "source_observation_ids": [],
            },
            "source_observations": [],
            "analysis_result": None,
        }],
    )
    assert audit["feature_pooling_from_query_plan"] is True
    assert audit["group_comparison_feature_mixing_risk"] is True
    assert audit["analysis_semantics_validated"] is False
    assert audit["analysis_semantics_status"] == "pending"
    assert audit["analysis_scientific_conclusion_eligible"] is False
    assert any("feature_dimension" in reason for reason in audit["root_causes"])


def test_query_semantics_audit_accepts_feature_aware_group_result() -> None:
    plans = []
    profiles = []
    for disease in ("healthy", "T2D"):
        plan = {
            "schemaVersion": "query-plan-v1",
            "root_entity": "sample",
            "relation_path": ["sample_to_abundance"],
            "select_fields": ["sample.disease", "abundance.feature", "abundance.value"],
            "aggregations": [],
            "filters": [{"field": "sample.disease", "operator": "in", "value": [disease]}],
            "group_by": [],
            "limit": 100,
        }
        profiles.append({
            "row_count": 2,
            "distinct_disease_values": [disease],
            "distinct_disease_count": 1,
            "distinct_abundance_feature_count": 2,
            "feature_pooling_risk": False,
            "sample_count_source": "returned_runtime_opaque_sample_key",
            "group_size_semantics": "sample_level",
        })
        plans.append({
            "java_call_index": len(plans) + 1,
            "selected_action": "execute_read_query",
            "query_plan": plan,
            "query_result_profile": profiles[-1],
        })

    audit = _query_semantics_audit(
        plans,
        [{
            "analysis_plan": {
                "analysis_type": "group_comparison",
                "feature_field": "abundance.feature",
                "source_observation_ids": [],
            },
            "source_observations": [],
            "analysis_result": {"feature_results": [{"featureName": "f1"}]},
        }],
    )

    assert audit["feature_aware_group_result"] is True
    assert audit["group_comparison_feature_mixing_risk"] is False
    assert audit["analysis_semantics_validated"] is True


def test_materializer_call_audit_reports_repair_count() -> None:
    records = [
        {
            "materializer_call": 1,
            "request_index": 1,
            "attempt": 1,
            "status": 200,
            "failure_reason": "schema_validation",
            "approved_actions": ["execute_read_query"],
            "execution_feedback": [],
        },
        {
            "materializer_call": 1,
            "request_index": 2,
            "attempt": 2,
            "status": 200,
            "failure_reason": None,
            "approved_actions": ["execute_read_query"],
            "execution_feedback": [],
            "normalized_response": {"actionName": "execute_read_query"},
        },
    ]
    audit = _materializer_call_audit(records)
    assert audit[0]["attempt_count"] == 2
    assert audit[0]["repair_count"] == 1
    assert audit[0]["first_pass_success"] is False
    assert audit[0]["first_failure_reason"] == "schema_validation"


def test_materializer_audit_distinguishes_generated_analysis_contract() -> None:
    generated = GeneratedAnalysisPlan(
        language="python",
        analysisType="stratified_analysis",
        code="result = {'metrics': {'count': len(rows)}, 'topFeatures': []}",
    ).model_dump(mode="json")
    context = {
        "actionName": "stratified_analysis",
        "workflow": "stratified_analysis",
        "approvedActions": ["stratified_analysis"],
        "sourceObservationIds": ["observation-" + "a" * 32],
        "availableSemanticFields": ["sample.disease", "sample.age", "abundance.value"],
    }

    assert _materializer_failure_reason(
        status=200,
        context=context,
        response=generated,
    ) is None
    audit = _materializer_call_audit([{
        "materializer_call": 1,
        "request_index": 1,
        "attempt": 1,
        "status": 200,
        "action_name": "stratified_analysis",
        "contract_kind": "generated_analysis_plan",
        "planner_method": "generate_analysis",
        "failure_reason": None,
        "approved_actions": ["stratified_analysis"],
        "normalized_response": generated,
    }])
    assert audit[0]["contract_kind"] == "generated_analysis_plan"
    assert audit[0]["planner_method"] == "generate_analysis"
    assert audit[0]["first_pass_success"] is True


def test_typed_materializer_audit_includes_feature_field_in_catalog_check() -> None:
    typed = {
        "schemaVersion": "analysis-plan-v2",
        "analysis_type": "stratified_comparison",
        "source_observation_ids": ["observation-" + "b" * 32],
        "outcome": "abundance.value",
        "feature_field": "abundance.feature",
        "group_field": "sample.disease",
        "covariates": [],
        "stratify_by": ["sample.age"],
        "validation_field": None,
        "method": {"family": "auto", "name": None, "parameters": {}},
        "analysis_goal": "compare across age strata",
        "metrics": ["effect_size"],
    }
    context = {
        "actionName": "stratified_analysis",
        "workflow": "stratified_analysis",
        "sourceObservationIds": ["observation-" + "b" * 32],
        "availableSemanticFields": ["abundance.value", "sample.disease", "sample.age"],
    }
    assert _materializer_failure_reason(
        status=200,
        context=context,
        response=typed,
    ) == "invalid_semantic_field_or_relation"


def test_action_audit_normalization_does_not_add_analysis_fields() -> None:
    action = {
        "actionId": "action-" + "a" * 32,
        "actionName": "execute_read_query",
        "rationale": "read the selected outcome",
        "arguments": {
            "actionName": "execute_read_query",
            "queryPlan": {
                "root_entity": "sample",
                "relation_path": [],
                "select_fields": ["sample.disease"],
                "aggregations": [],
                "filters": [],
                "group_by": [],
                "limit": 10,
            },
            "limit": 10,
        },
    }
    normalized = _normalise_materializer_response(action)
    assert isinstance(normalized, dict)
    assert "covariates" not in normalized
    assert "stratify_by" not in normalized
