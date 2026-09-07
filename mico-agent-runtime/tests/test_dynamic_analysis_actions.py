from __future__ import annotations

import ast
import json
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from mico_agent_runtime.contracts.analysis import AnalysisPlan, AnalysisResult
from mico_agent_runtime.contracts.generated_analysis import (
    AnalysisPlannerContext,
    GeneratedAnalysisPlan,
    GeneratedAnalysisPlannerResult,
    TypedAnalysisPlannerResult,
)
from mico_agent_runtime.contracts.materialization import AnalysisPlan as TypedAnalysisPlan
from mico_agent_runtime.contracts.materialization import QueryPlan
from mico_agent_runtime.graph.scientific_workflow import (
    _execute_analysis_action,
    _payload_semantic_fields,
)
from mico_agent_runtime.contracts.research import (
    AnalyzeProjectionAction,
    AnalyzeProjectionArguments,
    CompareGroupsAction,
    FinishAction,
    FinishArguments,
    InspectCohortAction,
    InspectCohortArguments,
    Observation,
    ProjectionAnalysisArguments,
    ResearchTask,
    ScientificObservationSummary,
    ScientificPlannerContext,
    validate_scientific_action,
)
from mico_agent_runtime.graph.generated_analysis import build_analysis_preview
from mico_agent_runtime.graph.generated_analysis import execute_generated_analysis
from mico_agent_runtime.ports.research_planner import (
    _canonicalize_cross_query_shape,
    _guard_dynamic_action,
    _normalize_generated_analysis_code,
    HttpResearchPlannerPort,
)
from mico_agent_runtime.ports.scientific_planner import ScientificPlannerResult
from mico_agent_runtime.runtime.scientific_service import ScientificRuntime
from mico_agent_runtime.contracts.trace_eval import TraceDecision
from mico_agent_runtime.transport.app import create_app
from tests.conftest import FakeJavaPort, completed_response
from tests.test_analysis_capability_registry import _catalog as capability_catalog
from tests.test_scientific_agent_loop import _semantic_catalog


NOW = datetime(2026, 8, 23, 12, 0, tzinfo=timezone.utc)


def test_generated_analysis_code_normalization_preserves_python_blocks() -> None:
    normalized = _normalize_generated_analysis_code(
        "result = {}\r\nif rows:\r\n\tresult['count'] = len(rows)\r\n"
    )

    assert "\nif rows:\n" in normalized
    ast.parse(normalized)


def test_http_analysis_planner_keeps_multiline_code_from_json() -> None:
    import httpx

    content = json.dumps({
        "language": "python",
        "analysisType": "bounded_summary",
        "code": "result = {}\nif rows:\n\tresult['count'] = len(rows)",
    })

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}]},
        )

    planner = HttpResearchPlannerPort(
        "https://example.invalid/v1",
        "deepseek-v4-flash",
        "unit-test-token",
        transport=httpx.MockTransport(handler),
    )
    try:
        result = planner.generate_analysis(AnalysisPlannerContext(
            questionSummary="bounded summary",
            workflow="compare_groups",
            columns=["a_value"],
            previewRows=[{"a_value": 1}],
        ))
    finally:
        planner.close()

    assert result.mode == "model"
    assert "\nif rows:\n" in result.plan.code
    ast.parse(result.plan.code)


def test_http_analysis_planner_accepts_extended_timeout() -> None:
    planner = HttpResearchPlannerPort(
        "https://example.invalid/v1",
        "gemini-3.5-flash-lite",
        "unit-test-token",
        timeout_seconds=300.0,
    )
    try:
        assert planner._client.timeout.read == 300.0
    finally:
        planner.close()


def test_http_analysis_planner_rejects_non_positive_timeout() -> None:
    with pytest.raises(ValueError, match="planner timeout is invalid"):
        HttpResearchPlannerPort(
            "https://example.invalid/v1",
            "gemini-3.5-flash-lite",
            "unit-test-token",
            timeout_seconds=0.0,
        )


def test_sandbox_allows_safe_none_comparison_in_generated_code() -> None:
    plan = GeneratedAnalysisPlan(
        language="python",
        analysisType="bounded_summary",
        code=(
            "values = [float(row['a_value']) for row in rows "
            "if row['a_value'] is not None]\n"
            "result = {'metrics': {'mean': sum(values) / len(values)}}"
        ),
    )

    result = execute_generated_analysis(
        plan,
        [{"a_value": 1.0}, {"a_value": None}, {"a_value": 3.0}],
        3,
    )

    assert result.metrics["mean"] == 2.0


def test_dynamic_preview_allows_only_catalog_proven_java_aliases() -> None:
    columns, rows = build_analysis_preview(
        {
            "columns": ["a_sample_age", "a_sample_patient_id", "unknown_value"],
            "rows": [{"a_sample_age": 30, "a_sample_patient_id": 99, "unknown_value": 1}],
        },
        allowed_columns={"a_sample_age"},
    )

    assert columns == ["a_sample_age"]
    assert rows == [{"a_sample_age": 30}]


def test_dynamic_java_aliases_map_back_to_verified_semantic_fields() -> None:
    fields, aliases = _payload_semantic_fields(
        {"schemaCatalog": _semantic_catalog()},
        {
            "columns": ["a_sample_metadata_gender", "a_sample_metadata_age_mean"],
            "rows": [],
        },
    )

    assert fields == ["sample_metadata.age", "sample_metadata.gender"]
    assert aliases == {"a_sample_metadata_age_mean", "a_sample_metadata_gender"}


def test_dynamic_guard_requires_numeric_read_after_non_numeric_java_result() -> None:
    context = ScientificPlannerContext(
        questionSummary="Compare the validated groups",
        intent="scientific_exploration",
        approvedActions=["inspect_cohort", "execute_read_query", "compare_groups"],
        remainingActionBudget=4,
        observations=[ScientificObservationSummary(
            observationId="observation-" + "a" * 32,
            actionName="execute_read_query",
            status="VALIDATED",
            source="java_controlled_read",
            rowCount=10,
            qualityCodes=["numeric_outcome_missing"],
        )],
    )

    assert _guard_dynamic_action(context, "compare_groups") == (
        "execute_read_query",
        "DYNAMIC_ANALYSIS_NUMERIC_READ_REQUIRED_REPAIRED",
    )


def test_dynamic_cross_validation_accepts_one_grouped_java_observation() -> None:
    observation = "observation-" + "b" * 32
    context = ScientificPlannerContext(
        questionSummary="Validate the numeric abundance outcome across projects",
        intent="focused_comparison",
        approvedActions=["execute_read_query", "cross_project_validate"],
        remainingActionBudget=3,
        observations=[ScientificObservationSummary(
            observationId=observation,
            actionName="execute_read_query",
            status="VALIDATED",
            source="java_controlled_read",
            rowCount=10,
            qualityCodes=["numeric_outcome_available"],
            queryPlanFields=["metadata.project", "abundance.value"],
        )],
    )

    assert _guard_dynamic_action(context, "cross_project_validate") == (
        "cross_project_validate",
        None,
    )


def test_cross_query_shape_normalization_preserves_model_selected_fields() -> None:
    context = ScientificPlannerContext(
        questionSummary="Validate abundance across projects",
        intent="focused_comparison",
        approvedActions=["execute_read_query"],
        remainingActionBudget=3,
        requiredSemanticFields=["metadata.project", "abundance.value"],
        requiredGroupField="metadata.project",
    )
    payload = {
        "actionName": "execute_read_query",
        "rationale": "read the selected fields",
        "arguments": {
            "actionName": "execute_read_query",
            "queryPlan": {
                "root_entity": "sample",
                "relation_path": ["sample_to_metadata", "sample_to_abundance"],
                "select_fields": ["metadata.project", "abundance.value"],
                "aggregations": [],
                "filters": [],
                "group_by": [],
                "limit": 100,
            },
            "limit": 100,
        },
    }

    normalized = _canonicalize_cross_query_shape(payload, context)
    plan = normalized["arguments"]["queryPlan"]
    assert plan["select_fields"] == ["metadata.project"]
    assert plan["group_by"] == ["metadata.project"]
    assert plan["aggregations"] == [{"field": "abundance.value", "op": "mean"}]


def test_cross_query_shape_keeps_raw_feature_projection_for_group_coverage_repair() -> None:
    context = ScientificPlannerContext(
        questionSummary="Fill the missing disease group with sample-level abundance rows",
        intent="focused_comparison",
        approvedActions=["execute_read_query"],
        remainingActionBudget=3,
        requiredSemanticFields=["sample.disease", "abundance.feature", "abundance.value"],
        requiredGroupField="sample.disease",
        executionFeedback=[
            "DYNAMIC_GROUP_COVERAGE_REQUIRED",
            "DYNAMIC_REQUESTED_GROUP_COVERAGE_REQUIRED",
        ],
    )
    payload = {
        "actionName": "execute_read_query",
        "rationale": "read the missing group",
        "arguments": {
            "actionName": "execute_read_query",
            "queryPlan": {
                "root_entity": "sample",
                "relation_path": ["sample_to_abundance"],
                "select_fields": ["sample.disease", "abundance.feature", "abundance.value"],
                "aggregations": [],
                "filters": [{"field": "sample.disease", "operator": "in", "value": ["T2D"]}],
                "group_by": [],
                "limit": 500,
            },
        },
    }

    normalized = _canonicalize_cross_query_shape(payload, context)
    assert normalized == payload


def _task() -> ResearchTask:
    return ResearchTask(
        runId="run-dynamic-analysis-00000000000000000000000000001",
        taskId="task-dynamic-analysis-00000000000000000000000000001",
        requesterId="principal-dynamic-analysis-00000000000000000000000001",
        traceId="trace-dynamic-analysis-00000000000000000000000000001",
        question="探索元数据覆盖后比较不同分组的微生物特征",
        requestedScopes=["mico:query:read", "mico:research:read"],
        allowedActions=["inspect_cohort", "compare_groups", "finish"],
        maxActions=3,
        createdAt=NOW,
    )


class DynamicPlanner:
    def __init__(self) -> None:
        self.calls = 0
        self.analysis_context: AnalysisPlannerContext | None = None

    def plan_action(self, context):
        self.calls += 1
        action_id = "action-" + f"{self.calls:032x}"
        if self.calls == 1:
            return ScientificPlannerResult(
                action=InspectCohortAction(
                    actionId=action_id,
                    actionName="inspect_cohort",
                    rationale="先检查可用元数据覆盖和分组分布",
                    arguments=InspectCohortArguments(
                        actionName="inspect_cohort",
                        sql="SELECT group_code, project_code, feature_name, abundance_value FROM approved_projection LIMIT 20",
                        limit=20,
                    ),
                ),
                mode="model",
            )
        if self.calls == 2:
            return ScientificPlannerResult(
                action=CompareGroupsAction(
                    actionId=action_id,
                    actionName="compare_groups",
                    rationale="仅基于已验证观察执行当前分组分析",
                    arguments=ProjectionAnalysisArguments(
                        actionName="compare_groups",
                        observationIds=[context.observations[-1].observationId],
                        analysisGoal="比较已返回分组的特征摘要",
                        dimensions=["group_code", "project_code"],
                    ),
                ),
                mode="model",
            )
        return ScientificPlannerResult(
            action=FinishAction(
                actionId=action_id,
                actionName="finish",
                rationale="当前有界分析已完成，停止继续调用",
                arguments=FinishArguments(actionName="finish", reasonCode="EVIDENCE_SUFFICIENT"),
            ),
            mode="model",
        )

    def generate_analysis(self, context: AnalysisPlannerContext) -> GeneratedAnalysisPlannerResult:
        self.analysis_context = context
        return GeneratedAnalysisPlannerResult(
            plan=GeneratedAnalysisPlan(
                language="python",
                analysisType="group_summary",
                code="result = {'metrics': {'row_count': len(rows)}, 'topFeatures': []}",
            ),
            mode="model",
        )


def test_dynamic_actions_are_closed_and_argument_discriminators_match() -> None:
    action = validate_scientific_action({
        "actionId": "action-" + "1" * 32,
        "actionName": "inspect_cohort",
        "rationale": "先检查元数据",
        "arguments": {
            "actionName": "inspect_cohort",
            "sql": "SELECT group_code FROM approved_projection LIMIT 10",
            "limit": 10,
        },
    })
    assert action.actionName == "inspect_cohort"

    with pytest.raises((ValidationError, ValueError)):
        validate_scientific_action({
            "actionId": "action-" + "2" * 32,
            "actionName": "compare_groups",
            "rationale": "比较已验证观察",
            "arguments": {
                "actionName": "stratified_analysis",
                "observationIds": ["observation-" + "3" * 32],
                "analysisGoal": "比较分组",
            },
        })


def test_analysis_plan_requires_observation_and_confounder_fields() -> None:
    with pytest.raises(ValidationError):
        AnalysisPlan(
            analysisId="analysis-" + "1" * 32,
            actionName="compare_groups",
            analysisGoal="比较分组",
        )
    with pytest.raises(ValidationError):
        AnalysisPlan(
            analysisId="analysis-" + "2" * 32,
            actionName="adjust_confounders",
            sourceObservationIds=["observation-" + "3" * 32],
            analysisGoal="调整混杂因素",
        )


def test_analysis_actions_require_their_minimum_observation_shape() -> None:
    observation = "observation-" + "3" * 32
    with pytest.raises(ValidationError):
        ProjectionAnalysisArguments(
            actionName="stratified_analysis",
            observationIds=[observation],
            analysisGoal="检查分层稳定性",
        )
    cross_project_valid = ProjectionAnalysisArguments(
        actionName="cross_project_validate",
        observationIds=[observation],
        analysisGoal="检查跨项目稳定性",
    )
    assert cross_project_valid.observationIds == [observation]
    cross_disease_valid = AnalysisPlan(
        analysisId="analysis-" + "4" * 32,
        actionName="cross_disease_validate",
        sourceObservationIds=[observation],
        analysisGoal="检查跨疾病可迁移性",
    )
    assert cross_disease_valid.sourceObservationIds == [observation]
    valid = ProjectionAnalysisArguments(
        actionName="stratified_analysis",
        observationIds=[observation],
        analysisGoal="检查分层稳定性",
        dimensions=["sample_metadata.project"],
    )
    assert valid.dimensions == ["sample_metadata.project"]

    confounder_valid = ProjectionAnalysisArguments(
        actionName="adjust_confounders",
        observationIds=[observation],
        analysisGoal="控制已批准的混杂维度",
        confounders=["sample_metadata.age", "gender"],
    )
    assert confounder_valid.confounders == ["sample_metadata.age", "gender"]


def test_scientific_loop_runs_metadata_inspection_then_generic_analysis() -> None:
    port = FakeJavaPort()

    def response(call):
        value = completed_response(call)
        value.data = {
            "columns": ["group_code", "project_code", "feature_name", "abundance_value"],
            "rows": [
                {"group_code": "g1", "project_code": "p1", "feature_name": "feature_a", "abundance_value": 0.7},
                {"group_code": "g2", "project_code": "p1", "feature_name": "feature_a", "abundance_value": 0.3},
            ],
        }
        value.rowCount = 2
        value.dataSnapshot.rowCount = 2
        return value

    port.response_factory = response
    planner = DynamicPlanner()
    result = ScientificRuntime(port, planner).run(_task())

    assert result["status"] == "COMPLETED"
    assert result["actionCount"] == 3
    assert port.calls == 1
    assert planner.analysis_context is not None
    assert planner.analysis_context.workflow == "compare_groups"
    report = result["report"]
    assert report.analysisResults[0].actionName == "compare_groups"
    assert report.analysisResults[0].sourceObservationIds
    assert report.analysisEvidence[0].snapshotPersistence == "transient"
    assert report.sourceMetadata[0].queryHash.startswith("sha256:")
    assert report.sourceMetadata[0].dataSnapshotId.startswith("transient-")


def test_sandbox_rejection_is_returned_as_opaque_feedback_before_fallback() -> None:
    class ReplanningAnalysisPlanner(DynamicPlanner):
        def __init__(self) -> None:
            super().__init__()
            self.analysis_contexts: list[AnalysisPlannerContext] = []

        def generate_analysis(self, context: AnalysisPlannerContext) -> GeneratedAnalysisPlannerResult:
            self.analysis_contexts.append(context)
            if len(self.analysis_contexts) == 1:
                return GeneratedAnalysisPlannerResult(
                    plan=GeneratedAnalysisPlan(
                        language="python",
                        analysisType="group_summary",
                        code="result = {'metrics': {}, 'topFeatures': []}; result.update({})",
                    ),
                    mode="model",
                )
            return GeneratedAnalysisPlannerResult(
                plan=GeneratedAnalysisPlan(
                    language="python",
                    analysisType="group_summary",
                    code="result = {'metrics': {'row_count': len(rows)}, 'topFeatures': []}",
                ),
                mode="model",
            )

    planner = ReplanningAnalysisPlanner()
    result = ScientificRuntime(FakeJavaPort(), planner).run(_task())

    assert result.status == "COMPLETED"
    assert len(planner.analysis_contexts) == 2
    assert planner.analysis_contexts[0].executionFeedback == []
    assert planner.analysis_contexts[1].executionFeedback == ["ANALYSIS_CODE_REJECTED"]
    assert "ANALYSIS_CODE_REPLAN" in result.fallbackCodes
    assert "ANALYSIS_CODE_DETERMINISTIC_FALLBACK" not in result.fallbackCodes


def test_analysis_result_does_not_accept_unbound_source_observation() -> None:
    with pytest.raises(ValidationError):
        AnalysisResult(
            analysisId="analysis-" + "1" * 32,
            actionName="compare_groups",
            status="COMPLETED",
            plannerMode="model",
            codeVersion="sandbox-python-v1",
            sourceObservationIds=["observation-" + "1" * 32],
            rowsAnalyzed=1,
            evidence=[],
            limitations=["input_projection_was_bounded"],
        )


def test_scientific_runtime_endpoint_uses_internal_auth_and_closed_report() -> None:
    port = FakeJavaPort()

    def response(call):
        value = completed_response(call)
        value.data = {
            "columns": ["group_code", "abundance_value"],
            "rows": [
                {"group_code": "g1", "abundance_value": 0.7},
                {"group_code": "g2", "abundance_value": 0.3},
            ],
        }
        value.rowCount = 2
        value.dataSnapshot.rowCount = 2
        return value

    port.response_factory = response
    app = create_app(
        env={"MICO_RUNTIME_INTERNAL_TOKEN": "runtime-token"},
        scientific_runtime=ScientificRuntime(port, DynamicPlanner()),
    )
    payload = _task().model_dump(mode="json")
    with TestClient(app) as client:
        assert client.post("/internal/runtime/scientific-runs", json=payload).status_code == 401
        invalid = client.post(
            "/internal/runtime/scientific-runs",
            json={**payload, "unknown": "field"},
            headers={"Authorization": "Bearer runtime-token"},
        )
        assert invalid.status_code == 400
        response = client.post(
            "/internal/runtime/scientific-runs",
            json=payload,
            headers={"Authorization": "Bearer runtime-token"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "COMPLETED"
        assert body["report"]["analysisResults"]
        assert "rawObservationPayloads" not in body
        assert "sourceSampleId" not in response.text


def test_dynamic_runtime_uses_typed_query_and_analysis_without_static_fallback() -> None:
    class TypedDynamicPlanner:
        dynamic_action_materialization = True

        def __init__(self) -> None:
            self.calls = 0
            self.analysis_calls = 0

        def plan_action(self, context):
            self.calls += 1
            action_id = "action-" + f"{self.calls:032x}"
            if self.calls == 1:
                return ScientificPlannerResult(
                    action=InspectCohortAction(
                        actionId=action_id,
                        actionName="inspect_cohort",
                        rationale="inspect the approved semantic read surface",
                        arguments=InspectCohortArguments(
                            actionName="inspect_cohort",
                            queryPlan=QueryPlan(
                                root_entity="sample_metadata",
                                select_fields=[
                                    "sample_metadata.gender",
                                    "sample_metadata.age",
                                ],
                                limit=20,
                            ),
                            limit=20,
                        ),
                    ),
                    mode="model",
                )
            if self.calls == 2:
                return ScientificPlannerResult(
                    action=CompareGroupsAction(
                        actionId=action_id,
                        actionName="compare_groups",
                        rationale="compare the validated observation",
                        arguments=ProjectionAnalysisArguments(
                            actionName="compare_groups",
                            observationIds=[context.observations[-1].observationId],
                            analysisGoal="compare the observed groups",
                        ),
                    ),
                    mode="model",
                )
            return ScientificPlannerResult(
                action=FinishAction(
                    actionId=action_id,
                    actionName="finish",
                    rationale="stop after the bounded analysis",
                    arguments=FinishArguments(
                        actionName="finish", reasonCode="EVIDENCE_SUFFICIENT"
                    ),
                ),
                mode="model",
            )

        def generate_typed_analysis(self, context: AnalysisPlannerContext):
            self.analysis_calls += 1
            return TypedAnalysisPlannerResult(
                plan=TypedAnalysisPlan(
                    analysis_type="group_comparison",
                    source_observation_ids=context.sourceObservationIds,
                    outcome="sample_metadata.age",
                    group_field="sample_metadata.gender",
                    metrics=["count", "mean", "effect_size"],
                ),
                mode="model",
            )

    port = FakeJavaPort()

    def response(call):
        value = completed_response(call)
        value.data = {
            "columns": ["a_sample_metadata_gender", "a_sample_metadata_age"],
            "rows": [
                {"a_sample_metadata_gender": "g1", "a_sample_metadata_age": 30},
                {"a_sample_metadata_gender": "g2", "a_sample_metadata_age": 40},
            ],
        }
        value.rowCount = 2
        value.dataSnapshot.rowCount = 2
        return value

    port.response_factory = response
    planner = TypedDynamicPlanner()
    result = ScientificRuntime(
        port,
        planner,
        schema_catalog=_semantic_catalog(),
    ).run(_task())

    assert result.status == "COMPLETED"
    assert result.actionCount == 3
    assert planner.analysis_calls == 1
    assert result.report.analysisResults[0].codeVersion == "typed-analysis-operator-v1"
    assert all(
        record.query_plan_validation_status in {"passed", None}
        for record in result.decisionRecords
    )
    assert "ANALYSIS_CODE_DETERMINISTIC_FALLBACK" not in result.fallbackCodes


def test_dynamic_runtime_rejects_legacy_sql_shape() -> None:
    class LegacyDynamicPlanner:
        dynamic_action_materialization = True

        def plan_action(self, _context):
            return ScientificPlannerResult(
                action=InspectCohortAction(
                    actionId="action-" + "e" * 32,
                    actionName="inspect_cohort",
                    rationale="legacy SQL must not enter the dynamic path",
                    arguments=InspectCohortArguments(
                        actionName="inspect_cohort",
                        sql="SELECT disease FROM patients LIMIT 10",
                        limit=10,
                    ),
                ),
                mode="model",
            )

    result = ScientificRuntime(
        FakeJavaPort(),
        LegacyDynamicPlanner(),
        schema_catalog=_semantic_catalog(),
    ).run(_task())

    assert result.status == "REJECTED"
    assert result.errorCode == "DYNAMIC_QUERY_PLAN_REQUIRED"


def test_dynamic_typed_analysis_executes_inferential_metric_without_sandbox_fallback() -> None:
    class UnsupportedMetricPlanner:
        dynamic_action_materialization = True

        def __init__(self) -> None:
            self.calls = 0
            self.typed_calls = 0
            self.code_calls = 0

        def plan_action(self, context):
            self.calls += 1
            action_id = "action-" + f"{self.calls:032x}"
            if self.calls == 1:
                return ScientificPlannerResult(
                    action=InspectCohortAction(
                        actionId=action_id,
                        actionName="inspect_cohort",
                        rationale="inspect the approved semantic read surface",
                        arguments=InspectCohortArguments(
                            actionName="inspect_cohort",
                            queryPlan=QueryPlan(
                                root_entity="sample_metadata",
                                select_fields=["sample_metadata.age"],
                                limit=20,
                            ),
                            limit=20,
                        ),
                    ),
                    mode="model",
                )
            if self.calls == 2:
                return ScientificPlannerResult(
                    action=CompareGroupsAction(
                        actionId=action_id,
                        actionName="compare_groups",
                        rationale="compare the validated observation",
                        arguments=ProjectionAnalysisArguments(
                            actionName="compare_groups",
                            observationIds=[context.observations[-1].observationId],
                            analysisGoal="compare the observed groups",
                        ),
                    ),
                    mode="model",
                )
            return ScientificPlannerResult(
                action=FinishAction(
                    actionId=action_id,
                    actionName="finish",
                    rationale="stop after the bounded analysis",
                    arguments=FinishArguments(
                        actionName="finish", reasonCode="EVIDENCE_SUFFICIENT"
                    ),
                ),
                mode="model",
            )

        def generate_typed_analysis(self, context: AnalysisPlannerContext):
            self.typed_calls += 1
            return TypedAnalysisPlannerResult(
                plan=TypedAnalysisPlan(
                    analysis_type="group_comparison",
                    source_observation_ids=context.sourceObservationIds,
                    outcome="sample_metadata.age",
                    group_field="sample_metadata.gender",
                    metrics=["p_value"],
                ),
                mode="model",
            )

        def generate_analysis(self, _context: AnalysisPlannerContext):
            self.code_calls += 1
            return GeneratedAnalysisPlannerResult(
                plan=GeneratedAnalysisPlan(
                    language="python",
                    analysisType="bounded_unsupported_metric",
                    code="result = {'metrics': {'row_count': len(rows)}, 'topFeatures': []}",
                ),
                mode="model",
            )

    port = FakeJavaPort()

    def response(call):
        value = completed_response(call)
        value.data = {
            "columns": ["a_sample_metadata_gender", "a_sample_metadata_age"],
            "rows": [
                {"a_sample_metadata_gender": "g1", "a_sample_metadata_age": 30},
                {"a_sample_metadata_gender": "g2", "a_sample_metadata_age": 40},
            ],
        }
        value.rowCount = 2
        value.dataSnapshot.rowCount = 2
        return value

    port.response_factory = response
    planner = UnsupportedMetricPlanner()
    result = ScientificRuntime(
        port,
        planner,
        schema_catalog=_semantic_catalog(),
    ).run(_task())

    assert result.status == "COMPLETED"
    assert planner.typed_calls == 1
    assert planner.code_calls == 0
    assert result.report.analysisResults[0].codeVersion == "typed-analysis-operator-v1"
    assert "ANALYSIS_CODE_DETERMINISTIC_FALLBACK" not in result.fallbackCodes


def test_dynamic_typed_failure_does_not_fallback_to_generated() -> None:
    class ReplanningTypedPlanner:
        dynamic_action_materialization = True

        def __init__(self) -> None:
            self.calls = 0
            self.typed_calls = 0
            self.code_contexts: list[AnalysisPlannerContext] = []

        def plan_action(self, context):
            self.calls += 1
            action_id = "action-" + f"{self.calls:032x}"
            if self.calls == 1:
                return ScientificPlannerResult(
                    action=InspectCohortAction(
                        actionId=action_id,
                        actionName="inspect_cohort",
                        rationale="inspect the approved semantic read surface",
                        arguments=InspectCohortArguments(
                            actionName="inspect_cohort",
                            queryPlan=QueryPlan(
                                root_entity="sample_metadata",
                                select_fields=["sample_metadata.age"],
                                limit=20,
                            ),
                            limit=20,
                        ),
                    ),
                    mode="model",
                )
            if self.calls == 2:
                return ScientificPlannerResult(
                    action=CompareGroupsAction(
                        actionId=action_id,
                        actionName="compare_groups",
                        rationale="use the validated observation",
                        arguments=ProjectionAnalysisArguments(
                            actionName="compare_groups",
                            observationIds=[context.observations[-1].observationId],
                            analysisGoal="bounded projection summary",
                        ),
                    ),
                    mode="model",
                )
            return ScientificPlannerResult(
                action=FinishAction(
                    actionId=action_id,
                    actionName="finish",
                    rationale="stop after the bounded analysis",
                    arguments=FinishArguments(
                        actionName="finish", reasonCode="EVIDENCE_SUFFICIENT"
                    ),
                ),
                mode="model",
            )

        def generate_typed_analysis(self, context: AnalysisPlannerContext):
            self.typed_calls += 1
            return TypedAnalysisPlannerResult(
                plan=TypedAnalysisPlan(
                    analysis_type="group_comparison",
                    source_observation_ids=context.sourceObservationIds,
                    outcome="sample_metadata.age",
                    group_field="sample_metadata.gender",
                    metrics=["confidence_interval"],
                ),
                mode="model",
            )

        def generate_analysis(self, context: AnalysisPlannerContext):
            self.code_contexts.append(context)
            code = (
                "result = {'metrics': {}, 'topFeatures': []}; result.update({})"
                if len(self.code_contexts) == 1
                else "result = {'metrics': {'row_count': len(rows)}, 'topFeatures': []}"
            )
            return GeneratedAnalysisPlannerResult(
                plan=GeneratedAnalysisPlan(
                    language="python",
                    analysisType="bounded_projection",
                    code=code,
                ),
                mode="model",
            )

    port = FakeJavaPort()

    def response(call):
        value = completed_response(call)
        value.data = {
            "columns": ["a_sample_metadata_age"],
            "rows": [
                {"a_sample_metadata_age": 30},
                {"a_sample_metadata_age": 40},
            ],
        }
        value.rowCount = 2
        value.dataSnapshot.rowCount = 2
        return value

    port.response_factory = response
    planner = ReplanningTypedPlanner()
    result = ScientificRuntime(
        port,
        planner,
        schema_catalog=_semantic_catalog(),
    ).run(_task())

    assert result.status == "FAILED"
    assert planner.typed_calls == 1
    assert planner.code_contexts == []
    assert result.errorCode == "ANALYSIS_TYPED_EXECUTION_FAILED"


def test_dynamic_typed_materializer_cannot_change_action_family() -> None:
    observation_id = "observation-" + "f" * 32
    action = CompareGroupsAction(
        actionId="action-" + "f" * 32,
        actionName="compare_groups",
        rationale="compare the validated groups",
        arguments=ProjectionAnalysisArguments(
            actionName="compare_groups",
            observationIds=[observation_id],
            analysisGoal="compare the validated groups",
        ),
    )
    observation = Observation(
        observationId=observation_id,
        actionId="action-" + "e" * 32,
        actionName="execute_read_query",
        status="VALIDATED",
        source="java_controlled_read",
        queryHash="sha256:" + "a" * 64,
        rowCount=2,
        generatedAt=NOW,
        schemaVersion="schema-v1",
        dataSnapshotId="transient-11111111-1111-4111-8111-111111111111",
        snapshotPersistence="transient",
        queryPlanFields=["sample_metadata.gender", "sample_metadata.age"],
    )

    class WrongTypedMaterializer:
        dynamic_action_materialization = True

        def generate_typed_analysis(self, context: AnalysisPlannerContext):
            return TypedAnalysisPlannerResult(
                plan=TypedAnalysisPlan(
                    analysis_type="confounder_adjustment",
                    source_observation_ids=context.sourceObservationIds,
                    outcome="sample_metadata.age",
                    group_field="sample_metadata.gender",
                    covariates=["sample_metadata.age"],
                    metrics=["count"],
                ),
                mode="model",
            )

    state = {
        "request": _task().model_copy(update={"allowedActions": ["compare_groups"]}),
        "schemaCatalog": _semantic_catalog(),
        "observations": [observation],
        "rawObservationPayloads": {
            observation_id: {
                "columns": ["a_sample_metadata_gender", "a_sample_metadata_age"],
                "rows": [
                    {"a_sample_metadata_gender": "g1", "a_sample_metadata_age": 30},
                    {"a_sample_metadata_gender": "g2", "a_sample_metadata_age": 40},
                ],
            }
        },
        "analysisPlans": [],
        "analysisResults": [],
        "analysisEvidence": [],
        "fallbackCodes": [],
        "decisionRecords": [],
        "currentAction": action,
    }
    result = _execute_analysis_action(state, action, WrongTypedMaterializer())
    assert result["status"] == "REJECTED"
    assert result["errorCode"] == "MATERIALIZATION_ACTION_MISMATCH"


def test_dynamic_typed_compare_plan_executes_typed_without_generated_fallback() -> None:
    """Exercise the complete registry-to-executor boundary for ``effect``."""

    observation_id = "observation-" + "9" * 32
    action = CompareGroupsAction(
        actionId="action-" + "9" * 32,
        actionName="compare_groups",
        rationale="compare the validated groups",
        arguments=ProjectionAnalysisArguments(
            actionName="compare_groups",
            observationIds=[observation_id],
            analysisGoal="compare the validated groups",
        ),
    )
    observation = Observation(
        observationId=observation_id,
        actionId="action-" + "8" * 32,
        actionName="execute_read_query",
        status="VALIDATED",
        source="java_controlled_read",
        queryHash="sha256:" + "9" * 64,
        rowCount=4,
        generatedAt=NOW,
        schemaVersion="schema-v1",
        dataSnapshotId="transient-99999999-9999-4999-8999-999999999999",
        snapshotPersistence="transient",
        queryPlanFields=["sample.disease", "abundance.value"],
    )

    class EffectTypedMaterializer:
        dynamic_action_materialization = True
        allow_deterministic_materializer_fallback = False

        def __init__(self) -> None:
            self.typed_calls = 0
            self.generated_calls = 0

        def generate_typed_analysis(self, context: AnalysisPlannerContext):
            self.typed_calls += 1
            return TypedAnalysisPlannerResult(
                plan=TypedAnalysisPlan(
                    analysis_type="group_comparison",
                    source_observation_ids=context.sourceObservationIds,
                    outcome="abundance.value",
                    group_field="sample.disease",
                    metrics=["effect", "p_value"],
                ),
                mode="model",
            )

        def generate_analysis(self, _context: AnalysisPlannerContext):
            self.generated_calls += 1
            raise AssertionError("typed compare must not call generated Python")

    planner = EffectTypedMaterializer()
    result = _execute_analysis_action(
        {
            "request": _task(),
            "schemaCatalog": capability_catalog(),
            "observations": [observation],
            "rawObservationPayloads": {
                observation_id: {
                    "columns": ["a_sample_disease", "a_abundance_value"],
                    "rows": [
                        {"a_sample_disease": "T2D", "a_abundance_value": 0.2},
                        {"a_sample_disease": "T2D", "a_abundance_value": 0.3},
                        {"a_sample_disease": "Healthy", "a_abundance_value": 0.8},
                        {"a_sample_disease": "Healthy", "a_abundance_value": 0.9},
                    ],
                },
            },
            "analysisPlans": [],
            "analysisResults": [],
            "analysisEvidence": [],
            "fallbackCodes": [],
            "decisionRecords": [],
            "currentAction": action,
        },
        action,
        planner,
    )

    assert result.get("errorCode") is None
    assert planner.typed_calls == 1
    assert planner.generated_calls == 0
    assert result["pendingPayload"].execution_mode == "typed"
    assert result["pendingPayload"].codeVersion == "typed-analysis-operator-v1"
    assert result["pendingPayload"].group_results
    assert result["pendingPayload"].metrics["effect"] == result["pendingPayload"].metrics["effect_size"]


def test_registry_generated_projection_executes_program_before_typed_operator() -> None:
    observation_id = "observation-" + "7" * 32
    action = AnalyzeProjectionAction(
        actionId="action-" + "7" * 32,
        actionName="analyze_projection",
        rationale="summarize the approved abundance projection",
        arguments=AnalyzeProjectionArguments(
            actionName="analyze_projection",
            observationId=observation_id,
            analysisGoal="summarize the approved abundance projection",
        ),
    )
    observation = Observation(
        observationId=observation_id,
        actionId="action-" + "6" * 32,
        actionName="execute_read_query",
        status="VALIDATED",
        source="java_controlled_read",
        queryHash="sha256:" + "7" * 64,
        rowCount=2,
        generatedAt=NOW,
        schemaVersion="schema-v1",
        dataSnapshotId="transient-77777777-7777-4777-8777-777777777777",
        snapshotPersistence="transient",
        queryPlanFields=["abundance.value"],
    )

    class ProjectionGeneratedMaterializer:
        dynamic_action_materialization = True
        allow_deterministic_materializer_fallback = False

        def __init__(self) -> None:
            self.typed_calls = 0
            self.code_calls = 0

        def generate_typed_analysis(self, context: AnalysisPlannerContext):
            self.typed_calls += 1
            return TypedAnalysisPlannerResult(
                plan=TypedAnalysisPlan(
                    analysis_type="projection",
                    source_observation_ids=context.sourceObservationIds,
                    outcome="abundance.value",
                    metrics=["count", "mean"],
                    analysis_goal="summarize the approved abundance projection",
                ),
                mode="model",
            )

        def generate_analysis(self, _context: AnalysisPlannerContext):
            self.code_calls += 1
            return GeneratedAnalysisPlannerResult(
                plan=GeneratedAnalysisPlan(
                    language="python",
                    analysisType="projection",
                    code=(
                        'values = [float(row["a_abundance_value"]) for row in rows]\n'
                        'result = {"metrics": {"count": float(len(values)), '
                        '"mean": sum(values) / len(values)}, "used_row_count": len(values)}'
                    ),
                ),
                mode="model",
            )

    planner = ProjectionGeneratedMaterializer()
    state = {
        "request": _task().model_copy(update={"allowedActions": ["analyze_projection"]}),
        "schemaCatalog": capability_catalog(),
        "observations": [observation],
        "rawObservationPayloads": {
            observation_id: {
                "columns": ["a_abundance_value", "raw_patient_id"],
                "rows": [
                    {"a_abundance_value": 0.2, "raw_patient_id": "not-bound"},
                    {"a_abundance_value": 0.8, "raw_patient_id": "not-bound"},
                ],
            }
        },
        "analysisPlans": [], "analysisResults": [], "analysisEvidence": [],
        "generatedAnalysisPrograms": [], "fallbackCodes": [],
        "decisionRecords": [TraceDecision(
            observationStateCode="OBSERVATION_READY",
            allowedActions=["analyze_projection"],
            chosenAction="analyze_projection",
        )],
        "currentAction": action,
    }
    result = _execute_analysis_action(state, action, planner)

    assert result.get("errorCode") is None
    assert planner.typed_calls == 1
    assert planner.code_calls == 1
    assert result["pendingPayload"].execution_mode == "generated"
    assert result["pendingPayload"].scientific_result_valid is True
    assert result["pendingPayload"].scientific_conclusion_eligible is False
    assert result["generatedAnalysisPrograms"][0]["required_columns"] == ["a_abundance_value"]
    assert result["decisionRecords"][-1].generated_code_hash
    assert result["decisionRecords"][-1].generated_program_hash


def test_dynamic_typed_materializer_deterministic_fallback_keeps_action() -> None:
    observation_id = "observation-" + "1" * 32
    action = CompareGroupsAction(
        actionId="action-" + "1" * 32,
        actionName="compare_groups",
        rationale="compare the validated groups",
        arguments=ProjectionAnalysisArguments(
            actionName="compare_groups",
            observationIds=[observation_id],
            analysisGoal="compare the validated groups",
        ),
    )
    observation = Observation(
        observationId=observation_id,
        actionId="action-" + "2" * 32,
        actionName="execute_read_query",
        status="VALIDATED",
        source="java_controlled_read",
        queryHash="sha256:" + "b" * 64,
        rowCount=2,
        generatedAt=NOW,
        schemaVersion="schema-v1",
        dataSnapshotId="transient-22222222-2222-4222-8222-222222222222",
        snapshotPersistence="transient",
        queryPlanFields=["sample_metadata.gender", "sample_metadata.age"],
    )

    class DeterministicTypedMaterializer:
        dynamic_action_materialization = True

        def generate_typed_analysis(self, context: AnalysisPlannerContext):
            return TypedAnalysisPlannerResult(
                plan=TypedAnalysisPlan(
                    analysis_type="group_comparison",
                    source_observation_ids=context.sourceObservationIds,
                    outcome="sample_metadata.age",
                    group_field="sample_metadata.gender",
                    metrics=["count", "mean", "effect_size"],
                ),
                mode="deterministic",
                fallbackCode="ANALYSIS_PLANNER_DETERMINISTIC_FALLBACK",
            )

    state = {
        "request": _task().model_copy(update={"allowedActions": ["compare_groups"]}),
        "schemaCatalog": _semantic_catalog(),
        "observations": [observation],
        "rawObservationPayloads": {
            observation_id: {
                "columns": ["a_sample_metadata_gender", "a_sample_metadata_age"],
                "rows": [
                    {"a_sample_metadata_gender": "g1", "a_sample_metadata_age": 30},
                    {"a_sample_metadata_gender": "g2", "a_sample_metadata_age": 40},
                ],
            }
        },
        "analysisPlans": [],
        "analysisResults": [],
        "analysisEvidence": [],
        "fallbackCodes": [],
        "decisionRecords": [],
        "currentAction": action,
    }
    result = _execute_analysis_action(state, action, DeterministicTypedMaterializer())
    assert result.get("errorCode") is None
    assert result["pendingPayload"].analysis_type == "group_comparison"
    assert result["pendingPayload"].plannerMode == "deterministic"
    assert "ANALYSIS_PLANNER_DETERMINISTIC_FALLBACK" in result["fallbackCodes"]
