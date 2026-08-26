from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from mico_agent_runtime.contracts.analysis import AnalysisPlan, AnalysisResult
from mico_agent_runtime.contracts.generated_analysis import (
    AnalysisPlannerContext,
    GeneratedAnalysisPlan,
    GeneratedAnalysisPlannerResult,
)
from mico_agent_runtime.contracts.research import (
    CompareGroupsAction,
    FinishAction,
    FinishArguments,
    InspectCohortAction,
    InspectCohortArguments,
    ProjectionAnalysisArguments,
    ResearchTask,
    validate_scientific_action,
)
from mico_agent_runtime.ports.scientific_planner import ScientificPlannerResult
from mico_agent_runtime.runtime.scientific_service import ScientificRuntime
from mico_agent_runtime.transport.app import create_app
from tests.conftest import FakeJavaPort, completed_response


NOW = datetime(2026, 8, 23, 12, 0, tzinfo=timezone.utc)


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
    with pytest.raises(ValidationError):
        ProjectionAnalysisArguments(
            actionName="cross_project_validate",
            observationIds=[observation],
            analysisGoal="检查跨项目稳定性",
        )
    with pytest.raises(ValidationError):
        AnalysisPlan(
            analysisId="analysis-" + "4" * 32,
            actionName="cross_disease_validate",
            sourceObservationIds=[observation],
            analysisGoal="检查跨疾病可迁移性",
        )
    valid = ProjectionAnalysisArguments(
        actionName="stratified_analysis",
        observationIds=[observation],
        analysisGoal="检查分层稳定性",
        dimensions=["project_code"],
    )
    assert valid.dimensions == ["project_code"]


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
