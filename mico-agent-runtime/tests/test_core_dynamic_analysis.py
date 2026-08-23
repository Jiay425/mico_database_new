from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from mico_agent_runtime.contracts.generated_analysis import (
    AnalysisPlannerContext,
    GeneratedAnalysisPlan,
    GeneratedAnalysisPlannerResult,
)
from mico_agent_runtime.contracts.intent import (
    IntentPlannerContext,
    IntentRoutePlan,
    IntentTaskRequest,
)
from mico_agent_runtime.graph.intent_workflow import redact_for_planner
from mico_agent_runtime.ports.research_planner import (
    DeterministicIntentPlanner,
    IntentPlannerResult,
)
from mico_agent_runtime.runtime.intent_service import IntentRuntime
from mico_agent_runtime.transport.app import create_app
from tests.conftest import FakeJavaPort, completed_response, make_call


NOW = datetime(2026, 8, 22, tzinfo=timezone.utc)


def request(question: str = "比较不同研究条件下的微生物丰度") -> IntentTaskRequest:
    return IntentTaskRequest(
        runId="run-core-00000000000000000000000000000001",
        taskId="task-core-00000000000000000000000000000001",
        requesterId="principal-00000000000000000000000000000001",
        requestedScopes=["mico:query:read"],
        question=question,
        allowedWorkflows=["dynamic_read_query"],
        createdAt=NOW,
        traceId="trace-core-00000000000000000000000000000001",
    )


class FakeModelPlanner:
    def __init__(self) -> None:
        self.intent_context: IntentPlannerContext | None = None
        self.analysis_context: AnalysisPlannerContext | None = None

    def route_intent(self, context: IntentPlannerContext) -> IntentPlannerResult:
        self.intent_context = context
        return IntentPlannerResult(
            plan=IntentRoutePlan(
                workflow="dynamic_read_query",
                responseMode="structured_analysis_job",
                safetyProfile="non_diagnostic",
                sqlDraft=(
                    "SELECT feature_name, abundance_value "
                    "FROM approved_feature_projection LIMIT 10"
                ),
            ),
            mode="model",
        )

    def generate_analysis(self, context: AnalysisPlannerContext) -> GeneratedAnalysisPlannerResult:
        self.analysis_context = context
        return GeneratedAnalysisPlannerResult(
            plan=GeneratedAnalysisPlan(
                language="python",
                analysisType="descriptive_summary",
                code=(
                    "result = {'metrics': {'row_count': len(rows)}, "
                    "'topFeatures': []}"
                ),
            ),
            mode="model",
        )


def test_model_intent_sql_java_preview_and_generated_python_form_one_graph() -> None:
    planner = FakeModelPlanner()
    port = FakeJavaPort()

    def response(call):
        value = completed_response(call)
        value.data = {
            "columns": ["feature_name", "abundance_value"],
            "rows": [
                {"feature_name": "feature_a", "abundance_value": 0.75},
                {"feature_name": "feature_b", "abundance_value": 0.25},
            ],
        }
        value.rowCount = 2
        value.dataSnapshot.rowCount = 2
        return value

    port.response_factory = response
    result = IntentRuntime(port, planner).run(request())

    assert result.status == "COMPLETED"
    assert result.workflow == "dynamic_read_query"
    assert result.plannerMode == "model"
    assert port.calls == 1
    assert port.last_call is not None
    assert port.last_call.toolName == "execute_read_query"
    assert port.last_call.arguments.sql.startswith("SELECT")
    assert result.report is not None
    assert result.report.rowCount == 2
    assert result.report.generatedAnalysis is not None
    assert result.report.generatedAnalysis.metrics["row_count"] == 2
    assert planner.intent_context is not None
    assert planner.analysis_context is not None


def test_planner_context_redacts_identifiers_before_model_access() -> None:
    redacted = redact_for_planner(
        "分析 SRR1518476 sourceSampleId=secret-value internalRecordId=18423 "
        "以及 https://internal.invalid/path"
    )
    assert "SRR1518476" not in redacted
    assert "secret-value" not in redacted
    assert "18423" not in redacted
    assert "https://" not in redacted


def test_deterministic_mode_is_explicit_and_does_not_fabricate_sql() -> None:
    port = FakeJavaPort()
    result = IntentRuntime(port, DeterministicIntentPlanner()).run(request())

    assert result.status == "REJECTED"
    assert result.errorCode == "QUERY_PLAN_REQUIRED"
    assert result.plannerMode == "deterministic"
    assert port.calls == 0


def test_runtime_endpoint_is_closed_and_fail_closed() -> None:
    port = FakeJavaPort()
    app = create_app(
        env={"MICO_RUNTIME_INTERNAL_TOKEN": "runtime-token"},
        intent_runtime=IntentRuntime(port, DeterministicIntentPlanner()),
    )
    with TestClient(app) as client:
        disabled = client.post(
            "/internal/runtime/intent-runs",
            json=request().model_dump(mode="json"),
        )
        assert disabled.status_code == 401
        invalid = client.post(
            "/internal/runtime/intent-runs",
            json={**request().model_dump(mode="json"), "unknown": "field"},
            headers={"Authorization": "Bearer runtime-token"},
        )
        assert invalid.status_code == 400
        valid = client.post(
            "/internal/runtime/intent-runs",
            json=request().model_dump(mode="json"),
            headers={"Authorization": "Bearer runtime-token"},
        )
        assert valid.status_code == 200
        assert valid.json()["errorCode"] == "QUERY_PLAN_REQUIRED"
    assert port.calls == 0
