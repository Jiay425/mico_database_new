"""Run the four deterministic P6 demo scenarios without external services.

This is an evaluation/demo asset, not a production adapter. It imports the
existing test-only Java Port fixtures so the demos exercise the same closed
LangGraph contracts without a Java process, network, database, or LLM.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from mico_agent_runtime.contracts.intent import (
    IntentPlannerContext,
    IntentRoutePlan,
    IntentTaskRequest,
)
from mico_agent_runtime.ports.research_planner import (
    DeterministicResearchPlanner,
    DeterministicResearchRoutePlanner,
    IntentPlannerResult,
)
from mico_agent_runtime.runtime.cohort_service import CohortRuntime
from mico_agent_runtime.runtime.differential_service import DifferentialRuntime
from mico_agent_runtime.runtime.intent_service import IntentRuntime
from mico_agent_runtime.runtime.research_service import ResearchRuntime

from tests.conftest import FakeJavaPort, completed_response
from tests.test_cohort_agent import FakeCohortPort, make_request as make_cohort_request
from tests.test_differential_agent import (
    FakeDifferentialPort,
    request as make_differential_request,
)
from tests.test_intent_routing import request as make_intent_request
from tests.test_research_agent import ResearchFakePort, make_request as make_research_request


@dataclass(frozen=True)
class DemoSummary:
    demoId: str
    status: str
    workflow: str | None
    toolCallCount: int
    snapshotCount: int
    snapshotPersistence: str | None
    replayable: bool
    plannerMode: str | None
    errorCode: str | None


class _DynamicQueryPlanner:
    """Model-shaped fixture: it proposes a query plan, but never calls a model."""

    def route_intent(self, _context: IntentPlannerContext) -> IntentPlannerResult:
        return IntentPlannerResult(
            plan=IntentRoutePlan(
                workflow="dynamic_read_query",
                responseMode="structured_analysis_job",
                safetyProfile="non_diagnostic",
                sqlDraft=(
                    "SELECT microbe_name_standard AS taxon_name, "
                    "sample_group AS group_key, abundance_value "
                    "FROM bounded_projection LIMIT 4"
                ),
            ),
            mode="model",
        )


class _DynamicQueryPort(FakeJavaPort):
    def execute(self, call):  # type: ignore[no-untyped-def]
        response = completed_response(call)
        response.data = {
            "columns": ["taxon_name", "group_key", "abundance_value"],
            "rows": [
                {"taxon_name": "Bacteroides vulgatus", "group_key": "case", "abundance_value": 0.8},
                {"taxon_name": "Bacteroides vulgatus", "group_key": "case", "abundance_value": 0.7},
                {"taxon_name": "Bacteroides vulgatus", "group_key": "control", "abundance_value": 0.2},
                {"taxon_name": "Bacteroides vulgatus", "group_key": "control", "abundance_value": 0.1},
            ],
        }
        self.last_call = call
        self.calls += 1
        return response


def _summary(demo_id: str, result: Any, tool_call_count: int, workflow: str) -> DemoSummary:
    report = getattr(result, "report", None)
    snapshot_values: list[Any] = []
    if report is not None:
        for field in ("evidence", "toolChain"):
            values = getattr(report, field, None)
            if isinstance(values, list):
                snapshot_values.extend(values)
        evidence = getattr(report, "evidence", None)
        if evidence is not None and not isinstance(evidence, list):
            snapshot_values.append(evidence)
        if getattr(report, "dataSnapshotId", None) is not None:
            snapshot_values.append(report)
    persistence = None
    persistence_values = {
        getattr(item, "snapshotPersistence", None)
        for item in snapshot_values
        if getattr(item, "snapshotPersistence", None) is not None
    }
    if len(persistence_values) == 1:
        persistence = next(iter(persistence_values))
    return DemoSummary(
        demoId=demo_id,
        status=str(getattr(result, "status", "FAILED")),
        workflow=getattr(result, "workflow", None) or workflow,
        toolCallCount=tool_call_count,
        snapshotCount=len(snapshot_values),
        snapshotPersistence=persistence,
        replayable=bool(getattr(report, "replayable", False)) if report is not None else False,
        plannerMode=getattr(result, "plannerMode", None),
        errorCode=getattr(result, "errorCode", None),
    )


def run_demos() -> list[DemoSummary]:
    research_port = ResearchFakePort()
    research = ResearchRuntime(
        research_port, DeterministicResearchPlanner()
    ).run(make_research_request())

    cohort_port = FakeCohortPort()
    cohort = CohortRuntime(
        cohort_port, DeterministicResearchRoutePlanner()
    ).run(make_cohort_request())

    differential_port = FakeDifferentialPort()
    differential = DifferentialRuntime(differential_port).run(make_differential_request())

    dynamic_port = _DynamicQueryPort()
    dynamic_request: IntentTaskRequest = make_intent_request(
        requestedScopes=["mico:query:read"],
        allowedWorkflows=["dynamic_read_query"],
        question="比较两组物种丰度的科研差异",
    )
    dynamic = IntentRuntime(dynamic_port, _DynamicQueryPlanner()).run(dynamic_request)

    return [
        _summary("sample_evidence", research, len(research_port.calls),
                 "sample_evidence_investigation"),
        _summary("t2d_healthy_feasibility", cohort, len(cohort_port.calls),
                 "t2d_healthy_cohort_feasibility"),
        _summary("t2d_differential", differential, len(differential_port.calls),
                 "t2d_differential_analysis"),
        _summary("dynamic_read_query", dynamic, dynamic_port.calls, "dynamic_read_query"),
    ]


if __name__ == "__main__":
    print(json.dumps([asdict(item) for item in run_demos()], ensure_ascii=False, sort_keys=True))
