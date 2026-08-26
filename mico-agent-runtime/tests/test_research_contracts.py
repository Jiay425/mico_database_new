from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from mico_agent_runtime.contracts.research import (
    Observation,
    ResearchEvidenceBinding,
    ResearchExplorationReport,
    ResearchPlannerContext,
    ResearchTask,
    validate_scientific_action,
)
from mico_agent_runtime.contracts.graph_rag import GroundedClaim, ReasoningStep
from mico_agent_runtime.contracts.unified_evidence import merge_unified_evidence
from tests.test_unified_evidence import _item, _path


NOW = datetime(2026, 8, 23, 12, 0, tzinfo=timezone.utc)
HASH = "sha256:" + "a" * 64


def test_research_task_is_closed_and_has_generic_budgeted_actions() -> None:
    task = ResearchTask(
        runId="run-001",
        taskId="task-001",
        requesterId="principal-001",
        traceId="trace-001",
        question="比较不同研究项目中的微生物丰度差异，并检查稳健性",
        requestedScopes=["mico:research:read", "mico:query:read"],
        allowedActions=["execute_read_query", "analyze_projection", "finish"],
        maxActions=6,
        createdAt=NOW,
    )
    assert task.dataContractVersion == "v1"
    assert task.maxActions == 6

    with pytest.raises(ValidationError):
        ResearchTask(
            **task.model_dump(),
            freeSql="SELECT 1",
        )


def test_action_union_accepts_only_the_dedicated_dynamic_query_action() -> None:
    action = validate_scientific_action(
        {
            "actionId": "action-" + "1" * 32,
            "actionName": "execute_read_query",
            "rationale": "先获取当前问题所需的有界数据事实",
            "arguments": {
                "actionName": "execute_read_query",
                "sql": "SELECT disease, COUNT(*) AS n FROM patients GROUP BY disease LIMIT 20",
                "limit": 20,
            },
        }
    )
    assert action.actionName == "execute_read_query"

    with pytest.raises(ValidationError):
        validate_scientific_action(
            {
                "actionId": "action-" + "1" * 32,
                "actionName": "execute_read_query",
                "rationale": "读取数据",
                "arguments": {
                    "actionName": "execute_read_query",
                    "sql": "DELETE FROM patients LIMIT 1",
                },
            }
        )


def test_observation_preserves_transient_snapshot_and_real_version_token() -> None:
    observation = Observation(
        observationId="observation-" + "2" * 32,
        actionId="action-" + "1" * 32,
        actionName="execute_read_query",
        status="VALIDATED",
        source="java_controlled_read",
        queryHash=HASH,
        rowCount=20,
        generatedAt=NOW,
        schemaVersion="p2j-research-contract-v1",
        dataSnapshotId="transient-550e8400-e29b-41d4-a716-446655440000",
        snapshotPersistence="transient",
        featureVersion="meta2db-species-v1",
        sourceBatch="meta2db-species-20260726:2015_Castro-NallarE",
        sampleRecordCount=20,
        sampleKeyCount=20,
    )
    assert observation.replayable is False
    assert observation.sourceBatch.endswith("2015_Castro-NallarE")

    invalid_observation = observation.model_dump()
    invalid_observation["snapshotPersistence"] = None
    with pytest.raises(ValidationError):
        Observation.model_validate(invalid_observation)


def test_report_requires_known_evidence_bindings() -> None:
    binding = ResearchEvidenceBinding(
        bindingId="binding-" + "3" * 32,
        observationId="observation-" + "2" * 32,
        evidenceReference="evidence-" + "4" * 32,
        supportStatus="supported",
        source="java_controlled_read",
        dataSnapshotId="transient-550e8400-e29b-41d4-a716-446655440000",
        queryHash=HASH,
        snapshotPersistence="transient",
    )
    report = ResearchExplorationReport(
        traceId="trace-001",
        runId="run-001",
        taskId="task-001",
        status="COMPLETED",
        evidenceBindings=[binding],
        limitations=["snapshot_is_transient_and_not_replayable"],
    )
    assert report.nonDiagnostic == "scientific_evidence_not_clinical_diagnosis"

    with pytest.raises(ValidationError):
        ResearchExplorationReport(
            traceId="trace-001",
            runId="run-001",
            taskId="task-001",
            status="COMPLETED",
            findings=[
                {
                    "findingId": "finding-" + "5" * 32,
                    "findingType": "descriptive",
                    "supportStatus": "supported",
                    "statement": "受控汇总已生成",
                    "evidenceBindingIds": ["binding-" + "6" * 32],
                }
            ],
            evidenceBindings=[binding],
            limitations=["snapshot_is_transient_and_not_replayable"],
        )


def test_report_accepts_only_evidence_bound_generated_claims_and_paths() -> None:
    candidate = merge_unified_evidence(
        vector_results=[],
        graph_results=[_item("graph", 0.8, _path())],
        java_observations=[],
        limit=5,
    )[0]
    path_id = candidate.reasoningPaths[0].pathId
    claim = GroundedClaim(
        claimId="claim-" + "1" * 32,
        statement="The retrieved graph path is retained as a source-bound assertion.",
        supportStatus="supported",
        evidenceIds=[candidate.candidateId],
        reasoningPathIds=[path_id],
    )
    step = ReasoningStep(
        stepIndex=1,
        description="The path is reported with its source-bound status.",
        supportStatus="supported",
        evidenceIds=[candidate.candidateId],
        reasoningPathIds=[path_id],
    )
    report = ResearchExplorationReport(
        traceId="trace-001",
        runId="run-001",
        taskId="task-001",
        status="COMPLETED",
        unifiedEvidence=[candidate],
        groundedClaims=[claim],
        groundedReasoningSteps=[step],
        generationMode="model_grounded",
        limitations=["scientific_evidence_is_not_clinical_diagnosis"],
    )
    assert report.groundedClaims[0].evidenceIds == [candidate.candidateId]

    with pytest.raises(ValidationError):
        ResearchExplorationReport(
            traceId="trace-001",
            runId="run-001",
            taskId="task-001",
            status="COMPLETED",
            unifiedEvidence=[candidate],
            groundedClaims=[claim.model_copy(update={
                "evidenceIds": ["evidence-" + "f" * 32],
            })],
            limitations=["scientific_evidence_is_not_clinical_diagnosis"],
        )

    conflicted = merge_unified_evidence(
        vector_results=[],
        graph_results=[_item("graph", 0.8, _path("conflicted"))],
        java_observations=[],
        limit=5,
    )[0]
    with pytest.raises(ValidationError):
        ResearchExplorationReport(
            traceId="trace-001",
            runId="run-001",
            taskId="task-001",
            status="COMPLETED",
            unifiedEvidence=[conflicted],
            groundedClaims=[GroundedClaim(
                claimId="claim-" + "2" * 32,
                statement="This must not upgrade a conflicted path.",
                supportStatus="supported",
                evidenceIds=[conflicted.candidateId],
                reasoningPathIds=[conflicted.reasoningPaths[0].pathId],
            )],
            limitations=["scientific_evidence_is_not_clinical_diagnosis"],
        )


def test_planner_context_is_redacted_and_closed() -> None:
    context = ResearchPlannerContext(
        questionSummary="比较病例和对照的微生物组成差异",
        intent="scientific_exploration",
        approvedActions=["execute_read_query", "retrieve_evidence", "finish"],
        remainingActionBudget=6,
    )
    assert "questionSummary" in context.model_dump()

    with pytest.raises(ValidationError):
        ResearchPlannerContext(
            questionSummary="sourceSampleId=secret",
            intent="scientific_exploration",
            approvedActions=["finish"],
            remainingActionBudget=1,
        )
