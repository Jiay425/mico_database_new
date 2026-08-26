from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from mico_agent_runtime.contracts.audit import AuditEvent
from mico_agent_runtime.contracts.intent import IntentRunResult, IntentTaskRequest
from mico_agent_runtime.contracts.research import ResearchExplorationReport
from mico_agent_runtime.contracts.unified_evidence import merge_unified_evidence
from mico_agent_runtime.runtime.persistence import (
    RuntimePersistenceConfigurationError,
    RuntimePersistenceCoordinator,
    build_optional_runtime_persistence,
)
from mico_agent_runtime.storage.crypto import RuntimeStateCipher
from mico_agent_runtime.storage.models import RuntimeStatus, ToolAuditRecord
from mico_agent_runtime.storage.mysql_store import AgentStepRow, ToolAuditRow
from mico_agent_runtime.transport.app import create_app
from tests.test_unified_evidence import _item, _java_observation


RUN = "run-" + "1" * 32
TASK = "task-" + "2" * 32
TRACE = "trace-" + "3" * 32
CALL = "call-" + "4" * 32
SNAPSHOT = "transient-550e8400-e29b-41d4-a716-446655440000"
NOW = datetime(2026, 8, 22, tzinfo=timezone.utc)


class FakeStore:
    persistent = True
    recoverable = False

    def __init__(self) -> None:
        self.runs = []
        self.steps = []
        self.audits = []
        self.statuses = []
        self.artifacts = []

    async def create_run(self, value):
        self.runs.append(value)
        return value

    async def load_run(self, run_id):
        return self.runs[0] if self.runs and self.runs[0].runId == run_id else None

    async def update_encrypted_state(self, run_id, encrypted_state_payload, *, updated_at):
        if not self.runs or self.runs[0].runId != run_id:
            raise RuntimeError("missing")
        self.runs[0].encryptedStatePayload = encrypted_state_payload
        self.runs[0].updatedAt = updated_at
        return self.runs[0]

    async def list_steps(self, run_id):
        return [value for value in self.steps if value.runId == run_id]

    async def append_step(self, value):
        self.steps.append(value)
        return value

    async def mark_run_status(self, run_id, status, **kwargs):
        self.statuses.append((run_id, status, kwargs))
        if self.runs and self.runs[0].runId == run_id:
            self.runs[0].status = status
            self.runs[0].failureCode = kwargs.get("failure_code")
            self.runs[0].currentSuccessfulStepId = kwargs.get("current_successful_step_id")
            if kwargs.get("updated_at") is not None:
                self.runs[0].updatedAt = kwargs["updated_at"]
        return self.runs[0]

    async def create_approval_ticket(self, value):
        return value

    async def append_tool_audit(self, value):
        self.audits.append(value)
        return value

    async def append_artifact(self, value):
        self.artifacts.append(value)
        return value


def request() -> IntentTaskRequest:
    return IntentTaskRequest.model_validate({
        "runId": RUN,
        "taskId": TASK,
        "requesterId": "principal-" + "5" * 32,
        "requestedScopes": ["mico:query:read"],
        "question": "请比较两组物种差异",
            "allowedWorkflows": ["dynamic_read_query"],
        "createdAt": datetime(2026, 8, 22, tzinfo=timezone.utc),
        "traceId": TRACE,
    })


def result() -> IntentRunResult:
    event = AuditEvent(
        traceId=TRACE,
        runId=RUN,
        node="execute_selected_workflow",
        toolName="execute_read_query",
        toolCallId=CALL,
        status="COMPLETED",
        dataSnapshotId=SNAPSHOT,
        snapshotPersistence="transient",
        errorCode="JAVA_TOOL_COMPLETED",
        occurredAt=datetime(2026, 8, 22, tzinfo=timezone.utc),
    )
    return IntentRunResult(
        traceId=TRACE,
        runId=RUN,
        taskId=TASK,
        status="COMPLETED",
        workflow="dynamic_read_query",
        report={
            "dataSnapshotId": SNAPSHOT,
            "snapshotPersistence": "transient",
            "queryHash": "sha256:" + "a" * 64,
            "schemaVersion": "v1",
            "rowCount": 8,
            "generatedAt": "2026-08-22T00:00:00Z",
            "sourceBatch": "merged_abundance",
        },
        auditEvents=[event],
    )


@pytest.mark.asyncio
async def test_persistence_encrypts_request_and_stores_only_safe_metadata() -> None:
    store = FakeStore()
    cipher = RuntimeStateCipher(b"k" * 32, "runtime-key-1")
    coordinator = RuntimePersistenceCoordinator(store, cipher)

    await coordinator.begin(request())
    envelope_text = cipher.serialize(store.runs[0].encryptedStatePayload)
    assert "sourceSampleId" not in envelope_text
    assert "patient_data_manager" not in envelope_text
    assert store.statuses[0][1] == RuntimeStatus.RUNNING

    await coordinator.finish(result())
    assert store.statuses[-1][1] == RuntimeStatus.COMPLETED
    assert len(store.steps) == 1
    assert len(store.audits) == 1
    persisted_step = store.steps[0].model_dump_json()
    persisted_audit = store.audits[0].model_dump_json()
    for secret in ("sourceSampleId", "internalRecordId", "SELECT", "payload", "Bearer"):
        assert secret.lower() not in (persisted_step + persisted_audit).lower()
    assert store.steps[0].snapshotMetadata is not None
    assert store.steps[0].snapshotMetadata.dataSnapshotId == SNAPSHOT


@pytest.mark.asyncio
async def test_persisted_progress_rebuilds_only_safe_step_and_snapshot_metadata() -> None:
    store = FakeStore()
    coordinator = RuntimePersistenceCoordinator(store, RuntimeStateCipher(b"k" * 32, "runtime-key-1"))

    await coordinator.begin(request())
    await coordinator.finish(result())

    progress = await coordinator.load_progress(RUN)
    assert progress is not None
    assert progress.runId == RUN
    assert progress.taskId == TASK
    assert progress.status == "COMPLETED"
    assert len(progress.events) == 1
    assert progress.events[0].node == "execute_tool"
    assert progress.events[0].dataSnapshotId == SNAPSHOT
    assert progress.events[0].snapshotPersistence == "transient"
    dumped = progress.model_dump_json()
    for forbidden in ("sourceSampleId", "internalRecordId", "SELECT", "payload", "Bearer"):
        assert forbidden.lower() not in dumped.lower()


@pytest.mark.asyncio
async def test_checkpoint_replaces_only_encrypted_state_and_never_plaintext_columns() -> None:
    store = FakeStore()
    cipher = RuntimeStateCipher(b"k" * 32, "runtime-key-1")
    coordinator = RuntimePersistenceCoordinator(store, cipher)
    await coordinator.begin(request())
    await coordinator.checkpoint({
        "runId": RUN,
        "taskId": TASK,
        "traceId": TRACE,
        "locator": {"internalRecordId": 18423, "sourceSampleId": "complete-source-sample"},
        "toolResult": {"data": "java-payload", "warning": "free warning"},
    })
    serialized = cipher.serialize(store.runs[0].encryptedStatePayload)
    assert "complete-source-sample" not in serialized
    assert "java-payload" not in serialized
    assert "free warning" not in serialized
    assert store.runs[0].status == RuntimeStatus.RUNNING
    assert store.runs[0].encryptedStatePayload.keyId == "runtime-key-1"


@pytest.mark.asyncio
async def test_non_opaque_ids_fail_before_store_write() -> None:
    store = FakeStore()
    coordinator = RuntimePersistenceCoordinator(store, RuntimeStateCipher(b"k" * 32, "runtime-key-1"))
    bad = request().model_copy(update={"runId": "run-sample-accession"})
    with pytest.raises(Exception) as error:
        await coordinator.begin(bad)
    assert str(error.value) == "RUNTIME_PERSISTENCE_ID_INVALID"
    assert store.runs == []


@pytest.mark.asyncio
async def test_unified_evidence_projection_reaches_trace_audit_and_artifact_metadata() -> None:
    store = FakeStore()
    coordinator = RuntimePersistenceCoordinator(
        store, RuntimeStateCipher(b"k" * 32, "runtime-key-1")
    )
    evidence = merge_unified_evidence(
        vector_results=[_item("vector", 0.8)],
        graph_results=[_item("graph", 0.7)],
        java_observations=[_java_observation()],
        limit=5,
    )
    report = ResearchExplorationReport(
        traceId=TRACE,
        runId=RUN,
        taskId=TASK,
        status="COMPLETED",
        unifiedEvidence=evidence,
        limitations=["snapshot_is_transient_and_not_replayable"],
    )
    events = [
        AuditEvent(
            traceId=TRACE,
            runId=RUN,
            node="execute_action",
            toolName="execute_read_query",
            toolCallId=CALL,
            status="COMPLETED",
            occurredAt=NOW,
        ),
        AuditEvent(
            traceId=TRACE,
            runId=RUN,
            node="synthesize_report",
            status="COMPLETED",
            occurredAt=NOW,
        ),
    ]
    value = IntentRunResult(
        traceId=TRACE,
        runId=RUN,
        taskId=TASK,
        status="COMPLETED",
        report=report,
        auditEvents=events,
    )

    await coordinator.begin(request())
    await coordinator.finish(value)

    assert len(store.steps) == 2
    assert all(step.evidenceProjection is not None for step in store.steps)
    projection = store.steps[-1].evidenceProjection
    assert projection is not None
    assert projection.candidateCount == 2
    assert projection.sourceRoutes == ["graph", "java", "vector"]
    assert projection.sourceBindingCount == 3
    assert projection.transientSnapshotCount == 1
    assert len(store.audits) == 1
    assert store.audits[0].evidenceProjection == projection
    assert len(store.artifacts) == 1
    assert store.artifacts[0].dataSnapshotId == SNAPSHOT

    for persisted in (*store.steps, *store.audits):
        dumped = persisted.model_dump_json()
        for forbidden in ("sourceSampleId", "internalRecordId", "payload", "cohortCondition"):
            assert forbidden.lower() not in dumped.lower()


def test_unified_projection_round_trips_through_existing_json_trace_columns() -> None:
    from mico_agent_runtime.storage.models import UnifiedEvidencePersistenceProjection

    projection = UnifiedEvidencePersistenceProjection(
        candidateCount=1,
        vectorCandidateCount=1,
        graphCandidateCount=0,
        javaCandidateCount=0,
        sourceRoutes=["vector"],
        sourceBindingCount=1,
        reasoningPathCount=0,
        conflictedPathCount=0,
        transientSnapshotCount=0,
        generatedAt=NOW,
    )
    from mico_agent_runtime.storage.models import AgentStepRecord

    step = AgentStepRecord(
        dataContractVersion="v1",
        stepId="step-" + "a" * 32,
        runId=RUN,
        nodeName="execute_tool",
        attemptNumber=1,
        status="COMPLETED",
        startedAt=NOW,
        evidenceProjection=projection,
    )
    row = AgentStepRow.from_model(step)
    restored = row.to_model()
    assert restored.evidenceProjection == projection
    assert row.snapshot_metadata is not None
    assert "evidenceProjection" in row.snapshot_metadata

    audit = ToolAuditRow.from_model(
        ToolAuditRecord(
            dataContractVersion="v1",
            auditId="audit-" + "b" * 32,
            runId=RUN,
            traceId=TRACE,
            toolName="execute_read_query",
            toolCallId=CALL,
            status="COMPLETED",
            durationMs=1,
            evidenceProjection=projection,
            createdAt=NOW,
        )
    )
    assert audit.to_model().evidenceProjection == projection


def test_persistence_is_disabled_without_explicit_switch() -> None:
    assert build_optional_runtime_persistence({}) is None


def test_enabled_persistence_without_key_fails_closed_without_connection() -> None:
    with pytest.raises(RuntimePersistenceConfigurationError) as error:
        build_optional_runtime_persistence({
            "MICO_AGENT_RUNTIME_MYSQL_ENABLED": "true",
            "MICO_AGENT_RUNTIME_DATABASE_URL": "mysql+asyncmy://runtime@127.0.0.1/mico_agent_runtime",
        })
    assert str(error.value) in {
        "RUNTIME_STATE_ENCRYPTION_KEY_INVALID",
        "RUNTIME_STATE_ENCRYPTION_KEY_ID_INVALID",
    }


def test_enabled_but_invalid_persistence_blocks_run_before_graph() -> None:
    class Runtime:
        def run(self, _request):
            raise AssertionError("graph must not execute")

    app = create_app(
        env={
            "MICO_RUNTIME_INTERNAL_TOKEN": "runtime-token",
            "MICO_AGENT_RUNTIME_MYSQL_ENABLED": "true",
            "MICO_AGENT_RUNTIME_DATABASE_URL": "mysql+asyncmy://runtime@127.0.0.1/mico_agent_runtime",
        },
        intent_runtime=Runtime(),
    )
    with TestClient(app) as client:
        response = client.post(
            "/internal/runtime/intent-runs",
            json=request().model_dump(mode="json"),
            headers={"Authorization": "Bearer runtime-token"},
        )
    assert response.status_code == 503
    assert response.json() == {
        "code": "RUNTIME_PERSISTENCE_NOT_CONFIGURED",
        "message": "Runtime persistence is not configured",
    }
