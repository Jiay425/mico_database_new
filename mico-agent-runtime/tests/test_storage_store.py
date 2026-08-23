from __future__ import annotations

from datetime import datetime, timezone

import pytest

from mico_agent_runtime.storage.crypto import EncryptedStateEnvelope
from mico_agent_runtime.storage.memory_store import InMemoryRuntimeStore
from mico_agent_runtime.storage.models import (
    AgentArtifactRecord,
    AgentRunRecord,
    AgentStepRecord,
    ApprovalTicketRecord,
    RuntimeStatus,
    ToolAuditRecord,
)
from mico_agent_runtime.storage.ports import (
    InvalidStateTransitionError,
    RunNotFoundError,
)
from tests.storage_fixtures import (
    APPROVAL_ID,
    ARTIFACT_ID,
    AUDIT_ID,
    CALL_ID,
    RUN_ID,
    SNAPSHOT_ID,
    STEP_ID,
    TASK_ID,
    TRACE_ID,
    PRINCIPAL_ID,
)


NOW = datetime(2026, 8, 21, tzinfo=timezone.utc)


def make_run() -> AgentRunRecord:
    return AgentRunRecord(
        dataContractVersion="v1",
        runId=RUN_ID,
        taskId=TASK_ID,
        traceId=TRACE_ID,
        status="QUEUED",
        createdAt=NOW,
        updatedAt=NOW,
        encryptedStatePayload=EncryptedStateEnvelope(
            algorithm="AES-256-GCM",
            keyId="key-runtime-v1",
            nonce="nonce-value",
            ciphertext="ciphertext-value",
        ),
        encryptionKeyId="key-runtime-v1",
    )


@pytest.mark.asyncio
async def test_in_memory_store_is_explicitly_non_persistent_and_controls_transitions() -> None:
    store = InMemoryRuntimeStore()
    assert store.persistent is False
    assert store.recoverable is False
    await store.create_run(make_run())
    await store.mark_run_status(RUN_ID, RuntimeStatus.RUNNING)
    await store.mark_run_status(RUN_ID, RuntimeStatus.WAITING_JOB)
    await store.mark_run_status(RUN_ID, RuntimeStatus.RUNNING)
    completed = await store.mark_run_status(RUN_ID, RuntimeStatus.COMPLETED)
    assert completed.status is RuntimeStatus.COMPLETED
    with pytest.raises(InvalidStateTransitionError):
        await store.mark_run_status(RUN_ID, RuntimeStatus.RUNNING)


@pytest.mark.asyncio
async def test_in_memory_store_requires_run_for_related_records() -> None:
    store = InMemoryRuntimeStore()
    step = AgentStepRecord(
        dataContractVersion="v1",
        stepId=STEP_ID,
        runId="run-" + "f" * 32,
        nodeName="validate_task",
        attemptNumber=1,
        status="RUNNING",
        startedAt=NOW,
    )
    with pytest.raises(RunNotFoundError):
        await store.append_step(step)


@pytest.mark.asyncio
async def test_in_memory_store_supports_typed_related_records_without_raw_payloads() -> None:
    store = InMemoryRuntimeStore()
    await store.create_run(make_run())
    step = AgentStepRecord(
        dataContractVersion="v1",
        stepId=STEP_ID,
        runId=RUN_ID,
        nodeName="validate_task",
        attemptNumber=1,
        status="COMPLETED",
        startedAt=NOW,
        endedAt=NOW,
        safeOutputCode="JAVA_TOOL_COMPLETED",
    )
    approval = ApprovalTicketRecord(
        dataContractVersion="v1",
        approvalId=APPROVAL_ID,
        runId=RUN_ID,
        operation="sensitive_batch_read",
        status="PENDING",
        requestedBy=PRINCIPAL_ID,
        requestedAt=NOW,
    )
    audit = ToolAuditRecord(
        dataContractVersion="v1",
        auditId=AUDIT_ID,
        runId=RUN_ID,
        traceId=TRACE_ID,
        toolName="execute_read_query",
        toolCallId=CALL_ID,
        status="COMPLETED",
        durationMs=12,
        createdAt=NOW,
    )
    artifact = AgentArtifactRecord(
        dataContractVersion="v1",
        artifactId=ARTIFACT_ID,
        runId=RUN_ID,
        artifactType="evidence_summary",
        schemaVersion="p2b1-v1",
        contentHash="sha256:" + "c" * 64,
        artifactStorageRef=f"artifact://{ARTIFACT_ID}",
        dataSnapshotId=SNAPSHOT_ID,
        createdAt=NOW,
    )
    assert (await store.append_step(step)).stepId == step.stepId
    assert (await store.create_approval_ticket(approval)).approvalId == approval.approvalId
    assert (await store.append_tool_audit(audit)).auditId == audit.auditId
    assert (await store.append_artifact(artifact)).artifactId == artifact.artifactId
