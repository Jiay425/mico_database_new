from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from mico_agent_runtime.storage.crypto import EncryptedStateEnvelope
from mico_agent_runtime.storage.models import (
    AgentArtifactRecord,
    AgentRunRecord,
    AgentStepRecord,
    ApprovalTicketRecord,
    SnapshotMetadata,
    ToolAuditRecord,
    UnifiedEvidencePersistenceProjection,
)
from tests.storage_fixtures import (
    ARTIFACT_ID,
    AUDIT_ID,
    CALL_ID,
    RUN_ID,
    SNAPSHOT_ID,
    STEP_ID,
    TASK_ID,
    TRACE_ID,
)


NOW = datetime(2026, 8, 21, tzinfo=timezone.utc)


def snapshot() -> SnapshotMetadata:
    return SnapshotMetadata(
        dataSnapshotId=SNAPSHOT_ID,
        source="java_agent_read_model",
        snapshotPersistence="transient",
        schemaVersion="p1b2-java-read-only-tool-executor-v1",
        queryHash="sha256:" + "a" * 64,
        rowCount=1,
        generatedAt=NOW,
        taxonomyVersion="species-v1",
        featureVersion="feature-v1",
        sourceBatch="meta2db-species-20260726:2015_Castro-NallarE",
    )


def run() -> AgentRunRecord:
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


def test_persisted_models_are_closed_and_snapshot_is_normalized() -> None:
    value = snapshot()
    assert "cohortCondition" not in value.model_dump()
    assert value.sourceBatch == "meta2db-species-20260726:2015_Castro-NallarE"
    with pytest.raises(ValidationError):
        SnapshotMetadata.model_validate({**value.model_dump(), "cohortCondition": "forbidden"})
    with pytest.raises(ValidationError):
        ToolAuditRecord.model_validate({
            "dataContractVersion": "v1",
            "auditId": AUDIT_ID,
            "runId": RUN_ID,
            "traceId": TRACE_ID,
            "toolName": "execute_read_query",
            "toolCallId": CALL_ID,
            "status": "COMPLETED",
            "durationMs": 10,
            "createdAt": NOW,
            "rawArguments": {"sourceSampleId": "forbidden"},
        })


@pytest.mark.parametrize("unsafe", [
    "SRR1518476",
    "2015_Castro-NallarE",
    "sourceSampleId=complete-source-sample",
    "internalRecordId=18423",
    "Bearer java-token",
    "Authorization: Bearer java-token",
    "s3://bucket/SRR1518476/report",
    r"C:\reports\SRR1518476.json",
    "postgresql://runtime@host/db",
    "fake payload fragment",
    "warning: arbitrary upstream text",
])
def test_step_codes_reject_locator_payload_token_database_text_and_free_warning(unsafe: str) -> None:
    with pytest.raises(ValueError):
        AgentStepRecord(
            dataContractVersion="v1",
            stepId=STEP_ID,
            runId=RUN_ID,
            nodeName="execute_tool",
            attemptNumber=1,
            status="FAILED",
            startedAt=NOW,
            safeOutputCode=unsafe,
        )


def test_step_codes_accept_only_runtime_owned_finite_codes() -> None:
    value = AgentStepRecord(
        dataContractVersion="v1",
        stepId=STEP_ID,
        runId=RUN_ID,
        nodeName="execute_tool",
        attemptNumber=1,
        status="COMPLETED",
        startedAt=NOW,
        safeInputCode="TASK_CONTRACT_VALIDATED",
        safeOutputCode="JAVA_TOOL_COMPLETED",
    )
    assert value.safeInputCode == "TASK_CONTRACT_VALIDATED"
    with pytest.raises(ValidationError):
        AgentStepRecord.model_validate({**value.model_dump(), "safeOutputCode": "UPSTREAM_WARNING"})


def test_unified_evidence_projection_is_closed_and_metadata_only() -> None:
    projection = UnifiedEvidencePersistenceProjection(
        candidateCount=3,
        vectorCandidateCount=1,
        graphCandidateCount=1,
        javaCandidateCount=1,
        sourceRoutes=["vector", "graph", "java"],
        sourceBindingCount=3,
        reasoningPathCount=2,
        conflictedPathCount=1,
        transientSnapshotCount=1,
        generatedAt=NOW,
    )
    step = AgentStepRecord(
        dataContractVersion="v1",
        stepId=STEP_ID,
        runId=RUN_ID,
        nodeName="execute_tool",
        attemptNumber=1,
        status="COMPLETED",
        startedAt=NOW,
        evidenceProjection=projection,
    )
    assert step.evidenceProjection is not None
    assert step.evidenceProjection.sourceBindingCount == 3
    dumped = step.model_dump_json()
    for forbidden in (
        "sourceSampleId",
        "internalRecordId",
        "cohortCondition",
        "java payload",
        "free warning",
    ):
        assert forbidden.lower() not in dumped.lower()
    with pytest.raises(ValidationError):
        UnifiedEvidencePersistenceProjection.model_validate({
            **projection.model_dump(),
            "sourceRoutes": ["vector", "vector"],
        })


@pytest.mark.parametrize("bad_code", ["SRR9999999", "MV_FEI1_t1Q14", "2015_Castro-NallarE", "warning_free_text"])
def test_control_codes_cannot_be_used_as_business_or_free_text_values(bad_code: str) -> None:
    with pytest.raises(ValidationError):
        AgentRunRecord.model_validate({**run().model_dump(), "failureCode": bad_code})
    with pytest.raises(ValidationError):
        AgentStepRecord(
            dataContractVersion="v1",
            stepId=STEP_ID,
            runId=RUN_ID,
            nodeName="execute_tool",
            attemptNumber=1,
            status="FAILED",
            startedAt=NOW,
            errorCode=bad_code,
        )
    with pytest.raises(ValidationError):
        ApprovalTicketRecord(
            dataContractVersion="v1",
            approvalId="approval-" + "b" * 32,
            runId=RUN_ID,
            operation="business_write",
            status="REJECTED",
            requestedBy="principal-" + "c" * 32,
            requestedAt=NOW,
            decisionCode=bad_code,
        )
    with pytest.raises(ValidationError):
        ToolAuditRecord(
            dataContractVersion="v1",
            auditId="audit-" + "d" * 32,
            runId=RUN_ID,
            traceId=TRACE_ID,
            toolName="execute_read_query",
            toolCallId=CALL_ID,
            status="FAILED",
            durationMs=1,
            errorCode=bad_code,
            createdAt=NOW,
        )


@pytest.mark.parametrize("bad_id", ["SRR9999999", "MV_FEI1_t1Q14", "2015_Castro-NallarE", RUN_ID])
def test_artifact_id_is_opaque_and_not_replaced_by_other_identifier_types(bad_id: str) -> None:
    with pytest.raises(ValidationError):
        AgentArtifactRecord(
            dataContractVersion="v1",
            artifactId=bad_id,
            runId=RUN_ID,
            artifactType="evidence_summary",
            schemaVersion="p2b1-v1",
            contentHash="sha256:" + "e" * 64,
            artifactStorageRef=f"artifact://{bad_id}",
            createdAt=NOW,
        )
    with pytest.raises(ValidationError):
        AgentArtifactRecord(
            dataContractVersion="v1",
            artifactId=ARTIFACT_ID,
            runId=RUN_ID,
            artifactType=bad_id,
            schemaVersion="p2b1-v1",
            contentHash="sha256:" + "e" * 64,
            artifactStorageRef=f"artifact://{ARTIFACT_ID}",
            createdAt=NOW,
        )


def test_runtime_id_prefixes_are_not_interchangeable() -> None:
    with pytest.raises(ValidationError):
        AgentRunRecord.model_validate({**run().model_dump(), "runId": ARTIFACT_ID})
    with pytest.raises(ValidationError):
        AgentRunRecord.model_validate({**run().model_dump(), "taskId": RUN_ID})
    with pytest.raises(ValidationError):
        AgentRunRecord.model_validate({**run().model_dump(), "traceId": RUN_ID})


def test_artifact_reference_is_opaque_and_bound_to_artifact_id() -> None:
    base = {
        "dataContractVersion": "v1",
        "artifactId": ARTIFACT_ID,
        "runId": RUN_ID,
        "artifactType": "evidence_summary",
        "schemaVersion": "p2b1-v1",
        "contentHash": "sha256:" + "b" * 64,
        "artifactStorageRef": f"artifact://{ARTIFACT_ID}",
        "createdAt": NOW,
    }
    assert AgentArtifactRecord(**base).artifactStorageRef == f"artifact://{ARTIFACT_ID}"
    for invalid in (
        "s3://bucket/SRR1518476/report",
        r"C:\reports\SRR1518476.json",
        "artifact://artifact-" + "f" * 32,
    ):
        with pytest.raises(ValueError):
            AgentArtifactRecord(**{**base, "artifactStorageRef": invalid})


@pytest.mark.parametrize("field", ["dataSnapshotId", "schemaVersion", "sourceBatch"])
@pytest.mark.parametrize("unsafe", ["SRR1518476", "s3://bucket/report", r"C:\reports\x.json"])
def test_structured_persistence_metadata_rejects_business_values_and_paths(field: str, unsafe: str) -> None:
    with pytest.raises(ValueError):
        SnapshotMetadata.model_validate({**snapshot().model_dump(), field: unsafe})


@pytest.mark.parametrize("invalid_snapshot_id", [
    "snapshot-" + "a" * 32,
    "SRR9999999",
    "MV_FEI1_t1Q14",
    "artifact://artifact-" + "b" * 32,
    "sourceSampleId=complete-source-sample",
    "https://example.invalid/evidence",
])
def test_java_transient_snapshot_id_is_exact_and_not_a_runtime_or_sample_id(invalid_snapshot_id: str) -> None:
    with pytest.raises(ValidationError):
        SnapshotMetadata.model_validate({**snapshot().model_dump(), "dataSnapshotId": invalid_snapshot_id})
    with pytest.raises(ValidationError):
        AgentArtifactRecord(
            dataContractVersion="v1",
            artifactId=ARTIFACT_ID,
            runId=RUN_ID,
            artifactType="evidence_summary",
            schemaVersion="p2b1-v1",
            contentHash="sha256:" + "f" * 64,
            artifactStorageRef=f"artifact://{ARTIFACT_ID}",
            dataSnapshotId=invalid_snapshot_id,
            createdAt=NOW,
        )


@pytest.mark.parametrize("unsafe", [
    "SELECT * FROM runtime_state",
    "Bearer opaque-token",
    "sourceSampleId=complete-source-sample",
    "internalRecordId=18423",
    "warning: free upstream warning",
    "{\"payload\":\"raw\"}",
])
def test_metadata_version_type_rejects_generic_dangerous_syntax_without_business_blacklist(unsafe: str) -> None:
    with pytest.raises(ValueError):
        SnapshotMetadata.model_validate({**snapshot().model_dump(), "sourceBatch": unsafe})


def test_legacy_free_text_field_names_are_closed() -> None:
    with pytest.raises(ValidationError):
        AgentStepRecord.model_validate({
            "dataContractVersion": "v1",
            "stepId": STEP_ID,
            "runId": RUN_ID,
            "nodeName": "execute_tool",
            "attemptNumber": 1,
            "status": "COMPLETED",
            "startedAt": NOW,
            "safeInputSummary": "SRR1518476",
        })
    with pytest.raises(ValidationError):
        AgentArtifactRecord.model_validate({
            "dataContractVersion": "v1",
            "artifactId": ARTIFACT_ID,
            "runId": RUN_ID,
            "artifactType": "evidence_summary",
            "schemaVersion": "p2b1-v1",
            "contentHash": "sha256:" + "d" * 64,
            "externalLocationRef": "s3://bucket/SRR1518476/report",
            "artifactStorageRef": f"artifact://{ARTIFACT_ID}",
            "createdAt": NOW,
        })


@pytest.mark.parametrize("invalid_hash", [
    "sha256:abcdef12",
    "sha256:" + "A" * 64,
    "hash-test",
    "sha512:" + "c" * 64,
])
def test_hash_fields_require_lowercase_sha256(invalid_hash: str) -> None:
    with pytest.raises(ValidationError):
        SnapshotMetadata.model_validate({**snapshot().model_dump(), "queryHash": invalid_hash})
    with pytest.raises(ValidationError):
        AgentArtifactRecord(
            dataContractVersion="v1",
            artifactId=ARTIFACT_ID,
            runId=RUN_ID,
            artifactType="evidence_summary",
            schemaVersion="p2b1-v1",
            contentHash=invalid_hash,
            artifactStorageRef=f"artifact://{ARTIFACT_ID}",
            createdAt=NOW,
        )


def test_run_model_requires_encrypted_state_fields() -> None:
    value = run()
    assert value.encryptedStatePayload.algorithm == "AES-256-GCM"
    with pytest.raises(ValidationError):
        AgentRunRecord.model_validate({**value.model_dump(), "encryptedStatePayload": "plain-state"})
