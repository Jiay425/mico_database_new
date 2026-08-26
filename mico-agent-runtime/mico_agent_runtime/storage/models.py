from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, field_validator, model_validator

from mico_agent_runtime.contracts.base import ClosedModel
from mico_agent_runtime.contracts.approval import ApprovalOperation
from mico_agent_runtime.contracts.tools import AllowedToolName, ToolStatus, TransientPersistence

from .crypto import EncryptedStateEnvelope
from .types import (
    ApprovalId,
    ArtifactId,
    ArtifactStorageRef,
    AuditId,
    HashValue,
    KeyIdentifier,
    DiseaseMappingVersion,
    FeatureVersion,
    ImportBatch,
    PrincipalId,
    RunId,
    SchemaVersion,
    JavaTransientSnapshotId,
    SourceBatch,
    StepId,
    TaxonomyVersion,
    TaskId,
    ToolCallId,
    TraceId,
)


StorageContractVersion = Literal["v1"]
RuntimeStepCode = Literal[
    "TASK_CONTRACT_VALIDATED",
    "POLICY_ALLOWED",
    "TOOL_PLAN_BUILT",
    "JAVA_TOOL_COMPLETED",
    "JAVA_TOOL_REJECTED",
    "JAVA_TOOL_FAILED",
    "EVIDENCE_METADATA_CREATED",
    "RUN_QUEUED",
    "RUN_COMPLETED",
    "RUN_FAILED",
    "APPROVAL_REQUESTED",
]
RuntimeControlCode = Annotated[
    str,
    StringConstraints(min_length=3, max_length=128, pattern=r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+$"),
]
ArtifactType = Literal["evidence_summary"]
EvidenceRoute = Literal["vector", "graph", "java"]
RuntimeNodeName = Literal[
    "validate_task",
    "policy_gate",
    "build_tool_plan",
    "execute_tool",
    "summarize_evidence",
    "human_review",
    "terminal",
]


class RuntimeStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    WAITING_JOB = "WAITING_JOB"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ApprovalStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"


_FORBIDDEN_FIELD_NAMES = {
    "patientid",
    "subjectid",
    "samplekey",
    "sourcesampleid",
    "internalrecordid",
    "cohortcondition",
    "rawarguments",
    "rawpayload",
    "requestheaders",
    "authorization",
    "token",
    "password",
    "privatekey",
}
def _walk_for_sensitive_content(value: object, field_name: str | None = None) -> None:
    if field_name and field_name.lower() in _FORBIDDEN_FIELD_NAMES:
        raise ValueError("sensitive runtime content is not persistable")
    if isinstance(value, dict):
        for key, child in value.items():
            _walk_for_sensitive_content(child, str(key))
    elif isinstance(value, (list, tuple)):
        for child in value:
            _walk_for_sensitive_content(child)


class PersistentSafeModel(ClosedModel):
    """Closed persistence model that rejects raw locator, payload, and credential material."""

    @field_validator("*")
    @classmethod
    def require_aware_datetime(cls, value: object) -> object:
        if isinstance(value, datetime) and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("persisted timestamps must include a timezone")
        return value

    @model_validator(mode="after")
    def reject_sensitive_content(self) -> "PersistentSafeModel":
        _walk_for_sensitive_content(self.model_dump(mode="python"))
        return self


class SnapshotMetadata(PersistentSafeModel):
    """Safe, normalized snapshot metadata; it has no cohort condition or payload field."""

    dataSnapshotId: JavaTransientSnapshotId
    source: Literal["java_agent_read_model", "runtime"]
    snapshotPersistence: TransientPersistence
    schemaVersion: SchemaVersion
    queryHash: HashValue
    rowCount: Annotated[int, Field(strict=True, ge=0)]
    generatedAt: datetime
    importBatch: ImportBatch | None = None
    diseaseMappingVersion: DiseaseMappingVersion | None = None
    taxonomyVersion: TaxonomyVersion | None = None
    featureVersion: FeatureVersion | None = None
    sourceBatch: SourceBatch | None = None


class UnifiedEvidencePersistenceProjection(PersistentSafeModel):
    """Safe counters for a unified evidence trace.

    This is deliberately a projection rather than a copy of evidence.  It
    records route/binding/path counts and conflict state only; source text,
    locator values, Java payloads and literature content remain transient.
    It is serialized inside the existing JSON metadata columns, so no schema
    migration is required for this additive trace projection.
    """

    candidateCount: Annotated[int, Field(strict=True, ge=0, le=50)]
    vectorCandidateCount: Annotated[int, Field(strict=True, ge=0, le=50)]
    graphCandidateCount: Annotated[int, Field(strict=True, ge=0, le=50)]
    javaCandidateCount: Annotated[int, Field(strict=True, ge=0, le=50)]
    sourceRoutes: list[EvidenceRoute] = Field(default_factory=list, max_length=3)
    sourceBindingCount: Annotated[int, Field(strict=True, ge=0, le=150)]
    reasoningPathCount: Annotated[int, Field(strict=True, ge=0, le=200)]
    conflictedPathCount: Annotated[int, Field(strict=True, ge=0, le=200)]
    transientSnapshotCount: Annotated[int, Field(strict=True, ge=0, le=50)]
    generatedAt: datetime

    @model_validator(mode="after")
    def validate_route_projection(self) -> "UnifiedEvidencePersistenceProjection":
        if len(self.sourceRoutes) != len(set(self.sourceRoutes)):
            raise ValueError("source routes must be unique")
        if self.vectorCandidateCount and "vector" not in self.sourceRoutes:
            raise ValueError("vector candidates require a vector route")
        if self.graphCandidateCount and "graph" not in self.sourceRoutes:
            raise ValueError("graph candidates require a graph route")
        if self.javaCandidateCount and "java" not in self.sourceRoutes:
            raise ValueError("Java candidates require a Java route")
        return self


class AgentRunRecord(PersistentSafeModel):
    dataContractVersion: StorageContractVersion
    runId: RunId
    taskId: TaskId
    traceId: TraceId
    status: RuntimeStatus
    createdAt: datetime
    updatedAt: datetime
    currentSuccessfulStepId: StepId | None = None
    failureCode: RuntimeControlCode | None = None
    encryptedStatePayload: EncryptedStateEnvelope
    encryptionKeyId: KeyIdentifier

    @model_validator(mode="after")
    def require_envelope_key_id(self) -> "AgentRunRecord":
        if self.encryptedStatePayload.keyId != self.encryptionKeyId:
            raise ValueError("encrypted state key ID does not match the run key ID")
        return self


class AgentStepRecord(PersistentSafeModel):
    dataContractVersion: StorageContractVersion
    stepId: StepId
    runId: RunId
    nodeName: RuntimeNodeName
    attemptNumber: Annotated[int, Field(strict=True, ge=1, le=100)]
    status: RuntimeStatus
    startedAt: datetime
    endedAt: datetime | None = None
    errorCode: RuntimeControlCode | None = None
    safeInputCode: RuntimeStepCode | None = None
    safeOutputCode: RuntimeStepCode | None = None
    snapshotMetadata: SnapshotMetadata | None = None
    evidenceProjection: UnifiedEvidencePersistenceProjection | None = None


class AgentArtifactRecord(PersistentSafeModel):
    dataContractVersion: StorageContractVersion
    artifactId: ArtifactId
    runId: RunId
    artifactType: ArtifactType
    schemaVersion: SchemaVersion
    contentHash: HashValue
    artifactStorageRef: ArtifactStorageRef
    dataSnapshotId: JavaTransientSnapshotId | None = None
    createdAt: datetime

    @model_validator(mode="after")
    def require_bound_artifact_reference(self) -> "AgentArtifactRecord":
        expected = f"artifact://{self.artifactId}"
        if self.artifactStorageRef != expected:
            raise ValueError("artifact storage reference must be bound to artifact ID")
        return self


class ApprovalTicketRecord(PersistentSafeModel):
    dataContractVersion: StorageContractVersion
    approvalId: ApprovalId
    runId: RunId
    operation: ApprovalOperation
    status: ApprovalStatus
    requestedBy: PrincipalId
    requestedAt: datetime
    decidedAt: datetime | None = None
    decidedBy: PrincipalId | None = None
    decisionCode: RuntimeControlCode | None = None


class ToolAuditRecord(PersistentSafeModel):
    dataContractVersion: StorageContractVersion
    auditId: AuditId
    runId: RunId
    traceId: TraceId
    toolName: AllowedToolName
    toolCallId: ToolCallId
    status: ToolStatus
    durationMs: Annotated[int, Field(strict=True, ge=0, le=86_400_000)]
    errorCode: RuntimeControlCode | None = None
    snapshotMetadata: SnapshotMetadata | None = None
    evidenceProjection: UnifiedEvidencePersistenceProjection | None = None
    createdAt: datetime
