from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from .base import ClosedModel, Identifier, NonEmptyText
from .research import ScientificActionName
from .tools import JavaDataSnapshot, ToolCallIdentifier, ToolStatus, TransientPersistence, TransientSnapshotId

EvidenceScope = Literal["study", "sample_candidates", "exact_record_profile_locator"]


class AuditEvent(ClosedModel):
    """Minimal audit record; it intentionally has no request or payload field."""

    traceId: Identifier
    runId: Identifier
    node: NonEmptyText
    actionName: ScientificActionName | None = None
    toolName: NonEmptyText | None = None
    toolCallId: ToolCallIdentifier | None = None
    status: ToolStatus
    errorCode: NonEmptyText | None = None
    dataSnapshotId: TransientSnapshotId | None = None
    snapshotPersistence: TransientPersistence | None = None
    occurredAt: datetime


class EvidenceSummary(ClosedModel):
    status: ToolStatus
    source: NonEmptyText | None = None
    rowCount: int | None = None
    schemaVersion: NonEmptyText | None = None
    dataSnapshotId: TransientSnapshotId | None = None
    dataSource: NonEmptyText | None = None
    importBatch: NonEmptyText | None = None
    diseaseMappingVersion: NonEmptyText | None = None
    taxonomyVersion: NonEmptyText | None = None
    featureVersion: NonEmptyText | None = None
    sourceBatch: NonEmptyText | None = None
    evidenceScope: EvidenceScope
    queryHash: NonEmptyText | None = None
    generatedAt: datetime | None = None
    snapshotPersistence: TransientPersistence | None = None
    replayable: bool = False
    warnings: list[NonEmptyText] = Field(default_factory=list)

    @classmethod
    def from_snapshot(cls, response_source: str | None, schema_version: str | None,
                      snapshot: JavaDataSnapshot, warnings: list[str], *,
                      evidence_scope: EvidenceScope) -> "EvidenceSummary":
        return cls(
            status="COMPLETED",
            source=response_source,
            rowCount=snapshot.rowCount,
            schemaVersion=schema_version,
            dataSnapshotId=snapshot.dataSnapshotId,
            dataSource=snapshot.dataSource,
            importBatch=snapshot.importBatch,
            diseaseMappingVersion=snapshot.diseaseMappingVersion,
            taxonomyVersion=snapshot.taxonomyVersion,
            featureVersion=snapshot.featureVersion,
            sourceBatch=snapshot.sourceBatch,
            evidenceScope=evidence_scope,
            queryHash=snapshot.queryHash,
            generatedAt=snapshot.generatedAt,
            snapshotPersistence=snapshot.snapshotPersistence,
            replayable=False,
            warnings=warnings,
        )


class RuntimeRunResult(ClosedModel):
    """Safe runtime result; raw Java data stays inside the in-memory graph state."""

    traceId: Identifier
    runId: Identifier
    taskId: Identifier
    status: ToolStatus
    errorCode: NonEmptyText | None = None
    evidenceSummary: EvidenceSummary | None = None
    auditEvents: list[AuditEvent] = Field(default_factory=list)
