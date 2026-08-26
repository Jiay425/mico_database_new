from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, Integer, String, Text, select
from sqlalchemy.dialects.mysql import JSON
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from .configuration import RuntimeStorageConfiguration
from .crypto import EncryptedStateEnvelope, RuntimeStateCipher
from .models import (
    AgentArtifactRecord,
    AgentRunRecord,
    AgentStepRecord,
    ApprovalTicketRecord,
    ApprovalStatus,
    RuntimeStatus,
    ToolAuditRecord,
    SnapshotMetadata,
    UnifiedEvidencePersistenceProjection,
)
from .ports import (
    DuplicateRecordError,
    RunNotFoundError,
    RuntimeStore,
    RuntimeStoreError,
    assert_status_transition,
    assert_approval_transition,
)
from .types import RunId, StepId


class RuntimeOrmBase(DeclarativeBase):
    pass


_STATUS_VALUES = "'QUEUED','RUNNING','WAITING_APPROVAL','WAITING_JOB','COMPLETED','FAILED','CANCELLED'"
_TOOL_STATUS_VALUES = "'COMPLETED','REJECTED','FAILED','NOT_IMPLEMENTED'"
_APPROVAL_STATUS_VALUES = "'PENDING','APPROVED','REJECTED','EXPIRED','CANCELLED'"
_APPROVAL_OPERATION_VALUES = "'read_only_research','statistical_analysis','sensitive_batch_read','bulk_export','external_publish','business_write'"
_RUNTIME_STEP_CODES = "'TASK_CONTRACT_VALIDATED','POLICY_ALLOWED','TOOL_PLAN_BUILT','JAVA_TOOL_COMPLETED','JAVA_TOOL_REJECTED','JAVA_TOOL_FAILED','EVIDENCE_METADATA_CREATED','RUN_QUEUED','RUN_COMPLETED','RUN_FAILED','APPROVAL_REQUESTED'"
_CONTROL_CODE_PATTERN = r"'^[A-Z][A-Z0-9]*(_[A-Z0-9]+)+$'"
_KEY_IDENTIFIER_PATTERN = r"'^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'"
_RUN_ID_PATTERN = r"'^run-[0-9a-f]{32}$'"
_TASK_ID_PATTERN = r"'^task-[0-9a-f]{32}$'"
_TRACE_ID_PATTERN = r"'^trace-[0-9a-f]{32}$'"
_STEP_ID_PATTERN = r"'^step-[0-9a-f]{32}$'"
_APPROVAL_ID_PATTERN = r"'^approval-[0-9a-f]{32}$'"
_AUDIT_ID_PATTERN = r"'^audit-[0-9a-f]{32}$'"
_ARTIFACT_ID_PATTERN = r"'^artifact-[0-9a-f]{32}$'"
_CALL_ID_PATTERN = r"'^call-[0-9a-f]{32}$'"
_PRINCIPAL_ID_PATTERN = r"'^principal-[0-9a-f]{32}$'"
_SNAPSHOT_ID_PATTERN = r"'^transient-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'"
_TOOL_NAMES = "'execute_read_query','literature_evidence'"
_ARTIFACT_TYPES = "'evidence_summary'"
_NODE_NAMES = "'validate_task','policy_gate','build_tool_plan','execute_tool','summarize_evidence','terminal'"
_MYSQL_TABLE_OPTIONS = {
    "mysql_engine": "InnoDB",
    "mysql_charset": "utf8mb4",
    "mysql_collate": "utf8mb4_bin",
}


def _utc_naive(value: datetime) -> datetime:
    """Convert an aware timestamp to MySQL DATETIME's UTC-without-zone form."""

    if value.tzinfo is None or value.utcoffset() is None:
        raise RuntimeStoreError("RUNTIME_TIMESTAMP_INVALID")
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _utc_aware(value: datetime) -> datetime:
    """Interpret a MySQL DATETIME value as UTC without exposing DB details."""

    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _trace_metadata_to_json(
    snapshot: SnapshotMetadata | None,
    projection: UnifiedEvidencePersistenceProjection | None,
) -> dict[str, Any] | None:
    """Encode only typed trace metadata into the existing JSON column."""

    if snapshot is None and projection is None:
        return None
    value: dict[str, Any] = (
        snapshot.model_dump(mode="json") if snapshot is not None else {}
    )
    if projection is not None:
        value["evidenceProjection"] = projection.model_dump(mode="json")
    return value


def _trace_metadata_from_json(
    value: dict[str, Any] | None,
) -> tuple[SnapshotMetadata | None, UnifiedEvidencePersistenceProjection | None]:
    """Decode legacy snapshot JSON plus the additive safe projection."""

    if value is None:
        return None, None
    if not isinstance(value, dict):
        raise RuntimeStoreError("RUNTIME_TRACE_METADATA_INVALID")
    snapshot_value = dict(value)
    projection_value = snapshot_value.pop("evidenceProjection", None)
    snapshot = SnapshotMetadata.model_validate(snapshot_value) if snapshot_value else None
    projection = (
        UnifiedEvidencePersistenceProjection.model_validate(projection_value)
        if projection_value is not None
        else None
    )
    return snapshot, projection


class AgentRunRow(RuntimeOrmBase):
    __tablename__ = "agent_run"

    run_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    task_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    trace_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    data_contract_version: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False, index=True)
    current_successful_step_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    failure_code: Mapped[str | None] = mapped_column(String(128), nullable=True)
    encrypted_state_payload: Mapped[str] = mapped_column(Text, nullable=False)
    encryption_key_id: Mapped[str] = mapped_column(String(128), nullable=False)

    __table_args__ = (
        CheckConstraint(f"status IN ({_STATUS_VALUES})", name="ck_agent_run_status"),
        CheckConstraint(f"run_id REGEXP {_RUN_ID_PATTERN}", name="ck_agent_run_id"),
        CheckConstraint(f"task_id REGEXP {_TASK_ID_PATTERN}", name="ck_agent_run_task_id"),
        CheckConstraint(f"trace_id REGEXP {_TRACE_ID_PATTERN}", name="ck_agent_run_trace_id"),
        CheckConstraint(f"current_successful_step_id IS NULL OR current_successful_step_id REGEXP {_STEP_ID_PATTERN}", name="ck_agent_run_success_step_id"),
        CheckConstraint(f"encryption_key_id REGEXP {_KEY_IDENTIFIER_PATTERN}", name="ck_agent_run_encryption_key_id"),
        CheckConstraint("data_contract_version = 'v1'", name="ck_agent_run_contract_version"),
        CheckConstraint(f"failure_code IS NULL OR failure_code REGEXP {_CONTROL_CODE_PATTERN}", name="ck_agent_run_failure_code"),
        Index("ix_agent_run_status_updated_at", "status", "updated_at"),
        _MYSQL_TABLE_OPTIONS,
    )

    @classmethod
    def from_model(cls, value: AgentRunRecord) -> "AgentRunRow":
        return cls(
            run_id=value.runId,
            task_id=value.taskId,
            trace_id=value.traceId,
            status=value.status.value,
            data_contract_version=value.dataContractVersion,
            created_at=_utc_naive(value.createdAt),
            updated_at=_utc_naive(value.updatedAt),
            current_successful_step_id=value.currentSuccessfulStepId,
            failure_code=value.failureCode,
            encrypted_state_payload=RuntimeStateCipher.serialize(value.encryptedStatePayload),
            encryption_key_id=value.encryptionKeyId,
        )

    def to_model(self) -> AgentRunRecord:
        return AgentRunRecord(
            dataContractVersion=self.data_contract_version,
            runId=self.run_id,
            taskId=self.task_id,
            traceId=self.trace_id,
            status=RuntimeStatus(self.status),
            createdAt=_utc_aware(self.created_at),
            updatedAt=_utc_aware(self.updated_at),
            currentSuccessfulStepId=self.current_successful_step_id,
            failureCode=self.failure_code,
            encryptedStatePayload=RuntimeStateCipher.deserialize(self.encrypted_state_payload),
            encryptionKeyId=self.encryption_key_id,
        )


class AgentStepRow(RuntimeOrmBase):
    __tablename__ = "agent_step"

    step_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), ForeignKey("agent_run.run_id"), nullable=False, index=True)
    data_contract_version: Mapped[str] = mapped_column(String(16), nullable=False)
    node_name: Mapped[str] = mapped_column(String(128), nullable=False)
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(128), nullable=True)
    safe_input_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    safe_output_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    snapshot_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    __table_args__ = (
        CheckConstraint(f"status IN ({_STATUS_VALUES})", name="ck_agent_step_status"),
        CheckConstraint(f"step_id REGEXP {_STEP_ID_PATTERN}", name="ck_agent_step_id"),
        CheckConstraint(f"run_id REGEXP {_RUN_ID_PATTERN}", name="ck_agent_step_run_id"),
        CheckConstraint("attempt_number >= 1 AND attempt_number <= 100", name="ck_agent_step_attempt"),
        CheckConstraint(f"node_name IN ({_NODE_NAMES})", name="ck_agent_step_node_name"),
        CheckConstraint("data_contract_version = 'v1'", name="ck_agent_step_contract_version"),
        CheckConstraint(f"error_code IS NULL OR error_code REGEXP {_CONTROL_CODE_PATTERN}", name="ck_agent_step_error_code"),
        CheckConstraint(f"safe_input_code IS NULL OR safe_input_code IN ({_RUNTIME_STEP_CODES})", name="ck_agent_step_input_code"),
        CheckConstraint(f"safe_output_code IS NULL OR safe_output_code IN ({_RUNTIME_STEP_CODES})", name="ck_agent_step_output_code"),
        Index("ix_agent_step_run_started_at", "run_id", "started_at"),
        _MYSQL_TABLE_OPTIONS,
    )

    @classmethod
    def from_model(cls, value: AgentStepRecord) -> "AgentStepRow":
        return cls(
            step_id=value.stepId,
            run_id=value.runId,
            data_contract_version=value.dataContractVersion,
            node_name=value.nodeName,
            attempt_number=value.attemptNumber,
            status=value.status.value,
            started_at=_utc_naive(value.startedAt),
            ended_at=_utc_naive(value.endedAt) if value.endedAt else None,
            error_code=value.errorCode,
            safe_input_code=value.safeInputCode,
            safe_output_code=value.safeOutputCode,
            snapshot_metadata=_trace_metadata_to_json(
                value.snapshotMetadata, value.evidenceProjection
            ),
        )

    def to_model(self) -> AgentStepRecord:
        snapshot, projection = _trace_metadata_from_json(self.snapshot_metadata)
        return AgentStepRecord(
            dataContractVersion=self.data_contract_version,
            stepId=self.step_id,
            runId=self.run_id,
            nodeName=self.node_name,
            attemptNumber=self.attempt_number,
            status=RuntimeStatus(self.status),
            startedAt=_utc_aware(self.started_at),
            endedAt=_utc_aware(self.ended_at) if self.ended_at else None,
            errorCode=self.error_code,
            safeInputCode=self.safe_input_code,
            safeOutputCode=self.safe_output_code,
            snapshotMetadata=snapshot,
            evidenceProjection=projection,
        )


class AgentArtifactRow(RuntimeOrmBase):
    __tablename__ = "agent_artifact"

    artifact_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), ForeignKey("agent_run.run_id"), nullable=False, index=True)
    data_contract_version: Mapped[str] = mapped_column(String(16), nullable=False)
    artifact_type: Mapped[str] = mapped_column(String(128), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(128), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(256), nullable=False)
    artifact_storage_ref: Mapped[str] = mapped_column(String(256), nullable=False)
    data_snapshot_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False, index=True)

    __table_args__ = (
        CheckConstraint(f"artifact_id REGEXP {_ARTIFACT_ID_PATTERN}", name="ck_agent_artifact_id"),
        CheckConstraint(f"run_id REGEXP {_RUN_ID_PATTERN}", name="ck_agent_artifact_run_id"),
        CheckConstraint(f"artifact_type IN ({_ARTIFACT_TYPES})", name="ck_agent_artifact_type"),
        CheckConstraint(f"data_snapshot_id IS NULL OR data_snapshot_id REGEXP {_SNAPSHOT_ID_PATTERN}", name="ck_agent_artifact_snapshot_id"),
        CheckConstraint("data_contract_version = 'v1'", name="ck_agent_artifact_contract_version"),
        CheckConstraint("content_hash REGEXP '^sha256:[0-9a-f]{64}$'", name="ck_agent_artifact_content_hash"),
        CheckConstraint("artifact_storage_ref REGEXP '^artifact://artifact-[0-9a-f]{32}$'", name="ck_agent_artifact_storage_ref"),
        CheckConstraint("artifact_storage_ref = CONCAT('artifact://', artifact_id)", name="ck_agent_artifact_storage_ref_binding"),
        _MYSQL_TABLE_OPTIONS,
    )

    @classmethod
    def from_model(cls, value: AgentArtifactRecord) -> "AgentArtifactRow":
        return cls(
            artifact_id=value.artifactId,
            run_id=value.runId,
            data_contract_version=value.dataContractVersion,
            artifact_type=value.artifactType,
            schema_version=value.schemaVersion,
            content_hash=value.contentHash,
            artifact_storage_ref=value.artifactStorageRef,
            data_snapshot_id=value.dataSnapshotId,
            created_at=_utc_naive(value.createdAt),
        )


class ApprovalTicketRow(RuntimeOrmBase):
    __tablename__ = "approval_ticket"

    approval_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), ForeignKey("agent_run.run_id"), nullable=False, index=True)
    data_contract_version: Mapped[str] = mapped_column(String(16), nullable=False)
    operation: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    requested_by: Mapped[str] = mapped_column(String(128), nullable=False)
    requested_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    decided_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    decision_code: Mapped[str | None] = mapped_column(String(128), nullable=True)

    __table_args__ = (
        CheckConstraint(f"status IN ({_APPROVAL_STATUS_VALUES})", name="ck_approval_ticket_status"),
        CheckConstraint(f"approval_id REGEXP {_APPROVAL_ID_PATTERN}", name="ck_approval_ticket_id"),
        CheckConstraint(f"run_id REGEXP {_RUN_ID_PATTERN}", name="ck_approval_ticket_run_id"),
        CheckConstraint(f"operation IN ({_APPROVAL_OPERATION_VALUES})", name="ck_approval_ticket_operation"),
        CheckConstraint(f"requested_by REGEXP {_PRINCIPAL_ID_PATTERN}", name="ck_approval_ticket_requested_by"),
        CheckConstraint(f"decided_by IS NULL OR decided_by REGEXP {_PRINCIPAL_ID_PATTERN}", name="ck_approval_ticket_decided_by"),
        CheckConstraint("data_contract_version = 'v1'", name="ck_approval_ticket_contract_version"),
        CheckConstraint(f"decision_code IS NULL OR decision_code REGEXP {_CONTROL_CODE_PATTERN}", name="ck_approval_ticket_decision_code"),
        _MYSQL_TABLE_OPTIONS,
    )

    @classmethod
    def from_model(cls, value: ApprovalTicketRecord) -> "ApprovalTicketRow":
        return cls(
            approval_id=value.approvalId,
            run_id=value.runId,
            data_contract_version=value.dataContractVersion,
            operation=value.operation,
            status=value.status.value,
            requested_by=value.requestedBy,
            requested_at=_utc_naive(value.requestedAt),
            decided_at=_utc_naive(value.decidedAt) if value.decidedAt else None,
            decided_by=value.decidedBy,
            decision_code=value.decisionCode,
        )

    def to_model(self) -> ApprovalTicketRecord:
        return ApprovalTicketRecord(
            dataContractVersion=self.data_contract_version,
            approvalId=self.approval_id,
            runId=self.run_id,
            operation=self.operation,
            status=ApprovalStatus(self.status),
            requestedBy=self.requested_by,
            requestedAt=_utc_aware(self.requested_at),
            decidedAt=_utc_aware(self.decided_at) if self.decided_at else None,
            decidedBy=self.decided_by,
            decisionCode=self.decision_code,
        )


class ToolAuditRow(RuntimeOrmBase):
    __tablename__ = "tool_audit"

    audit_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), ForeignKey("agent_run.run_id"), nullable=False, index=True)
    trace_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    data_contract_version: Mapped[str] = mapped_column(String(16), nullable=False)
    tool_name: Mapped[str] = mapped_column(String(128), nullable=False)
    tool_call_id: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(128), nullable=True)
    snapshot_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False, index=True)

    __table_args__ = (
        CheckConstraint(f"status IN ({_TOOL_STATUS_VALUES})", name="ck_tool_audit_status"),
        CheckConstraint(f"audit_id REGEXP {_AUDIT_ID_PATTERN}", name="ck_tool_audit_id"),
        CheckConstraint(f"run_id REGEXP {_RUN_ID_PATTERN}", name="ck_tool_audit_run_id"),
        CheckConstraint(f"trace_id REGEXP {_TRACE_ID_PATTERN}", name="ck_tool_audit_trace_id"),
        CheckConstraint(f"tool_call_id REGEXP {_CALL_ID_PATTERN}", name="ck_tool_audit_call_id"),
        CheckConstraint(f"tool_name IN ({_TOOL_NAMES})", name="ck_tool_audit_tool_name"),
        CheckConstraint("duration_ms >= 0", name="ck_tool_audit_duration"),
        CheckConstraint("data_contract_version = 'v1'", name="ck_tool_audit_contract_version"),
        CheckConstraint(f"error_code IS NULL OR error_code REGEXP {_CONTROL_CODE_PATTERN}", name="ck_tool_audit_error_code"),
        Index("ix_tool_audit_run_created_at", "run_id", "created_at"),
        _MYSQL_TABLE_OPTIONS,
    )

    @classmethod
    def from_model(cls, value: ToolAuditRecord) -> "ToolAuditRow":
        return cls(
            audit_id=value.auditId,
            run_id=value.runId,
            trace_id=value.traceId,
            data_contract_version=value.dataContractVersion,
            tool_name=value.toolName,
            tool_call_id=value.toolCallId,
            status=value.status,
            duration_ms=value.durationMs,
            error_code=value.errorCode,
            snapshot_metadata=_trace_metadata_to_json(
                value.snapshotMetadata, value.evidenceProjection
            ),
            created_at=_utc_naive(value.createdAt),
        )

    def to_model(self) -> ToolAuditRecord:
        snapshot, projection = _trace_metadata_from_json(self.snapshot_metadata)
        return ToolAuditRecord(
            dataContractVersion=self.data_contract_version,
            auditId=self.audit_id,
            runId=self.run_id,
            traceId=self.trace_id,
            toolName=self.tool_name,
            toolCallId=self.tool_call_id,
            status=self.status,
            durationMs=self.duration_ms,
            errorCode=self.error_code,
            snapshotMetadata=snapshot,
            evidenceProjection=projection,
            createdAt=_utc_aware(self.created_at),
        )


class MySqlRuntimeStore(RuntimeStore):
    """Async MySQL 8 adapter for the independent runtime database only."""

    persistent = True
    recoverable = False

    def __init__(self, configuration: RuntimeStorageConfiguration) -> None:
        self._configuration = configuration
        self._engine: AsyncEngine = create_async_engine(
            configuration.database_url,
            pool_pre_ping=True,
            future=True,
        )
        self._sessions = async_sessionmaker(self._engine, expire_on_commit=False)

    @classmethod
    def from_environment(cls, env: dict[str, str] | None = None) -> "MySqlRuntimeStore":
        return cls(RuntimeStorageConfiguration.from_environment(env))

    async def close(self) -> None:
        await self._engine.dispose()

    async def create_run(self, run: AgentRunRecord) -> AgentRunRecord:
        try:
            async with self._sessions.begin() as session:
                session.add(AgentRunRow.from_model(run))
        except IntegrityError:
            raise self._map_integrity_error("RUNTIME_RUN_ALREADY_EXISTS") from None
        return run.model_copy(deep=True)

    async def load_run(self, run_id: RunId) -> AgentRunRecord | None:
        async with self._sessions() as session:
            result = await session.execute(select(AgentRunRow).where(AgentRunRow.run_id == run_id))
            row = result.scalar_one_or_none()
            return row.to_model() if row else None

    async def update_encrypted_state(
        self,
        run_id: RunId,
        encrypted_state_payload: EncryptedStateEnvelope,
        *,
        updated_at: datetime,
    ) -> AgentRunRecord:
        async with self._sessions.begin() as session:
            result = await session.execute(
                select(AgentRunRow).where(AgentRunRow.run_id == run_id).with_for_update()
            )
            row = result.scalar_one_or_none()
            if row is None:
                raise RunNotFoundError("RUNTIME_RUN_NOT_FOUND")
            if row.encryption_key_id != encrypted_state_payload.keyId:
                raise RuntimeStoreError("RUNTIME_STATE_KEY_ID_MISMATCH")
            row.encrypted_state_payload = RuntimeStateCipher.serialize(encrypted_state_payload)
            row.updated_at = _utc_naive(updated_at)
            return row.to_model()

    async def list_steps(self, run_id: RunId) -> list[AgentStepRecord]:
        async with self._sessions() as session:
            result = await session.execute(
                select(AgentStepRow)
                .where(AgentStepRow.run_id == run_id)
                .order_by(AgentStepRow.started_at.asc(), AgentStepRow.step_id.asc())
            )
            return [row.to_model() for row in result.scalars().all()]

    async def append_step(self, step: AgentStepRecord) -> AgentStepRecord:
        try:
            async with self._sessions.begin() as session:
                await self._require_run(session, step.runId)
                session.add(AgentStepRow.from_model(step))
        except IntegrityError:
            raise self._map_integrity_error("RUNTIME_STEP_ALREADY_EXISTS") from None
        return step.model_copy(deep=True)

    async def mark_run_status(
        self,
        run_id: RunId,
        status: RuntimeStatus,
        *,
        failure_code: str | None = None,
        current_successful_step_id: StepId | None = None,
        updated_at: datetime | None = None,
    ) -> AgentRunRecord:
        async with self._sessions.begin() as session:
            result = await session.execute(
                select(AgentRunRow).where(AgentRunRow.run_id == run_id).with_for_update()
            )
            row = result.scalar_one_or_none()
            if row is None:
                raise RunNotFoundError("RUNTIME_RUN_NOT_FOUND")
            assert_status_transition(RuntimeStatus(row.status), status)
            row.status = status.value
            row.failure_code = failure_code
            row.current_successful_step_id = current_successful_step_id
            row.updated_at = _utc_naive(updated_at or datetime.now(timezone.utc))
            return row.to_model()

    async def create_approval_ticket(self, ticket: ApprovalTicketRecord) -> ApprovalTicketRecord:
        try:
            async with self._sessions.begin() as session:
                await self._require_run(session, ticket.runId)
                session.add(ApprovalTicketRow.from_model(ticket))
        except IntegrityError:
            raise self._map_integrity_error("RUNTIME_APPROVAL_ALREADY_EXISTS") from None
        return ticket.model_copy(deep=True)

    async def load_approval(self, approval_id) -> ApprovalTicketRecord | None:
        async with self._sessions() as session:
            result = await session.execute(
                select(ApprovalTicketRow).where(ApprovalTicketRow.approval_id == approval_id)
            )
            row = result.scalar_one_or_none()
            return row.to_model() if row else None

    async def decide_approval(
        self,
        approval_id,
        status: ApprovalStatus,
        *,
        decision_code: str,
        decided_by,
        decided_at: datetime,
    ) -> ApprovalTicketRecord:
        async with self._sessions.begin() as session:
            result = await session.execute(
                select(ApprovalTicketRow)
                .where(ApprovalTicketRow.approval_id == approval_id)
                .with_for_update()
            )
            row = result.scalar_one_or_none()
            if row is None:
                raise RunNotFoundError("RUNTIME_APPROVAL_NOT_FOUND")
            assert_approval_transition(ApprovalStatus(row.status), status)
            row.status = status.value
            row.decision_code = decision_code
            row.decided_by = decided_by
            row.decided_at = _utc_naive(decided_at)
            return row.to_model()

    async def append_tool_audit(self, audit: ToolAuditRecord) -> ToolAuditRecord:
        try:
            async with self._sessions.begin() as session:
                await self._require_run(session, audit.runId)
                session.add(ToolAuditRow.from_model(audit))
        except IntegrityError:
            raise self._map_integrity_error("RUNTIME_AUDIT_ALREADY_EXISTS") from None
        return audit.model_copy(deep=True)

    async def append_artifact(self, artifact: AgentArtifactRecord) -> AgentArtifactRecord:
        try:
            async with self._sessions.begin() as session:
                await self._require_run(session, artifact.runId)
                session.add(AgentArtifactRow.from_model(artifact))
        except IntegrityError:
            raise self._map_integrity_error("RUNTIME_ARTIFACT_ALREADY_EXISTS") from None
        return artifact.model_copy(deep=True)

    @staticmethod
    def _map_integrity_error(code: str) -> DuplicateRecordError:
        """Map integrity conflicts without exposing SQL or connection details."""

        return DuplicateRecordError(code)

    @staticmethod
    async def _require_run(session: AsyncSession, run_id: RunId) -> AgentRunRow:
        result = await session.execute(select(AgentRunRow).where(AgentRunRow.run_id == run_id))
        row = result.scalar_one_or_none()
        if row is None:
            raise RunNotFoundError("RUNTIME_RUN_NOT_FOUND")
        return row
