from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import datetime, timezone

from .crypto import EncryptedStateEnvelope

from .models import (
    AgentArtifactRecord,
    AgentRunRecord,
    AgentStepRecord,
    ApprovalTicketRecord,
    ApprovalStatus,
    RuntimeStatus,
    ToolAuditRecord,
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


class InMemoryRuntimeStore(RuntimeStore):
    """Test-only store; state is neither persistent nor recoverable."""

    persistent = False
    recoverable = False

    def __init__(self) -> None:
        self._runs: dict[str, AgentRunRecord] = {}
        self._steps: dict[str, AgentStepRecord] = {}
        self._tickets: dict[str, ApprovalTicketRecord] = {}
        self._audits: dict[str, ToolAuditRecord] = {}
        self._artifacts: dict[str, AgentArtifactRecord] = {}
        self._lock = asyncio.Lock()

    async def create_run(self, run: AgentRunRecord) -> AgentRunRecord:
        async with self._lock:
            if run.runId in self._runs:
                raise DuplicateRecordError("RUNTIME_RUN_ALREADY_EXISTS")
            self._runs[run.runId] = deepcopy(run)
            return deepcopy(run)

    async def load_run(self, run_id: RunId) -> AgentRunRecord | None:
        async with self._lock:
            value = self._runs.get(run_id)
            return deepcopy(value) if value else None

    async def update_encrypted_state(
        self,
        run_id: RunId,
        encrypted_state_payload: EncryptedStateEnvelope,
        *,
        updated_at: datetime,
    ) -> AgentRunRecord:
        async with self._lock:
            run = self._require_run(run_id)
            if run.encryptionKeyId != encrypted_state_payload.keyId:
                raise RuntimeStoreError("RUNTIME_STATE_KEY_ID_MISMATCH")
            run.encryptedStatePayload = deepcopy(encrypted_state_payload)
            run.updatedAt = updated_at
            return deepcopy(run)

    async def list_steps(self, run_id: RunId) -> list[AgentStepRecord]:
        async with self._lock:
            self._require_run(run_id)
            values = [value for value in self._steps.values() if value.runId == run_id]
            values.sort(key=lambda value: (value.startedAt, value.stepId))
            return deepcopy(values)

    async def append_step(self, step: AgentStepRecord) -> AgentStepRecord:
        async with self._lock:
            self._require_run(step.runId)
            if step.stepId in self._steps:
                raise DuplicateRecordError("RUNTIME_STEP_ALREADY_EXISTS")
            self._steps[step.stepId] = deepcopy(step)
            return deepcopy(step)

    async def mark_run_status(
        self,
        run_id: RunId,
        status: RuntimeStatus,
        *,
        failure_code: str | None = None,
        current_successful_step_id: StepId | None = None,
        updated_at: datetime | None = None,
    ) -> AgentRunRecord:
        async with self._lock:
            current = self._require_run(run_id)
            assert_status_transition(current.status, status)
            current.status = status
            current.failureCode = failure_code
            current.currentSuccessfulStepId = current_successful_step_id
            current.updatedAt = updated_at or datetime.now(timezone.utc)
            return deepcopy(current)

    async def create_approval_ticket(self, ticket: ApprovalTicketRecord) -> ApprovalTicketRecord:
        async with self._lock:
            self._require_run(ticket.runId)
            if ticket.approvalId in self._tickets:
                raise DuplicateRecordError("RUNTIME_APPROVAL_ALREADY_EXISTS")
            self._tickets[ticket.approvalId] = deepcopy(ticket)
            return deepcopy(ticket)

    async def load_approval(self, approval_id) -> ApprovalTicketRecord | None:
        async with self._lock:
            value = self._tickets.get(approval_id)
            return deepcopy(value) if value else None

    async def decide_approval(
        self,
        approval_id,
        status: ApprovalStatus,
        *,
        decision_code: str,
        decided_by,
        decided_at,
    ) -> ApprovalTicketRecord:
        async with self._lock:
            ticket = self._tickets.get(approval_id)
            if ticket is None:
                raise RunNotFoundError("RUNTIME_APPROVAL_NOT_FOUND")
            assert_approval_transition(ticket.status, status)
            ticket.status = status
            ticket.decisionCode = decision_code
            ticket.decidedBy = decided_by
            ticket.decidedAt = decided_at
            return deepcopy(ticket)

    async def append_tool_audit(self, audit: ToolAuditRecord) -> ToolAuditRecord:
        async with self._lock:
            self._require_run(audit.runId)
            if audit.auditId in self._audits:
                raise DuplicateRecordError("RUNTIME_AUDIT_ALREADY_EXISTS")
            self._audits[audit.auditId] = deepcopy(audit)
            return deepcopy(audit)

    async def append_artifact(self, artifact: AgentArtifactRecord) -> AgentArtifactRecord:
        async with self._lock:
            self._require_run(artifact.runId)
            if artifact.artifactId in self._artifacts:
                raise DuplicateRecordError("RUNTIME_ARTIFACT_ALREADY_EXISTS")
            self._artifacts[artifact.artifactId] = deepcopy(artifact)
            return deepcopy(artifact)

    def _require_run(self, run_id: RunId) -> AgentRunRecord:
        run = self._runs.get(run_id)
        if run is None:
            raise RunNotFoundError("RUNTIME_RUN_NOT_FOUND")
        return run
