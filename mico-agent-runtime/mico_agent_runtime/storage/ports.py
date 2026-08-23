from __future__ import annotations

from datetime import datetime
from typing import Protocol

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
from .types import ApprovalId, RunId, StepId, PrincipalId


class RuntimeStoreError(RuntimeError):
    """Base error for the independent runtime state store."""


class RunNotFoundError(RuntimeStoreError):
    pass


class DuplicateRecordError(RuntimeStoreError):
    pass


class InvalidStateTransitionError(RuntimeStoreError):
    pass


class RuntimeStore(Protocol):
    """Persistence port; no LangGraph private checkpoint format is accepted."""

    persistent: bool
    recoverable: bool

    async def create_run(self, run: AgentRunRecord) -> AgentRunRecord:
        ...

    async def load_run(self, run_id: RunId) -> AgentRunRecord | None:
        ...

    async def update_encrypted_state(
        self,
        run_id: RunId,
        encrypted_state_payload: EncryptedStateEnvelope,
        *,
        updated_at: datetime,
    ) -> AgentRunRecord:
        """Replace only the encrypted checkpoint envelope for one run."""
        ...

    async def list_steps(self, run_id: RunId) -> list[AgentStepRecord]:
        """Return only the safe, typed step projections for one run."""
        ...

    async def append_step(self, step: AgentStepRecord) -> AgentStepRecord:
        ...

    async def mark_run_status(
        self,
        run_id: RunId,
        status: RuntimeStatus,
        *,
        failure_code: str | None = None,
        current_successful_step_id: StepId | None = None,
        updated_at: datetime | None = None,
    ) -> AgentRunRecord:
        ...

    async def create_approval_ticket(self, ticket: ApprovalTicketRecord) -> ApprovalTicketRecord:
        ...

    async def load_approval(self, approval_id: ApprovalId) -> ApprovalTicketRecord | None:
        ...

    async def decide_approval(
        self,
        approval_id: ApprovalId,
        status: ApprovalStatus,
        *,
        decision_code: str,
        decided_by: PrincipalId,
        decided_at: datetime,
    ) -> ApprovalTicketRecord:
        ...

    async def append_tool_audit(self, audit: ToolAuditRecord) -> ToolAuditRecord:
        ...

    async def append_artifact(self, artifact: AgentArtifactRecord) -> AgentArtifactRecord:
        ...


_ALLOWED_TRANSITIONS: dict[RuntimeStatus, frozenset[RuntimeStatus]] = {
    RuntimeStatus.QUEUED: frozenset({RuntimeStatus.RUNNING, RuntimeStatus.CANCELLED}),
    RuntimeStatus.RUNNING: frozenset({
        RuntimeStatus.WAITING_APPROVAL,
        RuntimeStatus.WAITING_JOB,
        RuntimeStatus.COMPLETED,
        RuntimeStatus.FAILED,
        RuntimeStatus.CANCELLED,
    }),
    RuntimeStatus.WAITING_APPROVAL: frozenset({RuntimeStatus.RUNNING, RuntimeStatus.FAILED, RuntimeStatus.CANCELLED}),
    RuntimeStatus.WAITING_JOB: frozenset({RuntimeStatus.RUNNING, RuntimeStatus.FAILED, RuntimeStatus.CANCELLED}),
    RuntimeStatus.COMPLETED: frozenset(),
    RuntimeStatus.FAILED: frozenset(),
    RuntimeStatus.CANCELLED: frozenset(),
}


def assert_status_transition(current: RuntimeStatus, target: RuntimeStatus) -> None:
    if target not in _ALLOWED_TRANSITIONS[current]:
        raise InvalidStateTransitionError("RUNTIME_STATE_TRANSITION_INVALID")


_ALLOWED_APPROVAL_TRANSITIONS: dict[ApprovalStatus, frozenset[ApprovalStatus]] = {
    ApprovalStatus.PENDING: frozenset({
        ApprovalStatus.APPROVED,
        ApprovalStatus.REJECTED,
        ApprovalStatus.EXPIRED,
        ApprovalStatus.CANCELLED,
    }),
    ApprovalStatus.APPROVED: frozenset(),
    ApprovalStatus.REJECTED: frozenset(),
    ApprovalStatus.EXPIRED: frozenset(),
    ApprovalStatus.CANCELLED: frozenset(),
}


def assert_approval_transition(current: ApprovalStatus, target: ApprovalStatus) -> None:
    if target not in _ALLOWED_APPROVAL_TRANSITIONS[current]:
        raise InvalidStateTransitionError("RUNTIME_APPROVAL_TRANSITION_INVALID")
