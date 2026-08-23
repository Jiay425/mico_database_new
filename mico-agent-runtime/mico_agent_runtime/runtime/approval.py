"""Fail-closed approval-ticket lifecycle over the Runtime Store."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from pydantic import TypeAdapter

from mico_agent_runtime.contracts.approval import (
    ApprovalDecisionRequest,
    ApprovalTicketRequest,
    ApprovalTicketResponse,
)
from mico_agent_runtime.governance.approval import ApprovalGate
from mico_agent_runtime.runtime.control import RunControlCoordinator, RunControlError
from mico_agent_runtime.storage.models import ApprovalStatus, ApprovalTicketRecord
from mico_agent_runtime.storage.ports import RuntimeStore, RuntimeStoreError
from mico_agent_runtime.storage.types import ApprovalId, PrincipalId


class ApprovalLifecycleError(RuntimeError):
    """Safe approval error; never contains SQL, payload or identity details."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class ApprovalCoordinator:
    """Create typed tickets and release execution only through run control.

    The ticket lifecycle is durable when the injected Runtime Store is durable.
    An APPROVED decision is fail-closed unless a controlled resume coordinator
    is injected and successfully resumes the associated run first. This avoids
    recording approval that has not actually released execution.
    """

    def __init__(self, store: RuntimeStore,
                 run_control_coordinator: RunControlCoordinator | None = None) -> None:
        self._store = store
        self._gate = ApprovalGate()
        self._run_control = run_control_coordinator

    async def request(self, value: ApprovalTicketRequest) -> ApprovalTicketResponse:
        decision = self._gate.require_decision(value.operation)
        if decision.code == "APPROVAL_NOT_REQUIRED":
            raise ApprovalLifecycleError("APPROVAL_NOT_REQUIRED")
        record = ApprovalTicketRecord(
            dataContractVersion="v1",
            approvalId="approval-" + uuid4().hex,
            runId=value.runId,
            operation=value.operation,
            status=ApprovalStatus.PENDING,
            requestedBy=value.requestedBy,
            requestedAt=datetime.now(timezone.utc),
            decisionCode="APPROVAL_REQUIRED",
        )
        try:
            stored = await self._store.create_approval_ticket(record)
        except RuntimeStoreError:
            raise ApprovalLifecycleError("APPROVAL_PERSISTENCE_FAILED") from None
        return _response(stored)

    async def get(self, approval_id: str) -> ApprovalTicketResponse | None:
        try:
            typed_id = TypeAdapter(ApprovalId).validate_python(approval_id)
            value = await self._store.load_approval(typed_id)
        except RuntimeStoreError:
            raise ApprovalLifecycleError("APPROVAL_PERSISTENCE_FAILED") from None
        except Exception:
            return None
        return _response(value) if value is not None else None

    async def decide(
        self,
        approval_id: str,
        value: ApprovalDecisionRequest,
        decided_by: str,
    ) -> ApprovalTicketResponse:
        try:
            typed_id = TypeAdapter(ApprovalId).validate_python(approval_id)
            typed_principal = TypeAdapter(PrincipalId).validate_python(decided_by)
        except Exception:
            raise ApprovalLifecycleError("APPROVAL_REQUEST_INVALID") from None
        target = ApprovalStatus.APPROVED if value.decision == "APPROVED" else ApprovalStatus.REJECTED
        code = "APPROVAL_GRANTED" if target == ApprovalStatus.APPROVED else "APPROVAL_REJECTED"
        if target == ApprovalStatus.APPROVED:
            run_id = await self._approval_run_id(typed_id)
            if self._run_control is None:
                raise ApprovalLifecycleError("APPROVAL_RESUME_NOT_AVAILABLE")
            try:
                resumed = await self._run_control.resume(run_id)
            except (RunControlError, RuntimeStoreError):
                raise ApprovalLifecycleError("APPROVAL_RESUME_NOT_AVAILABLE") from None
            if resumed.status not in {"RUNNING", "COMPLETED"} or resumed.controlCode != "RUN_RESUMED":
                raise ApprovalLifecycleError("APPROVAL_RESUME_NOT_AVAILABLE")
        try:
            stored = await self._store.decide_approval(
                typed_id,
                target,
                decision_code=code,
                decided_by=typed_principal,
                decided_at=datetime.now(timezone.utc),
            )
        except RuntimeStoreError as exc:
            if str(exc) == "RUNTIME_APPROVAL_NOT_FOUND":
                raise ApprovalLifecycleError("APPROVAL_NOT_FOUND") from None
            if str(exc) == "RUNTIME_APPROVAL_TRANSITION_INVALID":
                raise ApprovalLifecycleError("APPROVAL_ALREADY_TERMINAL") from None
            raise ApprovalLifecycleError("APPROVAL_PERSISTENCE_FAILED") from None
        return _response(stored)

    async def _approval_run_id(self, approval_id: ApprovalId) -> str:
        """Load only the typed run binding needed before approval release."""

        ticket = await self._store.load_approval(approval_id)
        if ticket is None:
            raise ApprovalLifecycleError("APPROVAL_NOT_FOUND")
        if ticket.status != ApprovalStatus.PENDING:
            raise ApprovalLifecycleError("APPROVAL_ALREADY_TERMINAL")
        return ticket.runId


def _response(value: ApprovalTicketRecord | None) -> ApprovalTicketResponse | None:
    if value is None:
        return None
    return ApprovalTicketResponse.model_validate(value.model_dump(mode="json"))
