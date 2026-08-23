from __future__ import annotations

from datetime import datetime, timezone
import re
from threading import RLock
from typing import Literal

from pydantic import Field, StringConstraints
from typing_extensions import Annotated

from .audit import AuditEvent
from .base import ClosedModel, Identifier, NonEmptyText
from .intent import ApprovedWorkflow, IntentRunResult
from .metrics import RuntimeMetrics
from .tools import AllowedToolName, ToolStatus, TransientPersistence, TransientSnapshotId


RuntimeProgressStatus = Literal[
    "QUEUED",
    "RUNNING",
    "WAITING_APPROVAL",
    "WAITING_JOB",
    "COMPLETED",
    "FAILED",
    "CANCELLED",
]
RuntimeProgressNode = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{1,63}$")]
RuntimeProgressErrorCode = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+$", max_length=128),
]


class RuntimeProgressEvent(ClosedModel):
    """Safe event projection; it contains no request or tool payload."""

    sequence: int = Field(strict=True, ge=1, le=1000)
    node: RuntimeProgressNode
    toolName: AllowedToolName | None = None
    status: ToolStatus
    errorCode: RuntimeProgressErrorCode | None = None
    dataSnapshotId: TransientSnapshotId | None = None
    snapshotPersistence: TransientPersistence | None = None
    occurredAt: datetime


class RuntimeProgressResponse(ClosedModel):
    """Ephemeral progress view, intentionally without trace or raw request data."""

    runId: Identifier
    taskId: Identifier
    workflow: ApprovedWorkflow | None = None
    status: RuntimeProgressStatus
    currentNode: RuntimeProgressNode | None = None
    errorCode: RuntimeProgressErrorCode | None = None
    events: list[RuntimeProgressEvent] = Field(default_factory=list, max_length=1000)
    metrics: RuntimeMetrics | None = None
    updatedAt: datetime


def _safe_error_code(value: str | None) -> str | None:
    if value is None:
        return None
    candidate = value.strip()
    if not candidate or re.fullmatch(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+$", candidate) is None:
        return None
    return candidate


def _runtime_status(status: ToolStatus) -> RuntimeProgressStatus:
    if status == "COMPLETED":
        return "COMPLETED"
    if status == "WAITING_APPROVAL":
        return "WAITING_APPROVAL"
    return "FAILED"


class RunProgressRegistry:
    """Process-local, safe progress registry.

    This is deliberately not a checkpoint or a durable audit store. It is a
    small adapter for the internal progress endpoint until the Runtime Store
    is connected to lifecycle execution in a later deployment phase.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._runs: dict[str, RuntimeProgressResponse] = {}

    def begin(self, run_id: str, task_id: str, workflow: ApprovedWorkflow | None = None) -> None:
        now = datetime.now(timezone.utc)
        with self._lock:
            self._runs[run_id] = RuntimeProgressResponse(
                runId=run_id,
                taskId=task_id,
                workflow=workflow,
                status="RUNNING",
                updatedAt=now,
            )

    def complete(self, result: IntentRunResult) -> None:
        status = _runtime_status(result.status)
        events = self._safe_events(result.auditEvents)
        now = datetime.now(timezone.utc)
        with self._lock:
            self._runs[result.runId] = RuntimeProgressResponse(
                runId=result.runId,
                taskId=result.taskId,
                workflow=result.workflow,
                status=status,
                currentNode=events[-1].node if events else None,
                errorCode=_safe_error_code(result.errorCode),
                events=events,
                metrics=result.metrics,
                updatedAt=now,
            )

    def record_event(self, event: AuditEvent) -> None:
        """Project one in-flight audit event without retaining raw state."""

        with self._lock:
            current = self._runs.get(event.runId)
            if current is None:
                return
            safe_event = self._safe_event(event, len(current.events) + 1)
            if safe_event is None:
                return
            events = (current.events + [safe_event])[:1000]
            status = current.status
            if event.node == "terminal":
                status = _runtime_status(event.status)
            self._runs[event.runId] = current.model_copy(update={
                "status": status,
                "currentNode": safe_event.node,
                "errorCode": safe_event.errorCode,
                "events": events,
                "updatedAt": safe_event.occurredAt,
            })

    def fail(self, run_id: str, task_id: str, code: str) -> None:
        now = datetime.now(timezone.utc)
        with self._lock:
            self._runs[run_id] = RuntimeProgressResponse(
                runId=run_id,
                taskId=task_id,
                status="FAILED",
                errorCode=_safe_error_code(code),
                updatedAt=now,
            )

    def get(self, run_id: str) -> RuntimeProgressResponse | None:
        with self._lock:
            value = self._runs.get(run_id)
            return value.model_copy(deep=True) if value is not None else None

    @staticmethod
    def _safe_events(events: list[AuditEvent]) -> list[RuntimeProgressEvent]:
        result: list[RuntimeProgressEvent] = []
        for sequence, event in enumerate(events[:1000], start=1):
            safe = RunProgressRegistry._safe_event(event, sequence)
            if safe is not None:
                result.append(safe)
        return result

    @staticmethod
    def _safe_event(event: AuditEvent, sequence: int) -> RuntimeProgressEvent | None:
        allowed_tools = set(AllowedToolName.__args__)
        try:
            tool_name = event.toolName if event.toolName in allowed_tools else None
            return RuntimeProgressEvent(
                sequence=sequence,
                node=event.node,
                toolName=tool_name,
                status=event.status,
                errorCode=_safe_error_code(event.errorCode),
                dataSnapshotId=event.dataSnapshotId,
                snapshotPersistence=event.snapshotPersistence,
                occurredAt=event.occurredAt,
            )
        except (TypeError, ValueError):
            return None
