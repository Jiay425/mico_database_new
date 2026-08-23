from __future__ import annotations

from typing import Any, TypedDict

from mico_agent_runtime.contracts.audit import AuditEvent
from mico_agent_runtime.contracts.intent import IntentRoutePlan, IntentTaskRequest
from mico_agent_runtime.workflow.catalog import WorkflowToolPlan


class IntentState(TypedDict, total=False):
    request: IntentTaskRequest | dict[str, Any]
    routePlan: IntentRoutePlan
    plannerMode: str
    workflow: str
    toolPlan: WorkflowToolPlan
    subResult: Any
    report: Any
    status: str
    errorCode: str
    auditEvents: list[AuditEvent]
    # Process-local only; never part of a wire or persistence model.
    progressCallback: Any
    reviewRequired: bool
    reviewPayload: dict[str, Any]
    reviewDecision: Any
