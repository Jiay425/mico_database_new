from __future__ import annotations

from typing import Any, TypedDict

from mico_agent_runtime.contracts.audit import AuditEvent
from mico_agent_runtime.contracts.evidence import (
    EvidenceQuery,
    EvidenceTaskRequest,
    LiteratureEvidenceItem,
)


class EvidenceState(TypedDict, total=False):
    request: EvidenceTaskRequest | dict[str, Any]
    queries: list[EvidenceQuery]
    evidence: list[LiteratureEvidenceItem]
    report: Any
    status: str
    errorCode: str
    auditEvents: list[AuditEvent]
