"""Closed approval-ticket wire contracts for the independent Runtime."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import Field, field_validator
from typing_extensions import Annotated

from .base import ClosedModel


ApprovalOperation = Literal[
    "read_only_research",
    "statistical_analysis",
    "sensitive_batch_read",
    "bulk_export",
    "external_publish",
    "business_write",
]
ApprovalStatusValue = Literal["PENDING", "APPROVED", "REJECTED", "EXPIRED", "CANCELLED"]
ApprovalDecision = Literal["APPROVED", "REJECTED"]
ApprovalCode = Annotated[
    str,
    Field(min_length=3, max_length=128, pattern=r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+$"),
]
RunReference = Annotated[str, Field(pattern=r"^run-[0-9a-f]{32}$")]
PrincipalReference = Annotated[str, Field(pattern=r"^principal-[0-9a-f]{32}$")]
ApprovalReference = Annotated[str, Field(pattern=r"^approval-[0-9a-f]{32}$")]


class ApprovalTicketRequest(ClosedModel):
    """Request for a ticket; no raw arguments or free-text justification."""

    runId: RunReference
    operation: ApprovalOperation
    requestedBy: PrincipalReference
    dataContractVersion: Literal["v1"] = "v1"


class ApprovalDecisionRequest(ClosedModel):
    """Decision body intentionally contains no caller-supplied identity."""

    decision: ApprovalDecision
    dataContractVersion: Literal["v1"] = "v1"


class ApprovalTicketResponse(ClosedModel):
    approvalId: ApprovalReference
    runId: RunReference
    operation: ApprovalOperation
    status: ApprovalStatusValue
    requestedBy: PrincipalReference
    requestedAt: datetime
    decidedAt: datetime | None = None
    decidedBy: PrincipalReference | None = None
    decisionCode: ApprovalCode | None = None
    dataContractVersion: Literal["v1"] = "v1"

    @field_validator("requestedAt", "decidedAt")
    @classmethod
    def require_aware_utc(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("approval timestamps must include a timezone")
        return value.astimezone(timezone.utc)


class ApprovalDisabledResponse(ClosedModel):
    code: Literal[
        "APPROVAL_WORKFLOW_NOT_CONFIGURED",
        "APPROVAL_DECISION_NOT_CONFIGURED",
        "APPROVAL_NOT_REQUIRED",
    ]
    message: str = "The approval workflow is not available"
