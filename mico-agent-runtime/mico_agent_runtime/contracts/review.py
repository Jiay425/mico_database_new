"""Closed contracts for graph-quality review and safe evidence-path display.

These contracts deliberately expose only versioned graph identifiers, bounded
literature labels and source references.  They do not carry business sample
locators, Java payloads, credentials or free-form approval text.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator
from typing_extensions import TypeAlias

from .base import ClosedModel
from .graph_rag import GraphPathStep, GraphPathStatus


GraphReviewStatus = Literal["PENDING", "APPROVED", "REJECTED"]
GraphReviewIssueCode = Literal[
    "LOW_CONFIDENCE_RELATION",
    "CONFLICTING_ASSERTIONS",
    "SPECULATIVE_RELATION",
    "CANDIDATE_ENDPOINT",
    "AMBIGUOUS_ENDPOINT",
]
GraphReviewDecisionCode = Literal[
    "RELATION_ACCEPTED",
    "RELATION_REJECTED",
]
GraphReviewSeverity = Literal["warning", "high"]
GraphPublicationDecision = Literal["APPROVED"]
GraphAssertionStatus = Literal["asserted", "negated", "speculative", "conflicted"]
PrincipalReference = Annotated[str, Field(pattern=r"^principal-[0-9a-f]{32}$")]
GraphVersion = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]{1,79}$")]
GraphBuildReference = Annotated[str, Field(pattern=r"^graph-build-[0-9a-f]{32}$")]
GraphRecordReference = Annotated[str, Field(min_length=1, max_length=192)]
ReviewReference = Annotated[str, Field(pattern=r"^review-[0-9a-f]{32}$")]
Sha256Reference = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]


def _reject_sensitive_display_text(value: str | None) -> str | None:
    if value is None:
        return None
    import re

    lowered = value.lower()
    forbidden = (
        "sourcesampleid=", "internalrecordid=", "cohortcondition",
        "authorization:", "bearer ", "jdbc:", "mysql://", "postgresql://",
        "ssh://", "file://", "http://", "https://", "select ", "insert ",
        "update ", "delete ", "drop ",
    )
    if any(marker in lowered for marker in forbidden):
        raise ValueError("unsafe evidence display text")
    if re.search(r"\b(?:srr|err|drr)\d+\b", lowered):
        raise ValueError("sample identifiers are not displayable in graph review")
    return value


class EvidencePathView(ClosedModel):
    """Safe path projection for review UI and audit explanations."""

    pathId: Annotated[str, Field(pattern=r"^path-[0-9a-f]{32}$")]
    status: GraphPathStatus
    hops: list[GraphPathStep] = Field(min_length=1, max_length=4)
    sourceDocumentIds: list[Annotated[str, Field(min_length=1, max_length=128)]] = Field(
        min_length=1, max_length=4
    )
    confidence: float = Field(ge=0.0, le=1.0)
    reviewRequired: bool = False

    @field_validator("sourceDocumentIds", mode="after")
    @classmethod
    def validate_document_ids(cls, value: list[str]) -> list[str]:
        return [_reject_sensitive_display_text(item) or item for item in value]

    @model_validator(mode="after")
    def validate_hops(self) -> "EvidencePathView":
        for hop in self.hops:
            _reject_sensitive_display_text(hop.fromEntity)
            _reject_sensitive_display_text(hop.toEntity)
            _reject_sensitive_display_text(hop.relation)
            _reject_sensitive_display_text(hop.evidenceChunkId)
        return self


class GraphReviewTicket(ClosedModel):
    """One low-confidence or conflicting graph record awaiting a reviewer."""

    reviewId: ReviewReference
    graphVersion: GraphVersion
    buildRunId: GraphBuildReference
    recordId: GraphRecordReference
    issueCode: GraphReviewIssueCode
    status: GraphReviewStatus = "PENDING"
    severity: GraphReviewSeverity
    confidence: float = Field(ge=0.0, le=1.0)
    sourceEntity: Annotated[str, Field(min_length=1, max_length=512)]
    relation: Annotated[str, Field(min_length=1, max_length=96)]
    targetEntity: Annotated[str, Field(min_length=1, max_length=512)]
    assertionStatus: GraphAssertionStatus
    evidenceChunkId: Annotated[str, Field(min_length=1, max_length=128)] | None = None
    evidenceExcerpt: Annotated[str, Field(min_length=1, max_length=1200)] | None = None
    path: EvidencePathView | None = None
    dataContractVersion: Literal["v1"] = "v1"

    @field_validator(
        "sourceEntity", "relation", "targetEntity", "evidenceChunkId", "evidenceExcerpt",
        mode="after",
    )
    @classmethod
    def validate_display_text(cls, value: str | None) -> str | None:
        return _reject_sensitive_display_text(value)


class GraphReviewDecision(ClosedModel):
    """A typed reviewer decision; identity is supplied by the trusted caller."""

    reviewId: ReviewReference
    decision: Literal["APPROVED", "REJECTED"]
    decisionCode: GraphReviewDecisionCode
    decidedBy: PrincipalReference
    decidedAt: datetime
    dataContractVersion: Literal["v1"] = "v1"

    @field_validator("decidedAt")
    @classmethod
    def require_aware_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("review decision timestamp must include timezone")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def bind_decision_code(self) -> "GraphReviewDecision":
        expected = "RELATION_ACCEPTED" if self.decision == "APPROVED" else "RELATION_REJECTED"
        if self.decisionCode != expected:
            raise ValueError("review decision code does not match decision")
        return self


class GraphReviewQueue(ClosedModel):
    """Deterministic review queue derived from one immutable staging build."""

    graphVersion: GraphVersion
    buildRunId: GraphBuildReference
    status: Literal["OPEN", "CLOSED"]
    items: list[GraphReviewTicket] = Field(max_length=10000)
    queueHash: Sha256Reference
    pendingCount: int = Field(strict=True, ge=0)
    approvedCount: int = Field(strict=True, ge=0)
    rejectedCount: int = Field(strict=True, ge=0)
    dataContractVersion: Literal["v1"] = "v1"

    @model_validator(mode="after")
    def validate_counts(self) -> "GraphReviewQueue":
        counts = {
            "pendingCount": sum(item.status == "PENDING" for item in self.items),
            "approvedCount": sum(item.status == "APPROVED" for item in self.items),
            "rejectedCount": sum(item.status == "REJECTED" for item in self.items),
        }
        if any(getattr(self, key) != value for key, value in counts.items()):
            raise ValueError("review queue counts do not match items")
        if self.status == "OPEN" and self.pendingCount == 0:
            raise ValueError("closed review queue is required when no items are pending")
        if self.status == "CLOSED" and self.pendingCount != 0:
            raise ValueError("open review queue is required while items are pending")
        return self


class GraphPublicationApproval(ClosedModel):
    """Approval bound to one exact review queue before version switching."""

    graphVersion: GraphVersion
    buildRunId: GraphBuildReference
    reviewQueueHash: Sha256Reference
    reviewedItemCount: int = Field(strict=True, ge=1, le=10000)
    decision: GraphPublicationDecision
    approvedBy: PrincipalReference
    approvedAt: datetime
    approvalCode: Literal["GRAPH_REVIEW_APPROVED"] = "GRAPH_REVIEW_APPROVED"
    dataContractVersion: Literal["v1"] = "v1"

    @field_validator("approvedAt")
    @classmethod
    def require_aware_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("publication approval timestamp must include timezone")
        return value.astimezone(timezone.utc)


class GraphReviewResumeCommand(ClosedModel):
    """Only the closed decision needed to resume an interrupted graph."""

    reviewId: ReviewReference
    decision: Literal["APPROVED", "REJECTED"]
    dataContractVersion: Literal["v1"] = "v1"
