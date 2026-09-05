from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, field_validator

from .base import ClosedModel, Identifier, NonEmptyText
from .evidence_report import EvidenceId, EvidenceReference, EvidenceReviewReport, EvidenceSource
from .graph_rag import GraphEvidencePath, ReasoningPath
from .retrieval import RerankBreakdown, RetrievalScope


EvidenceDirection = Literal["supporting", "contrary", "context"]
KnowledgeRetrievalMode = Literal["auto", "vector", "graph", "hybrid"]
EvidenceRunStatus = Literal["COMPLETED", "INSUFFICIENT_EVIDENCE", "REJECTED", "FAILED"]


class EvidenceTaskRequest(ClosedModel):
    """Generic, closed literature retrieval request for a later evidence stage."""

    runId: Identifier
    taskId: Identifier
    requesterId: Identifier
    traceId: Identifier
    topic: Annotated[str, StringConstraints(min_length=1, max_length=1024)]
    taxonNames: list[Annotated[str, StringConstraints(min_length=1, max_length=512)]] = Field(
        default_factory=list, max_length=20
    )
    requestedDirections: list[EvidenceDirection] = Field(min_length=1, max_length=3)
    retrievalMode: KnowledgeRetrievalMode = "hybrid"
    # Optional request/UI-supplied document boundary.  The default remains a
    # deliberately explicit global search for ordinary corpus questions.
    retrievalScope: RetrievalScope = Field(default_factory=RetrievalScope)
    limit: int = Field(default=10, strict=True, ge=1, le=10)
    createdAt: datetime

    @field_validator("createdAt")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("createdAt must include a timezone")
        return value.astimezone(timezone.utc)

    @field_validator("requestedDirections", "taxonNames")
    @classmethod
    def reject_duplicates(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("duplicate evidence values are not allowed")
        return value


class EvidenceQuery(ClosedModel):
    topic: Annotated[str, StringConstraints(min_length=1, max_length=1024)]
    taxonName: Annotated[str, StringConstraints(min_length=1, max_length=512)] | None = None
    direction: EvidenceDirection
    retrievalMode: KnowledgeRetrievalMode = "hybrid"
    retrievalScope: RetrievalScope = Field(default_factory=RetrievalScope)
    limit: int = Field(strict=True, ge=1, le=10)


class LiteratureEvidenceItem(ClosedModel):
    evidenceId: EvidenceId
    taxonName: Annotated[str, StringConstraints(min_length=1, max_length=512)] | None = None
    source: EvidenceSource
    externalId: NonEmptyText
    title: Annotated[str, StringConstraints(min_length=1, max_length=512)]
    journal: Annotated[str, StringConstraints(min_length=1, max_length=256)] | None = None
    publicationYear: int = Field(strict=True, ge=1900, le=2100)
    direction: EvidenceDirection
    summary: Annotated[str, StringConstraints(min_length=1, max_length=2000)]
    evidenceTier: Literal["fulltext", "abstract", "metadata"] = "metadata"
    retrievalRoute: Literal["vector", "sparse", "graph", "hybrid"] | None = None
    retrievalModel: Literal["gemini-embedding-2", "fulltext-tfidf-cosine-v1", "postgres-fulltext-bm25-v1"] | None = None
    sourceChunkId: Annotated[str, StringConstraints(min_length=1, max_length=128)] | None = None
    retrievalScore: float = Field(default=0.0, ge=0.0)
    sourceExcerpt: Annotated[str, StringConstraints(min_length=1, max_length=1200)] | None = None
    vectorScore: float = Field(default=0.0, ge=0.0)
    sparseScore: float = Field(default=0.0, ge=0.0)
    graphScore: float = Field(default=0.0, ge=0.0)
    rerankScore: float = Field(default=0.0, ge=0.0)
    rerankBreakdown: RerankBreakdown | None = None
    graphPaths: list[GraphEvidencePath] = Field(default_factory=list, max_length=4)
    reasoningPaths: list[ReasoningPath] = Field(default_factory=list, max_length=4)
    retrievalSources: list[Literal["vector", "sparse", "graph"]] = Field(default_factory=list, max_length=3)


class EvidenceRunResult(ClosedModel):
    traceId: Identifier
    runId: Identifier
    taskId: Identifier
    status: EvidenceRunStatus
    errorCode: str | None = None
    report: EvidenceReviewReport | None = None
