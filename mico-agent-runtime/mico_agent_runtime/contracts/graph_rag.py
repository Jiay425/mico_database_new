from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from .base import ClosedModel


GraphPathStatus = Literal["supported", "speculative", "conflicted", "partial", "unsupported"]


class GraphPathStep(ClosedModel):
    """One auditable hop in a literature graph path.

    Node identifiers are intentionally not exposed.  The path contains
    display labels and the exact full-text chunk that supports the hop.
    """

    fromEntity: Annotated[str, StringConstraints(min_length=1, max_length=512)]
    relation: Annotated[str, StringConstraints(min_length=1, max_length=96)]
    toEntity: Annotated[str, StringConstraints(min_length=1, max_length=512)]
    evidenceChunkId: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    supportStatus: GraphPathStatus = "supported"


class GraphEvidencePath(ClosedModel):
    pathId: Annotated[str, StringConstraints(pattern=r"^path-[0-9a-f]{32}$")]
    status: GraphPathStatus
    hops: list[GraphPathStep] = Field(min_length=1, max_length=4)
    sourceDocumentIds: list[Annotated[str, StringConstraints(min_length=1, max_length=128)]] = Field(
        min_length=1, max_length=4
    )
    confidence: float = Field(ge=0.0, le=1.0)


class ReasoningPath(ClosedModel):
    """Explicit multi-hop path used by reranking and grounded generation."""

    pathId: Annotated[str, StringConstraints(pattern=r"^path-[0-9a-f]{32}$")]
    status: GraphPathStatus
    hops: list[GraphPathStep] = Field(min_length=1, max_length=4)
    sourceDocumentIds: list[Annotated[str, StringConstraints(min_length=1, max_length=128)]] = Field(
        min_length=1, max_length=4
    )
    confidence: float = Field(ge=0.0, le=1.0)
    hopCount: int = Field(strict=True, ge=1, le=4)
    evidenceChunkIds: list[Annotated[str, StringConstraints(min_length=1, max_length=128)]] = Field(
        min_length=1, max_length=4
    )
    pathScore: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_path_shape(self) -> "ReasoningPath":
        expected_chunks = list(dict.fromkeys(hop.evidenceChunkId for hop in self.hops))
        if self.hopCount != len(self.hops):
            raise ValueError("hopCount must equal hops length")
        if self.evidenceChunkIds != expected_chunks:
            raise ValueError("evidenceChunkIds must match path hops")
        return self

    @classmethod
    def from_graph_path(cls, path: GraphEvidencePath) -> "ReasoningPath":
        return cls(
            pathId=path.pathId,
            status=path.status,
            hops=path.hops,
            sourceDocumentIds=path.sourceDocumentIds,
            confidence=path.confidence,
            hopCount=len(path.hops),
            evidenceChunkIds=list(dict.fromkeys(hop.evidenceChunkId for hop in path.hops)),
            pathScore=path.confidence,
        )

    def to_graph_path(self) -> GraphEvidencePath:
        return GraphEvidencePath(
            pathId=self.pathId,
            status=self.status,
            hops=self.hops,
            sourceDocumentIds=self.sourceDocumentIds,
            confidence=self.confidence,
        )


class GroundedClaim(ClosedModel):
    """A structured claim envelope; unsupported hops cannot be presented as facts."""

    claimId: Annotated[str, StringConstraints(pattern=r"^claim-[0-9a-f]{32}$")]
    statement: Annotated[str, StringConstraints(min_length=1, max_length=1200)]
    supportStatus: GraphPathStatus
    evidenceIds: list[Annotated[str, StringConstraints(pattern=r"^evidence-[0-9a-f]{32}$")]] = Field(
        min_length=1, max_length=10
    )
    reasoningPathIds: list[Annotated[str, StringConstraints(pattern=r"^path-[0-9a-f]{32}$")]] = Field(
        default_factory=list, max_length=10
    )


class SynthesisEvidence(ClosedModel):
    """Bounded literature context supplied to a generator."""

    evidenceId: Annotated[str, StringConstraints(pattern=r"^evidence-[0-9a-f]{32}$")]
    title: Annotated[str, StringConstraints(min_length=1, max_length=512)]
    summary: Annotated[str, StringConstraints(min_length=1, max_length=2000)]
    graphPaths: list[GraphEvidencePath] = Field(default_factory=list, max_length=4)
    reasoningPaths: list[ReasoningPath] = Field(default_factory=list, max_length=4)

    @model_validator(mode="after")
    def sync_reasoning_paths(self) -> "SynthesisEvidence":
        if self.graphPaths and not self.reasoningPaths:
            self.reasoningPaths = [ReasoningPath.from_graph_path(path) for path in self.graphPaths]
        elif self.reasoningPaths and not self.graphPaths:
            self.graphPaths = [path.to_graph_path() for path in self.reasoningPaths]
        return self


class ModelGroundedClaim(ClosedModel):
    """Closed model output before Runtime assigns the final opaque claim ID."""

    statement: Annotated[str, StringConstraints(min_length=1, max_length=1200)]
    supportStatus: GraphPathStatus
    evidenceIds: list[Annotated[str, StringConstraints(pattern=r"^evidence-[0-9a-f]{32}$")]] = Field(
        min_length=1, max_length=10
    )
    reasoningPathIds: list[Annotated[str, StringConstraints(pattern=r"^path-[0-9a-f]{32}$")]] = Field(
        default_factory=list, max_length=10
    )


class ReasoningStep(ClosedModel):
    """One source-bound reasoning step, not hidden chain-of-thought."""

    stepIndex: int = Field(strict=True, ge=1, le=20)
    description: Annotated[str, StringConstraints(min_length=1, max_length=1200)]
    supportStatus: GraphPathStatus
    evidenceIds: list[Annotated[str, StringConstraints(pattern=r"^evidence-[0-9a-f]{32}$")]] = Field(
        min_length=1, max_length=10
    )
    reasoningPathIds: list[Annotated[str, StringConstraints(pattern=r"^path-[0-9a-f]{32}$")]] = Field(
        default_factory=list, max_length=10
    )


class ModelReasoningStep(ClosedModel):
    """Closed model proposal before Runtime assigns/validates evidence binding."""

    stepIndex: int = Field(strict=True, ge=1, le=20)
    description: Annotated[str, StringConstraints(min_length=1, max_length=1200)]
    supportStatus: GraphPathStatus
    evidenceIds: list[Annotated[str, StringConstraints(pattern=r"^evidence-[0-9a-f]{32}$")]] = Field(
        min_length=1, max_length=10
    )
    reasoningPathIds: list[Annotated[str, StringConstraints(pattern=r"^path-[0-9a-f]{32}$")]] = Field(
        default_factory=list, max_length=10
    )


class EvidenceSynthesisContext(ClosedModel):
    questionSummary: Annotated[str, StringConstraints(min_length=1, max_length=1200)]
    evidence: list[SynthesisEvidence] = Field(min_length=1, max_length=20)


class EvidenceSynthesisResult(ClosedModel):
    claims: list[GroundedClaim] = Field(default_factory=list, max_length=20)
    reasoningSteps: list[ReasoningStep] = Field(default_factory=list, max_length=20)
    conclusion: Annotated[str, StringConstraints(min_length=1, max_length=1600)] | None = None
    mode: Literal["deterministic", "model"]
    fallbackCode: str | None = None
