from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StringConstraints

from .base import ClosedModel, NonEmptyText
from .graph_rag import GraphEvidencePath, GroundedClaim, ReasoningPath, ReasoningStep
from .retrieval import RerankBreakdown, RetrievalPlan


EvidenceSource = Literal["pubmed", "crossref", "arxiv", "semantic_scholar", "internal_knowledge"]
EvidenceId = Annotated[str, StringConstraints(pattern=r"^evidence-[0-9a-f]{32}$")]


class EvidenceReference(ClosedModel):
    evidenceId: EvidenceId
    source: EvidenceSource
    externalId: NonEmptyText
    title: Annotated[str, StringConstraints(min_length=1, max_length=512)]
    journal: Annotated[str, StringConstraints(min_length=1, max_length=256)] | None = None
    publicationYear: int = Field(strict=True, ge=1900, le=2100)
    direction: Literal["supporting", "contrary", "context"]
    summary: Annotated[str, StringConstraints(min_length=1, max_length=2000)]
    evidenceTier: Literal["fulltext", "abstract", "metadata"] = "metadata"
    retrievalRoute: Literal["vector", "graph", "hybrid"] | None = None
    retrievalModel: Literal["gemini-embedding-2", "fulltext-tfidf-cosine-v1"] | None = None
    sourceChunkId: Annotated[str, StringConstraints(min_length=1, max_length=128)] | None = None
    retrievalScore: float = Field(default=0.0, ge=0.0)
    sourceExcerpt: Annotated[str, StringConstraints(min_length=1, max_length=1200)] | None = None
    vectorScore: float = Field(default=0.0, ge=0.0)
    graphScore: float = Field(default=0.0, ge=0.0)
    rerankScore: float = Field(default=0.0, ge=0.0)
    rerankBreakdown: RerankBreakdown | None = None
    graphPaths: list[GraphEvidencePath] = Field(default_factory=list, max_length=4)
    reasoningPaths: list[ReasoningPath] = Field(default_factory=list, max_length=4)
    retrievalSources: list[Literal["vector", "graph"]] = Field(default_factory=list, max_length=2)


class EvidenceReviewReport(ClosedModel):
    """Generic literature metadata report; it is not tied to a fixed disease."""

    status: Literal["COMPLETED", "INSUFFICIENT_EVIDENCE", "REJECTED", "FAILED"]
    topic: Annotated[str, StringConstraints(min_length=1, max_length=1024)]
    queryType: Literal["semantic_fact", "relation", "multi_hop", "composite"] = "semantic_fact"
    routeConfidence: float = Field(default=0.0, ge=0.0, le=1.0)
    retrievalBranches: list[Literal["vector", "graph"]] = Field(default_factory=list, max_length=2)
    retrievalPlan: RetrievalPlan | None = None
    queriesExecuted: int = Field(strict=True, ge=0, le=64)
    references: list[EvidenceReference] = Field(max_length=50)
    limitations: list[Literal[
        "literature_is_external_evidence",
        "metadata_only_until_full_text_review",
        "fulltext_corpus_v1",
        "retrieval_route_is_provenance_labeled",
        "candidate_taxon_edges_require_review",
        "graph_paths_are_source_bound",
        "multi_hop_claims_require_review",
        "external_evidence_does_not_override_internal_data",
        "evidence_is_not_causal_or_clinical_advice",
    ]] = Field(min_length=1)
    nonDiagnostic: Literal["not_clinical_diagnostic_or_treatment_advice"]
    generationMode: Literal["deterministic_grounded", "model_grounded"] = "deterministic_grounded"
    groundedClaims: list[GroundedClaim] = Field(default_factory=list, max_length=20)
    reasoningSteps: list[ReasoningStep] = Field(default_factory=list, max_length=20)
    conclusion: Annotated[str, StringConstraints(min_length=1, max_length=1600)] | None = None
