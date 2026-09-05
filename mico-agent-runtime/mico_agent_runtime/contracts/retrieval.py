from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import Field, StringConstraints, model_validator

from .base import ClosedModel


RetrievalBranch = Literal["vector", "sparse", "graph"]
RetrievalMode = Literal["vector", "graph", "hybrid"]
RetrievalQueryType = Literal["semantic_fact", "relation", "multi_hop", "composite"]
RetrievalScopeType = Literal["global", "document", "document_set"]


class RetrievalScope(ClosedModel):
    """Explicit corpus boundary shared by every retrieval branch.

    ``allowedDocumentIds`` are source-stable PMCIDs in the current medical
    corpus.  They must originate from the real request/UI context, never from
    hidden evaluation provenance.
    """

    scopeType: RetrievalScopeType = "global"
    allowedDocumentIds: list[str] = Field(default_factory=list, max_length=32)

    @model_validator(mode="after")
    def validate_scope(self) -> "RetrievalScope":
        ids = list(dict.fromkeys(value.strip() for value in self.allowedDocumentIds if value.strip()))
        if len(ids) != len(self.allowedDocumentIds):
            raise ValueError("allowedDocumentIds must be non-empty and unique")
        if self.scopeType == "global" and ids:
            raise ValueError("global scope must not contain document IDs")
        if self.scopeType == "document" and len(ids) != 1:
            raise ValueError("document scope requires exactly one document ID")
        if self.scopeType == "document_set" and len(ids) < 2:
            raise ValueError("document_set scope requires at least two document IDs")
        return self


class RetrievalPlan(ClosedModel):
    """Closed execution plan shared by vector and graph retrieval ports."""

    planId: str = Field(pattern=r"^plan-[0-9a-f]{32}$")
    queryType: RetrievalQueryType
    retrievalMode: RetrievalMode
    retrievalBranches: list[RetrievalBranch] = Field(min_length=1, max_length=3)
    topK: int = Field(strict=True, ge=1, le=20)
    maxHops: int = Field(strict=True, ge=0, le=3)
    minConfidence: float = Field(ge=0.65, le=1.0)
    requireEvidencePaths: bool = True
    graphVersion: str = Field(
        default="fulltext-provenance-graphrag-v3",
        pattern=r"^[a-z0-9][a-z0-9-]{1,79}$",
    )
    embeddingVersion: str = Field(default="gemini-embedding-2", pattern=r"^[a-z0-9][a-z0-9-]{1,79}$")
    # Keep the unweighted RRF path as the safe default until reviewed qrels
    # demonstrate that query-type weights/listwise features improve ranking.
    # Experimental ranking stages remain explicitly selectable via
    # ``fusion_version="rrf-v2-listwise-v1"``.
    fusionVersion: str = Field(default="rrf-standard-v1", pattern=r"^[a-z0-9][a-z0-9-]{1,79}$")
    rerankerVersion: str = Field(default="deterministic-listwise-v1", pattern=r"^[a-z0-9][a-z0-9-]{1,79}$")
    retrievalScope: RetrievalScope = Field(default_factory=RetrievalScope)
    querySummary: str = Field(min_length=1, max_length=1024)


class RerankBreakdown(ClosedModel):
    """Auditable score components; no hidden model reasoning is stored."""

    vectorContribution: float = Field(ge=0.0, le=1.0)
    graphContribution: float = Field(ge=0.0, le=1.0)
    reciprocalRankContribution: float = Field(ge=0.0, le=1.0)
    pathSupportContribution: float = Field(ge=0.0, le=1.0)
    sourceDiversityContribution: float = Field(ge=0.0, le=1.0)
    hopPenalty: float = Field(ge=0.0, le=1.0)
    statusPenalty: float = Field(ge=0.0, le=1.0)
    finalScore: float = Field(ge=0.0, le=1.0)
    # BM25/FTS is deliberately exposed separately from dense cosine and graph
    # path support.  A default keeps older persisted evidence records valid.
    sparseContribution: float = Field(default=0.0, ge=0.0, le=1.0)
    # Optional final listwise feature. It defaults to zero for legacy records.
    listwiseContribution: float = Field(default=0.0, ge=0.0, le=1.0)


def build_retrieval_plan(
    *,
    query_summary: str,
    query_type: RetrievalQueryType,
    retrieval_mode: RetrievalMode,
    retrieval_branches: list[RetrievalBranch] | tuple[RetrievalBranch, ...],
    top_k: int = 10,
    graph_version: str = "fulltext-provenance-graphrag-v3",
    fusion_version: str = "rrf-standard-v1",
    retrieval_scope: RetrievalScope | None = None,
) -> RetrievalPlan:
    branches = list(dict.fromkeys(retrieval_branches))
    if retrieval_mode == "vector":
        branches = ["vector"]
    elif retrieval_mode == "graph":
        branches = ["graph"]
    else:
        # Hybrid is always a three-way candidate collection.  Callers may
        # explicitly disable a branch for controlled experiments, but the
        # production default includes dense, sparse and graph retrieval.
        branches = [branch for branch in ("vector", "sparse", "graph") if branch in branches]
        if not branches:
            branches = ["vector", "sparse", "graph"]
    max_hops = 3 if query_type in {"relation", "multi_hop", "composite"} and "graph" in branches else 0
    scope = retrieval_scope or RetrievalScope()
    canonical = json.dumps(
        {
            "queryType": query_type,
            "retrievalMode": retrieval_mode,
            "branches": branches,
            "topK": top_k,
            "maxHops": max_hops,
            "graphVersion": graph_version,
            "embeddingVersion": "gemini-embedding-2",
            "fusionVersion": fusion_version,
            "rerankerVersion": "deterministic-listwise-v1",
            "retrievalScope": scope.model_dump(mode="json"),
            "querySummary": query_summary[:1024],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    plan_id = "plan-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]
    return RetrievalPlan(
        planId=plan_id,
        queryType=query_type,
        retrievalMode=retrieval_mode,
        retrievalBranches=branches,
        topK=top_k,
        maxHops=max_hops,
        minConfidence=0.65,
        requireEvidencePaths="graph" in branches,
        graphVersion=graph_version,
        embeddingVersion="gemini-embedding-2",
        fusionVersion=fusion_version,
        rerankerVersion="deterministic-listwise-v1",
        retrievalScope=scope,
        querySummary=query_summary[:1024],
    )
