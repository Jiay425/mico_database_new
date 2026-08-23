from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import Field, StringConstraints

from .base import ClosedModel


RetrievalBranch = Literal["vector", "graph"]
RetrievalMode = Literal["vector", "graph", "hybrid"]
RetrievalQueryType = Literal["semantic_fact", "relation", "multi_hop", "composite"]


class RetrievalPlan(ClosedModel):
    """Closed execution plan shared by vector and graph retrieval ports."""

    planId: str = Field(pattern=r"^plan-[0-9a-f]{32}$")
    queryType: RetrievalQueryType
    retrievalMode: RetrievalMode
    retrievalBranches: list[RetrievalBranch] = Field(min_length=1, max_length=2)
    topK: int = Field(strict=True, ge=1, le=20)
    maxHops: int = Field(strict=True, ge=0, le=3)
    minConfidence: float = Field(ge=0.65, le=1.0)
    requireEvidencePaths: bool = True
    graphVersion: str = Field(
        default="fulltext-provenance-graphrag-v3",
        pattern=r"^[a-z0-9][a-z0-9-]{1,79}$",
    )
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


def build_retrieval_plan(
    *,
    query_summary: str,
    query_type: RetrievalQueryType,
    retrieval_mode: RetrievalMode,
    retrieval_branches: list[RetrievalBranch] | tuple[RetrievalBranch, ...],
    top_k: int = 10,
    graph_version: str = "fulltext-provenance-graphrag-v3",
) -> RetrievalPlan:
    branches = list(dict.fromkeys(retrieval_branches))
    if retrieval_mode == "vector":
        branches = ["vector"]
    elif retrieval_mode == "graph":
        branches = ["graph"]
    else:
        branches = [branch for branch in ("vector", "graph") if branch in branches] or ["vector", "graph"]
    max_hops = 3 if query_type in {"relation", "multi_hop", "composite"} and "graph" in branches else 0
    canonical = json.dumps(
        {
            "queryType": query_type,
            "retrievalMode": retrieval_mode,
            "branches": branches,
            "topK": top_k,
            "maxHops": max_hops,
            "graphVersion": graph_version,
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
        querySummary=query_summary[:1024],
    )

