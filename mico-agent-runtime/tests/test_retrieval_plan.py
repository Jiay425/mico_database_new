from __future__ import annotations

import pytest
from pydantic import ValidationError

from mico_agent_runtime.contracts.evidence import EvidenceQuery, LiteratureEvidenceItem
from mico_agent_runtime.contracts.graph_rag import GraphEvidencePath, GraphPathStep, ReasoningPath
from mico_agent_runtime.contracts.retrieval import RetrievalPlan, build_retrieval_plan
from mico_agent_runtime.knowledge.reranking import merge_and_rerank


def _path(status: str = "supported") -> GraphEvidencePath:
    return GraphEvidencePath(
        pathId="path-22222222222222222222222222222222",
        status=status,
        hops=[
            GraphPathStep(
                fromEntity="type 2 diabetes",
                relation="ASSOCIATED_WITH",
                toEntity="Akkermansia muciniphila",
                evidenceChunkId="PMC1-A001",
                supportStatus=status,
            )
        ],
        sourceDocumentIds=["PMCID:PMC1"],
        confidence=0.9,
    )


def _item(route: str, score: float, path: GraphEvidencePath | None = None) -> LiteratureEvidenceItem:
    paths = [path] if path else []
    return LiteratureEvidenceItem(
        evidenceId="evidence-22222222222222222222222222222222",
        source="internal_knowledge",
        externalId="PMCID:PMC1#PMC1-A001",
        title="Full-text evidence",
        publicationYear=2025,
        direction="context",
        summary="Bounded source evidence.",
        evidenceTier="fulltext",
        retrievalRoute=route,
        retrievalModel="fulltext-tfidf-cosine-v1",
        sourceChunkId="PMC1-A001",
        retrievalScore=score,
        sourceExcerpt="A bounded excerpt.",
        vectorScore=score if route == "vector" else 0.0,
        graphScore=score if route == "graph" else 0.0,
        rerankScore=score,
        graphPaths=paths,
        retrievalSources=[route],
    )


def test_retrieval_plan_is_closed_and_sets_multi_hop_budget() -> None:
    plan = build_retrieval_plan(
        query_summary="diabetes microbiome mechanism",
        query_type="composite",
        retrieval_mode="hybrid",
        retrieval_branches=["vector", "graph"],
        top_k=10,
    )
    assert plan.planId.startswith("plan-")
    assert plan.retrievalBranches == ["vector", "graph"]
    assert plan.maxHops == 3
    assert plan.requireEvidencePaths is True
    with pytest.raises(ValidationError):
        RetrievalPlan.model_validate({**plan.model_dump(), "topK": 21})


def test_reasoning_path_requires_exact_hop_and_evidence_shape() -> None:
    path = ReasoningPath.from_graph_path(_path())
    assert path.hopCount == 1
    assert path.evidenceChunkIds == ["PMC1-A001"]
    with pytest.raises(ValidationError):
        ReasoningPath(
            pathId=path.pathId,
            status=path.status,
            hops=path.hops,
            sourceDocumentIds=path.sourceDocumentIds,
            confidence=path.confidence,
            hopCount=2,
            evidenceChunkIds=path.evidenceChunkIds,
            pathScore=path.pathScore,
        )


def test_reranker_merges_branches_and_exposes_score_breakdown() -> None:
    query = EvidenceQuery(
        topic="type 2 diabetes microbiome",
        direction="context",
        retrievalMode="hybrid",
        limit=5,
    )
    result = merge_and_rerank(
        query,
        [_item("vector", 0.8)],
        [_item("graph", 0.7, _path())],
    )
    assert len(result) == 1
    item = result[0]
    assert item.retrievalRoute == "hybrid"
    assert item.retrievalSources == ["graph", "vector"]
    assert item.reasoningPaths[0].hopCount == 1
    assert item.rerankBreakdown is not None
    assert item.rerankBreakdown.finalScore == item.rerankScore
    assert item.rerankScore == item.retrievalScore


def test_conflicted_path_is_penalized_and_retains_status() -> None:
    query = EvidenceQuery(
        topic="type 2 diabetes microbiome",
        direction="context",
        retrievalMode="graph",
        limit=5,
    )
    supported = merge_and_rerank(query, [], [_item("graph", 0.8, _path("supported"))])[0]
    conflicted = merge_and_rerank(query, [], [_item("graph", 0.8, _path("conflicted"))])[0]
    assert conflicted.reasoningPaths[0].status == "conflicted"
    assert conflicted.rerankBreakdown is not None
    assert conflicted.rerankBreakdown.statusPenalty > supported.rerankBreakdown.statusPenalty
    assert conflicted.rerankScore < supported.rerankScore
