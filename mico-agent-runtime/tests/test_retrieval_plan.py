from __future__ import annotations

import pytest
from pydantic import ValidationError

from datetime import datetime, timezone

from mico_agent_runtime.contracts.evidence import EvidenceQuery, EvidenceTaskRequest, LiteratureEvidenceItem
from mico_agent_runtime.contracts.graph_rag import GraphEvidencePath, GraphPathStep, ReasoningPath
from mico_agent_runtime.contracts.retrieval import RetrievalPlan, RetrievalScope, build_retrieval_plan
from mico_agent_runtime.knowledge.database_retriever import (
    DatabaseKnowledgeSearchPort,
    _normalize_graph_query,
    _path_relevance,
    _scope_document_ids,
)
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


def test_standard_rrf_is_the_default_until_qrels_calibrate_experimental_fusion() -> None:
    plan = build_retrieval_plan(
        query_summary="gut microbiome and diabetes",
        query_type="semantic_fact",
        retrieval_mode="hybrid",
        retrieval_branches=["vector", "sparse", "graph"],
    )
    assert plan.fusionVersion == "rrf-standard-v1"


def test_document_scope_is_explicit_and_changes_the_auditable_plan() -> None:
    scope = RetrievalScope(scopeType="document", allowedDocumentIds=["PMC7915767"])
    plan = build_retrieval_plan(
        query_summary="What did this study report about the microbiome?",
        query_type="semantic_fact",
        retrieval_mode="vector",
        retrieval_branches=["vector"],
        retrieval_scope=scope,
    )
    global_plan = build_retrieval_plan(
        query_summary="What did this study report about the microbiome?",
        query_type="semantic_fact",
        retrieval_mode="vector",
        retrieval_branches=["vector"],
    )
    assert plan.retrievalScope == scope
    assert plan.planId != global_plan.planId
    assert _scope_document_ids(scope) == ["PMC7915767"]
    assert _scope_document_ids(RetrievalScope()) is None


def test_retrieval_scope_rejects_ambiguous_document_boundaries() -> None:
    with pytest.raises(ValidationError):
        RetrievalScope(scopeType="document", allowedDocumentIds=[])
    with pytest.raises(ValidationError):
        RetrievalScope(scopeType="global", allowedDocumentIds=["PMC7915767"])
    with pytest.raises(ValidationError):
        RetrievalScope(scopeType="document_set", allowedDocumentIds=["PMC7915767"])


def test_evidence_task_carries_document_scope_into_its_queries() -> None:
    task = EvidenceTaskRequest(
        runId="run-12345678",
        taskId="task-12345678",
        requesterId="user-12345678",
        traceId="trace-12345678",
        topic="What did the supplied study report?",
        requestedDirections=["context"],
        retrievalScope=RetrievalScope(scopeType="document", allowedDocumentIds=["PMC7915767"]),
        createdAt=datetime.now(timezone.utc),
    )
    assert task.retrievalScope.scopeType == "document"
    assert task.retrievalScope.allowedDocumentIds == ["PMC7915767"]


def test_database_retriever_refuses_a_plan_from_another_document_scope() -> None:
    query = EvidenceQuery(
        topic="What did this study report?",
        direction="context",
        retrievalMode="vector",
        retrievalScope=RetrievalScope(scopeType="document", allowedDocumentIds=["PMC7915767"]),
        limit=5,
    )
    mismatched_plan = build_retrieval_plan(
        query_summary=query.topic,
        query_type="semantic_fact",
        retrieval_mode="vector",
        retrieval_branches=["vector"],
    )
    # Scope validation runs before configuration, database, or embedding use.
    port = object.__new__(DatabaseKnowledgeSearchPort)
    with pytest.raises(ValueError, match="scope must match"):
        port.search_parallel(query, branches=("vector",), plan=mismatched_plan)


def test_document_scoped_request_routes_to_dense_by_default() -> None:
    query = EvidenceQuery(
        topic="What did this supplied study report about LPS?",
        direction="context",
        retrievalMode="hybrid",
        retrievalScope=RetrievalScope(scopeType="document", allowedDocumentIds=["PMC7915767"]),
        limit=5,
    )
    port = object.__new__(DatabaseKnowledgeSearchPort)
    seen: list[tuple[str, ...]] = []
    port.search_parallel = lambda _query, branches, plan=None: (seen.append(branches) or [])  # type: ignore[method-assign]

    assert port.search(query) == []
    assert seen == [("vector",)]


def test_graph_query_normalization_expands_surface_forms_without_changing_graph_data() -> None:
    normalized = _normalize_graph_query("Does type II diabetes mellitus involve TLR-4?")
    assert "type 2 diabetes" in normalized
    assert "toll-like receptor 4" in normalized


def test_query_specific_path_score_penalizes_a_frequent_intermediate_hub() -> None:
    direct_nodes = [
        {"label": "lipopolysaccharide", "nodeType": "metabolite"},
        {"label": "toll-like receptor 4", "nodeType": "concept"},
    ]
    direct_relationships = [{"relation": "ACTIVATES", "confidence": 0.92}]
    hub_nodes = [
        direct_nodes[0],
        {"label": "inflammation", "nodeType": "hostprocess"},
        direct_nodes[1],
    ]
    hub_relationships = [
        {"relation": "PROMOTES", "confidence": 0.92},
        {"relation": "MEDIATES", "confidence": 0.90},
    ]
    topic = "How does lipopolysaccharide activate toll-like receptor 4?"
    direct = _path_relevance(direct_nodes, direct_relationships, _normalize_graph_query(topic).split(), topic, {}, 100)
    through_hub = _path_relevance(
        hub_nodes, hub_relationships, _normalize_graph_query(topic).split(), topic,
        {"inflammation": 100}, 100,
    )
    assert direct > through_hub




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


def test_hybrid_does_not_displace_strong_vector_hit_for_branch_coverage() -> None:
    query = EvidenceQuery(
        topic="type 2 diabetes microbiome",
        direction="context",
        retrievalMode="hybrid",
        limit=1,
    )
    vector = _item("vector", 0.92)
    graph = _item("graph", 0.99, _path()).model_copy(update={
        "evidenceId": "evidence-44444444444444444444444444444444",
        "externalId": "PMCID:PMC2#PMC2-A001",
        "sourceChunkId": "PMC2-A001",
    })

    result = merge_and_rerank(query, [vector], [graph])

    assert len(result) == 1
    assert result[0].sourceChunkId == "PMC1-A001"
    assert result[0].retrievalSources == ["vector"]


def test_three_way_rrf_keeps_sparse_provenance_and_auditable_features() -> None:
    query = EvidenceQuery(
        topic="type 2 diabetes microbiome relationship",
        direction="context",
        retrievalMode="hybrid",
        limit=3,
    )
    dense = _item("vector", 0.8)
    sparse = _item("sparse", 0.7).model_copy(update={
        "evidenceId": "evidence-55555555555555555555555555555555",
        "externalId": "PMCID:PMC2#PMC2-A001",
        "sourceChunkId": "PMC2-A001",
        "sparseScore": 0.7,
    })
    graph = _item("graph", 0.7, _path()).model_copy(update={
        "evidenceId": "evidence-66666666666666666666666666666666",
        "externalId": "PMCID:PMC3#PMC3-A001",
        "sourceChunkId": "PMC3-A001",
    })
    plan = build_retrieval_plan(
        query_summary=query.topic,
        query_type="relation",
        retrieval_mode="hybrid",
        retrieval_branches=["vector", "sparse", "graph"],
    )
    result = merge_and_rerank(query, [dense], [graph], plan, sparse_results=[sparse])
    assert result
    assert {route for item in result for route in item.retrievalSources} == {"vector", "sparse", "graph"}
    assert result[0].rerankBreakdown is not None
    assert result[0].rerankBreakdown.sparseContribution >= 0.0
    assert result[0].rerankBreakdown.listwiseContribution >= 0.0


def test_standard_rrf_is_a_clean_fusion_baseline_without_listwise_stage() -> None:
    query = EvidenceQuery(
        topic="type 2 diabetes microbiome relationship",
        direction="context",
        retrievalMode="hybrid",
        limit=3,
    )
    plan = build_retrieval_plan(
        query_summary=query.topic,
        query_type="relation",
        retrieval_mode="hybrid",
        retrieval_branches=["vector", "sparse", "graph"],
        fusion_version="rrf-standard-v1",
    )
    result = merge_and_rerank(
        query,
        [_item("vector", 0.8)],
        [_item("graph", 0.7, _path())],
        plan,
        sparse_results=[_item("sparse", 0.6)],
    )
    assert result
    assert all(item.rerankBreakdown is not None for item in result)
    assert all(item.rerankBreakdown.listwiseContribution == 0.0 for item in result)
