from __future__ import annotations

from datetime import datetime, timezone

from mico_agent_runtime.contracts.evidence import LiteratureEvidenceItem
from mico_agent_runtime.contracts.graph_rag import GraphEvidencePath, GraphPathStep
from mico_agent_runtime.contracts.research import Observation
from mico_agent_runtime.contracts.unified_evidence import merge_unified_evidence


NOW = datetime(2026, 8, 23, 12, 0, tzinfo=timezone.utc)


def _path(status: str = "supported") -> GraphEvidencePath:
    return GraphEvidencePath(
        pathId="path-33333333333333333333333333333333",
        status=status,
        hops=[GraphPathStep(
            fromEntity="disease concept",
            relation="ASSOCIATED_WITH",
            toEntity="microbe concept",
            evidenceChunkId="chunk-001",
            supportStatus=status,
        )],
        sourceDocumentIds=["document-001"],
        confidence=0.9 if status == "supported" else 0.2,
    )


def _item(route: str, score: float, path: GraphEvidencePath | None = None) -> LiteratureEvidenceItem:
    return LiteratureEvidenceItem(
        evidenceId="evidence-33333333333333333333333333333333",
        source="internal_knowledge",
        externalId="document-001#chunk-001",
        title="Bounded literature result",
        publicationYear=2025,
        direction="context",
        summary="A source-bound literature summary.",
        evidenceTier="fulltext",
        retrievalRoute=route,
        retrievalModel="gemini-embedding-2" if route == "vector" else None,
        sourceChunkId="chunk-001",
        retrievalScore=score,
        vectorScore=score if route == "vector" else 0.0,
        graphScore=score if route == "graph" else 0.0,
        graphPaths=[path] if path else [],
        retrievalSources=[route],
    )


def _java_observation(status: str = "VALIDATED") -> Observation:
    return Observation(
        observationId="observation-33333333333333333333333333333333",
        actionId="action-33333333333333333333333333333333",
        actionName="execute_read_query",
        status=status,
        source="java_controlled_read",
        queryHash="sha256:" + "3" * 64,
        rowCount=4,
        generatedAt=NOW,
        schemaVersion="p1b2-java-read-only-tool-executor-v1",
        dataSnapshotId="transient-550e8400-e29b-41d4-a716-446655440000",
        snapshotPersistence="transient",
    )


def test_three_routes_merge_with_source_bindings_and_transient_java_evidence() -> None:
    result = merge_unified_evidence(
        vector_results=[_item("vector", 0.82)],
        graph_results=[_item("graph", 0.74, _path())],
        java_observations=[_java_observation()],
        limit=3,
    )

    literature = next(item for item in result if item.kind == "literature_chunk")
    java = next(item for item in result if item.kind == "java_observation")
    assert literature.sourceRoutes == ["vector", "graph"]
    assert {binding.origin for binding in literature.sourceBindings} == {"pgvector", "neo4j"}
    assert literature.reasoningPaths[0].hopCount == 1
    assert java.sourceRoutes == ["java"]
    assert java.sourceBindings[0].snapshotPersistence == "transient"
    assert java.sourceBindings[0].dataSnapshotId.startswith("transient-")
    assert java.sourceBindings[0].queryHash == "sha256:" + "3" * 64


def test_unified_rerank_preserves_route_coverage_under_limit() -> None:
    result = merge_unified_evidence(
        vector_results=[_item("vector", 0.82)],
        graph_results=[_item("graph", 0.74, _path())],
        java_observations=[_java_observation()],
        limit=2,
    )
    routes = {route for item in result for route in item.sourceRoutes}
    assert {"vector", "graph", "java"}.issubset(routes)
    assert len(result) == 2


def test_conflicted_path_is_penalized_without_being_deleted() -> None:
    supported = merge_unified_evidence(
        vector_results=[],
        graph_results=[_item("graph", 0.8, _path("supported"))],
        java_observations=[],
        limit=5,
    )[0]
    conflicted = merge_unified_evidence(
        vector_results=[],
        graph_results=[_item("graph", 0.8, _path("conflicted"))],
        java_observations=[],
        limit=5,
    )[0]
    assert conflicted.supportStatus == "conflicted"
    assert conflicted.rerankBreakdown.conflictPenalty > supported.rerankBreakdown.conflictPenalty
    assert conflicted.rerankScore < supported.rerankScore


def test_vector_candidate_does_not_claim_unbound_graph_reasoning_path() -> None:
    result = merge_unified_evidence(
        vector_results=[_item("vector", 0.82, _path("conflicted"))],
        graph_results=[],
        java_observations=[],
        limit=5,
    )
    assert result[0].sourceRoutes == ["vector"]
    assert result[0].reasoningPaths == []
