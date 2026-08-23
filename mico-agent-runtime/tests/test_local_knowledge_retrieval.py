from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from mico_agent_runtime.contracts.evidence import EvidenceQuery
from mico_agent_runtime.contracts.evidence import LiteratureEvidenceItem
from mico_agent_runtime.contracts.intent import IntentTaskRequest
from mico_agent_runtime.knowledge.local_retriever import (
    LocalKnowledgeIndexConfiguration,
    LocalKnowledgeIndexConfigurationError,
    LocalKnowledgeSearchPort,
)
from mico_agent_runtime.transport.app import create_app
from mico_agent_runtime.runtime.intent_service import IntentRuntime
from mico_agent_runtime.ports.research_planner import DeterministicIntentPlanner


def write_index(tmp_path):
    (tmp_path / "medical_knowledge_manifest.json").write_text(json.dumps({
        "indexVersion": "fulltext-knowledge-index-v1",
        "corpusScope": "fulltext_only",
        "evidenceTier": "fulltext",
        "paperCount": 2,
    }), encoding="utf-8")
    (tmp_path / "medical_vector_meta.json").write_text(json.dumps({
        "indexVersion": "fulltext-tfidf-cosine-v1",
        "idf": {"type": 1.0, "diabetes": 1.0, "gut": 1.0, "microbiome": 1.0},
    }), encoding="utf-8")
    rows = [
        {
            "evidenceTier": "fulltext", "chunkId": "PMC1-A001", "pmcid": "PMC1", "pmid": "1",
            "title": "Type 2 diabetes gut microbiome", "year": "2024", "journal": "Test Journal",
            "vector": [["type", 0.7], ["diabetes", 0.7], ["gut", 0.1], ["microbiome", 0.1]],
        },
        {
            "evidenceTier": "fulltext", "chunkId": "PMC2-A001", "pmcid": "PMC2", "pmid": "2",
            "title": "Unrelated paper", "year": "2023", "journal": "Test Journal",
            "vector": [["gut", 0.9], ["microbiome", 0.4]],
        },
    ]
    with (tmp_path / "medical_vector_index.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    graph = [
        {"recordType": "node", "nodeId": "term:type_2_diabetes", "label": "type_2_diabetes",
         "aliases": ["type 2 diabetes", "t2d"]},
        {"recordType": "node", "nodeId": "topic:t2d", "label": "t2d"},
        {"recordType": "edge", "edgeId": "edge-1", "source": "chunk:PMC1-A001",
         "relation": "MENTIONS", "target": "term:type_2_diabetes", "evidenceChunkId": "PMC1-A001"},
        {"recordType": "edge", "edgeId": "edge-2", "source": "paper:PMC1",
         "relation": "HAS_TOPIC", "target": "topic:t2d", "evidenceChunkId": "PMC1-A001"},
    ]
    with (tmp_path / "medical_knowledge_graph.jsonl").open("w", encoding="utf-8") as handle:
        for row in graph:
            handle.write(json.dumps(row) + "\n")


def test_local_fulltext_hybrid_search_preserves_provenance(tmp_path) -> None:
    write_index(tmp_path)
    port = LocalKnowledgeSearchPort(LocalKnowledgeIndexConfiguration(tmp_path))
    results = port.search(EvidenceQuery(
        topic="type 2 diabetes gut microbiome",
        direction="supporting",
        retrievalMode="hybrid",
        limit=5,
    ))
    assert results
    assert results[0].evidenceTier == "fulltext"
    assert results[0].source == "internal_knowledge"
    assert results[0].retrievalRoute == "hybrid"
    assert results[0].sourceChunkId == "PMC1-A001"
    assert results[0].externalId == "PMCID:PMC1#PMC1-A001"
    assert results[0].sourceExcerpt is None


def test_graph_retrieval_returns_bounded_source_bound_multi_hop_paths(tmp_path) -> None:
    write_index(tmp_path)
    with (tmp_path / "medical_knowledge_graph.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "recordType": "node", "nodeId": "candidate_taxon:abc",
            "nodeType": "candidate_taxon", "label": "Bacteroides testus",
        }) + "\n")
        handle.write(json.dumps({
            "recordType": "edge", "edgeId": "edge-3",
            "source": "term:type_2_diabetes", "relation": "ASSOCIATED_WITH_CANDIDATE",
            "target": "candidate_taxon:abc", "evidenceChunkId": "PMC1-A001",
        }) + "\n")
        handle.write(json.dumps({
            "recordType": "edge", "edgeId": "edge-4",
            "source": "chunk:PMC1-A001", "relation": "MENTIONS_CANDIDATE",
            "target": "candidate_taxon:abc", "evidenceChunkId": "PMC1-A001",
        }) + "\n")
    port = LocalKnowledgeSearchPort(LocalKnowledgeIndexConfiguration(tmp_path))
    results = port.search(EvidenceQuery(
        topic="type 2 diabetes", direction="context", retrievalMode="graph", limit=5
    ))
    assert results and results[0].graphPaths
    assert any(len(path.hops) == 2 for path in results[0].graphPaths)
    assert all(
        step.evidenceChunkId
        for item in results
        for path in item.graphPaths
        for step in path.hops
    )
    assert results[0].graphScore > 0
    assert results[0].rerankScore == results[0].retrievalScore


def test_local_index_configuration_is_fail_closed(tmp_path) -> None:
    with pytest.raises(LocalKnowledgeIndexConfigurationError):
        LocalKnowledgeIndexConfiguration.from_environment({})
    with pytest.raises(LocalKnowledgeIndexConfigurationError):
        LocalKnowledgeIndexConfiguration.from_environment({
            "MICO_LOCAL_KNOWLEDGE_ENABLED": "true",
        })


def test_evidence_endpoint_uses_explicit_local_fulltext_index(tmp_path) -> None:
    write_index(tmp_path)
    app = create_app(env={
        "MICO_RUNTIME_INTERNAL_TOKEN": "runtime-test-token",
        "MICO_LOCAL_KNOWLEDGE_ENABLED": "true",
        "MICO_LOCAL_KNOWLEDGE_INDEX_DIR": str(tmp_path),
    })
    with TestClient(app) as client:
        response = client.post(
            "/internal/runtime/evidence-runs",
            headers={"Authorization": "Bearer runtime-test-token"},
            json={
                "runId": "run-knowledge-1",
                "taskId": "task-knowledge-1",
                "requesterId": "principal-knowledge-1",
                "traceId": "trace-knowledge-1",
                "topic": "type 2 diabetes gut microbiome",
                "requestedDirections": ["supporting"],
                "retrievalMode": "hybrid",
                "limit": 2,
                "createdAt": datetime.now(timezone.utc).isoformat(),
            },
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["report"]["references"][0]["evidenceTier"] == "fulltext"
    assert payload["report"]["references"][0]["retrievalRoute"] == "hybrid"
    assert "fulltext_corpus_v1" in payload["report"]["limitations"]


class FakeKnowledgePort:
    def __init__(self) -> None:
        self.calls = 0

    def search(self, query: EvidenceQuery) -> list[LiteratureEvidenceItem]:
        self.calls += 1
        return [LiteratureEvidenceItem(
            evidenceId="evidence-11111111111111111111111111111111",
            source="internal_knowledge",
            externalId="PMCID:PMC1#PMC1-A001",
            title="Full-text evidence",
            publicationYear=2024,
            direction=query.direction,
            summary="Full-text evidence chunk selected by hybrid.",
            evidenceTier="fulltext",
            retrievalRoute="hybrid",
            sourceChunkId="PMC1-A001",
            retrievalScore=0.9,
        )]


def test_natural_language_knowledge_route_does_not_call_java() -> None:
    class NoJavaPort:
        def execute(self, _call):
            raise AssertionError("knowledge route must not call Java data tool")

    knowledge = FakeKnowledgePort()
    runtime = IntentRuntime(NoJavaPort(), DeterministicIntentPlanner(), knowledge_port=knowledge)
    request = IntentTaskRequest(
        runId="run-knowledge-1",
        taskId="task-knowledge-1",
        requesterId="principal-knowledge-1",
        requestedScopes=["mico:evidence:read"],
        question="请查找 2 型糖尿病与肠道微生物组的文献证据",
        allowedWorkflows=["dynamic_read_query", "knowledge_retrieval"],
        createdAt=datetime.now(timezone.utc),
        traceId="trace-knowledge-1",
    )
    result = runtime.run(request)
    assert result.status == "COMPLETED"
    assert result.workflow == "knowledge_retrieval"
    assert result.report.references[0].evidenceTier == "fulltext"
    assert knowledge.calls == 1
