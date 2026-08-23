from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("MICO_KNOWLEDGE_INTEGRATION", "").lower() != "true",
    reason="real pgvector/Neo4j integration is opt-in",
)


class _StoredVectorEmbedding:
    modelName = "gemini-embedding-2"

    def __init__(self, values: list[float]) -> None:
        self._values = values

    def embed_query(self, _query: str) -> list[float]:
        return self._values

    def embed_document(self, _title: str | None, _text: str) -> list[float]:
        return self._values


def test_real_pgvector_and_neo4j_hybrid_retrieval() -> None:
    import psycopg
    from pgvector.psycopg import register_vector

    from mico_agent_runtime.contracts.evidence import EvidenceQuery
    from mico_agent_runtime.knowledge.database_config import KnowledgeStoreConfiguration
    from mico_agent_runtime.knowledge.database_retriever import DatabaseKnowledgeSearchPort

    configuration = KnowledgeStoreConfiguration.from_environment()
    with psycopg.connect(configuration.vectorDatabaseUrl) as connection:
        register_vector(connection)
        with connection.cursor() as cursor:
            cursor.execute("SELECT embedding FROM knowledge_document LIMIT 1")
            stored = cursor.fetchone()[0]
    values = stored.to_list() if hasattr(stored, "to_list") else list(stored)
    port = DatabaseKnowledgeSearchPort(configuration, _StoredVectorEmbedding(values))
    results = port.search_parallel(
        EvidenceQuery(
            topic="gut microbiome diabetes relationships",
            direction="context",
            retrievalMode="hybrid",
            limit=10,
        ),
        branches=("vector", "graph"),
    )
    assert results
    assert any("vector" in item.retrievalSources for item in results)
    assert any("graph" in item.retrievalSources for item in results)
    assert any(item.graphPaths for item in results)
    assert all(item.evidenceTier == "fulltext" for item in results)
