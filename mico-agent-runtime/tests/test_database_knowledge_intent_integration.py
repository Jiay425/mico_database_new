from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("MICO_KNOWLEDGE_INTENT_INTEGRATION", "").lower() != "true",
    reason="real database-backed LangGraph intent integration is opt-in",
)


class _StoredVectorEmbedding:
    modelName = "gemini-embedding-2"

    def __init__(self, values: list[float]) -> None:
        self._values = values

    def embed_query(self, _query: str) -> list[float]:
        return self._values

    def embed_document(self, _title: str | None, _text: str) -> list[float]:
        return self._values


class _NoJavaPort:
    def execute(self, _call):
        raise AssertionError("literature route must not call Java business tools")


def test_real_database_backend_is_used_by_main_langgraph_intent_route() -> None:
    import psycopg
    from pgvector.psycopg import register_vector

    from mico_agent_runtime.contracts.intent import IntentTaskRequest
    from mico_agent_runtime.knowledge.database_config import KnowledgeStoreConfiguration
    from mico_agent_runtime.knowledge.database_retriever import DatabaseKnowledgeSearchPort
    from mico_agent_runtime.ports.research_planner import DeterministicIntentPlanner
    from mico_agent_runtime.runtime.intent_service import IntentRuntime

    configuration = KnowledgeStoreConfiguration.from_environment()
    with psycopg.connect(configuration.vectorDatabaseUrl) as connection:
        register_vector(connection)
        with connection.cursor() as cursor:
            cursor.execute("SELECT embedding FROM knowledge_document LIMIT 1")
            stored = cursor.fetchone()[0]
    values = stored.to_list() if hasattr(stored, "to_list") else list(stored)
    knowledge = DatabaseKnowledgeSearchPort(configuration, _StoredVectorEmbedding(values))
    request = IntentTaskRequest(
        runId="run-db-knowledge-000000000000000000000000000001",
        taskId="task-db-knowledge-000000000000000000000000000001",
        requesterId="principal-db-knowledge-00000000000000000000000000001",
        requestedScopes=["mico:evidence:read"],
        question="请检索 2 型糖尿病与肠道微生物组的文献证据和机制关系",
        allowedWorkflows=["knowledge_retrieval"],
        createdAt=datetime.now(timezone.utc),
        traceId="trace-db-knowledge-000000000000000000000000000001",
    )
    result = IntentRuntime(
        _NoJavaPort(), DeterministicIntentPlanner(), knowledge_port=knowledge
    ).run(request)
    assert result.status == "COMPLETED"
    assert result.workflow == "knowledge_retrieval"
    assert result.report.retrievalBranches == ["vector", "graph"]
    assert result.report.references
    assert any(item.retrievalRoute == "hybrid" for item in result.report.references)
    assert any(item.graphPaths for item in result.report.references)
