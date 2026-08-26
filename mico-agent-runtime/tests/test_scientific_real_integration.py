from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from mico_agent_runtime.knowledge.database_retriever import DatabaseKnowledgeSearchPort
from mico_agent_runtime.knowledge.synthesis import build_graph_rag_synthesis_port
from mico_agent_runtime.ports.java_agent import HttpJavaAgentToolPort
from mico_agent_runtime.ports.research_planner import build_intent_planner
from mico_agent_runtime.ports.schema_catalog import JavaSchemaCatalogPort
from mico_agent_runtime.runtime.scientific_service import ScientificRuntime
from mico_agent_runtime.transport.app import create_app
from mico_agent_runtime.contracts.research import ResearchTask


_REQUIRED = (
    "MICO_JAVA_AGENT_TOOL_BASE_URL",
    "MICO_AGENT_INTERNAL_TOKEN",
    "MICO_KNOWLEDGE_INTEGRATION",
    "MICO_KNOWLEDGE_VECTOR_ENABLED",
    "MICO_KNOWLEDGE_GRAPH_ENABLED",
    "MICO_KNOWLEDGE_VECTOR_DATABASE_URL",
    "MICO_KNOWLEDGE_NEO4J_URI",
    "MICO_KNOWLEDGE_NEO4J_USER",
    "MICO_KNOWLEDGE_NEO4J_PASSWORD",
    "MICO_LOCAL_KNOWLEDGE_INDEX_DIR",
    "MICO_GEMINI_EMBEDDING_ENABLED",
    "MICO_GEMINI_API_KEY",
    "MICO_RESEARCH_PLANNER_BASE_URL",
    "MICO_RESEARCH_PLANNER_MODEL",
    "MICO_RESEARCH_PLANNER_TOKEN",
    "MICO_GRAPH_RAG_GENERATOR_BASE_URL",
    "MICO_GRAPH_RAG_GENERATOR_MODEL",
    "MICO_GRAPH_RAG_GENERATOR_TOKEN",
)


pytestmark = pytest.mark.skipif(
    any(not os.environ.get(name, "").strip() for name in _REQUIRED),
    reason="real Java, DeepSeek, Gemini, pgvector and Neo4j integration is opt-in",
)


def test_real_fastapi_scientific_graph_uses_java_and_hybrid_knowledge() -> None:
    env = dict(os.environ)
    env["MICO_KNOWLEDGE_GRAPH_VERSION"] = "fulltext-provenance-graphrag-v3"
    java_port = HttpJavaAgentToolPort.from_environment(env)
    knowledge_port = DatabaseKnowledgeSearchPort.from_environment(env)
    try:
        planner = build_intent_planner(env)
        runtime = ScientificRuntime(
            java_port,
            planner,
            knowledge_port=knowledge_port,
            schema_catalog_port=JavaSchemaCatalogPort(java_port),
            synthesis_port=build_graph_rag_synthesis_port(env),
        )
        task = ResearchTask(
            runId="run-real-scientific-000000000000000000000000000001",
            taskId="task-real-scientific-000000000000000000000000000001",
            requesterId="principal-real-scientific-000000000000000000000000001",
            traceId="trace-real-scientific-000000000000000000000000000001",
            question="探索微生物与疾病相关研究证据，并检查现有数据中可验证的分组与特征",
            requestedScopes=["mico:query:read", "mico:research:read", "mico:evidence:read"],
            allowedActions=[
                "inspect_cohort",
                "compare_groups",
                "stratified_analysis",
                "adjust_confounders",
                "cross_project_validate",
                "cross_disease_validate",
                "retrieve_evidence",
                "finish",
            ],
            maxActions=4,
        )
        app = create_app(
            env={**env, "MICO_RUNTIME_INTERNAL_TOKEN": "integration-runtime-token"},
            scientific_runtime=runtime,
            knowledge_port=knowledge_port,
        )
        with TestClient(app) as client:
            response = client.post(
                "/internal/runtime/scientific-runs",
                json=task.model_dump(mode="json"),
                headers={"Authorization": "Bearer integration-runtime-token"},
            )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "COMPLETED"
        assert body["report"] is not None
        assert body["report"]["sourceMetadata"]
        assert "sourceSampleId" not in response.text
        assert "internalRecordId" not in response.text
    finally:
        java_port.close()
        knowledge_port.close()
