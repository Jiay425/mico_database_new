from __future__ import annotations

import pytest

from mico_agent_runtime.knowledge.database_config import (
    KnowledgeStoreConfiguration,
    KnowledgeStoreConfigurationError,
)
from mico_agent_runtime.knowledge.database_schema import vector_schema_sql
from mico_agent_runtime.transport.app import _resolve_knowledge_backend


def _env() -> dict[str, str]:
    return {
        "MICO_KNOWLEDGE_VECTOR_ENABLED": "true",
        "MICO_KNOWLEDGE_GRAPH_ENABLED": "true",
        "MICO_KNOWLEDGE_VECTOR_DATABASE_URL": "postgresql://test_user@127.0.0.1:55432/mico_knowledge",
        "MICO_KNOWLEDGE_NEO4J_URI": "bolt://127.0.0.1:57687",
        "MICO_KNOWLEDGE_NEO4J_USER": "neo4j",
        "MICO_KNOWLEDGE_NEO4J_PASSWORD": "test_only_password",
        "MICO_LOCAL_KNOWLEDGE_INDEX_DIR": "references/knowledge/medical/rag",
    }


def test_independent_knowledge_store_configuration_is_closed() -> None:
    configuration = KnowledgeStoreConfiguration.from_environment(_env())
    assert configuration.vectorDimension == 3072
    assert configuration.vectorDatabaseUrl.endswith("/mico_knowledge")


@pytest.mark.parametrize(
    "url,code",
    [
        ("mysql://test_user@127.0.0.1/mico_knowledge", "KNOWLEDGE_VECTOR_URL_NOT_POSTGRES"),
        ("postgresql://test_user@127.0.0.1/patient_data_manager", "KNOWLEDGE_VECTOR_DATABASE_FORBIDDEN"),
        ("postgresql://test_user@127.0.0.1/mico_agent_runtime", "KNOWLEDGE_VECTOR_DATABASE_FORBIDDEN"),
        ("postgresql://test_user@127.0.0.1/mico_knowledge?sslmode=disable", "KNOWLEDGE_VECTOR_URL_QUERY_FORBIDDEN"),
        ("postgresql://test_user@127.0.0.1/", "KNOWLEDGE_VECTOR_URL_INVALID"),
    ],
)
def test_vector_store_rejects_wrong_database_boundaries(url: str, code: str) -> None:
    env = _env()
    env["MICO_KNOWLEDGE_VECTOR_DATABASE_URL"] = url
    with pytest.raises(KnowledgeStoreConfigurationError, match=code):
        KnowledgeStoreConfiguration.from_environment(env)


def test_schema_is_pgvector_and_does_not_reference_business_database() -> None:
    sql = "\n".join(vector_schema_sql(3072)).lower()
    assert "create extension if not exists vector" in sql
    assert "halfvec(3072)" in sql
    assert "using hnsw" in sql
    assert "halfvec_cosine_ops" in sql
    assert "jsonb" in sql
    assert "patient_data_manager" not in sql
    assert "mico_agent_runtime" not in sql


def test_knowledge_schema_contains_document_chunk_and_manifest_assets() -> None:
    sql = "\n".join(vector_schema_sql()).lower()
    assert "knowledge_index_manifest" in sql
    assert "knowledge_document" in sql
    assert "knowledge_chunk" in sql
    assert "on delete cascade" in sql


def test_configured_database_stores_are_not_silently_bypassed_by_local_default() -> None:
    env = _env()
    assert _resolve_knowledge_backend(env) == "database"
    env.pop("MICO_KNOWLEDGE_VECTOR_ENABLED")
    env.pop("MICO_KNOWLEDGE_GRAPH_ENABLED")
    assert _resolve_knowledge_backend(env) == "local"
