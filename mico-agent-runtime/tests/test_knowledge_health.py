from pathlib import Path
import json

from mico_agent_runtime.knowledge.health import probe_knowledge_backend


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "MICO_KNOWLEDGE_RETRIEVAL_BACKEND": "database",
        "MICO_KNOWLEDGE_VECTOR_ENABLED": "true",
        "MICO_KNOWLEDGE_GRAPH_ENABLED": "true",
        "MICO_KNOWLEDGE_VECTOR_DATABASE_URL": "postgresql://test_user@127.0.0.1:55432/mico_knowledge",
        "MICO_KNOWLEDGE_NEO4J_URI": "bolt://127.0.0.1:57687",
        "MICO_KNOWLEDGE_NEO4J_USER": "neo4j",
        "MICO_KNOWLEDGE_NEO4J_PASSWORD": "test-only-password",
        "MICO_LOCAL_KNOWLEDGE_INDEX_DIR": str(tmp_path),
    }


def test_config_only_health_check_never_starts_embedding_or_retrieval(
    tmp_path: Path,
) -> None:
    report = probe_knowledge_backend(_env(tmp_path), probe_connectivity=False)
    assert report["configuration_valid"] is True
    assert report["embedding_calls"] == 0
    assert report["knowledge_retrieval_probe_pass"] == "NOT_RUN_QUOTA_GUARD"
    assert report["vector"] is None
    assert report["graph"] is None


def test_invalid_knowledge_configuration_fails_closed_without_external_calls(
    tmp_path: Path,
) -> None:
    env = _env(tmp_path)
    env["MICO_KNOWLEDGE_VECTOR_ENABLED"] = "false"
    report = probe_knowledge_backend(env, probe_connectivity=True)
    assert report["configuration_valid"] is False
    assert report["knowledge_wiring_pass"] is False
    assert report["knowledge_connectivity_pass"] is False
    assert report["embedding_calls"] == 0
    assert report["knowledge_retrieval_probe_pass"] == "NOT_RUN_QUOTA_GUARD"


def test_embedding_config_matches_index_manifest_without_provider_call(tmp_path: Path) -> None:
    env = _env(tmp_path)
    env.update({
        "MICO_GEMINI_EMBEDDING_ENABLED": "true",
        "MICO_GEMINI_API_KEY": "test-only",
        "MICO_GEMINI_EMBEDDING_MODEL": "gemini-embedding-2",
        "MICO_KNOWLEDGE_VECTOR_DIMENSION": "3072",
        "MICO_KNOWLEDGE_CHUNK_VERSION": "chunk-v2",
        "MICO_KNOWLEDGE_CHUNK_VARIANT": "medium",
        "MICO_KNOWLEDGE_GRAPH_VERSION": "fulltext-provenance-graphrag-v3",
    })
    (tmp_path / "medical_gemini_paper_embedding_meta.json").write_text(json.dumps({
        "model": "gemini-embedding-2", "dimension": 3072, "chunkCount": 10,
    }), encoding="utf-8")
    (tmp_path / "medical_gemini_chunk_embedding_index_v2_medium_meta.json").write_text(json.dumps({
        "model": "gemini-embedding-2", "dimension": 3072, "chunkCount": 10,
        "chunkVersion": "chunk-v2", "variant": "medium",
    }), encoding="utf-8")
    (tmp_path / "medical_knowledge_graph_v3_manifest.json").write_text(json.dumps({
        "graphVersion": "fulltext-provenance-graphrag-v3",
    }), encoding="utf-8")
    report = probe_knowledge_backend(env, probe_connectivity=False)
    contract = report["embedding_contract"]
    assert contract["configuration_valid"] is True
    assert contract["asset_contract_status"] == "PASS"
    assert report["embedding_calls"] == 0


def test_embedding_config_audit_reports_stale_chunk_contract_without_provider_call(tmp_path: Path) -> None:
    env = _env(tmp_path)
    env.update({
        "MICO_GEMINI_EMBEDDING_ENABLED": "true",
        "MICO_GEMINI_API_KEY": "test-only",
        "MICO_GEMINI_EMBEDDING_MODEL": "gemini-embedding-2",
    })
    (tmp_path / "medical_gemini_paper_embedding_meta.json").write_text(json.dumps({
        "model": "gemini-embedding-2", "dimension": 3072, "chunkCount": 10,
    }), encoding="utf-8")
    (tmp_path / "medical_gemini_chunk_embedding_index_v2_medium_meta.json").write_text(json.dumps({
        "model": "gemini-embedding-2", "dimension": 3072, "chunkCount": 5,
        "chunkVersion": "chunk-v2", "variant": "medium",
    }), encoding="utf-8")
    (tmp_path / "medical_knowledge_graph_v3_manifest.json").write_text(json.dumps({
        "graphVersion": "fulltext-provenance-graphrag-v3",
    }), encoding="utf-8")
    report = probe_knowledge_backend(env, probe_connectivity=False)
    contract = report["embedding_contract"]
    assert contract["configuration_valid"] is True
    assert contract["asset_contract_status"] == "MISMATCH"
    assert report["embedding_calls"] == 0
