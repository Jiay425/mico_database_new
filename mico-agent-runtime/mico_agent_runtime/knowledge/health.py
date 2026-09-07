"""Read-only health checks for the independent knowledge backend.

The preflight deliberately stops at wiring, TCP/database connectivity, and
schema/index visibility.  It never constructs an embedding request and never
executes a retrieval query, so it is safe to run while Gemini quota is being
conserved.
"""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .database_config import KnowledgeStoreConfiguration
from .embeddings import GeminiEmbeddingConfiguration


def _endpoint(value: str) -> dict[str, object]:
    parsed = urlparse(value)
    return {
        "scheme": parsed.scheme,
        "host": parsed.hostname,
        "port": parsed.port,
        "database": parsed.path.lstrip("/") or None,
    }


def _error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {str(exc)[:512]}"


def _probe_vector_store(configuration: KnowledgeStoreConfiguration) -> dict[str, Any]:
    expected_tables = (
        "knowledge_index_manifest",
        "knowledge_document",
        "knowledge_chunk",
    )
    result: dict[str, Any] = {
        "configured_endpoint": _endpoint(configuration.vectorDatabaseUrl),
        "connected": False,
        "schema_tables": {name: False for name in expected_tables},
        "vector_extension": False,
        "document_count": None,
        "chunk_count": None,
        "manifest_count": None,
        "error": None,
    }
    if importlib.util.find_spec("psycopg") is None:
        result["error"] = "DEPENDENCY_MISSING:psycopg"
        return result
    try:
        import psycopg

        with psycopg.connect(configuration.vectorDatabaseUrl, connect_timeout=3) as connection:
            result["connected"] = True
            with connection.cursor() as cursor:
                for table in expected_tables:
                    cursor.execute("SELECT to_regclass(%s)", (table,))
                    result["schema_tables"][table] = cursor.fetchone()[0] is not None
                cursor.execute(
                    "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')"
                )
                result["vector_extension"] = bool(cursor.fetchone()[0])
                for table, key in (
                    ("knowledge_document", "document_count"),
                    ("knowledge_chunk", "chunk_count"),
                    ("knowledge_index_manifest", "manifest_count"),
                ):
                    cursor.execute(f"SELECT count(*) FROM {table}")
                    result[key] = int(cursor.fetchone()[0])
    except Exception as exc:  # pragma: no cover - exercised with integration env
        result["error"] = _error(exc)
    result["schema_ready"] = bool(
        all(result["schema_tables"].values()) and result["vector_extension"]
    )
    return result


def _probe_graph_store(configuration: KnowledgeStoreConfiguration) -> dict[str, Any]:
    result: dict[str, Any] = {
        "configured_endpoint": _endpoint(configuration.neo4jUri),
        "connected": False,
        "query_ok": False,
        "error": None,
    }
    if importlib.util.find_spec("neo4j") is None:
        result["error"] = "DEPENDENCY_MISSING:neo4j"
        return result
    driver = None
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            configuration.neo4jUri,
            auth=(configuration.neo4jUser, configuration.neo4jPassword),
        )
        driver.verify_connectivity()
        result["connected"] = True
        with driver.session() as session:
            record = session.run("RETURN 1 AS ok").single()
            result["query_ok"] = bool(record and record["ok"] == 1)
    except Exception as exc:  # pragma: no cover - exercised with integration env
        result["error"] = _error(exc)
    finally:
        if driver is not None:
            try:
                driver.close()
            except Exception:
                pass
    return result


def _index_assets(configuration: KnowledgeStoreConfiguration) -> dict[str, Any]:
    directory = Path(configuration.indexDirectory)
    manifest = directory / "medical_knowledge_manifest.json"
    result: dict[str, Any] = {
        "directory_configured": bool(str(directory)),
        "directory_exists": directory.is_dir(),
        "manifest_exists": manifest.is_file(),
        "manifest_valid": False,
        "error": None,
    }
    if manifest.is_file():
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            result["manifest_valid"] = isinstance(payload, dict)
        except Exception as exc:  # pragma: no cover - malformed local asset
            result["error"] = _error(exc)
    return result


def _embedding_contract(
    configuration: KnowledgeStoreConfiguration,
    env: Mapping[str, str] | None,
) -> dict[str, Any]:
    """Audit provider/index compatibility without making an embedding call.

    The database is the authority for the live index, but the checked-in asset
    manifests are still useful for catching a stale model, dimension, or
    incomplete chunk-vector asset before a quota-consuming retrieval probe.
    """
    source = env if env is not None else None
    result: dict[str, Any] = {
        "configuration_valid": False,
        "error_code": None,
        "model": None,
        "configured_dimension": configuration.vectorDimension,
        "configured_chunk_version": configuration.chunkVersion,
        "configured_chunk_variant": configuration.chunkVariant,
        "paper_asset": None,
        "chunk_asset": None,
        "graph_asset": None,
        "asset_contract_status": "NOT_AUDITED",
        "asset_configuration_consistent": False,
        "document_dense_ready": False,
        "chunk_v2_dense_ready": False,
        "legacy_chunk_dense_ready": False,
        "vector_capability": "NOT_AUDITED",
    }
    try:
        embedding = GeminiEmbeddingConfiguration.from_environment(source)
        result["configuration_valid"] = True
        result["model"] = embedding.modelName
    except Exception as exc:
        # These exception messages are stable internal codes and never contain
        # the API key or a query.  Keep them so an E2E trace does not collapse
        # every configuration failure into the opaque exception class.
        result["error_code"] = str(exc) or type(exc).__name__
        return result

    directory = Path(configuration.indexDirectory)
    def load(name: str) -> dict[str, Any] | None:
        path = directory / name
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {"invalid": True}
        if not isinstance(payload, dict):
            return {"invalid": True}
        return {
            "index_version": payload.get("indexVersion"),
            "graph_version": payload.get("graphVersion"),
            "model": payload.get("model"),
            "dimension": payload.get("dimension"),
            "granularity": payload.get("granularity"),
            "chunk_count": payload.get("chunkCount"),
            "chunk_version": payload.get("chunkVersion"),
            "variant": payload.get("variant"),
        }

    result["paper_asset"] = load("medical_gemini_paper_embedding_meta.json")
    result["chunk_asset"] = load("medical_gemini_chunk_embedding_index_v2_medium_meta.json")
    graph = load("medical_knowledge_graph_v3_manifest.json")
    result["graph_asset"] = graph
    paper = result["paper_asset"]
    chunk = result["chunk_asset"]
    graph_model = graph.get("graph_version") if isinstance(graph, dict) else None
    paper_ok = isinstance(paper, dict) and paper.get("model") == embedding.modelName and paper.get("dimension") == configuration.vectorDimension
    chunk_ok = isinstance(chunk, dict) and chunk.get("model") == embedding.modelName and chunk.get("dimension") == configuration.vectorDimension
    graph_ok = bool(graph_model is None or graph_model == configuration.graphVersion)
    chunk_version_ok = isinstance(chunk, dict) and chunk.get("chunk_version") == configuration.chunkVersion and chunk.get("variant") == configuration.chunkVariant
    result["document_dense_ready"] = paper_ok
    result["chunk_v2_dense_ready"] = chunk_ok and isinstance(chunk, dict) and chunk.get("chunk_version") == "chunk-v2" and chunk.get("variant") == "medium"
    # The legacy v1 asset is intentionally retained for sparse/provenance
    # compatibility.  It is not a dense retrieval asset.
    result["legacy_chunk_dense_ready"] = False
    result["asset_configuration_consistent"] = bool(paper_ok and chunk_ok and graph_ok and chunk_version_ok)
    # A partial chunk asset is a real limitation, not a reason to claim that
    # document-level vectors provide complete chunk-level dense retrieval.
    if paper_ok and chunk_ok and graph_ok and chunk_version_ok:
        result["asset_contract_status"] = "PARTIAL_CHUNK_COVERAGE" if int(chunk.get("chunk_count") or 0) < int(paper.get("chunk_count") or 0) else "PASS"
    else:
        result["asset_contract_status"] = "MISMATCH"
    if result["asset_configuration_consistent"]:
        result["vector_capability"] = "PARTIAL" if not result["legacy_chunk_dense_ready"] else "FULL"
    else:
        result["vector_capability"] = "UNAVAILABLE"
    return result


def probe_knowledge_backend(
    env: Mapping[str, str] | None = None,
    *,
    probe_connectivity: bool = True,
) -> dict[str, Any]:
    """Return a redacted, read-only knowledge-store health report.

    ``probe_connectivity=False`` performs only configuration/wiring checks,
    which is useful for unit tests and offline CI.  Retrieval is intentionally
    not part of this function: ``retrieval_probe_pass`` remains an explicit
    ``NOT_RUN_QUOTA_GUARD`` status until a separately authorized probe uses a
    cached embedding or a caller-supplied vector.
    """

    source = env if env is not None else {}
    resolver = ""
    if env is not None:
        resolver = str(source.get("MICO_KNOWLEDGE_RETRIEVAL_BACKEND", "")).strip().lower()
    else:
        import os

        resolver = os.environ.get("MICO_KNOWLEDGE_RETRIEVAL_BACKEND", "").strip().lower()
    result: dict[str, Any] = {
        "knowledge_wiring_pass": False,
        "knowledge_connectivity_pass": False,
        "knowledge_retrieval_probe_pass": "NOT_RUN_QUOTA_GUARD",
        "embedding_calls": 0,
        "resolver_backend": resolver or None,
        "configuration_valid": False,
        "configuration_error": None,
        "embedding_contract": None,
        "dependencies": {
            "psycopg": importlib.util.find_spec("psycopg") is not None,
            "neo4j": importlib.util.find_spec("neo4j") is not None,
            "pgvector": importlib.util.find_spec("pgvector") is not None,
        },
        "vector": None,
        "graph": None,
        "index_assets": None,
    }
    try:
        configuration = KnowledgeStoreConfiguration.from_environment(env)
        result["configuration_valid"] = True
    except Exception as exc:
        result["configuration_error"] = _error(exc)
        return result

    result["knowledge_wiring_pass"] = resolver in {"database", ""} and all(
        result["dependencies"].values()
    )
    result["index_assets"] = _index_assets(configuration)
    result["embedding_contract"] = _embedding_contract(configuration, env)
    if probe_connectivity:
        result["vector"] = _probe_vector_store(configuration)
        result["graph"] = _probe_graph_store(configuration)
        result["knowledge_connectivity_pass"] = bool(
            result["vector"]["connected"]
            and result["vector"]["schema_ready"]
            and result["graph"]["connected"]
            and result["graph"]["query_ok"]
        )
    return result


__all__ = ["probe_knowledge_backend"]
