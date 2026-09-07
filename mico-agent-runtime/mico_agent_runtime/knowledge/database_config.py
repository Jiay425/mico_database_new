from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Mapping
from urllib.parse import urlparse


class KnowledgeStoreConfigurationError(ValueError):
    """The independent knowledge stores are not safely configured."""


_GRAPH_VERSION_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,79}$")
_ASSET_VERSION_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,79}$")


def _required(source: Mapping[str, str], name: str) -> str:
    value = source.get(name, "").strip()
    if not value or "\x00" in value:
        raise KnowledgeStoreConfigurationError(f"{name}_MISSING")
    return value


def _validate_vector_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"postgresql", "postgresql+psycopg"}:
        raise KnowledgeStoreConfigurationError("KNOWLEDGE_VECTOR_URL_NOT_POSTGRES")
    if not parsed.hostname or not parsed.path or parsed.path == "/":
        raise KnowledgeStoreConfigurationError("KNOWLEDGE_VECTOR_URL_INVALID")
    database = parsed.path.lstrip("/")
    if database in {"", "patient_data_manager", "mico_agent_runtime"}:
        raise KnowledgeStoreConfigurationError("KNOWLEDGE_VECTOR_DATABASE_FORBIDDEN")
    if parsed.query or parsed.fragment:
        raise KnowledgeStoreConfigurationError("KNOWLEDGE_VECTOR_URL_QUERY_FORBIDDEN")
    return value


@dataclass(frozen=True)
class KnowledgeStoreConfiguration:
    vectorDatabaseUrl: str
    neo4jUri: str
    neo4jUser: str
    neo4jPassword: str
    indexDirectory: Path
    vectorDimension: int = 3072
    # Internal candidate pools.  The public evidence response remains bounded
    # by EvidenceQuery.limit (currently Top-10); fusion/reranking needs a much
    # wider pool to measure and improve Recall@50.
    vectorTopK: int = 50
    graphTopK: int = 50
    sparseTopK: int = 50
    graphVersion: str = "fulltext-provenance-graphrag-v3"
    # v1/legacy remains queryable for backwards compatibility, but it has no
    # dense vectors.  The default must identify the canonical dense asset used
    # by the database retriever.
    chunkVersion: str = "chunk-v2"
    chunkVariant: str = "medium"

    @classmethod
    def from_environment(
        cls,
        env: Mapping[str, str] | None = None,
    ) -> "KnowledgeStoreConfiguration":
        import os

        source = os.environ if env is None else env
        if source.get("MICO_KNOWLEDGE_VECTOR_ENABLED", "").strip().lower() != "true":
            raise KnowledgeStoreConfigurationError("KNOWLEDGE_VECTOR_DISABLED")
        if source.get("MICO_KNOWLEDGE_GRAPH_ENABLED", "").strip().lower() != "true":
            raise KnowledgeStoreConfigurationError("KNOWLEDGE_GRAPH_DISABLED")
        vector_url = _validate_vector_url(_required(source, "MICO_KNOWLEDGE_VECTOR_DATABASE_URL"))
        neo4j_uri = _required(source, "MICO_KNOWLEDGE_NEO4J_URI")
        if not neo4j_uri.startswith(("bolt://", "neo4j://", "neo4j+s://", "neo4j+ssc://")):
            raise KnowledgeStoreConfigurationError("KNOWLEDGE_GRAPH_URI_INVALID")
        user = _required(source, "MICO_KNOWLEDGE_NEO4J_USER")
        password = _required(source, "MICO_KNOWLEDGE_NEO4J_PASSWORD")
        index_directory = Path(_required(source, "MICO_LOCAL_KNOWLEDGE_INDEX_DIR")).expanduser()
        try:
            dimension = int(source.get("MICO_KNOWLEDGE_VECTOR_DIMENSION", "3072"))
            vector_top_k = int(source.get("MICO_KNOWLEDGE_VECTOR_TOP_K", "50"))
            sparse_top_k = int(source.get("MICO_KNOWLEDGE_SPARSE_TOP_K", "50"))
            graph_top_k = int(source.get("MICO_KNOWLEDGE_GRAPH_TOP_K", "50"))
        except ValueError as exc:
            raise KnowledgeStoreConfigurationError("KNOWLEDGE_LIMIT_INVALID") from exc
        if dimension <= 0 or dimension > 16000:
            raise KnowledgeStoreConfigurationError("KNOWLEDGE_VECTOR_DIMENSION_INVALID")
        if not 1 <= vector_top_k <= 100 or not 1 <= sparse_top_k <= 100 or not 1 <= graph_top_k <= 100:
            raise KnowledgeStoreConfigurationError("KNOWLEDGE_LIMIT_INVALID")
        graph_version = source.get(
            "MICO_KNOWLEDGE_GRAPH_VERSION",
            "fulltext-provenance-graphrag-v3",
        ).strip()
        if not _GRAPH_VERSION_RE.fullmatch(graph_version):
            raise KnowledgeStoreConfigurationError("KNOWLEDGE_GRAPH_VERSION_INVALID")
        chunk_version = source.get("MICO_KNOWLEDGE_CHUNK_VERSION", "chunk-v2").strip()
        chunk_variant = source.get("MICO_KNOWLEDGE_CHUNK_VARIANT", "medium").strip()
        if not _ASSET_VERSION_RE.fullmatch(chunk_version):
            raise KnowledgeStoreConfigurationError("KNOWLEDGE_CHUNK_VERSION_INVALID")
        if not _ASSET_VERSION_RE.fullmatch(chunk_variant):
            raise KnowledgeStoreConfigurationError("KNOWLEDGE_CHUNK_VARIANT_INVALID")
        return cls(
            vectorDatabaseUrl=vector_url,
            neo4jUri=neo4j_uri,
            neo4jUser=user,
            neo4jPassword=password,
            indexDirectory=index_directory,
            vectorDimension=dimension,
            vectorTopK=vector_top_k,
            sparseTopK=sparse_top_k,
            graphTopK=graph_top_k,
            graphVersion=graph_version,
            chunkVersion=chunk_version,
            chunkVariant=chunk_variant,
        )
