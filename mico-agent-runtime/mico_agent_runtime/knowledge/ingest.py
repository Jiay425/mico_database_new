from __future__ import annotations

import json
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .database_config import KnowledgeStoreConfiguration
from .database_schema import load_manifest, vector_schema_sql
from .graph_pipeline import GraphPipelineError, load_graph_manifest
from .embeddings import GeminiEmbeddingPort


_VERSIONED_GRAPH_STATUSES = {"staging", "review_pending", "approved", "published"}


def validate_versioned_graph_ingestion(manifest: Any, publish: bool = False) -> None:
    """Apply the publication gate before opening a Neo4j driver.

    This is intentionally a pure validation boundary so the CLI and unit tests
    cannot accidentally turn a ``--publish`` flag into an approval decision.
    """
    if manifest.status not in _VERSIONED_GRAPH_STATUSES:
        raise GraphPipelineError("KNOWLEDGE_GRAPH_VERSION_NOT_STAGING")
    if publish and manifest.status != "approved":
        raise GraphPipelineError("KNOWLEDGE_GRAPH_PUBLICATION_APPROVAL_REQUIRED")
    if publish and (
        not manifest.publicationApprovalHash
        or not manifest.approvedBy
        or not manifest.approvedAt
    ):
        raise GraphPipelineError("KNOWLEDGE_GRAPH_PUBLICATION_APPROVAL_MISSING")


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _batched(values: list[dict[str, Any]], size: int = 500) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(values), size):
        yield values[start:start + size]


def _database_asset_manifest(directory: Path) -> dict[str, Any]:
    """Build the database-store manifest from the actual dense/vector and v3 assets."""
    manifest = dict(load_manifest(directory))
    embedding_meta_path = directory / "medical_gemini_paper_embedding_meta.json"
    graph_meta_path = directory / "medical_knowledge_graph_v3_manifest.json"
    if embedding_meta_path.exists():
        embedding_meta = json.loads(embedding_meta_path.read_text(encoding="utf-8"))
        manifest["indexVersion"] = embedding_meta["indexVersion"]
        manifest["paperCount"] = embedding_meta["documentCount"]
        manifest["chunkCount"] = embedding_meta["chunkCount"]
    if graph_meta_path.exists():
        graph_meta = json.loads(graph_meta_path.read_text(encoding="utf-8"))
        manifest["graphVersion"] = graph_meta["graphVersion"]
    return manifest


def initialize_vector_store(configuration: KnowledgeStoreConfiguration, manifest: dict[str, Any]) -> None:
    import psycopg

    with psycopg.connect(configuration.vectorDatabaseUrl) as connection:
        for statement in vector_schema_sql(configuration.vectorDimension):
            connection.execute(statement)
        connection.execute(
            """
            INSERT INTO knowledge_index_manifest
                (index_version, graph_version, corpus_scope, evidence_tier,
                 paper_count, chunk_count, created_at, manifest)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (index_version) DO UPDATE SET
                graph_version = EXCLUDED.graph_version,
                paper_count = EXCLUDED.paper_count,
                chunk_count = EXCLUDED.chunk_count,
                created_at = EXCLUDED.created_at,
                manifest = EXCLUDED.manifest
            """,
            (
                manifest["indexVersion"], manifest["graphVersion"], manifest["corpusScope"],
                manifest["evidenceTier"], manifest["paperCount"], manifest["chunkCount"],
                datetime.now(timezone.utc), json.dumps(manifest),
            ),
        )


def ingest_vector_store(configuration: KnowledgeStoreConfiguration) -> dict[str, int]:
    import psycopg
    from pgvector import HalfVector
    from pgvector.psycopg import register_vector

    directory = configuration.indexDirectory
    manifest = _database_asset_manifest(directory)
    dense = list(_jsonl(directory / "medical_gemini_paper_embedding_index.jsonl"))
    chunks = list(_jsonl(directory / "medical_chunks.jsonl"))
    if len(dense) != manifest["paperCount"] or len(chunks) != manifest["chunkCount"]:
        raise ValueError("knowledge corpus does not match manifest")
    initialize_vector_store(configuration, manifest)
    chunk_embeddings_enabled = os.environ.get(
        "MICO_KNOWLEDGE_CHUNK_EMBEDDINGS_ENABLED", "false"
    ).strip().lower() == "true"
    embedding_port = GeminiEmbeddingPort.from_environment() if chunk_embeddings_enabled else None
    embedded_chunks = 0
    with psycopg.connect(configuration.vectorDatabaseUrl) as connection:
        register_vector(connection)
        with connection.cursor() as cursor:
            for row in dense:
                if len(row["embedding"]) != configuration.vectorDimension:
                    raise ValueError("embedding dimension mismatch")
                cursor.execute(
                    """
                    INSERT INTO knowledge_document
                        (document_id, pmcid, pmid, doi, title, journal, publication_year,
                         topic, source_url, evidence_tier, embedding_model,
                         embedding_index_version, embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (document_id) DO UPDATE SET
                        pmid = EXCLUDED.pmid, doi = EXCLUDED.doi, title = EXCLUDED.title,
                        journal = EXCLUDED.journal, publication_year = EXCLUDED.publication_year,
                        topic = EXCLUDED.topic, source_url = EXCLUDED.source_url,
                        embedding = EXCLUDED.embedding, metadata = EXCLUDED.metadata
                    """,
                    (
                        str(row["paperId"]), row["pmcid"], row.get("pmid"), row.get("doi"),
                        row["title"], row.get("journal"), int(row.get("year") or 2000),
                        row.get("topic"), row["sourceUrl"], row["evidenceTier"],
                        row["model"], row["indexVersion"], HalfVector(row["embedding"]),
                        json.dumps({"chunkCount": row.get("chunkCount"), "sourceUrl": row.get("sourceUrl")}),
                    ),
                )
            ordinal_by_document: dict[str, int] = {}
            for row in chunks:
                document_id = str(row["pmcid"])
                ordinal = ordinal_by_document.get(document_id, 0)
                ordinal_by_document[document_id] = ordinal + 1
                chunk_embedding = None
                if embedding_port is not None:
                    chunk_embedding = HalfVector(
                        embedding_port.embed_document(row.get("title"), str(row["text"]))
                    )
                    embedded_chunks += 1
                cursor.execute(
                    """
                    INSERT INTO knowledge_chunk
                        (chunk_id, document_id, pmcid, pmid, title, journal, publication_year,
                         topic, section, doi, source_url, evidence_tier, ordinal, text, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        section = EXCLUDED.section, text = EXCLUDED.text,
                        ordinal = EXCLUDED.ordinal, metadata = EXCLUDED.metadata,
                        embedding = COALESCE(EXCLUDED.embedding, knowledge_chunk.embedding)
                    """,
                    (
                        row["chunkId"], document_id, row["pmcid"], row.get("pmid"), row["title"],
                        row.get("journal"), int(row.get("year") or 2000), row.get("topic"),
                        row.get("section"), row.get("doi"), row["sourceUrl"], "fulltext",
                        ordinal, row["text"], json.dumps({"tokenCount": row.get("tokenCount")}),
                        chunk_embedding,
                    ),
                )
    if embedding_port is not None:
        embedding_port.close()
    return {"documents": len(dense), "chunks": len(chunks), "chunkEmbeddings": embedded_chunks}


def ingest_chunk_variant(
    configuration: KnowledgeStoreConfiguration,
    chunks_path: Path,
    manifest_path: Path,
    graph_version: str,
) -> dict[str, Any]:
    """Add one chunk-v2 variant without replacing the v1 corpus.

    This path intentionally does not call the embedding provider.  It first
    registers the schema/asset manifest, verifies that all source documents
    already exist, and then inserts rows tagged with chunk version, variant,
    embedding version and graph version.  The primary key remains the opaque
    chunk id, while the version columns make accidental cross-variant reads
    impossible for configured retrievers.
    """
    import psycopg

    if not chunks_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("KNOWLEDGE_CHUNK_VARIANT_ASSET_MISSING")
    asset = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunk_version = str(asset.get("chunkVersion") or "")
    variant = str(asset.get("variant") or "")
    summary = asset.get("summary") or {}
    if chunk_version != "chunk-v2" or variant not in {"small", "medium", "large"}:
        raise ValueError("KNOWLEDGE_CHUNK_VARIANT_MANIFEST_INVALID")
    rows = list(_jsonl(chunks_path))
    expected = int(summary.get("chunkCount") or 0)
    papers = int(summary.get("paperCount") or 0)
    if not rows or len(rows) != expected:
        raise ValueError("KNOWLEDGE_CHUNK_VARIANT_COUNT_MISMATCH")
    if len({str(row.get("chunkId") or "") for row in rows}) != len(rows):
        raise ValueError("KNOWLEDGE_CHUNK_VARIANT_DUPLICATE_ID")
    pmcids = {str(row.get("pmcid") or "") for row in rows}
    if len(pmcids) != papers or "" in pmcids:
        raise ValueError("KNOWLEDGE_CHUNK_VARIANT_PAPER_MISMATCH")
    index_version = f"fulltext-gemini-{chunk_version}-{variant}-v1"
    db_manifest = {
        "indexVersion": index_version,
        "graphVersion": graph_version,
        "corpusScope": "fulltext_only",
        "evidenceTier": "fulltext",
        "paperCount": papers,
        "chunkCount": len(rows),
        "chunkVersion": chunk_version,
        "chunkVariant": variant,
        "chunkAsset": chunks_path.name,
        "graphAsset": f"medical_knowledge_graph_{graph_version}.jsonl",
        "embeddingStatus": "pending",
        "sourceManifest": manifest_path.name,
    }
    initialize_vector_store(configuration, db_manifest)
    ordinal_by_pmcid: dict[str, int] = {}
    with psycopg.connect(configuration.vectorDatabaseUrl) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT pmcid FROM knowledge_document WHERE pmcid = ANY(%s)",
                (list(pmcids),),
            )
            existing_pmcids = {str(row[0]) for row in cursor.fetchall()}
            missing_documents = sorted(pmcids - existing_pmcids)
            if missing_documents:
                raise ValueError("KNOWLEDGE_CHUNK_VARIANT_DOCUMENT_MISSING")
            for row in rows:
                pmcid = str(row["pmcid"])
                ordinal = ordinal_by_pmcid.get(pmcid, 0)
                ordinal_by_pmcid[pmcid] = ordinal + 1
                metadata = {
                    "tokenCount": row.get("tokenCount"),
                    "chunkType": row.get("chunkType"),
                    "subsection": row.get("subsection"),
                    "paragraphIndex": row.get("paragraphIndex"),
                    "paragraphEndIndex": row.get("paragraphEndIndex"),
                    "sentenceStart": row.get("sentenceStart"),
                    "sentenceEnd": row.get("sentenceEnd"),
                    "charStart": row.get("charStart"),
                    "charEnd": row.get("charEnd"),
                    "offsetBase": row.get("offsetBase"),
                    "embeddingText": row.get("embeddingText"),
                }
                cursor.execute(
                    """
                    INSERT INTO knowledge_chunk
                        (chunk_id, document_id, pmcid, pmid, title, journal, publication_year,
                         topic, section, doi, source_url, evidence_tier, ordinal, text, metadata,
                         embedding, chunk_version, chunk_variant, embedding_model,
                         embedding_version, graph_version)
                    SELECT %s, d.document_id, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                           'fulltext', %s, %s, %s, NULL, %s, %s, NULL, NULL, %s
                    FROM knowledge_document d
                    WHERE d.pmcid = %s
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        document_id = EXCLUDED.document_id, pmcid = EXCLUDED.pmcid,
                        pmid = EXCLUDED.pmid, title = EXCLUDED.title, journal = EXCLUDED.journal,
                        publication_year = EXCLUDED.publication_year, topic = EXCLUDED.topic,
                        section = EXCLUDED.section, doi = EXCLUDED.doi,
                        source_url = EXCLUDED.source_url, ordinal = EXCLUDED.ordinal,
                        text = EXCLUDED.text, metadata = EXCLUDED.metadata,
                        chunk_version = EXCLUDED.chunk_version, chunk_variant = EXCLUDED.chunk_variant,
                        graph_version = EXCLUDED.graph_version,
                        embedding = COALESCE(EXCLUDED.embedding, knowledge_chunk.embedding),
                        embedding_model = COALESCE(EXCLUDED.embedding_model, knowledge_chunk.embedding_model),
                        embedding_version = COALESCE(EXCLUDED.embedding_version, knowledge_chunk.embedding_version)
                    """,
                    (
                        str(row["chunkId"]), pmcid, row.get("pmid"), row.get("title") or "",
                        row.get("journal"), int(row.get("year") or 2000), row.get("topic"),
                        row.get("section"), row.get("doi"), row.get("sourceUrl") or "",
                        ordinal, row.get("text") or "", json.dumps(metadata, ensure_ascii=False),
                        chunk_version, variant, graph_version, pmcid,
                    ),
                )
            cursor.execute(
                """
                SELECT count(*), count(DISTINCT chunk_id),
                       count(*) FILTER (WHERE document_id IS NULL),
                       count(*) FILTER (WHERE pmcid IS NULL OR pmcid = ''),
                       count(*) FILTER (WHERE text IS NULL OR text = ''),
                       count(*) FILTER (WHERE section IS NULL OR section = ''),
                       count(*) FILTER (WHERE embedding IS NOT NULL)
                FROM knowledge_chunk
                WHERE chunk_version = %s AND chunk_variant = %s
                """,
                (chunk_version, variant),
            )
            count_row = cursor.fetchone()
            cursor.execute(
                """
                SELECT section, count(*)
                FROM knowledge_chunk
                WHERE chunk_version = %s AND chunk_variant = %s
                GROUP BY section ORDER BY section
                """,
                (chunk_version, variant),
            )
            sections = {str(row[0] or "unknown"): int(row[1]) for row in cursor.fetchall()}
    integrity = {
        "expectedChunks": len(rows),
        "databaseChunks": int(count_row[0]),
        "duplicateChunkId": int(count_row[1] - count_row[0]),
        "missingDocumentId": int(count_row[2]),
        "missingPmcid": int(count_row[3]),
        "missingText": int(count_row[4]),
        "missingSection": int(count_row[5]),
        "embeddingNonNull": int(count_row[6]),
        "sectionCoverage": sections,
    }
    integrity["valid"] = (
        integrity["expectedChunks"] == integrity["databaseChunks"]
        and integrity["duplicateChunkId"] == 0
        and integrity["missingDocumentId"] == 0
        and integrity["missingPmcid"] == 0
        and integrity["missingText"] == 0
    )
    return {
        "chunkVersion": chunk_version,
        "chunkVariant": variant,
        "graphVersion": graph_version,
        "indexVersion": index_version,
        "integrity": integrity,
    }


def ingest_graph_store(configuration: KnowledgeStoreConfiguration) -> dict[str, int]:
    from neo4j import GraphDatabase

    rows = list(_jsonl(configuration.indexDirectory / "medical_knowledge_graph.jsonl"))
    nodes = [row for row in rows if row.get("recordType") == "node"]
    edges = [row for row in rows if row.get("recordType") == "edge"]
    driver = GraphDatabase.driver(
        configuration.neo4jUri,
        auth=(configuration.neo4jUser, configuration.neo4jPassword),
    )
    try:
        with driver.session() as session:
            session.run(
                "CREATE CONSTRAINT knowledge_entity_id IF NOT EXISTS FOR (n:KnowledgeEntity) REQUIRE n.nodeId IS UNIQUE"
            ).consume()
            session.run(
                "CREATE INDEX knowledge_entity_label IF NOT EXISTS FOR (n:KnowledgeEntity) ON (n.label)"
            ).consume()
            node_query = """
                UNWIND $rows AS row
                MERGE (n:KnowledgeEntity {nodeId: row.nodeId})
                SET n.nodeType = row.nodeType,
                    n.label = row.label,
                    n.evidenceTier = row.evidenceTier,
                    n.pmcid = row.pmcid,
                    n.pmid = row.pmid,
                    n.doi = row.doi,
                    n.sourceUrl = row.sourceUrl,
                    n.aliases = row.aliases,
                    n.extractionMethod = row.extractionMethod
            """
            for batch in _batched(nodes):
                session.run(node_query, rows=batch).consume()
            edge_query = """
                UNWIND $rows AS row
                MATCH (source:KnowledgeEntity {nodeId: row.source}),
                      (target:KnowledgeEntity {nodeId: row.target})
                MERGE (source)-[r:KNOWLEDGE_RELATION {edgeId: row.edgeId}]->(target)
                SET r.relation = row.relation,
                    r.evidenceChunkId = row.evidenceChunkId,
                    r.evidenceTier = row.evidenceTier,
                    r.assertionStatus = row.assertionStatus,
                    r.extractionMethod = row.extractionMethod
            """
            for batch in _batched(edges):
                session.run(edge_query, rows=batch).consume()
    finally:
        driver.close()
    return {"nodes": len(nodes), "edges": len(edges)}


def ingest_graph_v3_store(configuration: KnowledgeStoreConfiguration) -> dict[str, int | str]:
    """Add the versioned semantic graph without deleting the v2 graph.

    v3 node IDs are version-prefixed, so the old graph remains available for
    rollback and comparison.  Runtime retrieval explicitly filters the v3
    graph version; this function never touches the business database.
    """
    from neo4j import GraphDatabase

    graph_path = configuration.indexDirectory / "medical_knowledge_graph_v3.jsonl"
    if not graph_path.exists():
        raise FileNotFoundError("KNOWLEDGE_GRAPH_V3_NOT_BUILT")
    rows = list(_jsonl(graph_path))
    nodes = [row for row in rows if row.get("recordType") == "node"]
    edges = [row for row in rows if row.get("recordType") == "edge"]
    if not nodes or not edges or any(row.get("graphVersion") != "fulltext-provenance-graphrag-v3" for row in rows):
        raise ValueError("KNOWLEDGE_GRAPH_V3_MANIFEST_INVALID")
    driver = GraphDatabase.driver(
        configuration.neo4jUri,
        auth=(configuration.neo4jUser, configuration.neo4jPassword),
    )
    graph_version = "fulltext-provenance-graphrag-v3"
    build_run_id = "graph-build-" + hashlib.sha256(
        f"{graph_version}|{len(nodes)}|{len(edges)}".encode("utf-8")
    ).hexdigest()[:32]
    manifest_path = configuration.indexDirectory / "medical_knowledge_graph_v3_manifest.json"
    created_at = datetime.now(timezone.utc).isoformat()
    if manifest_path.exists():
        try:
            created_at = str(json.loads(manifest_path.read_text(encoding="utf-8")).get("createdAt") or created_at)
        except (OSError, ValueError, TypeError):
            pass
    try:
        with driver.session() as session:
            # Replace only the rebuildable v3 knowledge graph.  The previous
            # v2 graph is intentionally left untouched for rollback/comparison.
            session.run(
                "MATCH (n:KnowledgeEntity {graphVersion: $version}) DETACH DELETE n",
                version="fulltext-provenance-graphrag-v3",
            ).consume()
            session.run(
                "CREATE INDEX knowledge_entity_graph_version IF NOT EXISTS FOR (n:KnowledgeEntity) ON (n.graphVersion)"
            ).consume()
            session.run(
                "CREATE INDEX knowledge_relation_graph_version IF NOT EXISTS FOR ()-[r:KNOWLEDGE_RELATION]-() ON (r.graphVersion)"
            ).consume()
            node_query = """
                UNWIND $rows AS row
                MERGE (n:KnowledgeEntity {nodeId: row.nodeId})
                SET n.graphVersion = row.graphVersion,
                    n.nodeType = row.nodeType,
                    n.entityType = row.entityType,
                    n.label = row.label,
                    n.canonicalLabel = row.canonicalLabel,
                    n.evidenceTier = row.evidenceTier,
                    n.pmcid = row.pmcid,
                    n.pmid = row.pmid,
                    n.doi = row.doi,
                    n.year = row.year,
                    n.sourceUrl = row.sourceUrl,
                    n.section = row.section,
                    n.aliases = row.aliases,
                    n.normalizationStatus = row.normalizationStatus,
                    n.extractionMethod = row.extractionMethod
            """
            for batch in _batched(nodes):
                session.run(node_query, rows=batch).consume()
            edge_query = """
                UNWIND $rows AS row
                MATCH (source:KnowledgeEntity {nodeId: row.source}),
                      (target:KnowledgeEntity {nodeId: row.target})
                MERGE (source)-[r:KNOWLEDGE_RELATION {edgeId: row.edgeId}]->(target)
                SET r.graphVersion = row.graphVersion,
                    r.relation = row.relation,
                    r.relationClass = row.relationClass,
                    r.evidenceChunkId = row.evidenceChunkId,
                    r.evidenceTier = row.evidenceTier,
                    r.confidence = row.confidence,
                    r.assertionStatus = row.assertionStatus,
                    r.extractionMethod = row.extractionMethod,
                    r.evidenceText = row.evidenceText,
                    r.evidenceStart = row.evidenceStart,
                    r.evidenceEnd = row.evidenceEnd
            """
            for batch in _batched(edges):
                session.run(edge_query, rows=batch).consume()
            # The legacy v3 asset predates the review manifest model.  Register
            # it as the explicitly retained published baseline so retrieval can
            # require a published build marker for every graph version.
            session.run(
                """
                MERGE (build:KnowledgeGraphBuild {graphVersion: $version})
                SET build.buildRunId = $build_run_id,
                    build.status = 'published',
                    build.createdAt = $created_at,
                    build.nodeCount = $node_count,
                    build.edgeCount = $edge_count
                """,
                version=graph_version,
                build_run_id=build_run_id,
                created_at=created_at,
                node_count=len(nodes),
                edge_count=len(edges),
            ).consume()
    finally:
        driver.close()
    return {
        "graphVersion": graph_version,
        "nodes": len(nodes),
        "edges": len(edges),
    }


def ingest_graph_versioned_store(
    configuration: KnowledgeStoreConfiguration,
    graph_path: Path | None = None,
    manifest_path: Path | None = None,
    publish: bool = False,
) -> dict[str, int | str]:
    """Ingest one validated staging graph without touching other versions.

    The version and build metadata are supplied by the graph manifest, never
    by a table name or free Cypher fragment.  This function is intentionally
    separate from the legacy v2/v3 loaders so a new build can be staged and
    inspected before an explicit publication switch.
    """
    from neo4j import GraphDatabase

    directory = configuration.indexDirectory
    graph_file = graph_path or directory / "medical_knowledge_graph_v4.jsonl"
    manifest_file = manifest_path or directory / "medical_knowledge_graph_v4_manifest.json"
    manifest = load_graph_manifest(manifest_file)
    if not graph_file.exists():
        raise GraphPipelineError("KNOWLEDGE_GRAPH_VERSION_NOT_STAGING")
    validate_versioned_graph_ingestion(manifest, publish=publish)
    rows = list(_jsonl(graph_file))
    nodes = [row for row in rows if row.get("recordType") == "node"]
    edges = [row for row in rows if row.get("recordType") == "edge"]
    if not nodes or not edges or any(row.get("graphVersion") != manifest.graphVersion for row in rows):
        raise GraphPipelineError("KNOWLEDGE_GRAPH_MANIFEST_MISMATCH")
    if len(nodes) != manifest.nodeCount or len(edges) != manifest.edgeCount:
        raise GraphPipelineError("KNOWLEDGE_GRAPH_COUNTS_MISMATCH")
    driver = GraphDatabase.driver(
        configuration.neo4jUri,
        auth=(configuration.neo4jUser, configuration.neo4jPassword),
    )
    try:
        with driver.session() as session:
            session.run(
                "CREATE INDEX knowledge_entity_graph_version IF NOT EXISTS FOR (n:KnowledgeEntity) ON (n.graphVersion)"
            ).consume()
            session.run(
                "CREATE INDEX knowledge_relation_graph_version IF NOT EXISTS FOR ()-[r:KNOWLEDGE_RELATION]-() ON (r.graphVersion)"
            ).consume()
            session.run(
                "MATCH (n:KnowledgeEntity {graphVersion: $version}) DETACH DELETE n",
                version=manifest.graphVersion,
            ).consume()
            node_query = """
                UNWIND $rows AS row
                MERGE (n:KnowledgeEntity {nodeId: row.nodeId})
                SET n.graphVersion = row.graphVersion,
                    n.graphBuildRunId = row.graphBuildRunId,
                    n.nodeType = row.nodeType,
                    n.entityType = row.entityType,
                    n.label = row.label,
                    n.canonicalLabel = row.canonicalLabel,
                    n.canonicalEntityId = row.canonicalEntityId,
                    n.ontologySource = row.ontologySource,
                    n.ontologyId = row.ontologyId,
                    n.evidenceTier = row.evidenceTier,
                    n.pmcid = row.pmcid,
                    n.pmid = row.pmid,
                    n.doi = row.doi,
                    n.year = row.year,
                    n.sourceUrl = row.sourceUrl,
                    n.section = row.section,
                    n.aliases = row.aliases,
                    n.normalizationStatus = row.normalizationStatus,
                    n.qualityStatus = row.qualityStatus,
                    n.extractionMethod = row.extractionMethod
            """
            for batch in _batched(nodes):
                session.run(node_query, rows=batch).consume()
            edge_query = """
                UNWIND $rows AS row
                MATCH (source:KnowledgeEntity {nodeId: row.source}),
                      (target:KnowledgeEntity {nodeId: row.target})
                MERGE (source)-[r:KNOWLEDGE_RELATION {edgeId: row.edgeId}]->(target)
                SET r.graphVersion = row.graphVersion,
                    r.graphBuildRunId = row.graphBuildRunId,
                    r.relation = row.relation,
                    r.relationClass = row.relationClass,
                    r.evidenceChunkId = row.evidenceChunkId,
                    r.evidenceTier = row.evidenceTier,
                    r.confidence = row.confidence,
                    r.assertionStatus = row.assertionStatus,
                    r.qualityStatus = row.qualityStatus,
                    r.extractionMethod = row.extractionMethod,
                    r.evidenceText = row.evidenceText,
                    r.evidenceStart = row.evidenceStart,
                    r.evidenceEnd = row.evidenceEnd
            """
            for batch in _batched(edges):
                session.run(edge_query, rows=batch).consume()
            session.run(
                """
                MERGE (build:KnowledgeGraphBuild {graphVersion: $version})
                SET build.buildRunId = $build_run_id,
                    build.status = $status,
                    build.createdAt = $created_at,
                    build.paperCount = $paper_count,
                    build.chunkCount = $chunk_count,
                    build.nodeCount = $node_count,
                    build.edgeCount = $edge_count,
                    build.semanticRelationCount = $semantic_count,
                    build.rejectedRecordCount = $rejected_count
                """,
                version=manifest.graphVersion,
                build_run_id=manifest.buildRunId,
                status="published" if publish else manifest.status,
                created_at=manifest.createdAt,
                paper_count=manifest.paperCount,
                chunk_count=manifest.chunkCount,
                node_count=manifest.nodeCount,
                edge_count=manifest.edgeCount,
                semantic_count=manifest.semanticRelationCount,
                rejected_count=manifest.rejectedRecordCount,
            ).consume()
            if publish:
                session.run(
                    """
                    MATCH (build:KnowledgeGraphBuild)
                    WHERE build.graphVersion <> $version AND build.status = 'published'
                    SET build.status = 'retired'
                    """,
                    version=manifest.graphVersion,
                ).consume()
    finally:
        driver.close()
    return {
        "graphVersion": manifest.graphVersion,
        "buildRunId": manifest.buildRunId,
        "status": "published" if publish else manifest.status,
        "nodes": len(nodes),
        "edges": len(edges),
    }


def ingest_all(configuration: KnowledgeStoreConfiguration) -> dict[str, Any]:
    # The default database load is the published v3 baseline.  The v4 asset is
    # intentionally a separate review/staging operation and is never silently
    # made active by the bulk loader.
    manifest = _database_asset_manifest(configuration.indexDirectory)
    return {
        "vector": ingest_vector_store(configuration),
        "graph": ingest_graph_v3_store(configuration),
        "indexVersion": manifest["indexVersion"],
        "graphVersion": "fulltext-provenance-graphrag-v3",
    }
