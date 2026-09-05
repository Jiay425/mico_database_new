#!/usr/bin/env python3
"""Import a versioned chunk embedding JSONL into PostgreSQL.

The importer never creates a missing chunk and never updates another variant:
both the chunk version and variant are part of the UPDATE predicate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mico_agent_runtime.knowledge.database_config import KnowledgeStoreConfiguration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import chunk-level embeddings")
    parser.add_argument("--embedding-index", required=True)
    parser.add_argument("--chunk-version", default="chunk-v2")
    parser.add_argument("--variant", required=True, choices=("small", "medium", "large"))
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()
    path = Path(args.embedding_index).resolve()
    rows = load_rows(path)
    if not rows:
        raise SystemExit("CHUNK_EMBEDDING_INDEX_EMPTY")
    expected_version = f"fulltext-gemini-{args.chunk_version}-{args.variant}-v1"
    for row in rows:
        if (
            row.get("chunkVersion") != args.chunk_version
            or row.get("variant") != args.variant
            or row.get("granularity") != "chunk"
            or row.get("indexVersion") != expected_version
        ):
            raise SystemExit("CHUNK_EMBEDDING_INDEX_METADATA_MISMATCH")
        if not row.get("chunkId") or not isinstance(row.get("embedding"), list):
            raise SystemExit("CHUNK_EMBEDDING_INDEX_ROW_INVALID")
    ids = [str(row["chunkId"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("CHUNK_EMBEDDING_INDEX_DUPLICATE_ID")
    dimensions = {len(row["embedding"]) for row in rows}
    if dimensions != {3072}:
        raise SystemExit(f"CHUNK_EMBEDDING_DIMENSION_INVALID: {sorted(dimensions)}")

    import psycopg
    from pgvector import HalfVector
    from pgvector.psycopg import register_vector

    configuration = KnowledgeStoreConfiguration.from_environment()
    if configuration.vectorDimension != 3072:
        raise SystemExit("CHUNK_EMBEDDING_DIMENSION_CONFIG_MISMATCH")
    with psycopg.connect(configuration.vectorDatabaseUrl) as connection:
        register_vector(connection)
        with connection.cursor() as cursor:
            updated = 0
            missing: list[str] = []
            for row in rows:
                cursor.execute(
                    """
                    UPDATE knowledge_chunk
                    SET embedding = %s,
                        embedding_model = %s,
                        embedding_version = %s
                    WHERE chunk_id = %s
                      AND chunk_version = %s
                      AND chunk_variant = %s
                    """,
                    (
                        HalfVector(row["embedding"]),
                        row.get("model") or "gemini-embedding-2",
                        row["indexVersion"],
                        str(row["chunkId"]),
                        args.chunk_version,
                        args.variant,
                    ),
                )
                if cursor.rowcount == 1:
                    updated += 1
                else:
                    missing.append(str(row["chunkId"]))
            if missing:
                raise ValueError("CHUNK_EMBEDDING_TARGET_MISSING")
            cursor.execute(
                """
                SELECT count(*), count(*) FILTER (WHERE embedding IS NOT NULL),
                       count(*) FILTER (WHERE embedding_model = %s),
                       count(*) FILTER (WHERE embedding_version = %s)
                FROM knowledge_chunk
                WHERE chunk_version = %s AND chunk_variant = %s
            """,
                ("gemini-embedding-2", expected_version, args.chunk_version, args.variant),
            )
            count_row = cursor.fetchone()
            embedding_status = "complete" if int(count_row[1]) == int(count_row[0]) else "partial"
            cursor.execute(
                """
                UPDATE knowledge_index_manifest
                SET manifest = manifest || %s::jsonb
                WHERE index_version = %s
                """,
                (
                    json.dumps({
                        "embeddingStatus": embedding_status,
                        "embeddedChunkCount": int(count_row[1]),
                        "lastEmbeddingImport": expected_version,
                    }),
                    expected_version,
                ),
            )
    result = {
        "embeddingIndex": str(path),
        "chunkVersion": args.chunk_version,
        "chunkVariant": args.variant,
        "indexVersion": expected_version,
        "vectorsRead": len(rows),
        "rowsUpdated": updated,
        "databaseVariantChunks": int(count_row[0]),
        "databaseEmbeddingNonNull": int(count_row[1]),
        "databaseEmbeddingModelMatches": int(count_row[2]),
        "databaseEmbeddingVersionMatches": int(count_row[3]),
        "embeddingStatus": embedding_status,
        "dimension": 3072,
        "valid": updated == len(rows) and int(count_row[1]) >= updated,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["valid"]:
        raise SystemExit("CHUNK_EMBEDDING_IMPORT_INTEGRITY_FAILED")


if __name__ == "__main__":
    main()
