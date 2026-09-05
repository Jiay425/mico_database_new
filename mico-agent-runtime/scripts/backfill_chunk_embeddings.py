"""Resumable opt-in backfill for chunk-level Gemini embeddings.

The script updates only NULL embeddings in the independent knowledge store.
It is intentionally explicit because each row is an external embedding API
call; running the script is never part of application startup.
"""

from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0, help="0 means all remaining chunks")
    parser.add_argument("--chunk-version", default="chunk-v2")
    parser.add_argument("--variant", choices=("small", "medium", "large"), default="medium")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.batch_size > 256 or args.limit < 0:
        raise SystemExit("invalid batch-size or limit")

    import psycopg
    from pgvector import HalfVector

    from mico_agent_runtime.knowledge.database_config import KnowledgeStoreConfiguration
    from mico_agent_runtime.knowledge.embeddings import GeminiEmbeddingPort

    config = KnowledgeStoreConfiguration.from_environment()
    with psycopg.connect(config.vectorDatabaseUrl) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT count(*) FROM knowledge_chunk
                WHERE evidence_tier = 'fulltext'
                  AND chunk_version = %s AND chunk_variant = %s
                  AND embedding IS NULL
                """,
                (args.chunk_version, args.variant),
            )
            remaining = int(cursor.fetchone()[0])
    if args.dry_run:
        print({"status": "DRY_RUN", "remaining": remaining})
        return
    embedding = GeminiEmbeddingPort.from_environment()
    updated = 0
    try:
        while remaining and (args.limit == 0 or updated < args.limit):
            take = min(args.batch_size, remaining, args.limit - updated if args.limit else remaining)
            with psycopg.connect(config.vectorDatabaseUrl) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT chunk_id, title, text, topic, section, metadata
                        FROM knowledge_chunk
                        WHERE evidence_tier = 'fulltext'
                          AND chunk_version = %s AND chunk_variant = %s
                          AND embedding IS NULL
                        ORDER BY chunk_id
                        LIMIT %s
                        """,
                        (args.chunk_version, args.variant, take),
                    )
                    rows = cursor.fetchall()
                    for chunk_id, title, text, topic, section, metadata in rows:
                        metadata = metadata if isinstance(metadata, dict) else {}
                        # Reuse the exact source text used by the offline
                        # chunk index whenever it was persisted.  Falling
                        # back to the canonical header keeps older rows
                        # deterministic without mixing v1 rows into v2.
                        embedding_text = str(metadata.get("embeddingText") or "")
                        if not embedding_text:
                            embedding_text = "\n".join([
                                f"Title: {title or 'none'}",
                                f"Topic: {topic or 'unknown'}",
                                f"Section: {section or '正文'}",
                                "Subsection: none",
                                str(text or ""),
                            ])
                        vector = embedding.embed_document(title, embedding_text)
                        if len(vector) != config.vectorDimension:
                            raise SystemExit("CHUNK_EMBEDDING_DIMENSION_MISMATCH")
                        cursor.execute(
                            """
                            UPDATE knowledge_chunk
                            SET embedding = %s,
                                embedding_model = %s,
                                embedding_version = %s
                            WHERE chunk_id = %s
                              AND chunk_version = %s AND chunk_variant = %s
                              AND embedding IS NULL
                            """,
                            (
                                HalfVector(vector),
                                "gemini-embedding-2",
                                f"fulltext-gemini-{args.chunk_version}-{args.variant}-v1",
                                chunk_id,
                                args.chunk_version,
                                args.variant,
                            ),
                        )
                    updated += len(rows)
            remaining -= len(rows)
            print({"status": "RUNNING", "updated": updated, "remaining": remaining})
            if not rows:
                break
    finally:
        embedding.close()
    print({
        "status": "COMPLETED",
        "updated": updated,
        "remaining": remaining,
        "chunkVersion": args.chunk_version,
        "variant": args.variant,
    })


if __name__ == "__main__":
    main()
