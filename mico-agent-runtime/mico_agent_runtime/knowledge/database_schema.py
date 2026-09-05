from __future__ import annotations

from pathlib import Path


def vector_schema_sql(dimension: int = 3072) -> tuple[str, ...]:
    """Return the fixed schema for the independent pgvector knowledge store."""
    if dimension <= 0 or dimension > 16000:
        raise ValueError("invalid vector dimension")
    return (
        "CREATE EXTENSION IF NOT EXISTS vector",
        """
        CREATE TABLE IF NOT EXISTS knowledge_index_manifest (
            index_version TEXT PRIMARY KEY,
            graph_version TEXT NOT NULL,
            corpus_scope TEXT NOT NULL CHECK (corpus_scope = 'fulltext_only'),
            evidence_tier TEXT NOT NULL CHECK (evidence_tier = 'fulltext'),
            paper_count INTEGER NOT NULL CHECK (paper_count >= 0),
            chunk_count INTEGER NOT NULL CHECK (chunk_count >= 0),
            created_at TIMESTAMPTZ NOT NULL,
            manifest JSONB NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS knowledge_document (
            document_id TEXT PRIMARY KEY,
            pmcid TEXT NOT NULL UNIQUE,
            pmid TEXT,
            doi TEXT,
            title TEXT NOT NULL,
            journal TEXT,
            publication_year INTEGER,
            topic TEXT,
            source_url TEXT NOT NULL,
            evidence_tier TEXT NOT NULL CHECK (evidence_tier = 'fulltext'),
            embedding_model TEXT NOT NULL,
            embedding_index_version TEXT NOT NULL,
            embedding HALFVEC(%d) NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb
        )
        """ % dimension,
        """
        CREATE TABLE IF NOT EXISTS knowledge_chunk (
            chunk_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES knowledge_document(document_id) ON DELETE CASCADE,
            pmcid TEXT NOT NULL,
            pmid TEXT,
            title TEXT NOT NULL,
            journal TEXT,
            publication_year INTEGER,
            topic TEXT,
            section TEXT,
            doi TEXT,
            source_url TEXT NOT NULL,
            evidence_tier TEXT NOT NULL CHECK (evidence_tier = 'fulltext'),
            ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
            text TEXT NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            chunk_version TEXT NOT NULL DEFAULT 'chunk-v1',
            chunk_variant TEXT NOT NULL DEFAULT 'legacy',
            embedding_model TEXT,
            embedding_version TEXT,
            graph_version TEXT
        )
        """,
        # Chunk-level dense retrieval is a separate migration so an existing
        # document-level store can be upgraded in place.  Ingestion may fill
        # this column incrementally; retrieval falls back to the document
        # vector until every chunk has a dense vector.
        """
        ALTER TABLE knowledge_chunk
        ADD COLUMN IF NOT EXISTS embedding HALFVEC(%d)
        """ % dimension,
        """
        ALTER TABLE knowledge_chunk
        ADD COLUMN IF NOT EXISTS chunk_version TEXT NOT NULL DEFAULT 'chunk-v1'
        """,
        """
        ALTER TABLE knowledge_chunk
        ADD COLUMN IF NOT EXISTS chunk_variant TEXT NOT NULL DEFAULT 'legacy'
        """,
        """
        ALTER TABLE knowledge_chunk
        ADD COLUMN IF NOT EXISTS embedding_model TEXT
        """,
        """
        ALTER TABLE knowledge_chunk
        ADD COLUMN IF NOT EXISTS embedding_version TEXT
        """,
        """
        ALTER TABLE knowledge_chunk
        ADD COLUMN IF NOT EXISTS graph_version TEXT
        """,
        """
        CREATE INDEX IF NOT EXISTS knowledge_document_embedding_hnsw
        ON knowledge_document USING hnsw (embedding halfvec_cosine_ops)
        """,
        """
        CREATE INDEX IF NOT EXISTS knowledge_chunk_document_idx
        ON knowledge_chunk (document_id, ordinal)
        """,
        """
        CREATE INDEX IF NOT EXISTS knowledge_chunk_variant_idx
        ON knowledge_chunk (chunk_version, chunk_variant, document_id, ordinal)
        """,
        """
        CREATE INDEX IF NOT EXISTS knowledge_chunk_fts_idx
        ON knowledge_chunk USING gin (
            to_tsvector('english', coalesce(title, '') || ' ' || coalesce(text, ''))
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS knowledge_chunk_embedding_hnsw
        ON knowledge_chunk USING hnsw (embedding halfvec_cosine_ops)
        """,
    )


def load_manifest(index_directory: Path) -> dict:
    import json

    return json.loads((index_directory / "medical_knowledge_manifest.json").read_text(encoding="utf-8"))
