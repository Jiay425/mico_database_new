#!/usr/bin/env python3
"""Ingest one validated chunk-v2 variant into the independent pgvector DB.

This command writes no embedding vectors.  It is the version-isolation step
that must happen before any provider call.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mico_agent_runtime.knowledge.database_config import KnowledgeStoreConfiguration
from mico_agent_runtime.knowledge.ingest import ingest_chunk_variant


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest a chunk-v2 variant")
    parser.add_argument("--chunks", default="medical_chunks_v2_medium.jsonl")
    parser.add_argument("--manifest", default="medical_rag_manifest_v2_medium.json")
    parser.add_argument(
        "--graph-version",
        default="fulltext-provenance-graphrag-v4-v2m-r3",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_root = Path(__file__).resolve().parents[1]
    workspace_root = runtime_root.parent
    knowledge_dir = workspace_root / "mico_database_new" / "references" / "knowledge" / "medical" / "rag"
    chunks_path = knowledge_dir / args.chunks
    manifest_path = knowledge_dir / args.manifest
    configuration = KnowledgeStoreConfiguration.from_environment()
    result = ingest_chunk_variant(configuration, chunks_path, manifest_path, args.graph_version)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["integrity"]["valid"]:
        raise SystemExit("KNOWLEDGE_CHUNK_VARIANT_INTEGRITY_FAILED")


if __name__ == "__main__":
    main()
