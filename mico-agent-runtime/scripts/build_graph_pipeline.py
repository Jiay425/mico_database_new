from __future__ import annotations

import argparse
import json
from pathlib import Path

from mico_agent_runtime.knowledge.graph_pipeline import (
    GRAPH_PIPELINE_VERSION,
    build_versioned_graph,
    save_graph_build,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a source-bound versioned graph")
    parser.add_argument(
        "--chunks",
        default="medical_chunks_v2_medium.jsonl",
        help="chunk JSONL filename under the medical rag directory",
    )
    parser.add_argument(
        "--graph-version",
        default="fulltext-provenance-graphrag-v4-v2m",
        help="new graph version; do not reuse an already published version",
    )
    parser.add_argument("--output", default="medical_knowledge_graph_v4_v2m.jsonl")
    parser.add_argument("--manifest", default="medical_knowledge_graph_v4_v2m_manifest.json")
    return parser.parse_args()


def _jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    args = parse_args()
    runtime_root = Path(__file__).resolve().parents[1]
    workspace_root = runtime_root.parent
    knowledge_dir = workspace_root / "mico_database_new" / "references" / "knowledge" / "medical" / "rag"
    chunks_path = knowledge_dir / args.chunks
    output_path = knowledge_dir / args.output
    manifest_path = knowledge_dir / args.manifest
    if not chunks_path.exists():
        raise SystemExit(f"CHUNK_ASSET_NOT_FOUND: {chunks_path}")
    if output_path.exists() or manifest_path.exists():
        raise SystemExit("GRAPH_OUTPUT_EXISTS: choose a new output/manifest name")
    rows = list(_jsonl(chunks_path))
    if not rows:
        raise SystemExit("FULLTEXT_CORPUS_EMPTY")
    result = build_versioned_graph(
        rows,
        args.graph_version,
        input_asset=chunks_path.name,
    )
    save_graph_build(result, output_path, manifest_path)
    print(json.dumps(result.manifest.model_dump(mode="json"), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
