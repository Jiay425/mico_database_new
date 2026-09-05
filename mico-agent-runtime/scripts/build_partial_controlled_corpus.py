#!/usr/bin/env python3
"""Create a fair, provider-free corpus containing only embedded chunks.

The resulting assets are for development diagnostics.  Dense, sparse, and
graph evaluators can all point at the same ``chunkIds`` universe while the
main medium corpus is still missing vectors.  Missing chunks are excluded,
not relabeled as irrelevant.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def _sha256_ids(ids: list[str]) -> str:
    digest = hashlib.sha256()
    for value in ids:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return "sha256:" + digest.hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(path)


def build(
    chunks_path: Path,
    embeddings_path: Path,
    output_dir: Path,
    graph_path: Path | None = None,
) -> dict[str, Any]:
    chunks = _read_jsonl(chunks_path)
    embedding_rows = _read_jsonl(embeddings_path)
    chunk_by_id = {str(row.get("chunkId")): row for row in chunks if row.get("chunkId")}
    embedding_by_id = {
        str(row.get("chunkId")): row
        for row in embedding_rows
        if row.get("chunkId") and str(row.get("chunkId")) in chunk_by_id
    }
    selected_ids = [str(row["chunkId"]) for row in chunks if str(row.get("chunkId")) in embedding_by_id]
    selected = [chunk_by_id[chunk_id] for chunk_id in selected_ids]
    selected_embeddings = [embedding_by_id[chunk_id] for chunk_id in selected_ids]
    if not selected:
        raise ValueError("CONTROLLED_CORPUS_EMPTY")

    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_output = output_dir / "medical_chunks_v2_medium_partial1001.jsonl"
    embedding_output = output_dir / "medical_gemini_chunk_embedding_index_v2_medium_partial1001.jsonl"
    _write_jsonl(chunk_output, selected)
    _write_jsonl(embedding_output, selected_embeddings)

    graph_output = None
    graph_stats: dict[str, int] = {}
    if graph_path is not None:
        graph_rows = _read_jsonl(graph_path)
        kept_edges = [
            row for row in graph_rows
            if row.get("recordType") == "edge"
            and str(row.get("evidenceChunkId") or "") in set(selected_ids)
        ]
        referenced_nodes = {
            str(node_id)
            for edge in kept_edges
            for node_id in (edge.get("source"), edge.get("target"))
            if node_id
        }
        kept_nodes = [
            row for row in graph_rows
            if row.get("recordType") == "node" and str(row.get("nodeId")) in referenced_nodes
        ]
        graph_rows_out = kept_nodes + kept_edges
        graph_output = output_dir / f"{graph_path.stem}_partial1001.jsonl"
        _write_jsonl(graph_output, graph_rows_out)
        graph_stats = {
            "inputRecords": len(graph_rows),
            "inputNodes": sum(row.get("recordType") == "node" for row in graph_rows),
            "inputEdges": sum(row.get("recordType") == "edge" for row in graph_rows),
            "outputNodes": len(kept_nodes),
            "outputEdges": len(kept_edges),
        }

    manifest = {
        "manifestVersion": "p2g-controlled-corpus-v1",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "status": "development_diagnostic_only",
        "corpusId": "dense-partial-corpus-v1",
        "source": {
            "chunks": str(chunks_path),
            "embeddings": str(embeddings_path),
            "graph": str(graph_path) if graph_path else None,
        },
        "selection": {
            "rule": "chunkId has a valid Gemini embedding record",
            "totalChunks": len(chunks),
            "selectedChunks": len(selected),
            "excludedChunks": len(chunks) - len(selected),
            "coverage": round(len(selected) / len(chunks), 8) if chunks else 0.0,
            "chunkIdHash": _sha256_ids(selected_ids),
        },
        "assets": {
            "chunks": str(chunk_output),
            "embeddings": str(embedding_output),
            "graph": str(graph_output) if graph_output else None,
        },
        "graph": graph_stats,
        "fairnessContract": {
            "denseSparseGraphShareExactChunkUniverse": True,
            "missingIsNotIrrelevant": True,
            "notForFinalModelSelection": True,
        },
    }
    manifest_path = output_dir / "p2g-controlled-corpus-v1.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args.chunks, args.embeddings, args.output_dir, args.graph)
    print(json.dumps({
        "status": "COMPLETED",
        "corpusId": manifest["corpusId"],
        "selectedChunks": manifest["selection"]["selectedChunks"],
        "excludedChunks": manifest["selection"]["excludedChunks"],
        "coverage": manifest["selection"]["coverage"],
        "graph": manifest["graph"],
        "manifest": str(args.output_dir / "p2g-controlled-corpus-v1.json"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
