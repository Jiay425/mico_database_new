#!/usr/bin/env python3
"""Build a versioned Gemini index with one vector per chunk-v2 record.

Safety contract:
  * default mode is a provider-free dry-run;
  * ``--execute`` is required for network/API calls;
  * output names contain the chunk version and variant, so paper-v1 and
    chunk-v2 assets cannot overwrite one another;
  * ``--resume`` only reuses vectors whose input fingerprint is unchanged.

This script is deliberately separate from the historical paper-level builder.
It is not run as part of chunk generation and it must not be used until a
variant has passed validation and received an approved evaluation budget.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mico_agent_runtime.knowledge.embeddings import (
    EmbeddingConfigurationError,
    EmbeddingRequestError,
    GeminiEmbeddingPort,
)


MODEL_NAME = "gemini-embedding-2"
CHUNK_VERSION = "chunk-v2"
MAX_REQUEST_ATTEMPTS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build chunk-level Gemini index")
    parser.add_argument("--chunks", required=True, help="validated chunk-v2 JSONL")
    parser.add_argument("--variant", required=True, choices=("small", "medium", "large"))
    parser.add_argument("--out-dir", default="references/knowledge/medical/rag")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--execute", action="store_true",
        help="actually call Gemini; without this flag the command is a provider-free dry-run",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("chunkId") and row.get("text"):
                    rows.append(row)
    return rows


def input_fingerprint(row: dict[str, Any]) -> str:
    payload = {
        "chunkVersion": row.get("chunkVersion"),
        "variant": row.get("variant"),
        "chunkId": row.get("chunkId"),
        "embeddingText": row.get("embeddingText") or row.get("text"),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def corpus_hash(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row.get("chunkId", "")).encode("utf-8"))
        digest.update(b"\0")
        digest.update(input_fingerprint(row).encode("ascii"))
        digest.update(b"\n")
    return "sha256:" + digest.hexdigest()


def output_paths(out_dir: Path, variant: str) -> tuple[Path, Path]:
    prefix = f"medical_gemini_chunk_embedding_index_v2_{variant}"
    return out_dir / f"{prefix}.jsonl", out_dir / f"{prefix}_meta.json"


def load_existing(path: Path, variant: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    values: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("chunkVersion") == CHUNK_VERSION and row.get("variant") == variant and row.get("chunkId"):
                values[str(row["chunkId"])] = row
    return values


def write_files(
    output_path: Path,
    meta_path: Path,
    rows: list[dict[str, Any]],
    vectors: dict[str, dict[str, Any]],
    source_path: Path,
    complete: bool,
) -> int:
    available = [row for row in rows if str(row["chunkId"]) in vectors]
    if not available:
        return 0
    dimension = len(vectors[str(available[0]["chunkId"])]["embedding"])
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for source in available:
            chunk_id = str(source["chunkId"])
            record = vectors[chunk_id]
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temp_path.replace(output_path)
    meta = {
        "indexVersion": f"fulltext-gemini-{CHUNK_VERSION}-{rows[0].get('variant', 'unknown')}-v1",
        "model": MODEL_NAME,
        "granularity": "chunk",
        "chunkVersion": CHUNK_VERSION,
        "variant": rows[0].get("variant"),
        "chunkCount": len(available),
        "dimension": dimension,
        "corpusHash": corpus_hash(rows),
        "inputAsset": str(source_path),
        "inputText": "embeddingText (title/topic/section/subsection + source chunk)",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "completeCorpus": bool(complete and len(available) == len(rows)),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return dimension


def main() -> None:
    args = parse_args()
    if args.limit < 0:
        raise SystemExit("EMBEDDING_LIMIT_INVALID")
    if not 1 <= args.workers <= 4:
        raise SystemExit("EMBEDDING_WORKERS_INVALID: max 4 to protect quota")
    root = Path(__file__).resolve().parents[1]
    chunks_path = (root / args.chunks).resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(chunks_path)
    if not rows:
        raise SystemExit("CHUNK_V2_CORPUS_EMPTY")
    invalid = [row for row in rows if row.get("chunkVersion") != CHUNK_VERSION or row.get("variant") != args.variant]
    if invalid:
        raise SystemExit("CHUNK_V2_VARIANT_MISMATCH")
    selected = rows[:args.limit] if args.limit else rows
    output_path, meta_path = output_paths(out_dir, args.variant)
    existing = load_existing(output_path, args.variant) if args.resume else {}
    # Never reuse an old vector if the input text changed.
    row_by_id = {str(row["chunkId"]): row for row in rows}
    existing = {
        key: value for key, value in existing.items()
        if key in row_by_id
        and value.get("inputFingerprint") == input_fingerprint(row_by_id[key])
    }
    pending = [row for row in selected if str(row["chunkId"]) not in existing]
    if not args.execute:
        print(json.dumps({
            "mode": "dry-run",
            "providerCalls": 0,
            "variant": args.variant,
            "input": str(chunks_path),
            "output": str(output_path),
            "selectedChunks": len(selected),
            "resumableVectors": len(existing),
            "pendingVectors": len(pending),
            "estimatedProviderCalls": len(pending),
            "corpusHash": corpus_hash(rows),
        }, ensure_ascii=False, indent=2))
        return

    embedder = GeminiEmbeddingPort.from_environment()
    vectors = dict(existing)

    def embed(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        for attempt in range(MAX_REQUEST_ATTEMPTS):
            try:
                text = str(row.get("embeddingText") or row["text"])
                vector = embedder.embed_document(str(row.get("title") or "none"), text)
                return str(row["chunkId"]), {
                    "indexVersion": f"fulltext-gemini-{CHUNK_VERSION}-{args.variant}-v1",
                    "model": MODEL_NAME,
                    "granularity": "chunk",
                    "chunkVersion": CHUNK_VERSION,
                    "variant": args.variant,
                    "chunkId": str(row["chunkId"]),
                    "pmcid": row.get("pmcid", ""),
                    "section": row.get("section", ""),
                    "sourceUrl": row.get("sourceUrl", ""),
                    "inputFingerprint": input_fingerprint(row),
                    "embedding": vector,
                }
            except EmbeddingRequestError as exc:
                if str(exc) == "GEMINI_EMBEDDING_QUOTA_EXHAUSTED":
                    raise
                if attempt == MAX_REQUEST_ATTEMPTS - 1:
                    raise
                time.sleep(2 ** attempt)
        raise AssertionError("unreachable")

    try:
        if args.workers == 1:
            # Do not enqueue the whole corpus when quota protection matters:
            # a non-transient quota error stops before another request starts.
            for index, row in enumerate(pending, start=1):
                chunk_id, record = embed(row)
                vectors[chunk_id] = record
                write_files(output_path, meta_path, selected, vectors, chunks_path, len(selected) == len(rows))
                print(f"embedded={index}/{len(pending)}", flush=True)
        else:
            # Parallel mode is still bounded to the selected process count;
            # callers that need the strictest quota behavior should use the
            # default single worker.
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(embed, row) for row in pending]
                for index, future in enumerate(as_completed(futures), start=1):
                    chunk_id, record = future.result()
                    vectors[chunk_id] = record
                    write_files(output_path, meta_path, selected, vectors, chunks_path, len(selected) == len(rows))
                    print(f"embedded={index}/{len(futures)}", flush=True)
    finally:
        embedder.close()
    dimension = write_files(output_path, meta_path, selected, vectors, chunks_path, len(selected) == len(rows))
    if len(vectors) < len(selected):
        raise SystemExit("EMBEDDING_INDEX_INCOMPLETE")
    print("MEDICAL_GEMINI_CHUNK_EMBEDDING_INDEX_BUILD_DONE")
    print(f"chunks={len(selected)}")
    print(f"dimension={dimension}")
    print(f"output={output_path}")


if __name__ == "__main__":
    try:
        main()
    except (EmbeddingConfigurationError, EmbeddingRequestError) as exc:
        print(str(exc))
        raise SystemExit(2) from None
