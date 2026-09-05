#!/usr/bin/env python3
"""Build the Gemini Embedding 2 index for the checked-in full-text corpus.

The dense index is paper-level: one vector per full-text PMCID. Retrieval
maps the ranked paper back to the existing chunk and provenance indexes, so
the final evidence still has an exact chunk-level source. This keeps the
initial 65-paper build bounded while preserving the GraphRAG evidence edge.

The API key is read only from the process environment by the Runtime adapter.
This script never writes credentials, request text, or provider responses to
logs. It writes only bounded local vector records and provenance metadata.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
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


INDEX_VERSION = "fulltext-gemini-paper-embedding-v1"
MODEL_NAME = "gemini-embedding-2"
MAX_DOCUMENT_CHARS = 12_000
MAX_REQUEST_ATTEMPTS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Gemini Embedding 2 paper index")
    parser.add_argument("--chunks", default="references/knowledge/medical/rag/medical_chunks.jsonl")
    parser.add_argument("--out-dir", default="references/knowledge/medical/rag")
    parser.add_argument("--limit", type=int, default=0, help="Optional bounded paper smoke-build limit")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def load_chunks(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if value.get("pmcid") and value.get("chunkId") and value.get("text"):
                    rows.append(value)
    return rows


def group_papers(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for row in rows:
        paper_id = str(row["pmcid"])
        paper = grouped.setdefault(paper_id, {
            "paperId": paper_id,
            "pmcid": paper_id,
            "pmid": row.get("pmid", ""),
            "topic": row.get("topic", "unknown"),
            "title": row.get("title", ""),
            "year": row.get("year", ""),
            "journal": row.get("journal", ""),
            "doi": row.get("doi", ""),
            "sourceUrl": row.get("sourceUrl", ""),
            "chunkIds": [],
            "texts": [],
        })
        paper["chunkIds"].append(str(row["chunkId"]))
        paper["texts"].append(str(row["text"]))
    return list(grouped.values())


def document_text(paper: dict[str, Any]) -> str:
    title = str(paper.get("title") or "none")
    content = "\n\n".join(str(value) for value in paper["texts"])
    return f"{title}\n\n{content}"[:MAX_DOCUMENT_CHARS]


def load_existing(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        return {}
    values: dict[str, list[float]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("indexVersion") == INDEX_VERSION and row.get("paperId"):
                values[str(row["paperId"])] = [float(item) for item in row["embedding"]]
    return values


def corpus_hash(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row["chunkId"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row["pmcid"]).encode("utf-8"))
        digest.update(b"\n")
    return "sha256:" + digest.hexdigest()


def write_index_files(
    out_dir: Path,
    selected: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
    vectors: dict[str, list[float]],
    complete_corpus: bool,
) -> int:
    available = [paper for paper in selected if str(paper["paperId"]) in vectors]
    if not available:
        return 0
    dimension = len(vectors[str(available[0]["paperId"])])
    output_path = out_dir / "medical_gemini_paper_embedding_index.jsonl"
    temp_path = output_path.with_suffix(".jsonl.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for paper in available:
            paper_id = str(paper["paperId"])
            handle.write(json.dumps({
                "indexVersion": INDEX_VERSION,
                "model": MODEL_NAME,
                "granularity": "paper",
                "evidenceTier": "fulltext",
                "paperId": paper_id,
                "pmcid": paper["pmcid"],
                "pmid": paper.get("pmid", ""),
                "topic": paper.get("topic", "unknown"),
                "title": paper.get("title", ""),
                "year": paper.get("year", ""),
                "journal": paper.get("journal", ""),
                "doi": paper.get("doi", ""),
                "sourceUrl": paper.get("sourceUrl", ""),
                "chunkCount": len(paper["chunkIds"]),
                "embedding": vectors[paper_id],
            }, ensure_ascii=False) + "\n")
    temp_path.replace(output_path)
    meta = {
        "indexVersion": INDEX_VERSION,
        "model": MODEL_NAME,
        "granularity": "paper",
        "evidenceTier": "fulltext",
        "documentCount": len(available),
        "chunkCount": sum(len(paper["chunkIds"]) for paper in selected),
        "dimension": dimension,
        "corpusHash": corpus_hash(all_rows),
        "documentInputFormat": "title: {title} | text: {fulltext_content_truncated_to_12000_chars}",
        "queryInputFormat": "task: search result | query: {content}",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "completeCorpus": complete_corpus and len(available) == len(selected),
    }
    (out_dir / "medical_gemini_paper_embedding_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return dimension


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    chunks_path = (root / args.chunks).resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_chunks(chunks_path)
    papers = group_papers(rows)
    if not papers:
        raise SystemExit("FULLTEXT_CORPUS_EMPTY")
    if args.limit < 0:
        raise SystemExit("EMBEDDING_LIMIT_INVALID")
    if args.workers < 1 or args.workers > 16:
        raise SystemExit("EMBEDDING_WORKERS_INVALID")
    selected = papers[:args.limit] if args.limit else papers
    embedder = GeminiEmbeddingPort.from_environment()
    output_path = out_dir / "medical_gemini_paper_embedding_index.jsonl"
    existing = load_existing(output_path) if args.resume else {}
    vectors: dict[str, list[float]] = dict(existing)
    pending = [paper for paper in selected if str(paper["paperId"]) not in vectors]

    def embed_paper(paper: dict[str, Any]) -> tuple[str, list[float]]:
        for attempt in range(MAX_REQUEST_ATTEMPTS):
            try:
                return str(paper["paperId"]), embedder.embed_document(
                    str(paper.get("title") or "none"),
                    document_text(paper),
                )
            except EmbeddingRequestError as exc:
                if str(exc) == "GEMINI_EMBEDDING_QUOTA_EXHAUSTED":
                    # A quota response is not transient.  Retrying here would
                    # multiply the provider charge and obscure the real gate.
                    raise
                if attempt == MAX_REQUEST_ATTEMPTS - 1:
                    raise
                time.sleep(2 ** attempt)
        raise AssertionError("unreachable")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(embed_paper, paper) for paper in pending]
        for index, future in enumerate(as_completed(futures), start=1):
            paper_id, vector = future.result()
            vectors[paper_id] = vector
            write_index_files(
                out_dir,
                selected,
                rows,
                vectors,
                complete_corpus=len(selected) == len(papers),
            )
            print(f"embedded={index}/{len(futures)}", flush=True)

    by_id = {str(paper["paperId"]): paper for paper in selected}
    missing = [paper_id for paper_id in by_id if paper_id not in vectors]
    if missing:
        raise SystemExit("EMBEDDING_INDEX_INCOMPLETE")
    dimension = write_index_files(
        out_dir,
        selected,
        rows,
        vectors,
        complete_corpus=len(selected) == len(papers),
    )
    meta = json.loads((out_dir / "medical_gemini_paper_embedding_meta.json").read_text(encoding="utf-8"))
    print("MEDICAL_GEMINI_EMBEDDING_INDEX_BUILD_DONE", flush=True)
    print(f"papers={len(selected)}", flush=True)
    print(f"chunks={meta['chunkCount']}", flush=True)
    print(f"dimension={dimension}", flush=True)
    print(f"model={meta['model']}", flush=True)
    print(f"complete_corpus={meta['completeCorpus']}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (EmbeddingConfigurationError, EmbeddingRequestError) as exc:
        # Never emit provider, transport, credential, request, or document
        # details from a CLI failure.
        print(str(exc))
        raise SystemExit(2) from None
