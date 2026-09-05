#!/usr/bin/env python3
"""Audit coverage and bias of a partial chunk-level embedding index.

The report is intentionally provider-free.  It answers whether a resumable
embedding run has produced an approximately representative corpus or whether
the missing vectors are concentrated by paper, topic, section, or chunk type.
It never treats a missing vector as a negative relevance judgment.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _rows(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    values.append(value)
    return values


def _coverage(rows: list[dict[str, Any]], embedded: set[str], field: str) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "embedded": 0})
    for row in rows:
        key = str(row.get(field) or "(empty)")
        grouped[key]["total"] += 1
        if str(row.get("chunkId") or "") in embedded:
            grouped[key]["embedded"] += 1
    result: list[dict[str, Any]] = []
    for key, values in grouped.items():
        total = values["total"]
        count = values["embedded"]
        result.append({
            field: key,
            "total": total,
            "embedded": count,
            "missing": total - count,
            "coverage": round(count / total, 8) if total else 0.0,
        })
    return sorted(result, key=lambda item: (-item["missing"], item[field]))


def build_report(chunks_path: Path, embeddings_path: Path) -> dict[str, Any]:
    chunks = _rows(chunks_path)
    embeddings = _rows(embeddings_path)
    chunk_ids = {str(row.get("chunkId") or "") for row in chunks if row.get("chunkId")}
    embedded_ids = {
        str(row.get("chunkId") or "")
        for row in embeddings
        if row.get("chunkId") and str(row.get("chunkId")) in chunk_ids
    }
    embedding_ids_in_corpus = [
        str(row.get("chunkId"))
        for row in embeddings
        if row.get("chunkId") and str(row.get("chunkId")) in chunk_ids
    ]
    missing_ids = sorted(chunk_ids - embedded_ids)
    total = len(chunk_ids)
    embedded = len(embedded_ids)
    by_paper = _coverage(chunks, embedded_ids, "pmcid")
    by_topic = _coverage(chunks, embedded_ids, "topic")
    by_section = _coverage(chunks, embedded_ids, "section")
    by_type = _coverage(chunks, embedded_ids, "chunkType")
    fully_missing = [row for row in by_paper if row["embedded"] == 0]
    partial = [row for row in by_paper if 0 < row["embedded"] < row["total"]]
    fully_covered = [row for row in by_paper if row["embedded"] == row["total"]]
    paper_coverages = [row["coverage"] for row in by_paper]
    return {
        "reportVersion": "p2g-embedding-coverage-v1",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "input": {
            "chunks": str(chunks_path),
            "embeddings": str(embeddings_path),
            "chunkCount": total,
            "embeddingRecordCount": len(embeddings),
        },
        "summary": {
            "embeddedChunkCount": embedded,
            "missingChunkCount": len(missing_ids),
            "coverage": round(embedded / total, 8) if total else 0.0,
            "duplicateEmbeddingRecords": len(embedding_ids_in_corpus) - len(set(embedding_ids_in_corpus)),
            "fullyMissingPaperCount": len(fully_missing),
            "partialPaperCount": len(partial),
            "fullyCoveredPaperCount": len(fully_covered),
            "paperCoverageMin": min(paper_coverages) if paper_coverages else 0.0,
            "paperCoverageMax": max(paper_coverages) if paper_coverages else 0.0,
        },
        "biasAssessment": {
            "representativeByPaper": not fully_missing and not partial,
            "hasConcentratedMissingPapers": bool(fully_missing or partial),
            "interpretation": (
                "Embedding records are concentrated by corpus order; partial Dense results "
                "must not be compared against full-corpus Sparse/Graph metrics."
                if fully_missing or partial
                else "No paper-level concentration detected; partial results remain diagnostic only."
            ),
        },
        "byPaper": by_paper,
        "byTopic": by_topic,
        "bySection": by_section,
        "byChunkType": by_type,
        "missingChunkIds": missing_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.chunks, args.embeddings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "COMPLETED",
        "embedded": report["summary"]["embeddedChunkCount"],
        "missing": report["summary"]["missingChunkCount"],
        "coverage": report["summary"]["coverage"],
        "fullyMissingPaperCount": report["summary"]["fullyMissingPaperCount"],
        "partialPaperCount": report["summary"]["partialPaperCount"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
