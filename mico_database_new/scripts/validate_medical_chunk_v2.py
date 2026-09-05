#!/usr/bin/env python3
"""Validate versioned medical chunk-v2 assets without touching a provider or DB.

The validator is intentionally independent from the builder.  It checks the
properties that matter before an expensive embedding/graph build: manifest
identity, unique IDs, paper coverage, bounded token windows, offsets and
embedding context headers.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]+|[\u4e00-\u9fff]{1,}")


def token_count(text: str) -> int:
    return len(TOKEN_RE.findall(text.lower()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate medical chunk-v2 JSONL")
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--manifest", required=True)
    return parser.parse_args()


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def main() -> int:
    args = parse_args()
    chunks_path = Path(args.chunks).resolve()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    params = manifest.get("params") or {}
    errors: list[str] = []
    if manifest.get("chunkVersion") != "chunk-v2":
        fail(errors, "manifest.chunkVersion must be chunk-v2")
    variant = str(manifest.get("variant") or "")
    if variant not in {"small", "medium", "large"}:
        fail(errors, f"unsupported variant: {variant}")

    rows: list[dict[str, Any]] = []
    with chunks_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                fail(errors, f"line {line_number}: invalid JSON: {exc}")
                continue
            rows.append(row)

    ids: set[str] = set()
    papers: set[str] = set()
    types: Counter[str] = Counter()
    below_min = 0
    above_max = 0
    missing_header = 0
    bad_offsets = 0
    min_tokens = int(params.get("minTokens", 0))
    max_tokens = int(params.get("maxTokens", 0))
    for index, row in enumerate(rows, start=1):
        chunk_id = str(row.get("chunkId") or "")
        pmcid = str(row.get("pmcid") or "")
        text = str(row.get("text") or "").strip()
        embedding_text = str(row.get("embeddingText") or "")
        if not chunk_id:
            fail(errors, f"row {index}: missing chunkId")
        elif chunk_id in ids:
            fail(errors, f"row {index}: duplicate chunkId {chunk_id}")
        ids.add(chunk_id)
        if not pmcid:
            fail(errors, f"row {index}: missing pmcid")
        papers.add(pmcid)
        if row.get("chunkVersion") != "chunk-v2" or row.get("variant") != variant:
            fail(errors, f"row {index}: version/variant mismatch")
        if not text:
            fail(errors, f"row {index}: empty text")
        measured = token_count(text)
        declared = int(row.get("tokenCount") or 0)
        if declared != measured:
            fail(errors, f"row {index}: tokenCount mismatch ({declared} != {measured})")
        if measured < min_tokens:
            below_min += 1
        if measured > max_tokens:
            above_max += 1
            fail(errors, f"row {index}: token count {measured} > maxTokens {max_tokens}")
        expected_header = (
            f"Title: {row.get('title') or 'none'}\n"
            f"Topic: {row.get('topic') or 'unknown'}\n"
            f"Section: {row.get('section') or '正文'}\n"
            f"Subsection: {row.get('subsection') or 'none'}\n"
        )
        if not embedding_text.startswith(expected_header) or not embedding_text.endswith(text):
            missing_header += 1
            fail(errors, f"row {index}: embeddingText header/body mismatch")
        try:
            start = int(row.get("charStart"))
            end = int(row.get("charEnd"))
            paragraph_start = int(row.get("paragraphIndex"))
            paragraph_end = int(row.get("paragraphEndIndex"))
            sentence_start = int(row.get("sentenceStart"))
            sentence_end = int(row.get("sentenceEnd"))
        except (TypeError, ValueError):
            bad_offsets += 1
            fail(errors, f"row {index}: non-integer offsets")
        else:
            if start < 0 or end < start or paragraph_start < 0 or paragraph_end <= paragraph_start:
                bad_offsets += 1
                fail(errors, f"row {index}: invalid paragraph/character offsets")
            if sentence_start < 0 or sentence_end <= sentence_start:
                bad_offsets += 1
                fail(errors, f"row {index}: invalid sentence offsets")
        types[str(row.get("chunkType") or "unknown")] += 1

    expected_count = int(((manifest.get("summary") or {}).get("chunkCount")) or 0)
    expected_papers = int(((manifest.get("summary") or {}).get("paperCount")) or 0)
    if expected_count != len(rows):
        fail(errors, f"manifest chunkCount {expected_count} != rows {len(rows)}")
    if expected_papers != len(papers):
        fail(errors, f"manifest paperCount {expected_papers} != PMCID count {len(papers)}")
    if not rows:
        fail(errors, "empty chunk asset")

    result = {
        "asset": str(chunks_path),
        "manifest": str(manifest_path),
        "chunkVersion": manifest.get("chunkVersion"),
        "variant": variant,
        "chunks": len(rows),
        "papers": len(papers),
        "chunkTypes": dict(sorted(types.items())),
        "belowMinTokens": below_min,
        "aboveMaxTokens": above_max,
        "missingEmbeddingHeaders": missing_header,
        "badOffsetRows": bad_offsets,
        "errors": errors,
        "valid": not errors,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
