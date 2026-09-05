"""Build a resumable Gemini query-embedding cache for query-set-v1.

The cache is intentionally separate from document embeddings and from the
18-case smoke cache.  It stores only model-scoped hashes and normalized
vectors through :class:`QueryEmbeddingCache`; raw questions never enter the
cache file.  A quota error stops the batch cleanly so a later ``--resume``
continues from the first missing query without repeating successful calls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mico_agent_runtime.knowledge.embeddings import (
    EmbeddingRequestError,
    GeminiEmbeddingPort,
    QueryEmbeddingCache,
)


def _load_questions(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items") or []
    if not isinstance(items, list):
        raise ValueError("QUERY_SET_ITEMS_INVALID")
    rows: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict) or not str(item.get("question") or "").strip():
            raise ValueError("QUERY_SET_QUESTION_MISSING")
        rows.append(item)
    return rows


def run(
    query_set: Path,
    output: Path,
    limit: int | None = None,
    execute: bool = False,
) -> dict[str, Any]:
    questions = _load_questions(query_set)
    selected = questions[:limit] if limit is not None else questions
    model = "gemini-embedding-2"
    cache = QueryEmbeddingCache(model, output, dimension=3072)
    pending: list[tuple[dict[str, Any], str]] = []
    for item in selected:
        # Dense retrieval must embed the user's complete question.  Stopword
        # removal and lexical alias expansion belong to BM25 only; applying
        # them here loses relation, negation, and synthesis intent.
        query_text = str(item["question"]).strip()
        if not query_text:
            raise ValueError(f"QUERY_TEXT_EMPTY:{item.get('queryId')}")
        key = cache._key(query_text)  # cache key is hash-only and never persisted as text
        if key not in cache._values:
            pending.append((item, query_text))
    if not execute:
        return {
            "status": "DRY_RUN",
            "querySet": str(query_set),
            "selected": len(selected),
            "pending": len(pending),
            "cached": len(selected) - len(pending),
            "cache": str(output),
            "model": model,
        }

    port = GeminiEmbeddingPort.from_environment()
    completed = 0
    quota_error: str | None = None
    try:
        for item, query_text in pending:
            try:
                cache.get_or_compute(query_text, lambda text=query_text: port.embed_query(text))
                completed += 1
                print(f"embedded={completed}/{len(pending)} queryId={item.get('queryId')}", flush=True)
            except EmbeddingRequestError as exc:
                quota_error = str(exc)
                print(quota_error, flush=True)
                break
    finally:
        port.close()
    return {
        "status": "QUOTA_EXHAUSTED" if quota_error else "COMPLETED",
        "querySet": str(query_set),
        "selected": len(selected),
        "pendingBeforeRun": len(pending),
        "embeddedThisRun": completed,
        "cachedAfterRun": len(cache._values),
        "remainingSelected": len(pending) - completed,
        "cache": str(output),
        "model": model,
        "dimension": 3072,
        "error": quota_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-set", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    result = run(args.query_set, args.output, args.limit, args.execute)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
