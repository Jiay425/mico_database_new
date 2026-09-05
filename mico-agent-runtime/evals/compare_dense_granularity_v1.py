"""Compare paper-level and chunk-level Dense rankings on query-set-v1.

This artifact deliberately reports rankings and coverage diagnostics only.
Without judged qrels it does not compute Recall, nDCG, MRR or precision and
cannot be used for model selection.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from mico_agent_runtime.knowledge.database_retriever import _query_terms
from p2g_partial_diagnostic_v1 import _cache_key, _dense_rank, _load_cache, _read_jsonl


def _dot(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return -1.0
    return sum(a * b for a, b in zip(left, right))


def _paper_rank(rows: list[dict[str, Any]], vector: list[float] | None, limit: int) -> list[dict[str, Any]]:
    if vector is None:
        return []
    scored = []
    for row in rows:
        paper_id = str(row.get("paperId") or row.get("pmcid") or "")
        embedding = row.get("embedding")
        if not paper_id or not isinstance(embedding, list):
            continue
        scored.append((paper_id, _dot(vector, [float(value) for value in embedding])))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return [
        {"paperId": paper_id, "score": round(float(score), 8), "rank": index}
        for index, (paper_id, score) in enumerate(scored[:limit], 1)
    ]


def run(
    query_set: Path,
    chunks_path: Path,
    chunk_embeddings_path: Path,
    paper_embeddings_path: Path,
    query_cache: Path,
    output: Path,
    top_k: int = 20,
) -> dict[str, Any]:
    query_payload = json.loads(query_set.read_text(encoding="utf-8"))
    queries = [item for item in (query_payload.get("items") or []) if isinstance(item, dict)]
    chunks = _read_jsonl(chunks_path)
    chunk_embeddings = _read_jsonl(chunk_embeddings_path)
    paper_embeddings = _read_jsonl(paper_embeddings_path)
    chunk_by_id = {
        str(row.get("chunkId")): row for row in chunks if row.get("chunkId")
    }
    chunks_by_paper: defaultdict[str, list[str]] = defaultdict(list)
    for row in chunks:
        chunk_id = str(row.get("chunkId") or "")
        paper_id = str(row.get("pmcid") or chunk_id.split("-", 1)[0])
        if chunk_id:
            chunks_by_paper[paper_id].append(chunk_id)
    chunk_vector_rows = {
        str(row.get("chunkId")): row for row in chunk_embeddings if row.get("chunkId")
    }
    chunk_rows = [
        {**chunk_by_id[chunk_id], "embedding": row.get("embedding")}
        for chunk_id, row in chunk_vector_rows.items()
        if chunk_id in chunk_by_id and isinstance(row.get("embedding"), list)
    ]
    model, cached = _load_cache(query_cache)
    cases: list[dict[str, Any]] = []
    missing: list[str] = []
    for item in queries:
        question = str(item.get("question") or "")
        terms = _query_terms(question)
        query_text = " ".join(terms)
        vector = cached.get(_cache_key(model, query_text))
        if vector is None:
            missing.append(str(item.get("queryId") or ""))
        chunk_top = _dense_rank(chunk_rows, chunk_vector_rows, vector)[:top_k]
        paper_top = _paper_rank(paper_embeddings, vector, top_k)
        mapped = []
        for row in paper_top:
            paper_id = row["paperId"]
            mapped.append({
                "paperId": paper_id,
                "chunkIds": chunks_by_paper.get(paper_id, [])[:4],
            })
        cases.append({
            "queryId": item.get("queryId"),
            "category": item.get("category"),
            "question": question,
            "denseQueryVectorCached": vector is not None,
            "chunkDenseTop20": chunk_top,
            "paperDenseTop20": paper_top,
            "paperToChunkPreview": mapped,
        })
    report = {
        "reportVersion": "p2g-dense-granularity-diagnostic-v1",
        "status": "DIAGNOSTIC_NO_QRELS" if not missing else "INCOMPLETE_QUERY_VECTORS",
        "evaluationBoundary": {
            "querySet": query_set.name,
            "querySetVersion": "query-set-v1",
            "chunkCorpus": "chunk-v2-medium",
            "chunkCount": len(chunk_rows),
            "paperCount": len(paper_embeddings),
            "topK": top_k,
            "qrelsStatus": "not_built",
            "metrics": "not computed without relevance judgments",
        },
        "assets": {
            "querySet": str(query_set),
            "chunks": str(chunks_path),
            "chunkEmbeddings": str(chunk_embeddings_path),
            "paperEmbeddings": str(paper_embeddings_path),
            "queryCache": str(query_cache),
        },
        "counts": {
            "queryCount": len(queries),
            "denseQueryVectors": len(queries) - len(missing),
            "missingDenseQueryVectors": len(missing),
        },
        "missingDenseQueryIds": missing,
        "cases": cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-set", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--chunk-embeddings", type=Path, required=True)
    parser.add_argument("--paper-embeddings", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()
    result = run(
        args.query_set,
        args.chunks,
        args.chunk_embeddings,
        args.paper_embeddings,
        args.query_cache,
        args.output,
        args.top_k,
    )
    print(json.dumps({"status": result["status"], "counts": result["counts"], "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
