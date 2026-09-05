"""Build the pre-Cross-Encoder candidate pool for Final Hybrid v2.

Global questions use Dense Top-30, PostgreSQL-style lexical Top-30 and
query-specific 1--3 hop graph-path Top-30.  Standard RRF merges ranks only and
emits a Top-40 candidate set for a later Cross-Encoder.  This script never
uses hidden source-PMCID metadata to narrow an otherwise global evaluation.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from graph_path_evidence_ranker_v2 import rank_path_evidence_graph
from mico_agent_runtime.knowledge.database_retriever import _query_terms, _query_type
from p2g_partial_diagnostic_v1 import _cache_key, _dense_rank, _load_cache, _read_jsonl, _rrf_rank, _sparse_rank


def _queries(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("items") or []
    if not isinstance(rows, list) or not rows:
        raise ValueError("QUERY_SET_EMPTY")
    return [row for row in rows if isinstance(row, dict) and str(row.get("question") or "").strip()]


def run(
    query_set: Path,
    chunks_path: Path,
    embeddings_path: Path,
    graph_path: Path,
    query_cache: Path,
    output: Path,
) -> dict[str, Any]:
    queries = _queries(query_set)
    chunks = _read_jsonl(chunks_path)
    graph_rows = _read_jsonl(graph_path)
    embeddings = _read_jsonl(embeddings_path)
    embedding_by_id = {
        str(row["chunkId"]): [float(value) for value in row["embedding"]]
        for row in embeddings if isinstance(row.get("embedding"), list)
    }
    model, cached = _load_cache(query_cache)
    missing: list[str] = []
    types: Counter[str] = Counter()
    cases: list[dict[str, Any]] = []
    for item in queries:
        question = str(item["question"])
        query_type = _query_type(question)
        terms = _query_terms(question)
        # Dense consumes the complete natural-language question.  Terms are
        # retained below for BM25 and graph-side entity/intent processing.
        vector = cached.get(_cache_key(model, question))
        if vector is None:
            missing.append(str(item.get("queryId") or ""))
        branches = {
            "dense_top30": _dense_rank(chunks, embedding_by_id, vector)[:30],
            "bm25_top30": _sparse_rank(chunks, terms)[:30],
            "graph_path_top30": rank_path_evidence_graph(graph_rows, question, query_type, limit=30),
        }
        branches["rrf_top40"] = _rrf_rank(
            {
                "dense": branches["dense_top30"],
                "sparse": branches["bm25_top30"],
                "graph": branches["graph_path_top30"],
            },
            (1.0, 1.0, 1.0),
            equal_weight=True,
        )[:40]
        membership: dict[str, dict[str, Any]] = {}
        for branch, rows in branches.items():
            for row in rows:
                chunk_id = str(row.get("chunkId") or "")
                if not chunk_id:
                    continue
                membership.setdefault(chunk_id, {"chunkId": chunk_id, "branches": {}})["branches"][branch] = {
                    "rank": int(row.get("rank") or 0),
                    "score": float(row.get("score") or 0.0),
                }
        types[query_type] += 1
        cases.append({
            "queryId": item.get("queryId"),
            "category": item.get("category"),
            "question": question,
            "queryTerms": terms,
            "retrievalScope": {"scopeType": "global", "allowedDocumentIds": []},
            "runtimeRouterType": query_type,
            "denseQueryVectorCached": vector is not None,
            "branches": branches,
            "candidatePool": sorted(membership.values(), key=lambda row: (-len(row["branches"]), row["chunkId"])),
            "candidatePoolSize": len(membership),
        })
    report = {
        "reportVersion": "final-hybrid-v2-pre-cross-encoder-v1",
        "status": "READY_FOR_CROSS_ENCODER" if not missing else "INCOMPLETE_QUERY_VECTORS",
        "evaluationBoundary": {
            "scope": "global_only",
            "corpus": "chunk-v2-medium",
            "dense": "Top30",
            "bm25": "Top30",
            "graph": "query-specific accepted semantic 1-3 hop paths, Top30",
            "fusion": "rrf-standard-v1 candidate merger",
            "rerankInput": "rrf_top40",
            "crossEncoder": "not executed",
            "graphImplementation": "local r4 staging export for evaluation; production Neo4j uses the same scope and path constraints after publication",
        },
        "assets": {"querySet": str(query_set), "chunks": str(chunks_path), "embeddings": str(embeddings_path), "graph": str(graph_path), "queryCache": str(query_cache)},
        "counts": {"queryCount": len(cases), "missingDenseQueryVectors": len(missing), "routerTypes": dict(types), "candidatePoolTotal": sum(case["candidatePoolSize"] for case in cases)},
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
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(
        query_set=args.query_set,
        chunks_path=args.chunks,
        embeddings_path=args.embeddings,
        graph_path=args.graph,
        query_cache=args.query_cache,
        output=args.output,
    )
    print(json.dumps({"status": report["status"], "counts": report["counts"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
