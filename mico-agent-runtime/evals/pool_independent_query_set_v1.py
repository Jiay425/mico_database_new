"""Build a provider-free Top-20 candidate pool for query-set-v1.

This is the handoff between frozen questions and relevance judging.  Dense,
sparse, graph and standard-RRF branches all operate on the same chunk-v2
medium corpus and are retained with their ranks/scores.  The graph branch is
explicitly marked as a local provenance proxy when the reviewed graph build
is still staging; it never invents evidence or labels.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from mico_agent_runtime.knowledge.database_retriever import _query_terms
from p2g_partial_diagnostic_v1 import (
    _cache_key,
    _dense_rank,
    _graph_rank,
    _load_cache,
    _read_jsonl,
    _rrf_rank,
    _sparse_rank,
)


def _load_query_set(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items") or []
    if not isinstance(items, list) or not items:
        raise ValueError("QUERY_SET_EMPTY")
    return [item for item in items if isinstance(item, dict) and str(item.get("question") or "").strip()]


def run(
    query_set: Path,
    chunks_path: Path,
    embeddings_path: Path,
    graph_path: Path,
    query_cache: Path,
    output: Path,
    top_k: int = 20,
    pool_branches: tuple[str, ...] = ("dense", "sparse", "graph", "rrf_standard"),
) -> dict[str, Any]:
    if top_k <= 0 or top_k > 100:
        raise ValueError("TOP_K_INVALID")
    if not pool_branches or any(branch not in {"dense", "sparse", "graph", "rrf_standard"} for branch in pool_branches):
        raise ValueError("POOL_BRANCHES_INVALID")
    queries = _load_query_set(query_set)
    chunks = _read_jsonl(chunks_path)
    embeddings = _read_jsonl(embeddings_path)
    graph_rows = _read_jsonl(graph_path)
    embedding_by_id = {
        str(row["chunkId"]): [float(value) for value in row["embedding"]]
        for row in embeddings
        if isinstance(row.get("embedding"), list)
    }
    model, cached = _load_cache(query_cache)
    chunk_ids = {str(row.get("chunkId")) for row in chunks}
    cases: list[dict[str, Any]] = []
    branch_counts: Counter[str] = Counter()
    missing_vectors: list[str] = []
    for item in queries:
        question = str(item["question"])
        terms = _query_terms(question)
        query_text = " ".join(terms)
        query_vector = cached.get(_cache_key(model, query_text))
        if query_vector is None:
            missing_vectors.append(str(item.get("queryId") or ""))
        branches = {
            "dense": _dense_rank(chunks, embedding_by_id, query_vector)[:top_k],
            "sparse": _sparse_rank(chunks, terms)[:top_k],
            "graph": _graph_rank(chunks, graph_rows, terms)[:top_k],
        }
        branches["rrf_standard"] = _rrf_rank(
            {name: rows for name, rows in branches.items()},
            (1.0, 1.0, 1.0),
            equal_weight=True,
        )[:top_k]
        memberships: dict[str, dict[str, Any]] = {}
        for branch, rows in branches.items():
            branch_counts[branch] += int(bool(rows))
            if branch not in pool_branches:
                continue
            for row in rows:
                chunk_id = str(row.get("chunkId") or "")
                if not chunk_id:
                    continue
                entry = memberships.setdefault(chunk_id, {"chunkId": chunk_id, "branches": {}})
                entry["branches"][branch] = {
                    "rank": int(row.get("rank") or 0),
                    "score": float(row.get("score") or 0.0),
                }
        cases.append({
            "queryId": item.get("queryId"),
            "category": item.get("category"),
            "question": question,
            "queryTerms": terms,
            "denseQueryVectorCached": query_vector is not None,
            "branches": branches,
            "candidatePool": sorted(
                memberships.values(),
                key=lambda row: (-len(row["branches"]), row["chunkId"]),
            ),
            "candidatePoolSize": len(memberships),
        })
    report = {
        "reportVersion": "p2g-independent-query-pool-v1",
        "status": "READY_FOR_BLIND_JUDGING" if not missing_vectors else "INCOMPLETE_QUERY_VECTORS",
        "evaluationBoundary": {
            "querySet": query_set.name,
            "querySetVersion": "query-set-v1",
            "corpus": "chunk-v2-medium",
            "chunkCount": len(chunks),
            "embeddingCount": len(embedding_by_id),
            "topKPerBranch": top_k,
            "branches": ["dense", "sparse", "graph", "rrf_standard"],
            "candidatePoolBranches": list(pool_branches),
            "graphImplementation": "local provenance proxy over reviewed r4 staging JSONL; not Neo4j publication",
            "qrelsStatus": "not_built",
            "humanJudgingRequired": True,
        },
        "assets": {
            "querySet": str(query_set),
            "chunks": str(chunks_path),
            "embeddings": str(embeddings_path),
            "graph": str(graph_path),
            "queryCache": str(query_cache),
        },
        "counts": {
            "queryCount": len(queries),
            "denseQueryVectors": len(queries) - len(missing_vectors),
            "missingDenseQueryVectors": len(missing_vectors),
            "branchWithResults": dict(branch_counts),
            "candidatePoolTotal": sum(case["candidatePoolSize"] for case in cases),
        },
        "missingDenseQueryIds": missing_vectors,
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
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--pool-branches", default="dense,sparse,graph,rrf_standard")
    args = parser.parse_args()
    report = run(
        args.query_set,
        args.chunks,
        args.embeddings,
        args.graph,
        args.query_cache,
        args.output,
        args.top_k,
        tuple(branch.strip() for branch in args.pool_branches.split(",") if branch.strip()),
    )
    print(json.dumps({
        "status": report["status"],
        "counts": report["counts"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
