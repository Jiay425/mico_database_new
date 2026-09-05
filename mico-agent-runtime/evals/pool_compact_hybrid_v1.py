"""Build the fixed-query, online-sized hybrid candidate pool.

This intentionally models the serving path rather than a large benchmark
pool: Dense and BM25 each produce eight candidates; the graph branch is only
enabled by the runtime router for relation or multi-hop questions.  Standard
RRF produces at most 15 candidates for content-only listwise reranking.
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


def _membership(branches: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for branch, rows in branches.items():
        for row in rows:
            chunk_id = str(row.get("chunkId") or "")
            if not chunk_id:
                continue
            item = values.setdefault(chunk_id, {"chunkId": chunk_id, "branches": {}})
            item["branches"][branch] = {
                "rank": int(row.get("rank") or 0),
                "score": float(row.get("score") or 0.0),
            }
    return sorted(values.values(), key=lambda row: (-len(row["branches"]), row["chunkId"]))


def run(query_set: Path, chunks_path: Path, embeddings_path: Path, graph_path: Path, query_cache: Path, output: Path) -> dict[str, Any]:
    chunks = _read_jsonl(chunks_path)
    embeddings = _read_jsonl(embeddings_path)
    graph_rows = _read_jsonl(graph_path)
    embedding_by_id = {
        str(row["chunkId"]): [float(value) for value in row["embedding"]]
        for row in embeddings if isinstance(row.get("embedding"), list)
    }
    model, cached = _load_cache(query_cache)
    missing: list[str] = []
    router_types: Counter[str] = Counter()
    graph_enabled = 0
    cases: list[dict[str, Any]] = []
    for item in _queries(query_set):
        question = str(item["question"])
        router_type = _query_type(question)
        vector = cached.get(_cache_key(model, question))
        if vector is None:
            missing.append(str(item.get("queryId") or ""))
        terms = _query_terms(question)
        dense8 = _dense_rank(chunks, embedding_by_id, vector)[:8]
        dense20 = _dense_rank(chunks, embedding_by_id, vector)[:20]
        bm25 = _sparse_rank(chunks, terms)[:8]
        graph = rank_path_evidence_graph(graph_rows, question, router_type, limit=8) if router_type in {"relation", "multi_hop"} else []
        if graph:
            graph_enabled += 1
        active = {"dense": dense8, "sparse": bm25}
        if graph:
            active["graph"] = graph
        rrf = _rrf_rank(active, (1.0, 1.0, 1.0), equal_weight=True, limit=15)
        # Dense@20 is deliberately carried only as a diagnostic.  It does
        # not enter RRF or listwise reranking.
        branches = {
            "dense_top8": dense8,
            "bm25_top8": bm25,
            "graph_top8": graph,
            "rrf_top15": rrf,
            "dense_top20_diagnostic": dense20,
        }
        router_types[router_type] += 1
        cases.append({
            "queryId": item.get("queryId"), "category": item.get("category"), "question": question,
            "queryTerms": terms, "runtimeRouterType": router_type,
            "graphEnabled": bool(graph), "denseQueryVectorCached": vector is not None,
            "branches": branches, "candidatePool": _membership(branches),
        })
    report = {
        "reportVersion": "compact-adaptive-hybrid-v1",
        "status": "READY_FOR_LISTWISE" if not missing else "INCOMPLETE_QUERY_VECTORS",
        "evaluationBoundary": {
            "label": "fixed-query internal engineering evaluation (Dev30); not an independent test set",
            "dense": "Top8", "bm25": "Top8", "graph": "Top8 only when the runtime router returns relation or multi_hop",
            "fusion": "standard equal-weight RRF Top15", "rerankerInput": "rrf_top15", "finalResult": "listwise Top5",
            "diagnosticOnly": "dense_top20_diagnostic is excluded from candidate fusion and reranking",
            "graphImplementation": "local r4 staging export; production Neo4j follows the same accepted semantic 1-3-hop path constraints",
        },
        "assets": {"querySet": str(query_set), "chunks": str(chunks_path), "embeddings": str(embeddings_path), "graph": str(graph_path), "queryCache": str(query_cache)},
        "counts": {"queryCount": len(cases), "missingDenseQueryVectors": len(missing), "routerTypes": dict(router_types), "graphEnabledQueries": graph_enabled, "candidatePoolTotal": sum(len(case["candidatePool"]) for case in cases)},
        "missingDenseQueryIds": missing, "cases": cases,
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
