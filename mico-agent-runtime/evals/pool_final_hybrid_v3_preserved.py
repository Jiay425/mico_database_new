"""Build the conservative final candidate pool: Dense30 + RRF unique20.

RRF is used only to decide which non-Dense BM25/Graph candidates supplement
the full Dense Top30.  It is never used as the final evidence ranking.
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-set", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    chunks = _read_jsonl(args.chunks)
    embeddings = _read_jsonl(args.embeddings)
    graph_rows = _read_jsonl(args.graph)
    embedding_by_id = {str(row["chunkId"]): [float(value) for value in row["embedding"]] for row in embeddings if isinstance(row.get("embedding"), list)}
    model, cache = _load_cache(args.query_cache)
    missing: list[str] = []
    cases: list[dict[str, Any]] = []
    types: Counter[str] = Counter()
    for item in _queries(args.query_set):
        question = str(item["question"])
        terms = _query_terms(question)
        query_type = _query_type(question)
        vector = cache.get(_cache_key(model, question))
        if vector is None:
            missing.append(str(item.get("queryId") or ""))
        branches: dict[str, list[dict[str, Any]]] = {
            "dense_top30": _dense_rank(chunks, embedding_by_id, vector)[:30],
            "bm25_top30": _sparse_rank(chunks, terms)[:30],
            "graph_path_top30": rank_path_evidence_graph(graph_rows, question, query_type, limit=30),
        }
        rrf_all = _rrf_rank(
            {"dense": branches["dense_top30"], "sparse": branches["bm25_top30"], "graph": branches["graph_path_top30"]},
            (1.0, 1.0, 1.0), equal_weight=True, limit=90,
        )
        dense_ids = {str(row["chunkId"]) for row in branches["dense_top30"]}
        supplements = [row for row in rrf_all if str(row["chunkId"]) not in dense_ids][:20]
        branches["rrf_non_dense_top20"] = supplements
        branches["preserved_candidate_top50"] = [
            *[{**row, "candidateSource": "dense_preserved"} for row in branches["dense_top30"]],
            *[{**row, "candidateSource": "rrf_non_dense"} for row in supplements],
        ]
        membership: dict[str, dict[str, Any]] = {}
        for name, rows in branches.items():
            for row in rows:
                chunk_id = str(row.get("chunkId") or "")
                if chunk_id:
                    membership.setdefault(chunk_id, {"chunkId": chunk_id, "branches": {}})["branches"][name] = {"rank": int(row.get("rank") or 0), "score": float(row.get("score") or 0.0)}
        cases.append({
            "queryId": item.get("queryId"), "category": item.get("category"), "question": question,
            "queryTerms": terms, "runtimeRouterType": query_type,
            "retrievalScope": {"scopeType": "global", "allowedDocumentIds": []},
            "denseQueryVectorCached": vector is not None, "branches": branches,
            "candidatePool": sorted(membership.values(), key=lambda value: (-len(value["branches"]), value["chunkId"])),
            "candidatePoolSize": len(membership),
        })
        types[query_type] += 1
    report = {
        "reportVersion": "final-hybrid-v3-dense30-rrfunique20-v1",
        "status": "READY_FOR_BLIND_JUDGING" if not missing else "INCOMPLETE_QUERY_VECTORS",
        "evaluationBoundary": {
            "dense": "complete original question embedding, Top30", "bm25": "lexical aliases, Top30",
            "graph": "accepted query-specific semantic 1-3 hop paths, Top30",
            "candidatePolicy": "preserve every Dense Top30; add the first 20 RRF-ranked candidates absent from Dense",
            "candidateLimit": 50, "finalRanker": "LLM listwise; not executed by this script",
        },
        "counts": {"queryCount": len(cases), "missingDenseQueryVectors": len(missing), "routerTypes": dict(types), "candidatePoolTotal": sum(case["candidatePoolSize"] for case in cases)},
        "cases": cases,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "counts": report["counts"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
