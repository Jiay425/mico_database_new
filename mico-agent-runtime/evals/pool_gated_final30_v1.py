"""Build held-out Final30 rankings using the existing runtime query-type router.

Dense is always available.  Only runtime-routed ``relation`` or ``multi_hop``
queries use equal-weight Dense+Graph RRF; sparse is intentionally excluded.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from mico_agent_runtime.knowledge.database_retriever import _query_terms, _query_type
from p2g_partial_diagnostic_v1 import _cache_key, _dense_rank, _graph_rank, _load_cache, _read_jsonl, _rrf_rank


def run(query_set: Path, chunks_path: Path, embeddings_path: Path, graph_path: Path, query_cache: Path, output: Path, top_k: int = 20) -> dict[str, Any]:
    queries = list(json.loads(query_set.read_text(encoding="utf-8")).get("items") or [])
    chunks, embeddings, graph_rows = _read_jsonl(chunks_path), _read_jsonl(embeddings_path), _read_jsonl(graph_path)
    by_id = {str(row["chunkId"]): [float(value) for value in row["embedding"]] for row in embeddings if isinstance(row.get("embedding"), list)}
    model, cache = _load_cache(query_cache)
    cases: list[dict[str, Any]] = []
    missing: list[str] = []
    router_counts: Counter[str] = Counter()
    for item in queries:
        question = str(item.get("question") or "")
        terms = _query_terms(question)
        vector = cache.get(_cache_key(model, " ".join(terms)))
        if vector is None:
            missing.append(str(item.get("queryId")))
        dense = _dense_rank(chunks, by_id, vector)[:top_k]
        router_type = _query_type(question)
        router_counts[router_type] += 1
        graph_enabled = router_type in {"relation", "multi_hop"}
        graph = _graph_rank(chunks, graph_rows, terms)[:top_k] if graph_enabled else []
        gated = _rrf_rank({"dense": dense, "sparse": [], "graph": graph}, (1.0, 1.0, 1.0), equal_weight=True)[:top_k] if graph_enabled else dense
        memberships: dict[str, dict[str, Any]] = {}
        for branch, rows in (("dense", dense), ("gated_graph_rag", gated)):
            for row in rows:
                chunk_id = str(row.get("chunkId") or "")
                entry = memberships.setdefault(chunk_id, {"chunkId": chunk_id, "branches": {}})
                entry["branches"][branch] = {"rank": int(row["rank"]), "score": float(row["score"])}
        cases.append({
            "queryId": item.get("queryId"), "category": item.get("category"), "question": question,
            "queryTerms": terms, "denseQueryVectorCached": vector is not None,
            "runtimeRouterType": router_type, "graphEnabled": graph_enabled,
            "branches": {"dense": dense, "graph": graph, "gated_graph_rag": gated},
            "candidatePool": sorted(memberships.values(), key=lambda row: (-len(row["branches"]), row["chunkId"])),
            "candidatePoolSize": len(memberships),
        })
    report = {
        "reportVersion": "p2g-gated-final30-pool-v1", "status": "READY_FOR_SINGLE_JUDGING" if not missing else "INCOMPLETE_QUERY_VECTORS",
        "evaluationBoundary": {
            "querySet": query_set.name, "corpus": "chunk-v2-medium", "chunkCount": len(chunks), "embeddingCount": len(by_id),
            "topK": top_k, "routerImplementation": "mico_agent_runtime.knowledge.database_retriever._query_type",
            "policy": "relation/multi_hop => equal-weight Dense+Graph RRF; all other runtime router types => Dense only; Sparse excluded.",
            "graphImplementation": "local provenance proxy over reviewed r4 staging JSONL; not Neo4j publication",
        },
        "assets": {"querySet": str(query_set), "queryCache": str(query_cache), "graph": str(graph_path)},
        "counts": {"queryCount": len(cases), "denseQueryVectors": len(cases) - len(missing), "missingDenseQueryVectors": len(missing), "candidatePoolTotal": sum(case["candidatePoolSize"] for case in cases), "runtimeRouterTypes": dict(router_counts), "graphEnabledQueries": sum(case["graphEnabled"] for case in cases)},
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
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()
    result = run(args.query_set, args.chunks, args.embeddings, args.graph, args.query_cache, args.output, args.top_k)
    print(json.dumps({"status": result["status"], "counts": result["counts"], "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
