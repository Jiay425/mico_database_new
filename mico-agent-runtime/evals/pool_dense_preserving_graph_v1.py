"""Runtime-router Dense Top15 + unique Graph Top5 evidence augmentation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from mico_agent_runtime.knowledge.database_retriever import _query_terms, _query_type
from p2g_partial_diagnostic_v1 import _cache_key, _dense_rank, _graph_rank, _load_cache, _read_jsonl
from graph_path_aware_ranker_v1 import rank_path_aware_graph
from graph_path_evidence_ranker_v2 import rank_path_evidence_graph


def _rerank(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**row, "rank": index} for index, row in enumerate(rows, 1)]


def _augment(dense: list[dict[str, Any]], graph: list[dict[str, Any]], graph_enabled: bool) -> list[dict[str, Any]]:
    if not graph_enabled:
        return dense[:20]
    head = list(dense[:15])
    selected = {str(row["chunkId"]) for row in head}
    supplement = []
    for row in graph:
        if str(row["chunkId"]) in selected:
            continue
        supplement.append(row)
        selected.add(str(row["chunkId"]))
        if len(supplement) == 5:
            break
    # Preserve an exact Top20 even when the graph has fewer than five unique
    # evidence chunks; only then backfill from the dense tail.
    if len(supplement) < 5:
        for row in dense[15:]:
            if str(row["chunkId"]) in selected:
                continue
            supplement.append(row)
            selected.add(str(row["chunkId"]))
            if len(supplement) == 5:
                break
    return _rerank(head + supplement)


def run(query_set: Path, chunks_path: Path, embeddings_path: Path, graph_path: Path, query_cache: Path, output: Path, graph_ranker: str = "proxy", document_scope_from_query: bool = False) -> dict[str, Any]:
    queries = list(json.loads(query_set.read_text(encoding="utf-8")).get("items") or [])
    chunks, embeddings, graph_rows = _read_jsonl(chunks_path), _read_jsonl(embeddings_path), _read_jsonl(graph_path)
    embeddings_by_id = {str(row["chunkId"]): [float(value) for value in row["embedding"]] for row in embeddings if isinstance(row.get("embedding"), list)}
    model, cache = _load_cache(query_cache)
    cases: list[dict[str, Any]] = []
    types: Counter[str] = Counter()
    missing: list[str] = []
    for item in queries:
        question = str(item.get("question") or "")
        terms = _query_terms(question)
        vector = cache.get(_cache_key(model, " ".join(terms)))
        if vector is None:
            missing.append(str(item.get("queryId")))
        allowed_pmcids = [str(value) for value in (item.get("sourcePmcids") or []) if str(value)] if document_scope_from_query else []
        scoped_chunks = [row for row in chunks if not allowed_pmcids or str(row.get("pmcid") or "") in set(allowed_pmcids)]
        # This is an evaluation-only simulation of a real request context.
        # It must never be used by an unscoped runtime request: sourcePmcids
        # are hidden provenance metadata of the frozen query set.
        scoped_graph_rows = graph_rows if not allowed_pmcids else [
            row for row in graph_rows
            if row.get("recordType") == "node"
            or str(row.get("evidenceChunkId") or "").split("-", 1)[0] in set(allowed_pmcids)
        ]
        dense = _dense_rank(scoped_chunks, embeddings_by_id, vector)[:20]
        router_type = _query_type(question)
        graph_enabled = router_type in {"relation", "multi_hop"}
        if not graph_enabled:
            graph = []
        elif graph_ranker == "path_aware":
            graph = rank_path_aware_graph(scoped_graph_rows, question, router_type, limit=20)
        elif graph_ranker == "path_evidence":
            graph = rank_path_evidence_graph(scoped_graph_rows, question, router_type, limit=20)
        else:
            graph = _graph_rank(scoped_chunks, scoped_graph_rows, terms)[:20]
        augmented = _augment(dense, graph, graph_enabled)
        membership: dict[str, dict[str, Any]] = {}
        for branch, rows in (("dense", dense), ("dense_preserving_graph", augmented)):
            for row in rows:
                chunk_id = str(row["chunkId"])
                membership.setdefault(chunk_id, {"chunkId": chunk_id, "branches": {}})["branches"][branch] = {"rank": row["rank"], "score": row["score"]}
        types[router_type] += 1
        cases.append({
            "queryId": item.get("queryId"), "category": item.get("category"), "question": question, "queryTerms": terms,
            "retrievalScope": {"scopeType": "document" if allowed_pmcids else "global", "allowedDocumentIds": allowed_pmcids},
            "runtimeRouterType": router_type, "graphEnabled": graph_enabled, "denseQueryVectorCached": vector is not None,
            "branches": {"dense": dense, "graph": graph, "dense_preserving_graph": augmented},
            "candidatePool": sorted(membership.values(), key=lambda row: (-len(row["branches"]), row["chunkId"])), "candidatePoolSize": len(membership),
        })
    result = {
        "reportVersion": "p2g-dense-preserving-graph-pool-v1", "status": "READY_FOR_JUDGING" if not missing else "INCOMPLETE_QUERY_VECTORS",
        "evaluationBoundary": {"topK": 20, "densePreserved": 15, "graphSupplement": 5, "sparse": "excluded", "routerImplementation": "mico_agent_runtime.knowledge.database_retriever._query_type", "graphRanker": graph_ranker, "documentScopeSimulation": document_scope_from_query, "scopeWarning": "When true, sourcePmcids are evaluation metadata used only to simulate a real caller-provided document context; this is not an unscoped runtime score.", "graphImplementation": "local provenance graph export over r4 staging JSONL; not Neo4j publication"},
        "assets": {"querySet": str(query_set), "queryCache": str(query_cache), "graph": str(graph_path)},
        "counts": {"queryCount": len(cases), "denseQueryVectors": len(cases) - len(missing), "missingDenseQueryVectors": len(missing), "candidatePoolTotal": sum(case["candidatePoolSize"] for case in cases), "runtimeRouterTypes": dict(types), "graphEnabledQueries": sum(case["graphEnabled"] for case in cases)},
        "missingDenseQueryIds": missing, "cases": cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-set", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graph-ranker", choices=("proxy", "path_aware", "path_evidence"), default="proxy")
    parser.add_argument("--document-scope-from-query", action="store_true", help="evaluation-only simulation of a caller-provided currentDocumentId")
    args = parser.parse_args()
    report = run(args.query_set, args.chunks, args.embeddings, args.graph, args.query_cache, args.output, args.graph_ranker, args.document_scope_from_query)
    print(json.dumps({"status": report["status"], "counts": report["counts"], "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
