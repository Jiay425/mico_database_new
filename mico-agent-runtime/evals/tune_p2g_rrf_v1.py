"""Grid-search three-way RRF weights on approved graded qrels only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from mico_agent_runtime.contracts.evidence import EvidenceQuery
from mico_agent_runtime.contracts.retrieval import build_retrieval_plan
from mico_agent_runtime.knowledge.database_config import KnowledgeStoreConfiguration
from mico_agent_runtime.knowledge.database_retriever import DatabaseKnowledgeSearchPort, _query_terms
from mico_agent_runtime.knowledge.embeddings import GeminiEmbeddingPort

K = 60.0


def _query_type(category: str) -> str:
    return {
        "single_fact": "semantic_fact",
        "relation": "relation",
        "multi_hop": "multi_hop",
        "document_synthesis": "composite",
    }.get(category, "semantic_fact")


def _metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    recall50: list[float] = []
    ndcg: list[float] = []
    mrr: list[float] = []
    precision: list[float] = []
    for row in rows:
        grades = {str(x["chunkId"]): int(x["grade"]) for x in row["qrels"]}
        ranked = row["ranked"]
        gains = [grades.get(chunk, 0) for chunk in ranked]
        relevant = {chunk for chunk, grade in grades.items() if grade > 0}
        recall50.append(len(set(ranked[:50]) & relevant) / max(1, len(relevant)))
        dcg = sum((2**grade - 1) / math.log2(index + 2) for index, grade in enumerate(gains[:10]))
        ideal = sorted(grades.values(), reverse=True)[:10]
        idcg = sum((2**grade - 1) / math.log2(index + 2) for index, grade in enumerate(ideal)) or 1.0
        ndcg.append(dcg / idcg)
        first = next((index + 1 for index, grade in enumerate(gains[:10]) if grade >= 2), None)
        mrr.append(1 / first if first else 0.0)
        precision.append(sum(grade >= 2 for grade in gains[:5]) / 5.0)
    count = max(1, len(rows))
    return {
        "recallAt50": sum(recall50) / count,
        "ndcgAt10": sum(ndcg) / count,
        "mrrAt10": sum(mrr) / count,
        "evidencePrecisionAt5": sum(precision) / count,
    }


def _cache(qrels: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("qrelsFingerprint") == qrels.get("qrelsFingerprint") and all(
            all(branch in case for branch in ("vector", "sparse", "graph"))
            for case in cached.get("cases", [])
        ):
            return cached["cases"]
    config = KnowledgeStoreConfiguration.from_environment()
    port = DatabaseKnowledgeSearchPort(
        config,
        GeminiEmbeddingPort.from_environment(),
        query_embedding_cache_path=path.with_name(".p2g-query-embedding-cache-v1.json"),
    )
    rows: list[dict[str, Any]] = []
    try:
        for case in qrels["cases"]:
            query = EvidenceQuery(topic=case["query"], direction="context", retrievalMode="hybrid", limit=10)
            plan = build_retrieval_plan(
                query_summary=case["query"],
                query_type=_query_type(case["category"]),
                retrieval_mode="hybrid",
                retrieval_branches=["vector", "sparse", "graph"],
                top_k=10,
                graph_version=config.graphVersion,
            )
            terms = _query_terms(query.topic)
            vector = port._vector_branch(query, terms, 50)
            sparse = port._sparse_branch(query, terms, 50)
            graph = port._graph_branch(query, terms, plan.maxHops, plan.minConfidence, query.topic, 50)
            rows.append({
                "caseId": case["caseId"],
                "category": case["category"],
                "qrels": case["qrels"],
                "vector": [item.sourceChunkId for item in vector if item.sourceChunkId],
                "sparse": [item.sourceChunkId for item in sparse if item.sourceChunkId],
                "graph": [item.sourceChunkId for item in graph if item.sourceChunkId],
            })
    finally:
        port.close()
    path.write_text(json.dumps({"qrelsFingerprint": qrels["qrelsFingerprint"], "cases": rows}, ensure_ascii=False), encoding="utf-8")
    return rows


def _rank(case: dict[str, Any], weights: tuple[float, float, float]) -> list[str]:
    scores: dict[str, float] = {}
    for branch, weight in zip(("vector", "sparse", "graph"), weights):
        for index, chunk in enumerate(case[branch], 1):
            scores[chunk] = scores.get(chunk, 0.0) + weight / (K + index)
    return [chunk for chunk, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:50]]


def _cache_from_eval(qrels: dict[str, Any], evaluation_path: Path) -> list[dict[str, Any]]:
    """Reuse an already completed retrieval run without querying providers again."""
    report = json.loads(evaluation_path.read_text(encoding="utf-8"))
    by_id = {str(row["caseId"]): row for row in report.get("cases", [])}
    if not by_id:
        raise SystemExit("CANDIDATE_EVAL_MISSING_CASES")
    rows: list[dict[str, Any]] = []
    for case in qrels["cases"]:
        row = by_id.get(str(case["caseId"]))
        pools = row.get("candidatePools") if row else None
        if not isinstance(pools, dict) or not all(branch in pools for branch in ("vector", "sparse", "graph")):
            raise SystemExit("CANDIDATE_EVAL_MISSING_THREE_WAY_POOLS")
        rows.append({
            "caseId": case["caseId"],
            "category": case["category"],
            "qrels": case["qrels"],
            "vector": pools["vector"],
            "sparse": pools["sparse"],
            "graph": pools["graph"],
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-eval", type=Path)
    parser.add_argument("--allow-provisional-diagnostic", action="store_true")
    args = parser.parse_args()
    qrels = json.loads(args.qrels.read_text(encoding="utf-8"))
    formal = qrels.get("humanAdjudicationStatus") == "approved"
    if not formal and not args.allow_provisional_diagnostic:
        raise SystemExit(
            "QRELS_NOT_APPROVED_FOR_MODEL_SELECTION: use --allow-provisional-diagnostic "
            "only for explicitly non-promotable diagnostics"
        )
    cached = _cache_from_eval(qrels, args.candidate_eval) if args.candidate_eval else _cache(qrels, args.cache)
    results: list[dict[str, Any]] = []
    for dense_step in range(21):
        for sparse_step in range(21 - dense_step):
            graph_step = 20 - dense_step - sparse_step
            weights = (dense_step / 20, sparse_step / 20, graph_step / 20)
            rows = [{**case, "ranked": _rank(case, weights)} for case in cached]
            results.append({
                "denseWeight": weights[0], "sparseWeight": weights[1], "graphWeight": weights[2],
                "metrics": _metrics(rows),
            })
    best = max(results, key=lambda row: (
        row["metrics"]["ndcgAt10"], row["metrics"]["mrrAt10"], row["metrics"]["recallAt50"]
    ))
    args.output.write_text(json.dumps({
        "status": "FORMAL" if formal else "DIAGNOSTIC_ONLY",
        "qrelsFingerprint": qrels["qrelsFingerprint"],
        "candidatePool": 50,
        "k": K,
        "fusionVersion": "rrf-v2-listwise-v1",
        "best": best,
        "grid": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(best, ensure_ascii=False))


if __name__ == "__main__":
    main()
