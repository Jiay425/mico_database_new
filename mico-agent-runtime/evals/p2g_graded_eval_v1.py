"""Evaluate the graded GraphRAG candidate pipeline on the 200-case qrels set.

The script is intentionally conservative: generated source-bound grades may
be used only with ``--allow-provisional-diagnostic`` and the report is marked
``DIAGNOSTIC_ONLY``. Formal model selection requires ``humanAdjudicationStatus
== approved`` and at least one non-null humanGrade per qrel.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

from mico_agent_runtime.contracts.evidence import EvidenceQuery
from mico_agent_runtime.contracts.retrieval import build_retrieval_plan
from mico_agent_runtime.knowledge.database_config import KnowledgeStoreConfiguration
from mico_agent_runtime.knowledge.database_retriever import (
    DatabaseKnowledgeSearchPort,
    _query_terms,
)
from mico_agent_runtime.knowledge.embeddings import GeminiEmbeddingPort
from mico_agent_runtime.knowledge.reranking import _rrf_score, merge_and_rerank


def _query_type(category: str) -> str:
    return {
        "single_fact": "semantic_fact",
        "relation": "relation",
        "multi_hop": "multi_hop",
        "document_synthesis": "composite",
    }.get(category, "semantic_fact")


def _doc_id(item: Any) -> str:
    return str(getattr(item, "externalId", "")).split("#", 1)[0]


def _grades(case: dict[str, Any], formal: bool) -> dict[str, int]:
    values: dict[str, int] = {}
    for qrel in case.get("qrels", []):
        value = qrel.get("humanGrade") if formal else qrel.get("grade")
        if value is not None:
            values[str(qrel["chunkId"])] = int(value)
    return values


def _metrics(ranked50: list[str], top10: list[str], grades: dict[str, int]) -> dict[str, float]:
    relevant = {chunk for chunk, grade in grades.items() if grade > 0}
    gains10 = [grades.get(chunk, 0) for chunk in top10[:10]]
    recall50 = len(set(ranked50[:50]) & relevant) / len(relevant) if relevant else 0.0
    dcg = sum((2**grade - 1) / math.log2(index + 2) for index, grade in enumerate(gains10))
    ideal = sorted(grades.values(), reverse=True)[:10]
    idcg = sum((2**grade - 1) / math.log2(index + 2) for index, grade in enumerate(ideal)) or 1.0
    first = next((index + 1 for index, grade in enumerate(gains10) if grade >= 2), None)
    precision = sum(grade >= 2 for grade in gains10[:5]) / 5.0
    return {
        "recallAt50": round(recall50, 8),
        "ndcgAt10": round(dcg / idcg, 8),
        "mrrAt10": round(1.0 / first, 8) if first else 0.0,
        "evidencePrecisionAt5": round(precision, 8),
    }


def _path_contract(items: list[Any]) -> tuple[float, float]:
    if not items:
        return 1.0, 1.0
    valid_paths = 0
    total_paths = 0
    bound = 0
    for item in items:
        bound += int(bool(item.sourceChunkId))
        for path in item.reasoningPaths:
            total_paths += 1
            valid_paths += int(
                1 <= path.hopCount <= 3
                and bool(path.sourceDocumentIds)
                and all(hop.evidenceChunkId for hop in path.hops)
            )
    return (
        valid_paths / total_paths if total_paths else 1.0,
        bound / len(items),
    )


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"caseCount": 0}
    return {
        "caseCount": len(rows),
        **{
            metric: round(sum(row["metrics"][metric] for row in rows) / len(rows), 8)
            for metric in ("recallAt50", "ndcgAt10", "mrrAt10", "evidencePrecisionAt5")
        },
        "pathCorrectness": round(sum(row["pathCorrectness"] for row in rows) / len(rows), 8),
        "sourceBindingRate": round(sum(row["sourceBindingRate"] for row in rows) / len(rows), 8),
        "latencyP50Ms": round(statistics.median(row["latencyMs"] for row in rows), 3),
        "latencyP95Ms": round(
            sorted(row["latencyMs"] for row in rows)[min(len(rows) - 1, math.ceil(len(rows) * 0.95) - 1)], 3
        ),
    }


def run(
    qrels_path: Path,
    output_path: Path,
    allow_provisional: bool = False,
    skip_dense: bool = False,
    embedding_cache_path: Path | None = None,
) -> dict[str, Any]:
    qrels = json.loads(qrels_path.read_text(encoding="utf-8"))
    formal = qrels.get("humanAdjudicationStatus") == "approved"
    if not formal and not allow_provisional:
        raise SystemExit(
            "QRELS_NOT_APPROVED_FOR_FORMAL_EVAL: pass --allow-provisional-diagnostic "
            "only for a report explicitly marked DIAGNOSTIC_ONLY"
        )
    if formal and not any(
        qrel.get("humanGrade") is not None
        for case in qrels.get("cases", [])
        for qrel in case.get("qrels", [])
    ):
        raise SystemExit("QRELS_APPROVED_WITHOUT_HUMAN_GRADES")

    config = KnowledgeStoreConfiguration.from_environment()
    if embedding_cache_path is None and not skip_dense:
        # Evaluation is repeat-heavy; make persistence the safe default so a
        # restarted run does not spend one provider request per case again.
        embedding_cache_path = qrels_path.parent / ".p2g-query-embedding-cache-v1.json"
    if skip_dense:
        class _DisabledEmbedding:
            modelName = "unavailable-skipped"

            def embed_query(self, _query: str) -> list[float]:
                raise RuntimeError("DENSE_BRANCH_SKIPPED")

            def embed_document(self, _title: str | None, _text: str) -> list[float]:
                raise RuntimeError("DENSE_BRANCH_SKIPPED")

            def close(self) -> None:
                return None

        embedding = _DisabledEmbedding()
    else:
        embedding = GeminiEmbeddingPort.from_environment()
    port = DatabaseKnowledgeSearchPort(
        config,
        embedding,
        query_embedding_cache_path=embedding_cache_path,
    )
    branches = ("vector", "sparse", "graph", "hybrid")
    rows_by_branch: dict[str, list[dict[str, Any]]] = {branch: [] for branch in branches}
    case_rows: list[dict[str, Any]] = []
    try:
        for case in qrels["cases"]:
            category = str(case["category"])
            query = EvidenceQuery(topic=case["query"], direction="context", retrievalMode="hybrid", limit=10)
            plan = build_retrieval_plan(
                query_summary=case["query"],
                query_type=_query_type(category),
                retrieval_mode="hybrid",
                retrieval_branches=["vector", "sparse", "graph"],
                top_k=10,
                graph_version=config.graphVersion,
            )
            terms = _query_terms(case["query"])
            started = time.perf_counter()
            vector = [] if skip_dense else port._vector_branch(query, terms, 50)
            vector_latency_ms = (time.perf_counter() - started) * 1000.0
            started = time.perf_counter()
            sparse = port._sparse_branch(query, terms, 50)
            sparse_latency_ms = (time.perf_counter() - started) * 1000.0
            started = time.perf_counter()
            graph = port._graph_branch(query, terms, plan.maxHops, plan.minConfidence, case["query"], 50)
            graph_latency_ms = (time.perf_counter() - started) * 1000.0
            started = time.perf_counter()
            hybrid = merge_and_rerank(query, vector, graph, plan, sparse_results=sparse)
            hybrid_latency_ms = (time.perf_counter() - started) * 1000.0
            branch_items = {"vector": vector, "sparse": sparse, "graph": graph, "hybrid": hybrid}
            grades = _grades(case, formal)
            vector_rank = {item.sourceChunkId: index for index, item in enumerate(vector, 1)}
            sparse_rank = {item.sourceChunkId: index for index, item in enumerate(sparse, 1)}
            graph_rank = {item.sourceChunkId: index for index, item in enumerate(graph, 1)}
            fused = {
                chunk: _rrf_score(vector_rank.get(chunk), sparse_rank.get(chunk), graph_rank.get(chunk), plan)
                for chunk in set(vector_rank) | set(sparse_rank) | set(graph_rank)
            }
            hybrid50 = [chunk for chunk, _ in sorted(fused.items(), key=lambda item: (-item[1], item[0]))[:50]]
            per_case: dict[str, Any] = {"caseId": case["caseId"], "category": category, "branches": {}}
            per_case["candidatePools"] = {
                "vector": [str(item.sourceChunkId) for item in vector if item.sourceChunkId],
                "sparse": [str(item.sourceChunkId) for item in sparse if item.sourceChunkId],
                "graph": [str(item.sourceChunkId) for item in graph if item.sourceChunkId],
                "hybridRrf": hybrid50,
            }
            branch_latencies = {
                "vector": vector_latency_ms,
                "sparse": sparse_latency_ms,
                "graph": graph_latency_ms,
                "hybrid": vector_latency_ms + sparse_latency_ms + graph_latency_ms + hybrid_latency_ms,
            }
            for branch, items in branch_items.items():
                ranked = [str(item.sourceChunkId) for item in items if item.sourceChunkId]
                ranked50 = hybrid50 if branch == "hybrid" else ranked[:50]
                top10 = [str(item.sourceChunkId) for item in items[:10] if item.sourceChunkId]
                path_correctness, binding_rate = _path_contract(items[:10])
                row = {
                    "metrics": _metrics(ranked50, top10, grades),
                    "pathCorrectness": round(path_correctness, 8),
                    "sourceBindingRate": round(binding_rate, 8),
                    "latencyMs": round(branch_latencies[branch], 3),
                    "resultCount": len(items),
                    "routeSources": sorted({source for item in items for source in item.retrievalSources}),
                }
                rows_by_branch[branch].append(row)
                per_case["branches"][branch] = row
            case_rows.append(per_case)
    finally:
        port.close()

    report = {
        "reportVersion": "p2g-graded-eval-v1",
        "status": "FORMAL" if formal else "DIAGNOSTIC_ONLY",
        "humanAdjudicationStatus": qrels.get("humanAdjudicationStatus"),
        "qrelsFingerprint": qrels.get("qrelsFingerprint"),
        "caseCount": len(qrels.get("cases", [])),
        "caseCountsByCategory": dict(Counter(case["category"] for case in qrels.get("cases", []))),
        "candidateLimit": 50,
        "topK": 10,
        "denseSkipped": skip_dense,
        "runtimeManifest": {
            "graphVersion": config.graphVersion,
            "embeddingVersion": "unavailable-skipped" if skip_dense else plan.embeddingVersion,
            "fusionVersion": plan.fusionVersion,
            "rerankerVersion": plan.rerankerVersion,
            "rrfK": 60,
            "queryEmbeddingCache": {
                "enabled": bool(embedding_cache_path),
                "path": str(embedding_cache_path) if embedding_cache_path else None,
                "hits": port._query_embedding_cache.hits,
                "misses": port._query_embedding_cache.misses,
            },
        },
        "metricsDefinition": {
            "recallAt50": "any grade > 0 in fused/branch candidate Top-50",
            "ndcgAt10": "graded gain 2^grade-1 over Top-10",
            "mrrAt10": "first grade >= 2 in Top-10",
            "evidencePrecisionAt5": "grade >= 2 proportion in Top-5",
        },
        "aggregateByBranch": {branch: _aggregate(rows) for branch, rows in rows_by_branch.items()},
        "aggregateByCategory": {
            category: {
                branch: _aggregate([
                    row["branches"][branch] for row in case_rows if row["category"] == category
                ])
                for branch in branches
            }
            for category in sorted({case["category"] for case in qrels.get("cases", [])})
        },
        "cases": case_rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-provisional-diagnostic", action="store_true")
    parser.add_argument("--skip-dense", action="store_true", help="diagnose sparse/graph while embedding provider is unavailable")
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        help="optional hash-keyed query-vector cache; prevents repeated Gemini calls across eval runs",
    )
    args = parser.parse_args()
    report = run(
        args.qrels,
        args.output,
        args.allow_provisional_diagnostic,
        args.skip_dense,
        args.embedding_cache,
    )
    print(json.dumps({
        "status": report["status"],
        "caseCount": report["caseCount"],
        "aggregateByBranch": report["aggregateByBranch"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
