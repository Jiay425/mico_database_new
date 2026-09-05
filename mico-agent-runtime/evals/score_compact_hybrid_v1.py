"""Score Dense Top5 vs compact adaptive Hybrid Listwise Top5 on Dev30."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def _metrics(ids: list[str], relevant: set[str], k: int = 5) -> dict[str, float]:
    hits = [chunk_id in relevant for chunk_id in ids[:k]]
    recall = sum(hits) / len(relevant) if relevant else 0.0
    first = next((index for index, hit in enumerate(hits, 1) if hit), None)
    dcg = sum(1.0 / math.log2(index + 1) for index, hit in enumerate(hits, 1) if hit)
    ideal = sum(1.0 / math.log2(index + 1) for index in range(1, min(k, len(relevant)) + 1))
    return {f"recallAt{k}": recall, f"hitAt{k}": float(bool(first)), f"mrrAt{k}": 1.0 / first if first else 0.0, f"nDCGAt{k}": dcg / ideal if ideal else 0.0}


def _average(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = ("recallAt5", "hitAt5", "mrrAt5", "nDCGAt5")
    return {key: round(sum(row[key] for row in rows) / len(rows), 6) if rows else 0.0 for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--judged", type=Path, required=True)
    parser.add_argument("--reranked", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pool = json.loads(args.pool.read_text(encoding="utf-8"))
    judged = {str(row.get("queryId")): row for row in json.loads(args.judged.read_text(encoding="utf-8")).get("queries") or []}
    reranked = {str(row.get("queryId")): row for row in json.loads(args.reranked.read_text(encoding="utf-8")).get("queries") or []}
    aggregate: dict[str, list[dict[str, float]]] = defaultdict(list)
    categories: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    per_query: list[dict[str, Any]] = []
    diagnostics: list[float] = []
    for case in pool.get("cases") or []:
        query_id = str(case["queryId"])
        if judged.get(query_id, {}).get("status") != "judged" or reranked.get(query_id, {}).get("status") != "reranked":
            raise RuntimeError(f"INCOMPLETE_QUERY:{query_id}")
        relevant = {str(label["chunkId"]) for label in judged[query_id].get("labels") or [] if label.get("relevant")}
        branches = case.get("branches") or {}
        dense = [str(row["chunkId"]) for row in (branches.get("dense_top8") or [])[:5]]
        hybrid = [str(row["chunkId"]) for row in reranked[query_id].get("top5") or []]
        values = {"dense_top5": _metrics(dense, relevant), "adaptive_hybrid_top5": _metrics(hybrid, relevant)}
        dense20 = [str(row["chunkId"]) for row in (branches.get("dense_top20_diagnostic") or [])]
        diagnostics.append(_metrics(dense20, relevant, 20)["recallAt20"])
        category = str(case.get("category"))
        for name, value in values.items():
            aggregate[name].append(value)
            categories[category][name].append(value)
        per_query.append({"queryId": query_id, "category": category, "relevantEvidenceCount": len(relevant), **values, "denseRecallAt20Diagnostic": diagnostics[-1]})
    systems = {name: _average(values) for name, values in aggregate.items()}
    report = {
        "reportVersion": "compact-adaptive-hybrid-dev30-metrics-v1",
        "evaluationBoundary": "Fixed Dev30 internal engineering comparison; one full-union LLM evidence judge with evidence-span validation. This is not an independent test set or human qrels benchmark.",
        "onlinePipeline": "Dense Top8 + BM25 Top8; graph Top8 only for runtime relation/multi_hop routes; equal RRF Top15; content-only listwise Top5.",
        "systems": systems,
        "hybridMinusDense": {key: round(systems["adaptive_hybrid_top5"][key] - systems["dense_top5"][key], 6) for key in systems["dense_top5"]},
        "diagnostics": {"denseRecallAt20": round(sum(diagnostics) / len(diagnostics), 6) if diagnostics else 0.0, "note": "diagnostic only; Dense Top20 is not part of the online Top5 path."},
        "byCategory": {category: {name: _average(values) for name, values in groups.items()} for category, groups in sorted(categories.items())},
        "perQuery": per_query,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPACT_INTERNAL_COMPARISON_COMPLETE", "systems": systems, "hybridMinusDense": report["hybridMinusDense"], "diagnostics": report["diagnostics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
