"""Compare Dense Top20 with independent LLM-listwise reranked Hybrid Top20."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def _metrics(ids: list[str], relevant: set[str]) -> dict[str, float]:
    binary = [item in relevant for item in ids]
    recall = sum(binary[:20]) / len(relevant) if relevant else 0.0
    hit5 = float(any(binary[:5]))
    first = next((index for index, value in enumerate(binary[:10], 1) if value), None)
    mrr = 1.0 / first if first else 0.0
    dcg = sum(1.0 / math.log2(index + 1) for index, value in enumerate(binary[:10], 1) if value)
    ideal = min(10, len(relevant))
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal + 1))
    return {"recallAt20": recall, "hitAt5": hit5, "mrrAt10": mrr, "nDCGAt10": dcg / idcg if idcg else 0.0}


def _average(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: round(sum(row[key] for row in rows) / len(rows), 6) if rows else 0.0 for key in ("recallAt20", "hitAt5", "mrrAt10", "nDCGAt10")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--judged", type=Path, required=True)
    parser.add_argument("--reranked", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pool = json.loads(args.pool.read_text(encoding="utf-8"))
    judgments = {str(row["queryId"]): row for row in json.loads(args.judged.read_text(encoding="utf-8")).get("queries") or []}
    reranked = {str(row["queryId"]): row for row in json.loads(args.reranked.read_text(encoding="utf-8")).get("queries") or []}
    rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    categories: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    per_query: list[dict[str, Any]] = []
    for case in pool.get("cases") or []:
        query_id = str(case["queryId"])
        if judgments.get(query_id, {}).get("status") != "judged" or reranked.get(query_id, {}).get("status") != "reranked":
            raise RuntimeError(f"INCOMPLETE_QUERY:{query_id}")
        relevant = {str(label["chunkId"]) for label in judgments[query_id].get("labels") or [] if label.get("relevant")}
        dense = [str(row["chunkId"]) for row in list((case.get("branches") or {}).get("dense_top30") or [])[:20]]
        hybrid = [str(row["chunkId"]) for row in reranked[query_id]["top20"]]
        values = {"dense": _metrics(dense, relevant), "hybrid_listwise": _metrics(hybrid, relevant)}
        category = str(case.get("category"))
        for name, value in values.items():
            rows[name].append(value)
            categories[category][name].append(value)
        per_query.append({"queryId": query_id, "category": category, "relevantEvidenceCount": len(relevant), **values})
    systems = {name: _average(value) for name, value in rows.items()}
    report = {
        "reportVersion": "final-hybrid-v2-listwise-dev30-v1",
        "evaluationBoundary": "Dev30 internal diagnostic; full three-way union single-LLM judgments, not an independent human qrels benchmark.",
        "systems": systems,
        "hybridMinusDense": {key: round(systems["hybrid_listwise"][key] - systems["dense"][key], 6) for key in systems["dense"]},
        "byCategory": {category: {name: _average(value) for name, value in values.items()} for category, values in sorted(categories.items())},
        "perQuery": per_query,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "LISTWISE_COMPARISON_COMPLETE", "systems": systems, "hybridMinusDense": report["hybridMinusDense"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
