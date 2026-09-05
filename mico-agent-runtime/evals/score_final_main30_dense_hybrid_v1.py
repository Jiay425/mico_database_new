"""Score the fixed main30 Dense-versus-Hybrid comparison from binary LLM labels."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def _score(ranked: list[dict[str, Any]], relevant: set[str]) -> dict[str, float]:
    ids = [str(item.get("chunkId")) for item in ranked]
    binary = [chunk_id in relevant for chunk_id in ids]
    recall20 = sum(binary[:20]) / len(relevant) if relevant else 0.0
    hit5 = 1.0 if any(binary[:5]) else 0.0
    first = next((index + 1 for index, value in enumerate(binary[:10]) if value), None)
    mrr10 = 1.0 / first if first else 0.0
    dcg = sum(1.0 / math.log2(index + 2) for index, value in enumerate(binary[:10]) if value)
    ideal = min(len(relevant), 10)
    idcg = sum(1.0 / math.log2(index + 2) for index in range(ideal))
    return {"recallAt20": recall20, "hitAt5": hit5, "mrrAt10": mrr10, "nDCGAt10": dcg / idcg if idcg else 0.0}


def _aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: _mean([row[key] for row in rows]) for key in ("recallAt20", "hitAt5", "mrrAt10", "nDCGAt10")}


def run(pool_path: Path, judged_path: Path, output_path: Path, hybrid_branch: str = "rrf_standard") -> dict[str, Any]:
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    judged = json.loads(judged_path.read_text(encoding="utf-8"))
    judgments = {str(row.get("queryId")): row for row in judged.get("queries") or []}
    cases = list(pool.get("cases") or [])
    incomplete = [str(case.get("queryId")) for case in cases if judgments.get(str(case.get("queryId")), {}).get("status") != "judged"]
    if incomplete:
        raise RuntimeError(f"JUDGMENTS_INCOMPLETE count={len(incomplete)}")
    scores: dict[str, list[dict[str, float]]] = defaultdict(list)
    category_scores: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    per_query: list[dict[str, Any]] = []
    for case in cases:
        query_id = str(case.get("queryId"))
        relevant = {str(label.get("chunkId")) for label in judgments[query_id].get("labels") or [] if label.get("relevant")}
        category = str(case.get("category"))
        values: dict[str, dict[str, float]] = {}
        for label, branch in (("dense", "dense"), ("hybrid", hybrid_branch)):
            value = _score(list((case.get("branches") or {}).get(branch) or []), relevant)
            scores[label].append(value)
            category_scores[category][label].append(value)
            values[label] = value
        per_query.append({"queryId": query_id, "category": category, "relevantEvidenceCount": len(relevant), "dense": values["dense"], "hybrid": values["hybrid"]})
    overall = {label: _aggregate(rows) for label, rows in scores.items()}
    relative = {key: round(overall["hybrid"][key] - overall["dense"][key], 6) for key in overall["dense"]}
    report = {
        "reportVersion": "p2g-final-main30-dense-hybrid-v1",
        "status": "FIXED_SET_SINGLE_JUDGE_COMPARISON",
        "evaluationScope": "Fixed independent 30-query medical set; binary single-LLM evidence judgments with exact-span requirement. This is a project benchmark, not a human-adjudicated external benchmark.",
        "pool": str(pool_path),
        "judgments": str(judged_path),
        "queryCount": len(cases),
        "hybridBranch": hybrid_branch,
        "systems": overall,
        "hybridMinusDense": relative,
        "byCategory": {category: {label: _aggregate(rows[label]) for label in ("dense", "hybrid")} for category, rows in sorted(category_scores.items())},
        "perQuery": per_query,
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--judged", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hybrid-branch", default="rrf_standard")
    args = parser.parse_args()
    report = run(args.pool, args.judged, args.output, args.hybrid_branch)
    print(json.dumps({"status": report["status"], "queryCount": report["queryCount"], "systems": report["systems"], "hybridMinusDense": report["hybridMinusDense"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
