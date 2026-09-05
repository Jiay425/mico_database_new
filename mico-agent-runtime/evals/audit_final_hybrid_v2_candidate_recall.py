"""Measure candidate coverage before any final reranking.

The denominator is every judged-relevant chunk in the per-query union of
Dense30, BM25 Top30 and graph-path Top30.  This makes the comparison a fair
candidate-generation diagnostic; it deliberately does not claim full-corpus
recall or final ranking quality.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


SYSTEMS: dict[str, tuple[str, ...]] = {
    "dense_top20": ("dense_top30",),
    "dense_top30": ("dense_top30",),
    "dense30_union_bm25_30": ("dense_top30", "bm25_top30"),
    "dense30_union_graph30": ("dense_top30", "graph_path_top30"),
    "three_way_union90": ("dense_top30", "bm25_top30", "graph_path_top30"),
    "rrf_top40": ("rrf_top40",),
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ids(case: dict[str, Any], system: str) -> set[str]:
    branches = case.get("branches") or {}
    if system == "dense_top20":
        return {str(row["chunkId"]) for row in list(branches.get("dense_top30") or [])[:20]}
    return {
        str(row["chunkId"])
        for branch in SYSTEMS[system]
        for row in list(branches.get(branch) or [])
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", required=True, type=Path)
    parser.add_argument("--judged", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    pool = _load(args.pool)
    judged = _load(args.judged)
    labels = {
        str(row["queryId"]): {
            str(label["chunkId"])
            for label in row.get("labels") or []
            if bool(label.get("relevant"))
        }
        for row in judged.get("queries") or []
        if row.get("status") == "judged"
    }
    missing = [str(case.get("queryId")) for case in pool.get("cases") or [] if str(case.get("queryId")) not in labels]
    if missing:
        raise RuntimeError(f"JUDGMENTS_INCOMPLETE count={len(missing)}")
    # Candidate recall has to judge the *entire* three-way union.  Treating
    # unjudged candidates as non-relevant would artificially favor Dense.
    unjudged = [
        str(case["queryId"])
        for case in pool.get("cases") or []
        if {
            str(row["chunkId"]) for row in case.get("candidatePool") or []
        } - {
            str(label["chunkId"]) for row in judged.get("queries") or []
            if str(row.get("queryId")) == str(case["queryId"])
            for label in row.get("labels") or []
        }
    ]
    if unjudged:
        raise RuntimeError(f"CANDIDATE_POOL_NOT_FULLY_JUDGED count={len(unjudged)}")

    per_system: dict[str, list[float]] = defaultdict(list)
    per_category: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    per_query: list[dict[str, Any]] = []
    for case in pool.get("cases") or []:
        query_id = str(case["queryId"])
        relevant = labels[query_id]
        scores: dict[str, float] = {}
        for system in SYSTEMS:
            selected = _ids(case, system)
            score = len(selected & relevant) / len(relevant) if relevant else 0.0
            scores[system] = score
            per_system[system].append(score)
            per_category[str(case.get("category"))][system].append(score)
        per_query.append({
            "queryId": query_id,
            "category": case.get("category"),
            "judgedRelevantCount": len(relevant),
            "candidateRecall": scores,
        })

    average = lambda rows: round(sum(rows) / len(rows), 6) if rows else 0.0
    report = {
        "reportVersion": "final-hybrid-v2-candidate-recall-audit-v1",
        "evaluationBoundary": (
            "Dev30 diagnostic. Denominator is judged relevance in the per-query "
            "three-way candidate union, not a full-corpus qrels set."
        ),
        "queryCount": len(per_query),
        "candidateRecall": {name: average(values) for name, values in per_system.items()},
        "byCategory": {
            category: {name: average(values) for name, values in systems.items()}
            for category, systems in sorted(per_category.items())
        },
        "perQuery": per_query,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "CANDIDATE_RECALL_READY", "queryCount": len(per_query), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
