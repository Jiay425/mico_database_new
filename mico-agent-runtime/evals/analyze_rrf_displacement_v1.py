from __future__ import annotations

"""Explain why weighted RRF moves evidence down in the provider-free smoke run.

This is deliberately a diagnostic, not a tuner.  The input cases use the
provenance-derived smoke gold and the controlled 1001-chunk corpus, so the
output is useful for debugging rank movement only and must not be reported as
the final retrieval evaluation.
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _rank_map(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {str(row["chunkId"]): int(row["rank"]) for row in rows}


def _first_gold_rank(rows: list[dict[str, Any]], gold: set[str], cutoff: int = 50) -> int | None:
    for row in rows[:cutoff]:
        if str(row["chunkId"]) in gold:
            return int(row["rank"])
    return None


def _reciprocal(rank: int | None) -> float:
    return 1.0 / rank if rank else 0.0


def _case_displacement(case_row: dict[str, Any]) -> dict[str, Any]:
    case = case_row["case"]
    gold = {str(value) for value in case.get("goldChunkIds") or []}
    branches = case_row.get("branches") or {}
    maps = {name: _rank_map(rows or []) for name, rows in branches.items()}
    present_gold = sorted(
        chunk_id
        for chunk_id in gold
        if any(chunk_id in maps.get(name, {}) for name in ("dense", "sparse", "graph"))
    )
    movements: list[dict[str, Any]] = []
    for chunk_id in present_gold:
        standard_rank = maps.get("rrf_standard", {}).get(chunk_id)
        weighted_rank = maps.get("rrf_weighted", {}).get(chunk_id)
        movement = None
        if standard_rank is not None and weighted_rank is not None:
            movement = weighted_rank - standard_rank
        movements.append(
            {
                "chunkId": chunk_id,
                "denseRank": maps.get("dense", {}).get(chunk_id),
                "sparseRank": maps.get("sparse", {}).get(chunk_id),
                "graphRank": maps.get("graph", {}).get(chunk_id),
                "standardRrfRank": standard_rank,
                "weightedRrfRank": weighted_rank,
                "weightedMinusStandard": movement,
                "standardInTop10": bool(standard_rank and standard_rank <= 10),
                "weightedInTop10": bool(weighted_rank and weighted_rank <= 10),
            }
        )

    standard_first = _first_gold_rank(branches.get("rrf_standard", []), gold)
    weighted_first = _first_gold_rank(branches.get("rrf_weighted", []), gold)
    standard_ndcg = float((case_row.get("metrics") or {}).get("rrf_standard", {}).get("ndcgAt10", 0.0))
    weighted_ndcg = float((case_row.get("metrics") or {}).get("rrf_weighted", {}).get("ndcgAt10", 0.0))
    return {
        "caseId": str(case.get("caseId")),
        "category": str(case.get("category")),
        "query": str(case.get("query")),
        "goldChunkCount": len(gold),
        "goldChunkCountPresentInBranches": len(present_gold),
        "standardFirstGoldRank": standard_first,
        "weightedFirstGoldRank": weighted_first,
        "firstGoldRankDelta": (
            weighted_first - standard_first
            if standard_first is not None and weighted_first is not None
            else None
        ),
        "standardNdcgAt10": standard_ndcg,
        "weightedNdcgAt10": weighted_ndcg,
        "ndcgDelta": round(weighted_ndcg - standard_ndcg, 8),
        "goldMovements": movements,
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    moved_down = [
        movement
        for row in rows
        for movement in row["goldMovements"]
        if movement.get("weightedMinusStandard") is not None
        and movement["weightedMinusStandard"] > 0
    ]
    moved_down_3 = [item for item in moved_down if item["weightedMinusStandard"] >= 3]
    case_hit_loss = [
        row for row in rows
        if row["standardFirstGoldRank"] is not None
        and row["standardFirstGoldRank"] <= 10
        and (row["weightedFirstGoldRank"] is None or row["weightedFirstGoldRank"] > 10)
    ]
    category_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        category_rows[row["category"]].append(row)
    by_category: dict[str, Any] = {}
    for category, category_items in sorted(category_rows.items()):
        deltas = [float(item["ndcgDelta"]) for item in category_items]
        by_category[category] = {
            "caseCount": len(category_items),
            "meanNdcgDelta": round(sum(deltas) / len(deltas), 8) if deltas else 0.0,
            "caseHitLossCount": sum(
                item in case_hit_loss for item in category_items
            ),
            "goldMovedDownCount": sum(
                1
                for item in category_items
                for movement in item["goldMovements"]
                if movement.get("weightedMinusStandard") is not None
                and movement["weightedMinusStandard"] > 0
            ),
        }
    return {
        "caseCount": len(rows),
        "caseHitLossCount": len(case_hit_loss),
        "goldMovementCount": len(moved_down),
        "goldMovedDownByAtLeast3Count": len(moved_down_3),
        "meanNdcgDelta": round(
            sum(float(row["ndcgDelta"]) for row in rows) / len(rows), 8
        ) if rows else 0.0,
        "weightedNdcgWorseCaseCount": sum(float(row["ndcgDelta"]) < 0 for row in rows),
        "byCategory": by_category,
        "largestDownwardMovements": sorted(
            moved_down,
            key=lambda item: (-int(item["weightedMinusStandard"]), item["chunkId"]),
        )[:25],
        "hitLossCases": [
            {
                "caseId": row["caseId"],
                "category": row["category"],
                "standardFirstGoldRank": row["standardFirstGoldRank"],
                "weightedFirstGoldRank": row["weightedFirstGoldRank"],
                "ndcgDelta": row["ndcgDelta"],
            }
            for row in case_hit_loss
        ],
    }


def run(input_path: Path, output_path: Path) -> dict[str, Any]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    rows = [_case_displacement(item) for item in payload.get("cases", [])]
    result = {
        "reportVersion": "p2g-rrf-displacement-v1",
        "status": "diagnostic_only",
        "source": str(input_path),
        "boundary": payload.get("evaluationBoundary"),
        "summary": _summary(rows),
        "cases": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("p2g-partial-diagnostic-v1.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("p2g-rrf-displacement-v1.json"))
    args = parser.parse_args()
    result = run(args.input, args.output)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
