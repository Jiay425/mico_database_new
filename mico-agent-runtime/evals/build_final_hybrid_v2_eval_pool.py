"""Reduce Final Hybrid v2 pre-rerank candidates to a fair Dense-vs-RRF Top-20 pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def run(source: Path, output: Path) -> dict[str, Any]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    cases: list[dict[str, Any]] = []
    for case in payload.get("cases") or []:
        branches = case.get("branches") or {}
        dense = list(branches.get("dense_top30") or [])[:20]
        hybrid = list(branches.get("rrf_top40") or [])[:20]
        candidates: dict[str, dict[str, Any]] = {}
        for branch_name, rows in (("dense", dense), ("hybrid_rrf", hybrid)):
            for row in rows:
                chunk_id = str(row.get("chunkId") or "")
                if not chunk_id:
                    continue
                candidates.setdefault(chunk_id, {"chunkId": chunk_id, "branches": {}})["branches"][branch_name] = {
                    "rank": int(row.get("rank") or 0), "score": float(row.get("score") or 0.0),
                }
        cases.append({
            "queryId": case.get("queryId"), "category": case.get("category"), "question": case.get("question"),
            "retrievalScope": case.get("retrievalScope"), "runtimeRouterType": case.get("runtimeRouterType"),
            "branches": {"dense": dense, "hybrid_rrf": hybrid},
            "candidatePool": sorted(candidates.values(), key=lambda row: (-len(row["branches"]), row["chunkId"])),
            "candidatePoolSize": len(candidates),
        })
    result = {
        "reportVersion": "final-hybrid-v2-dev30-rrf-top20-eval-pool-v1",
        "status": "READY_FOR_BLIND_JUDGING",
        "evaluationBoundary": {
            "dense": "Dense Top20", "hybrid": "Dense30 + BM25 30 + graph path 30 -> standard RRF Top20",
            "crossEncoder": "not executed", "judgeExposure": "branch/rank/provenance hidden",
        },
        "sourcePool": str(source), "counts": {"queryCount": len(cases), "candidatePoolTotal": sum(case["candidatePoolSize"] for case in cases)},
        "cases": cases,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.source, args.output)
    print(json.dumps({"status": report["status"], "counts": report["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
