"""Merge model-origin collection batches without importing failed cases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def merge(paths: list[Path]) -> dict[str, Any]:
    traces: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    excluded_failures: list[str] = []
    seen_cases: set[str] = set()
    source_files: list[str] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        services = payload.get("servicesObserved") or {}
        if payload.get("mode") != "real_run":
            raise ValueError(f"MODEL_ORIGIN_SOURCE_NOT_REAL_RUN:{path.name}")
        if services.get("plannerModelUsed") is not True:
            raise ValueError(f"MODEL_ORIGIN_PLANNER_PROVENANCE_INVALID:{path.name}")
        by_trace = {trace.get("traceId"): trace for trace in payload.get("traces", [])}
        source_files.append(path.name)
        for score in payload.get("scores", []):
            case_id = score.get("caseId")
            if case_id in seen_cases:
                raise ValueError(f"MODEL_ORIGIN_DUPLICATE_CASE:{case_id}")
            seen_cases.add(case_id)
            if score.get("status") == "PASS":
                trace = by_trace.get(score.get("traceId"))
                if not trace:
                    raise ValueError(f"MODEL_ORIGIN_PASS_TRACE_MISSING:{case_id}")
                traces.append(trace)
                scores.append(score)
            else:
                excluded_failures.append(case_id)
    scores.sort(key=lambda item: item["caseId"])
    traces_by_id = {trace["traceId"]: trace for trace in traces}
    return {
        "schemaVersion": "p2j4-dpo-v4-model-origin-collection-merge-v1",
        "mode": "model_origin_collection_merge",
        "status": "PASS_ONLY_MERGED",
        "sourceFiles": source_files,
        "plannerModelUsedRequired": True,
        "caseCount": len(scores),
        "excludedFailureCaseCount": len(excluded_failures),
        "excludedFailureCaseIds": excluded_failures,
        "completedCaseIds": [score["caseId"] for score in scores],
        "traces": [traces_by_id[score["traceId"]] for score in scores],
        "scores": scores,
        "trainingStarted": False,
        "pairConstructionStarted": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = merge(args.input)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "caseCount", "excludedFailureCaseCount")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
