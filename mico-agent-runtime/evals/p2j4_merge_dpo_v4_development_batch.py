"""Merge an offline-rescored development batch with failed-case repairs.

Successful primary cases are immutable.  A repair may replace only a primary
FAIL for the same case ID; the merged asset must be all PASS before it can be
registered as preserved in the DPO v4 collection plan.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def merge(primary_path: Path, repair_paths: list[Path]) -> dict[str, Any]:
    primary = _load(primary_path)
    if primary.get("mode") != "offline_rescore":
        raise ValueError("DPO_V4_PRIMARY_RESCORE_REQUIRED")
    primary_services = primary.get("servicesObserved") or {}
    if not isinstance(primary_services, dict):
        raise ValueError("DPO_V4_PRIMARY_SERVICES_PROVENANCE_INVALID")
    traces = {item["traceId"]: item for item in primary.get("traces", [])}
    scores = {item["caseId"]: item for item in primary.get("scores", [])}
    repaired: list[str] = []
    repair_services: dict[str, dict[str, Any]] = {}
    for repair_path in repair_paths:
        repair = _load(repair_path)
        if repair.get("mode") != "real_run":
            raise ValueError("DPO_V4_REPAIR_REAL_RUN_REQUIRED")
        services = repair.get("servicesObserved") or {}
        if not isinstance(services, dict):
            raise ValueError("DPO_V4_REPAIR_SERVICES_PROVENANCE_INVALID")
        repair_services[repair_path.name] = services
        repair_traces = {item["traceId"]: item for item in repair.get("traces", [])}
        for score in repair.get("scores", []):
            case_id = score["caseId"]
            if case_id not in scores or scores[case_id].get("status") != "FAIL":
                raise ValueError("DPO_V4_REPAIR_MUST_REPLACE_PRIMARY_FAILURE")
            if score.get("status") != "PASS" or score.get("traceId") not in repair_traces:
                raise ValueError("DPO_V4_REPAIR_NOT_PASS")
            traces.pop(scores[case_id]["traceId"], None)
            traces[score["traceId"]] = repair_traces[score["traceId"]]
            scores[case_id] = score
            repaired.append(case_id)
    ordered_scores = [scores[case_id] for case_id in sorted(scores)]
    if any(score.get("status") != "PASS" for score in ordered_scores):
        raise ValueError("DPO_V4_MERGED_BATCH_NOT_ALL_PASS")
    return {
        "schemaVersion": "p2j4-dpo-v4-development-merge-v1",
        "mode": "development_trace_merge",
        "status": "FROZEN_ALL_PASS",
        "sourcePrimary": primary_path.name,
        "sourceRepairs": [path.name for path in repair_paths],
        "servicesObserved": primary_services,
        "repairServicesObserved": repair_services,
        "preservedPrimaryPassCaseCount": len(ordered_scores) - len(repaired),
        "repairedFailureCaseIds": repaired,
        "caseCount": len(ordered_scores),
        "completedCaseIds": [score["caseId"] for score in ordered_scores],
        "traces": [traces[score["traceId"]] for score in ordered_scores],
        "scores": ordered_scores,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--repair", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = merge(args.primary, args.repair)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "caseCount", "preservedPrimaryPassCaseCount", "repairedFailureCaseIds")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
