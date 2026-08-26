"""Audit per-decision provenance for the DPO v4 policy-source traces.

This report is intentionally conservative: payload-level plannerModelUsed is
not evidence that every decision was a raw model decision.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "evals" / "p2j4-dpo-v4-policy-source-runs"
RUNS = (
    "canary-compact-001.json",
    "batch-01-final.json",
    "batch-02-final.json",
    "batch-03-final.json",
    "batch-04-final.json",
    "batch-05-final.json",
)


def audit() -> dict[str, Any]:
    traces: list[dict[str, Any]] = []
    fallback_codes: Counter[str] = Counter()
    decision_codes: Counter[str] = Counter()
    decisions = 0
    missing_raw_action = 0
    missing_repair_code = 0
    trace_rows: list[dict[str, Any]] = []

    for name in RUNS:
        payload = json.loads((RUN_DIR / name).read_text(encoding="utf-8"))
        payload_services = payload.get("servicesObserved", {})
        for trace in payload.get("traces", []):
            traces.append(trace)
            codes = list(trace.get("fallbackCodes", []))
            fallback_codes.update(codes)
            local_decisions = trace.get("decisions", [])
            trace_rows.append({
                "run": name,
                "traceId": trace.get("traceId"),
                "plannerModelUsedPayload": payload_services.get("plannerModelUsed"),
                "fallbackCodes": codes,
                "decisionCount": len(local_decisions),
                "decisionLevelRawActionFieldsPresent": all(
                    d.get("raw_action") is not None for d in local_decisions
                ),
                "decisionLevelRepairCodeFieldsPresent": all(
                    d.get("repair_code") is not None for d in local_decisions
                ),
            })
            for decision in local_decisions:
                decisions += 1
                decision_codes.update([decision.get("decisionCode") or "MISSING"])
                missing_raw_action += decision.get("raw_action") is None
                missing_repair_code += decision.get("repair_code") is None

    return {
        "schemaVersion": "p2j4-dpo-v4-policy-source-provenance-audit-v1",
        "status": "PROVENANCE_UNKNOWN_NOT_MODEL_ORIGIN",
        "trainingStarted": False,
        "pairConstructionStarted": False,
        "traceCount": len(traces),
        "decisionCount": decisions,
        "payloadPlannerModelUsedAllTrue": all(
            row["plannerModelUsedPayload"] is True for row in trace_rows
        ),
        "decisionLevel": {
            "rawActionPresentCount": decisions - missing_raw_action,
            "rawActionMissingCount": missing_raw_action,
            "repairCodePresentCount": decisions - missing_repair_code,
            "repairCodeMissingCount": missing_repair_code,
            "modelOriginEligibleCount": 0,
        },
        "fallbackCodeCounts": dict(sorted(fallback_codes.items())),
        "decisionCodeCounts": dict(sorted(decision_codes.items())),
        "classification": {
            "sourceClass": "controlled_runtime",
            "plannerOrigin": "provenance_unknown",
            "reason": "No per-decision raw model action to final action plus repair attribution is persisted.",
        },
        "traces": trace_rows,
        "nextAction": "retain_as_runtime_contract_asset_and_collect_only_targeted_provenance_complete_traces",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "traceCount": result["traceCount"],
        "decisionCount": result["decisionCount"],
        "rawActionMissingCount": result["decisionLevel"]["rawActionMissingCount"],
        "modelOriginEligibleCount": result["decisionLevel"]["modelOriginEligibleCount"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
