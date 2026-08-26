"""Create a resumable, success-preserving real Trace collection plan.

The plan is an execution manifest only.  It does not construct a runtime,
call a model, or alter a trace.  Each case occurs exactly once and each batch
has its own checkpoint; rerunning a completed checkpoint uses the existing
runner's explicit ``--resume`` path rather than repeating completed cases.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
TASK_SET = ROOT / "p2j4-dpo-v4-development-task-set-v1.json"
PRESERVED_CANARY = ROOT / "p2j4-dpo-v4-development-runs" / "canary-001-rescored.json"
PRESERVED_BATCH = ROOT / "p2j4-dpo-v4-development-runs" / "batch-01-final.json"


def _preserved_completed_cases() -> list[str]:
    """Return only auditable, offline-rescored PASS canary cases."""

    if not PRESERVED_CANARY.exists():
        return []
    payload = json.loads(PRESERVED_CANARY.read_text(encoding="utf-8"))
    if payload.get("mode") != "offline_rescore" or payload.get("summary", {}).get("fail") != 0:
        raise ValueError("DPO_V4_PRESERVED_CANARY_AUDIT_INVALID")
    canary_cases = [item["caseId"] for item in payload.get("scores", []) if item.get("status") == "PASS"]
    if not PRESERVED_BATCH.exists():
        return canary_cases
    batch = json.loads(PRESERVED_BATCH.read_text(encoding="utf-8"))
    if batch.get("status") != "FROZEN_ALL_PASS":
        raise ValueError("DPO_V4_PRESERVED_BATCH_AUDIT_INVALID")
    return sorted(set(canary_cases) | set(batch.get("completedCaseIds", [])))


def build() -> dict[str, Any]:
    payload = json.loads(TASK_SET.read_text(encoding="utf-8"))
    metadata = payload["developmentCaseMetadata"]
    preserved = _preserved_completed_cases()
    if not set(preserved).issubset(metadata):
        raise ValueError("DPO_V4_PRESERVED_CANARY_CASE_UNKNOWN")
    by_family: dict[str, list[str]] = defaultdict(list)
    for case in payload["cases"]:
        if case["caseId"] in preserved:
            continue
        by_family[metadata[case["caseId"]]["dpoV4Family"]].append(case["caseId"])

    # Interleave the scarce actions across five 16-case batches.  This gives
    # early evidence of coverage without treating an incomplete batch as a
    # training source, and avoids a single all-or-nothing 80-call process.
    batches: list[list[str]] = [[] for _ in range(5)]
    family_order = (
        "bounded_read", "stratified_state", "confounder_state",
        "cross_disease_state", "combined_legal_choice", "justified_stop",
    )
    cursor = 0
    for family in family_order:
        for case_id in by_family[family]:
            batches[cursor % len(batches)].append(case_id)
            cursor += 1
    for batch in batches:
        batch.sort()
    flattened = [case_id for batch in batches for case_id in batch]
    expected_pending = set(metadata) - set(preserved)
    if len(flattened) != len(set(flattened)) or set(flattened) != expected_pending:
        raise ValueError("DPO_V4_DEVELOPMENT_BATCH_PARTITION_INVALID")

    return {
        "schemaVersion": "p2j4-dpo-v4-development-execution-plan-v1",
        "status": "READY_NOT_EXECUTED",
        "taskSet": TASK_SET.name,
        "caseCount": len(metadata),
        "preservedCompletedCaseIds": preserved,
        "pendingCaseCount": len(flattened),
        "batchCount": len(batches),
        "sourcePolicy": {
            "successfulCasesNeverRerun": True,
            "failedOrInterruptedCasesResumeOnly": True,
            "allPassRequiredBeforePairExtraction": True,
            "evaluationSetsExcluded": ["Golden50", "Hard30", "Test70", "OOD30", "Runtime20"],
        },
        "batches": [
            {
                "batchId": f"dpo-v4-dev-batch-{index:02d}",
                "caseIds": batch,
                "caseCount": len(batch),
                "checkpoint": f"evals/p2j4-dpo-v4-development-runs/batch-{index:02d}.json",
                "status": "PENDING",
            }
            for index, batch in enumerate(batches, 1)
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"], "caseCount": result["caseCount"],
        "batchCount": result["batchCount"],
        "batchSizes": [item["caseCount"] for item in result["batches"]],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
