"""Extract auditable action-repair-derived DPO candidates.

This file is intentionally fail-closed.  A repair-derived candidate exists
only when a model-origin decision contains both a raw action and a different
runtime final action, and the runtime emitted a repair code.  Stop-reason
repairs, semantic-guard decisions, and missing provenance are not converted
into preference pairs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
INPUTS = (
    ROOT / "evals" / "p2j4-dpo-v4-targeted-weak-action-runs-final-model-pass-v1.json",
    ROOT / "evals" / "p2j4-dpo-v4-targeted-weak-action-runs-v2-final-pass-v1.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-collection-runs-v3-final-pass.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-collection-runs-v4-final-pass.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-cross-only-runs-v5-final-pass.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-collection-runs-v6-final-pass.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-collection-runs-v7-final-pass.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-collection-runs-v8-final-pass.json",
)

ACTION_REPAIR_CODES = {
    "SCIENTIFIC_PLANNER_STRATIFIED_ORDER_REPAIRED": "evidence_requirement",
    "SCIENTIFIC_PLANNER_REPEATED_READ_REPAIRED": "efficiency",
    "SCIENTIFIC_RUNTIME_OBSERVATION_REBOUND": "evidence_requirement",
}


def _hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _trace_files(paths: list[Path] | None) -> list[Path]:
    return paths or list(INPUTS)


def build(paths: list[Path] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scanned = 0
    eligible = 0
    for path in _trace_files(paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for trace in payload.get("traces", []):
            for decision in trace.get("decisions", []):
                scanned += 1
                raw = decision.get("raw_action")
                final = decision.get("final_action") or decision.get("selected_action")
                codes = list(dict.fromkeys(decision.get("repair_codes") or ([decision["repair_code"]] if decision.get("repair_code") else [])))
                action_codes = [code for code in codes if code in ACTION_REPAIR_CODES]
                if decision.get("planner_origin") != "model":
                    continue
                if not raw or not final or raw == final or not action_codes:
                    continue
                eligible += 1
                allowed = list(dict.fromkeys(decision.get("allowedActions") or []))
                state = {
                    "observationStateCode": decision.get("observationStateCode"),
                    "observationEvidenceBindingCount": decision.get("observationEvidenceBindingCount"),
                    "observationSourceRoutes": decision.get("observationSourceRoutes", []),
                    "historySummary": decision.get("state_summary", ""),
                    "candidateActions": allowed,
                }
                dimension = ACTION_REPAIR_CODES[action_codes[0]]
                rows.append({
                    "source_kind": "runtime_repair_derived",
                    "source_trace_id": trace.get("traceId"),
                    "source_file": path.name,
                    "state_signature": _hash(state),
                    "task_family": "runtime_repair_contract",
                    "task_kind": "open_exploration",
                    "observation_flags": [decision.get("observationStateCode")],
                    "history_actions": [],
                    "candidate_actions": allowed,
                    "state_summary": decision.get("state_summary", ""),
                    "planner_origin": "model",
                    "raw_action": raw,
                    "final_action": final,
                    "repair_code": action_codes[0],
                    "repair_codes": codes,
                    "chosen_action": final,
                    "rejected_action": raw,
                    "chosen_origin": "policy_contract",
                    "rejected_origin": "model_raw",
                    "preference_type": "action",
                    "preference_dimension": dimension,
                    "preference_codes": [action_codes[0]],
                    "preference_rationale": f"Runtime contract selected {final} over the model raw action {raw}: {action_codes[0]}.",
                    "review_status": "CONTRACT_REVIEW_REQUIRED",
                })

    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        unique.setdefault((row["state_signature"], row["chosen_action"], row["rejected_action"]), row)
    result = list(unique.values())
    audit = {
        "schemaVersion": "p2j4-dpo-v4-repair-derived-candidates-audit-v1",
        "status": "REVIEW_REQUIRED_NO_ACTION_REPAIR_CANDIDATES" if not result else "REVIEW_REQUIRED",
        "trainingStarted": False,
        "pairConstructionStarted": False,
        "inputFiles": [str(p) for p in _trace_files(paths)],
        "scannedDecisionCount": scanned,
        "rawEligibleCandidateCount": eligible,
        "uniqueCandidateCount": len(result),
        "chosenActionCounts": dict(Counter(row["chosen_action"] for row in result)),
        "rejectedActionCounts": dict(Counter(row["rejected_action"] for row in result)),
        "repairCodeCounts": dict(Counter(row["repair_code"] for row in result)),
        "rule": "planner_origin=model AND raw_action != final_action AND action repair code present",
        "excludedStopReasonRepairs": True,
        "nextAction": "collect_or_audit_true_model_action_replacements",
    }
    return result, audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append")
    args = parser.parse_args()
    rows, audit = build(args.input)
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")
    args.audit_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ("status", "scannedDecisionCount", "uniqueCandidateCount")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
