"""Audit existing SFT/Trace sources before DPO v4 pair construction.

This is an audit-only asset. It does not create preference pairs, alter SFT
data, or infer per-decision ``planner_origin`` from a payload-level planner
flag.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
POLICY_RUNS = ROOT / "evals" / "p2j4-dpo-v4-policy-source-runs"
SFT_FILES = (
    ROOT / "sft-data" / "decision-freeze-v2-staging" / "train.jsonl",
    ROOT / "sft-data" / "decision-freeze-v2-staging" / "validation.jsonl",
)
NEW_RUNS = (
    "canary-compact-001.json",
    "batch-01-final.json",
    "batch-02-final.json",
    "batch-03-final.json",
    "batch-04-final.json",
    "batch-05-final.json",
)
TARGETED_RUNS = (
    "p2j4-dpo-v4-targeted-weak-action-runs-final-model-pass-v1.json",
    "p2j4-dpo-v4-targeted-weak-action-runs-v2-final-pass-v1.json",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _state_signature(state: dict[str, Any], selected: str, alternatives: list[str], stop: str | None) -> str:
    value = {
        "task_kind": state.get("task_kind"),
        "goal_code": state.get("goal_code"),
        "observation_flags": state.get("observation_flags", []),
        "history_actions": state.get("history_actions", []),
        "candidate_actions": state.get("candidate_actions", []),
        "state_summary": state.get("state_summary", ""),
        "selected_action": selected,
        "alternative_actions": alternatives,
        "stop_reason": stop,
    }
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]


def _canonical_state_signature(state: dict[str, Any], selected: str, stop: str | None) -> str:
    """Cross-source key using stable structured fields only.

    Alternative lists and free-form state summaries differed between the
    legacy audit and SFT export. They are useful for pair review, but must
    not prevent recognizing the same Trace/action state across exports.
    """
    value = {
        "task_kind": state.get("task_kind"),
        "goal_code": state.get("goal_code"),
        "observation_flags": state.get("observation_flags", []),
        "history_actions": state.get("history_actions", []),
        "candidate_actions": state.get("candidate_actions", []),
        "selected_action": selected,
        "stop_reason": stop,
    }
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]


def _sft_real_rows() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for path in SFT_FILES:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("metadata", {}).get("provenance") != "enriched_real_trace_v5":
                continue
            state = json.loads(row["messages"][1]["content"])["policy_state"]
            target = json.loads(row["messages"][2]["content"])
            result.append({
                "source": "real_sft_v2",
                "plannerOrigin": "unverified",
                "sourceTraceId": row["metadata"].get("source_trace_id"),
                "stateSignature": _state_signature(
                    state,
                    target["selected_action"],
                    target["alternative_actions"],
                    target.get("stop_reason"),
                ),
                "canonicalStateSignature": _canonical_state_signature(
                    state, target["selected_action"], target.get("stop_reason")
                ),
                "selectedAction": target["selected_action"],
                "candidateWidth": len(state.get("candidate_actions", [])),
                "rawAction": None,
                "finalAction": target["selected_action"],
                "repairCode": None,
            })
    return result


def _legacy_real_rows() -> list[dict[str, Any]]:
    payload = _read_json(ROOT / "evals" / "p2j4-dpo-v4-real-source-audit-20260825.json")
    result: list[dict[str, Any]] = []
    for item in payload.get("candidates", []):
        actions = item.get("candidateActions", item.get("candidate_actions", item.get("allowedActions", [])))
        selected = item.get("selectedAction", item.get("selected_action"))
        if not actions or not selected:
            continue
        alternatives = item.get("legalUnperformedAlternatives", item.get("alternative_actions", []))
        stop = item.get("stopReason", item.get("stop_reason"))
        normalized_state = {
            "task_kind": item.get("taskKind", item.get("task_kind")),
            "goal_code": item.get("goalCode", item.get("goal_code")),
            "observation_flags": item.get("observationFlags", item.get("observation_flags", [])),
            "history_actions": item.get("historyActions", item.get("history_actions", [])),
            "candidate_actions": actions,
            "state_summary": item.get("stateSummary", item.get("state_summary", "")),
        }
        result.append({
            "source": "legacy_real_source_audit",
            "plannerOrigin": "unverified",
            "sourceTraceId": item.get("sourceTraceId"),
                "stateSignature": _state_signature(
                normalized_state,
                selected,
                alternatives,
                stop,
                ),
                "canonicalStateSignature": _canonical_state_signature(normalized_state, selected, stop),
                "selectedAction": selected,
                "candidateWidth": len(actions),
                "rawAction": None,
                "finalAction": selected,
                "repairCode": None,
        })
    return result


def _policy_source_rows() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for name in NEW_RUNS:
        payload = _read_json(POLICY_RUNS / name)
        for trace in payload.get("traces", []):
            for decision in trace.get("decisions", []):
                selected = decision.get("chosenAction")
                actions = decision.get("allowedActions", [])
                if not selected or not actions:
                    continue
                fallback_codes = trace.get("fallbackCodes", [])
                result.append({
                    "source": "new_policy_source_trace",
                    "sourceClass": "controlled_runtime",
                    "plannerOrigin": "provenance_unknown",
                    "sourceTraceId": trace.get("traceId"),
                    "stateSignature": hashlib.sha256(
                        json.dumps({
                            "stateSummary": decision.get("state_summary", ""),
                            "candidateActions": actions,
                            "chosenAction": selected,
                        }, ensure_ascii=False, sort_keys=True).encode("utf-8")
                    ).hexdigest()[:20],
                    "canonicalStateSignature": hashlib.sha256(
                        json.dumps({
                            "observation_state": decision.get("observationStateCode"),
                            "candidateActions": actions,
                            "chosenAction": selected,
                            "stopReason": decision.get("stop_reason"),
                        }, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
                    ).hexdigest()[:20],
                    "selectedAction": selected,
                    "candidateWidth": len(actions),
                    "rawAction": decision.get("raw_action"),
                    "finalAction": selected,
                    "repairCode": decision.get("repair_code"),
                    "traceFallbackCodes": fallback_codes,
                    "decisionLevelRepairEvidence": bool(decision.get("raw_action") or decision.get("repair_code")),
                })
    return result


def _targeted_source_rows() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for targeted_run in TARGETED_RUNS:
        payload = _read_json(ROOT / "evals" / targeted_run)
        for trace in payload.get("traces", []):
            for decision in trace.get("decisions", []):
                selected = decision.get("chosenAction")
                actions = decision.get("allowedActions", [])
                if not selected or not actions:
                    continue
                canonical = {
                    "observationStateCode": decision.get("observationStateCode"),
                    "candidateActions": actions,
                    "chosenAction": selected,
                    "historySummary": decision.get("state_summary", ""),
                }
                signature = hashlib.sha256(
                    json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest()[:20]
                result.append({
                    "source": "targeted_weak_action_trace",
                    "sourceClass": "targeted_real_state_difference_candidate",
                    "plannerOrigin": decision.get("planner_origin", "provenance_unknown"),
                    "sourceTraceId": trace.get("traceId"),
                    "stateSignature": signature,
                    "canonicalStateSignature": signature,
                    "selectedAction": selected,
                    "candidateWidth": len(actions),
                    "rawAction": decision.get("raw_action"),
                    "finalAction": decision.get("final_action", selected),
                    "repairCode": decision.get("repair_code"),
                    "repairCodes": decision.get("repair_codes", []),
                })
    return result


def audit() -> dict[str, Any]:
    real_sft = _sft_real_rows()
    legacy = _legacy_real_rows()
    policy = _policy_source_rows()
    targeted = _targeted_source_rows()
    all_rows = [*legacy, *real_sft, *policy, *targeted]
    legacy_sft = [*legacy, *real_sft]
    legacy_sft_by_canonical = {row["canonicalStateSignature"]: row for row in legacy_sft}
    unique_by_signature = {
        row["canonicalStateSignature"]: row
        for row in [*legacy_sft_by_canonical.values(), *policy, *targeted]
    }
    exact_trace_action_overlap = len({(row["sourceTraceId"], row["selectedAction"]) for row in legacy} &
                                    {(row["sourceTraceId"], row["selectedAction"]) for row in real_sft})
    action_counts = Counter(row["selectedAction"] for row in unique_by_signature.values())
    width_counts = Counter(row["candidateWidth"] for row in unique_by_signature.values())
    source_counts = Counter(row["source"] for row in unique_by_signature.values())
    origin_counts = Counter(row["plannerOrigin"] for row in unique_by_signature.values())
    trace_counts = {
        source: len({row["sourceTraceId"] for row in rows if row["sourceTraceId"]})
        for source, rows in {
            "legacy_real_source_audit": legacy,
            "real_sft_v2": real_sft,
            "new_policy_source_trace": policy,
            "targeted_weak_action_trace": targeted,
        }.items()
    }
    return {
        "schemaVersion": "p2j4-dpo-v4-existing-source-pool-audit-v1",
        "status": "PAIR_REVIEW_REQUIRED",
        "trainingStarted": False,
        "pairConstructionStarted": False,
        "sourceCountsRaw": {
            "legacyRealRuntimeLikeStates": len(legacy),
            "realSftRows": len(real_sft),
            "newPolicySourceDecisionStates": len(policy),
            "targetedWeakActionDecisionStates": len(targeted),
            "controlledSftRowsExcluded": 500,
            "dpoV3PairsExcluded": 750,
        },
        "sourceTraceCountsRaw": trace_counts,
        "legacySftCanonicalUnionStateCount": len(legacy_sft_by_canonical),
        "legacySftExactTraceActionOverlapCount": exact_trace_action_overlap,
        "uniqueStateCountAfterCanonicalDedup": len(unique_by_signature),
        "sourceCountsAfterCanonicalDedup": dict(sorted(source_counts.items())),
        "plannerOriginCountsAfterCanonicalDedup": dict(sorted(origin_counts.items())),
        "chosenActionCountsAfterCanonicalDedup": dict(sorted(action_counts.items())),
        "candidateWidthCountsAfterCanonicalDedup": dict(sorted(width_counts.items())),
        "width4to7Count": sum(count for width, count in width_counts.items() if 4 <= width <= 7),
        "boundedWidth2to3Count": sum(count for width, count in width_counts.items() if 2 <= width <= 3),
        "reviewGates": {
            "perDecisionPlannerOriginAuditRequired": True,
            "newPolicySourceCountsAsControlledRuntimeUntilProven": True,
            "controlledSftMayNotCountAsModelOrigin": True,
            "dpoV3MayNotBeReused": True,
            "taskFamilySplitRequired": True,
            "manualPreferenceReviewRequired": True,
        },
        "nextAction": "repair_cross_source_dedup_then_audit_decision_provenance_before_pair_construction",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "uniqueStateCountAfterCanonicalDedup": result["uniqueStateCountAfterCanonicalDedup"],
        "sourceCountsAfterCanonicalDedup": result["sourceCountsAfterCanonicalDedup"],
        "plannerOriginCountsAfterCanonicalDedup": result["plannerOriginCountsAfterCanonicalDedup"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
