"""Stage DPO v4 review candidates without freezing preference pairs."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SFT_FILES = (
    ROOT / "sft-data" / "decision-freeze-v2-staging" / "train.jsonl",
    ROOT / "sft-data" / "decision-freeze-v2-staging" / "validation.jsonl",
)
TARGETED_FILES = (
    ROOT / "evals" / "p2j4-dpo-v4-targeted-weak-action-runs-final-model-pass-v1.json",
    ROOT / "evals" / "p2j4-dpo-v4-targeted-weak-action-runs-v2-final-pass-v1.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-collection-runs-v3-final-pass.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-collection-runs-v4-final-pass.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-cross-only-runs-v5-final-pass.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-collection-runs-v6-final-pass.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-collection-runs-v7-final-pass.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-collection-runs-v8-final-pass.json",
)
LEGACY_FILE = ROOT / "evals" / "p2j4-dpo-v4-real-source-audit-20260825.json"
REPAIR_DERIVED_FILE = ROOT / "evals" / "p2j4-dpo-v4-repair-derived-candidates-20260826.jsonl"


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:20]


def _sft_candidates() -> list[dict[str, Any]]:
    rows = []
    for path in SFT_FILES:
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            metadata = record.get("metadata", {})
            if metadata.get("provenance") != "enriched_real_trace_v5":
                continue
            state = json.loads(record["messages"][1]["content"])["policy_state"]
            target = json.loads(record["messages"][2]["content"])
            chosen = target["selected_action"]
            rejected = metadata.get("dpo_rejected_action")
            rows.append({
                "source_kind": "real_sft_state",
                "source_trace_id": metadata.get("source_trace_id"),
                "state_signature": metadata.get("state_signature") or _hash(state),
                "task_family": metadata.get("task_family"),
                "task_kind": state.get("task_kind"),
                "goal_code": state.get("goal_code"),
                "observation_flags": state.get("observation_flags", []),
                "history_actions": state.get("history_actions", []),
                "candidate_actions": state.get("candidate_actions", []),
                "state_summary": state.get("state_summary", ""),
                "planner_origin": "provenance_unknown",
                "chosen_action": chosen,
                "rejected_action": rejected,
                "preference_type": None,
                "preference_dimension": None,
                "preference_rationale": None,
                "review_status": "REVIEW_REQUIRED",
            })
    return rows


def _targeted_candidates() -> list[dict[str, Any]]:
    rows = []
    for path in TARGETED_FILES:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for trace in payload.get("traces", []):
            for decision in trace.get("decisions", []):
                chosen = decision.get("final_action") or decision.get("chosenAction")
                actions = decision.get("allowedActions", [])
                if not chosen or not actions:
                    continue
                state = {
                    "observationStateCode": decision.get("observationStateCode"),
                    "candidateActions": actions,
                    "historySummary": decision.get("state_summary", ""),
                }
                alternatives = [action for action in actions if action != chosen]
                for rejected in alternatives:
                    rows.append({
                    "source_kind": "targeted_real_state_difference",
                    "source_trace_id": trace.get("traceId"),
                    "state_signature": _hash(state),
                    "task_family": "targeted_weak_action",
                    "task_kind": "open_exploration",
                    "goal_code": None,
                    "observation_flags": [decision.get("observationStateCode")],
                    "history_actions": [],
                    "candidate_actions": actions,
                    "state_summary": decision.get("state_summary", ""),
                    "planner_origin": decision.get("planner_origin", "provenance_unknown"),
                    "raw_action": decision.get("raw_action"),
                    "final_action": decision.get("final_action"),
                    "repair_code": decision.get("repair_code"),
                    "chosen_action": chosen,
                    "rejected_action": rejected,
                    "preference_type": None,
                    "preference_dimension": None,
                    "preference_rationale": None,
                    "review_status": "REVIEW_REQUIRED_NO_PREFERENCE_RATIONALE",
                    })
    return rows


def _legacy_candidates() -> list[dict[str, Any]]:
    payload = json.loads(LEGACY_FILE.read_text(encoding="utf-8"))
    rows = []
    for item in payload.get("candidates", []):
        actions = item.get("candidateActions", [])
        chosen = item.get("selectedAction")
        if not actions or not chosen:
            continue
        state = {
            "taskKind": item.get("taskKind"), "goalCode": item.get("goalCode"),
            "observationFlags": item.get("observationFlags", []),
            "historyActions": item.get("historyActions", []),
            "candidateActions": actions, "selectedAction": chosen,
        }
        signature = _hash(state)
        for rejected in item.get("legalUnperformedAlternatives", []):
            if rejected == chosen or rejected not in actions:
                continue
            rows.append({
                "source_kind": "legacy_real_state",
                "source_trace_id": item.get("sourceTraceId"),
                "state_signature": signature,
                "task_family": item.get("taskFamily"),
                "task_kind": item.get("taskKind"),
                "goal_code": item.get("goalCode"),
                "observation_flags": item.get("observationFlags", []),
                "history_actions": item.get("historyActions", []),
                "candidate_actions": actions,
                "state_summary": item.get("stateSummary", ""),
                "planner_origin": "provenance_unknown",
                "chosen_action": chosen,
                "rejected_action": rejected,
                "preference_type": None,
                "preference_dimension": None,
                "preference_rationale": None,
                "review_status": "REVIEW_REQUIRED_NO_PREFERENCE_RATIONALE",
            })
    return rows


def _repair_derived_candidates() -> list[dict[str, Any]]:
    if not REPAIR_DERIVED_FILE.exists():
        return []
    return [
        json.loads(line)
        for line in REPAIR_DERIVED_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = [*_sft_candidates(), *_legacy_candidates(), *_targeted_candidates(), *_repair_derived_candidates()]
    unique_states: dict[str, dict[str, Any]] = {}
    unique_pairs: dict[tuple[str, str, str | None], dict[str, Any]] = {}
    for row in raw:
        unique_states.setdefault(row["state_signature"], row)
        pair_key = (row["state_signature"], row["chosen_action"], row.get("rejected_action"))
        unique_pairs.setdefault(pair_key, row)
    rows = list(unique_pairs.values())
    pair_action_counts = Counter(row["chosen_action"] for row in rows)
    model_origin_pairs = sum(row["planner_origin"] == "model" for row in rows)
    model_origin_usable_pairs = sum(
        row["planner_origin"] == "model"
        and row.get("preference_type") == "action"
        and row.get("preference_dimension")
        and row.get("preference_rationale")
        and row.get("chosen_action") in row.get("candidate_actions", [])
        and row.get("rejected_action") in row.get("candidate_actions", [])
        and row.get("chosen_action") != row.get("rejected_action")
        for row in rows
    )
    width_4_to_7 = sum(4 <= len(row["candidate_actions"]) <= 7 for row in rows)
    summary = {
        "schemaVersion": "p2j4-dpo-v4-review-candidates-audit-v1",
        "status": "REVIEW_ONLY_NOT_DPO_PAIRS",
        "trainingStarted": False,
        "pairConstructionStarted": False,
        "rawCandidateCount": len(raw),
        "uniqueCandidateCount": len(unique_states),
        "uniquePairCandidateCount": len(unique_pairs),
        "pairChosenActionCounts": dict(pair_action_counts),
        "modelOriginPairCount": model_origin_pairs,
        "modelOriginPairRatio": model_origin_pairs / len(rows) if rows else 0.0,
        "modelOriginUsablePairCount": model_origin_usable_pairs,
        "modelOriginUsablePairRatio": model_origin_usable_pairs / len(rows) if rows else 0.0,
        "width4to7PairCount": width_4_to_7,
        "width4to7PairRatio": width_4_to_7 / len(rows) if rows else 0.0,
        "dpoV4Gate": {
            "pairCountRangePass": 500 <= len(rows) <= 700,
            "modelOrigin70PercentPass": model_origin_pairs / len(rows) >= 0.70 if rows else False,
            "preferenceRationalePass": False,
            "trainingAllowed": False,
        },
        "sourceKindCounts": dict(Counter(row["source_kind"] for row in rows)),
        "plannerOriginCounts": dict(Counter(row["planner_origin"] for row in rows)),
        "chosenActionCounts": dict(Counter(row["chosen_action"] for row in rows)),
        "candidateWidthCounts": dict(Counter(len(row["candidate_actions"]) for row in rows)),
        "basicSftChosenRejectedEligible": sum(
            row["source_kind"] == "real_sft_state"
            and row.get("rejected_action") in row["candidate_actions"]
            and row.get("rejected_action") != row["chosen_action"]
            for row in rows
        ),
        "missingPreferenceRationaleCount": sum(not row.get("preference_rationale") for row in rows),
        "structuredRationaleCount": sum(
            bool(row.get("preference_dimension"))
            and bool(row.get("preference_codes"))
            and bool(row.get("preference_rationale"))
            for row in rows
        ),
        "nextAction": "human_or_approved_review_of_preference_dimension_and_rationale",
    }
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--audit-output", type=Path, required=True); args = parser.parse_args()
    rows, summary = build()
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    args.audit_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
