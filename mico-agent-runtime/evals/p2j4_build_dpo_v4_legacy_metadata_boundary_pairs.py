"""Select a small, explicitly labelled metadata-first boundary subset.

These source rows originate from the legacy real-trace audit but lack
per-decision planner provenance.  They are therefore never counted as
model-origin.  The subset is capped so obvious metadata-first boundaries stay
minority evidence rather than becoming the DPO objective.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ACTIONS = {
    "inspect_cohort", "execute_read_query", "compare_groups", "analyze_projection",
    "stratified_analysis", "adjust_confounders", "cross_project_validate",
    "cross_disease_validate", "retrieve_evidence", "finish",
}
REJECT_ORDER = (
    "compare_groups", "analyze_projection", "stratified_analysis", "adjust_confounders",
    "cross_project_validate", "cross_disease_validate", "retrieve_evidence", "finish",
)


def _hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _state(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_kind": row["taskKind"],
        "goal_code": row["goalCode"],
        "observation_flags": list(row["observationFlags"]),
        "history_actions": list(row["historyActions"]),
        "candidate_actions": list(row["candidateActions"]),
        "state_summary": row["stateSummary"],
    }


def _prompt(state: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "Choose exactly one next action from the supplied de-identified policy state."},
        {"role": "user", "content": json.dumps({"policy_state": state}, ensure_ascii=False, sort_keys=True)},
    ]


def _eligible(row: dict[str, Any]) -> bool:
    if row.get("hardCaseClass") or row.get("selectedAction") != "inspect_cohort":
        return False
    try:
        state = _state(row)
    except (KeyError, TypeError):
        return False
    flags = set(state["observation_flags"])
    candidates = state["candidate_actions"]
    return (
        state["history_actions"] == []
        and {"NO_OBSERVATION", "METADATA_FIRST"}.issubset(flags)
        and 4 <= len(candidates) <= 7
        and len(candidates) == len(set(candidates))
        and all(action in ACTIONS for action in candidates)
        and any(action != "inspect_cohort" for action in row.get("legalUnperformedAlternatives", []))
    )


def _rejected(row: dict[str, Any], index: int) -> str | None:
    legal = set(row.get("legalUnperformedAlternatives") or [])
    candidates = set(row.get("candidateActions") or [])
    ordered = [action for action in REJECT_ORDER if action in legal and action in candidates]
    return ordered[index % len(ordered)] if ordered else None


def build(path: Path, target: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    excluded: Counter[str] = Counter()
    for row in payload.get("candidates", []):
        if not _eligible(row):
            excluded["NOT_METADATA_BOUNDARY_ELIGIBLE"] += 1
            continue
        grouped[row["taskFamily"]].append(row)
    for family in grouped:
        grouped[family].sort(key=lambda item: (item["sourceTraceId"], item["stateSummary"]))
    families = sorted(grouped)
    selected: list[dict[str, Any]] = []
    signatures: set[str] = set()
    cursors: Counter[str] = Counter()
    while len(selected) < target:
        progressed = False
        for family in families:
            items = grouped[family]
            while cursors[family] < len(items):
                row = items[cursors[family]]
                cursors[family] += 1
                state = _state(row)
                signature = _hash(state)
                if signature in signatures:
                    excluded["DUPLICATE_NORMALIZED_STATE"] += 1
                    continue
                rejected = _rejected(row, len(selected))
                if rejected is None:
                    excluded["NO_LEGAL_REJECTED_ACTION"] += 1
                    continue
                rationale = (
                    "No validated observation exists and the state explicitly requires metadata first, "
                    "so inspect_cohort establishes the bounded cohort context before any downstream action."
                )
                selected.append({
                    "id": "dpo-v4-legacy-metadata-" + _hash([signature, "inspect_cohort", rejected]),
                    "prompt": _prompt(state),
                    "chosen": json.dumps({"selected_action": "inspect_cohort", "decision_reason": rationale, "alternative_actions": [rejected], "stop_reason": None}, ensure_ascii=False, sort_keys=True),
                    "rejected": json.dumps({"selected_action": rejected, "decision_reason": "Listed unperformed alternative retained as a metadata-first boundary comparison.", "alternative_actions": ["inspect_cohort"], "stop_reason": None}, ensure_ascii=False, sort_keys=True),
                    **state,
                    "state_signature": signature,
                    "task_family": row["taskFamily"],
                    "source_trace_id": row["sourceTraceId"],
                    "source_file": path.name,
                    "case_id": None,
                    "source_kind": "hard_development_trace",
                    "scenario_class": "legacy_metadata_boundary",
                    "planner_origin": None,
                    "provenance_class": "provenance_unknown_legacy",
                    "raw_action": None,
                    "final_action": "inspect_cohort",
                    "chosen_action": "inspect_cohort",
                    "rejected_action": rejected,
                    "hard_case_class": "metadata_first_boundary",
                    "repair_codes": [],
                    "preference_type": "action",
                    "preference_dimension": "efficiency",
                    "preference_codes": ["METADATA_FIRST_REQUIRED", "NO_VALIDATED_OBSERVATION"],
                    "preference_rationale": rationale,
                    "preference_rationale_source": "legacy_state_contract_metadata_boundary_v1",
                    "review_status": "SEMANTIC_REVIEW_REQUIRED",
                })
                signatures.add(signature)
                progressed = True
                break
            if len(selected) >= target:
                break
        if not progressed:
            break
    audit = {
        "schemaVersion": "p2j4-dpo-v4-legacy-metadata-boundary-audit-v1",
        "status": "SEMANTIC_REVIEW_REQUIRED" if len(selected) == target else "BLOCKED",
        "trainingStarted": False,
        "pairConstructionStarted": True,
        "targetPairCount": target,
        "selectedPairCount": len(selected),
        "uniqueStateCount": len(signatures),
        "taskFamilyCounts": dict(Counter(item["task_family"] for item in selected)),
        "rejectedActionCounts": dict(Counter(item["rejected_action"] for item in selected)),
        "excludedCounts": dict(excluded),
        "provenanceUnknown": True,
        "rules": [
            "only NO_OBSERVATION plus METADATA_FIRST states are selected",
            "only non-hard-labelled legacy source rows are used",
            "selected action is always inspect_cohort and is never claimed model-origin",
            "one pair per normalized state",
            "target is capped to keep this obvious boundary below five percent of a 500-pair freeze",
        ],
    }
    return selected, audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--target", type=int, default=23)
    args = parser.parse_args()
    if not 1 <= args.target <= 24:
        raise SystemExit("DPO_V4_METADATA_BOUNDARY_TARGET_MUST_BE_1_TO_24")
    rows, audit = build(args.input, args.target)
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    args.audit_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: audit[key] for key in ("status", "selectedPairCount", "uniqueStateCount", "rejectedActionCounts", "excludedCounts")}, ensure_ascii=False))
    return 0 if audit["status"] != "BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
