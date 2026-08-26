"""Fail-closed audit for the flat DPO-v4 staging schema."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ACTIONS = {
    "inspect_cohort", "execute_read_query", "compare_groups", "analyze_projection",
    "stratified_analysis", "adjust_confounders", "cross_project_validate",
    "cross_disease_validate", "retrieve_evidence", "finish",
}
SOURCES = {"model_origin_policy_trace", "hard_development_trace"}
DIMENSIONS = {"efficiency", "evidence_requirement", "exploration_depth", "reason_quality", "stop_boundary"}
# A raw action selected by the model is not automatically a real-policy
# source.  Matrix/weak-action tasks deliberately constrain the route and are
# useful only as controlled contract data.  Keep this list narrow and
# explicit: adding a scenario requires a separate provenance review.
REAL_POLICY_SCENARIOS = {"real_policy_source"}


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _audit(path: Path) -> dict[str, Any]:
    rows = _read(path)
    errors: Counter[str] = Counter()
    chosen_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    scenario_counts: Counter[str] = Counter()
    width_counts: Counter[int] = Counter()
    dimension_counts: Counter[str] = Counter()
    state_counts: Counter[str] = Counter()
    split_states: dict[str, set[str]] = defaultdict(set)
    family_splits: dict[str, set[str]] = defaultdict(set)
    pair_keys: set[tuple[str, str, str]] = set()
    pending_review = 0
    model_origin_valid = 0
    real_policy_valid = 0
    hard_valid = 0

    required = {
        "id", "prompt", "chosen", "rejected", "task_kind", "goal_code", "task_family",
        "observation_flags", "history_actions", "candidate_actions", "state_summary",
        "state_signature", "source_kind", "scenario_class", "provenance_class",
        "chosen_action", "rejected_action", "preference_type", "preference_dimension",
        "preference_codes", "preference_rationale", "preference_rationale_source",
        "review_status", "reviewer_label", "review_basis", "split", "state_group_id",
    }
    for row in rows:
        missing = required - set(row)
        if missing:
            errors["MISSING_FIELDS"] += 1
            continue
        source = row["source_kind"]
        source_counts[source] += 1
        scenario_counts[row["scenario_class"]] += 1
        width = len(row["candidate_actions"]) if isinstance(row["candidate_actions"], list) else -1
        width_counts[width] += 1
        dimension_counts[row["preference_dimension"]] += 1
        if row["split"] not in {"train", "val"}:
            errors["SPLIT_LABEL_INVALID"] += 1
        split_states[row["split"]].add(row["state_signature"])
        family_splits[row["task_family"]].add(row["split"])
        if row["review_status"] not in {"RULE_AUDITED", "FROZEN_APPROVED"}:
            pending_review += 1
        candidates = row["candidate_actions"]
        if not isinstance(candidates, list) or not 4 <= len(candidates) <= 7:
            errors["CANDIDATE_WIDTH_INVALID"] += 1
        elif len(candidates) != len(set(candidates)) or any(action not in ACTIONS for action in candidates):
            errors["CANDIDATE_ACTION_INVALID"] += 1
        if row["chosen_action"] not in candidates or row["rejected_action"] not in candidates:
            errors["PAIR_ACTION_NOT_CANDIDATE"] += 1
        if row["chosen_action"] == row["rejected_action"]:
            errors["ACTION_PAIR_UNCHANGED"] += 1
        try:
            chosen = json.loads(row["chosen"])
            rejected = json.loads(row["rejected"])
            state = json.loads(row["prompt"][1]["content"])["policy_state"]
        except (IndexError, KeyError, TypeError, json.JSONDecodeError):
            errors["RECORD_SHAPE_INVALID"] += 1
            continue
        if chosen.get("selected_action") != row["chosen_action"] or rejected.get("selected_action") != row["rejected_action"]:
            errors["COMPLETION_ACTION_MISMATCH"] += 1
        for field in ("task_kind", "goal_code", "observation_flags", "history_actions", "candidate_actions", "state_summary"):
            if state.get(field) != row[field]:
                errors["PROMPT_STATE_MISMATCH_" + field] += 1
        key = (row["state_signature"], row["chosen_action"], row["rejected_action"])
        if key in pair_keys:
            errors["DUPLICATE_PAIR_KEY"] += 1
        pair_keys.add(key)
        state_counts[row["state_signature"]] += 1
        if state_counts[row["state_signature"]] > 3:
            errors["STATE_MULTIPLICITY_ABOVE_3"] += 1
        if source not in SOURCES:
            errors["SOURCE_KIND_INVALID"] += 1
        if row["preference_type"] != "action":
            errors["PREFERENCE_TYPE_INVALID"] += 1
        if row["preference_dimension"] not in DIMENSIONS:
            errors["PREFERENCE_DIMENSION_INVALID"] += 1
        if not row["preference_codes"] or not row["preference_rationale"] or not row["preference_rationale_source"]:
            errors["PREFERENCE_RATIONALE_INCOMPLETE"] += 1
        chosen_counts[row["chosen_action"]] += 1
        if source == "model_origin_policy_trace":
            ok = (
                row.get("planner_origin") == "model"
                and row.get("provenance_class") == "model_origin_raw"
                and row.get("raw_action") == row.get("final_action") == row["chosen_action"]
                and not row.get("repair_codes")
            )
            if ok:
                model_origin_valid += 1
                if row.get("scenario_class") in REAL_POLICY_SCENARIOS:
                    real_policy_valid += 1
            else:
                errors["MODEL_PROVENANCE_CONTRACT_INVALID"] += 1
        elif source == "hard_development_trace":
            if row["scenario_class"] == "runtime_repair_derived":
                ok = (
                    row.get("planner_origin") == "model"
                    and row.get("provenance_class") == "runtime_repair_derived"
                    and row.get("raw_action") == row["rejected_action"]
                    and row.get("final_action") == row["chosen_action"]
                    and bool(row.get("repair_codes"))
                )
            elif row["scenario_class"] in {"legacy_hard_case", "legacy_metadata_boundary"}:
                ok = (
                    row.get("planner_origin") is None
                    and row.get("provenance_class") == "provenance_unknown_legacy"
                    and row.get("raw_action") is None
                    and bool(row.get("hard_case_class"))
                )
                if row["scenario_class"] == "legacy_metadata_boundary":
                    ok = ok and (
                        row.get("chosen_action") == "inspect_cohort"
                        and row.get("history_actions") == []
                        and {"NO_OBSERVATION", "METADATA_FIRST"}.issubset(set(row.get("observation_flags") or []))
                    )
            else:
                ok = False
            if ok:
                hard_valid += 1
            else:
                errors["HARD_PROVENANCE_CONTRACT_INVALID"] += 1

    total = len(rows)
    model_rate = source_counts["model_origin_policy_trace"] / total if total else 0.0
    real_policy_rate = real_policy_valid / total if total else 0.0
    hard_rate = source_counts["hard_development_trace"] / total if total else 0.0
    positive_counts = [chosen_counts[action] for action in ACTIONS]
    if total < 500 or total > 700:
        errors["PAIR_COUNT_OUTSIDE_500_700"] += 1
    if set(chosen_counts) != ACTIONS:
        errors["CHOSEN_ACTION_COVERAGE_INCOMPLETE"] += 1
    if not positive_counts or min(positive_counts) == 0 or max(positive_counts) > 3 * min(positive_counts):
        errors["CHOSEN_ACTION_IMBALANCE_OVER_3X"] += 1
    if total and sum(count for width, count in width_counts.items() if 4 <= width <= 7) / total < 0.85:
        errors["CANDIDATE_WIDTH_BELOW_85_PERCENT"] += 1
    if total and model_rate < 0.70:
        errors["MODEL_ORIGIN_BELOW_70_PERCENT"] += 1
    if total and real_policy_rate < 0.70:
        errors["REAL_POLICY_SOURCE_BELOW_70_PERCENT"] += 1
    if total and hard_rate < 0.20:
        errors["HARD_DEVELOPMENT_BELOW_20_PERCENT"] += 1
    overlap = split_states.get("train", set()) & split_states.get("val", set())
    if overlap:
        errors["STATE_CROSSES_TRAIN_VAL"] += len(overlap)
    crossing_families = sorted(family for family, splits in family_splits.items() if len(splits) > 1)
    if crossing_families:
        errors["TASK_FAMILY_CROSSES_TRAIN_VAL"] += len(crossing_families)
    if pending_review:
        errors["SEMANTIC_REVIEW_PENDING"] += pending_review
    result = {
        "schemaVersion": "p2j4-dpo-v4-pair-contract-audit-v2",
        "status": "PASS" if not errors else "FAIL",
        "freezeEligible": not errors,
        "recordCount": total,
        "uniqueStateCount": len(state_counts),
        "maxPairsPerState": max(state_counts.values()) if state_counts else 0,
        "chosenActionCounts": dict(chosen_counts),
        "sourceCounts": dict(source_counts),
        "scenarioClassCounts": dict(scenario_counts),
        "modelOriginValidCount": model_origin_valid,
        "realPolicySourceValidCount": real_policy_valid,
        "hardProvenanceValidCount": hard_valid,
        "modelOriginRate": model_rate,
        "realPolicySourceRate": real_policy_rate,
        "hardDevelopmentRate": hard_rate,
        "candidateWidthCounts": dict(sorted(width_counts.items())),
        "preferenceDimensionCounts": dict(dimension_counts),
        "splitCounts": dict(Counter(row.get("split") for row in rows)),
        "validationFamilyCount": len({row.get("task_family") for row in rows if row.get("split") == "val"}),
        "crossSplitStateCount": len(overlap),
        "crossSplitFamilyCount": len(crossing_families),
        "pendingSemanticReviewCount": pending_review,
        "errors": dict(sorted(errors.items())),
        "gatePolicy": {
            "pairCount": "500-700",
            "chosenActionCoverage": "10/10",
            "maxPairsPerState": 3,
            "modelOriginRate": ">=0.70",
            "realPolicySourceRate": ">=0.70 (scenario_class=real_policy_source only)",
            "hardDevelopmentRate": ">=0.20",
            "candidateWidth4to7Rate": ">=0.85",
            "taskFamilySplit": "disjoint",
            "semanticReviewStatus": "RULE_AUDITED or FROZEN_APPROVED",
        },
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = _audit(args.input)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "status", "freezeEligible", "recordCount", "uniqueStateCount", "sourceCounts",
        "chosenActionCounts", "splitCounts", "errors",
    )}, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
