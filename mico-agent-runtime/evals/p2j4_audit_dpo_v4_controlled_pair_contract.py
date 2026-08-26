"""Fail-closed contract audit for the Controlled DPO-v4 dataset."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from mico_agent_runtime.ports.decision_policy import DecisionPolicyOutput


ACTIONS = {
    "inspect_cohort", "execute_read_query", "compare_groups", "analyze_projection",
    "stratified_analysis", "adjust_confounders", "cross_project_validate",
    "cross_disease_validate", "retrieve_evidence", "finish",
}
SOURCES = {"model_selected_controlled", "runtime_repair_derived", "legacy_boundary"}
SCENARIOS = {
    "controlled_state_matrix", "controlled_weak_action_source", "model_origin_collection",
    "runtime_repair_derived", "legacy_metadata_boundary", "legacy_hard_case",
    "legacy_controlled_supplement",
}
STATE_FIELDS = ("task_kind", "goal_code", "observation_flags", "history_actions", "candidate_actions", "state_summary")


def _sft_system_template(path: Path) -> str:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    templates = {
        row["messages"][0]["content"]
        for row in rows
        if len(row.get("messages", [])) == 3 and row["messages"][0].get("role") == "system"
    }
    if len(templates) != 1:
        raise SystemExit("DPO_V4_SFT_SYSTEM_TEMPLATE_NOT_UNIQUE")
    return next(iter(templates))


def audit(path: Path, sft_train: Path | None = None) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    sft_system = _sft_system_template(sft_train) if sft_train is not None else None
    errors: Counter[str] = Counter()
    chosen: Counter[str] = Counter()
    dimensions: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    scenarios: Counter[str] = Counter()
    states: Counter[str] = Counter()
    pair_keys: set[tuple[str, str, str]] = set()
    split_states: dict[str, set[str]] = defaultdict(set)
    family_splits: dict[str, set[str]] = defaultdict(set)
    reason_only = 0
    for row in rows:
        prompt = row.get("prompt")
        if not isinstance(prompt, list) or len(prompt) != 2 or [message.get("role") for message in prompt] != ["system", "user"]:
            errors["PROMPT_SHAPE_INVALID"] += 1
            continue
        if sft_system is not None and prompt[0].get("content") != sft_system:
            errors["SFT_PROMPT_CONTRACT_MISMATCH"] += 1
            continue
        try:
            state = json.loads(prompt[1]["content"])["policy_state"]
            chosen_completion = DecisionPolicyOutput.model_validate(json.loads(row["chosen"]))
            rejected_completion = DecisionPolicyOutput.model_validate(json.loads(row["rejected"]))
        except Exception:
            errors["RECORD_OR_COMPLETION_SCHEMA_INVALID"] += 1
            continue
        for field in STATE_FIELDS:
            if state.get(field) != row.get(field):
                errors["PROMPT_STATE_MISMATCH_" + field] += 1
        candidates = row.get("candidate_actions")
        if not isinstance(candidates, list) or not 4 <= len(candidates) <= 7 or len(candidates) != len(set(candidates)):
            errors["CANDIDATE_SET_INVALID"] += 1
        elif any(action not in ACTIONS for action in candidates):
            errors["CANDIDATE_ACTION_INVALID"] += 1
        chosen_action = row.get("chosen_action")
        rejected_action = row.get("rejected_action")
        if chosen_completion.selected_action != chosen_action or rejected_completion.selected_action != rejected_action:
            errors["COMPLETION_ACTION_MISMATCH"] += 1
        if chosen_action == rejected_action or chosen_action not in candidates or rejected_action not in candidates:
            errors["PAIR_ACTION_INVALID"] += 1
        key = (row.get("state_signature"), chosen_action, rejected_action)
        if key in pair_keys:
            errors["DUPLICATE_PAIR"] += 1
        pair_keys.add(key)
        states[row["state_signature"]] += 1
        if states[row["state_signature"]] > 2:
            errors["STATE_MULTIPLICITY_ABOVE_2"] += 1
        source = row.get("source_kind")
        scenario = row.get("scenario_class")
        sources[source] += 1
        scenarios[scenario] += 1
        if source not in SOURCES or scenario not in SCENARIOS:
            errors["SOURCE_CLASS_INVALID"] += 1
        if source == "model_selected_controlled":
            if not (row.get("planner_origin") == "model" and row.get("provenance_class") == "model_origin_raw" and row.get("raw_action") == row.get("final_action") == chosen_action and not row.get("repair_codes")):
                errors["MODEL_CONTROLLED_PROVENANCE_INVALID"] += 1
        elif source == "runtime_repair_derived":
            if not (row.get("planner_origin") == "model" and row.get("raw_action") == rejected_action and row.get("final_action") == chosen_action and row.get("repair_codes")):
                errors["RUNTIME_REPAIR_PROVENANCE_INVALID"] += 1
        elif source == "legacy_boundary":
            if not (row.get("planner_origin") is None and row.get("provenance_class") == "provenance_unknown_legacy" and row.get("raw_action") is None):
                errors["LEGACY_PROVENANCE_INVALID"] += 1
        if row.get("review_status") != "RULE_AUDITED" or not row.get("review_basis"):
            errors["RULE_REVIEW_MISSING"] += 1
        if row.get("preference_type") != "action":
            reason_only += 1
        if not row.get("preference_codes") or not row.get("preference_rationale"):
            errors["RATIONALE_INCOMPLETE"] += 1
        chosen[chosen_action] += 1
        dimensions[row.get("preference_dimension")] += 1
        split = row.get("split")
        if split not in {"train", "val"}:
            errors["SPLIT_INVALID"] += 1
        split_states[split].add(row["state_signature"])
        family_splits[row["task_family"]].add(split)
    total = len(rows)
    counts = [chosen[action] for action in ACTIONS]
    if not 350 <= total <= 500:
        errors["PAIR_COUNT_OUTSIDE_350_500"] += 1
    if set(chosen) != ACTIONS:
        errors["ACTION_COVERAGE_INCOMPLETE"] += 1
    if not counts or min(counts) == 0 or max(counts) > 2 * min(counts):
        errors["ACTION_IMBALANCE_OVER_2X"] += 1
    if dimensions["evidence_requirement"] > total * 0.30:
        errors["EVIDENCE_REQUIREMENT_ABOVE_30_PERCENT"] += 1
    overlap = split_states["train"] & split_states["val"]
    crossing_families = [family for family, splits in family_splits.items() if len(splits) > 1]
    if overlap:
        errors["STATE_CROSSES_SPLIT"] += len(overlap)
    if crossing_families:
        errors["FAMILY_CROSSES_SPLIT"] += len(crossing_families)
    if sources["runtime_repair_derived"] != 17:
        errors["REPAIR_COUNT_NOT_17"] += 1
    result = {
        "schemaVersion": "p2j4-dpo-v4-controlled-pair-contract-audit-v1",
        "status": "PASS" if not errors else "FAIL",
        "freezeEligible": not errors,
        "trainingStarted": False,
        "recordCount": total,
        "uniqueStateCount": len(states),
        "maxPairsPerState": max(states.values(), default=0),
        "chosenActionCounts": dict(chosen),
        "chosenMaxMinRatio": max(counts) / min(counts) if counts and min(counts) else None,
        "sourceCounts": dict(sources),
        "scenarioClassCounts": dict(scenarios),
        "preferenceDimensionCounts": dict(dimensions),
        "evidenceRequirementRate": dimensions["evidence_requirement"] / total if total else 0.0,
        "reasonOnlyCount": reason_only,
        "splitCounts": dict(Counter(row.get("split") for row in rows)),
        "crossSplitStateCount": len(overlap),
        "crossSplitFamilyCount": len(crossing_families),
        "errors": dict(errors),
        "datasetClaim": "Controlled Agent Policy Preference Dataset",
        "sftPromptContractChecked": sft_system is not None,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sft-train", type=Path)
    args = parser.parse_args()
    result = audit(args.input, args.sft_train)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
