"""Build a bounded, locally reviewed DPO-v4 model-origin staging set.

The input rows are actual model decisions with structured Runtime state.  A
row is promoted only when the selected action is supported by an explicit
state obligation; all other candidate rows remain review-only.  This script
does not call a model, invent a missing field, or claim human approval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ACTIONS = {
    "inspect_cohort",
    "execute_read_query",
    "compare_groups",
    "analyze_projection",
    "stratified_analysis",
    "adjust_confounders",
    "cross_project_validate",
    "cross_disease_validate",
    "retrieve_evidence",
    "finish",
}


def _hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _prompt(row: dict[str, Any]) -> list[dict[str, str]]:
    state = {
        "task_kind": row["task_kind"],
        "goal_code": row["goal_code"],
        "observation_flags": row["observation_flags"],
        "history_actions": row["history_actions"],
        "candidate_actions": row["candidate_actions"],
        "state_summary": row["state_summary"],
    }
    return [
        {
            "role": "system",
            "content": "Choose exactly one next action from the supplied de-identified policy state.",
        },
        {
            "role": "user",
            "content": json.dumps({"policy_state": state}, ensure_ascii=False, sort_keys=True),
        },
    ]


def _required(row: dict[str, Any]) -> set[str]:
    return set(row.get("task_required_actions") or [])


def _review_rule(row: dict[str, Any]) -> tuple[str, list[str], str] | None:
    """Return dimension, codes, rationale basis for a state-supported action."""

    chosen = row.get("chosen_action")
    history = set(row.get("history_actions") or [])
    flags = set(row.get("observation_flags") or [])
    required = _required(row)
    if chosen not in ACTIONS:
        return None
    if chosen == "execute_read_query" and "execute_read_query" not in history:
        return (
            "efficiency",
            ["BOUNDED_READ_REQUIRED", "OBSERVATION_FIRST"],
            "The state has a validated observation but no completed bounded read, so the read step is the next efficient evidence operation.",
        )
    if chosen == "compare_groups" and "compare_groups" not in history and (
        chosen in required or row.get("goal_code") == "group_comparison"
    ):
        return (
            "evidence_requirement",
            ["GROUP_COMPARISON_REQUIRED", "VALIDATED_OBSERVATION_AVAILABLE"],
            "The approved state calls for a bounded group comparison before downstream analysis; the rejected action remains legal but is not the current comparison obligation.",
        )
    if chosen == "stratified_analysis" and "stratified_analysis" not in history and (
        chosen in required or "ANALYSIS_REQUIRED" in flags or "compare_groups" in history
    ):
        return (
            "exploration_depth",
            ["STRATIFICATION_AFTER_OBSERVATION", "ANALYSIS_STEP_REQUIRED"],
            "The validated comparison state supports a bounded stratified follow-up, which resolves the next analysis obligation before reporting.",
        )
    if chosen == "analyze_projection" and "analyze_projection" not in history and (
        chosen in required or "ANALYSIS_REQUIRED" in flags
    ):
        return (
            "exploration_depth",
            ["PROJECTION_ANALYSIS_REQUIRED", "VALIDATED_OBSERVATION_AVAILABLE"],
            "The state has a validated Java observation and the projection analysis is the next bounded analysis obligation.",
        )
    if chosen == "adjust_confounders" and "adjust_confounders" not in history and (
        chosen in required or "ANALYSIS_COMPLETE" in flags
    ):
        return (
            "evidence_requirement",
            ["CONFOUNDER_ADJUSTMENT_REQUIRED", "ANALYSIS_ALREADY_AVAILABLE"],
            "The state already contains an analyzable result and calls for confounder adjustment before treating the pattern as stable.",
        )
    if chosen == "cross_project_validate" and "cross_project_validate" not in history and (
        chosen in required or "CROSS_PROJECT_REQUIRED" in flags
    ):
        return (
            "evidence_requirement",
            ["CROSS_PROJECT_VALIDATION_REQUIRED", "MULTI_OBSERVATION_STATE"],
            "The state has enough validated observations for the required cross-project check; stopping or switching analysis would leave that evidence obligation unresolved.",
        )
    if chosen == "cross_disease_validate" and "cross_disease_validate" not in history and (
        chosen in required or "DISEASE_VALIDATION_REQUIRED" in flags
    ):
        return (
            "evidence_requirement",
            ["CROSS_DISEASE_VALIDATION_REQUIRED", "MULTI_OBSERVATION_STATE"],
            "The state has enough validated observations for the required cross-disease check before an observational conclusion.",
        )
    if chosen == "retrieve_evidence" and "retrieve_evidence" not in history and (
        chosen in required or {"PROJECT_VALIDATED", "CONFOUNDER_ADJUSTED", "EVIDENCE_REQUIRED"} & flags
    ):
        return (
            "evidence_requirement",
            ["BOUNDED_EVIDENCE_RETRIEVAL", "DATA_OBSERVATION_ALREADY_AVAILABLE"],
            "The data-side observation is already available and bounded evidence retrieval is the next support step before reporting.",
        )
    if chosen == "finish" and required.issubset(history | {chosen}) and (
        "ANALYSIS_COMPLETE" in flags
        or "EVIDENCE_GROUNDED" in flags
        or "EVIDENCE_RETRIEVED" in flags
        or "PROJECT_VALIDATED" in flags
        or "CONFOUNDER_ADJUSTED" in flags
    ):
        return (
            "stop_boundary",
            ["EVIDENCE_OBLIGATION_SATISFIED", "NO_UNRESOLVED_REQUIRED_ACTION"],
            "All task-required actions are complete and the validated evidence state supports stopping rather than adding an unrequested operation.",
        )
    return None


def _valid_row(row: dict[str, Any]) -> bool:
    required = (
        "task_kind",
        "goal_code",
        "observation_flags",
        "history_actions",
        "candidate_actions",
        "state_summary",
        "state_signature",
        "chosen_action",
        "rejected_action",
    )
    if any(row.get(field) is None for field in required):
        return False
    candidates = row["candidate_actions"]
    return (
        isinstance(candidates, list)
        and 4 <= len(candidates) <= 7
        and len(candidates) == len(set(candidates))
        and row["chosen_action"] in candidates
        and row["rejected_action"] in candidates
        and row["chosen_action"] != row["rejected_action"]
        and row.get("planner_origin") == "model"
        and row.get("raw_action") == row.get("final_action") == row.get("chosen_action")
        and not row.get("decision_repair_codes")
    )


def _pair(row: dict[str, Any], review: tuple[str, list[str], str]) -> dict[str, Any]:
    dimension, codes, rationale = review
    pair_id = "dpo-v4-model-" + _hash([
        row["state_signature"], row["chosen_action"], row["rejected_action"]
    ])
    chosen = {
        "selected_action": row["chosen_action"],
        "decision_reason": rationale,
        "alternative_actions": [row["rejected_action"]],
        "stop_reason": "EVIDENCE_SUFFICIENT" if row["chosen_action"] == "finish" else None,
    }
    rejected = {
        "selected_action": row["rejected_action"],
        "decision_reason": "Legal alternative retained for the preference comparison.",
        "alternative_actions": [row["chosen_action"]],
        "stop_reason": None,
    }
    task_family = ":".join(
        str(value)
        for value in (
            row.get("scenario_class") or "unknown_source",
            row["task_kind"],
            row["goal_code"],
        )
    )
    return {
        "id": pair_id,
        "prompt": _prompt(row),
        "chosen": json.dumps(chosen, ensure_ascii=False, sort_keys=True),
        "rejected": json.dumps(rejected, ensure_ascii=False, sort_keys=True),
        "task_kind": row["task_kind"],
        "goal_code": row["goal_code"],
        "task_family": task_family,
        "observation_flags": row["observation_flags"],
        "history_actions": row["history_actions"],
        "candidate_actions": row["candidate_actions"],
        "state_summary": row["state_summary"],
        "state_signature": row["state_signature"],
        "source_trace_id": row.get("source_trace_id"),
        "source_file": row.get("source_file"),
        "case_id": row.get("case_id"),
        "planner_provider": row.get("planner_provider"),
        "planner_model": row.get("planner_model"),
        "observation_count": row.get("observation_count"),
        "task_required_actions": row.get("task_required_actions", []),
        "source_kind": "model_origin_policy_trace",
        "scenario_class": row.get("scenario_class"),
        "planner_origin": "model",
        "provenance_class": "model_origin_raw",
        "raw_action": row["raw_action"],
        "final_action": row["final_action"],
        "chosen_action": row["chosen_action"],
        "rejected_action": row["rejected_action"],
        "preference_type": "action",
        "preference_dimension": dimension,
        "preference_codes": codes,
        "preference_rationale": rationale,
        "preference_rationale_source": "local_state_obligation_review_v1",
        "review_status": "LOCAL_SEMANTIC_REVIEWED_PENDING_FINAL_FREEZE",
    }


def build(rows: list[dict[str, Any]], target: int, max_per_state: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eligible: list[tuple[dict[str, Any], tuple[str, list[str], str]]] = []
    excluded: Counter[str] = Counter()
    for row in rows:
        if not _valid_row(row):
            excluded["ROW_CONTRACT_OR_PROVENANCE_INVALID"] += 1
            continue
        review = _review_rule(row)
        if review is None:
            excluded["NO_EXPLICIT_STATE_OBLIGATION"] += 1
            continue
        eligible.append((row, review))

    # Prefer state diversity.  A deterministic round-robin by chosen action
    # lets the caller audit the balancing decision and avoids cloning one state
    # into a large block of pairs.
    by_action: dict[str, list[tuple[dict[str, Any], tuple[str, list[str], str]]]] = defaultdict(list)
    for item in eligible:
        by_action[item[0]["chosen_action"]].append(item)
    for action in by_action:
        by_action[action].sort(key=lambda item: (item[0]["state_signature"], item[0]["rejected_action"]))

    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, str, str]] = set()
    state_counts: Counter[str] = Counter()
    target_per_action = max(1, target // max(1, len(ACTIONS) - 1))
    # inspect_cohort is Runtime-owned in this source pool and therefore has no
    # raw model-origin rows; it is supplied later by the hard/boundary pool.
    quotas = {action: target_per_action for action in sorted(ACTIONS) if action != "inspect_cohort"}
    quotas["finish"] += target - sum(quotas.values())

    def take(action: str, limit: int) -> None:
        count = 0
        for row, review in by_action.get(action, []):
            key = (row["state_signature"], row["chosen_action"], row["rejected_action"])
            if key in selected_keys or state_counts[row["state_signature"]] >= max_per_state:
                continue
            selected_keys.add(key)
            state_counts[row["state_signature"]] += 1
            selected.append(_pair(row, review))
            count += 1
            if count >= limit or len(selected) >= target:
                break

    for action, quota in quotas.items():
        take(action, quota)
    if len(selected) < target:
        remaining = [item for action in sorted(by_action) for item in by_action[action]]
        for row, review in remaining:
            key = (row["state_signature"], row["chosen_action"], row["rejected_action"])
            if key in selected_keys or state_counts[row["state_signature"]] >= max_per_state:
                continue
            selected_keys.add(key)
            state_counts[row["state_signature"]] += 1
            selected.append(_pair(row, review))
            if len(selected) >= target:
                break

    audit = {
        "schemaVersion": "p2j4-dpo-v4-semantic-model-pairs-audit-v1",
        "status": "LOCAL_SEMANTIC_REVIEWED_PENDING_FINAL_FREEZE" if selected else "BLOCKED",
        "trainingStarted": False,
        "pairConstructionStarted": True,
        "inputRowCount": len(rows),
        "eligibleRowCount": len(eligible),
        "selectedPairCount": len(selected),
        "uniqueStateCount": len(state_counts),
        "maxPairsPerState": max(state_counts.values()) if state_counts else 0,
        "chosenActionCounts": dict(Counter(row["chosen_action"] for row in selected)),
        "preferenceDimensionCounts": dict(Counter(row["preference_dimension"] for row in selected)),
        "candidateWidthCounts": dict(Counter(len(row["candidate_actions"]) for row in selected)),
        "scenarioClassCounts": dict(Counter(row.get("scenario_class") for row in selected)),
        "excludedCounts": dict(excluded),
        "rules": [
            "all structured state fields are copied from the trace",
            "chosen and rejected are distinct legal candidate actions",
            "only raw model decisions with raw_action==final_action enter this pool",
            "decision-level Runtime repairs are excluded from the pure model pool",
            "at most max_per_state pairs are selected for one normalized state",
            "rationale is structured and locally auditable, not a free-form template",
            "inspect_cohort remains a Runtime-owned boundary action and is not fabricated as model-origin",
        ],
        "nextAction": "merge_hard_boundary_pairs_then_run_leakage_and_split_audits",
    }
    return selected, audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--target", type=int, default=400)
    parser.add_argument("--max-per-state", type=int, default=3)
    args = parser.parse_args()
    rows, audit = build(_read(args.input), args.target, args.max_per_state)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    args.audit_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: audit[key] for key in (
        "status", "selectedPairCount", "uniqueStateCount", "maxPairsPerState", "chosenActionCounts", "excludedCounts"
    )}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
