"""Fail-closed audit for a future DPO v4 pair staging file.

The v4 contract prevents the exact v3 failure modes: insufficient action
coverage, narrow synthetic candidate sets, reason-only domination, eval-source
leakage, and an unlabelled preference rationale.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ACTIONS = {
    "inspect_cohort", "execute_read_query", "compare_groups", "analyze_projection",
    "stratified_analysis", "adjust_confounders", "cross_project_validate",
    "cross_disease_validate", "retrieve_evidence", "finish",
}
SOURCE_KINDS = {"real_training_trace", "hard_development_trace", "controlled_boundary"}
PREFERENCE_TYPES = {"action", "reason"}


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def audit(path: Path) -> dict[str, Any]:
    records = _records(path)
    errors: list[str] = []
    chosen_actions: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    preference_counts: Counter[str] = Counter()
    width_counts: Counter[int] = Counter()
    signatures: set[tuple[str, str, str]] = set()
    state_counts: Counter[str] = Counter()
    for item in records:
        meta = item.get("metadata", {})
        prompt = item.get("prompt", [])
        try:
            state = json.loads(prompt[1]["content"])["policy_state"]
            chosen = json.loads(item["chosen"])
            rejected = json.loads(item["rejected"])
        except (IndexError, KeyError, TypeError, json.JSONDecodeError):
            errors.append("RECORD_SHAPE_INVALID")
            continue
        source = meta.get("source_kind")
        preference = meta.get("preference_type")
        if source not in SOURCE_KINDS:
            errors.append("SOURCE_KIND_INVALID")
        if preference not in PREFERENCE_TYPES:
            errors.append("PREFERENCE_TYPE_INVALID")
        if meta.get("evaluation_source"):
            errors.append("EVALUATION_SOURCE_FORBIDDEN")
        candidates = state.get("candidate_actions", [])
        width_counts[len(candidates)] += 1
        if chosen.get("selected_action") not in candidates or rejected.get("selected_action") not in candidates:
            errors.append("PAIR_ACTION_NOT_CANDIDATE")
        if chosen.get("selected_action") not in ACTIONS or rejected.get("selected_action") not in ACTIONS:
            errors.append("ILLEGAL_ACTION_LABEL")
        if preference == "action" and chosen.get("selected_action") == rejected.get("selected_action"):
            errors.append("ACTION_PAIR_UNCHANGED")
        if preference == "reason" and chosen.get("selected_action") != rejected.get("selected_action"):
            errors.append("REASON_PAIR_ACTION_CHANGED")
        if not meta.get("preference_rationale"):
            errors.append("PREFERENCE_RATIONALE_MISSING")
        signature = meta.get("state_signature")
        chosen_action = chosen.get("selected_action")
        rejected_action = rejected.get("selected_action")
        pair_signature = (signature or "", chosen_action or "", rejected_action or "")
        if not signature or pair_signature in signatures:
            errors.append("PAIR_SIGNATURE_DUPLICATE_OR_MISSING")
        signatures.add(pair_signature)
        if signature:
            state_counts[signature] += 1
            if state_counts[signature] > 4:
                errors.append("STATE_PAIR_MULTIPLICITY_ABOVE_4")
        chosen_actions[chosen_action or "UNKNOWN"] += 1
        source_counts[source or "UNKNOWN"] += 1
        preference_counts[preference or "UNKNOWN"] += 1
    total = len(records)
    wide = sum(count for width, count in width_counts.items() if 4 <= width <= 7)
    if total < 500 or total > 700:
        errors.append("PAIR_COUNT_OUTSIDE_500_700")
    if set(chosen_actions) != ACTIONS:
        errors.append("CHOSEN_ACTION_COVERAGE_INCOMPLETE")
    positive_counts = [chosen_actions[action] for action in ACTIONS]
    if not positive_counts or min(positive_counts) == 0 or max(positive_counts) > 3 * min(positive_counts):
        errors.append("CHOSEN_ACTION_IMBALANCE_OVER_3X")
    if total and wide / total < 0.85:
        errors.append("RUNTIME_LIKE_WIDTH_BELOW_85_PERCENT")
    if total and preference_counts["action"] / total < 0.80:
        errors.append("ACTION_PREFERENCE_BELOW_80_PERCENT")
    if total and source_counts["real_training_trace"] / total < 0.70:
        errors.append("REAL_TRACE_SOURCE_BELOW_70_PERCENT")
    if total and source_counts["hard_development_trace"] / total < 0.20:
        errors.append("HARD_DEVELOPMENT_SOURCE_BELOW_20_PERCENT")
    if total and source_counts["controlled_boundary"] / total > 0.10:
        errors.append("CONTROLLED_SOURCE_ABOVE_10_PERCENT")
    return {
        "schemaVersion": "p2j4-dpo-v4-pair-contract-audit-v1",
        "status": "PASS" if not errors else "FAIL",
        "recordCount": total,
        "chosenActionCounts": dict(chosen_actions),
        "sourceCounts": dict(source_counts),
        "preferenceCounts": dict(preference_counts),
        "candidateWidthCounts": dict(sorted(width_counts.items())),
        "uniqueStateCount": len(state_counts),
        "maxPairsPerState": max(state_counts.values()) if state_counts else 0,
        "runtimeLikeWidthRate": wide / total if total else 0.0,
        "errors": sorted(set(errors)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.input)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "recordCount", "errors")}, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
