"""Merge DPO-v4 pair pools without hiding provenance or cloning states.

This is a staging merge only.  It preserves model-origin, repair-derived, and
legacy-hard provenance, removes exact pair duplicates, caps each normalized
state at three pairs, and assigns whole task-family groups to train/val.
It does not claim semantic approval or create a frozen training set.
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
SOURCE_ORDER = (
    "model_origin_policy_trace",
    "hard_development_trace",
)


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _validate(row: dict[str, Any]) -> str | None:
    required = {
        "id", "prompt", "chosen", "rejected", "task_kind", "goal_code",
        "task_family", "observation_flags", "history_actions", "candidate_actions",
        "state_summary", "state_signature", "source_kind", "scenario_class",
        "provenance_class", "chosen_action", "rejected_action", "preference_type",
        "preference_dimension", "preference_codes", "preference_rationale",
        "review_status",
    }
    missing = sorted(required - set(row))
    if missing:
        return "MISSING_FIELDS:" + ",".join(missing)
    if row["source_kind"] not in SOURCE_ORDER:
        return "SOURCE_KIND_INVALID"
    if row["preference_type"] != "action":
        return "PREFERENCE_TYPE_INVALID"
    candidates = row["candidate_actions"]
    if not isinstance(candidates, list) or not 4 <= len(candidates) <= 7:
        return "CANDIDATE_WIDTH_INVALID"
    if len(candidates) != len(set(candidates)) or any(action not in ACTIONS for action in candidates):
        return "CANDIDATE_ACTION_INVALID"
    if row["chosen_action"] not in candidates or row["rejected_action"] not in candidates:
        return "PAIR_ACTION_NOT_CANDIDATE"
    if row["chosen_action"] == row["rejected_action"]:
        return "PAIR_ACTION_UNCHANGED"
    try:
        chosen = json.loads(row["chosen"])
        rejected = json.loads(row["rejected"])
    except (TypeError, json.JSONDecodeError):
        return "COMPLETION_JSON_INVALID"
    if chosen.get("selected_action") != row["chosen_action"] or rejected.get("selected_action") != row["rejected_action"]:
        return "COMPLETION_ACTION_MISMATCH"
    prompt = row["prompt"]
    try:
        state = json.loads(prompt[1]["content"])["policy_state"]
    except (IndexError, KeyError, TypeError, json.JSONDecodeError):
        return "PROMPT_STATE_INVALID"
    for field in ("task_kind", "goal_code", "observation_flags", "history_actions", "candidate_actions", "state_summary"):
        if state.get(field) != row[field]:
            return "PROMPT_STATE_MISMATCH:" + field
    if not row["preference_dimension"] or not row["preference_codes"] or not row["preference_rationale"]:
        return "PREFERENCE_RATIONALE_INCOMPLETE"
    return None


def _validation_families(rows: list[dict[str, Any]], ratio: float) -> tuple[set[str], dict[str, int]]:
    counts = Counter(row["task_family"] for row in rows)
    families = sorted(counts)
    target = round(len(rows) * ratio)
    minimum = max(1, round(len(rows) * 0.10))
    maximum = round(len(rows) * 0.35)
    # Dynamic programming over pair counts avoids exponential degradation
    # once scenario-derived task families exceed twenty.
    subsets: dict[int, tuple[str, ...]] = {0: ()}
    for family in families:
        additions: dict[int, tuple[str, ...]] = {}
        for total, subset in list(subsets.items()):
            new_total = total + counts[family]
            if new_total > maximum:
                continue
            candidate = (*subset, family)
            existing = subsets.get(new_total) or additions.get(new_total)
            if existing is None or candidate < existing:
                additions[new_total] = candidate
        subsets.update(additions)
    options = [
        (total, subset) for total, subset in subsets.items()
        if minimum <= total <= maximum and subset and len(subset) < len(families)
    ]
    if options:
        _, best_subset = min(
            options,
            key=lambda item: (
                abs(item[0] - target),
                0 if len(item[1]) >= 2 else 1,
                item[0],
                item[1],
            ),
        )
        selected = set(best_subset)
    else:
        selected = {families[0]} if families else set()
    return selected, dict(counts)


def merge(paths: list[Path], validation_ratio: float, max_per_state: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pools: list[tuple[str, dict[str, Any]]] = []
    excluded: Counter[str] = Counter()
    input_counts: Counter[str] = Counter()
    for path in paths:
        for row in _read(path):
            input_counts[row.get("source_kind", "UNKNOWN")] += 1
            reason = _validate(row)
            if reason:
                excluded[reason] += 1
                continue
            pools.append((path.name, row))

    pools.sort(key=lambda item: (
        SOURCE_ORDER.index(item[1]["source_kind"]),
        item[1]["state_signature"],
        item[1]["chosen_action"],
        item[1]["rejected_action"],
        item[1]["id"],
    ))
    output: list[dict[str, Any]] = []
    pair_keys: set[tuple[str, str, str]] = set()
    state_counts: Counter[str] = Counter()
    for source_file, row in pools:
        key = (row["state_signature"], row["chosen_action"], row["rejected_action"])
        if key in pair_keys:
            excluded["EXACT_PAIR_DUPLICATE"] += 1
            continue
        if state_counts[row["state_signature"]] >= max_per_state:
            excluded["STATE_PAIR_CAP_REACHED"] += 1
            continue
        item = dict(row)
        item["merged_source_file"] = source_file
        item["state_group_id"] = row["state_signature"]
        output.append(item)
        pair_keys.add(key)
        state_counts[row["state_signature"]] += 1

    val_families, family_counts = _validation_families(output, validation_ratio)
    for row in output:
        row["split"] = "val" if row["task_family"] in val_families else "train"

    split_states: dict[str, set[str]] = defaultdict(set)
    for row in output:
        split_states[row["split"]].add(row["state_signature"])
    overlap = split_states.get("train", set()) & split_states.get("val", set())
    source_counts = Counter(row["source_kind"] for row in output)
    model_count = source_counts["model_origin_policy_trace"]
    hard_count = source_counts["hard_development_trace"]
    audit = {
        "schemaVersion": "p2j4-dpo-v4-staging-merge-audit-v1",
        "status": "STAGING_REVIEW_REQUIRED" if output else "BLOCKED",
        "trainingStarted": False,
        "pairConstructionStarted": True,
        "inputCounts": dict(input_counts),
        "inputValidCount": len(pools),
        "mergedPairCount": len(output),
        "uniqueStateCount": len(state_counts),
        "maxPairsPerState": max(state_counts.values()) if state_counts else 0,
        "sourceCounts": dict(source_counts),
        "modelOriginRate": model_count / len(output) if output else 0.0,
        "hardDevelopmentRate": hard_count / len(output) if output else 0.0,
        "chosenActionCounts": dict(Counter(row["chosen_action"] for row in output)),
        "candidateWidthCounts": dict(Counter(len(row["candidate_actions"]) for row in output)),
        "preferenceDimensionCounts": dict(Counter(row["preference_dimension"] for row in output)),
        "taskFamilyCounts": family_counts,
        "validationFamilies": sorted(val_families),
        "splitCounts": dict(Counter(row["split"] for row in output)),
        "splitStateOverlapCount": len(overlap),
        "excludedCounts": dict(excluded),
        "rules": [
            "model-origin pool has priority when exact pair keys collide",
            "exact (state, chosen, rejected) duplicates are removed",
            "no normalized state receives more than max_per_state pairs",
            "whole task families, not individual pairs, are assigned to validation",
            "legacy hard and runtime-repair rows retain hard-development provenance",
            "staging status is not a frozen-training approval",
        ],
        "nextAction": "run_v2_contract_semantic_and_leakage_audits",
    }
    return output, audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--repair", type=Path, required=True)
    parser.add_argument("--hard", type=Path, required=True)
    parser.add_argument("--metadata-boundary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--validation-ratio", type=float, default=0.20)
    parser.add_argument("--max-per-state", type=int, default=3)
    args = parser.parse_args()
    paths = [args.model, args.repair, args.hard]
    if args.metadata_boundary is not None:
        paths.append(args.metadata_boundary)
    rows, audit = merge(paths, args.validation_ratio, args.max_per_state)
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    args.audit_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: audit[key] for key in (
        "status", "mergedPairCount", "uniqueStateCount", "maxPairsPerState",
        "sourceCounts", "chosenActionCounts", "splitCounts", "validationFamilies", "excludedCounts",
    )}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
