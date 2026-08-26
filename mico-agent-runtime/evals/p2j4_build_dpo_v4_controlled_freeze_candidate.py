"""Build a no-API controlled DPO-v4 freeze candidate from existing assets.

This performs subtraction and deterministic rebalancing only.  It does not
run a model, replay a Trace, invent state fields, or claim open-policy source
provenance.  Existing source artifacts are read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import p2j4_build_dpo_v4_semantic_pairs as semantic
from p2j4_merge_dpo_v4_staging import _validation_families


ACTION_CAPS = {
    "inspect_cohort": 23,
    "execute_read_query": 46,
    "compare_groups": 24,
    "analyze_projection": 36,
    "stratified_analysis": 37,
    "adjust_confounders": 46,
    "cross_project_validate": 46,
    "cross_disease_validate": 46,
    "retrieve_evidence": 46,
    "finish": 46,
}

DIMENSION_BY_ACTION = {
    "inspect_cohort": "efficiency",
    "execute_read_query": "efficiency",
    "compare_groups": "evidence_requirement",
    "analyze_projection": "exploration_depth",
    "stratified_analysis": "exploration_depth",
    "adjust_confounders": "reason_quality",
    "cross_project_validate": "exploration_depth",
    "cross_disease_validate": "exploration_depth",
    "retrieve_evidence": "evidence_requirement",
    "finish": "stop_boundary",
}


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _pair_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return row["state_signature"], row["chosen_action"], row["rejected_action"]


def _source_class(row: dict[str, Any]) -> str:
    scenario = row.get("scenario_class")
    if scenario == "runtime_repair_derived":
        return "runtime_repair_derived"
    if scenario in {"legacy_metadata_boundary", "legacy_hard_case", "legacy_controlled_supplement"}:
        return "legacy_boundary"
    return "model_selected_controlled"


def _normalize(row: dict[str, Any]) -> dict[str, Any]:
    item = dict(row)
    item["source_kind_original"] = item.get("source_kind")
    item["source_kind"] = _source_class(item)
    item["dataset_objective"] = "agent_policy_preference_controlled_v4"
    item["open_policy_source_claimed"] = False
    item["preference_type"] = "action"
    item["preference_dimension"] = DIMENSION_BY_ACTION[item["chosen_action"]]
    item["dimension_assignment_basis"] = "chosen_action_policy_objective_v1"
    item["reviewer_label"] = "p2j4_controlled_offline_rule_audit_v1"
    scenario = item.get("scenario_class")
    item["review_basis"] = (
        "RUNTIME_REPAIR_CONTRACT" if item["source_kind"] == "runtime_repair_derived"
        else "LEGACY_METADATA_BOUNDARY_CONTRACT" if scenario == "legacy_metadata_boundary"
        else "LEGACY_LABELLED_HARD_CONTRACT" if scenario == "legacy_hard_case"
        else "LEGACY_STRUCTURED_STATE_OBLIGATION_CONTRACT" if scenario == "legacy_controlled_supplement"
        else "MODEL_RAW_DECISION_CONTROLLED_TASK_CONTRACT"
    )
    # Both preference completions must obey the same closed Runtime output
    # shape.  A rejected finish remains a schema-valid (but less preferred)
    # action and therefore still carries the required stop_reason.
    for field in ("chosen", "rejected"):
        completion = json.loads(item[field])
        completion["stop_reason"] = (
            "EVIDENCE_SUFFICIENT" if completion.get("selected_action") == "finish" else None
        )
        item[field] = json.dumps(completion, ensure_ascii=False, sort_keys=True)
    item["review_status"] = "RULE_AUDITED"
    item["reviewed_at"] = "2026-08-26"
    return item


class _Dinic:
    def __init__(self, size: int) -> None:
        self.graph: list[list[list[int]]] = [[] for _ in range(size)]

    def add(self, left: int, right: int, capacity: int) -> list[int]:
        forward = [right, capacity, len(self.graph[right])]
        backward = [left, 0, len(self.graph[left])]
        self.graph[left].append(forward)
        self.graph[right].append(backward)
        return forward

    def flow(self, source: int, sink: int) -> int:
        total = 0
        while True:
            level = [-1] * len(self.graph)
            level[source] = 0
            queue = [source]
            for node in queue:
                for edge in self.graph[node]:
                    if edge[1] and level[edge[0]] < 0:
                        level[edge[0]] = level[node] + 1
                        queue.append(edge[0])
            if level[sink] < 0:
                return total
            cursor = [0] * len(self.graph)

            def send(node: int, amount: int) -> int:
                if node == sink:
                    return amount
                while cursor[node] < len(self.graph[node]):
                    edge = self.graph[node][cursor[node]]
                    if edge[1] and level[edge[0]] == level[node] + 1:
                        pushed = send(edge[0], min(amount, edge[1]))
                        if pushed:
                            edge[1] -= pushed
                            self.graph[edge[0]][edge[2]][1] += pushed
                            return pushed
                    cursor[node] += 1
                return 0

            while True:
                pushed = send(source, 10**9)
                if not pushed:
                    break
                total += pushed


def _component_validation(rows: list[dict[str, Any]], ratio: float = 0.20) -> set[int]:
    """Choose validation connected components of the state/family graph."""
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for row in rows:
        union("s:" + row["state_signature"], "f:" + row["task_family"])
    components: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        components[find("s:" + row["state_signature"])].append(index)
    target = round(len(rows) * ratio)
    maximum = round(len(rows) * 0.35)
    subsets: dict[int, tuple[str, ...]] = {0: ()}
    for component, indexes in sorted(components.items()):
        additions: dict[int, tuple[str, ...]] = {}
        for total, chosen in list(subsets.items()):
            new_total = total + len(indexes)
            if new_total <= maximum:
                additions.setdefault(new_total, (*chosen, component))
        subsets.update(additions)
    options = [(total, chosen) for total, chosen in subsets.items() if total > 0]
    _, selected = min(options, key=lambda item: (abs(item[0] - target), item[0], item[1]))
    return {index for component in selected for index in components[component]}


def build(candidate_path: Path, reviewed_path: Path, legacy_hard_path: Path | None, legacy_supplement_path: Path | None, max_per_state: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reviewed = _read(reviewed_path)
    mandatory = [
        row for row in reviewed
        if row.get("scenario_class") in {"runtime_repair_derived", "legacy_metadata_boundary"}
    ]
    if legacy_hard_path is not None:
        mandatory.extend(_read(legacy_hard_path))
    if legacy_supplement_path is not None:
        mandatory.extend(_read(legacy_supplement_path))
    # Build every already-eligible model pair locally instead of inheriting the
    # previous 400-row quota.  This exposes enough state diversity for a 2-pair
    # cap without any new inference call.
    model_rows: list[dict[str, Any]] = []
    for row in semantic._read(candidate_path):
        if not semantic._valid_row(row):
            continue
        review = semantic._review_rule(row)
        if review is not None:
            model_rows.append(semantic._pair(row, review))

    model_rows.sort(key=lambda row: (
        row["chosen_action"], row["state_signature"], row["rejected_action"], row["id"]
    ))
    by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in model_rows:
        by_action[row["chosen_action"]].append(row)

    selected: list[dict[str, Any]] = []
    pair_keys: set[tuple[str, str, str]] = set()
    state_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    excluded: Counter[str] = Counter()

    def take(row: dict[str, Any], mandatory_row: bool = False) -> bool:
        key = _pair_key(row)
        if key in pair_keys:
            excluded["EXACT_PAIR_DUPLICATE"] += 1
            return False
        if state_counts[row["state_signature"]] >= max_per_state:
            excluded["STATE_PAIR_CAP_REACHED_MANDATORY" if mandatory_row else "STATE_PAIR_CAP_REACHED"] += 1
            return False
        if not mandatory_row and action_counts[row["chosen_action"]] >= ACTION_CAPS[row["chosen_action"]]:
            excluded["ACTION_CAP_REACHED"] += 1
            return False
        selected.append(row)
        pair_keys.add(key)
        state_counts[row["state_signature"]] += 1
        action_counts[row["chosen_action"]] += 1
        return True

    # The 17 model-error repairs and 23 metadata boundaries are the highest
    # value minority sources and are preserved before controlled model rows.
    for row in sorted(mandatory, key=lambda item: (_source_class(item), item["state_signature"], item["id"])):
        take(row, mandatory_row=True)

    # Reserve scarce action coverage before the global matching.  These pools
    # have far fewer eligible rows than execute/finish and otherwise lose
    # shared state slots despite being essential to ten-action coverage.
    for action in ("analyze_projection", "compare_groups", "stratified_analysis"):
        for row in by_action.get(action, []):
            if action_counts[action] >= ACTION_CAPS[action]:
                break
            take(row)

    # Exact offline b-matching: action quotas on one side and normalized-state
    # capacity two on the other.  This avoids action-order bias without any
    # stochastic sampling or inference call.
    actions = sorted(ACTION_CAPS)
    states = sorted({row["state_signature"] for row in model_rows})
    action_node = {action: index + 1 for index, action in enumerate(actions)}
    state_node = {state: len(action_node) + index + 1 for index, state in enumerate(states)}
    source = 0
    sink = len(action_node) + len(state_node) + 1
    network = _Dinic(sink + 1)
    requested = 0
    for action in actions:
        remaining = max(0, ACTION_CAPS[action] - action_counts[action])
        requested += remaining
        network.add(source, action_node[action], remaining)
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in model_rows:
        if _pair_key(row) not in pair_keys:
            buckets[(row["chosen_action"], row["state_signature"])].append(row)
    edge_refs: dict[tuple[str, str], tuple[list[int], int]] = {}
    for (action, state), rows in sorted(buckets.items()):
        capacity = min(len(rows), max(0, max_per_state - state_counts[state]))
        if capacity:
            edge = network.add(action_node[action], state_node[state], capacity)
            edge_refs[(action, state)] = (edge, capacity)
    for state in states:
        network.add(state_node[state], sink, max(0, max_per_state - state_counts[state]))
    achieved = network.flow(source, sink)
    for key, (edge, original_capacity) in edge_refs.items():
        used = original_capacity - edge[1]
        for row in buckets[key][:used]:
            take(row)
    if achieved < requested:
        excluded["UNFILLED_BALANCED_QUOTA"] += requested - achieved

    normalized = [_normalize(row) for row in selected]
    validation_indexes = _component_validation(normalized, 0.20)
    family_counts = dict(Counter(row["task_family"] for row in normalized))
    for index, row in enumerate(normalized):
        row["split"] = "val" if index in validation_indexes else "train"
        row["state_group_id"] = row["state_signature"]

    split_states: dict[str, set[str]] = defaultdict(set)
    family_splits: dict[str, set[str]] = defaultdict(set)
    for row in normalized:
        split_states[row["split"]].add(row["state_signature"])
        family_splits[row["task_family"]].add(row["split"])
    chosen = Counter(row["chosen_action"] for row in normalized)
    dimensions = Counter(row["preference_dimension"] for row in normalized)
    sources = Counter(row["source_kind"] for row in normalized)
    total = len(normalized)
    positive = list(chosen.values())
    errors: list[str] = []
    if not positive or max(positive) > 2 * min(positive):
        errors.append("CHOSEN_ACTION_IMBALANCE_OVER_2X")
    if max(state_counts.values(), default=0) > max_per_state:
        errors.append("STATE_MULTIPLICITY_ABOVE_LIMIT")
    if dimensions["evidence_requirement"] > total * 0.30:
        errors.append("EVIDENCE_REQUIREMENT_ABOVE_30_PERCENT")
    if split_states["train"] & split_states["val"]:
        errors.append("STATE_CROSSES_SPLIT")
    if any(len(splits) > 1 for splits in family_splits.values()):
        errors.append("FAMILY_CROSSES_SPLIT")
    if sources["runtime_repair_derived"] != 17:
        errors.append("RUNTIME_REPAIR_NOT_FULLY_PRESERVED")
    if not 20 <= sources["legacy_boundary"] <= 120:
        errors.append("LEGACY_BOUNDARY_OUTSIDE_20_120")
    audit = {
        "schemaVersion": "p2j4-dpo-v4-controlled-freeze-candidate-audit-v1",
        "status": "PASS" if not errors else "FAIL",
        "trainingStarted": False,
        "apiCalls": 0,
        "traceRuns": 0,
        "recordCount": total,
        "uniqueStateCount": len(state_counts),
        "maxPairsPerState": max(state_counts.values(), default=0),
        "sourceCounts": dict(sources),
        "chosenActionCounts": dict(chosen),
        "chosenMaxMinRatio": max(positive) / min(positive) if positive else None,
        "preferenceDimensionCounts": dict(dimensions),
        "evidenceRequirementRate": dimensions["evidence_requirement"] / total if total else 0.0,
        "splitCounts": dict(Counter(row["split"] for row in normalized)),
        "taskFamilyCounts": family_counts,
        "crossSplitStateCount": len(split_states["train"] & split_states["val"]),
        "crossSplitFamilyCount": sum(len(splits) > 1 for splits in family_splits.values()),
        "excludedCounts": dict(excluded),
        "errors": errors,
        "datasetClaim": "Controlled Agent Policy Preference Dataset; no open-policy provenance claim",
    }
    return normalized, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--reviewed", type=Path, required=True)
    parser.add_argument("--legacy-hard", type=Path)
    parser.add_argument("--legacy-supplement", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--max-per-state", type=int, default=2)
    args = parser.parse_args()
    rows, audit = build(args.candidates, args.reviewed, args.legacy_hard, args.legacy_supplement, args.max_per_state)
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    args.audit_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: audit[key] for key in ("status", "recordCount", "uniqueStateCount", "maxPairsPerState", "sourceCounts", "chosenActionCounts", "chosenMaxMinRatio", "preferenceDimensionCounts", "evidenceRequirementRate", "splitCounts", "errors")}, ensure_ascii=False))
    return 0 if audit["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
