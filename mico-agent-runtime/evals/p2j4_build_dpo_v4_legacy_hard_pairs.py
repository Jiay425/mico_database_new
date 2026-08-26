"""Build review-only pairs from explicitly labelled legacy hard cases.

The legacy audit contains state fields and an oracle-selected action, but it
does not contain per-step planner provenance.  This builder therefore keeps
the rows as hard-development evidence and marks provenance as unknown.  It
never promotes unlabeled legacy rows, invents a model action, or overwrites
the source audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
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
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("candidates", []))


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
        {
            "role": "system",
            "content": "Choose exactly one next action from the supplied de-identified policy state.",
        },
        {
            "role": "user",
            "content": json.dumps({"policy_state": state}, ensure_ascii=False, sort_keys=True),
        },
    ]


def _rationale(row: dict[str, Any], chosen: str, rejected: str) -> tuple[str, list[str], str]:
    hard_case = row["hardCaseClass"]
    decision_reason = row.get("decisionReason") or ""
    if hard_case == "confounder_trap":
        dimension = "evidence_requirement"
        codes = ["CONFOUNDER_TRAP_BOUNDARY", "OBSERVATIONAL_EVIDENCE_REQUIRED"]
        basis = "The source state is explicitly labelled confounder_trap."
    elif hard_case == "premature_stop":
        dimension = "stop_boundary"
        codes = ["PREMATURE_STOP_BOUNDARY", "UNRESOLVED_EVIDENCE_OBLIGATION"]
        basis = "The source state is explicitly labelled premature_stop."
    else:
        raise ValueError(f"unsupported hard case: {hard_case}")
    if decision_reason:
        basis += f" Recorded decision basis: {decision_reason}."
    rationale = (
        f"Prefer {chosen} over {rejected}: {basis} The rejected action remains a listed "
        "unperformed alternative, but requires separate semantic review before training."
    )
    return dimension, codes, rationale


def _pair(row: dict[str, Any], state: dict[str, Any], signature: str, rejected: str) -> dict[str, Any]:
    chosen = row["selectedAction"]
    dimension, codes, rationale = _rationale(row, chosen, rejected)
    return {
        "id": "dpo-v4-legacy-hard-" + _hash([signature, chosen, rejected]),
        "prompt": _prompt(state),
        "chosen": json.dumps(
            {
                "selected_action": chosen,
                "decision_reason": rationale,
                "alternative_actions": [rejected],
                "stop_reason": "EVIDENCE_SUFFICIENT" if chosen == "finish" else None,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        "rejected": json.dumps(
            {
                "selected_action": rejected,
                "decision_reason": "Listed legal unperformed alternative; semantic review is required.",
                "alternative_actions": [chosen],
                "stop_reason": None,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        **state,
        "state_signature": signature,
        "task_family": row["taskFamily"],
        "source_trace_id": row["sourceTraceId"],
        "source_file": "p2j4-dpo-v4-real-source-audit-20260825.json",
        "case_id": None,
        "source_kind": "hard_development_trace",
        "scenario_class": "legacy_hard_case",
        "planner_origin": None,
        "provenance_class": "provenance_unknown_legacy",
        "raw_action": None,
        "final_action": chosen,
        "chosen_action": chosen,
        "rejected_action": rejected,
        "hard_case_class": row["hardCaseClass"],
        "repair_codes": [],
        "preference_type": "action",
        "preference_dimension": dimension,
        "preference_codes": codes,
        "preference_rationale": rationale,
        "preference_rationale_source": "legacy_decision_reason_plus_hard_case_class_v1",
        "review_status": "SEMANTIC_REVIEW_REQUIRED",
    }


def build(path: Path, max_per_state: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    states_seen: set[str] = set()
    state_counts: Counter[str] = Counter()
    for row in _read(path):
        if not row.get("hardCaseClass"):
            excluded["HARD_CASE_CLASS_MISSING"] += 1
            continue
        try:
            state = _state(row)
        except (KeyError, TypeError):
            excluded["STRUCTURED_FIELDS_INCOMPLETE"] += 1
            continue
        candidates = state["candidate_actions"]
        chosen = row.get("selectedAction")
        alternatives = list(row.get("legalUnperformedAlternatives") or [])
        if not (4 <= len(candidates) <= 7):
            excluded["CANDIDATE_WIDTH_OUTSIDE_4_7"] += 1
            continue
        if len(candidates) != len(set(candidates)) or any(action not in ACTIONS for action in candidates):
            excluded["CANDIDATE_ACTION_CONTRACT_INVALID"] += 1
            continue
        if chosen not in candidates or chosen not in ACTIONS:
            excluded["CHOSEN_ACTION_INVALID"] += 1
            continue
        alternatives = [
            action for action in alternatives
            if action in candidates and action in ACTIONS and action != chosen
            and action not in set(state["history_actions"])
        ]
        if not alternatives:
            excluded["NO_LEGAL_UNPERFORMED_ALTERNATIVE"] += 1
            continue
        signature = _hash(state)
        states_seen.add(signature)
        for rejected in alternatives:
            if state_counts[signature] >= max_per_state:
                excluded["STATE_PAIR_CAP_REACHED"] += 1
                break
            key = (signature, chosen, rejected)
            if any((item["state_signature"], item["chosen_action"], item["rejected_action"]) == key for item in output):
                excluded["DUPLICATE_PAIR"] += 1
                continue
            output.append(_pair(row, state, signature, rejected))
            state_counts[signature] += 1
    audit = {
        "schemaVersion": "p2j4-dpo-v4-legacy-hard-pairs-audit-v1",
        "status": "SEMANTIC_REVIEW_REQUIRED" if output else "BLOCKED",
        "trainingStarted": False,
        "pairConstructionStarted": True,
        "sourceFile": str(path),
        "sourceCandidateCount": len(_read(path)),
        "labelledHardStateCount": len(states_seen),
        "pairCount": len(output),
        "uniqueStateCount": len(state_counts),
        "maxPairsPerState": max(state_counts.values()) if state_counts else 0,
        "chosenActionCounts": dict(Counter(item["chosen_action"] for item in output)),
        "rejectedActionCounts": dict(Counter(item["rejected_action"] for item in output)),
        "hardCaseClassCounts": dict(Counter(item["hard_case_class"] for item in output)),
        "candidateWidthCounts": dict(Counter(len(item["candidate_actions"]) for item in output)),
        "excludedCounts": dict(excluded),
        "provenanceUnknown": True,
        "rules": [
            "only nonempty hardCaseClass rows are used",
            "selectedAction is treated as legacy oracle action, never as model-origin",
            "each rejected action comes from legalUnperformedAlternatives",
            "all six structured state fields are copied without fabrication",
            "at most max_per_state pairs are emitted per normalized state",
            "all rows remain semantic-review-required",
        ],
        "nextAction": "merge_only_after_pair_contract_and_semantic_review_audit",
    }
    return output, audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--max-per-state", type=int, default=3)
    args = parser.parse_args()
    rows, audit = build(args.input, args.max_per_state)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    args.audit_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: audit[key] for key in (
        "status", "pairCount", "uniqueStateCount", "maxPairsPerState",
        "chosenActionCounts", "excludedCounts",
    )}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
