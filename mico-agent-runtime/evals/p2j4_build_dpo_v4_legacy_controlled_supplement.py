"""Select a small, state-obligation-audited legacy supplement offline."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


TARGETS = {"analyze_projection": 15, "stratified_analysis": 11, "compare_groups": 12}


def _hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _eligible(row: dict[str, Any]) -> bool:
    action = row.get("selectedAction")
    flags = set(row.get("observationFlags") or [])
    history = set(row.get("historyActions") or [])
    if row.get("hardCaseClass") or action in history or action not in TARGETS:
        return False
    if action == "analyze_projection":
        return "ANALYSIS_AVAILABLE" in flags and "OBSERVATION_VALIDATED" in flags
    if action == "stratified_analysis":
        return "OBSERVATION_VALIDATED" in flags and bool({"JAVA_OBSERVED", "ANALYSIS_AVAILABLE"} & flags)
    return (
        row.get("goalCode") == "group_comparison"
        or bool({"CROSS_PROJECT_REQUIRED", "CONFOUNDER_PRESENT", "PROJECT_IMBALANCE"} & flags)
    )


def _pair(row: dict[str, Any], rejected: str) -> dict[str, Any]:
    chosen = row["selectedAction"]
    state = {
        "task_kind": row["taskKind"],
        "goal_code": row["goalCode"],
        "observation_flags": list(row["observationFlags"]),
        "history_actions": list(row["historyActions"]),
        "candidate_actions": list(row["candidateActions"]),
        "state_summary": row["stateSummary"],
    }
    signature = _hash(state)
    if chosen == "analyze_projection":
        codes = ["ANALYSIS_AVAILABLE", "PROJECTION_ANALYSIS_NEXT"]
        rationale = "A validated analyzable observation is available, so bounded projection analysis is preferred before switching to the legal alternative."
    elif chosen == "stratified_analysis":
        codes = ["OBSERVATION_VALIDATED", "STRATIFIED_FOLLOWUP_SUPPORTED"]
        rationale = "The validated observation supports a bounded stratified follow-up before the legal alternative."
    else:
        codes = ["GROUP_COMPARISON_REQUIRED", "VALIDATED_OBSERVATION_AVAILABLE"]
        rationale = "The current state still requires the bounded group comparison before the legal downstream alternative."
    chosen_json = {"selected_action": chosen, "decision_reason": rationale, "alternative_actions": [rejected], "stop_reason": None}
    rejected_json = {"selected_action": rejected, "decision_reason": "Legal alternative retained for controlled preference comparison.", "alternative_actions": [chosen], "stop_reason": None}
    return {
        "id": "dpo-v4-legacy-controlled-" + _hash([signature, chosen, rejected]),
        "prompt": [
            {"role": "system", "content": "Choose exactly one next action from the supplied de-identified policy state."},
            {"role": "user", "content": json.dumps({"policy_state": state}, ensure_ascii=False, sort_keys=True)},
        ],
        "chosen": json.dumps(chosen_json, ensure_ascii=False, sort_keys=True),
        "rejected": json.dumps(rejected_json, ensure_ascii=False, sort_keys=True),
        **state,
        "state_signature": signature,
        "task_family": row["taskFamily"],
        "source_trace_id": row["sourceTraceId"],
        "source_file": "p2j4-dpo-v4-real-source-audit-20260825.json",
        "case_id": None,
        "source_kind": "hard_development_trace",
        "scenario_class": "legacy_controlled_supplement",
        "planner_origin": None,
        "provenance_class": "provenance_unknown_legacy",
        "raw_action": None,
        "final_action": chosen,
        "chosen_action": chosen,
        "rejected_action": rejected,
        "hard_case_class": "controlled_state_obligation",
        "repair_codes": [],
        "preference_type": "action",
        "preference_dimension": "exploration_depth" if chosen != "compare_groups" else "evidence_requirement",
        "preference_codes": codes,
        "preference_rationale": rationale,
        "preference_rationale_source": "legacy_structured_state_obligation_v1",
        "review_status": "RULE_AUDITED",
    }


def build(path: Path, excluded_signatures: set[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = sorted(payload.get("candidates", []), key=lambda row: (row.get("selectedAction", ""), row.get("sourceTraceId", "")))
    output: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    states: set[str] = set()
    excluded: Counter[str] = Counter()
    for row in rows:
        if not _eligible(row):
            excluded["STATE_OBLIGATION_NOT_ELIGIBLE"] += 1
            continue
        action = row["selectedAction"]
        if counts[action] >= TARGETS[action]:
            continue
        candidates = list(row.get("candidateActions") or [])
        alternatives = [item for item in row.get("legalUnperformedAlternatives") or [] if item in candidates and item != action and item not in set(row.get("historyActions") or [])]
        if not 4 <= len(candidates) <= 7 or action not in candidates or not alternatives:
            excluded["PAIR_CONTRACT_INVALID"] += 1
            continue
        pair = _pair(row, alternatives[0])
        if pair["state_signature"] in excluded_signatures:
            excluded["FROZEN_EVAL_STATE_EXCLUDED"] += 1
            continue
        if pair["state_signature"] in states:
            excluded["DUPLICATE_STATE"] += 1
            continue
        output.append(pair)
        states.add(pair["state_signature"])
        counts[action] += 1
    errors = [f"TARGET_NOT_MET:{action}" for action, target in TARGETS.items() if counts[action] != target]
    audit = {
        "schemaVersion": "p2j4-dpo-v4-legacy-controlled-supplement-audit-v1",
        "status": "PASS" if not errors else "FAIL",
        "trainingStarted": False,
        "apiCalls": 0,
        "traceRuns": 0,
        "pairCount": len(output),
        "uniqueStateCount": len(states),
        "chosenActionCounts": dict(counts),
        "errors": errors,
        "excludedCounts": dict(excluded),
        "provenanceClaim": "legacy structured state obligation; not model-origin",
    }
    return output, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--exclude-signature", action="append", default=[])
    args = parser.parse_args()
    rows, audit = build(args.input, set(args.exclude_signature))
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    args.audit_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False))
    return 0 if audit["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
