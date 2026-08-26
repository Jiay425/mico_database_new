"""Fail-closed rule review for the final DPO-v4 staging candidate set.

The review is deliberately explicit about its nature: it is a deterministic
state/provenance audit, not an asserted human annotation.  Every accepted
pair receives a reviewer label and basis; any failed row remains in the
output with REVIEW_FAILED so a later stage cannot silently omit it.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _base(row: dict[str, Any]) -> str | None:
    try:
        chosen = json.loads(row["chosen"])["selected_action"]
        rejected = json.loads(row["rejected"])["selected_action"]
        state = json.loads(row["prompt"][1]["content"])["policy_state"]
    except (IndexError, KeyError, TypeError, json.JSONDecodeError):
        return "RECORD_SHAPE_INVALID"
    for field in ("task_kind", "goal_code", "observation_flags", "history_actions", "candidate_actions", "state_summary"):
        if state.get(field) != row.get(field):
            return "PROMPT_STATE_MISMATCH_" + field
    if chosen != row.get("chosen_action") or rejected != row.get("rejected_action"):
        return "COMPLETION_ACTION_MISMATCH"
    if chosen == rejected or chosen not in row.get("candidate_actions", []) or rejected not in row.get("candidate_actions", []):
        return "PAIR_ACTION_CONTRACT_INVALID"
    if not row.get("preference_dimension") or not row.get("preference_codes") or not row.get("preference_rationale"):
        return "PREFERENCE_RATIONALE_INCOMPLETE"
    return None


def _review(row: dict[str, Any]) -> tuple[bool, str]:
    error = _base(row)
    if error:
        return False, error
    source = row.get("source_kind")
    scenario = row.get("scenario_class")
    chosen = row["chosen_action"]
    rejected = row["rejected_action"]
    history = set(row.get("history_actions") or [])
    if source == "model_origin_policy_trace":
        ok = (
            row.get("planner_origin") == "model"
            and row.get("provenance_class") == "model_origin_raw"
            and row.get("raw_action") == row.get("final_action") == chosen
            and not row.get("repair_codes")
            and chosen not in history
        )
        return ok, "MODEL_RAW_DECISION_CONTRACT" if ok else "MODEL_RAW_PROVENANCE_INVALID"
    if source != "hard_development_trace":
        return False, "SOURCE_KIND_INVALID"
    if scenario == "runtime_repair_derived":
        ok = (
            row.get("planner_origin") == "model"
            and row.get("provenance_class") == "runtime_repair_derived"
            and row.get("raw_action") == rejected
            and row.get("final_action") == chosen
            and bool(row.get("repair_codes"))
            and chosen not in history
        )
        return ok, "RUNTIME_REPAIR_CONTRACT" if ok else "RUNTIME_REPAIR_PROVENANCE_INVALID"
    if scenario == "legacy_hard_case":
        ok = (
            row.get("planner_origin") is None
            and row.get("provenance_class") == "provenance_unknown_legacy"
            and row.get("hard_case_class") in {"confounder_trap", "premature_stop"}
            and row.get("raw_action") is None
            and chosen not in history
            and rejected not in history
        )
        return ok, "LEGACY_LABELLED_HARD_CONTRACT" if ok else "LEGACY_HARD_CONTRACT_INVALID"
    if scenario == "legacy_metadata_boundary":
        flags = set(row.get("observation_flags") or [])
        ok = (
            row.get("planner_origin") is None
            and row.get("provenance_class") == "provenance_unknown_legacy"
            and row.get("hard_case_class") == "metadata_first_boundary"
            and row.get("raw_action") is None
            and chosen == "inspect_cohort"
            and not history
            and {"NO_OBSERVATION", "METADATA_FIRST"}.issubset(flags)
        )
        return ok, "LEGACY_METADATA_BOUNDARY_CONTRACT" if ok else "LEGACY_METADATA_BOUNDARY_INVALID"
    return False, "HARD_SCENARIO_UNRECOGNIZED"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    rows = _rows(args.input)
    output: list[dict[str, Any]] = []
    errors: Counter[str] = Counter()
    basis_counts: Counter[str] = Counter()
    for row in rows:
        item = dict(row)
        ok, basis = _review(item)
        item["reviewer_label"] = "p2j4_dpo_v4_rule_audit_v1"
        item["review_basis"] = basis
        item["reviewed_at"] = "2026-08-26"
        item["review_status"] = "RULE_AUDITED" if ok else "REVIEW_FAILED"
        output.append(item)
        basis_counts[basis] += 1
        if not ok:
            errors[basis] += 1
    audit = {
        "schemaVersion": "p2j4-dpo-v4-rule-review-audit-v1",
        "status": "PASS" if not errors else "FAIL",
        "trainingStarted": False,
        "inputPairCount": len(rows),
        "ruleAuditedCount": sum(row["review_status"] == "RULE_AUDITED" for row in output),
        "reviewFailedCount": sum(row["review_status"] == "REVIEW_FAILED" for row in output),
        "reviewBasisCounts": dict(basis_counts),
        "errors": dict(errors),
        "reviewerLabel": "p2j4_dpo_v4_rule_audit_v1",
        "reviewNature": "deterministic provenance and state-contract audit; not a claimed human annotation",
    }
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in output) + "\n", encoding="utf-8")
    args.audit_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: audit[key] for key in ("status", "inputPairCount", "ruleAuditedCount", "reviewFailedCount", "errors")}, ensure_ascii=False))
    return 0 if audit["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
