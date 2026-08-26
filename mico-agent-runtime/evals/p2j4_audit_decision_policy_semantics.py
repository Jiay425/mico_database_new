"""Rule-based, per-record semantic audit for the frozen Decision policy data.

This does not claim to replace expert review.  It checks the closed Agent
policy obligations that must hold before a record can be used for training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evals.p2j4_build_decision_freeze_v2 import DPO_OUTPUT, SFT_OUTPUT, _jsonl, _policy_state, _target


def _issues(state: dict[str, Any], target: dict[str, Any]) -> list[str]:
    flags = set(state["observation_flags"])
    history = state["history_actions"]
    selected = target["selected_action"]
    stop = target["stop_reason"]
    issues: list[str] = []
    if "NO_OBSERVATION" in flags and selected not in {"inspect_cohort", "execute_read_query"}:
        issues.append("NO_OBSERVATION_NON_READ_ACTION")
    if selected in history and selected != "finish":
        issues.append("REDUNDANT_SELECTED_ACTION")
    if "CONFOUNDER_PRESENT" in flags and "CONFOUNDER_ADJUSTED" not in flags and selected == "finish":
        issues.append("CONFOUNDER_PREMATURE_FINISH")
    if "CROSS_PROJECT_REQUIRED" in flags and "PROJECT_VALIDATED" not in flags and selected == "finish":
        issues.append("CROSS_PROJECT_PREMATURE_FINISH")
    if "CROSS_DISEASE_REQUIRED" in flags and "CROSS_DISEASE_VALIDATED" not in flags and selected == "finish":
        issues.append("CROSS_DISEASE_PREMATURE_FINISH")
    if "EVIDENCE_CONFLICT" in flags and selected == "finish" and stop != "QUALITY_RISK":
        issues.append("CONFLICT_STOP_NOT_QUALITY_RISK")
    if "EVIDENCE_INCOMPLETE" in flags and selected == "finish" and stop != "QUALITY_RISK":
        issues.append("INCOMPLETE_EVIDENCE_STOP_NOT_QUALITY_RISK")
    if selected == "adjust_confounders" and "CONFOUNDER_PRESENT" not in flags:
        issues.append("ADJUST_WITHOUT_CONFOUNDER_SIGNAL")
    if selected == "cross_project_validate" and "CROSS_PROJECT_REQUIRED" not in flags:
        issues.append("CROSS_PROJECT_WITHOUT_OBLIGATION")
    if selected == "cross_disease_validate" and "CROSS_DISEASE_REQUIRED" not in flags:
        issues.append("CROSS_DISEASE_WITHOUT_OBLIGATION")
    if selected == "retrieve_evidence" and not flags.intersection({"ANALYSIS_AVAILABLE", "PROJECT_VALIDATED", "CROSS_DISEASE_VALIDATED", "EVIDENCE_INCOMPLETE", "EVIDENCE_CONFLICT"}):
        issues.append("RETRIEVE_WITHOUT_DATA_OR_EVIDENCE_OBLIGATION")
    if selected == "execute_read_query" and "BOUNDED_READ_REQUIRED" not in flags:
        issues.append("READ_WITHOUT_BOUNDED_READ_OBLIGATION")
    if selected == "finish" and not stop:
        issues.append("FINISH_WITHOUT_STOP_REASON")
    return issues


def audit() -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for split in ("train", "validation"):
        for record in _jsonl(SFT_OUTPUT / f"{split}.jsonl"):
            state, target = _policy_state(record), _target(record)
            results.append({
                "id": record["id"], "split": split,
                "provenance": record["metadata"]["provenance"],
                "hardCaseClass": record["metadata"]["hard_case_class"],
                "selectedAction": target["selected_action"],
                "issues": _issues(state, target),
            })
    invalid = [item for item in results if item["issues"]]
    return {
        "schemaVersion": "p2j4-decision-policy-semantic-audit-v1",
        "status": "PASS" if not invalid else "FAIL",
        "recordCount": len(results),
        "invalidCount": len(invalid),
        "invalidByProvenance": {
            key: sum(1 for item in invalid if item["provenance"] == key)
            for key in sorted({item["provenance"] for item in invalid})
        },
        "issues": invalid,
        "trainingStarted": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "recordCount", "invalidCount", "invalidByProvenance")}, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
