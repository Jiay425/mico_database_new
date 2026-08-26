"""Build reviewed-candidate repairs for missing next-obligation signals."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SYSTEM = (
    "You are Mico's Scientific Agent decision policy. Choose the next action "
    "from the candidate actions in the policy state. Use only the supplied "
    "structured state and action history. Return one JSON object and no Markdown. "
    "The JSON keys must be exactly: selected_action, decision_reason, "
    "alternative_actions, stop_reason. Use stop_reason only when selected_action "
    "is finish."
)


def record(
    case_id: str,
    task_kind: str,
    goal_code: str,
    flags: list[str],
    history: list[str],
    candidates: list[str],
    action: str,
    reason: str,
    stop_reason: str | None = None,
) -> dict[str, Any]:
    if action not in candidates:
        raise ValueError(case_id + ": target is not a candidate")
    if action == "finish" and not stop_reason:
        raise ValueError(case_id + ": finish needs stop reason")
    if action != "finish" and stop_reason is not None:
        raise ValueError(case_id + ": non-terminal stop reason")
    state = {
        "task_kind": task_kind,
        "goal_code": goal_code,
        "observation_flags": flags,
        "history_actions": history,
        "candidate_actions": candidates,
        "state_summary": (
            f"observation_state={flags[0]}; evidence_bindings={len(history)}; "
            f"source_routes={'java' if 'JAVA_OBSERVED' in flags else 'none'}; "
            f"prior_actions={','.join(history) if history else 'none'}; "
            f"flags={','.join(flags)}"
        ),
    }
    target = {
        "selected_action": action,
        "decision_reason": reason,
        "alternative_actions": [item for item in candidates if item != action],
        "stop_reason": stop_reason,
    }
    return {
        "id": case_id,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps({"policy_state": state}, ensure_ascii=False, sort_keys=True)},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False, separators=(",", ":"))},
        ],
    }


def build() -> list[dict[str, Any]]:
    return [
        record("repair-obligation-001", "data_fact", "cohort_fact", ["OBSERVATION_VALIDATED", "JAVA_OBSERVED", "BOUNDED_READ_REQUIRED", "METADATA_FIRST"], ["inspect_cohort"], ["inspect_cohort", "execute_read_query", "finish"], "execute_read_query", "the metadata observation is valid but the requested bounded fact still requires a read"),
        record("repair-obligation-002", "data_fact", "species_coverage", ["OBSERVATION_VALIDATED", "JAVA_OBSERVED", "BOUNDED_READ_REQUIRED", "METADATA_FIRST"], ["inspect_cohort"], ["inspect_cohort", "execute_read_query", "finish"], "execute_read_query", "retrieve the requested species coverage after metadata validation"),
        record("repair-obligation-003", "data_fact", "project_coverage", ["OBSERVATION_VALIDATED", "JAVA_OBSERVED", "BOUNDED_READ_REQUIRED", "PROJECT_DIMENSION"], ["inspect_cohort"], ["inspect_cohort", "execute_read_query", "finish"], "execute_read_query", "the project-level fact is not complete until the bounded read is executed"),
        record("repair-obligation-004", "focused_analysis", "group_comparison", ["OBSERVATION_VALIDATED", "JAVA_OBSERVED", "ANALYSIS_REQUIRED"], ["inspect_cohort", "compare_groups"], ["analyze_projection", "adjust_confounders", "finish"], "analyze_projection", "the groups were compared, but the validated projection still needs analysis"),
        record("repair-obligation-005", "focused_analysis", "group_comparison", ["OBSERVATION_VALIDATED", "JAVA_OBSERVED", "ANALYSIS_REQUIRED", "CONFOUNDER_PRESENT"], ["inspect_cohort", "compare_groups"], ["adjust_confounders", "analyze_projection", "finish"], "adjust_confounders", "the comparison has a confounder signal that must be controlled before interpretation"),
        record("repair-obligation-006", "focused_analysis", "group_comparison", ["OBSERVATION_VALIDATED", "JAVA_OBSERVED", "ANALYSIS_AVAILABLE", "ANALYSIS_COMPLETE"], ["inspect_cohort", "compare_groups", "analyze_projection"], ["retrieve_evidence", "finish"], "finish", "the bounded comparison and completed projection analysis are sufficient", "EVIDENCE_SUFFICIENT"),
        record("repair-obligation-007", "open_exploration", "cross_project_stability", ["OBSERVATION_VALIDATED", "JAVA_OBSERVED", "CROSS_PROJECT_REQUIRED"], ["inspect_cohort", "compare_groups"], ["cross_project_validate", "adjust_confounders", "finish"], "cross_project_validate", "the observed pattern must be checked across projects before a conclusion"),
        record("repair-obligation-008", "open_exploration", "cross_project_stability", ["OBSERVATION_VALIDATED", "JAVA_OBSERVED", "PROJECT_VALIDATED", "EVIDENCE_REQUIRED"], ["inspect_cohort", "compare_groups", "cross_project_validate"], ["retrieve_evidence", "finish"], "retrieve_evidence", "cross-project stability is observed but external evidence is still required"),
        record("repair-obligation-009", "open_exploration", "cross_project_stability", ["EVIDENCE_RETRIEVED", "EVIDENCE_GROUNDED", "JAVA_OBSERVED", "PROJECT_VALIDATED"], ["inspect_cohort", "compare_groups", "cross_project_validate", "retrieve_evidence"], ["finish", "retrieve_evidence"], "finish", "the finding is validated across projects and the retrieved evidence is grounded", "EVIDENCE_SUFFICIENT"),
        record("repair-obligation-010", "open_exploration", "disease_specificity", ["OBSERVATION_VALIDATED", "JAVA_OBSERVED", "ANALYSIS_AVAILABLE", "DISEASE_VALIDATION_REQUIRED"], ["inspect_cohort", "compare_groups", "analyze_projection"], ["cross_disease_validate", "retrieve_evidence", "finish"], "cross_disease_validate", "the analyzed finding still requires a disease-specificity check"),
        record("repair-obligation-011", "evidence_review", "evidence_grounding", ["OBSERVATION_VALIDATED", "JAVA_OBSERVED", "ANALYSIS_AVAILABLE", "EVIDENCE_REQUIRED"], ["inspect_cohort", "compare_groups", "analyze_projection"], ["retrieve_evidence", "finish"], "retrieve_evidence", "the analyzed observation needs bounded evidence before reporting"),
        record("repair-obligation-012", "evidence_review", "evidence_grounding", ["EVIDENCE_RETRIEVED", "EVIDENCE_GROUNDED", "JAVA_OBSERVED", "ANALYSIS_AVAILABLE", "ANALYSIS_COMPLETE"], ["inspect_cohort", "compare_groups", "analyze_projection", "retrieve_evidence"], ["finish", "retrieve_evidence"], "finish", "the analyzed observation is grounded and can now be reported", "EVIDENCE_SUFFICIENT"),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("evals/p2j4-state-obligation-repair-v1"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = build()
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("repair IDs are not unique")
    output = args.output_dir / "records.jsonl"
    output.write_text("".join(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n" for case in cases), encoding="utf-8")
    manifest = {
        "schemaVersion": "p2j4-state-obligation-repair-v1",
        "status": "OWNER_REVIEW_REQUIRED",
        "trainingStarted": False,
        "recordCount": len(cases),
        "sourceFailures": ["data-fact-004", "ood-focused-09", "ood-focused-11", "ood-focused-12", "ood-open-15", "ood-open-18", "ood-open-19", "ood-open-22", "ood-evidence-29"],
        "reasonReviewCases": ["ood-open-20", "ood-evidence-30"],
        "recordsFile": output.name,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "inputFields": ["task_kind", "goal_code", "observation_flags", "history_actions", "candidate_actions", "state_summary"],
        "targetFields": ["selected_action", "decision_reason", "alternative_actions", "stop_reason"],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "recordCount": len(cases), "sha256": manifest["sha256"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
