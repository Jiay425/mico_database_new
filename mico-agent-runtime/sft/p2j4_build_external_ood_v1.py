"""Build a fixed 30-case OOD Decision-policy evaluation set.

The set is deliberately policy-state only.  It tests unseen compositions of
goal codes, observation flags, action history, and candidate actions without
putting raw questions, disease names, SQL, IDs, or measurements across the
Decision-SFT boundary.
"""

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


def _record(
    case_id: str,
    family: str,
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
        raise ValueError(f"action is not a candidate: {case_id}")
    alternatives = [item for item in candidates if item != action]
    if action == "finish" and not stop_reason:
        raise ValueError(f"finish needs a stop reason: {case_id}")
    if action != "finish" and stop_reason is not None:
        raise ValueError(f"non-terminal action has a stop reason: {case_id}")
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
            f"ood_family={family}"
        ),
    }
    target = {
        "selected_action": action,
        "decision_reason": reason,
        "alternative_actions": alternatives,
        "stop_reason": stop_reason,
    }
    return {
        "id": case_id,
        "ood_family": family,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps({"policy_state": state}, ensure_ascii=False, sort_keys=True)},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False, separators=(",", ":"))},
        ],
    }


def build_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    add = cases.append

    # Data facts: the unseen composition is that a validated metadata read is
    # not itself the requested species/project coverage result.
    add(_record("ood-datafact-01", "metadata_then_projection", "data_fact", "cohort_fact", ["NO_OBSERVATION", "METADATA_FIRST"], [], ["inspect_cohort", "execute_read_query", "finish"], "inspect_cohort", "inspect metadata before reading a bounded projection"))
    add(_record("ood-datafact-02", "metadata_then_projection", "data_fact", "cohort_fact", ["OBSERVATION_VALIDATED", "JAVA_OBSERVED", "QUERY_REQUIRED"], ["inspect_cohort"], ["inspect_cohort", "execute_read_query", "finish"], "execute_read_query", "run the bounded read needed for the requested fact"))
    add(_record("ood-datafact-03", "metadata_then_projection", "data_fact", "species_coverage", ["OBSERVATION_VALIDATED", "JAVA_OBSERVED", "QUERY_REQUIRED"], ["inspect_cohort"], ["execute_read_query", "finish"], "execute_read_query", "retrieve the bounded coverage result after metadata validation"))
    add(_record("ood-datafact-04", "metadata_then_projection", "data_fact", "project_coverage", ["OBSERVATION_VALIDATED", "JAVA_OBSERVED", "QUERY_REQUIRED", "PROJECT_DIMENSION"], ["inspect_cohort"], ["inspect_cohort", "execute_read_query", "finish"], "execute_read_query", "query the requested project-level fact within the approved read boundary"))
    add(_record("ood-datafact-05", "validated_fact_stop", "data_fact", "species_coverage", ["OBSERVATION_VALIDATED", "JAVA_OBSERVED", "QUERY_EXECUTED"], ["inspect_cohort", "execute_read_query"], ["execute_read_query", "finish"], "finish", "the requested bounded fact has been validated", "EVIDENCE_SUFFICIENT"))
    add(_record("ood-datafact-06", "validated_fact_stop", "data_fact", "project_coverage", ["OBSERVATION_VALIDATED", "JAVA_OBSERVED", "QUERY_EXECUTED", "PROJECT_DIMENSION"], ["inspect_cohort", "execute_read_query"], ["execute_read_query", "finish"], "finish", "the requested project-level fact has been validated", "EVIDENCE_SUFFICIENT"))

    # Focused analysis: unseen combinations distinguish a valid comparison,
    # a confounder trap, and a completed analysis.
    add(_record("ood-focused-07", "focused_path", "focused_analysis", "group_comparison", ["NO_OBSERVATION", "METADATA_FIRST"], [], ["inspect_cohort", "compare_groups", "finish"], "inspect_cohort", "inspect cohort structure before comparing groups"))
    add(_record("ood-focused-08", "focused_path", "focused_analysis", "group_comparison", ["OBSERVATION_VALIDATED", "JAVA_OBSERVED", "GROUPS_AVAILABLE"], ["inspect_cohort"], ["compare_groups", "adjust_confounders", "finish"], "compare_groups", "compare the approved groups using the validated cohort"))
    add(_record("ood-focused-09", "focused_path", "focused_analysis", "group_comparison", ["OBSERVATION_VALIDATED", "JAVA_OBSERVED", "GROUPS_COMPARED"], ["inspect_cohort", "compare_groups"], ["analyze_projection", "adjust_confounders", "finish"], "analyze_projection", "summarize the validated comparison before concluding"))
    add(_record("ood-focused-10", "focused_path", "focused_analysis", "group_comparison", ["OBSERVATION_VALIDATED", "JAVA_OBSERVED", "GROUPS_COMPARED", "CONFOUNDER_PRESENT", "PROJECT_IMBALANCE"], ["inspect_cohort", "compare_groups"], ["adjust_confounders", "analyze_projection", "finish"], "adjust_confounders", "control the observed confounding before interpreting the comparison"))
    add(_record("ood-focused-11", "focused_path", "focused_analysis", "confounder_control", ["OBSERVATION_VALIDATED", "JAVA_OBSERVED", "CONFOUNDER_ADJUSTED"], ["inspect_cohort", "compare_groups", "adjust_confounders"], ["analyze_projection", "cross_project_validate", "finish"], "analyze_projection", "summarize the comparison after confounder adjustment"))
    add(_record("ood-focused-12", "focused_path", "focused_analysis", "group_comparison", ["OBSERVATION_VALIDATED", "JAVA_OBSERVED", "ANALYSIS_AVAILABLE"], ["inspect_cohort", "compare_groups", "analyze_projection"], ["retrieve_evidence", "finish"], "finish", "the bounded comparison and analysis are complete", "EVIDENCE_SUFFICIENT"))

    # Open exploration: each state changes the next scientific obligation.
    add(_record("ood-open-13", "exploration_path", "open_exploration", "scientific_exploration", ["NO_OBSERVATION", "METADATA_FIRST"], [], ["inspect_cohort", "compare_groups", "finish"], "inspect_cohort", "inspect the cohort before proposing an exploratory finding"))
    add(_record("ood-open-14", "exploration_path", "open_exploration", "cross_project_stability", ["OBSERVATION_VALIDATED", "JAVA_OBSERVED", "PROJECT_IMBALANCE"], ["inspect_cohort", "compare_groups"], ["cross_project_validate", "adjust_confounders", "finish"], "cross_project_validate", "test whether the observed pattern is stable across projects"))
    add(_record("ood-open-15", "exploration_path", "open_exploration", "cross_project_stability", ["OBSERVATION_VALIDATED", "JAVA_OBSERVED", "PROJECT_VALIDATED"], ["inspect_cohort", "compare_groups", "cross_project_validate"], ["retrieve_evidence", "adjust_confounders", "finish"], "retrieve_evidence", "add bounded external evidence after cross-project validation"))
    add(_record("ood-open-16", "exploration_path", "open_exploration", "cross_project_stability", ["EVIDENCE_RETRIEVED", "JAVA_OBSERVED", "PROJECT_VALIDATED"], ["inspect_cohort", "compare_groups", "cross_project_validate", "retrieve_evidence"], ["finish", "retrieve_evidence"], "finish", "data stability and evidence review are complete", "EVIDENCE_SUFFICIENT"))
    add(_record("ood-open-17", "exploration_path", "open_exploration", "confounder_control", ["OBSERVATION_VALIDATED", "JAVA_OBSERVED", "CONFOUNDER_PRESENT", "PROJECT_IMBALANCE"], ["inspect_cohort", "compare_groups"], ["adjust_confounders", "cross_project_validate", "finish"], "adjust_confounders", "control the detected confounding before judging the finding"))
    add(_record("ood-open-18", "exploration_path", "open_exploration", "confounder_control", ["OBSERVATION_VALIDATED", "JAVA_OBSERVED", "CONFOUNDER_ADJUSTED"], ["inspect_cohort", "compare_groups", "adjust_confounders"], ["cross_project_validate", "retrieve_evidence", "finish"], "cross_project_validate", "check whether the adjusted finding repeats across projects"))
    add(_record("ood-open-19", "exploration_path", "open_exploration", "confounder_control", ["OBSERVATION_VALIDATED", "JAVA_OBSERVED", "CONFOUNDER_ADJUSTED", "PROJECT_VALIDATED"], ["inspect_cohort", "compare_groups", "adjust_confounders", "cross_project_validate"], ["retrieve_evidence", "finish"], "retrieve_evidence", "seek bounded evidence after adjustment and stability checks"))
    add(_record("ood-open-20", "exploration_path", "open_exploration", "confounder_control", ["EVIDENCE_RETRIEVED", "JAVA_OBSERVED", "CONFOUNDER_ADJUSTED", "PROJECT_VALIDATED"], ["inspect_cohort", "compare_groups", "adjust_confounders", "cross_project_validate", "retrieve_evidence"], ["finish", "retrieve_evidence"], "finish", "the adjusted, replicated, and grounded finding is ready to report", "EVIDENCE_SUFFICIENT"))
    add(_record("ood-open-21", "exploration_path", "open_exploration", "disease_specificity", ["OBSERVATION_VALIDATED", "JAVA_OBSERVED", "ANALYSIS_AVAILABLE"], ["inspect_cohort", "compare_groups", "analyze_projection"], ["cross_disease_validate", "retrieve_evidence", "finish"], "cross_disease_validate", "test whether the finding is specific to the target disease context"))
    add(_record("ood-open-22", "exploration_path", "open_exploration", "disease_specificity", ["OBSERVATION_VALIDATED", "JAVA_OBSERVED", "DISEASE_VALIDATED"], ["inspect_cohort", "compare_groups", "analyze_projection", "cross_disease_validate"], ["retrieve_evidence", "finish"], "retrieve_evidence", "ground the disease-specificity result with bounded evidence"))
    add(_record("ood-open-23", "exploration_path", "open_exploration", "disease_specificity", ["EVIDENCE_RETRIEVED", "JAVA_OBSERVED", "DISEASE_VALIDATED"], ["inspect_cohort", "compare_groups", "analyze_projection", "cross_disease_validate", "retrieve_evidence"], ["finish", "retrieve_evidence"], "finish", "specificity validation and evidence grounding are complete", "EVIDENCE_SUFFICIENT"))

    # Evidence review/conflict: a conflict is a downgrade/quality-risk state,
    # not a license to claim a biomarker.
    add(_record("ood-evidence-24", "evidence_conflict", "evidence_review", "evidence_conflict_resolution", ["NO_OBSERVATION", "METADATA_FIRST", "EVIDENCE_CONFLICT"], [], ["inspect_cohort", "retrieve_evidence", "finish"], "inspect_cohort", "establish the data observation before resolving evidence conflict"))
    add(_record("ood-evidence-25", "evidence_conflict", "evidence_review", "evidence_conflict_resolution", ["OBSERVATION_VALIDATED", "JAVA_OBSERVED", "EVIDENCE_CONFLICT"], ["inspect_cohort", "compare_groups"], ["retrieve_evidence", "finish"], "retrieve_evidence", "seek additional bounded evidence to characterize the conflicting claims"))
    add(_record("ood-evidence-26", "evidence_conflict", "evidence_review", "evidence_conflict_resolution", ["EVIDENCE_RETRIEVED", "JAVA_OBSERVED", "EVIDENCE_CONFLICT", "QUALITY_RISK"], ["inspect_cohort", "compare_groups", "retrieve_evidence"], ["finish", "retrieve_evidence"], "finish", "the conflict remains and the report must downgrade the claim", "QUALITY_RISK"))
    add(_record("ood-evidence-27", "evidence_conflict", "open_exploration", "evidence_conflict_resolution", ["EVIDENCE_RETRIEVED", "EVIDENCE_CONFLICT", "QUALITY_RISK"], ["inspect_cohort", "retrieve_evidence"], ["finish", "retrieve_evidence"], "finish", "remaining conflict is an explicit quality risk, not a positive conclusion", "QUALITY_RISK"))
    add(_record("ood-evidence-28", "evidence_grounding", "evidence_review", "evidence_grounding", ["NO_OBSERVATION", "METADATA_FIRST"], [], ["inspect_cohort", "retrieve_evidence", "finish"], "inspect_cohort", "inspect the data basis before retrieving supporting evidence"))
    add(_record("ood-evidence-29", "evidence_grounding", "evidence_review", "evidence_grounding", ["OBSERVATION_VALIDATED", "JAVA_OBSERVED", "ANALYSIS_AVAILABLE"], ["inspect_cohort", "compare_groups", "analyze_projection"], ["retrieve_evidence", "finish"], "retrieve_evidence", "ground the analyzed observation with bounded evidence"))
    add(_record("ood-evidence-30", "evidence_grounding", "evidence_review", "evidence_grounding", ["EVIDENCE_RETRIEVED", "JAVA_OBSERVED", "ANALYSIS_AVAILABLE"], ["inspect_cohort", "compare_groups", "analyze_projection", "retrieve_evidence"], ["finish", "retrieve_evidence"], "finish", "the analyzed observation has supporting evidence and is ready to report", "EVIDENCE_SUFFICIENT"))

    if len(cases) != 30:
        raise AssertionError(f"expected 30 OOD cases, got {len(cases)}")
    if len({case["id"] for case in cases}) != len(cases):
        raise AssertionError("OOD case IDs must be unique")
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("evals/p2j4-external-ood-v1"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = build_cases()
    records_path = args.output_dir / "records.jsonl"
    records_path.write_text(
        "".join(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n" for case in cases),
        encoding="utf-8",
    )
    manifest = {
        "schemaVersion": "p2j4-external-ood-decision-v1",
        "status": "OWNER_REVIEW_REQUIRED",
        "trainingStarted": False,
        "scope": "deidentified_policy_state_composition_ood",
        "recordCount": len(cases),
        "families": {family: sum(case["ood_family"] == family for case in cases) for family in sorted({case["ood_family"] for case in cases})},
        "recordsFile": records_path.name,
        "sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
        "excludedFromModelBoundary": ["raw_question", "disease_name", "sql", "sample_id", "numeric_measurement"],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "recordCount": len(cases), "outputDir": str(args.output_dir), "sha256": manifest["sha256"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
