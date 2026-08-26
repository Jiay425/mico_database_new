"""Build a second targeted pool with distinct candidate widths and paths."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mico_agent_runtime.contracts.research import ScientificObservationSummary, ScientificPlannerContext
from mico_agent_runtime.ports.scientific_planner import semantic_scientific_next_action


VERSION = "p2j4-dpo-v4-targeted-weak-actions-v2"
FORBIDDEN = ["direct_mysql", "unbounded_export", "diagnosis", "causal_claim"]
TARGETS = (
    ("execute", [
        ["inspect_cohort", "execute_read_query", "finish"],
        ["inspect_cohort", "execute_read_query", "compare_groups", "finish"],
        ["inspect_cohort", "execute_read_query", "analyze_projection", "finish"],
        ["inspect_cohort", "execute_read_query", "retrieve_evidence", "finish"],
        ["inspect_cohort", "execute_read_query", "compare_groups", "analyze_projection", "finish"],
        ["inspect_cohort", "execute_read_query", "compare_groups", "retrieve_evidence", "finish"],
        ["inspect_cohort", "execute_read_query", "stratified_analysis", "analyze_projection", "finish"],
        ["inspect_cohort", "execute_read_query", "adjust_confounders", "analyze_projection", "finish"],
        ["inspect_cohort", "execute_read_query", "cross_project_validate", "finish"],
        ["inspect_cohort", "execute_read_query", "cross_disease_validate", "finish"],
        ["inspect_cohort", "execute_read_query", "compare_groups", "analyze_projection", "retrieve_evidence", "finish"],
        ["inspect_cohort", "execute_read_query", "compare_groups", "stratified_analysis", "analyze_projection", "finish"],
    ]),
    ("cross-disease", [
        ["inspect_cohort", "compare_groups", "cross_disease_validate", "analyze_projection", "finish"],
        ["inspect_cohort", "compare_groups", "cross_disease_validate", "retrieve_evidence", "finish"],
        ["inspect_cohort", "compare_groups", "cross_disease_validate", "analyze_projection", "retrieve_evidence", "finish"],
        ["inspect_cohort", "compare_groups", "cross_disease_validate", "cross_project_validate", "finish"],
        ["inspect_cohort", "compare_groups", "cross_disease_validate", "adjust_confounders", "analyze_projection", "finish"],
        ["inspect_cohort", "compare_groups", "cross_disease_validate", "stratified_analysis", "analyze_projection", "finish"],
        ["inspect_cohort", "compare_groups", "cross_disease_validate", "retrieve_evidence", "analyze_projection", "finish"],
        ["inspect_cohort", "compare_groups", "cross_disease_validate", "cross_project_validate", "analyze_projection", "finish"],
        ["inspect_cohort", "compare_groups", "cross_disease_validate", "adjust_confounders", "cross_project_validate", "finish"],
        ["inspect_cohort", "compare_groups", "cross_disease_validate", "stratified_analysis", "retrieve_evidence", "finish"],
        ["inspect_cohort", "compare_groups", "cross_disease_validate", "cross_project_validate", "retrieve_evidence", "finish"],
        ["inspect_cohort", "compare_groups", "cross_disease_validate", "adjust_confounders", "analyze_projection", "retrieve_evidence", "finish"],
    ]),
    ("stratified", [
        ["inspect_cohort", "compare_groups", "stratified_analysis", "analyze_projection", "finish"],
        ["inspect_cohort", "compare_groups", "stratified_analysis", "retrieve_evidence", "finish"],
        ["inspect_cohort", "compare_groups", "stratified_analysis", "analyze_projection", "retrieve_evidence", "finish"],
        ["inspect_cohort", "compare_groups", "stratified_analysis", "cross_project_validate", "finish"],
        ["inspect_cohort", "compare_groups", "stratified_analysis", "adjust_confounders", "analyze_projection", "finish"],
        ["inspect_cohort", "compare_groups", "stratified_analysis", "cross_disease_validate", "analyze_projection", "finish"],
        ["inspect_cohort", "compare_groups", "stratified_analysis", "retrieve_evidence", "analyze_projection", "finish"],
        ["inspect_cohort", "compare_groups", "stratified_analysis", "cross_project_validate", "analyze_projection", "finish"],
        ["inspect_cohort", "compare_groups", "stratified_analysis", "adjust_confounders", "cross_project_validate", "finish"],
        ["inspect_cohort", "compare_groups", "stratified_analysis", "cross_disease_validate", "retrieve_evidence", "finish"],
        ["inspect_cohort", "compare_groups", "stratified_analysis", "cross_project_validate", "retrieve_evidence", "finish"],
        ["inspect_cohort", "compare_groups", "stratified_analysis", "adjust_confounders", "analyze_projection", "retrieve_evidence", "finish"],
    ]),
)


def _obs(index: int, action: str) -> ScientificObservationSummary:
    return ScientificObservationSummary(
        observationId=f"observation-{index:032x}", actionName=action, status="VALIDATED",
        source="java_controlled_read" if action in {"inspect_cohort", "compare_groups"} else "python_bounded_analysis", rowCount=1,
    )


def build() -> tuple[dict, dict]:
    cases, preflight = [], []
    number = 1
    for family, paths in TARGETS:
        for path in paths:
            case_id = f"p2j4-dpo-v4-targeted-v2-{family}-{number:03d}"
            question = (
                "Use the current validated state and the declared bounded evidence boundary. "
                "Select only the necessary approved continuation, keep conclusions observational, "
                "and do not expose raw records."
            )
            cases.append({
                "schemaVersion": VERSION, "caseId": case_id, "kind": "open_exploration", "question": question,
                "expectedStatus": "COMPLETED", "requiredSources": ["java"], "allowedActions": list(path),
                "requiredActions": list(path), "forbiddenActions": FORBIDDEN, "minEvidenceBindings": 2,
                "maxActionCount": len(path), "expectedStopReason": "EVIDENCE_SUFFICIENT",
                "requiresNonDiagnostic": True, "allowedActionPaths": [list(path)],
            })
            states = []
            for prefix in range(1, len(path)):
                history = list(path[:prefix])
                context = ScientificPlannerContext(
                    questionSummary=question, intent="scientific_exploration", approvedActions=list(path),
                    remainingActionBudget=len(path) - prefix,
                    observations=[_obs(i, action) for i, action in enumerate(history, 1)], schemaCatalog=None,
                )
                semantic = semantic_scientific_next_action(context)
                states.append({"history": history, "semanticAction": semantic.actionName if semantic else None})
            preflight.append({"caseId": case_id, "targetAction": family, "path": path, "states": states})
            number += 1
    return (
        {"schemaVersion": VERSION, "name": "DPO v4 targeted weak-action state paths v2", "purpose": "targeted_state_difference_only", "caseCount": len(cases), "trainingStarted": False, "reviewStatus": "SEMANTIC_PREFLIGHT_REQUIRED", "cases": cases},
        {"schemaVersion": VERSION + "-preflight", "caseCount": len(cases), "allNonInitialStatesSemanticNone": all(s["semanticAction"] is None for c in preflight for s in c["states"]), "cases": preflight},
    )


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--task-output", type=Path, required=True); parser.add_argument("--preflight-output", type=Path, required=True); args = parser.parse_args()
    tasks, audit = build(); args.task_output.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); args.preflight_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"caseCount": tasks["caseCount"], "semanticNone": audit["allNonInitialStatesSemanticNone"]}, ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
