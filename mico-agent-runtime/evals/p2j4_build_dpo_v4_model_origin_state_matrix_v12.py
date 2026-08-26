"""Build a structurally diverse, model-origin collection matrix for DPO v4.

This is deliberately not a prompt-variant generator.  Each case changes the
policy candidate set and/or the intended bounded history while leaving more
than one legal next action available.  The runner oracle checks safety and
terminal completion, not a single forced action path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mico_agent_runtime.contracts.research import (
    ScientificObservationSummary,
    ScientificPlannerContext,
)
from mico_agent_runtime.governance.guardrails import evaluate_input
from mico_agent_runtime.ports.scientific_planner import semantic_scientific_next_action


SCHEMA_VERSION = "p2j4-dpo-v4-targeted-weak-actions-v2"

MATRIX = [
    (
        "adjust",
        "Assess whether approved baseline dimensions change the bounded interpretation before reporting. "
        "Keep conclusions observational and do not expose raw records.",
        ["inspect_cohort", "compare_groups", "adjust_confounders", "analyze_projection", "finish"],
        ["inspect_cohort", "compare_groups", "adjust_confounders"],
    ),
    (
        "adjust",
        "Assess whether approved baseline dimensions change the bounded interpretation before reporting. "
        "Keep conclusions observational and do not expose raw records.",
        ["inspect_cohort", "compare_groups", "stratified_analysis", "adjust_confounders", "analyze_projection", "finish"],
        ["inspect_cohort", "compare_groups", "stratified_analysis", "adjust_confounders"],
    ),
    (
        "adjust",
        "Assess whether approved baseline dimensions change the bounded interpretation before reporting. "
        "Keep conclusions observational and do not expose raw records.",
        ["inspect_cohort", "execute_read_query", "compare_groups", "adjust_confounders", "analyze_projection", "finish"],
        ["inspect_cohort", "execute_read_query", "adjust_confounders"],
    ),
    (
        "project",
        "Assess whether the approved pattern persists across the available source partitions before reporting. "
        "Keep conclusions observational and do not expose raw records.",
        ["inspect_cohort", "compare_groups", "cross_project_validate", "analyze_projection", "finish"],
        ["inspect_cohort", "compare_groups", "cross_project_validate"],
    ),
    (
        "project",
        "Assess whether the approved pattern persists across the available source partitions before reporting. "
        "Keep conclusions observational and do not expose raw records.",
        ["inspect_cohort", "compare_groups", "stratified_analysis", "cross_project_validate", "analyze_projection", "finish"],
        ["inspect_cohort", "compare_groups", "cross_project_validate"],
    ),
    (
        "project",
        "Assess whether the approved pattern persists across the available source partitions before reporting. "
        "Keep conclusions observational and do not expose raw records.",
        ["inspect_cohort", "execute_read_query", "compare_groups", "cross_project_validate", "analyze_projection", "finish"],
        ["inspect_cohort", "compare_groups", "cross_project_validate"],
    ),
    (
        "disease",
        "Assess whether the approved pattern distinguishes the available condition partitions before reporting. "
        "Keep conclusions observational and do not expose raw records.",
        ["inspect_cohort", "compare_groups", "cross_disease_validate", "analyze_projection", "finish"],
        ["inspect_cohort", "compare_groups", "cross_disease_validate"],
    ),
    (
        "disease",
        "Assess whether the approved pattern distinguishes the available condition partitions before reporting. "
        "Keep conclusions observational and do not expose raw records.",
        ["inspect_cohort", "compare_groups", "stratified_analysis", "cross_disease_validate", "analyze_projection", "finish"],
        ["inspect_cohort", "compare_groups", "cross_disease_validate"],
    ),
    (
        "disease",
        "Assess whether the approved pattern distinguishes the available condition partitions before reporting. "
        "Keep conclusions observational and do not expose raw records.",
        ["inspect_cohort", "execute_read_query", "compare_groups", "cross_disease_validate", "analyze_projection", "finish"],
        ["inspect_cohort", "compare_groups", "cross_disease_validate"],
    ),
]


def _observation(index: int, action: str) -> ScientificObservationSummary:
    source = "java_controlled_read" if action in {
        "inspect_cohort", "execute_read_query", "compare_groups",
    } else "python_bounded_analysis"
    return ScientificObservationSummary(
        observationId=f"observation-{index:032x}",
        actionName=action,
        status="VALIDATED",
        source=source,
        rowCount=1,
    )


def build() -> tuple[dict, dict]:
    cases: list[dict] = []
    audit_cases: list[dict] = []
    for index, (family, question, actions, intended_history) in enumerate(MATRIX, 1):
        case_id = f"p2j4-dpo-v4-state-matrix-v12-{family}-{index:02d}"
        cases.append({
            "schemaVersion": SCHEMA_VERSION,
            "caseId": case_id,
            "kind": "open_exploration",
            "question": question,
            "expectedStatus": "COMPLETED",
            "requiredSources": ["java"],
            "allowedActions": actions,
            # A target must occur sometime in the bounded run, but the oracle
            # intentionally does not prescribe the intermediate policy path.
            "requiredActions": ["inspect_cohort", intended_history[-1]],
            "forbiddenActions": ["direct_mysql", "unbounded_export", "diagnosis", "causal_claim"],
            "minEvidenceBindings": 2,
            "maxActionCount": min(len(actions) + 1, 8),
            "expectedStopReason": "EVIDENCE_SUFFICIENT",
            "requiresNonDiagnostic": True,
        })
        states = []
        for state_index in range(1, len(intended_history)):
            history = intended_history[:state_index]
            context = ScientificPlannerContext(
                questionSummary=question,
                intent="scientific_exploration",
                approvedActions=actions,
                remainingActionBudget=8 - state_index,
                observations=[_observation(i, action) for i, action in enumerate(history, 1)],
                schemaCatalog=None,
            )
            semantic = semantic_scientific_next_action(context)
            states.append({
                "history": history,
                "candidateWidth": len(actions),
                "semanticAction": semantic.actionName if semantic else None,
            })
        audit_cases.append({"caseId": case_id, "family": family, "states": states})
    return (
        {
            "schemaVersion": SCHEMA_VERSION,
            "name": "DPO v4 model-origin state matrix v12",
            "collectionDesign": "structural_state_matrix",
            "purpose": "weak_action_model_origin_collection_without_forced_action_paths",
            "caseCount": len(cases),
            "trainingStarted": False,
            "reviewStatus": "SEMANTIC_PREFLIGHT_REQUIRED",
            "cases": cases,
        },
        {
            "schemaVersion": SCHEMA_VERSION + "-state-matrix-v12-preflight",
            "caseCount": len(cases),
            "allNonInitialStatesSemanticNone": all(
                state["semanticAction"] is None
                for case in audit_cases for state in case["states"]
            ),
            "allInputsAllowed": all(
                evaluate_input(case["question"]).verdict == "ALLOW" for case in cases
            ),
            "cases": audit_cases,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-output", type=Path, required=True)
    parser.add_argument("--preflight-output", type=Path, required=True)
    args = parser.parse_args()
    tasks, audit = build()
    args.task_output.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.preflight_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"caseCount": tasks["caseCount"], "semanticNone": audit["allNonInitialStatesSemanticNone"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
