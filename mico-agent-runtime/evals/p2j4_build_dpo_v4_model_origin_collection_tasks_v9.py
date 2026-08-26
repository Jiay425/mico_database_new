"""Build a fresh, diverse model-origin collection set after the v8 audit.

This set is intentionally new: earlier successful runs are never rerun.  The
questions vary the redacted goal code while every case remains observational,
bounded, and free of raw scientific values.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mico_agent_runtime.contracts.research import ScientificObservationSummary, ScientificPlannerContext
from mico_agent_runtime.ports.scientific_planner import semantic_scientific_next_action


VERSION = "p2j4-dpo-v4-targeted-weak-actions-v2"
CASES = [
    ("cohort", ["inspect_cohort", "compare_groups", "analyze_projection", "finish"], "Compare the approved groups and summarize the validated observation; keep the conclusion observational and do not expose raw records."),
    ("projection", ["inspect_cohort", "execute_read_query", "analyze_projection", "retrieve_evidence", "finish"], "Summarize the validated projection, then add bounded evidence if needed; keep the conclusion observational and do not expose raw records."),
    ("comparison", ["inspect_cohort", "execute_read_query", "compare_groups", "stratified_analysis", "analyze_projection", "finish"], "Compare the approved groups and check whether the observed difference persists across the available strata; keep the conclusion observational and do not expose raw records."),
    ("stability", ["inspect_cohort", "execute_read_query", "compare_groups", "cross_project_validate", "analyze_projection", "finish"], "Assess whether the approved observation is stable across projects before summarizing it; keep the conclusion observational and do not expose raw records."),
    ("disease", ["inspect_cohort", "execute_read_query", "compare_groups", "cross_disease_validate", "retrieve_evidence", "finish"], "Assess disease specificity only after the approved comparison, then retrieve bounded evidence if required; keep the conclusion observational and do not expose raw records."),
]


def _obs(i: int, action: str) -> ScientificObservationSummary:
    source = "java_controlled_read" if action in {"inspect_cohort", "execute_read_query", "compare_groups"} else "python_bounded_analysis"
    return ScientificObservationSummary(observationId=f"observation-{i:032x}", actionName=action, status="VALIDATED", source=source, rowCount=1)


def build() -> tuple[dict, dict]:
    cases, preflight = [], []
    index = 1
    for family, actions, question in CASES:
        for variant in range(8):
            case_id = f"p2j4-dpo-v4-model-origin-v9-{family}-{variant + 1:02d}"
            case_question = f"{question} This is bounded review variant {variant + 1}; use only the current redacted state."
            cases.append({
                "schemaVersion": VERSION, "caseId": case_id, "kind": "open_exploration", "question": case_question,
                "expectedStatus": "COMPLETED", "requiredSources": ["java"], "allowedActions": actions,
                "requiredActions": actions, "forbiddenActions": ["direct_mysql", "unbounded_export", "diagnosis", "causal_claim"],
                "minEvidenceBindings": 2, "maxActionCount": len(actions), "expectedStopReason": "EVIDENCE_SUFFICIENT",
                "requiresNonDiagnostic": True, "allowedActionPaths": [actions],
            })
            states = []
            for prefix in range(1, len(actions)):
                history = list(actions[:prefix])
                context = ScientificPlannerContext(
                    questionSummary=case_question, intent="scientific_exploration", approvedActions=actions,
                    remainingActionBudget=len(actions) - prefix,
                    observations=[_obs(i, action) for i, action in enumerate(history, 1)], schemaCatalog=None,
                )
                semantic = semantic_scientific_next_action(context)
                states.append({"history": history, "semanticAction": semantic.actionName if semantic else None})
            preflight.append({"caseId": case_id, "target": family, "states": states})
            index += 1
    tasks = {"schemaVersion": VERSION, "name": "DPO v4 model-origin collection v9", "purpose": "diverse_model_origin_state_collection", "caseCount": len(cases), "trainingStarted": False, "reviewStatus": "SEMANTIC_PREFLIGHT_REQUIRED", "cases": cases}
    audit = {"schemaVersion": VERSION + "-model-origin-v9-preflight", "caseCount": len(cases), "allNonInitialStatesSemanticNone": all(s["semanticAction"] is None for c in preflight for s in c["states"]), "candidateWidthCounts": {str(len(actions)): sum(1 for family, actions, question in CASES)}, "cases": preflight}
    return tasks, audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-output", type=Path, required=True)
    parser.add_argument("--preflight-output", type=Path, required=True)
    args = parser.parse_args()
    tasks, audit = build()
    args.task_output.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.preflight_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"caseCount": tasks["caseCount"], "semanticNone": audit["allNonInitialStatesSemanticNone"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
