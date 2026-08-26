"""Build provenance-focused model-origin collection tasks with varied action sets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mico_agent_runtime.contracts.research import ScientificObservationSummary, ScientificPlannerContext
from mico_agent_runtime.ports.scientific_planner import semantic_scientific_next_action


VERSION = "p2j4-dpo-v4-targeted-weak-actions-v2"
FORBIDDEN = ["direct_mysql", "unbounded_export", "diagnosis", "causal_claim"]
BASES = {
    "execute": ["inspect_cohort", "execute_read_query", "finish"],
    "cross_disease": ["inspect_cohort", "compare_groups", "cross_disease_validate", "finish"],
    "stratified": ["inspect_cohort", "compare_groups", "stratified_analysis", "finish"],
}
EXTRAS = [
    (),
    ("analyze_projection",),
    ("retrieve_evidence",),
    ("adjust_confounders",),
    ("cross_project_validate",),
    ("analyze_projection", "retrieve_evidence"),
    ("adjust_confounders", "analyze_projection"),
    ("cross_project_validate", "retrieve_evidence"),
    ("stratified_analysis", "analyze_projection"),
    ("cross_disease_validate", "analyze_projection"),
]


def _obs(i: int, action: str) -> ScientificObservationSummary:
    return ScientificObservationSummary(
        observationId=f"observation-{i:032x}", actionName=action, status="VALIDATED",
        source="java_controlled_read" if action in {"inspect_cohort", "compare_groups"} else "python_bounded_analysis", rowCount=1,
    )


def build() -> tuple[dict, dict]:
    cases, preflight = [], []
    n = 1
    for family, base in BASES.items():
        for extra in EXTRAS:
            allowed = list(dict.fromkeys([*base, *extra]))
            case_id = f"p2j4-dpo-v4-model-origin-v3-{family.replace('_', '-')}-{n:03d}"
            question = (
                "Use the current validated state and the approved action set. "
                "Choose the next necessary bounded operation, preserve observational conclusions, "
                "and stop after the evidence obligation is satisfied without exposing raw records."
            )
            cases.append({
                "schemaVersion": VERSION, "caseId": case_id, "kind": "open_exploration",
                "question": question, "expectedStatus": "COMPLETED", "requiredSources": ["java"],
                "allowedActions": allowed, "requiredActions": list(base), "forbiddenActions": FORBIDDEN,
                "minEvidenceBindings": 2, "maxActionCount": len(base), "expectedStopReason": "EVIDENCE_SUFFICIENT",
                "requiresNonDiagnostic": True, "allowedActionPaths": [list(base)],
            })
            states = []
            for prefix in range(1, len(base)):
                history = list(base[:prefix])
                context = ScientificPlannerContext(
                    questionSummary=question, intent="scientific_exploration", approvedActions=allowed,
                    remainingActionBudget=len(base) - prefix,
                    observations=[_obs(i, action) for i, action in enumerate(history, 1)], schemaCatalog=None,
                )
                semantic = semantic_scientific_next_action(context)
                states.append({"history": history, "semanticAction": semantic.actionName if semantic else None})
            preflight.append({"caseId": case_id, "target": family, "allowedActions": allowed, "states": states})
            n += 1
    return (
        {"schemaVersion": VERSION, "name": "DPO v4 model-origin collection v3", "purpose": "provenance_complete_model_origin_collection", "caseCount": len(cases), "trainingStarted": False, "reviewStatus": "SEMANTIC_PREFLIGHT_REQUIRED", "cases": cases},
        {"schemaVersion": VERSION + "-model-origin-v3-preflight", "caseCount": len(cases), "allNonInitialStatesSemanticNone": all(s["semanticAction"] is None for c in preflight for s in c["states"]), "cases": preflight},
    )


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--task-output", type=Path, required=True); parser.add_argument("--preflight-output", type=Path, required=True); args = parser.parse_args()
    tasks, audit = build(); args.task_output.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); args.preflight_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); print(json.dumps({"caseCount": tasks["caseCount"], "semanticNone": audit["allNonInitialStatesSemanticNone"]}, ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
