"""Build model-origin tasks whose required paths cover underrepresented actions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mico_agent_runtime.contracts.research import ScientificObservationSummary, ScientificPlannerContext
from mico_agent_runtime.ports.scientific_planner import semantic_scientific_next_action


VERSION = "p2j4-dpo-v4-targeted-weak-actions-v2"
BASES = {
    "evidence": ["inspect_cohort", "retrieve_evidence", "finish"],
    "adjust": ["inspect_cohort", "compare_groups", "adjust_confounders", "finish"],
    "projection": ["inspect_cohort", "execute_read_query", "analyze_projection", "finish"],
    "cross_project": ["inspect_cohort", "compare_groups", "cross_project_validate", "finish"],
}
EXTRAS = [
    (),
    ("analyze_projection",),
    ("retrieve_evidence",),
    ("adjust_confounders",),
    ("cross_project_validate",),
    ("stratified_analysis",),
    ("analyze_projection", "retrieve_evidence"),
    ("adjust_confounders", "cross_project_validate"),
]


def _obs(i: int, action: str) -> ScientificObservationSummary:
    source = "java_controlled_read" if action in {"inspect_cohort", "compare_groups"} else "python_bounded_analysis"
    return ScientificObservationSummary(observationId=f"observation-{i:032x}", actionName=action, status="VALIDATED", source=source, rowCount=1)


def build() -> tuple[dict, dict]:
    cases, preflight = [], []
    index = 1
    for family, base in BASES.items():
        for extra in EXTRAS:
            allowed = list(dict.fromkeys([*base, *extra]))
            case_id = f"p2j4-dpo-v4-model-origin-v7-{family.replace('_', '-')}-{index:03d}"
            question = (
                "Use the current validated scientific state and approved bounded actions. "
                "Choose the next action required by the evidence obligation, preserve observational conclusions, "
                "and stop when the bounded evidence is sufficient without exposing raw records."
            )
            cases.append({
                "schemaVersion": VERSION, "caseId": case_id, "kind": "open_exploration",
                "question": question, "expectedStatus": "COMPLETED", "requiredSources": ["java"],
                "allowedActions": allowed, "requiredActions": list(base),
                "forbiddenActions": ["direct_mysql", "unbounded_export", "diagnosis", "causal_claim"],
                "minEvidenceBindings": 2, "maxActionCount": len(base),
                "expectedStopReason": "EVIDENCE_SUFFICIENT", "requiresNonDiagnostic": True,
                "allowedActionPaths": [list(base)],
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
            preflight.append({"caseId": case_id, "target": family, "states": states})
            index += 1
    tasks = {"schemaVersion": VERSION, "name": "DPO v4 model-origin collection v7", "purpose": "underrepresented_action_model_origin_collection", "caseCount": len(cases), "trainingStarted": False, "reviewStatus": "SEMANTIC_PREFLIGHT_REQUIRED", "cases": cases}
    audit = {"schemaVersion": VERSION + "-model-origin-v7-preflight", "caseCount": len(cases), "allNonInitialStatesSemanticNone": all(s["semanticAction"] is None for c in preflight for s in c["states"]), "cases": preflight}
    return tasks, audit


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--task-output", type=Path, required=True); parser.add_argument("--preflight-output", type=Path, required=True); args = parser.parse_args()
    tasks, audit = build(); args.task_output.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); args.preflight_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); print(json.dumps({"caseCount": tasks["caseCount"], "semanticNone": audit["allNonInitialStatesSemanticNone"]}, ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
