"""Build targeted weak-action tasks for the next Gemini Flash Lite batch."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from mico_agent_runtime.contracts.research import ScientificObservationSummary, ScientificPlannerContext
from mico_agent_runtime.ports.scientific_planner import semantic_scientific_next_action

VERSION = "p2j4-dpo-v4-targeted-weak-actions-v2"
CASES = [
    ("read", ["inspect_cohort", "execute_read_query", "compare_groups", "finish"], "Read the next bounded observation before comparing the approved groups; keep conclusions observational and do not expose raw records."),
    ("strata", ["inspect_cohort", "compare_groups", "stratified_analysis", "analyze_projection", "finish"], "Check whether the approved group observation changes across the available strata, then summarize it observationally without exposing raw records."),
    ("project", ["inspect_cohort", "compare_groups", "cross_project_validate", "analyze_projection", "finish"], "Validate whether the approved observation is stable across projects before summarizing it; keep conclusions observational and do not expose raw records."),
    ("evidence", ["inspect_cohort", "retrieve_evidence", "analyze_projection", "finish"], "Add bounded supporting evidence after the approved observation, then summarize only what is supported; do not expose raw records."),
]

def _obs(i: int, action: str) -> ScientificObservationSummary:
    source = "java_controlled_read" if action in {"inspect_cohort", "compare_groups", "execute_read_query"} else "python_bounded_analysis"
    return ScientificObservationSummary(observationId=f"observation-{i:032x}", actionName=action, status="VALIDATED", source=source, rowCount=1)

def build() -> tuple[dict, dict]:
    cases, states = [], []
    for family, actions, base_question in CASES:
        for i in range(6):
            case_id = f"p2j4-dpo-v4-model-origin-v10-{family}-{i+1:02d}"
            question = f"{base_question} This is independent bounded review variant {i+1}."
            cases.append({"schemaVersion": VERSION, "caseId": case_id, "kind": "open_exploration", "question": question, "expectedStatus": "COMPLETED", "requiredSources": ["java"], "allowedActions": actions, "requiredActions": actions, "forbiddenActions": ["direct_mysql", "unbounded_export", "diagnosis", "causal_claim"], "minEvidenceBindings": 2, "maxActionCount": len(actions), "expectedStopReason": "EVIDENCE_SUFFICIENT", "requiresNonDiagnostic": True, "allowedActionPaths": [actions]})
            row = []
            for n in range(1, len(actions)):
                history = actions[:n]
                ctx = ScientificPlannerContext(questionSummary=question, intent="scientific_exploration", approvedActions=actions, remainingActionBudget=len(actions)-n, observations=[_obs(j, a) for j, a in enumerate(history, 1)], schemaCatalog=None)
                semantic = semantic_scientific_next_action(ctx)
                row.append({"history": history, "semanticAction": semantic.actionName if semantic else None})
            states.append({"caseId": case_id, "target": family, "states": row})
    return ({"schemaVersion": VERSION, "name": "DPO v4 model-origin collection v10", "purpose": "targeted_weak_action_model_origin_collection", "caseCount": len(cases), "trainingStarted": False, "reviewStatus": "SEMANTIC_PREFLIGHT_REQUIRED", "cases": cases}, {"schemaVersion": VERSION+"-model-origin-v10-preflight", "caseCount": len(cases), "allNonInitialStatesSemanticNone": all(s["semanticAction"] is None for c in states for s in c["states"]), "cases": states})

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--task-output",type=Path,required=True); p.add_argument("--preflight-output",type=Path,required=True); a=p.parse_args(); tasks, audit=build(); a.task_output.write_text(json.dumps(tasks,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); a.preflight_output.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps({"caseCount":tasks["caseCount"],"semanticNone":audit["allNonInitialStatesSemanticNone"]},ensure_ascii=False)); return 0
if __name__ == "__main__": raise SystemExit(main())
