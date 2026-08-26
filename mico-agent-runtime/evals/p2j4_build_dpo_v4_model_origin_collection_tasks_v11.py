"""Build narrow, model-origin weak-action paths after v9/v10 audits."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from mico_agent_runtime.contracts.research import ScientificObservationSummary, ScientificPlannerContext
from mico_agent_runtime.ports.scientific_planner import semantic_scientific_next_action

VERSION = "p2j4-dpo-v4-targeted-weak-actions-v2"
TARGETS = [
    ("project", "cross_project_validate", "Validate stability across the approved partitions"),
    ("disease", "cross_disease_validate", "Validate specificity across approved disease partitions"),
    ("adjust", "adjust_confounders", "Continue the bounded interpretation before summarizing the approved observation"),
]

def build() -> tuple[dict, dict]:
    cases, audit_cases = [], []
    for family, target, statement in TARGETS:
        # Cross validation is valid only after two observations exist.  The
        # inspect + compare prefix supplies exactly that Runtime contract;
        # adjustment can operate over the initial validated observation.
        actions = (
            ["inspect_cohort", "compare_groups", target, "analyze_projection", "finish"]
            if target in {"cross_project_validate", "cross_disease_validate"}
            else ["inspect_cohort", target, "analyze_projection", "finish"]
        )
        for i in range(6):
            question = f"{statement} in this bounded scientific review. Keep conclusions observational and do not expose raw records. Independent variant {i + 1}."
            case_id = f"p2j4-dpo-v4-model-origin-v11-{family}-{i + 1:02d}"
            cases.append({"schemaVersion": VERSION, "caseId": case_id, "kind": "open_exploration", "question": question, "expectedStatus": "COMPLETED", "requiredSources": ["java"], "allowedActions": actions, "requiredActions": actions, "forbiddenActions": ["direct_mysql", "unbounded_export", "diagnosis", "causal_claim"], "minEvidenceBindings": 2, "maxActionCount": len(actions), "expectedStopReason": "EVIDENCE_SUFFICIENT", "requiresNonDiagnostic": True, "allowedActionPaths": [actions]})
            states = []
            for n in range(1, len(actions)):
                history = actions[:n]
                ctx = ScientificPlannerContext(questionSummary=question, intent="scientific_exploration", approvedActions=actions, remainingActionBudget=len(actions) - n, observations=[ScientificObservationSummary(observationId=f"observation-{j:032x}", actionName=a, status="VALIDATED", source="java_controlled_read", rowCount=1) for j, a in enumerate(history)], schemaCatalog=None)
                semantic = semantic_scientific_next_action(ctx)
                states.append({"history": history, "semanticAction": semantic.actionName if semantic else None})
            audit_cases.append({"caseId": case_id, "target": family, "states": states})
    return ({"schemaVersion": VERSION, "name": "DPO v4 model-origin collection v11", "purpose": "narrow_weak_action_model_origin_collection", "caseCount": len(cases), "trainingStarted": False, "reviewStatus": "SEMANTIC_PREFLIGHT_REQUIRED", "cases": cases}, {"schemaVersion": VERSION + "-model-origin-v11-preflight", "caseCount": len(cases), "allNonInitialStatesSemanticNone": all(s["semanticAction"] is None for c in audit_cases for s in c["states"]), "cases": audit_cases})

def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--task-output", type=Path, required=True); p.add_argument("--preflight-output", type=Path, required=True); a = p.parse_args(); tasks, audit = build(); a.task_output.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); a.preflight_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); print(json.dumps({"caseCount": tasks["caseCount"], "semanticNone": audit["allNonInitialStatesSemanticNone"]}, ensure_ascii=False)); return 0
if __name__ == "__main__": raise SystemExit(main())
