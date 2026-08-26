"""Build a semantic-neutral v14 source pool for weak model actions.

The task wording names a scientific obligation without invoking the Runtime
semantic guard.  The runner still owns execution and the final oracle; this
file only creates new collection tasks and a local semantic preflight.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

from mico_agent_runtime.contracts.research import ScientificObservationSummary, ScientificPlannerContext
from mico_agent_runtime.governance.guardrails import evaluate_input
from mico_agent_runtime.ports.scientific_planner import semantic_scientific_next_action


VERSION = "p2j4-dpo-v4-targeted-weak-actions-v2"
OPTIONAL = (
    "execute_read_query", "compare_groups", "stratified_analysis", "analyze_projection",
    "adjust_confounders", "cross_project_validate", "cross_disease_validate", "retrieve_evidence",
)
PROFILES = (
    ("compare", "Review the approved group summaries as the next bounded operation before reporting.", "compare_groups"),
    ("stratified", "Review the approved subgroup dimensions as the next bounded operation before reporting.", "stratified_analysis"),
    ("projection", "Review the approved joint projection as the next bounded operation before reporting.", "analyze_projection"),
    ("external", "Check the approved external context after the validated observation before reporting.", "retrieve_evidence"),
)


def _obs(index: int, action: str) -> ScientificObservationSummary:
    return ScientificObservationSummary(
        observationId=f"observation-{index:032x}",
        actionName=action,
        status="VALIDATED",
        source="java_controlled_read" if action in {"inspect_cohort", "execute_read_query", "compare_groups"} else "python_bounded_analysis",
        rowCount=1,
    )


def _menus(target: str) -> list[list[str]]:
    pool = [action for action in OPTIONAL if action != target]
    menus: list[list[str]] = []
    for width in (2, 3, 4, 5):
        for selected in combinations(pool, width):
            actions = ["inspect_cohort", "execute_read_query", *selected, target, "finish"]
            if 5 <= len(actions) <= 7:
                menus.append(actions)
    return menus


def build() -> tuple[dict, dict]:
    cases: list[dict] = []
    preflight_cases: list[dict] = []
    number = 1
    for family, question, target in PROFILES:
        menus = _menus(target)
        for variant in range(15):
            actions = menus[(variant * 7 + len(target)) % len(menus)]
            # ScientificTask rejects duplicate allowedActions.  The menu is
            # assembled from a fixed prefix plus the sampled alternatives and
            # target, so de-duplicate at the generator boundary before a task
            # can reach the runner.
            actions = list(dict.fromkeys(actions))
            case_id = f"p2j4-dpo-v4-weak-v14-{family}-{variant + 1:02d}"
            full_question = question + " Keep conclusions observational and do not expose raw records."
            cases.append({
                "schemaVersion": VERSION,
                "caseId": case_id,
                "kind": "open_exploration",
                "question": full_question,
                "expectedStatus": "COMPLETED",
                "requiredSources": ["java"],
                "allowedActions": actions,
                "requiredActions": ["inspect_cohort", target],
                "forbiddenActions": ["direct_mysql", "unbounded_export", "diagnosis", "causal_claim"],
                "minEvidenceBindings": 2,
                "maxActionCount": 8,
                "expectedStopReason": "EVIDENCE_SUFFICIENT",
                "requiresNonDiagnostic": True,
            })
            states: list[dict] = []
            intended = ["inspect_cohort", "execute_read_query", target]
            for index in range(1, len(intended)):
                context = ScientificPlannerContext(
                    questionSummary=full_question,
                    intent="scientific_exploration",
                    approvedActions=actions,
                    remainingActionBudget=8 - index,
                    observations=[_obs(i, action) for i, action in enumerate(intended[:index], 1)],
                    schemaCatalog=None,
                )
                semantic = semantic_scientific_next_action(context)
                states.append({"history": intended[:index], "semanticAction": semantic.actionName if semantic else None})
            preflight_cases.append({"caseId": case_id, "family": family, "targetAction": target, "candidateWidth": len(actions), "states": states})
            number += 1
    task_set = {
        "schemaVersion": VERSION,
        "name": "DPO v4 weak model-action source tasks v14",
        "purpose": "weak_action_model_origin_collection_with_structured_decision_fields",
        "caseCount": len(cases),
        "trainingStarted": False,
        "reviewStatus": "SEMANTIC_PREFLIGHT_REQUIRED",
        "cases": cases,
    }
    audit = {
        "schemaVersion": VERSION + "-weak-v14-preflight",
        "caseCount": len(cases),
        "allInputsAllowed": all(evaluate_input(case["question"]).verdict == "ALLOW" for case in cases),
        "allNonInitialStatesSemanticNone": all(state["semanticAction"] is None for case in preflight_cases for state in case["states"]),
        "targetCounts": {family: sum(case["family"] == family for case in preflight_cases) for family, _, _ in PROFILES},
        "candidateWidthCounts": {str(width): sum(case["candidateWidth"] == width for case in preflight_cases) for width in range(4, 8)},
        "cases": preflight_cases,
    }
    return task_set, audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-output", type=Path, required=True)
    parser.add_argument("--preflight-output", type=Path, required=True)
    args = parser.parse_args()
    tasks, audit = build()
    args.task_output.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.preflight_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: audit[key] for key in ("caseCount", "allInputsAllowed", "allNonInitialStatesSemanticNone", "targetCounts", "candidateWidthCounts")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
