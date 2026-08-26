"""Build model-participating (semantic-none) DPO v4 Trace source tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mico_agent_runtime.contracts.research import ScientificObservationSummary, ScientificPlannerContext
from mico_agent_runtime.ports.scientific_planner import semantic_scientific_next_action


ROOT = Path(__file__).resolve().parent
VERSION = "p2j4-dpo-v4-policy-source-v1"

# Wording intentionally contains no semantic-helper trigger vocabulary.  The
# action set and closed state, rather than key-word routing, are the policy
# input under test.
FAMILIES = (
    ("compact", ("inspect_cohort", "compare_groups", "analyze_projection", "finish")),
    ("evidence", ("inspect_cohort", "compare_groups", "retrieve_evidence", "finish")),
    ("strata", ("inspect_cohort", "compare_groups", "stratified_analysis", "analyze_projection", "finish")),
    ("adjust", ("inspect_cohort", "compare_groups", "adjust_confounders", "analyze_projection", "finish")),
    ("replication", ("inspect_cohort", "compare_groups", "cross_project_validate", "retrieve_evidence", "finish")),
    ("specificity", ("inspect_cohort", "compare_groups", "cross_disease_validate", "retrieve_evidence", "finish")),
)
CONTEXTS = (
    "Use only the approved scientific actions and the declared bounded budget.",
    "A validated initial observation is available for the next bounded step.",
    "Keep the work within the declared evidence and analysis scope.",
    "Choose the next necessary approved action without repeating completed work.",
    "Return an observational, non-diagnostic scientific result.",
    "Use the current validated state and do not expand the approved action set.",
    "Select a bounded continuation that preserves the declared evidence boundary.",
    "Keep conclusions observational and do not expose raw records.",
    "The next step must be justified by the available approved actions.",
    "Complete only the necessary bounded work before stopping.",
)


def _observation(index: int, action: str) -> ScientificObservationSummary:
    return ScientificObservationSummary(
        observationId=f"observation-{index:032x}", actionName=action,
        status="VALIDATED",
        source="java_controlled_read" if action in {"inspect_cohort", "compare_groups"}
        else "python_bounded_analysis" if action != "retrieve_evidence" else "knowledge_hybrid",
        rowCount=1,
    )


def build() -> tuple[dict, dict]:
    cases, preflight = [], []
    number = 1
    for family, path in FAMILIES:
        for context in CONTEXTS:
            case_id = f"p2j4-dpo-v4-policy-{family}-{number:03d}"
            case = {
                "schemaVersion": VERSION, "caseId": case_id, "kind": "open_exploration",
                "question": context, "expectedStatus": "COMPLETED",
                "requiredSources": ["java", "vector"] if "retrieve_evidence" in path else ["java"],
                "allowedActions": list(path), "requiredActions": ["inspect_cohort", "finish"],
                "forbiddenActions": ["direct_mysql", "unbounded_export", "causal_claim", "diagnosis"],
                "minEvidenceBindings": 2, "maxActionCount": len(path),
                "expectedStopReason": None, "requiresNonDiagnostic": True,
                "allowedActionPaths": [],
            }
            states = []
            for prefix in range(1, len(path)):
                history = list(path[:prefix])
                policy = ScientificPlannerContext(
                    questionSummary=context, intent="scientific_exploration", approvedActions=list(path),
                    remainingActionBudget=len(path) - prefix,
                    observations=[_observation(i, action) for i, action in enumerate(history, 1)],
                    schemaCatalog=None,
                )
                semantic = semantic_scientific_next_action(policy)
                states.append({"history": history, "semanticAction": semantic.actionName if semantic else None})
            cases.append(case)
            preflight.append({"caseId": case_id, "states": states})
            number += 1
    task_set = {
        "schemaVersion": VERSION, "name": "Mico DPO v4 model-policy Trace sources",
        "purpose": "development_only_model_participation_required", "caseCount": len(cases),
        "reviewStatus": "SEMANTIC_PREFLIGHT_REQUIRED", "kindDistribution": {"open_exploration": len(cases)},
        "trainingStarted": False, "cases": cases,
    }
    audit = {
        "schemaVersion": "p2j4-dpo-v4-policy-source-preflight-v1", "caseCount": len(cases),
        "allNonInitialStatesSemanticNone": all(
            state["semanticAction"] is None for item in preflight for state in item["states"]
        ), "cases": preflight,
    }
    return task_set, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-output", type=Path, required=True)
    parser.add_argument("--preflight-output", type=Path, required=True)
    args = parser.parse_args()
    tasks, audit = build()
    args.task_output.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.preflight_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"caseCount": tasks["caseCount"], "semanticNone": audit["allNonInitialStatesSemanticNone"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
