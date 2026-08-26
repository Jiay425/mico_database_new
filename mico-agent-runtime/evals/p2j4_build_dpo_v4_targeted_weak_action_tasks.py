"""Build a small, semantic-neutral task pool for weak-action provenance traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mico_agent_runtime.contracts.research import ScientificObservationSummary, ScientificPlannerContext
from mico_agent_runtime.ports.scientific_planner import semantic_scientific_next_action


VERSION = "p2j4-dpo-v4-targeted-weak-actions-v1"
FORBIDDEN = ["direct_mysql", "unbounded_export", "diagnosis", "causal_claim"]
SPECIFICATIONS = (
    ("execute", ("inspect_cohort", "execute_read_query", "finish"), "approved bounded read shape"),
    ("cross-disease", ("inspect_cohort", "compare_groups", "cross_disease_validate", "analyze_projection", "finish"), "approved cohort partition specificity"),
    ("stratified", ("inspect_cohort", "compare_groups", "stratified_analysis", "analyze_projection", "finish"), "approved subgroup resolution"),
)
VARIANTS = (
    "Use the current validated state and preserve the approved evidence boundary.",
    "Continue only with the next necessary bounded operation; do not repeat completed work.",
    "Keep the result observational and do not expose raw records or identifiers.",
    "Use the declared action path and stop when the validated result is sufficient.",
    "The next operation must be supported by the current approved observations.",
    "Do not expand the action set or invent an unapproved query shape.",
    "Prefer an evidence-backed continuation over a premature summary.",
    "Complete the bounded state transition before producing the final result.",
)


def _observation(index: int, action: str) -> ScientificObservationSummary:
    return ScientificObservationSummary(
        observationId=f"observation-{index:032x}",
        actionName=action,
        status="VALIDATED",
        source="java_controlled_read" if action in {"inspect_cohort", "compare_groups"}
        else "python_bounded_analysis",
        rowCount=1,
    )


def build() -> tuple[dict, dict]:
    cases: list[dict] = []
    preflight: list[dict] = []
    number = 1
    for family, path, state_label in SPECIFICATIONS:
        for variant in VARIANTS:
            case_id = f"p2j4-dpo-v4-targeted-{family}-{number:03d}"
            question = (
                f"State-difference target variant {variant}: {state_label} is the next bounded obligation. "
                "Use only approved actions, preserve observational conclusions, and do not expose raw records."
            )
            case = {
                "schemaVersion": VERSION,
                "caseId": case_id,
                "kind": "open_exploration",
                "question": question,
                "expectedStatus": "COMPLETED",
                "requiredSources": ["java"],
                "allowedActions": list(path),
                "requiredActions": list(path),
                "forbiddenActions": FORBIDDEN,
                "minEvidenceBindings": 2,
                "maxActionCount": len(path),
                "expectedStopReason": "EVIDENCE_SUFFICIENT",
                "requiresNonDiagnostic": True,
                "allowedActionPaths": [list(path)],
            }
            cases.append(case)
            states = []
            for prefix in range(1, len(path)):
                history = list(path[:prefix])
                context = ScientificPlannerContext(
                    questionSummary=question,
                    intent="scientific_exploration",
                    approvedActions=list(path),
                    remainingActionBudget=len(path) - prefix,
                    observations=[_observation(i, action) for i, action in enumerate(history, 1)],
                    schemaCatalog=None,
                )
                semantic = semantic_scientific_next_action(context)
                states.append({"history": history, "semanticAction": semantic.actionName if semantic else None})
            preflight.append({"caseId": case_id, "targetAction": family, "states": states})
            number += 1
    result = {
        "schemaVersion": VERSION,
        "name": "DPO v4 targeted weak-action provenance tasks",
        "purpose": "targeted_training_only_provenance_complete_trace_collection",
        "caseCount": len(cases),
        "trainingStarted": False,
        "reviewStatus": "SEMANTIC_PREFLIGHT_REQUIRED",
        "sourceClass": "targeted_real_state_difference_candidate",
        "cases": cases,
    }
    audit = {
        "schemaVersion": VERSION + "-preflight",
        "caseCount": len(cases),
        "targetCounts": {family: len(VARIANTS) for family, _, _ in SPECIFICATIONS},
        "allNonInitialStatesSemanticNone": all(
            state["semanticAction"] is None for item in preflight for state in item["states"]
        ),
        "cases": preflight,
    }
    return result, audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-output", type=Path, required=True)
    parser.add_argument("--preflight-output", type=Path, required=True)
    args = parser.parse_args()
    tasks, audit = build()
    args.task_output.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.preflight_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"caseCount": tasks["caseCount"], "targetCounts": audit["targetCounts"], "semanticNone": audit["allNonInitialStatesSemanticNone"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
