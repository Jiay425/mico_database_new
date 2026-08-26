"""Build the next structurally distinct DPO v4 model-origin collection plan.

The matrix varies action menus and bounded prior histories.  It never relies
on case identifiers or wording variants to claim a new policy state.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

from mico_agent_runtime.contracts.research import ScientificObservationSummary, ScientificPlannerContext
from mico_agent_runtime.governance.guardrails import evaluate_input
from mico_agent_runtime.ports.decision_policy import build_decision_policy_state
from mico_agent_runtime.ports.scientific_planner import semantic_scientific_next_action


SCHEMA_VERSION = "p2j4-dpo-v4-targeted-weak-actions-v2"
PROFILES = (
    ("adjust", "Assess whether approved baseline dimensions change the bounded interpretation before reporting.", "adjust_confounders"),
    ("project", "Assess whether the approved pattern persists across the available source partitions before reporting.", "cross_project_validate"),
    ("disease", "Assess whether the approved pattern distinguishes the available condition partitions before reporting.", "cross_disease_validate"),
)
OPTIONAL = (
    "compare_groups", "stratified_analysis", "adjust_confounders",
    "cross_project_validate", "cross_disease_validate", "analyze_projection", "retrieve_evidence",
)


def _observation(index: int, action: str) -> ScientificObservationSummary:
    return ScientificObservationSummary(
        observationId=f"observation-{index:032x}",
        actionName=action,
        status="VALIDATED",
        source="java_controlled_read" if action in {"inspect_cohort", "execute_read_query", "compare_groups"} else "python_bounded_analysis",
        rowCount=1,
    )


def _actions(target: str, offset: int) -> list[str]:
    # Four through seven candidates, all action names are part of the actual
    # policy input.  Target is required eventually, never as a forced next step.
    pool = [action for action in OPTIONAL if action != target]
    menus = [menu for size in range(1, 4) for menu in combinations(pool, size)]
    selected = menus[offset % len(menus)]
    # inspect + a bounded second Java observation make the two-observation
    # contract reachable, while the unique optional menu remains genuinely
    # visible to the model as alternatives (width 5--7).
    return ["inspect_cohort", "execute_read_query", *selected, target, "finish"]


def _history(target: str, actions: list[str], offset: int) -> list[str]:
    optional = next(
        action for action in actions
        if action not in {"inspect_cohort", "execute_read_query", target, "finish"}
    )
    return ["inspect_cohort", "execute_read_query", optional, target]


def _signature(state: dict) -> str:
    return json.dumps({key: state[key] for key in (
        "task_kind", "goal_code", "observation_flags", "history_actions", "candidate_actions", "state_summary",
    )}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build() -> tuple[dict, dict]:
    cases: list[dict] = []
    audits: list[dict] = []
    planned_states: set[str] = set()
    for profile_index, (family, question, target) in enumerate(PROFILES):
        for variant in range(20):
            actions = _actions(target, profile_index * 20 + variant)
            history = _history(target, actions, variant)
            case_id = f"p2j4-dpo-v4-state-matrix-v13-{family}-{variant + 1:02d}"
            cases.append({
                "schemaVersion": SCHEMA_VERSION,
                "caseId": case_id,
                "kind": "open_exploration",
                "question": question + " Keep conclusions observational and do not expose raw records.",
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
            state_audit = []
            for state_index in range(1, len(history)):
                prefix = history[:state_index]
                context = ScientificPlannerContext(
                    questionSummary=question,
                    intent="scientific_exploration",
                    approvedActions=actions,
                    remainingActionBudget=8 - state_index,
                    observations=[_observation(i, action) for i, action in enumerate(prefix, 1)],
                    schemaCatalog=None,
                )
                state = build_decision_policy_state(context)
                planned_states.add(_signature(state))
                semantic = semantic_scientific_next_action(context)
                state_audit.append({
                    "history": prefix,
                    "candidateWidth": len(actions),
                    "semanticAction": semantic.actionName if semantic else None,
                })
            audits.append({"caseId": case_id, "family": family, "states": state_audit})
    return (
        {
            "schemaVersion": SCHEMA_VERSION,
            "name": "DPO v4 model-origin state matrix v13",
            "collectionDesign": "structural_state_matrix",
            "caseCount": len(cases),
            "trainingStarted": False,
            "reviewStatus": "SEMANTIC_SCOPE_AND_STATE_PREFLIGHT_REQUIRED",
            "cases": cases,
        },
        {
            "schemaVersion": SCHEMA_VERSION + "-state-matrix-v13-preflight",
            "caseCount": len(cases),
            "plannedUniquePolicyStateCount": len(planned_states),
            "allInputsAllowed": all(evaluate_input(case["question"]).verdict == "ALLOW" for case in cases),
            "allNonInitialStatesSemanticNone": all(
                state["semanticAction"] is None for audit in audits for state in audit["states"]
            ),
            "candidateWidthCounts": {
                str(width): sum(1 for case in cases if len(case["allowedActions"]) == width)
                for width in range(4, 8)
            },
            "cases": audits,
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
    print(json.dumps({key: audit[key] for key in (
        "caseCount", "plannedUniquePolicyStateCount", "allInputsAllowed", "allNonInitialStatesSemanticNone", "candidateWidthCounts",
    )}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
