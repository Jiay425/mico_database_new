"""Build non-path-constrained, real-runtime DPO-v4 policy-source tasks.

The task only defines the scientific objective, approved capability boundary,
and completion/safety contract.  It never supplies a required analytic route
or an allowed action path.  A later trace qualifies as a real-policy source
only if its *decision-level* provenance proves model raw-action ownership.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mico_agent_runtime.contracts.research import ScientificObservationSummary, ScientificPlannerContext
from mico_agent_runtime.ports.scientific_planner import semantic_scientific_next_action


VERSION = "p2j4-dpo-v4-policy-source-v1"
REAL_CLASS = "real_policy_source"
FORBIDDEN = ["direct_mysql", "unbounded_export", "causal_claim", "diagnosis"]

# These are naturalistic scientific objectives, not oracle paths.  The wording
# deliberately avoids semantic-helper trigger vocabulary; the local preflight
# below proves that all post-initial states remain model-planned.
FAMILIES = (
    ("group_pattern", "Compare the approved study groups and determine which bounded checks are still needed before an observational summary.",
     ["inspect_cohort", "compare_groups", "analyze_projection", "stratified_analysis", "adjust_confounders", "retrieve_evidence", "finish"]),
    ("partition_stability", "Assess whether the bounded study pattern remains stable across independently curated study partitions before an observational summary.",
     ["inspect_cohort", "compare_groups", "cross_project_validate", "analyze_projection", "retrieve_evidence", "finish"]),
    ("condition_specificity", "Assess whether the bounded study pattern is limited to the focal condition when contrasted with approved comparison conditions.",
     ["inspect_cohort", "compare_groups", "cross_disease_validate", "analyze_projection", "retrieve_evidence", "finish"]),
    ("contextual_adjustment", "Check whether approved recorded context dimensions change the bounded group pattern before any observational summary.",
     ["inspect_cohort", "compare_groups", "adjust_confounders", "stratified_analysis", "analyze_projection", "retrieve_evidence", "finish"]),
    ("subgroup_resolution", "After an approved group-level observation, determine whether a bounded subgroup view is needed before reporting the observation.",
     ["inspect_cohort", "compare_groups", "stratified_analysis", "analyze_projection", "retrieve_evidence", "finish"]),
    ("evidence_context", "Place an approved bounded group observation in the available scientific context before an observational summary.",
     ["inspect_cohort", "compare_groups", "analyze_projection", "retrieve_evidence", "cross_project_validate", "finish"]),
    ("read_then_compare", "Characterize the approved cohort scope and then determine the necessary bounded comparison work before an observational summary.",
     ["inspect_cohort", "execute_read_query", "compare_groups", "analyze_projection", "retrieve_evidence", "finish"]),
    ("conservative_stop", "Complete only the bounded checks justified by the approved observation and stop once an observational result is adequately supported.",
     ["inspect_cohort", "compare_groups", "analyze_projection", "retrieve_evidence", "finish"]),
)
VARIANTS = (
    "Keep conclusions observational and do not expose raw records.",
    "Use only the declared bounded tools and preserve the approved evidence boundary.",
    "Do not repeat completed work or infer individual-level conclusions.",
    "Use only de-identified aggregate observations in the final result.",
    "Do not make a causal claim or provide a diagnostic interpretation.",
    "Choose only actions justified by the current validated observations.",
    "Stop when further work would not add an approved evidentiary check.",
    "Keep the result concise, non-diagnostic, and evidence-preserving.",
    "Use the available approved context without exposing source records.",
    "Do not expand the task beyond the bounded scientific objective.",
)


def _observation(index: int, action: str) -> ScientificObservationSummary:
    return ScientificObservationSummary(
        observationId=f"observation-{index:032x}",
        actionName=action,
        status="VALIDATED",
        source=("knowledge_hybrid" if action == "retrieve_evidence" else
                "python_bounded_analysis" if action not in {"inspect_cohort", "execute_read_query", "compare_groups"}
                else "java_controlled_read"),
        rowCount=1,
    )


def build() -> tuple[dict, dict]:
    cases: list[dict] = []
    preflight: list[dict] = []
    for family, objective, actions in FAMILIES:
        for number, boundary in enumerate(VARIANTS, start=1):
            case_id = f"p2j4-dpo-v4-real-policy-v15-{family}-{number:02d}"
            question = f"{objective} {boundary}"
            # There is no required analytic action beyond the universal entry
            # and completion boundary.  In particular, no required action
            # equals the allowed action list and no path is supplied.
            required = ["inspect_cohort", "finish"]
            cases.append({
                "schemaVersion": VERSION,
                "caseId": case_id,
                "family": family,
                "collectionClass": REAL_CLASS,
                "kind": "open_exploration",
                "question": question,
                "expectedStatus": "COMPLETED",
                "requiredSources": ["java"],
                "allowedActions": actions,
                "requiredActions": required,
                "forbiddenActions": FORBIDDEN,
                "minEvidenceBindings": 2,
                "maxActionCount": min(8, len(actions)),
                "expectedStopReason": None,
                "requiresNonDiagnostic": True,
                "allowedActionPaths": [],
            })
            states = []
            for prefix in range(1, len(actions)):
                history = actions[:prefix]
                context = ScientificPlannerContext(
                    questionSummary=question,
                    intent="scientific_exploration",
                    approvedActions=actions,
                    remainingActionBudget=len(actions) - prefix,
                    observations=[_observation(index, action) for index, action in enumerate(history, start=1)],
                    schemaCatalog=None,
                )
                semantic = semantic_scientific_next_action(context)
                states.append({"history": history, "semanticAction": semantic.actionName if semantic else None})
            preflight.append({"caseId": case_id, "family": family, "states": states})
    task_set = {
        "schemaVersion": VERSION,
        "name": "Mico DPO v4 real policy-source collection v15",
        "purpose": "real_runtime_non_path_constrained_model_policy_collection",
        "collectionClass": REAL_CLASS,
        "caseCount": len(cases),
        "reviewStatus": "SEMANTIC_PREFLIGHT_REQUIRED",
        "trainingStarted": False,
        "cases": cases,
    }
    audit = {
        "schemaVersion": "p2j4-dpo-v4-real-policy-source-v15-preflight",
        "status": "PASS" if all(item["semanticAction"] is None for case in preflight for item in case["states"]) else "FAIL",
        "caseCount": len(cases),
        "semanticNoneForAllPostInitialStates": all(item["semanticAction"] is None for case in preflight for item in case["states"]),
        "noFixedActionPaths": all(not case["allowedActionPaths"] for case in cases),
        "requiredActionsStrictSubset": all(set(case["requiredActions"]) < set(case["allowedActions"]) for case in cases),
        "candidateWidth4to7": all(4 <= len(case["allowedActions"]) <= 7 for case in cases),
        "cases": preflight,
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
    print(json.dumps({key: audit[key] for key in ("status", "caseCount", "semanticNoneForAllPostInitialStates", "noFixedActionPaths", "requiredActionsStrictSubset", "candidateWidth4to7")}, ensure_ascii=False))
    return 0 if audit["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
