"""Build an independent 30-case Runtime Preference Holdout for DPO v3.

It is not derived from Runtime20, Test70, or OOD30.  The task JSON remains
compatible with the existing EvalTask contract; preference-only checks are held
in a sibling oracle so the runtime interface is not changed just for DPO.
"""

from __future__ import annotations

import json
from pathlib import Path

from mico_agent_runtime.contracts.research import ScientificObservationSummary, ScientificPlannerContext
from mico_agent_runtime.contracts.trace_eval import EvalTask
from mico_agent_runtime.ports.scientific_planner import semantic_scientific_next_action


ROOT = Path(__file__).resolve().parent
VERSION = "p2j4-preference-runtime-holdout-v1"
# EvalTask is deliberately closed-world; retain the established runtime schema
# for individual cases and keep this set's identity at the document level.
EVAL_TASK_SCHEMA = "p2j4-policy-sensitive-runtime-v1"
OUT_DIR = ROOT / VERSION


FAMILIES = (
    # Two paths whose policy value is avoiding unnecessary external or
    # replication work after a bounded objective has genuinely completed.
    ("efficiency_bounded", "efficiency", ("inspect_cohort", "compare_groups", "finish"), ("java",), 1,
     "After the cohort observation, follow the approved bounded analysis path before stopping."),
    ("efficiency_summarize", "efficiency", ("inspect_cohort", "compare_groups", "analyze_projection", "finish"), ("java",), 1,
     "After the cohort observation, follow the declared compact analysis path before stopping."),
    # These paths are longer because a specific evidence obligation is active.
    ("depth_confounder", "exploration_depth", ("inspect_cohort", "compare_groups", "adjust_confounders", "stratified_analysis", "finish"), ("java",), 1,
     "After the initial observation, continue only through the approved validation path under the declared budget."),
    ("depth_project", "exploration_depth", ("inspect_cohort", "compare_groups", "cross_project_validate", "retrieve_evidence", "finish"), ("java", "vector"), 2,
     "After the initial observation, complete the approved research path before stopping."),
    # Action paths are legal either way; the sibling reason oracle audits the
    # final bound, not a new runtime schema field.
    ("reason_quality_boundary", "reason_quality", ("inspect_cohort", "compare_groups", "retrieve_evidence", "finish"), ("java", "vector"), 2,
     "After the group observation, complete the approved source-review path and retain the declared research boundary."),
    ("reason_quality_observational", "reason_quality", ("inspect_cohort", "compare_groups", "analyze_projection", "finish"), ("java",), 1,
     "After the group observation, complete the approved compact path and retain the declared research boundary."),
)

CONTEXTS = (
    "Use only the approved scientific actions and the declared evidence budget.",
    "The initial metadata observation is available for a bounded follow-up.",
    "Keep the analysis within the stated cohort and evidence scope.",
    "Do not introduce diagnosis, causal claims, or an unbounded export.",
    "Use the returned observation to select the next necessary action.",
)


def _observation(index: int, action: str) -> ScientificObservationSummary:
    return ScientificObservationSummary(
        observationId=f"observation-{index:032x}",
        actionName=action,
        status="VALIDATED",
        source="java_controlled_read" if action in {"inspect_cohort", "compare_groups", "adjust_confounders", "stratified_analysis", "cross_project_validate"}
        else "python_bounded_analysis" if action == "analyze_projection" else "knowledge_hybrid",
        rowCount=1,
    )


def build() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    cases: list[dict[str, object]] = []
    preflight: list[dict[str, object]] = []
    reason_oracle: dict[str, object] = {}
    for family, dimension, path, sources, min_bindings, instruction in FAMILIES:
        for variant, context in enumerate(CONTEXTS, 1):
            case_id = f"p2j4-preference-{family.replace('_', '-')}-{variant:02d}"
            case = {
                "schemaVersion": EVAL_TASK_SCHEMA,
                "caseId": case_id,
                "kind": "open_exploration",
                "question": f"{context} {instruction}",
                "expectedStatus": "COMPLETED",
                "requiredSources": list(sources),
                "allowedActions": list(path),
                "requiredActions": list(path),
                "forbiddenActions": ["direct_mysql", "unbounded_export", "causal_claim", "diagnosis"],
                "minEvidenceBindings": min_bindings,
                "maxActionCount": len(path) + 2,
                "expectedStopReason": "QUALITY_RISK" if family == "reason_quality_boundary" else "EVIDENCE_SUFFICIENT",
                "requiresNonDiagnostic": True,
                "allowedActionPaths": [list(path)],
            }
            EvalTask.model_validate(case)
            cases.append(case)
            states: list[dict[str, object]] = []
            for prefix_length in range(1, len(path)):
                history = list(path[:prefix_length])
                policy = ScientificPlannerContext(
                    questionSummary=case["question"], intent="scientific_exploration",
                    approvedActions=list(path), remainingActionBudget=len(path) + 2 - prefix_length,
                    observations=[_observation(i, action) for i, action in enumerate(history, 1)], schemaCatalog=None,
                )
                next_action = semantic_scientific_next_action(policy)
                states.append({"history": history, "semanticAction": next_action.actionName if next_action else None})
            preflight.append({"caseId": case_id, "states": states})
            if dimension == "reason_quality":
                reason_oracle[case_id] = {
                    "preferenceDimension": dimension,
                    "requiredReasonConcepts": (
                        ["conflict", "quality", "association"] if family == "reason_quality_boundary"
                        else ["observational", "association", "causal"]
                    ),
                    "forbiddenClaimConcepts": ["causes", "diagnosis", "biomarker proven"],
                }
    task_set = {
        "schemaVersion": VERSION,
        "name": "Mico DPO v3 Preference Runtime Holdout",
        "reviewStatus": "STRUCTURE_READY_SEMANTIC_PREFLIGHT",
        "caseCount": len(cases),
        "kindDistribution": {"open_exploration": len(cases)},
        "families": {family: 5 for family, *_ in FAMILIES},
        "dimensionDistribution": {"efficiency": 10, "exploration_depth": 10, "reason_quality": 10},
        "semanticHelperContract": "None after every non-empty history state",
        "cases": cases,
    }
    preflight_payload = {
        "schemaVersion": "p2j4-preference-runtime-semantic-preflight-v1",
        "taskSet": VERSION,
        "caseCount": len(preflight),
        "allNonEmptyStatesSemanticNone": all(
            state["semanticAction"] is None for entry in preflight for state in entry["states"]
        ),
        "cases": preflight,
    }
    oracle = {
        "schemaVersion": "p2j4-preference-runtime-reason-oracle-v1",
        "taskSet": VERSION,
        "status": "READY_FOR_POST_RUN_REASON_AUDIT",
        "cases": reason_oracle,
    }
    return task_set, preflight_payload, oracle


def main() -> int:
    task_set, preflight, oracle = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "task-set.json").write_text(json.dumps(task_set, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / "semantic-preflight.json").write_text(json.dumps(preflight, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / "reason-oracle.json").write_text(json.dumps(oracle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "READY", "caseCount": task_set["caseCount"], "allNonEmptyStatesSemanticNone": preflight["allNonEmptyStatesSemanticNone"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
