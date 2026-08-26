"""Build the policy-sensitive Runtime Eval set.

The questions deliberately avoid the semantic helper's explicit obligation
vocabulary.  After the mandatory first metadata inspection, the next action
must therefore be selected by the configured Decision planner.  This builder
also performs a local semantic preflight for every non-empty history state.
It does not open a socket or call a model.
"""

from __future__ import annotations

import json
from pathlib import Path

from mico_agent_runtime.contracts.research import ScientificPlannerContext
from mico_agent_runtime.contracts.trace_eval import EvalTask
from mico_agent_runtime.contracts.research import ScientificObservationSummary
from mico_agent_runtime.ports.scientific_planner import semantic_scientific_next_action


ROOT = Path(__file__).resolve().parent
VERSION = "p2j4-policy-sensitive-runtime-v1"
TASK_OUTPUT = ROOT / "p2j4-policy-sensitive-runtime-v1" / "task-set.json"
PREFLIGHT_OUTPUT = ROOT / "p2j4-policy-sensitive-runtime-v1" / "semantic-preflight.json"

PATHS: tuple[tuple[str, tuple[str, ...], str, tuple[str, ...], int], ...] = (
    (
        "compare",
        ("inspect_cohort", "compare_groups", "finish"),
        "After the cohort observation, characterize the group pattern before any further source review.",
        ("java",),
        1,
    ),
    (
        "analyze",
        ("inspect_cohort", "compare_groups", "analyze_projection", "finish"),
        "After the cohort observation, summarize the bounded projection before any further source review.",
        ("java",),
        1,
    ),
    (
        "retrieve",
        ("inspect_cohort", "compare_groups", "retrieve_evidence", "finish"),
        "After the group pattern has been summarized, perform a bounded source review before stopping.",
        ("java", "vector"),
        2,
    ),
    (
        "combined",
        ("inspect_cohort", "compare_groups", "analyze_projection", "retrieve_evidence", "finish"),
        "After the group pattern is summarized, complete the bounded projection and source review before stopping.",
        ("java", "vector"),
        2,
    ),
)

VARIANT_CONTEXTS = (
    "The metadata snapshot is complete and the observed groups are ready for a bounded next step.",
    "The initial read is validated and only the approved action path may be used.",
    "A compact group contrast has been returned from the first read.",
    "The first observation is available for a bounded analysis under the declared budget.",
    "The task should remain limited to the declared evidence and analysis budget.",
)


def _observation(index: int, action: str) -> ScientificObservationSummary:
    return ScientificObservationSummary(
        observationId="observation-" + f"{index:032x}",
        actionName=action,
        status="VALIDATED",
        source="java_controlled_read" if action in {"inspect_cohort", "compare_groups"}
        else "python_bounded_analysis" if action == "analyze_projection"
        else "knowledge_hybrid",
        rowCount=1,
    )


def build() -> tuple[dict[str, object], dict[str, object]]:
    cases: list[dict[str, object]] = []
    preflight: list[dict[str, object]] = []
    for family, path, question, sources, min_bindings in PATHS:
        for variant, context in enumerate(VARIANT_CONTEXTS, 1):
            case_id = f"p2j4-policy-sensitive-{family}-{variant:02d}"
            allowed = list(path)
            case = {
                "schemaVersion": VERSION,
                "caseId": case_id,
                "kind": "open_exploration",
                "question": f"{context} {question}",
                "expectedStatus": "COMPLETED",
                "requiredSources": list(sources),
                "allowedActions": allowed,
                "requiredActions": allowed,
                "forbiddenActions": ["direct_mysql", "unbounded_export", "causal_claim", "diagnosis"],
                "minEvidenceBindings": min_bindings,
                "maxActionCount": 7,
                "expectedStopReason": "EVIDENCE_SUFFICIENT",
                "requiresNonDiagnostic": True,
                "allowedActionPaths": [allowed],
            }
            EvalTask.model_validate(case)
            cases.append(case)

            states: list[dict[str, object]] = []
            for prefix_length in range(1, len(path)):
                history = list(path[:prefix_length])
                context = ScientificPlannerContext(
                    questionSummary=case["question"],
                    intent="scientific_exploration",
                    approvedActions=allowed,
                    remainingActionBudget=7 - prefix_length,
                    observations=[_observation(i, action) for i, action in enumerate(history, 1)],
                    schemaCatalog=None,
                )
                action = semantic_scientific_next_action(context)
                states.append({
                    "history": history,
                    "semanticAction": action.actionName if action else None,
                })
            preflight.append({"caseId": case_id, "states": states})

    task_payload = {
        "schemaVersion": VERSION,
        "name": "Mico Policy-Sensitive Runtime Eval v1",
        "reviewStatus": "STRUCTURE_READY_SEMANTIC_PREFLIGHT",
        "caseCount": len(cases),
        "kindDistribution": {"open_exploration": len(cases)},
        "families": {family: 5 for family, *_ in PATHS},
        "semanticHelperContract": "None after every non-empty history state",
        "cases": cases,
    }
    preflight_payload = {
        "schemaVersion": "p2j4-policy-sensitive-semantic-preflight-v1",
        "taskSet": VERSION,
        "caseCount": len(preflight),
        "allNonEmptyStatesSemanticNone": all(
            state["semanticAction"] is None
            for case in preflight
            for state in case["states"]
        ),
        "cases": preflight,
    }
    return task_payload, preflight_payload


def main() -> int:
    task_payload, preflight_payload = build()
    TASK_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    TASK_OUTPUT.write_text(json.dumps(task_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PREFLIGHT_OUTPUT.write_text(json.dumps(preflight_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "READY",
        "taskSet": str(TASK_OUTPUT),
        "preflight": str(PREFLIGHT_OUTPUT),
        "caseCount": task_payload["caseCount"],
        "allNonEmptyStatesSemanticNone": preflight_payload["allNonEmptyStatesSemanticNone"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
