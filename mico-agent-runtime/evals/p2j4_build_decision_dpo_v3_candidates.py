"""Build auditable DPO v3 *candidates* for Mico's decision policy.

This is deliberately separate from DPO v1/v2.  DPO v3 does not recast
"correct action vs plainly invalid action" SFT labels as preferences.  Its
pairs express a policy preference between two schema-valid, candidate actions,
or between two rationales for the same action.

The output is a reviewable candidate pool, not a training freeze.  The paired
audit script selects a balanced 750-record staging set and emits a human-review
packet before any A100 job is permitted.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evals" / "p2j4-decision-dpo-v3-candidates-20260825.json"
BASE_RUNTIME = ROOT / "evals" / "p2j4-qwen-runtime-paired-v1" / "qwen-base.json"

SYSTEM = (
    "You are Mico's Scientific Agent decision policy. Choose the next action "
    "from the candidate actions in the policy state. Use only the supplied "
    "structured state and action history. Return one JSON object and no Markdown. "
    "The JSON keys must be exactly: selected_action, decision_reason, "
    "alternative_actions, stop_reason. Use stop_reason only when selected_action "
    "is finish."
)
ALL_ACTIONS = {
    "execute_read_query", "inspect_cohort", "compare_groups", "stratified_analysis",
    "adjust_confounders", "cross_project_validate", "cross_disease_validate",
    "retrieve_evidence", "analyze_projection", "finish",
}
DIMENSIONS = ("efficiency", "exploration_depth", "reason_quality")


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()[:20]


def _completion(action: str, reason: str, candidates: list[str], stop: str | None = None) -> str:
    if action == "finish":
        # The rejected output remains schema-valid.  Its *preference* can be
        # wrong for the evidence state, but it must not be structurally invalid.
        stop = stop or "EVIDENCE_SUFFICIENT"
    elif stop is not None:
        raise ValueError("non-finish action cannot use stop reason")
    return _canonical({
        "selected_action": action,
        "decision_reason": reason,
        "alternative_actions": [item for item in candidates if item != action],
        "stop_reason": stop,
    })


def _state(
    *, kind: str, goal: str, flags: list[str], history: list[str], candidates: list[str], profile: str,
) -> dict[str, Any]:
    return {
        "task_kind": kind,
        "goal_code": goal,
        "observation_flags": flags,
        "history_actions": history,
        "candidate_actions": candidates,
        "state_summary": (
            f"goal={goal}; profile={profile}; flags={','.join(flags)}; "
            f"history={','.join(history) if history else 'none'}; candidates={','.join(candidates)}"
        ),
    }


@dataclass(frozen=True)
class Scenario:
    dimension: str
    name: str
    kind: str
    goal: str
    flags: tuple[str, ...]
    history: tuple[str, ...]
    candidates: tuple[str, ...]
    chosen_action: str
    rejected_action: str
    chosen_reason: str
    rejected_reason: str
    chosen_stop: str | None = None
    rejected_stop: str | None = None
    source_kind: str = "controlled_policy_contract"
    source_ids: tuple[str, ...] = ()


# Every action in this table is permitted by the state contract.  In particular,
# the rejected action is not an unknown tool or a schema-invalid response.
SCENARIOS: tuple[Scenario, ...] = (
    Scenario("efficiency", "data_before_external_evidence", "focused_analysis", "bounded_group_comparison",
             ("COHORT_VALIDATED", "COMPARISON_PENDING", "EVIDENCE_NOT_REQUIRED_YET"),
             ("inspect_cohort",), ("compare_groups", "retrieve_evidence", "finish"),
             "compare_groups", "retrieve_evidence",
             "the validated cohort still needs its bounded internal comparison; external retrieval is useful later but premature now",
             "retrieve external evidence before completing the available internal comparison"),
    Scenario("efficiency", "summarize_before_external_evidence", "focused_analysis", "effect_summary",
             ("COHORT_VALIDATED", "COMPARISON_COMPLETE", "PROJECTION_PENDING"),
             ("inspect_cohort", "compare_groups"), ("analyze_projection", "retrieve_evidence", "finish"),
             "analyze_projection", "retrieve_evidence",
             "a bounded comparison exists but has not been summarized; analyze it before using external evidence",
             "retrieve evidence before summarizing the completed comparison"),
    Scenario("efficiency", "scope_limited_stop", "focused_analysis", "bounded_group_comparison",
             ("COHORT_VALIDATED", "COMPARISON_COMPLETE", "ANALYSIS_AVAILABLE", "PROJECT_BALANCED", "TASK_SCOPE_SATISFIED"),
             ("inspect_cohort", "compare_groups", "analyze_projection"), ("finish", "cross_project_validate", "retrieve_evidence"),
             "finish", "cross_project_validate",
             "the requested bounded comparison is complete, projects are balanced, and no extra validation obligation is active",
             "run cross-project validation despite no project-risk flag and a completed scoped objective",
             "EVIDENCE_SUFFICIENT", None),
    Scenario("efficiency", "evidence_complete_stop", "open_exploration", "cross_project_stability",
             ("COHORT_VALIDATED", "PROJECT_VALIDATED", "EVIDENCE_RETRIEVED", "EVIDENCE_CONSISTENT", "TASK_SCOPE_SATISFIED"),
             ("inspect_cohort", "compare_groups", "cross_project_validate", "retrieve_evidence"),
             ("finish", "retrieve_evidence", "cross_disease_validate"),
             "finish", "retrieve_evidence",
             "the required replication and evidence checks are complete; another retrieval would not add a stated obligation",
             "repeat evidence retrieval although the evidence state is already complete and consistent",
             "EVIDENCE_SUFFICIENT", None),
    Scenario("efficiency", "no_repeat_projection", "focused_analysis", "effect_summary",
             ("COHORT_VALIDATED", "COMPARISON_COMPLETE", "ANALYSIS_AVAILABLE", "PROJECTION_SUMMARIZED", "EVIDENCE_REQUIRED"),
             ("inspect_cohort", "compare_groups", "analyze_projection"), ("retrieve_evidence", "cross_project_validate", "finish"),
             "retrieve_evidence", "cross_project_validate",
             "the projection is already summarized and evidence is the remaining obligation; retrieve it rather than add an unrequired validation branch",
             "run a further project validation although no project-risk obligation is active"),
    Scenario("exploration_depth", "project_validation_before_retrieval", "open_exploration", "cross_project_stability",
             ("COHORT_VALIDATED", "COMPARISON_COMPLETE", "ANALYSIS_AVAILABLE", "CROSS_PROJECT_REQUIRED", "PROJECT_IMBALANCE"),
             ("inspect_cohort", "compare_groups", "analyze_projection"), ("cross_project_validate", "retrieve_evidence", "finish"),
             "cross_project_validate", "retrieve_evidence",
             "project imbalance creates a replication obligation; validate stability across projects before interpreting literature support",
             "retrieve evidence before resolving the active project replication risk"),
    Scenario("exploration_depth", "adjust_before_stratify", "focused_analysis", "confounder_adjustment",
             ("COHORT_VALIDATED", "COMPARISON_COMPLETE", "ANALYSIS_AVAILABLE", "CONFOUNDER_PRESENT", "AGE_SEX_IMBALANCE"),
             ("inspect_cohort", "compare_groups"), ("adjust_confounders", "stratified_analysis", "finish"),
             "adjust_confounders", "stratified_analysis",
             "a measured age or sex imbalance should be adjusted in the primary comparison before an optional subgroup analysis",
             "start a subgroup analysis before adjusting the documented cohort confounder"),
    Scenario("exploration_depth", "specificity_before_literature", "open_exploration", "disease_specificity",
             ("COHORT_VALIDATED", "COMPARISON_COMPLETE", "ANALYSIS_AVAILABLE", "CROSS_DISEASE_REQUIRED"),
             ("inspect_cohort", "compare_groups", "analyze_projection"), ("cross_disease_validate", "retrieve_evidence", "finish"),
             "cross_disease_validate", "retrieve_evidence",
             "the stated objective is specificity, so resolve the available cross-disease comparison before external interpretation",
             "retrieve external evidence before testing the required disease-specificity contrast"),
    Scenario("exploration_depth", "conflict_requires_boundary_evidence", "open_exploration", "evidence_conflict_resolution",
             ("COHORT_VALIDATED", "ANALYSIS_AVAILABLE", "EVIDENCE_CONFLICT", "EVIDENCE_INCOMPLETE"),
             ("inspect_cohort", "compare_groups", "analyze_projection"), ("retrieve_evidence", "finish", "cross_project_validate"),
             "retrieve_evidence", "finish",
             "conflicting and incomplete evidence has not met the evidence obligation; retrieve bounded corroborating evidence before stopping",
             "stop despite an explicit unresolved evidence conflict"),
    Scenario("exploration_depth", "project_validation_before_stop", "open_exploration", "cross_project_stability",
             ("COHORT_VALIDATED", "COMPARISON_COMPLETE", "CROSS_PROJECT_REQUIRED", "PARTIAL_PROJECT_COVERAGE"),
             ("inspect_cohort", "compare_groups"), ("cross_project_validate", "finish", "retrieve_evidence"),
             "cross_project_validate", "finish",
             "partial project coverage leaves the requested stability question unresolved, so validate before drawing a boundary",
             "stop while the requested cross-project stability check remains incomplete"),
    Scenario("reason_quality", "qualified_finish_reason", "focused_analysis", "bounded_group_comparison",
             ("COHORT_VALIDATED", "COMPARISON_COMPLETE", "ANALYSIS_AVAILABLE", "PROJECT_BALANCED", "TASK_SCOPE_SATISFIED"),
             ("inspect_cohort", "compare_groups", "analyze_projection"), ("finish", "retrieve_evidence"),
             "finish", "finish",
             "the requested bounded comparison is complete and balanced; report an observational association only and stop without implying causality",
             "evidence sufficient",
             "EVIDENCE_SUFFICIENT", "EVIDENCE_SUFFICIENT"),
    Scenario("reason_quality", "quality_risk_finish_reason", "open_exploration", "evidence_conflict_resolution",
             ("COHORT_VALIDATED", "ANALYSIS_AVAILABLE", "EVIDENCE_CONFLICT", "EVIDENCE_INCOMPLETE", "RETRIEVAL_EXHAUSTED"),
             ("inspect_cohort", "compare_groups", "retrieve_evidence"), ("finish", "retrieve_evidence"),
             "finish", "finish",
             "available evidence remains conflicted after the bounded retrieval; stop with a quality-risk boundary and do not present a biomarker claim",
             "evidence sufficient",
             "QUALITY_RISK", "EVIDENCE_SUFFICIENT"),
    Scenario("reason_quality", "confounder_reason", "focused_analysis", "confounder_adjustment",
             ("COHORT_VALIDATED", "COMPARISON_COMPLETE", "CONFOUNDER_PRESENT", "PROJECT_IMBALANCE"),
             ("inspect_cohort", "compare_groups"), ("adjust_confounders", "stratified_analysis", "finish"),
             "adjust_confounders", "adjust_confounders",
             "age, sex, or project imbalance remains in the observed comparison; adjust these measured confounders before interpreting the association",
             "perform the next appropriate validation step"),
    Scenario("reason_quality", "replication_reason", "open_exploration", "cross_project_stability",
             ("COHORT_VALIDATED", "COMPARISON_COMPLETE", "CROSS_PROJECT_REQUIRED", "PROJECT_IMBALANCE"),
             ("inspect_cohort", "compare_groups"), ("cross_project_validate", "retrieve_evidence", "finish"),
             "cross_project_validate", "cross_project_validate",
             "the preliminary association may be project-specific, so test direction and availability across eligible projects before claiming stability",
             "perform more validation"),
    Scenario("reason_quality", "retrieval_reason", "open_exploration", "evidence_conflict_resolution",
             ("COHORT_VALIDATED", "ANALYSIS_AVAILABLE", "EVIDENCE_CONFLICT", "EVIDENCE_INCOMPLETE"),
             ("inspect_cohort", "compare_groups", "analyze_projection"), ("retrieve_evidence", "finish"),
             "retrieve_evidence", "retrieve_evidence",
             "the data observation is available but external evidence conflicts or is incomplete; retrieve bounded evidence to characterize agreement rather than assert causality",
             "look up literature"),
)


PROFILES = (
    "age_shift", "sex_shift", "country_shift", "project_shift", "balanced_cohort",
    "partial_replication", "direction_gap", "effect_gap", "literature_gap", "graph_gap",
    "vector_graph_conflict", "cross_disease_gap", "bounded_scope", "quality_boundary", "replication_complete",
)


def _context_features(index: int) -> tuple[list[str], str]:
    """Produce de-identified but semantically meaningful state variation.

    The dimensions mirror metadata conditions the Agent already reasons about;
    they are not paraphrase IDs.  Combining them gives distinct decision states
    without inserting diseases, user questions, SQL, or raw sample identifiers.
    """
    projects = ("PROJECTS_2", "PROJECTS_3", "PROJECTS_4", "PROJECTS_5", "PROJECTS_6", "PROJECTS_7", "PROJECTS_8")
    countries = ("COUNTRIES_2", "COUNTRIES_3", "COUNTRIES_4", "COUNTRIES_5")
    age = ("AGE_SHIFT_LOW", "AGE_SHIFT_MODERATE", "AGE_SHIFT_HIGH")
    sex = ("SEX_BALANCED", "SEX_IMBALANCE")
    direction = ("EFFECT_DIRECTION_STABLE", "EFFECT_DIRECTION_MIXED")
    coverage = ("EVIDENCE_COVERAGE_LOW", "EVIDENCE_COVERAGE_MEDIUM", "EVIDENCE_COVERAGE_HIGH")
    values = []
    for choices in (projects, countries, age, sex, direction, coverage):
        values.append(choices[index % len(choices)])
        index //= len(choices)
    return values, "; ".join(values)


def _scenario_count(dimension: str) -> int:
    return {"efficiency": 500, "exploration_depth": 300, "reason_quality": 300}[dimension]


def _runtime_failure_ids() -> tuple[str, ...]:
    """Return only completed, policy-induced Base failure case IDs.

    Infrastructure-preflight and accidental-concurrent-write artefacts are not
    loaded and cannot become DPO provenance.
    """
    payload = json.loads(BASE_RUNTIME.read_text(encoding="utf-8"))
    allowed = {"DECISION_ACCURACY_FAILURE", "TRACE_ACTION_REPEATED", "TOOL_EFFICIENCY_FAILURE"}
    ids: set[str] = set()
    for item in payload.get("badCases", []):
        if item.get("failureCode") in allowed and item.get("reviewDecision") != "INFRASTRUCTURE_EXCLUDED":
            ids.add(item["caseId"])
    return tuple(sorted(ids))


def build() -> dict[str, Any]:
    runtime_ids = _runtime_failure_ids()
    by_dimension = {name: [s for s in SCENARIOS if s.dimension == name] for name in DIMENSIONS}
    records: list[dict[str, Any]] = []
    for dimension in DIMENSIONS:
        total = _scenario_count(dimension)
        scenarios = by_dimension[dimension]
        for index in range(total):
            scenario = scenarios[index % len(scenarios)]
            profile = PROFILES[(index // len(scenarios)) % len(PROFILES)]
            candidates = list(scenario.candidates)
            context_flags, context_description = _context_features(index)
            # Do not place a low-coverage or mixed-direction flag into a
            # preferred evidence-sufficient finish state.  Those flags carry a
            # validation obligation and would make a stop pair internally
            # ambiguous.  The remaining cohort dimensions still give ample,
            # meaningful state variation for the selected 70 examples/family.
            if scenario.chosen_action == "finish" and scenario.chosen_stop == "EVIDENCE_SUFFICIENT":
                context_flags = [flag for flag in context_flags if not (
                    flag.startswith("EFFECT_DIRECTION_") or flag.startswith("EVIDENCE_COVERAGE_")
                )]
                context_description = "; ".join(context_flags)
            state = _state(kind=scenario.kind, goal=scenario.goal, flags=[*scenario.flags, *context_flags],
                           history=list(scenario.history), candidates=candidates, profile=profile)
            state["state_summary"] += f"; metadata_context={context_description}"
            chosen = _completion(scenario.chosen_action, scenario.chosen_reason, candidates, scenario.chosen_stop)
            rejected = _completion(scenario.rejected_action, scenario.rejected_reason, candidates, scenario.rejected_stop)
            # A small, explicit subset is derived from the audited repeated-action
            # pattern.  It remains a policy-pattern source rather than pretending
            # the controlled state is an unmodified Runtime observation.
            source_kind = scenario.source_kind
            source_ids = list(scenario.source_ids)
            if dimension == "efficiency" and scenario.name == "no_repeat_projection" and runtime_ids:
                source_kind = "runtime_failure_pattern_derived"
                source_ids = [runtime_ids[index % len(runtime_ids)]]
            state_signature = _hash({key: state[key] for key in (
                "task_kind", "goal_code", "observation_flags", "history_actions", "candidate_actions", "state_summary")})
            # A family is a decision-obligation template, not cosmetic wording.
            # The profile is retained separately as de-identified state context.
            family = f"{dimension}:{scenario.name}"
            records.append({
                "id": f"dpo-v3-candidate-{dimension}-{index:03d}",
                "prompt": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": _canonical({"policy_state": state})},
                ],
                "chosen": chosen,
                "rejected": rejected,
                "metadata": {
                    "candidate_status": "AUTO_ELIGIBLE",
                    "preference_dimension": dimension,
                    "preference_reason": (
                        "choose the minimal necessary next step" if dimension == "efficiency"
                        else "choose deeper validation only when the active evidence obligation requires it"
                        if dimension == "exploration_depth"
                        else "prefer a qualified, state-grounded rationale over a generic rationale"
                    ),
                    "scenario_name": scenario.name,
                    "task_family": family,
                    "family_variant": profile,
                    "state_signature": state_signature,
                    "source_kind": source_kind,
                    "source_ids": source_ids,
                    "policy_induced": True,
                    "review_priority": "high" if source_kind == "runtime_failure_pattern_derived" or dimension != "efficiency" else "normal",
                },
            })
    return {
        "schemaVersion": "p2j4-decision-dpo-v3-candidate-v1",
        "status": "REVIEW_REQUIRED",
        "purpose": "DPO v3 policy preference candidates; not a training freeze",
        "candidateCount": len(records),
        "dimensionCounts": {name: sum(1 for item in records if item["metadata"]["preference_dimension"] == name) for name in DIMENSIONS},
        "excludedSources": [
            "qwen-base-infrastructure-preflight-failure.json",
            "qwen-base-concurrent-write-aborted.json",
        ],
        "runtimeFailureSourceCaseCount": len(runtime_ids),
        "records": records,
    }


def main() -> None:
    payload = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("status", "candidateCount", "dimensionCounts", "runtimeFailureSourceCaseCount")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
