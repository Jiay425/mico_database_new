"""Build a reviewable 50-case Evidence Conflict repair candidate pool.

These are structured policy states, not claims about a new biological cohort.
They are intentionally kept outside the frozen v1 SFT split until owner
review confirms the stop-reason oracle.  The generator varies the state phase,
evidence route, history, approved alternatives, and evidence-binding count so
the pool is not a repeated copy of the one discovered Bad Case.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


GOAL = "evidence_conflict_resolution"
TASK_KIND = "open_exploration"
TASK_FAMILY = "open_exploration:evidence_conflict_resolution:evidence_conflict_repair_v1"
HARD_CLASS = "evidence_conflict"


REASON_VARIANTS = {
    "inspect": [
        "inspect the cohort before interpreting the conflicting evidence",
        "establish the metadata and cohort context before resolving the evidence conflict",
        "start with a bounded cohort inspection because the conflict has no validated data context yet",
        "inspect metadata first so the conflicting sources are not treated as a standalone finding",
    ],
    "retrieve": [
        "retrieve bounded evidence to resolve the conflict after the data comparison is available",
        "add another grounded evidence route before making a conclusion because the sources disagree",
        "seek additional bounded literature or graph evidence to distinguish the conflicting claims",
        "continue evidence retrieval because the current conflict is unresolved and the observation is not sufficient",
    ],
    "finish": [
        "stop with a bounded quality-risk conclusion because the retrieved sources remain conflicted",
        "finish only with the conflict disclosed; the current evidence is not sufficient for a stronger claim",
        "end the exploration with downgraded confidence because the evidence conflict remains unresolved",
        "use a quality-risk stop because the available evidence cannot support an unqualified conclusion",
    ],
}


def _candidate(
    *,
    index: int,
    phase: str,
    flags: list[str],
    history: list[str],
    actions: list[str],
    selected: str,
    bindings: int,
    routes: list[str],
    reason_key: str,
    evidence_profile: str,
) -> dict[str, Any]:
    reason = REASON_VARIANTS[reason_key][index % len(REASON_VARIANTS[reason_key])]
    state_summary = (
        f"phase={phase}; evidence_profile={evidence_profile}; observation_state="
        f"{'NO_OBSERVATION' if not history else 'EVIDENCE_RETRIEVED_REQUIRES_GROUNDING' if 'retrieve_evidence' in history else 'VALIDATED_OBSERVATION'}; "
        f"evidence_bindings={bindings}; source_routes={','.join(routes) if routes else 'none'}; "
        f"prior_actions={','.join(history) if history else 'none'}; conflict=EVIDENCE_CONFLICT"
    )
    alternatives = [action for action in actions if action != selected]
    material = json.dumps({
        "task_kind": TASK_KIND,
        "goal_code": GOAL,
        "observation_flags": flags,
        "history_actions": history,
        "candidate_actions": actions,
        "state_summary": state_summary,
        "decision_reason": reason,
        "selected_action": selected,
        "alternative_actions": alternatives,
        "stop_reason": "QUALITY_RISK" if selected == "finish" else None,
    }, ensure_ascii=False, sort_keys=True)
    signature = hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
    return {
        "schemaVersion": "p2j4-decision-dataset-v3",
        "sourceTraceId": f"repair-trace-{signature}",
        "task_kind": TASK_KIND,
        "goal_code": GOAL,
        "task_family": TASK_FAMILY,
        "hard_case_class": HARD_CLASS,
        "observation_flags": flags,
        "history_actions": history,
        "candidate_actions": actions,
        "state_summary": state_summary,
        "decision_reason": reason,
        "selected_action": selected,
        "alternative_actions": alternatives,
        "stop_reason": "QUALITY_RISK" if selected == "finish" else None,
        "reviewStatus": "OWNER_REVIEW_PENDING",
        "repairPhase": phase,
        "signatureHash": signature,
        "sourceRun": "evidence_conflict_repair_v1",
    }


def build() -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    index = 0
    no_observation_profiles = [
        "metadata_scope_unknown",
        "project_scope_unknown",
        "confounder_scope_unknown",
        "cohort_balance_unknown",
        "evidence_route_unverified",
        "vector_claim_without_cohort",
        "graph_claim_without_cohort",
        "literature_claim_without_cohort",
        "cross_disease_scope_unknown",
        "multi_feature_scope_unknown",
    ]

    # 10 states: conflict is known before any data observation.  The safe next
    # step remains metadata-first inspection, not a premature conclusion.
    for variant in range(10):
        index += 1
        flags = ["NO_OBSERVATION", "EVIDENCE_CONFLICT", "METADATA_FIRST"]
        if variant % 2:
            flags.append("EVIDENCE_INCOMPLETE")
        candidates.append(_candidate(
            index=index,
            phase="NO_OBSERVATION",
            flags=flags,
            history=[],
            actions=["inspect_cohort", "retrieve_evidence", "finish"],
            selected="inspect_cohort",
            bindings=0,
            routes=[],
            reason_key="inspect",
            evidence_profile=no_observation_profiles[variant],
        ))

    # 14 states: data has been compared but the conflict has not yet received
    # a bounded evidence resolution attempt.
    validated_profiles = [
        "java_comparison_only",
        "java_comparison_project_validated",
        "java_comparison_confounder_adjusted",
        "java_comparison_project_and_confounder_adjusted",
        "java_two_project_comparison",
        "java_multi_project_comparison",
        "java_age_adjusted_comparison",
        "java_sex_adjusted_comparison",
        "java_country_checked_comparison",
        "java_batch_checked_comparison",
        "java_group_difference_unresolved",
        "java_direction_conflict_unresolved",
        "java_feature_conflict_unresolved",
        "java_disease_scope_conflict",
    ]
    for variant in range(14):
        index += 1
        flags = [
            "OBSERVATION_VALIDATED", "JAVA_OBSERVED", "ANALYSIS_AVAILABLE",
            "EVIDENCE_CONFLICT", "EVIDENCE_INCOMPLETE", "METADATA_FIRST",
        ]
        if variant % 3 == 1:
            flags.append("PROJECT_VALIDATED")
        if variant % 3 == 2:
            flags.append("CONFOUNDER_ADJUSTED")
        candidates.append(_candidate(
            index=index,
            phase="VALIDATED_BEFORE_RETRIEVAL",
            flags=flags,
            history=["inspect_cohort", "compare_groups"],
            actions=["inspect_cohort", "compare_groups", "retrieve_evidence", "finish"],
            selected="retrieve_evidence",
            bindings=2 + (variant % 3),
            routes=["java"],
            reason_key="retrieve",
            evidence_profile=validated_profiles[variant],
        ))

    # 16 states: evidence was retrieved but remains incomplete/conflicted.  A
    # bounded QUALITY_RISK finish is the oracle, not EVIDENCE_SUFFICIENT.
    retrieved_profiles = [
        "vector_incomplete_after_comparison",
        "vector_conflict_after_project_validation",
        "vector_conflict_after_confounder_adjustment",
        "vector_conflict_after_project_and_confounder_adjustment",
        "vector_direction_disagreement",
        "vector_effect_size_disagreement",
        "vector_replication_gap",
        "vector_metadata_context_gap",
        "vector_disease_scope_gap",
        "vector_feature_scope_gap",
        "vector_literature_alignment_gap",
        "vector_graph_alignment_gap",
        "vector_single_project_risk",
        "vector_single_country_risk",
        "vector_age_balance_risk",
        "vector_sex_balance_risk",
    ]
    for variant in range(16):
        index += 1
        flags = [
            "OBSERVATION_VALIDATED", "EVIDENCE_RETRIEVED", "EVIDENCE_INCOMPLETE",
            "EVIDENCE_CONFLICT", "JAVA_OBSERVED", "VECTOR_EVIDENCE",
            "ANALYSIS_AVAILABLE",
        ]
        if variant % 2:
            flags.append("PROJECT_VALIDATED")
        if variant % 3 == 2:
            flags.append("CONFOUNDER_ADJUSTED")
        candidates.append(_candidate(
            index=index,
            phase="RETRIEVED_CONFLICT_REMAINS",
            flags=flags,
            history=["inspect_cohort", "compare_groups", "retrieve_evidence"],
            actions=["inspect_cohort", "compare_groups", "retrieve_evidence", "finish"],
            selected="finish",
            bindings=3 + (variant % 4),
            routes=["java", "vector"],
            reason_key="finish",
            evidence_profile=retrieved_profiles[variant],
        ))

    # 10 states: the available actions are intentionally narrower and the
    # conflict is still not safe to upgrade to a positive conclusion.
    terminal_profiles = [
        "quality_risk_vector_only",
        "quality_risk_project_checked",
        "quality_risk_confounder_checked",
        "quality_risk_project_and_confounder_checked",
        "quality_risk_direction_conflict",
        "quality_risk_replication_gap",
        "quality_risk_metadata_gap",
        "quality_risk_disease_scope_gap",
        "quality_risk_feature_scope_gap",
        "quality_risk_claim_downgrade_required",
    ]
    for variant in range(10):
        index += 1
        flags = [
            "OBSERVATION_VALIDATED", "EVIDENCE_RETRIEVED", "EVIDENCE_CONFLICT",
            "JAVA_OBSERVED", "VECTOR_EVIDENCE", "ANALYSIS_AVAILABLE",
        ]
        if variant % 2:
            flags.append("PROJECT_VALIDATED")
        candidates.append(_candidate(
            index=index,
            phase="QUALITY_RISK_TERMINAL",
            flags=flags,
            history=["inspect_cohort", "compare_groups", "retrieve_evidence"],
            actions=["retrieve_evidence", "finish"],
            selected="finish",
            bindings=4 + (variant % 3),
            routes=["java", "vector"],
            reason_key="finish",
            evidence_profile=terminal_profiles[variant],
        ))

    signatures = [candidate["signatureHash"] for candidate in candidates]
    if len(candidates) != 50 or len(set(signatures)) != 50:
        raise ValueError("EVIDENCE_CONFLICT_REPAIR_COUNT_OR_SIGNATURE_INVALID")
    if any(
        candidate["selected_action"] == "finish"
        and candidate["stop_reason"] != "QUALITY_RISK"
        for candidate in candidates
    ):
        raise ValueError("EVIDENCE_CONFLICT_FINISH_ORACLE_INVALID")
    return {
        "schemaVersion": "p2j4-evidence-conflict-repair-candidates-v1",
        "status": "OWNER_REVIEW_PENDING",
        "trainingStarted": False,
        "candidateCount": len(candidates),
        "sourceRun": "evidence_conflict_repair_v1",
        "purpose": "repair_reason_and_stop_semantics_only",
        "notExternalBiologicalCohort": True,
        "phaseCounts": {
            "NO_OBSERVATION": 10,
            "VALIDATED_BEFORE_RETRIEVAL": 14,
            "RETRIEVED_CONFLICT_REMAINS": 16,
            "QUALITY_RISK_TERMINAL": 10,
        },
        "oracleContract": {
            "conflictFlagRequired": True,
            "finishStopReason": "QUALITY_RISK",
            "forbiddenFinishStopReason": "EVIDENCE_SUFFICIENT",
            "rawQuestionIncluded": False,
            "rawPayloadIncluded": False,
        },
        "candidates": candidates,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build evidence-conflict repair candidates")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = build()
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, TypeError) as exc:
        print(json.dumps({"status": "INVALID", "errorCode": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({
        "status": result["status"],
        "candidateCount": result["candidateCount"],
        "phaseCounts": result["phaseCounts"],
        "trainingStarted": result["trainingStarted"],
        "output": str(args.output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
