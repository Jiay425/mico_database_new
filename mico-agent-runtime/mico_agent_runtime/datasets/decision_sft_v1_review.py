"""Offline semantic review and adjudication for Decision SFT v1.

The reviewer in this module is the repository agent itself.  It does not call
Gemini, Qwen, DeepSeek, or any other external model.  It reads every
contract-valid record, applies the fixed semantic rubric, records any target or
provenance revision, and runs the deterministic validator again before
writing a new split.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from mico_agent_runtime.contracts.decision_state import ScientificDecisionState
from mico_agent_runtime.contracts.schema_catalog import SchemaSemanticCatalog
from mico_agent_runtime.datasets.decision_sft_v1 import (
    HARD_CASE_CLASSES,
    SOURCE_TYPES,
    STATE_BLOCKS,
    DecisionSftRecord,
    _build_state,
    _canonical_json,
    _coarse_state_signature,
    _dump_json,
    _dump_jsonl,
    _load_catalog,
    _reason,
    _split_leakage,
    _walk_keys,
    normalized_state_hash,
    validate_record,
)
from mico_agent_runtime.runtime.action_availability import ALL_SCIENTIFIC_ACTIONS


REVIEW_SCHEMA_VERSION = "decision-sft-v1-review"
REVIEW_VERDICTS = ("PASS", "REVISE", "REJECT")
HIGH_RISK_HARD_CASES = frozenset(
    {
        "objective_action_confusion",
        "availability_boundary",
        "redundant_action",
        "premature_finish",
        "missed_finish",
        "observation_sensitivity",
        "heterogeneity_conflict",
    }
)
# The frozen test must exercise every taxonomy bucket, not only the policy
# high-risk subset.  This prevents missing-project/covariate/evidence cases
# from being hidden by a favorable aggregate score.
REQUIRED_TEST_HARD_CASES = frozenset(HARD_CASE_CLASSES)
REVIEW_TARGET_COUNTS = {"train": 390, "validation": 65, "test": 65}


def _review_target_counts(total: int) -> dict[str, int]:
    """Keep fixed holdouts while allowing semantic review rejects."""

    holdout = REVIEW_TARGET_COUNTS["validation"] + REVIEW_TARGET_COUNTS["test"]
    if total < holdout:
        raise ValueError(f"REVIEW_SPLIT_TOO_SMALL:{total}")
    return {
        "train": total - holdout,
        "validation": REVIEW_TARGET_COUNTS["validation"],
        "test": REVIEW_TARGET_COUNTS["test"],
    }


def _read_records(path: Path) -> list[DecisionSftRecord]:
    records: list[DecisionSftRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            records.append(DecisionSftRecord.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            raise ValueError(f"REVIEW_INPUT_INVALID:{path}:{line_number}:{exc}") from exc
    return records


def _state_internal_issues(record: DecisionSftRecord) -> list[str]:
    """Check State facts and derived progress without choosing a strategy."""

    state = record.state
    data = state.data_state
    analysis = state.analysis_state
    evidence = state.evidence_state
    progress = state.progress
    issues: list[str] = []

    if data.has_tabular_data:
        if data.row_count <= 0:
            issues.append("tabular_data_without_rows")
        if data.group_state.group_count != len(data.group_state.group_sizes):
            issues.append("group_count_size_mismatch")
        if data.group_state.group_count == 0:
            issues.append("tabular_data_without_group")
        if data.group_state.group_field is None:
            issues.append("group_field_missing_for_tabular_data")
        if data.group_state.group_field not in data.available_dimensions:
            issues.append("group_field_not_in_dimensions")
        if data.sample_count is not None and data.sample_count != sum(data.group_state.group_sizes.values()):
            issues.append("sample_count_size_mismatch")
    else:
        if data.row_count != 0:
            issues.append("no_tabular_data_with_rows")
        if data.group_state.group_count != 0 or data.group_state.group_sizes:
            issues.append("no_tabular_data_with_groups")
        if data.group_state.group_field is not None:
            issues.append("no_tabular_data_with_group_field")
        if data.available_dimensions or data.available_outcomes:
            issues.append("no_tabular_data_with_semantic_fields")
        if data.sample_count is not None and data.sample_count != 0:
            issues.append("no_tabular_data_with_samples")

    project = data.project_state
    has_project_dimension = "metadata.project" in data.available_dimensions
    if project.project_count == 0:
        if project.has_project_field:
            issues.append("project_flag_count_mismatch")
        if has_project_dimension:
            issues.append("project_dimension_without_projects")
    else:
        if not project.has_project_field:
            issues.append("project_count_without_project_flag")
        if not has_project_dimension:
            issues.append("project_count_without_project_dimension")

    completed = set(progress.completed_actions)
    if progress.action_count != sum(progress.action_counts.values()):
        issues.append("action_count_mismatch")
    if progress.last_action is not None and progress.last_action not in completed:
        issues.append("last_action_not_completed")

    status_to_action = {
        "group_comparison": (analysis.group_comparison.status, "compare_groups"),
        "confounder_assessment": (analysis.confounder_adjustment.status, "adjust_confounders"),
        "cross_project_validation": (analysis.cross_project_validation.status, "cross_project_validate"),
        "stratified_analysis": (analysis.stratified_analysis.status, "stratified_analysis"),
        "projection_analysis": (analysis.projection_analysis.status, "analyze_projection"),
        "cross_disease_validation": (analysis.cross_disease_validation.status, "cross_disease_validate"),
        "evidence_support": (evidence.status, "retrieve_evidence"),
    }
    for objective, (status, action) in status_to_action.items():
        if status == "completed" and action not in completed:
            issues.append(f"completed_status_missing_action:{objective}")
        if status == "completed" and objective in state.task.objectives:
            if objective in progress.remaining_objectives:
                issues.append(f"completed_status_remaining_objective:{objective}")

    if analysis.group_comparison.status == "completed":
        group = data.group_state
        if group.group_count != 2 or not data.available_outcomes or group.group_field is None:
            issues.append("completed_group_comparison_without_inputs")
        if analysis.group_comparison.effect_size is None and analysis.group_comparison.mean_difference is None:
            issues.append("completed_group_comparison_without_effect")
    if analysis.confounder_adjustment.status == "completed":
        if analysis.group_comparison.status != "completed":
            issues.append("completed_adjustment_without_group_comparison")
        if not data.covariate_state.available_covariates:
            issues.append("completed_adjustment_without_covariate")
        if not analysis.confounder_adjustment.adjusted_covariates:
            issues.append("completed_adjustment_without_adjusted_covariates")
    if analysis.cross_project_validation.status == "completed":
        cross = analysis.cross_project_validation
        if project.project_count < 2 or not project.has_project_field:
            issues.append("completed_cross_project_without_two_projects")
        if cross.project_count < 2:
            issues.append("completed_cross_project_count_too_low")
    if analysis.stratified_analysis.status == "completed":
        stratified = analysis.stratified_analysis
        if not stratified.stratify_fields or stratified.stratum_count < 1:
            issues.append("completed_stratification_without_strata")
    if analysis.projection_analysis.status == "completed" and analysis.projection_analysis.result_count < 1:
        issues.append("completed_projection_without_results")
    if analysis.cross_disease_validation.status == "completed" and analysis.cross_disease_validation.disease_count < 2:
        issues.append("completed_cross_disease_without_two_diseases")
    if evidence.status == "completed":
        if evidence.evidence_count < evidence.support_count + evidence.conflict_count + evidence.context_count:
            issues.append("evidence_direction_counts_exceed_total")

    done_by_objective = {
        "group_comparison": analysis.group_comparison.status == "completed",
        "projection_analysis": analysis.projection_analysis.status == "completed",
        "stratified_analysis": analysis.stratified_analysis.status == "completed",
        "confounder_assessment": analysis.confounder_adjustment.status == "completed",
        "cross_project_validation": analysis.cross_project_validation.status == "completed",
        "cross_disease_validation": analysis.cross_disease_validation.status == "completed",
        "evidence_support": evidence.status == "completed",
    }
    expected_remaining = [
        objective
        for objective in state.task.objectives
        if not done_by_objective.get(objective, False)
    ]
    if expected_remaining != list(progress.remaining_objectives):
        issues.append("remaining_objectives_mismatch")

    action = record.target.selected_action
    status_by_action = {
        "compare_groups": analysis.group_comparison.status,
        "adjust_confounders": analysis.confounder_adjustment.status,
        "cross_project_validate": analysis.cross_project_validation.status,
        "stratified_analysis": analysis.stratified_analysis.status,
        "analyze_projection": analysis.projection_analysis.status,
        "cross_disease_validate": analysis.cross_disease_validation.status,
        "retrieve_evidence": evidence.status,
    }
    if action in status_by_action and status_by_action[action] == "completed":
        issues.append(f"redundant_completed_action:{action}")
    if action == "execute_read_query" and action in completed:
        # A repeat read is defensible only when the State explicitly shows a
        # missing capability that a new bounded observation could discover.
        can_refresh = (
            data.group_state.group_count < 2
            or not data.covariate_state.available_covariates
            or ("cross_project_validation" in progress.remaining_objectives and project.project_count < 2)
            or ("evidence_support" in progress.remaining_objectives and record.metadata.availability_profile == "catalog_no_knowledge")
        )
        if not can_refresh:
            issues.append("redundant_completed_action:execute_read_query")

    if action == "finish":
        if progress.remaining_objectives:
            issues.append("finish_with_remaining_objectives")
        if "finish" not in state.action_space.available_actions:
            issues.append("finish_not_available")
    elif not progress.remaining_objectives:
        # A non-finish target after all user objectives are complete is a
        # semantic missed-finish label, even if the action is technically
        # executable.
        issues.append("nonfinish_when_objectives_complete")

    return issues


def _hard_case_issues(record: DecisionSftRecord) -> list[str]:
    state = record.state
    data = state.data_state
    analysis = state.analysis_state
    evidence = state.evidence_state
    progress = state.progress
    action = record.target.selected_action
    hard = record.metadata.hard_case_class
    issues: list[str] = []
    if hard == "availability_boundary":
        blocked = False
        if data.project_state.project_count < 2:
            blocked |= "cross_project_validate" not in state.action_space.available_actions
        if not data.covariate_state.available_covariates:
            blocked |= "adjust_confounders" not in state.action_space.available_actions
        if data.group_state.group_count < 2:
            blocked |= "compare_groups" not in state.action_space.available_actions
        if record.metadata.availability_profile == "catalog_no_knowledge":
            blocked |= "retrieve_evidence" not in state.action_space.available_actions
        if not blocked:
            issues.append("availability_boundary_without_blocked_capability")
    elif hard == "objective_action_confusion":
        if action in {
            "group_comparison",
            "projection_analysis",
            "confounder_assessment",
            "cross_project_validation",
            "cross_disease_validation",
            "evidence_support",
        }:
            issues.append("objective_used_as_action")
    elif hard == "missing_project_dimension":
        if data.project_state.project_count != 0 or "metadata.project" in data.available_dimensions:
            issues.append("missing_project_case_has_project_dimension")
        if action == "cross_project_validate":
            issues.append("missing_project_case_selected_unavailable_action")
    elif hard == "missing_covariate":
        if data.covariate_state.available_covariates or action == "adjust_confounders":
            issues.append("missing_covariate_case_inconsistent")
    elif hard == "missing_evidence":
        if evidence.status == "completed" or action != "retrieve_evidence":
            issues.append("missing_evidence_case_inconsistent")
    elif hard == "insufficient_data":
        if data.group_state.group_count >= 2 or action == "compare_groups":
            issues.append("insufficient_data_case_inconsistent")
    elif hard == "premature_finish":
        if not progress.remaining_objectives or "finish" in state.action_space.available_actions or action == "finish":
            issues.append("premature_finish_case_inconsistent")
    elif hard == "missed_finish":
        if progress.remaining_objectives or "finish" not in state.action_space.available_actions or action != "finish":
            issues.append("missed_finish_case_inconsistent")
    elif hard == "heterogeneity_conflict":
        if analysis.cross_project_validation.status != "completed":
            issues.append("heterogeneity_case_without_cross_project_result")
        if analysis.cross_project_validation.heterogeneity not in {"low", "medium", "high"}:
            issues.append("heterogeneity_case_without_level")
    return issues


def _semantic_review(record: DecisionSftRecord, catalog: SchemaSemanticCatalog) -> list[str]:
    _, contract_issues = validate_record(record.model_dump(mode="json"), catalog=catalog)
    return list(dict.fromkeys(contract_issues + _state_internal_issues(record) + _hard_case_issues(record)))


def _replacement_action(record: DecisionSftRecord) -> str | None:
    state = record.state
    data = state.data_state
    analysis = state.analysis_state
    available = state.action_space.available_actions
    if record.target.selected_action == "execute_read_query":
        if analysis.group_comparison.status == "completed" and data.covariate_state.available_covariates:
            if "adjust_confounders" in available and analysis.confounder_adjustment.status != "completed":
                return "adjust_confounders"
        if "retrieve_evidence" in available and state.evidence_state.status != "completed":
            return "retrieve_evidence"
    if record.target.selected_action == "compare_groups" and analysis.group_comparison.status == "completed":
        if "adjust_confounders" in available and data.covariate_state.available_covariates:
            return "adjust_confounders"
        if "retrieve_evidence" in available:
            return "retrieve_evidence"
    if record.target.selected_action == "adjust_confounders" and analysis.confounder_adjustment.status == "completed":
        if "stratified_analysis" in available:
            return "stratified_analysis"
        if "retrieve_evidence" in available:
            return "retrieve_evidence"
    if record.target.selected_action == "retrieve_evidence" and state.evidence_state.status == "completed":
        if "finish" in available:
            return "finish"
    return None


def _rewrite_target(record: DecisionSftRecord, action: str | None = None) -> DecisionSftRecord:
    payload = record.model_dump(mode="json")
    selected = action or record.target.selected_action
    payload["target"]["selected_action"] = selected
    payload["target"]["decision_reason"] = _reason(
        selected,
        record.state,
        record.metadata.hard_case_class,
    )
    payload["target"]["alternative_actions"] = [
        value
        for value in payload["target"].get("alternative_actions", [])
        if value in record.state.action_space.available_actions and value != selected
    ][:2]
    payload["target"]["stop_reason"] = (
        payload["target"].get("stop_reason") if selected == "finish" else None
    )
    if selected == "finish" and payload["target"]["stop_reason"] is None:
        payload["target"]["stop_reason"] = "EVIDENCE_SUFFICIENT"
    return DecisionSftRecord.model_validate(payload)


def _supplemental_records(catalog: SchemaSemanticCatalog, start_index: int = 521) -> list[DecisionSftRecord]:
    """Add reviewed, controlled boundary coverage without hand-written actions."""

    records: list[DecisionSftRecord] = []
    index = start_index
    variant = 700

    def add(
        *,
        action: str,
        hard_case: str,
        source_type: str,
        kwargs: dict[str, Any],
        knowledge_available: bool = True,
    ) -> None:
        nonlocal index, variant
        state, profile = _build_state(
            catalog=catalog,
            variant=variant,
            desired_action=action,
            knowledge_available=knowledge_available,
            **kwargs,
        )
        raw = _record_for_review(
            sample_index=index,
            trajectory_id=f"trajectory-review-v1-{index:04d}",
            turn_index=0,
            state=state,
            source_type=source_type,
            availability_profile=profile,
            action=action,
            hard_case_class=hard_case,
            validation_dimension_counts=kwargs.get("validation_dimension_counts"),
        )
        records.append(DecisionSftRecord.model_validate(raw))
        index += 1
        variant += 1

    # Availability boundary: project 0 / 1, no covariate, no evidence route,
    # and insufficient group coverage.  The selected action is always an
    # executable bounded read; the unavailable scientific Action is not used
    # as a label.
    for _ in range(6):
        add(
            action="execute_read_query",
            hard_case="availability_boundary",
            source_type="boundary_state",
            kwargs={
                "mode": "groups",
                "group_status": "completed",
                "adjustment_status": "completed",
                "covariates": ["sample.age"],
                "project_count": 0,
                "group_effect": 0.4,
                "adjusted_effect": 0.2,
            },
        )
    for _ in range(5):
        add(
            action="execute_read_query",
            hard_case="availability_boundary",
            source_type="boundary_state",
            kwargs={
                "mode": "project",
                "group_status": "completed",
                "adjustment_status": "completed",
                "covariates": ["sample.age"],
                "project_count": 1,
                "group_effect": 0.4,
                "adjusted_effect": 0.2,
            },
        )
    for _ in range(4):
        add(
            action="execute_read_query",
            hard_case="availability_boundary",
            source_type="boundary_state",
            kwargs={
                "mode": "groups",
                "group_status": "completed",
                "covariates": [],
                "group_effect": 0.4,
            },
        )
    for _ in range(3):
        add(
            action="execute_read_query",
            hard_case="availability_boundary",
            source_type="boundary_state",
            kwargs={
                "mode": "groups",
                "group_status": "completed",
                "adjustment_status": "completed",
                "covariates": ["sample.age"],
                "group_effect": 0.4,
                "adjusted_effect": 0.2,
            },
            knowledge_available=False,
        )
    for _ in range(3):
        add(
            action="execute_read_query",
            hard_case="availability_boundary",
            source_type="boundary_state",
            kwargs={"mode": "single_group", "covariates": []},
        )

    # Redundancy cases teach the policy to move beyond an already completed
    # observation; all selected labels remain executable and useful.
    for _ in range(7):
        add(
            action="adjust_confounders",
            hard_case="redundant_action",
            source_type="controlled_state",
            kwargs={
                "mode": "groups",
                "group_status": "completed",
                "covariates": ["sample.age"],
                "group_effect": 0.5,
            },
        )
    for _ in range(3):
        add(
            action="retrieve_evidence",
            hard_case="redundant_action",
            source_type="controlled_state",
            kwargs={
                "mode": "groups",
                "group_status": "completed",
                "adjustment_status": "completed",
                "covariates": ["sample.age"],
                "group_effect": 0.5,
                "adjusted_effect": 0.2,
            },
        )
    for _ in range(3):
        add(
            action="stratified_analysis",
            hard_case="redundant_action",
            source_type="controlled_state",
            kwargs={
                "mode": "groups",
                "group_status": "completed",
                "adjustment_status": "completed",
                "covariates": ["sample.age"],
                "group_effect": 0.5,
                "adjusted_effect": 0.2,
            },
        )
    for _ in range(3):
        add(
            action="cross_disease_validate",
            hard_case="redundant_action",
            source_type="controlled_state",
            kwargs={
                "mode": "groups",
                "group_status": "completed",
                "covariates": ["sample.age"],
                "dimensions": [
                    "sample.disease",
                    "disease.name",
                    "abundance.feature",
                    "sample.age",
                ],
                "validation_dimension_counts": {"disease.name": 2},
                "group_effect": 0.5,
            },
        )

    # Premature-finish cases retain unfinished objectives and therefore have no
    # finish in hard availability.
    for _ in range(5):
        add(
            action="compare_groups",
            hard_case="premature_finish",
            source_type="boundary_state",
            kwargs={"mode": "groups", "group_status": "not_started", "covariates": ["sample.age"]},
        )
    for _ in range(4):
        add(
            action="adjust_confounders",
            hard_case="premature_finish",
            source_type="boundary_state",
            kwargs={
                "mode": "groups",
                "group_status": "completed",
                "adjustment_status": "not_started",
                "covariates": ["sample.age"],
                "group_effect": 0.5,
            },
        )
    for _ in range(3):
        add(
            action="execute_read_query",
            hard_case="premature_finish",
            source_type="boundary_state",
            kwargs={
                "mode": "groups",
                "group_status": "completed",
                "adjustment_status": "completed",
                "covariates": ["sample.age"],
                "group_effect": 0.5,
                "adjusted_effect": 0.2,
            },
        )
    for _ in range(2):
        add(
            action="retrieve_evidence",
            hard_case="premature_finish",
            source_type="boundary_state",
            kwargs={
                "mode": "groups",
                "group_status": "completed",
                "adjustment_status": "completed",
                "covariates": ["sample.age"],
                "group_effect": 0.5,
                "adjusted_effect": 0.2,
            },
        )
    return records


def _record_for_review(
    *,
    sample_index: int,
    trajectory_id: str,
    turn_index: int,
    state: ScientificDecisionState,
    source_type: str,
    availability_profile: str,
    action: str,
    hard_case_class: str | None,
    validation_dimension_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    target = {
        "selected_action": action,
        "decision_reason": _reason(action, state, hard_case_class),
        "alternative_actions": [],
        "stop_reason": "EVIDENCE_SUFFICIENT" if action == "finish" else None,
    }
    return {
        "schema_version": "decision-sft-v1",
        "sample_id": f"decision-sft-v1-{sample_index:06d}",
        "source_type": source_type,
        "trajectory_id": trajectory_id,
        "turn_index": turn_index,
        "state": state.model_dump(mode="json"),
        "target": target,
        "metadata": {
            "state_origin": source_type,
            "label_origin": "expert_review",
            "training_eligible": True,
            "hard_case_class": hard_case_class,
            "paired_state_group": None,
            "availability_profile": availability_profile,
            "validation_dimension_counts": {
                str(field_id): int(count)
                for field_id, count in (validation_dimension_counts or {}).items()
            },
            "review_status": "approved",
            "review_method": "expert_rule_v1",
        },
    }


def _pair_issues(records: Iterable[DecisionSftRecord]) -> dict[str, list[str]]:
    groups: defaultdict[str, list[DecisionSftRecord]] = defaultdict(list)
    for record in records:
        if record.metadata.paired_state_group:
            groups[record.metadata.paired_state_group].append(record)
    result: dict[str, list[str]] = defaultdict(list)
    for group_id, members in groups.items():
        if len(members) != 2:
            for member in members:
                result[member.sample_id].append("paired_group_size_invalid")
            continue
        left, right = sorted(members, key=lambda record: record.turn_index)
        left_state = left.state.model_dump(mode="json")
        right_state = right.state.model_dump(mode="json")
        if _canonical_json(left_state) == _canonical_json(right_state):
            result[left.sample_id].append("paired_states_identical")
            result[right.sample_id].append("paired_states_identical")
        if left.target.decision_reason == right.target.decision_reason:
            result[left.sample_id].append("paired_reason_ignores_state_change")
            result[right.sample_id].append("paired_reason_ignores_state_change")
        if left.metadata.hard_case_class == "heterogeneity_conflict" or right.metadata.hard_case_class == "heterogeneity_conflict":
            expected = {
                left.state.analysis_state.cross_project_validation.heterogeneity,
                right.state.analysis_state.cross_project_validation.heterogeneity,
            }
            if expected != {"low", "high"}:
                result[left.sample_id].append("heterogeneity_pair_missing_low_high")
                result[right.sample_id].append("heterogeneity_pair_missing_low_high")
            for member in (left, right):
                level = member.state.analysis_state.cross_project_validation.heterogeneity
                if level not in member.target.decision_reason:
                    result[member.sample_id].append("heterogeneity_fact_not_in_reason")
        # The generated pairs intentionally change one observation family plus
        # its deterministic progress projection.  A very large diff indicates
        # that a future generator changed several unrelated facts at once.
        changed = sum(
            left_state.get(block) != right_state.get(block)
            for block in STATE_BLOCKS
        )
        if changed > 3:
            result[left.sample_id].append("paired_state_changes_too_many_blocks")
            result[right.sample_id].append("paired_state_changes_too_many_blocks")
    return dict(result)


def _split_review_records(records: list[DecisionSftRecord]) -> dict[str, list[DecisionSftRecord]]:
    target_counts = _review_target_counts(len(records))
    groups: defaultdict[str, list[DecisionSftRecord]] = defaultdict(list)
    for record in records:
        key = record.metadata.paired_state_group or record.trajectory_id
        groups[key].append(record)

    ordered = sorted(
        groups.items(),
        key=lambda item: hashlib.sha256(item[0].encode("utf-8")).hexdigest(),
    )
    assigned: dict[str, str] = {}
    result: dict[str, list[DecisionSftRecord]] = {name: [] for name in target_counts}
    remaining = dict(target_counts)

    # Reserve at least one complete group for every important test bucket.
    for hard_case in sorted(REQUIRED_TEST_HARD_CASES):
        candidates = [
            (key, members)
            for key, members in ordered
            if key not in assigned
            and any(member.metadata.hard_case_class == hard_case for member in members)
            and len(members) <= remaining["test"]
        ]
        if not candidates:
            raise ValueError(f"REVIEW_TEST_BUCKET_UNAVAILABLE:{hard_case}")
        key, members = candidates[0]
        assigned[key] = "test"
        result["test"].extend(members)
        remaining["test"] -= len(members)

    for key, members in ordered:
        if key in assigned:
            continue
        size = len(members)
        possible = [name for name in target_counts if remaining[name] >= size]
        if not possible:
            raise ValueError(f"REVIEW_SPLIT_CAPACITY_UNMET:{remaining}")
        chosen = max(possible, key=lambda name: (remaining[name], name == "test"))
        assigned[key] = chosen
        result[chosen].extend(members)
        remaining[chosen] -= size

    if any(value != 0 for value in remaining.values()):
        raise ValueError(f"REVIEW_SPLIT_CAPACITY_UNMET:{remaining}")
    for values in result.values():
        values.sort(key=lambda record: record.sample_id)
    return result


def _near_duplicate_audit(records: list[DecisionSftRecord], splits: Mapping[str, list[DecisionSftRecord]]) -> dict[str, Any]:
    buckets: defaultdict[str, list[DecisionSftRecord]] = defaultdict(list)
    for record in records:
        if record.metadata.paired_state_group:
            continue
        buckets[_coarse_state_signature(record)].append(record)
    groups = {key: members for key, members in buckets.items() if len(members) > 1}
    split_of = {
        record.sample_id: split
        for split, values in splits.items()
        for record in values
    }
    cross_split = {
        key: sorted({split_of[member.sample_id] for member in members})
        for key, members in groups.items()
        if len({split_of[member.sample_id] for member in members}) > 1
    }
    return {
        "near_duplicate_count": sum(len(members) - 1 for members in groups.values()),
        "near_duplicate_groups": {
            key: [member.sample_id for member in members]
            for key, members in list(sorted(groups.items()))[:20]
        },
        "cross_split_groups": cross_split,
    }


def _compact_review_example(record: DecisionSftRecord) -> dict[str, Any]:
    state = record.state
    data = state.data_state
    analysis = state.analysis_state
    return {
        "sample_id": record.sample_id,
        "source_type": record.source_type,
        "hard_case_class": record.metadata.hard_case_class,
        "available_actions": list(state.action_space.available_actions),
        "key_facts": {
            "has_tabular_data": data.has_tabular_data,
            "row_count": data.row_count,
            "group_count": data.group_state.group_count,
            "project_count": data.project_state.project_count,
            "group_comparison_status": analysis.group_comparison.status,
            "confounder_status": analysis.confounder_adjustment.status,
            "cross_project_status": analysis.cross_project_validation.status,
            "heterogeneity": analysis.cross_project_validation.heterogeneity,
            "evidence_status": state.evidence_state.status,
            "remaining_objectives": list(state.progress.remaining_objectives),
        },
        "decision_reason": record.target.decision_reason,
    }


def _review_examples(records: Iterable[DecisionSftRecord], *, per_action: int = 3) -> dict[str, list[dict[str, Any]]]:
    """Expose compact, non-sensitive examples for the audit report.

    The full records remain in ``reviewed_v1.jsonl``.  The report only needs
    enough State facts to let a reviewer spot-check every fixed Action without
    copying raw rows, identifiers, or provenance into model input.
    """

    result: dict[str, list[dict[str, Any]]] = {action: [] for action in ALL_SCIENTIFIC_ACTIONS}
    for record in sorted(records, key=lambda value: value.sample_id):
        action = record.target.selected_action
        if len(result[action]) >= per_action:
            continue
        result[action].append(_compact_review_example(record))
    return result


def _review_hard_case_examples(
    records: Iterable[DecisionSftRecord], *, per_case: int = 3
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {hard: [] for hard in HARD_CASE_CLASSES}
    for record in sorted(records, key=lambda value: value.sample_id):
        hard = record.metadata.hard_case_class
        if hard is None or len(result[hard]) >= per_case:
            continue
        result[hard].append(_compact_review_example(record))
    return result


def review_decision_sft_v1(
    input_dir: Path,
    *,
    catalog_path: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Review, supplement, revalidate and split the existing v1 records."""

    output_dir = output_dir or input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog = _load_catalog(catalog_path)
    source_path = input_dir / "approved.jsonl"
    original = _read_records(source_path)
    if len(original) != 480:
        raise ValueError(f"REVIEW_EXPECTED_480_AUTO_VALID:{len(original)}")

    # Preserve the pre-review input byte-for-byte at the record level.
    _dump_jsonl(output_dir / "candidate_v1.jsonl", (record.model_dump(mode="json") for record in original))

    pair_issues = _pair_issues(original)
    review_entries: list[dict[str, Any]] = []
    revised_records: list[DecisionSftRecord] = []
    verdict_counts: Counter[str] = Counter()
    revision_counts: Counter[str] = Counter()
    rejected_records: list[dict[str, Any]] = []

    def review_one(record: DecisionSftRecord, *, is_new: bool = False) -> None:
        original_target = record.target.model_dump(mode="json")
        original_metadata = record.metadata.model_dump(mode="json")
        issues = _semantic_review(record, catalog)
        issues.extend(pair_issues.get(record.sample_id, []))
        issues = list(dict.fromkeys(issues))
        revised = record
        revision_reason: str | None = None
        metadata_changed = False

        # The generated adjustment examples used an inaccurate hard-case tag:
        # the label itself is valid, but it is not an objective/action
        # confusion.  Correct only the offline taxonomy, never the State or
        # scientific target.
        if (
            record.metadata.hard_case_class == "objective_action_confusion"
            and record.target.selected_action == "adjust_confounders"
        ):
            payload = record.model_dump(mode="json")
            payload["metadata"]["hard_case_class"] = None
            revised = DecisionSftRecord.model_validate(payload)
            metadata_changed = True
            revision_counts["hard_case_reclassification"] += 1
            revision_reason = "The target is a valid confounder Action; only the inaccurate objective/action hard-case tag was removed."
            issues.append("hard_case_reclassified")

        if any(issue.startswith("redundant_completed_action:") for issue in issues):
            replacement = _replacement_action(revised)
            if replacement is None:
                rejected_records.append({"sample_id": record.sample_id, "reason": "no_safe_replacement_for_redundant_action"})
            else:
                revised = _rewrite_target(revised, replacement)
                revision_reason = revision_reason or f"The original label repeated a completed action; revised to executable {replacement}."
                issues.append("target_revised_for_redundancy")

        # Repair only closed-contract presentation errors; a strategy choice
        # that is unavailable remains a reject rather than being guessed by
        # this reviewer.
        if "alternative_action_unavailable" in issues:
            revised = _rewrite_target(revised)
            revision_reason = revision_reason or "Removed alternatives that were not executable in the current State."
            issues.append("alternatives_revised")
        if "ungrounded_reason" in issues:
            revised = _rewrite_target(revised)
            revision_reason = revision_reason or "Rewrote the rationale from State facts without changing the Action."
            issues.append("reason_revised")
        if "finish_contract_violation" in issues:
            revised = _rewrite_target(revised)
            revision_reason = revision_reason or "Repaired the closed finish/stop_reason shape."
            issues.append("finish_contract_revised")

        post_issues = _semantic_review(revised, catalog)
        post_issues.extend(pair_issues.get(revised.sample_id, []))
        post_issues = list(dict.fromkeys(post_issues))
        hard_reject_markers = {
            "availability_state_mismatch",
            "selected_action_unavailable",
            "provenance_mismatch",
            "paired_state_group_missing",
            "paired_states_identical",
            "paired_group_size_invalid",
            "paired_state_changes_too_many_blocks",
            "remaining_objectives_mismatch",
        }
        if post_issues and any(issue in hard_reject_markers or issue.startswith((
            "completed_",
            "no_tabular_data_",
            "tabular_data_",
            "project_",
            "group_",
            "evidence_",
            "finish_with_",
            "finish_not_",
            "nonfinish_",
            "missing_",
            "insufficient_",
            "premature_",
            "missed_",
            "heterogeneity_",
        )) for issue in post_issues):
            verdict = "REJECT"
            rejected_records.append({"sample_id": record.sample_id, "reason": post_issues})
        elif post_issues:
            verdict = "REVISE"
        elif metadata_changed or revised.target != record.target:
            verdict = "REVISE"
        else:
            verdict = "PASS"

        if verdict == "REJECT":
            revision_counts["state_rejected"] += 1
        verdict_counts[verdict] += 1
        if revised.target.selected_action != record.target.selected_action:
            revision_counts["action_changed"] += 1
        elif revised.target.decision_reason != record.target.decision_reason:
            revision_counts["reason_only_revision"] += 1
        if revised.target.alternative_actions != record.target.alternative_actions:
            revision_counts["alternative_revision"] += 1
        if revised.target.stop_reason != record.target.stop_reason:
            revision_counts["finish_contract_revision"] += 1

        final_issues = post_issues or (["metadata_reclassified"] if metadata_changed else issues if verdict != "PASS" else [])
        entry = {
            "sample_id": record.sample_id,
            "source_type": record.source_type,
            "trajectory_id": record.trajectory_id,
            "turn_index": record.turn_index,
            "original_target": original_target,
            "review_verdict": verdict,
            "review_issues": final_issues,
            "revised_target": revised.target.model_dump(mode="json"),
            "revision_reason": revision_reason,
            "original_metadata": original_metadata,
            "revised_metadata": revised.metadata.model_dump(mode="json"),
            "second_pass_verdict": None,
            "second_pass_issues": [],
            "new_sample": is_new,
        }
        review_entries.append(entry)
        if verdict != "REJECT":
            revised_records.append(revised)

    for record in original:
        review_one(record)
    additions = _supplemental_records(catalog)
    for record in additions:
        review_one(record, is_new=True)

    if rejected_records:
        # A rejected State must not silently disappear.  The final dataset may
        # continue only after the rejection is visible in the report.
        _dump_json(output_dir / "review_rejected.json", rejected_records)

    # Deduplicate again after metadata/target revision and supplementation.
    unique: dict[str, DecisionSftRecord] = {}
    duplicate_ids: list[dict[str, str]] = []
    for record in revised_records:
        key = normalized_state_hash(record)
        if key in unique:
            duplicate_ids.append({"sample_id": record.sample_id, "duplicate_of": unique[key].sample_id})
            continue
        unique[key] = record
    final_records = list(unique.values())
    if duplicate_ids:
        _dump_json(output_dir / "review_duplicates.json", duplicate_ids)
    splits = _split_review_records(final_records)
    _dump_jsonl(output_dir / "reviewed_v1.jsonl", (record.model_dump(mode="json") for record in final_records))
    for split, values in splits.items():
        _dump_jsonl(output_dir / f"reviewed_{split}.jsonl", (record.model_jsonl_record() for record in values))

    # High-risk second pass is an independent re-read of the revised records;
    # it does not use the first-pass verdict as evidence.
    second_pass_counts: Counter[str] = Counter()
    by_id = {record.sample_id: record for record in final_records}
    for entry in review_entries:
        record = by_id.get(entry["sample_id"])
        original_hard_case = entry.get("original_metadata", {}).get("hard_case_class")
        needs_second_pass = (
            entry["review_verdict"] == "REVISE"
            or (record is not None and record.metadata.hard_case_class in HIGH_RISK_HARD_CASES)
            or original_hard_case in HIGH_RISK_HARD_CASES
        )
        if record is None or not needs_second_pass:
            continue
        second = list(dict.fromkeys(_semantic_review(record, catalog) + pair_issues.get(record.sample_id, [])))
        entry["second_pass_verdict"] = "PASS" if not second else "REJECT"
        entry["second_pass_issues"] = second
        second_pass_counts[entry["second_pass_verdict"]] += 1

    # Post-review deterministic validation and privacy/contract checks.
    post_validation_issues: list[dict[str, Any]] = []
    final_hashes: set[str] = set()
    model_violations: Counter[str] = Counter()
    for record in final_records:
        _, issues = validate_record(record.model_dump(mode="json"), catalog=catalog)
        if issues:
            post_validation_issues.append({"sample_id": record.sample_id, "issues": issues})
        key = normalized_state_hash(record)
        if key in final_hashes:
            model_violations["exact_duplicate"] += 1
        final_hashes.add(key)
        model_record = record.model_jsonl_record()
        if tuple(model_record["input"]["state"]) != STATE_BLOCKS:
            model_violations["state_blocks"] += 1
        if _walk_keys(model_record["input"]) or _walk_keys(model_record["output"]):
            model_violations["forbidden_keys"] += 1
        if not record.metadata.training_eligible:
            model_violations["training_ineligible"] += 1

    near_duplicates = _near_duplicate_audit(final_records, splits)
    leakage = _split_leakage(splits)
    hard_distribution = Counter(
        record.metadata.hard_case_class
        for record in final_records
        if record.metadata.hard_case_class
    )
    source_distribution = Counter(record.source_type for record in final_records)
    action_distribution = Counter(record.target.selected_action for record in final_records)
    frozen_test_hard = {
        record.metadata.hard_case_class
        for record in splits["test"]
        if record.metadata.hard_case_class
    }
    second_pass_ok = all(
        entry["second_pass_verdict"] == "PASS"
        for entry in review_entries
        if entry["second_pass_verdict"] is not None
    )
    final_ids = {record.sample_id for record in final_records}
    rejected_ids = {item["sample_id"] for item in rejected_records}
    checks = {
        "reviewed_at_least_480": len(original) == 480,
        "final_count_500_520": 500 <= len(final_records) <= 520,
        "all_final_review_entries_pass_or_revise": all(
            entry["review_verdict"] in {"PASS", "REVISE"}
            for entry in review_entries
            if entry["sample_id"] in final_ids
        ),
        "no_rejected_final_records": not (final_ids & rejected_ids),
        "rejected_records_excluded": not (final_ids & rejected_ids),
        "post_review_contract_valid": not post_validation_issues,
        "fixed_action_set": set(action_distribution).issubset(set(ALL_SCIENTIFIC_ACTIONS)) and all(action_distribution[action] >= 15 for action in ALL_SCIENTIFIC_ACTIONS),
        "selected_action_available_100pct": not any("selected_action_unavailable" in item["issues"] for item in post_validation_issues),
        "alternative_actions_available_100pct": not any("alternative_action_unavailable" in item["issues"] for item in post_validation_issues),
        "finish_contract_100pct": not any("finish_contract_violation" in item["issues"] for item in post_validation_issues),
        "old_fields_absent": not model_violations.get("forbidden_keys"),
        "raw_rows_absent": not model_violations.get("forbidden_keys"),
        "raw_identity_absent": not model_violations.get("forbidden_keys"),
        "exact_duplicates_zero": not duplicate_ids and not model_violations.get("exact_duplicate"),
        "near_duplicates_zero": near_duplicates["near_duplicate_count"] == 0,
        "near_duplicate_split_leakage_zero": not near_duplicates["cross_split_groups"],
        "sample_id_unique": len({record.sample_id for record in final_records}) == len(final_records),
        "pair_trajectory_split_leakage_zero": leakage["overlap_count"] == 0,
        "frozen_test_hard_cases_covered": REQUIRED_TEST_HARD_CASES.issubset(frozen_test_hard),
        "availability_boundary_at_least_30": hard_distribution["availability_boundary"] >= 30,
        "redundant_action_at_least_20": hard_distribution["redundant_action"] >= 20,
        "premature_finish_at_least_20": hard_distribution["premature_finish"] >= 20,
        "training_eligible_100pct": all(record.metadata.training_eligible for record in final_records),
        "high_risk_second_pass": second_pass_ok,
        "no_training_started": True,
    }
    ready = all(checks.values())
    review_audit = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "review_origin": "repository_agent",
        "external_model_calls": 0,
        "candidate_v1_count": len(original),
        "total_reviewed": len(review_entries),
        "pass": verdict_counts["PASS"],
        "revise": verdict_counts["REVISE"],
        "reject": verdict_counts["REJECT"],
        "new_samples_added": len(additions),
        "final_approved": len(final_records),
        "action_changed_by_review": revision_counts["action_changed"],
        "reason_only_revision": revision_counts["reason_only_revision"],
        "alternative_action_revision": revision_counts["alternative_revision"],
        "finish_contract_revision": revision_counts["finish_contract_revision"],
        "hard_case_reclassification": revision_counts["hard_case_reclassification"],
        "state_rejected": revision_counts["state_rejected"],
        "source_type_distribution": {source: source_distribution.get(source, 0) for source in SOURCE_TYPES},
        "action_distribution": {action: action_distribution.get(action, 0) for action in ALL_SCIENTIFIC_ACTIONS},
        "hard_case_distribution": {hard: hard_distribution.get(hard, 0) for hard in HARD_CASE_CLASSES},
        "split_distribution": {split: len(values) for split, values in splits.items()},
        "split_target_distribution": _review_target_counts(len(final_records)),
        "review_rejection_reasons": dict(Counter(item["reason"] if isinstance(item["reason"], str) else "state_or_pair_issue" for item in rejected_records)),
        "post_review_validation_issues": post_validation_issues,
        "duplicate_count_after_review": len(duplicate_ids),
        "near_duplicate_audit": near_duplicates,
        "pair_trajectory_split_leakage": leakage,
        "frozen_test_hard_cases": sorted(frozen_test_hard),
        "examples_by_action": _review_examples(final_records),
        "examples_by_hard_case": _review_hard_case_examples(final_records),
        "second_pass": {
            "hard_case_count": sum(second_pass_counts.values()),
            "pass": second_pass_counts["PASS"],
            "reject": second_pass_counts["REJECT"],
        },
        "review_process": [
            "deterministic contract validation",
            "repository-agent full semantic review",
            "high-risk bucket second-pass adjudication",
            "deterministic post-review validation",
        ],
        "checks": checks,
        "training_started": False,
        "training_eligible": ready,
        "DECISION_SFT_V1_DATA_READY": ready,
        "artifacts": {
            "candidate_v1": "candidate_v1.jsonl",
            "review_report": "review_report.jsonl",
            "reviewed_v1": "reviewed_v1.jsonl",
            "train": "reviewed_train.jsonl",
            "validation": "reviewed_validation.jsonl",
            "frozen_test": "reviewed_test.jsonl",
            "manifest": "review_manifest.json",
        },
    }
    manifest = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "training_started": False,
        "review_origin": "repository_agent",
        "external_model_calls": 0,
        "reviewed_count": len(review_entries),
        "final_approved": len(final_records),
        "split_counts": {split: len(values) for split, values in splits.items()},
        "state_blocks": list(STATE_BLOCKS),
        "actions": list(ALL_SCIENTIFIC_ACTIONS),
        "pair_and_trajectory_disjoint": leakage["overlap_count"] == 0,
        "near_duplicate_split_disjoint": not near_duplicates["cross_split_groups"],
        "training_eligible": ready,
        "DECISION_SFT_V1_DATA_READY": ready,
    }
    _dump_jsonl(output_dir / "review_report.jsonl", review_entries)
    _dump_json(output_dir / "review_audit.json", review_audit)
    _dump_json(output_dir / "review_manifest.json", manifest)
    return review_audit


__all__ = [
    "HIGH_RISK_HARD_CASES",
    "REVIEW_SCHEMA_VERSION",
    "review_decision_sft_v1",
]
