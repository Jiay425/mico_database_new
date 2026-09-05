"""Decision-State-native SFT v1 data engineering.

This module freezes the serving-shaped ``ScientificDecisionState`` ->
``DecisionPolicyOutput`` contract before producing candidates.  It is an
offline builder only: it never starts a model, calls an LLM, or launches a
training job.

The three rules below are intentionally repeated in code and in the dataset
README:

1. Objective is not Action.
2. The selected action must be executable in the current state.
3. Training state must match serving state.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

from pydantic import Field, ValidationError, field_validator

from mico_agent_runtime.contracts.base import ClosedModel, Identifier
from mico_agent_runtime.contracts.decision_state import (
    ScientificDecisionState,
)
from mico_agent_runtime.ports.decision_policy import DecisionPolicyOutput
from mico_agent_runtime.runtime.action_availability import (
    ALL_SCIENTIFIC_ACTIONS,
    ActionAvailabilityContext,
    compute_available_actions,
)
from mico_agent_runtime.contracts.schema_catalog import SchemaSemanticCatalog


DECISION_SFT_SCHEMA_VERSION = "decision-sft-v1"
MODEL_RECORD_SCHEMA_VERSION = "decision-sft-v1-model-jsonl"
SOURCE_TYPES = (
    "real_trace",
    "controlled_state",
    "boundary_state",
    "paired_state",
    "repair_derived",
)
HARD_CASE_CLASSES = (
    "objective_action_confusion",
    "availability_boundary",
    "redundant_action",
    "premature_finish",
    "missed_finish",
    "observation_sensitivity",
    "heterogeneity_conflict",
    "missing_evidence",
    "missing_project_dimension",
    "missing_covariate",
    "insufficient_data",
)
STATE_BLOCKS = (
    "task",
    "data_state",
    "analysis_state",
    "evidence_state",
    "progress",
    "action_space",
)
LEGACY_OR_LEAKAGE_KEYS = frozenset(
    {
        "goal_code",
        "observation_flags",
        "candidate_actions",
        "state_summary",
        "next_action_hint",
        "recommended_action",
        "required_action",
        "planner_hint",
        "strategy_hint",
        "should_adjust",
        "should_adjust_confounders",
        "raw_rows",
        "rows",
        "patient_id",
        "sample_id_value",
        "source_sample_id",
        "analysis_sample_key",
    }
)
_GENERIC_REASON_RE = re.compile(
    r"(?i)^(?:the\s+)?(?:next\s+)?(?:appropriate|best|logical)\s+action\s+is\s+\w+[.!]?$"
)

class DecisionSftMetadata(ClosedModel):
    """Dataset-only provenance; this object is never exported to the model."""

    state_origin: Literal[
        "real_trace",
        "controlled_state",
        "boundary_state",
        "paired_state",
        "repair_derived",
    ]
    label_origin: Literal["verified_trace", "expert_review", "llm_proposal_reviewed"]
    training_eligible: bool
    hard_case_class: Literal[
        "objective_action_confusion",
        "availability_boundary",
        "redundant_action",
        "premature_finish",
        "missed_finish",
        "observation_sensitivity",
        "heterogeneity_conflict",
        "missing_evidence",
        "missing_project_dimension",
        "missing_covariate",
        "insufficient_data",
    ] | None = None
    paired_state_group: Identifier | None = None
    availability_profile: Literal["catalog_knowledge", "catalog_no_knowledge"] = "catalog_knowledge"
    # Controlled states may explicitly carry the observed-value cardinality
    # of an independent validation dimension.  This is provenance only; it is
    # never exported in the model-facing JSONL record.
    validation_dimension_counts: dict[str, int] = Field(default_factory=dict)
    review_status: Literal["approved", "pending", "rejected"] = "approved"
    review_method: Literal["expert_rule_v1"] = "expert_rule_v1"


class DecisionSftRecord(ClosedModel):
    """One auditable single-step State -> Action training example."""

    schema_version: Literal["decision-sft-v1"] = DECISION_SFT_SCHEMA_VERSION
    sample_id: Identifier
    source_type: Literal[
        "real_trace",
        "controlled_state",
        "boundary_state",
        "paired_state",
        "repair_derived",
    ]
    trajectory_id: Identifier
    turn_index: int = Field(strict=True, ge=0)
    state: ScientificDecisionState
    target: DecisionPolicyOutput
    metadata: DecisionSftMetadata

    @field_validator("trajectory_id")
    @classmethod
    def trajectory_must_be_nonempty(cls, value: str) -> str:
        return value

    def model_input(self) -> dict[str, Any]:
        """Return the exact serving-shaped input, without provenance."""

        state = self.state.model_dump(mode="json")
        if tuple(state) != STATE_BLOCKS:
            raise ValueError("DECISION_SFT_STATE_BLOCKS_CHANGED")
        return {"decision_type": "scientific_action", "state": state}

    def model_output(self) -> dict[str, Any]:
        return self.target.model_dump(mode="json")

    def model_jsonl_record(self) -> dict[str, Any]:
        return {"input": self.model_input(), "output": self.model_output()}


def _walk_keys(value: Any, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text in LEGACY_OR_LEAKAGE_KEYS:
                found.append(f"{path}.{key_text}")
            found.extend(_walk_keys(child, f"{path}.{key_text}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_walk_keys(child, f"{path}[{index}]"))
    return found


def _state_fact_tokens(state: ScientificDecisionState) -> set[str]:
    """Return safe, observable State facts usable for rationale grounding.

    The previous implementation only listed a few status fields.  That made
    perfectly grounded pair examples such as ``heterogeneity=low`` fail the
    validator because neither the metric key nor its enum value was included.
    Walking the already-validated Decision State keeps this check generic
    while still excluding raw rows/identities (those cannot exist in the
    contract).  Only scalar leaves are retained.  Field names such as
    ``data_state`` are deliberately excluded because matching a generic word
    like ``state`` would otherwise let boilerplate rationale pass.
    """
    tokens: set[str] = set()

    def collect(value: Any) -> None:
        if isinstance(value, Mapping):
            for child in value.values():
                collect(child)
            return
        if isinstance(value, list):
            for child in value:
                collect(child)
            return
        if value is not None:
            tokens.add(str(value))

    collect(state.model_dump(mode="json"))
    return {token.lower() for token in tokens if token}


def _reason_is_grounded(state: ScientificDecisionState, reason: str) -> bool:
    normalized = " ".join(reason.lower().split())
    if _GENERIC_REASON_RE.fullmatch(normalized):
        return False
    if len(normalized) < 24:
        return False
    # Require at least one observable State fact.  The builder emits several
    # such facts; this also rejects a bulk template such as "choose X next".
    for token in _state_fact_tokens(state):
        if len(token) < 3:
            continue
        # Facts such as ``true`` or ``low`` should not match as a substring of
        # an unrelated English word.  Semantic IDs containing ``.`` remain
        # supported by the same lightweight boundary check.
        pattern = rf"(?<![a-z0-9_]){re.escape(token)}(?![a-z0-9_])"
        if re.search(pattern, normalized):
            return True
    return False


def _availability_context(
    metadata: DecisionSftMetadata,
    catalog: SchemaSemanticCatalog,
) -> ActionAvailabilityContext:
    return ActionAvailabilityContext(
        catalog=catalog,
        knowledge_available=metadata.availability_profile == "catalog_knowledge",
        distinct_counts=metadata.validation_dimension_counts,
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalized_state_hash(record: DecisionSftRecord) -> str:
    """Hash the full State and target, excluding provenance/sample identity."""

    payload = {"state": record.state.model_dump(mode="json"), "target": record.target.model_dump(mode="json")}
    return "sha256:" + hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def state_only_hash(record: DecisionSftRecord) -> str:
    return "sha256:" + hashlib.sha256(
        _canonical_json(record.state.model_dump(mode="json")).encode("utf-8")
    ).hexdigest()


def _coarse_state_signature(record: DecisionSftRecord) -> str:
    """Build an audit-only near-duplicate signature.

    Numeric values are bucketed, but statuses, dimensions, objectives and
    action availability remain visible.  Paired groups are excluded by the
    caller because their minimal difference is intentional.
    """

    payload = copy.deepcopy(record.state.model_dump(mode="json"))

    def normalize(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): normalize(child) for key, child in sorted(value.items())}
        if isinstance(value, list):
            return [normalize(child) for child in value]
        if isinstance(value, bool) or value is None:
            return value
        if isinstance(value, int):
            return f"int_bucket_{value // 10}"
        if isinstance(value, float):
            return f"float_bucket_{round(value, 1)}"
        if isinstance(value, str):
            return " ".join(value.lower().split())
        return value

    return "sha256:" + hashlib.sha256(_canonical_json(normalize(payload)).encode("utf-8")).hexdigest()


def validate_record(
    raw: Mapping[str, Any],
    *,
    catalog: SchemaSemanticCatalog,
) -> tuple[DecisionSftRecord | None, list[str]]:
    """Validate one raw candidate and return closed error codes.

    Validation is intentionally stricter than Pydantic alone: availability is
    recomputed from the actual Catalog, and the model-facing state is checked
    for legacy hints, raw rows and generic rationale text.
    """

    if not isinstance(raw, Mapping):
        return None, ["schema_invalid"]
    issues: list[str] = []
    state_raw = raw.get("state")
    target_raw = raw.get("target")
    legacy_paths = _walk_keys(state_raw) + _walk_keys(target_raw)
    if legacy_paths:
        if any(path.endswith(".rows") or path.endswith(".raw_rows") for path in legacy_paths):
            issues.append("raw_rows")
        if any(path.endswith(".patient_id") or path.endswith(".analysis_sample_key") for path in legacy_paths):
            issues.append("pii_or_identity")
        if any(
            path.rsplit(".", 1)[-1]
            in {"goal_code", "observation_flags", "candidate_actions", "state_summary", "next_action_hint", "recommended_action", "required_action", "planner_hint", "strategy_hint", "should_adjust", "should_adjust_confounders"}
            for path in legacy_paths
        ):
            issues.append("legacy_or_leakage_field")
    if isinstance(target_raw, Mapping):
        selected_raw = target_raw.get("selected_action")
        if selected_raw in {
            "group_comparison",
            "projection_analysis",
            "confounder_assessment",
            "cross_project_validation",
            "cross_disease_validation",
            "evidence_support",
        }:
            issues.append("objective_action_confusion")
    try:
        record = DecisionSftRecord.model_validate(raw)
    except ValidationError as exc:
        text = str(exc).lower()
        if "stop_reason" in text or "finish" in text:
            issues.append("finish_contract_violation")
        if "less than or equal to 0" in text or "greater than or equal to 0" in text:
            issues.append("negative_count")
        issues.append("schema_invalid")
        return None, list(dict.fromkeys(issues))
    except (TypeError, ValueError):
        issues.append("schema_invalid")
        return None, list(dict.fromkeys(issues))

    expected = compute_available_actions(record.state, _availability_context(record.metadata, catalog))
    actual = record.state.action_space.available_actions
    if record.metadata.state_origin != record.source_type:
        issues.append("provenance_mismatch")
    if record.source_type == "paired_state" and not record.metadata.paired_state_group:
        issues.append("paired_state_group_missing")
    if actual != expected:
        issues.append("availability_state_mismatch")
    if record.target.selected_action not in expected:
        issues.append("selected_action_unavailable")
    if any(action not in expected for action in record.target.alternative_actions):
        issues.append("alternative_action_unavailable")
    if record.target.selected_action == "finish":
        if record.target.stop_reason not in {
            "EVIDENCE_SUFFICIENT",
            "NO_NEW_INFORMATION",
            "QUALITY_RISK",
            "ACTION_BUDGET_EXHAUSTED",
            "UPSTREAM_REJECTED",
            "UNSUPPORTED_ACTION",
            "USER_REQUESTED_STOP",
        }:
            issues.append("finish_contract_violation")
    elif record.target.stop_reason is not None:
        issues.append("finish_contract_violation")
    if not _reason_is_grounded(record.state, record.target.decision_reason):
        issues.append("ungrounded_reason")
    if record.metadata.training_eligible is not True:
        issues.append("training_ineligible")
    if record.metadata.review_status != "approved":
        issues.append("review_not_approved")
    return record, list(dict.fromkeys(issues))


def _dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _dump_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values), encoding="utf-8")


def _load_catalog(catalog_path: Path) -> SchemaSemanticCatalog:
    return SchemaSemanticCatalog.model_validate(json.loads(catalog_path.read_text(encoding="utf-8")))


def _task_payload(query_variant: int, *, objectives: list[str] | None = None) -> dict[str, Any]:
    queries = (
        "比较 T2D 和 Healthy 的菌群差异，并评估年龄影响。",
        "判断 T2D 与 Healthy 的微生物丰度差异是否受年龄混杂。",
        "评估两组菌群差异在不同项目中的稳定性，并结合文献解释。",
        "比较疾病组的 abundance.value，并检查协变量和证据一致性。",
        "分析 T2D/Healthy 的特征差异、年龄调整结果与跨项目异质性。",
        "研究目标组之间的微生物差异是否可重复且有文献支持。",
        "确认两组观测差异，必要时进行分层和混杂因素评估。",
        "围绕疾病分组、年龄和项目稳定性完成结构化科研分析。",
    )
    target_feature_variants = (
        ["gut_microbiome"],
        ["microbial_abundance"],
        ["taxonomic_features"],
        ["abundance_profile"],
    )
    requested_stratifier_variants = (
        [],
        ["sample.age"],
        ["sample.gender"],
        ["sample.country"],
    )
    return {
        "query": queries[query_variant % len(queries)],
        "objectives": objectives or [
            "group_comparison",
            "confounder_assessment",
            "cross_project_validation",
            "evidence_support",
        ],
        "constraints": {
            "disease_groups": ["T2D", "Healthy"],
            "target_features": target_feature_variants[(query_variant // 8) % len(target_feature_variants)],
            "focus_covariates": ["age"],
            "requested_projects": [],
            "requested_stratifiers": requested_stratifier_variants[(query_variant // 32) % len(requested_stratifier_variants)],
        },
    }


def _default_analysis() -> dict[str, Any]:
    return {
        "group_comparison": {"status": "not_started"},
        "confounder_adjustment": {"status": "not_started"},
        "cross_project_validation": {"status": "not_started"},
        "stratified_analysis": {"status": "not_started"},
        "projection_analysis": {"status": "not_started"},
        "cross_disease_validation": {"status": "not_started"},
    }


def _data_payload(
    *,
    mode: str,
    variant: int,
    group_status: str = "not_started",
    adjustment_status: str = "not_started",
    cross_project_status: str = "not_started",
    stratified_status: str = "not_started",
    projection_status: str = "not_started",
    cross_disease_status: str = "not_started",
    evidence_status: str = "not_started",
    evidence_consistency: str = "unknown",
    adjusted_effect: float | None = None,
    group_effect: float | None = None,
    project_heterogeneity: str = "unknown",
    positive_projects: int = 0,
    negative_projects: int = 0,
    project_count: int = 0,
    covariates: list[str] | None = None,
    dimensions: list[str] | None = None,
    group_count: int = 2,
    group_sizes: dict[str, int] | None = None,
    knowledge_available: bool = True,
) -> tuple[dict[str, Any], str]:
    has_data = mode != "empty"
    if mode == "single_group":
        group_count = 1
        group_sizes = {"T2D": 20 + variant % 10}
    elif group_sizes is None:
        group_sizes = {
            "T2D": 40 + variant % 16,
            "Healthy": 42 + (variant * 3) % 17,
        }
    if mode == "empty":
        group_count = 0
        group_sizes = {}
    if dimensions is None:
        dimensions = [] if not has_data else ["sample.disease", "abundance.feature"]
        if covariates:
            dimensions.append("sample.age")
        if project_count:
            dimensions.append("metadata.project")
    outcome_fields = [] if not has_data else ["abundance.value"]
    row_count = 0 if not has_data else 6000 + variant * 37
    sample_count = None if not has_data else sum(group_sizes.values())
    feature_count = None if not has_data else 20 + variant % 75
    if has_data and "sample.disease" not in dimensions:
        dimensions = [*dimensions, "sample.disease"]
    group_field = "sample.disease" if group_count else None
    analysis = _default_analysis()
    analysis["group_comparison"].update({"status": group_status, "effect_size": group_effect})
    if group_effect is not None:
        analysis["group_comparison"].update({"mean_difference": group_effect, "p_value": 0.01 + (variant % 7) / 100})
    analysis["confounder_adjustment"].update({"status": adjustment_status, "adjusted_effect_size": adjusted_effect})
    if adjusted_effect is not None:
        analysis["confounder_adjustment"].update({
            "adjusted_group_effect": adjusted_effect,
            "adjusted_p_value": 0.02 + (variant % 8) / 100,
            "adjusted_covariates": list(covariates or []),
            "used_row_count": row_count,
            "dropped_row_count": variant % 5,
        })
    analysis["cross_project_validation"].update({
        "status": cross_project_status,
        "project_count": project_count,
        "positive_project_count": positive_projects,
        "negative_project_count": negative_projects,
        "neutral_project_count": max(0, project_count - positive_projects - negative_projects),
        "effect_min": -0.6 if negative_projects else 0.4 if project_count else None,
        "effect_max": 0.7 if project_count else None,
        "heterogeneity": project_heterogeneity,
    })
    analysis["stratified_analysis"].update({
        "status": stratified_status,
        "stratify_fields": ["sample.age"] if covariates else [],
        "stratum_count": 2 if covariates else 0,
        "heterogeneity": "medium" if stratified_status == "completed" else "unknown",
    })
    analysis["projection_analysis"].update({"status": projection_status, "result_count": 2 if projection_status == "completed" else 0})
    analysis["cross_disease_validation"].update({"status": cross_disease_status, "disease_count": group_count, "heterogeneity": "low" if cross_disease_status == "completed" else "unknown"})
    task = _task_payload(variant)
    completed_actions: list[str] = []
    action_counts: dict[str, int] = {}
    completion_pairs = (
        ("execute_read_query", True, has_data),
        ("compare_groups", group_status == "completed", group_status == "completed"),
        ("adjust_confounders", adjustment_status == "completed", adjustment_status == "completed"),
        ("cross_project_validate", cross_project_status == "completed", cross_project_status == "completed"),
        ("stratified_analysis", stratified_status == "completed", stratified_status == "completed"),
        ("analyze_projection", projection_status == "completed", projection_status == "completed"),
        ("cross_disease_validate", cross_disease_status == "completed", cross_disease_status == "completed"),
        ("retrieve_evidence", evidence_status == "completed", evidence_status == "completed"),
    )
    for action, completed, countable in completion_pairs:
        if completed and countable:
            completed_actions.append(action)
            action_counts[action] = 1
    # Preserve a realistic execution order for a state that has several facts.
    order = {action: index for index, action in enumerate(ALL_SCIENTIFIC_ACTIONS)}
    completed_actions.sort(key=lambda action: order[action])
    objective_to_done = {
        "group_comparison": group_status == "completed",
        "confounder_assessment": adjustment_status == "completed",
        "cross_project_validation": cross_project_status == "completed",
        "stratified_analysis": stratified_status == "completed",
        "projection_analysis": projection_status == "completed",
        "cross_disease_validation": cross_disease_status == "completed",
        "evidence_support": evidence_status == "completed",
    }
    remaining = [objective for objective in task["objectives"] if not objective_to_done.get(objective, False)]
    last_action = completed_actions[-1] if completed_actions else None
    payload = {
        "task": task,
        "data_state": {
            "has_tabular_data": has_data,
            "row_count": row_count,
            "sample_count": sample_count,
            "patient_count": None,
            "feature_count": feature_count,
            "available_dimensions": list(dict.fromkeys(dimensions)),
            "available_outcomes": outcome_fields,
            "group_state": {"group_field": group_field, "group_count": group_count, "group_sizes": group_sizes},
            "project_state": {"has_project_field": bool(project_count), "project_count": project_count},
            "covariate_state": {"available_covariates": list(covariates or []), "imbalance": {field: "high" if variant % 3 == 0 else "low" for field in (covariates or [])}},
            "data_quality": {"missingness_level": "low" if has_data else "unknown", "sample_size_level": "sufficient" if has_data and sum(group_sizes.values()) >= 40 else "insufficient" if has_data else "unknown"},
        },
        "analysis_state": analysis,
        "evidence_state": {
            "status": evidence_status,
            "evidence_count": 6 if evidence_status == "completed" else 0,
            "support_count": 4 if evidence_status == "completed" else 0,
            "conflict_count": 1 if evidence_status == "completed" else 0,
            "context_count": 1 if evidence_status == "completed" else 0,
            "consistency": evidence_consistency,
        },
        "progress": {
            "completed_actions": completed_actions,
            "action_counts": action_counts,
            "last_action": last_action,
            "remaining_objectives": remaining,
            "action_count": sum(action_counts.values()),
        },
        "action_space": {"available_actions": []},
    }
    profile = "catalog_knowledge" if knowledge_available else "catalog_no_knowledge"
    return payload, profile


def _build_state(
    *,
    catalog: SchemaSemanticCatalog,
    variant: int,
    mode: str,
    desired_action: str,
    knowledge_available: bool = True,
    **kwargs: Any,
) -> tuple[ScientificDecisionState, str]:
    validation_dimension_counts = kwargs.pop("validation_dimension_counts", {})
    if not isinstance(validation_dimension_counts, Mapping):
        raise TypeError("validation_dimension_counts must be a mapping")
    validation_dimension_counts = {
        str(field_id): int(count)
        for field_id, count in validation_dimension_counts.items()
    }
    payload, profile = _data_payload(
        mode=mode,
        variant=variant,
        knowledge_available=knowledge_available,
        **kwargs,
    )
    state = ScientificDecisionState.model_validate(payload)
    metadata = DecisionSftMetadata(
        state_origin="controlled_state",
        label_origin="expert_review",
        training_eligible=True,
        availability_profile=profile,
        validation_dimension_counts=validation_dimension_counts,
    )
    available = compute_available_actions(state, _availability_context(metadata, catalog))
    payload["action_space"]["available_actions"] = available
    state = ScientificDecisionState.model_validate(payload)
    if desired_action not in state.action_space.available_actions:
        raise ValueError(f"BUILDER_DESIRED_ACTION_UNAVAILABLE:{desired_action}")
    return state, profile


def _reason(action: str, state: ScientificDecisionState, hard_case_class: str | None = None) -> str:
    data = state.data_state
    analysis = state.analysis_state
    if hard_case_class in {"objective_action_confusion", "missing_project_dimension", "availability_boundary"} and action == "execute_read_query":
        if hard_case_class == "availability_boundary":
            if data.group_state.group_count < 2:
                return f"group_count={data.group_state.group_count} is below the two-group operator requirement, so compare_groups is unavailable; execute_read_query is the executable action for a new bounded observation."
            if not data.covariate_state.available_covariates:
                return "available_covariates=none while confounder_assessment remains in remaining_objectives; adjust_confounders is unavailable, so execute_read_query can obtain another bounded observation."
            if data.project_state.project_count < 2 and "cross_project_validation" in state.progress.remaining_objectives:
                return f"cross_project_validation remains in remaining_objectives, but project_count={data.project_state.project_count} is below the two-project requirement; execute_read_query is the executable action for a new bounded observation."
            return f"evidence_support remains in remaining_objectives while evidence_state.status={state.evidence_state.status}; execute_read_query is available even though the evidence route is unavailable in this state."
        return f"cross_project_validation remains in remaining_objectives, but metadata.project is absent and project_count={data.project_state.project_count}; execute_read_query is the executable action for a new bounded observation."
    if hard_case_class == "missing_covariate" and action in {"execute_read_query", "compare_groups"}:
        return f"confounder_assessment remains in remaining_objectives, but available_covariates={','.join(data.covariate_state.available_covariates) or 'none'}; {action} is executable without claiming an adjustment that the State cannot support."
    if hard_case_class == "missing_evidence" and action == "execute_read_query":
        return f"evidence_support remains in remaining_objectives while the evidence source is unavailable; evidence_state.status={state.evidence_state.status}, so execute_read_query is the available next observation."
    if hard_case_class == "insufficient_data" and action == "execute_read_query":
        return f"group_count={data.group_state.group_count} is below the two-group operator requirement, so compare_groups is not executable; execute_read_query can seek a better bounded observation."
    if hard_case_class == "premature_finish":
        return f"remaining_objectives={','.join(state.progress.remaining_objectives) or 'none'} and finish is not available; {action} must advance an executable unfinished objective instead of ending early."
    if hard_case_class == "redundant_action":
        status_by_action = {
            "compare_groups": analysis.group_comparison.status,
            "adjust_confounders": analysis.confounder_adjustment.status,
            "cross_project_validate": analysis.cross_project_validation.status,
            "stratified_analysis": analysis.stratified_analysis.status,
            "analyze_projection": analysis.projection_analysis.status,
            "cross_disease_validate": analysis.cross_disease_validation.status,
            "retrieve_evidence": state.evidence_state.status,
        }
        action_status = status_by_action.get(action, "not_started")
        return f"last_action={state.progress.last_action}, {action}.status={action_status}, group_comparison.status={analysis.group_comparison.status}, and action_count={state.progress.action_count}; {action} avoids repeating a completed observation and advances the current State."
    if hard_case_class == "heterogeneity_conflict":
        cross = analysis.cross_project_validation
        if action == "retrieve_evidence":
            return f"cross_project_validation.heterogeneity={cross.heterogeneity} with positive_project_count={cross.positive_project_count} and negative_project_count={cross.negative_project_count}; retrieve_evidence can ground the stability interpretation."
        return f"cross_project_validation.heterogeneity={cross.heterogeneity} with effect_min={cross.effect_min} and effect_max={cross.effect_max}; {action} can probe the conflicting project pattern."
    if hard_case_class == "observation_sensitivity" and action in {"retrieve_evidence", "stratified_analysis"}:
        adjusted = analysis.confounder_adjustment.adjusted_effect_size
        return f"adjusted_effect_size={adjusted}, confounder_adjustment.status={analysis.confounder_adjustment.status}, and remaining_objectives={','.join(state.progress.remaining_objectives)}; {action} responds to the updated observation."
    if action == "inspect_cohort":
        return f"The State has_tabular_data={data.has_tabular_data} and row_count={data.row_count}; inspect_cohort can establish a bounded cohort profile before interpretation."
    if action == "execute_read_query":
        return f"The current observation has row_count={data.row_count}, group_count={data.group_state.group_count}, and remaining objectives={','.join(state.progress.remaining_objectives)}; execute_read_query can obtain the next catalog-bounded observation."
    if action == "compare_groups":
        return f"group_field={data.group_state.group_field}, group_count={data.group_state.group_count}, available_outcomes={','.join(data.available_outcomes)}, and group_comparison.status={analysis.group_comparison.status}; compare_groups is executable now."
    if action == "analyze_projection":
        return f"available_outcomes={','.join(data.available_outcomes)} and projection_analysis.status={analysis.projection_analysis.status}; analyze_projection can summarize the observed projection without changing the State contract."
    if action == "stratified_analysis":
        return f"sample.age is an available covariate, group_count={data.group_state.group_count}, and stratified_analysis.status={analysis.stratified_analysis.status}; stratified_analysis can test the observed strata."
    if action == "adjust_confounders":
        return f"group_comparison.status={analysis.group_comparison.status}, available_covariates={','.join(data.covariate_state.available_covariates)}, and confounder_adjustment.status={analysis.confounder_adjustment.status}; adjust_confounders is executable."
    if action == "cross_project_validate":
        return f"metadata.project is present with project_count={data.project_state.project_count}, group_count={data.group_state.group_count}, and cross_project_validation.status={analysis.cross_project_validation.status}; cross_project_validate can assess project stability."
    if action == "cross_disease_validate":
        return f"group_field={data.group_state.group_field} has group_count={data.group_state.group_count}, and cross_disease_validation.status={analysis.cross_disease_validation.status}; cross_disease_validate can compare the observed disease dimension."
    if action == "retrieve_evidence":
        return f"evidence_state.status={state.evidence_state.status}, evidence_count={state.evidence_state.evidence_count}, and evidence_support remains in {','.join(state.progress.remaining_objectives)}; retrieve_evidence can add grounded context."
    if action == "finish":
        return f"remaining_objectives is empty, group_comparison.status={analysis.group_comparison.status}, and evidence consistency={state.evidence_state.consistency}; finish is executable with completed required work."
    raise ValueError(f"UNKNOWN_ACTION:{action}")


def _record(
    *,
    sample_index: int,
    trajectory_id: str,
    turn_index: int,
    state: ScientificDecisionState,
    source_type: str,
    state_origin: str,
    availability_profile: str,
    action: str,
    hard_case_class: str | None,
    paired_state_group: str | None = None,
    validation_dimension_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    target = DecisionPolicyOutput(
        selected_action=action,
        decision_reason=_reason(action, state, hard_case_class),
        alternative_actions=[],
        stop_reason="EVIDENCE_SUFFICIENT" if action == "finish" else None,
    )
    metadata = DecisionSftMetadata(
        state_origin=state_origin,
        label_origin="expert_review",
        training_eligible=True,
        hard_case_class=hard_case_class,
        paired_state_group=paired_state_group,
        availability_profile=availability_profile,
        validation_dimension_counts={
            str(field_id): int(count)
            for field_id, count in (validation_dimension_counts or {}).items()
        },
    )
    return {
        "schema_version": DECISION_SFT_SCHEMA_VERSION,
        "sample_id": f"decision-sft-v1-{sample_index:06d}",
        "source_type": source_type,
        "trajectory_id": trajectory_id,
        "turn_index": turn_index,
        "state": state.model_dump(mode="json"),
        "target": target.model_dump(mode="json"),
        "metadata": metadata.model_dump(mode="json"),
    }


def _build_pair_records(catalog: SchemaSemanticCatalog, start_index: int) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    index = start_index
    pair_number = 0
    for pair_kind in ("group_progression", "adjustment_progression", "effect_response", "heterogeneity"):
        for variant in range(10):
            pair_number += 1
            pair_id = f"pair-v1-{pair_number:03d}"
            trajectory_id = f"trajectory-pair-v1-{pair_number:03d}"
            common = {"variant": 100 + variant + pair_number * 3, "mode": "groups"}
            if pair_kind == "group_progression":
                left, profile = _build_state(catalog=catalog, desired_action="compare_groups", group_status="not_started", covariates=["sample.age"], **common)
                right, profile = _build_state(catalog=catalog, desired_action="adjust_confounders", group_status="completed", covariates=["sample.age"], group_effect=0.5, **common)
                hard = "observation_sensitivity"
            elif pair_kind == "adjustment_progression":
                left, profile = _build_state(catalog=catalog, desired_action="adjust_confounders", group_status="completed", covariates=["sample.age"], group_effect=0.58, **common)
                right, profile = _build_state(catalog=catalog, desired_action="stratified_analysis", group_status="completed", adjustment_status="completed", covariates=["sample.age"], group_effect=0.58, adjusted_effect=0.21, **common)
                hard = "observation_sensitivity"
            elif pair_kind == "effect_response":
                left, profile = _build_state(catalog=catalog, desired_action="retrieve_evidence", group_status="completed", adjustment_status="completed", covariates=["sample.age"], group_effect=0.62, adjusted_effect=0.59, **common)
                right, profile = _build_state(catalog=catalog, desired_action="stratified_analysis", group_status="completed", adjustment_status="completed", covariates=["sample.age"], group_effect=0.62, adjusted_effect=0.18, **common)
                hard = "observation_sensitivity"
            else:
                project_common = {**common, "mode": "project", "project_count": 4, "group_status": "completed", "adjustment_status": "completed", "covariates": ["sample.age"], "group_effect": 0.62, "adjusted_effect": 0.18, "cross_project_status": "completed", "positive_projects": 4, "negative_projects": 0, "project_heterogeneity": "low"}
                left, profile = _build_state(catalog=catalog, desired_action="retrieve_evidence", **project_common)
                project_conflict = {**project_common, "positive_projects": 2, "negative_projects": 2, "project_heterogeneity": "high"}
                right, profile = _build_state(catalog=catalog, desired_action="stratified_analysis", **project_conflict)
                hard = "heterogeneity_conflict"
            records.append(_record(sample_index=index, trajectory_id=trajectory_id, turn_index=1, state=left, source_type="paired_state", state_origin="paired_state", availability_profile=profile, action="compare_groups" if pair_kind == "group_progression" else "adjust_confounders" if pair_kind == "adjustment_progression" else "retrieve_evidence", hard_case_class=hard, paired_state_group=pair_id))
            index += 1
            records.append(_record(sample_index=index, trajectory_id=trajectory_id, turn_index=2, state=right, source_type="paired_state", state_origin="paired_state", availability_profile=profile, action="adjust_confounders" if pair_kind == "group_progression" else "stratified_analysis", hard_case_class=hard, paired_state_group=pair_id))
            index += 1
    return records, index


SINGLE_ACTION_COUNTS: dict[str, int] = {
    "inspect_cohort": 45,
    "execute_read_query": 65,
    "compare_groups": 45,
    "analyze_projection": 40,
    "stratified_analysis": 15,
    "adjust_confounders": 35,
    "cross_project_validate": 40,
    "cross_disease_validate": 40,
    "retrieve_evidence": 25,
    "finish": 50,
}


def _single_spec(action: str, variant: int) -> tuple[dict[str, Any], str, str | None, str, bool]:
    """Return state-builder kwargs, source type, hard class, profile, and knowledge flag."""

    # A deliberately explicit boundary block supplies the S3-style examples.
    if action == "execute_read_query" and variant < 15:
        return ({"mode": "groups", "group_status": "completed", "adjustment_status": "completed", "covariates": ["sample.age"], "group_effect": 0.4, "adjusted_effect": 0.2, "project_count": 0}, "boundary_state", "objective_action_confusion" if variant < 8 else "missing_project_dimension", "catalog_knowledge", True)
    if action == "execute_read_query" and variant < 25:
        return ({"mode": "single_group", "group_count": 1, "dimensions": ["sample.disease"], "covariates": []}, "boundary_state", "insufficient_data", "catalog_knowledge", True)
    if action == "execute_read_query" and variant < 35:
        return ({"mode": "groups", "group_status": "completed", "covariates": [], "project_count": 0}, "boundary_state", "missing_covariate", "catalog_knowledge", True)
    if action == "execute_read_query":
        if variant < 45:
            if variant < 40:
                # One observed project is a real availability boundary:
                # ``has_project_field`` is true, but cross-project validation
                # still requires at least two projects.  The label remains a
                # new executable read, never the unavailable validation
                # Action.
                return ({"mode": "project", "group_status": "completed", "adjustment_status": "completed", "covariates": ["sample.age"], "group_effect": 0.4, "adjusted_effect": 0.2, "project_count": 1}, "boundary_state", "availability_boundary", "catalog_knowledge", True)
            return ({"mode": "empty", "covariates": []}, "boundary_state", "availability_boundary", "catalog_no_knowledge", False)
        return ({"mode": "empty", "covariates": []}, "controlled_state", None, "catalog_knowledge", True)
    if action == "inspect_cohort":
        return ({"mode": "empty", "covariates": []}, "controlled_state", None, "catalog_knowledge", True)
    if action == "compare_groups" and variant < 12:
        return ({"mode": "groups", "group_status": "not_started", "covariates": []}, "boundary_state", "missing_covariate", "catalog_knowledge", True)
    if action == "compare_groups":
        return ({"mode": "groups", "group_status": "not_started", "covariates": ["sample.age"]}, "controlled_state", None, "catalog_knowledge", True)
    if action == "analyze_projection":
        return ({"mode": "groups", "projection_status": "not_started", "covariates": ["sample.age"]}, "controlled_state", None, "catalog_knowledge", True)
    if action == "stratified_analysis":
        return ({"mode": "groups", "group_status": "completed", "stratified_status": "not_started", "covariates": ["sample.age"], "group_effect": 0.4}, "controlled_state", None, "catalog_knowledge", True)
    if action == "adjust_confounders":
        if variant < 20:
            return ({"mode": "groups", "group_status": "completed", "covariates": ["sample.age"], "adjustment_status": "not_started", "group_effect": 0.62}, "repair_derived", "objective_action_confusion", "catalog_knowledge", True)
        return ({"mode": "groups", "group_status": "completed", "covariates": ["sample.age"], "adjustment_status": "not_started", "group_effect": 0.4}, "controlled_state", None, "catalog_knowledge", True)
    if action == "cross_project_validate":
        return ({"mode": "project", "group_status": "completed", "adjustment_status": "completed", "cross_project_status": "not_started", "project_count": 2 + variant % 6, "positive_projects": 0, "negative_projects": 0, "covariates": ["sample.age"], "group_effect": 0.5, "adjusted_effect": 0.25}, "controlled_state", None, "catalog_knowledge", True)
    if action == "cross_disease_validate":
        # Cross-disease validation is only executable when the state exposes
        # a second, independent disease dimension.  Keep that capability
        # explicit in the controlled fixture and record its observed
        # cardinality as dataset provenance; the serving State remains the
        # unchanged six-block contract.
        return ({
            "mode": "groups",
            "group_status": "completed",
            "cross_disease_status": "not_started",
            "covariates": ["sample.age"],
            "dimensions": [
                "sample.disease",
                "disease.name",
                "abundance.feature",
                "sample.age",
            ],
            "validation_dimension_counts": {"disease.name": 2},
            "group_effect": 0.3,
        }, "controlled_state", None, "catalog_knowledge", True)
    if action == "retrieve_evidence":
        if variant < 8:
            return ({"mode": "groups", "group_status": "completed", "adjustment_status": "completed", "covariates": ["sample.age"], "evidence_status": "not_started", "group_effect": 0.3, "adjusted_effect": 0.2}, "boundary_state", "missing_evidence", "catalog_knowledge", True)
        if variant < 16:
            return ({"mode": "groups", "group_status": "completed", "adjustment_status": "completed", "covariates": ["sample.age"], "evidence_status": "not_started", "group_effect": 0.3, "adjusted_effect": 0.2}, "controlled_state", "missing_evidence", "catalog_knowledge", True)
        return ({"mode": "groups", "group_status": "completed", "adjustment_status": "completed", "covariates": ["sample.age"], "evidence_status": "not_started", "group_effect": 0.3, "adjusted_effect": 0.2}, "controlled_state", None, "catalog_knowledge", True)
    if action == "finish":
        return ({"mode": "project", "group_status": "completed", "adjustment_status": "completed", "cross_project_status": "completed", "stratified_status": "completed", "projection_status": "completed", "cross_disease_status": "completed", "evidence_status": "completed", "evidence_consistency": "mostly_supportive", "project_count": 3 + variant % 5, "positive_projects": 3 + variant % 3, "covariates": ["sample.age"], "group_effect": 0.4, "adjusted_effect": 0.2}, "controlled_state", "missed_finish", "catalog_knowledge", True)
    raise ValueError(action)


def _build_single_records(catalog: SchemaSemanticCatalog, start_index: int) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    index = start_index
    global_variant = 0
    for action, count in SINGLE_ACTION_COUNTS.items():
        for variant in range(count):
            kwargs, source_type, hard_case, profile, knowledge = _single_spec(action, variant)
            # Dedicated redundancy and premature-finish cases are layered onto
            # otherwise executable alternatives, never onto the requested label.
            if global_variant % 17 == 0 and action in {"adjust_confounders", "retrieve_evidence", "execute_read_query"}:
                kwargs = {**kwargs, "group_status": "completed", "adjustment_status": "not_started", "covariates": ["sample.age"]}
                hard_case = "redundant_action"
            if global_variant % 23 == 0 and action in {"execute_read_query", "compare_groups", "retrieve_evidence"}:
                hard_case = "premature_finish"
            state, actual_profile = _build_state(catalog=catalog, variant=global_variant + 300, desired_action=action, knowledge_available=knowledge, **kwargs)
            # Ensure provenance profile is generated from the actual hard
            # availability context, not a handwritten action list.
            records.append(_record(
                sample_index=index,
                trajectory_id=f"trajectory-single-v1-{global_variant:04d}",
                turn_index=0,
                state=state,
                source_type=source_type,
                state_origin=source_type,
                availability_profile=actual_profile,
                action=action,
                hard_case_class=hard_case,
                validation_dimension_counts=kwargs.get("validation_dimension_counts"),
            ))
            index += 1
            global_variant += 1
    return records, index


def _invalid_candidates(valid_records: list[dict[str, Any]], start_index: int) -> list[dict[str, Any]]:
    invalid: list[dict[str, Any]] = []

    def clone(position: int) -> dict[str, Any]:
        value = copy.deepcopy(valid_records[position % len(valid_records)])
        value["sample_id"] = f"decision-sft-v1-invalid-{start_index + len(invalid):06d}"
        value["trajectory_id"] = f"trajectory-invalid-v1-{len(invalid):04d}"
        return value

    for _ in range(8):
        value = clone(1)
        value["target"]["selected_action"] = "cross_project_validate"
        invalid.append(value)
    for _ in range(8):
        value = clone(2)
        value["target"]["alternative_actions"] = ["cross_project_validate"]
        invalid.append(value)
    for _ in range(6):
        value = clone(3)
        value["target"]["selected_action"] = "finish"
        value["target"]["stop_reason"] = None
        invalid.append(value)
    for _ in range(6):
        value = clone(4)
        value["state"]["goal_code"] = "leak"
        invalid.append(value)
    for _ in range(3):
        value = clone(5)
        value["state"]["rows"] = []
        invalid.append(value)
    for _ in range(3):
        value = clone(6)
        value["state"]["data_state"]["patient_id"] = 123
        invalid.append(value)
    for _ in range(3):
        value = clone(7)
        value["state"]["data_state"]["row_count"] = -1
        invalid.append(value)
    for _ in range(3):
        value = clone(8)
        value["state"]["data_state"]["group_state"]["group_count"] = -1
        invalid.append(value)
    for _ in range(3):
        value = clone(9)
        value["target"]["selected_action"] = "cross_project_validation"
        invalid.append(value)
    return invalid


def _duplicate_candidates(valid_records: list[dict[str, Any]], start_index: int) -> list[dict[str, Any]]:
    duplicates: list[dict[str, Any]] = []
    for index, source in enumerate(valid_records[:20]):
        value = copy.deepcopy(source)
        value["sample_id"] = f"decision-sft-v1-duplicate-{start_index + index:06d}"
        value["trajectory_id"] = f"trajectory-duplicate-v1-{index:04d}"
        duplicates.append(value)
    return duplicates


def _split_groups(records: list[DecisionSftRecord]) -> dict[str, list[DecisionSftRecord]]:
    groups: defaultdict[str, list[DecisionSftRecord]] = defaultdict(list)
    for record in records:
        key = record.metadata.paired_state_group or record.trajectory_id
        groups[key].append(record)
    ordered_groups = sorted(
        groups.items(),
        key=lambda item: hashlib.sha256(item[0].encode("utf-8")).hexdigest(),
    )
    target = {"train": 360, "validation": 60, "test": 60}
    result: dict[str, list[DecisionSftRecord]] = {"train": [], "validation": [], "test": []}
    remaining = dict(target)
    for key, members in ordered_groups:
        del key
        size = len(members)
        possible = [name for name in target if remaining[name] >= size]
        if not possible:
            possible = list(target)
        chosen = max(possible, key=lambda name: (remaining[name], name == "test"))
        result[chosen].extend(members)
        remaining[chosen] -= size
    # The deterministic capacities above are exactly attainable for 400
    # singleton groups plus 40 pairs.  Fail closed if future generation
    # changes that shape.
    if any(value != 0 for value in remaining.values()):
        raise ValueError(f"DECISION_SFT_SPLIT_CAPACITY_UNMET:{remaining}")
    for values in result.values():
        values.sort(key=lambda record: record.sample_id)
    return result


def _split_leakage(split_records: Mapping[str, list[DecisionSftRecord]]) -> dict[str, Any]:
    groups = {
        split: {
            record.metadata.paired_state_group or record.trajectory_id
            for record in records
        }
        for split, records in split_records.items()
    }
    overlaps = {
        f"{left}_{right}": sorted(groups[left] & groups[right])
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }
    return {"overlap_count": sum(len(values) for values in overlaps.values()), "overlaps": overlaps}


def _audit_examples(records: list[DecisionSftRecord]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        action = record.target.selected_action
        if len(result[action]) >= 3:
            continue
        state = record.state
        result[action].append({
            "sample_id": record.sample_id,
            "source_type": record.source_type,
            "hard_case_class": record.metadata.hard_case_class,
            "available_actions": state.action_space.available_actions,
            "key_facts": {
                "group_count": state.data_state.group_state.group_count,
                "project_count": state.data_state.project_state.project_count,
                "group_comparison_status": state.analysis_state.group_comparison.status,
                "confounder_status": state.analysis_state.confounder_adjustment.status,
                "heterogeneity": state.analysis_state.cross_project_validation.heterogeneity,
                "remaining_objectives": state.progress.remaining_objectives,
            },
            "decision_reason": record.target.decision_reason,
        })
    return dict(result)


def build_decision_sft_v1(
    output_dir: Path,
    *,
    catalog_path: Path,
    manual_review_completed: bool = False,
) -> dict[str, Any]:
    """Build, validate, split and audit the frozen v1 candidate pool."""

    catalog = _load_catalog(catalog_path)
    pair_records, next_index = _build_pair_records(catalog, 1)
    single_records, next_index = _build_single_records(catalog, next_index)
    valid_raw = pair_records + single_records
    invalid_raw = _invalid_candidates(valid_raw, next_index)
    duplicate_raw = _duplicate_candidates(valid_raw, next_index + len(invalid_raw))
    raw_candidates = valid_raw + invalid_raw + duplicate_raw
    _dump_jsonl(output_dir / "all_candidates.jsonl", raw_candidates)

    accepted: list[DecisionSftRecord] = []
    rejection_reasons: Counter[str] = Counter()
    validation_issue_records: list[dict[str, Any]] = []
    seen_hashes: dict[str, str] = {}
    valid_hash_counts: Counter[str] = Counter()
    for raw in raw_candidates:
        record, issues = validate_record(raw, catalog=catalog)
        if issues:
            for issue in issues:
                rejection_reasons[issue] += 1
            validation_issue_records.append({"sample_id": raw.get("sample_id"), "issues": issues})
            continue
        if record is None:
            rejection_reasons["schema_invalid"] += 1
            continue
        key = normalized_state_hash(record)
        valid_hash_counts[key] += 1
        if key in seen_hashes:
            rejection_reasons["exact_duplicate"] += 1
            validation_issue_records.append({"sample_id": record.sample_id, "issues": ["exact_duplicate", "duplicate_of:" + seen_hashes[key]]})
            continue
        seen_hashes[key] = record.sample_id
        accepted.append(record)

    splits = _split_groups(accepted)
    _dump_jsonl(output_dir / "approved.jsonl", (record.model_dump(mode="json") for record in accepted))
    for split, records in splits.items():
        _dump_jsonl(output_dir / f"{split}.jsonl", (record.model_jsonl_record() for record in records))

    leakage = _split_leakage(splits)
    near_buckets: defaultdict[str, list[str]] = defaultdict(list)
    for record in accepted:
        if record.metadata.paired_state_group:
            continue
        near_buckets[_coarse_state_signature(record)].append(record.sample_id)
    near_groups = {key: ids for key, ids in near_buckets.items() if len(ids) > 1}
    near_duplicate_count = sum(len(ids) - 1 for ids in near_groups.values())

    action_counts = Counter(record.target.selected_action for record in accepted)
    source_counts = Counter(record.source_type for record in accepted)
    hard_counts = Counter(record.metadata.hard_case_class for record in accepted if record.metadata.hard_case_class)
    # Keep zero-valued categories in the report.  A missing ``real_trace``
    # bucket must be visible rather than silently looking like an omitted
    # audit dimension.
    action_distribution = {
        action: action_counts.get(action, 0) for action in ALL_SCIENTIFIC_ACTIONS
    }
    source_distribution = {
        source_type: source_counts.get(source_type, 0) for source_type in SOURCE_TYPES
    }
    hard_distribution = {
        hard_case: hard_counts.get(hard_case, 0) for hard_case in HARD_CASE_CLASSES
    }
    split_counts = {split: len(records) for split, records in splits.items()}
    approved_sample_id_counts = Counter(record.sample_id for record in accepted)
    duplicate_approved_sample_ids = {
        sample_id: count
        for sample_id, count in approved_sample_id_counts.items()
        if count > 1
    }
    approved_violations: Counter[str] = Counter()
    model_contract_violations: Counter[str] = Counter()
    for record in accepted:
        expected = compute_available_actions(record.state, _availability_context(record.metadata, catalog))
        if record.target.selected_action not in expected:
            approved_violations["selected_action_unavailable"] += 1
        if any(action not in expected for action in record.target.alternative_actions):
            approved_violations["alternative_action_unavailable"] += 1
        if record.target.selected_action == "finish" and record.target.stop_reason is None:
            approved_violations["finish_contract"] += 1
        if record.target.selected_action != "finish" and record.target.stop_reason is not None:
            approved_violations["finish_contract"] += 1
        input_record = record.model_jsonl_record()
        if tuple(input_record["input"]["state"]) != STATE_BLOCKS:
            model_contract_violations["state_blocks"] += 1
        if _walk_keys(input_record["input"]) or _walk_keys(input_record["output"]):
            model_contract_violations["forbidden_keys"] += 1

    hard_required = {
        "objective_action_confusion",
        "availability_boundary",
        "observation_sensitivity",
        "heterogeneity_conflict",
        "premature_finish",
        "missed_finish",
        "redundant_action",
        "missing_evidence",
        "missing_project_dimension",
        "missing_covariate",
        "insufficient_data",
    }
    checks = {
        "approved_count_300_500": 300 <= len(accepted) <= 500,
        "six_block_state_only": not model_contract_violations,
        "fixed_action_set": set(action_counts).issubset(set(ALL_SCIENTIFIC_ACTIONS)) and all(action_counts[action] >= 15 for action in ALL_SCIENTIFIC_ACTIONS),
        "selected_action_available_100pct": not approved_violations.get("selected_action_unavailable"),
        "alternative_actions_available_100pct": not approved_violations.get("alternative_action_unavailable"),
        "finish_contract_100pct": not approved_violations.get("finish_contract"),
        # These are readiness checks on the records that would actually be
        # exported to the model.  Intentionally rejected candidates are
        # retained in the audit pool and must not make an otherwise clean
        # approved split fail the gate.
        "legacy_fields_absent": not model_contract_violations.get("forbidden_keys"),
        "raw_rows_absent": not model_contract_violations.get("forbidden_keys"),
        "exact_duplicates_removed": (
            len(seen_hashes) == len(accepted)
            and sum(max(count - 1, 0) for count in valid_hash_counts.values())
            == rejection_reasons.get("exact_duplicate", 0)
        ),
        "pair_trajectory_split_leakage_zero": leakage["overlap_count"] == 0,
        "sample_id_unique": not duplicate_approved_sample_ids,
        "hard_cases_covered": hard_required.issubset(hard_counts),
        "manual_review_completed": manual_review_completed,
    }
    ready = all(checks.values())
    audit = {
        "schema_version": DECISION_SFT_SCHEMA_VERSION,
        "model_record_schema_version": MODEL_RECORD_SCHEMA_VERSION,
        "training_started": False,
        "candidate_training_eligible": True,
        "training_eligible": manual_review_completed,
        "candidate_count": len(raw_candidates),
        "validation_attempt_count": len(raw_candidates),
        "validated_candidate_count": len(accepted) + sum(1 for item in validation_issue_records if "exact_duplicate" not in item.get("issues", [])),
        "contract_valid_before_dedup": len(accepted) + rejection_reasons.get("exact_duplicate", 0),
        "contract_rejected_before_dedup": len(raw_candidates) - len(accepted) - rejection_reasons.get("exact_duplicate", 0),
        "approved_count": len(accepted),
        "rejected_count": len(raw_candidates) - len(accepted),
        "rejection_reason_counts": dict(rejection_reasons),
        "action_distribution": action_distribution,
        "source_type_distribution": source_distribution,
        "hard_case_distribution": hard_distribution,
        "real_trace_eligible_count": source_distribution["real_trace"],
        "automatic_contract_approval": True,
        "human_review_pending_count": 0 if manual_review_completed else len(accepted),
        "split_distribution": split_counts,
        "duplicate_count": rejection_reasons.get("exact_duplicate", 0),
        "near_duplicate_count": near_duplicate_count,
        "near_duplicate_groups": dict(list(sorted(near_groups.items()))[:20]),
        "availability_violations": dict(approved_violations),
        "schema_violations": dict(rejection_reasons),
        "finish_contract_violations": rejection_reasons.get("finish_contract_violation", 0),
        "objective_action_confusion_labels": rejection_reasons.get("objective_action_confusion", 0),
        "pair_trajectory_split_leakage": leakage,
        "duplicate_approved_sample_ids": duplicate_approved_sample_ids,
        "hard_case_required": sorted(hard_required),
        "hard_case_missing": sorted(hard_required - set(hard_counts)),
        "manual_review_completed": manual_review_completed,
        "label_policy": "expert_rule_v1; no LLM batch labels; eligible source traces were unavailable",
        "checks": checks,
        "DECISION_SFT_V1_DATA_READY": ready,
        "examples_by_action": _audit_examples(accepted),
        "artifacts": {
            "candidate_pool": "all_candidates.jsonl",
            "approved": "approved.jsonl",
            "train": "train.jsonl",
            "validation": "validation.jsonl",
            "frozen_test": "test.jsonl",
            "manifest": "manifest.json",
        },
    }
    manifest = {
        "schema_version": DECISION_SFT_SCHEMA_VERSION,
        "model_record_schema_version": MODEL_RECORD_SCHEMA_VERSION,
        "training_started": False,
        "approved_count": len(accepted),
        "automatic_contract_approval": True,
        "human_review_pending_count": 0 if manual_review_completed else len(accepted),
        "action_distribution": action_distribution,
        "source_type_distribution": source_distribution,
        "hard_case_distribution": hard_distribution,
        "input_contract": {
            "decision_type": "scientific_action",
            "state_blocks": list(STATE_BLOCKS),
            "forbidden_state_fields": sorted(LEGACY_OR_LEAKAGE_KEYS),
        },
        "target_contract": {
            "fields": ["selected_action", "decision_reason", "alternative_actions", "stop_reason"],
            "actions": list(ALL_SCIENTIFIC_ACTIONS),
            "stop_reasons": [
                "EVIDENCE_SUFFICIENT",
                "NO_NEW_INFORMATION",
                "QUALITY_RISK",
                "ACTION_BUDGET_EXHAUSTED",
                "UPSTREAM_REJECTED",
                "UNSUPPORTED_ACTION",
                "USER_REQUESTED_STOP",
            ],
        },
        "split_policy": {
            "unit": "trajectory_id_or_paired_state_group",
            "random_row_split": False,
            "pair_and_trajectory_disjoint": leakage["overlap_count"] == 0,
            "target_counts": {"train": 360, "validation": 60, "test": 60},
            "frozen_test": True,
        },
        "split_counts": split_counts,
        "sample_id_unique": not duplicate_approved_sample_ids,
        # Individual records carry the intended post-review eligibility flag;
        # this manifest-level flag is the safety gate consumed by downstream
        # training tooling and therefore stays false until human review is
        # explicitly completed.
        "candidate_training_eligible": True,
        "training_eligible": manual_review_completed,
        "manual_review_completed": manual_review_completed,
        "DECISION_SFT_V1_DATA_READY": ready,
    }
    schema_doc = {
        "schema_version": DECISION_SFT_SCHEMA_VERSION,
        "record_fields": ["schema_version", "sample_id", "source_type", "trajectory_id", "turn_index", "state", "target", "metadata"],
        "model_input": {"decision_type": "scientific_action", "state_blocks": list(STATE_BLOCKS)},
        "model_output": ["selected_action", "decision_reason", "alternative_actions", "stop_reason"],
        "rules": [
            "Objective is not Action.",
            "The selected action must be executable in the current state.",
            "Training state must match serving state.",
        ],
    }
    _dump_json(output_dir / "audit_report.json", audit)
    _dump_json(output_dir / "manifest.json", manifest)
    _dump_json(output_dir / "schema.json", schema_doc)
    _dump_json(output_dir / "validation_issues.json", {"count": len(validation_issue_records), "items": validation_issue_records})
    return audit


__all__ = [
    "DECISION_SFT_SCHEMA_VERSION",
    "HARD_CASE_CLASSES",
    "MODEL_RECORD_SCHEMA_VERSION",
    "SOURCE_TYPES",
    "DecisionSftMetadata",
    "DecisionSftRecord",
    "build_decision_sft_v1",
    "normalized_state_hash",
    "state_only_hash",
    "validate_record",
]
