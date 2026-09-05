"""Hard technical availability for the ten Scientific Agent actions.

This module deliberately does not rank or select scientific actions.  It
answers only whether an action has the minimum runtime inputs, permission and
tool conditions needed for an attempt.  The policy-facing state receives only
``available_actions``; the blocked reason map is kept as runtime/debug data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from mico_agent_runtime.contracts.decision_state import (
    ScientificDecisionState,
    ScientificObjective,
)
from mico_agent_runtime.contracts.research import (
    ResearchScope,
    ScientificActionName,
)
from mico_agent_runtime.contracts.schema_catalog import SchemaSemanticCatalog


ALL_SCIENTIFIC_ACTIONS: tuple[ScientificActionName, ...] = (
    "inspect_cohort",
    "execute_read_query",
    "compare_groups",
    "analyze_projection",
    "stratified_analysis",
    "adjust_confounders",
    "cross_project_validate",
    "cross_disease_validate",
    "retrieve_evidence",
    "finish",
)


@dataclass(frozen=True)
class ActionAvailabilityContext:
    """Runtime facts that are not part of the compressed Decision State."""

    allowed_actions: Sequence[ScientificActionName] = ALL_SCIENTIFIC_ACTIONS
    requested_scopes: Sequence[ResearchScope] = (
        "mico:query:read",
        "mico:research:read",
        "mico:evidence:read",
    )
    catalog: SchemaSemanticCatalog | None = None
    knowledge_available: bool = True
    remaining_action_budget: int | None = None
    max_actions: int | None = None
    action_count: int | None = None
    distinct_counts: Mapping[str, int] = field(default_factory=dict)
    # Optional signatures are supplied by callers that can prove a repeated
    # action has the same input.  Action history alone is intentionally not a
    # hard block because a later query/State may make the action meaningful.
    completed_signatures: Mapping[ScientificActionName, Sequence[str]] = field(
        default_factory=dict
    )
    current_signatures: Mapping[ScientificActionName, str] = field(default_factory=dict)
    # Runtime-only lifecycle input.  Blocked objectives remain incomplete in
    # analysis state and are reported as limitations; they simply must not
    # make the terminal ``finish`` action impossible forever.
    blocked_objectives: Sequence[ScientificObjective] = ()


@dataclass(frozen=True)
class ActionAvailabilityResult:
    """Availability plus internal diagnostics excluded from Qwen State."""

    available_actions: list[ScientificActionName]
    blocked_reasons: dict[ScientificActionName, str]

    @property
    def blocked_actions(self) -> list[ScientificActionName]:
        return list(self.blocked_reasons)


def _catalog_fields(catalog: SchemaSemanticCatalog | None) -> dict[
    str, tuple[str, frozenset[str], str, bool]
]:
    if catalog is None:
        return {}
    fields: dict[str, tuple[str, frozenset[str], str, bool]] = {}
    for entity in catalog.entities:
        entity_id = entity.entityId or entity.entityName
        for catalog_field in entity.fields:
            field_id = catalog_field.fieldId or f"{entity_id}.{catalog_field.name}"
            fields[field_id] = (
                catalog_field.dataType,
                frozenset(catalog_field.scientificCapabilities),
                catalog_field.semanticStatus,
                catalog_field.sensitive,
            )
    return fields


def _budget_available(
    state: ScientificDecisionState,
    context: ActionAvailabilityContext,
) -> bool:
    if context.remaining_action_budget is not None:
        return context.remaining_action_budget > 0
    if context.max_actions is not None:
        count = (
            context.action_count
            if context.action_count is not None
            else state.progress.action_count
        )
        return count < context.max_actions
    return True


def _scope_available(
    action: ScientificActionName,
    requested_scopes: set[ResearchScope],
) -> bool:
    required = {
        "execute_read_query": "mico:query:read",
        "inspect_cohort": "mico:query:read",
        "compare_groups": "mico:research:read",
        "stratified_analysis": "mico:research:read",
        "adjust_confounders": "mico:research:read",
        "cross_project_validate": "mico:research:read",
        "cross_disease_validate": "mico:research:read",
        "analyze_projection": "mico:research:read",
        "retrieve_evidence": "mico:evidence:read",
        "finish": None,
    }[action]
    return required is None or required in requested_scopes


def _outcomes(state: ScientificDecisionState) -> list[str]:
    data = state.data_state
    # ``available_outcomes`` is the current contract.  The second name is
    # accepted for forward compatibility with the original design proposal,
    # without adding a policy hint or changing the State contract here.
    values = list(getattr(data, "available_outcomes", []) or [])
    values.extend(list(getattr(data, "available_numeric_fields", []) or []))
    return list(dict.fromkeys(values))


def _dimensions(state: ScientificDecisionState) -> set[str]:
    return set(state.data_state.available_dimensions)


def _group_ready(state: ScientificDecisionState) -> bool:
    group = state.data_state.group_state
    return (
        state.data_state.has_tabular_data
        and bool(_outcomes(state))
        and group.group_field is not None
        and group.group_field in _dimensions(state)
        # The currently registered typed group, stratified, confounder, and
        # cross-validation operators require exactly two observed groups.
        # Keep this as a hard capability fact; the Policy still decides what
        # to do when a fresh read can change the observed coverage.
        and group.group_count == 2
    )


def _has_stratifier(
    state: ScientificDecisionState,
    fields: Mapping[str, tuple[str, frozenset[str], str, bool]],
) -> bool:
    dimensions = _dimensions(state)
    if not dimensions:
        return False
    if not fields:
        # Without Catalog proof a field cannot be promoted to a stratifier.
        return False
    return any(
        field_id in dimensions
        and metadata[2] == "verified"
        and not metadata[3]
        and "stratifier" in metadata[1]
        for field_id, metadata in fields.items()
    )


def _has_project_dimension(
    state: ScientificDecisionState,
    fields: Mapping[str, tuple[str, frozenset[str], str, bool]],
) -> bool:
    if not state.data_state.project_state.has_project_field:
        return False
    dimensions = _dimensions(state)
    # A project count/flag is not sufficient by itself.  The field must be
    # both present in the current observation and proven by the semantic
    # Catalog as a verified, non-sensitive dimension.  In particular, do not
    # promote a covariate such as ``sample.age`` (or an unverified field whose
    # name happens to end in ``project``) into a cross-project capability.
    if not fields:
        return False
    return any(
        field_id in dimensions
        and field_id.rsplit(".", 1)[-1].lower() in {"project", "project_name"}
        and metadata[2] == "verified"
        and not metadata[3]
        and "dimension" in metadata[1]
        for field_id, metadata in fields.items()
    )


def _has_disease_dimension(
    state: ScientificDecisionState,
    fields: Mapping[str, tuple[str, frozenset[str], str, bool]],
    distinct_counts: Mapping[str, int],
) -> bool:
    group_field = state.data_state.group_state.group_field
    candidates = [
        field_id for field_id in _dimensions(state)
        if (
            field_id.rsplit(".", 1)[-1].lower() in {"disease", "disease_name"}
            or field_id.lower() == "disease.name"
        )
    ]
    group_field = state.data_state.group_state.group_field
    for field_id in candidates:
        if fields:
            metadata = fields.get(field_id)
            if not metadata or metadata[2] != "verified" or metadata[3] \
                    or "dimension" not in metadata[1]:
                continue
        # Cross-disease validation needs an independent validation dimension;
        # reusing the same field as the current group would only repeat the
        # original comparison and is rejected by AnalysisPlan v2.
        if field_id == group_field:
            continue
        if distinct_counts.get(field_id, 0) >= 2:
            return True
    return False


def _repeated_signature_blocked(
    action: ScientificActionName,
    context: ActionAvailabilityContext,
) -> bool:
    current = context.current_signatures.get(action)
    if current is None:
        return False
    return current in set(context.completed_signatures.get(action, ()))


def _finish_ready(
    state: ScientificDecisionState,
    blocked_objectives: Sequence[ScientificObjective] = (),
) -> bool:
    statuses = {
        "group_comparison": state.analysis_state.group_comparison.status,
        "projection_analysis": state.analysis_state.projection_analysis.status,
        "stratified_analysis": state.analysis_state.stratified_analysis.status,
        "confounder_assessment": state.analysis_state.confounder_adjustment.status,
        "cross_project_validation": state.analysis_state.cross_project_validation.status,
        "cross_disease_validation": state.analysis_state.cross_disease_validation.status,
        "evidence_support": state.evidence_state.status,
    }
    blocked = set(blocked_objectives)
    return all(
        statuses.get(objective) == "completed" or objective in blocked
        for objective in state.task.objectives
    )


def evaluate_action_availability(
    state: ScientificDecisionState,
    context: ActionAvailabilityContext | None = None,
) -> ActionAvailabilityResult:
    """Compute available actions and retain blocked reasons for diagnostics."""

    context = context or ActionAvailabilityContext()
    fields = _catalog_fields(context.catalog)
    dimensions = _dimensions(state)
    outcomes = _outcomes(state)
    group_ready = _group_ready(state)
    budget_available = _budget_available(state, context)
    requested_scopes = set(context.requested_scopes)
    allowed = set(context.allowed_actions)
    reasons: dict[ScientificActionName, str] = {}

    def block(action: ScientificActionName, reason: str) -> None:
        if action in allowed:
            reasons[action] = reason

    for action in ALL_SCIENTIFIC_ACTIONS:
        if action not in allowed:
            continue
        if not _scope_available(action, requested_scopes):
            block(action, "MISSING_REQUIRED_SCOPE")
            continue
        if action != "finish" and not budget_available:
            block(action, "ACTION_BUDGET_EXHAUSTED")
            continue
        if action == "inspect_cohort":
            # Inspection is a metadata/profile operation.  Its minimum hard
            # requirements are read scope and budget; a catalog may be the
            # thing this action is intended to discover.
            if _repeated_signature_blocked(action, context):
                block(action, "IDENTICAL_ACTION_SIGNATURE_ALREADY_COMPLETED")
        elif action == "execute_read_query":
            if context.catalog is None or not fields:
                block(action, "SEMANTIC_CATALOG_UNAVAILABLE")
            elif _repeated_signature_blocked(action, context):
                block(action, "IDENTICAL_ACTION_SIGNATURE_ALREADY_COMPLETED")
        elif action == "compare_groups":
            if not state.data_state.has_tabular_data:
                block(action, "NO_TABULAR_OBSERVATION")
            elif not outcomes:
                block(action, "NO_AVAILABLE_OUTCOME")
            elif not state.data_state.group_state.group_field:
                block(action, "NO_GROUP_DIMENSION")
            elif state.data_state.group_state.group_field not in dimensions:
                block(action, "GROUP_DIMENSION_NOT_IN_OBSERVATION")
            elif state.data_state.group_state.group_count != 2:
                block(action, "EXACTLY_TWO_GROUPS_REQUIRED")
        elif action == "adjust_confounders":
            if not group_ready:
                block(action, "GROUP_COMPARISON_INPUTS_UNAVAILABLE")
            elif not state.data_state.covariate_state.available_covariates:
                block(action, "NO_AVAILABLE_COVARIATE")
        elif action == "stratified_analysis":
            if not group_ready:
                block(action, "GROUP_COMPARISON_INPUTS_UNAVAILABLE")
            elif not _has_stratifier(state, fields):
                block(action, "NO_AVAILABLE_STRATIFIER")
        elif action == "cross_project_validate":
            if not group_ready:
                block(action, "GROUP_COMPARISON_INPUTS_UNAVAILABLE")
            elif not _has_project_dimension(state, fields):
                block(action, "PROJECT_DIMENSION_NOT_IN_OBSERVATION")
            elif state.data_state.project_state.project_count < 2:
                block(action, "INSUFFICIENT_PROJECT_COUNT")
        elif action == "cross_disease_validate":
            if not group_ready:
                block(action, "GROUP_COMPARISON_INPUTS_UNAVAILABLE")
            elif not _has_disease_dimension(state, fields, context.distinct_counts):
                block(action, "DISEASE_DIMENSION_UNAVAILABLE")
            # The typed cross-disease operator is not registered yet.  A real
            # disease dimension is still enough to attempt the existing
            # generated action; the Capability Registry validates the
            # concrete plan after Qwen/DeepSeek materialization.
        elif action == "analyze_projection":
            if not state.data_state.has_tabular_data:
                block(action, "NO_TABULAR_OBSERVATION")
            elif not outcomes:
                block(action, "NO_AVAILABLE_OUTCOME")
        elif action == "retrieve_evidence":
            if not state.task.query.strip():
                block(action, "NO_RESEARCH_QUERY")
            elif not context.knowledge_available:
                block(action, "KNOWLEDGE_SOURCE_UNAVAILABLE")
        elif action == "finish" and not _finish_ready(
            state,
            context.blocked_objectives,
        ):
            block(action, "REQUIRED_OBJECTIVE_INCOMPLETE")

    available = [
        action for action in ALL_SCIENTIFIC_ACTIONS
        if action in allowed and action not in reasons
    ]
    return ActionAvailabilityResult(available_actions=available, blocked_reasons=reasons)


def compute_available_actions(
    state: ScientificDecisionState,
    runtime_context: ActionAvailabilityContext | None = None,
) -> list[ScientificActionName]:
    """Return only the hard-executable actions for the current State."""

    return evaluate_action_availability(state, runtime_context).available_actions


__all__ = [
    "ALL_SCIENTIFIC_ACTIONS",
    "ActionAvailabilityContext",
    "ActionAvailabilityResult",
    "compute_available_actions",
    "evaluate_action_availability",
]
