"""Runtime-only lifecycle for requested scientific objectives.

``ScientificDecisionState`` intentionally remains the frozen six-block policy
contract.  Objective feasibility is a Runtime concern: it is derived from
validated observations, the semantic catalog, and hard availability reasons,
then retained outside the policy payload so an infeasible objective is never
silently relabelled as completed.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any, Literal, Mapping

from mico_agent_runtime.contracts.decision_state import (
    ScientificDecisionState,
    ScientificObjective,
)
from mico_agent_runtime.contracts.schema_catalog import SchemaSemanticCatalog


ObjectiveResolutionStatus = Literal["active", "completed", "blocked"]


_OBJECTIVE_STATUS_FIELDS: dict[ScientificObjective, tuple[str, str]] = {
    "group_comparison": ("analysis_state", "group_comparison"),
    "projection_analysis": ("analysis_state", "projection_analysis"),
    "stratified_analysis": ("analysis_state", "stratified_analysis"),
    "confounder_assessment": ("analysis_state", "confounder_adjustment"),
    "cross_project_validation": ("analysis_state", "cross_project_validation"),
    "cross_disease_validation": ("analysis_state", "cross_disease_validation"),
    "evidence_support": ("evidence_state", "evidence_state"),
}

_OBJECTIVE_ACTION: dict[ScientificObjective, str] = {
    "group_comparison": "compare_groups",
    "projection_analysis": "analyze_projection",
    "stratified_analysis": "stratified_analysis",
    "confounder_assessment": "adjust_confounders",
    "cross_project_validation": "cross_project_validate",
    "cross_disease_validation": "cross_disease_validate",
    "evidence_support": "retrieve_evidence",
}

# Limitation provenance is derived from Runtime-owned capability evidence.  A
# blocked objective caused by an absent data dimension is different from one
# blocked because the knowledge backend or embedding configuration is down.
# Keep the mapping explicit so a provider cannot invent a limitation category.
_ENVIRONMENT_LIMITATION_REASON_CODES = frozenset({
    "KNOWLEDGE_SOURCE_UNAVAILABLE",
    "KNOWLEDGE_BACKEND_UNAVAILABLE",
    "EMBEDDING_CONFIGURATION_UNAVAILABLE",
    "EMBEDDING_CONFIGURATION_ERROR",
})


def limitation_code_for_resolution(
    status: str,
    *,
    reason_code: str | None = None,
) -> str | None:
    """Map a terminal objective resolution to its limitation provenance.

    This is Runtime/report metadata only.  It never enters the six-block
    Decision State and never selects an Action.  Unknown blocked reasons keep
    the historical data-limitation default until a concrete capability code
    is added to the closed mapping above.
    """

    if status != "blocked":
        return None
    if reason_code in _ENVIRONMENT_LIMITATION_REASON_CODES:
        return "objective_unavailable_due_to_environment"
    return "objective_unavailable_due_to_data"


def limitation_codes_from_resolution(
    resolution: Mapping[str, Any] | None,
) -> list[str]:
    """Return distinct limitation codes, correcting stale derived values.

    Historical traces may contain the old single data code.  This projection
    uses the preserved reason code to correct only derived report metadata;
    callers must not rewrite the original trace payload.
    """

    raw = resolution if isinstance(resolution, Mapping) else {}
    entries = raw.get("resolutions", [])
    if not isinstance(entries, list):
        entries = []
    codes: list[str] = []
    for item in entries:
        if not isinstance(item, Mapping) or item.get("status") != "blocked":
            continue
        code = limitation_code_for_resolution(
            "blocked",
            reason_code=(
                str(item.get("reason_code"))
                if isinstance(item.get("reason_code"), str)
                else None
            ),
        )
        if code and code not in codes:
            codes.append(code)
    if not codes and raw.get("blocked_objectives"):
        codes.append("objective_unavailable_due_to_data")
    return codes


@dataclass(frozen=True)
class ObjectiveResolution:
    """One requested objective's Runtime-owned lifecycle state."""

    objective: ScientificObjective
    status: ObjectiveResolutionStatus
    reason_code: str | None = None
    reason: str | None = None
    action: str | None = None
    # Runtime trace metadata.  These are deliberately absent from the
    # policy-facing Decision State.
    resolved_at_step: int | None = None
    limitation_code: str | None = None

    def model_dump(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "status": self.status,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "action": self.action,
            "resolved_at_step": self.resolved_at_step,
            "limitation_code": self.limitation_code,
        }


@dataclass(frozen=True)
class ObjectiveResolutionSummary:
    """Immutable resolution snapshot stored in Runtime state, not Policy state."""

    resolutions: tuple[ObjectiveResolution, ...]

    @property
    def active_objectives(self) -> tuple[ScientificObjective, ...]:
        return tuple(
            item.objective for item in self.resolutions if item.status == "active"
        )

    @property
    def completed_objectives(self) -> tuple[ScientificObjective, ...]:
        return tuple(
            item.objective for item in self.resolutions if item.status == "completed"
        )

    @property
    def blocked_objectives(self) -> tuple[ScientificObjective, ...]:
        return tuple(
            item.objective for item in self.resolutions if item.status == "blocked"
        )

    @property
    def workflow_can_finish(self) -> bool:
        """No satisfiable required objective remains."""

        return not self.active_objectives

    @property
    def all_requested_objectives_completed(self) -> bool:
        """Blocked objectives are limitations, not successful analyses."""

        return not self.blocked_objectives and not self.active_objectives

    @property
    def limitations_present(self) -> bool:
        return bool(self.blocked_objectives)

    def model_dump(self) -> dict[str, Any]:
        return {
            "resolutions": [item.model_dump() for item in self.resolutions],
            "active_objectives": list(self.active_objectives),
            "completed_objectives": list(self.completed_objectives),
            "blocked_objectives": list(self.blocked_objectives),
            "workflow_can_finish": self.workflow_can_finish,
            "all_requested_objectives_completed": self.all_requested_objectives_completed,
            "limitations_present": self.limitations_present,
        }


def persist_objective_resolution(
    resolution: ObjectiveResolutionSummary,
    *,
    previous: Mapping[str, Any] | None = None,
    resolved_at_step: int = 0,
) -> dict[str, Any]:
    """Serialize Runtime objective lifecycle without changing Policy State.

    ``resolved_at_step`` is the action-count snapshot at which an objective
    first became terminal (completed or blocked).  The metadata is deliberately
    kept in the Runtime-only ``objectiveResolution`` map and is never copied
    into the six-block ``ScientificDecisionState`` sent to a model.
    """

    prior_entries: dict[str, Mapping[str, Any]] = {}
    if isinstance(previous, Mapping):
        raw_entries = previous.get("resolutions")
        if isinstance(raw_entries, list):
            prior_entries = {
                str(item.get("objective")): item
                for item in raw_entries
                if isinstance(item, Mapping) and item.get("objective")
            }

    payload = resolution.model_dump()
    serialized: list[dict[str, Any]] = []
    for item in payload["resolutions"]:
        objective = str(item["objective"])
        prior = prior_entries.get(objective, {})
        status = item["status"]
        prior_status = prior.get("status")
        prior_step = prior.get("resolved_at_step")
        if status in {"completed", "blocked"}:
            if isinstance(prior_step, int) and prior_step >= 0:
                terminal_step: int | None = prior_step
            elif prior_status == status:
                terminal_step = resolved_at_step
            else:
                terminal_step = resolved_at_step
        else:
            terminal_step = None
        entry = dict(item)
        entry["resolved_at_step"] = terminal_step
        entry["limitation_code"] = limitation_code_for_resolution(
            status,
            reason_code=(
                str(item.get("reason_code"))
                if isinstance(item.get("reason_code"), str)
                else None
            ),
        )
        serialized.append(entry)
    payload["resolutions"] = serialized
    return payload


def completion_semantics_from_resolution(
    resolution: Mapping[str, Any] | None,
    *,
    workflow_completed: bool,
    remaining_objectives: Sequence[str] = (),
) -> dict[str, Any]:
    """Build terminal completion semantics from the Runtime lifecycle.

    This projection is trace/report metadata.  It deliberately does not add
    fields to ``ScientificDecisionState`` or select/replace a Scientific
    Action.  A blocked objective therefore remains a limitation even when a
    finish decision closes the workflow.
    """

    raw = resolution if isinstance(resolution, Mapping) else {}
    active = [str(item) for item in raw.get("active_objectives", []) if isinstance(item, str)]
    completed = [str(item) for item in raw.get("completed_objectives", []) if isinstance(item, str)]
    blocked = [str(item) for item in raw.get("blocked_objectives", []) if isinstance(item, str)]
    all_completed_value = raw.get("all_requested_objectives_completed")
    all_completed = (
        bool(all_completed_value)
        if isinstance(all_completed_value, bool)
        else not active and not blocked
    )
    limitations = bool(raw.get("limitations_present")) or bool(blocked)
    resolution_entries = raw.get("resolutions", [])
    if not isinstance(resolution_entries, list):
        resolution_entries = []
    limitation_codes = limitation_codes_from_resolution(raw)
    return {
        "workflow_completed": bool(workflow_completed),
        "all_requested_objectives_completed": all_completed,
        "limitations_present": limitations,
        "active_objectives": active,
        "completed_objectives": completed,
        "blocked_objectives": blocked,
        "limitation_codes": list(dict.fromkeys(limitation_codes)),
        "limitation_code": limitation_codes[0] if limitation_codes else None,
        "remaining_objectives_invariant": list(remaining_objectives) == active,
    }


def _field_label(field_id: str) -> str:
    return field_id.rsplit(".", 1)[-1].lower()


def _catalog_dimension_ids(
    catalog: SchemaSemanticCatalog | None,
    labels: set[str],
) -> set[str]:
    if catalog is None:
        return set()
    result: set[str] = set()
    for entity in catalog.entities:
        for field in entity.fields:
            field_id = field.fieldId
            if (
                field_id
                and _field_label(field_id) in labels
                and field.semanticStatus == "verified"
                and not field.sensitive
                and "dimension" in field.scientificCapabilities
            ):
                result.add(field_id)
    return result


def _query_attempted_dimension(
    state: ScientificDecisionState,
    catalog: SchemaSemanticCatalog | None,
    labels: set[str],
    observations: Sequence[object],
) -> bool:
    """Whether an observed bounded read explicitly attempted this dimension."""

    ids = _catalog_dimension_ids(catalog, labels)
    if not ids:
        return False
    for observation in observations:
        fields = set(getattr(observation, "queryPlanFields", ()) or ())
        relations = set(getattr(observation, "queryPlanRelationPath", ()) or ())
        if fields.intersection(ids):
            return True
        if "sample_to_metadata" in relations and "project" in labels:
            return True
    return False


def _observed_project_attempt(
    state: ScientificDecisionState,
    catalog: SchemaSemanticCatalog | None,
    *,
    explicit_attempt: bool,
    observations: Sequence[object],
) -> bool:
    """Read the Runtime proof attached by the graph without changing the DTO.

    The graph attaches the Observation sequence transiently while resolving
    lifecycle state.  ``ScientificDecisionState`` itself remains closed and
    does not acquire a seventh block.
    """

    if explicit_attempt:
        return True
    return _query_attempted_dimension(
        state,
        catalog,
        {"project", "project_name"},
        observations,
    )


def _terminal_block_reason(
    objective: ScientificObjective,
    state: ScientificDecisionState,
    *,
    catalog: SchemaSemanticCatalog | None,
    availability_reasons: Mapping[str, str],
    knowledge_available: bool,
    explicit_dimension_attempts: Mapping[str, bool],
    observations: Sequence[object],
) -> tuple[str, str] | None:
    """Return a block only when the current data capability is proven absent.

    A merely unobserved optional field remains *active*: the Policy may issue
    another bounded read.  A dimension explicitly attempted by Java with no
    usable values, or a Catalog with no verified capability at all, is a
    terminal data boundary and may be reported as a limitation.
    """

    action = _OBJECTIVE_ACTION[objective]
    reason = availability_reasons.get(action)

    if objective == "cross_project_validation":
        project = state.data_state.project_state
        catalog_project_ids = _catalog_dimension_ids(
            catalog,
            {"project", "project_name"},
        )
        attempted = _observed_project_attempt(
            state,
            catalog,
            explicit_attempt=bool(explicit_dimension_attempts.get("project")),
            observations=observations,
        )
        if (
            project.project_count < 2
            and (
                # A zero-coverage observation is proof even when another
                # missing input (for example group coverage) also blocks the
                # plain availability check.  Do not require the latter
                # reason to be the one that won the preliminary evaluation.
                (attempted and not project.has_project_field)
                or (
                    reason in {"PROJECT_DIMENSION_NOT_IN_OBSERVATION", "INSUFFICIENT_PROJECT_COUNT"}
                    and attempted
                )
                # A Catalog with no verified project capability is a permanent
                # boundary. If the capability exists, require an explicit read
                # attempt before concluding that this dataset cannot support it.
                or (not catalog_project_ids and catalog is not None)
            )
        ):
            if project.project_count < 2 and project.has_project_field:
                return (
                    "INSUFFICIENT_PROJECT_COUNT",
                    f"cross-project validation requires at least two projects; observed {project.project_count}",
                )
            return (
                "PROJECT_DIMENSION_UNAVAILABLE",
                "cross-project validation is unavailable because no valid project dimension is present",
            )

    if objective == "cross_disease_validation":
        catalog_disease_ids = _catalog_dimension_ids(
            catalog,
            {"disease", "disease_name"},
        )
        group_field = state.data_state.group_state.group_field
        independent_disease_ids = {
            field_id for field_id in catalog_disease_ids
            if field_id != group_field
        }
        if (
            reason == "DISEASE_DIMENSION_UNAVAILABLE"
            and not independent_disease_ids
            and catalog is not None
        ):
            return (
                "VALIDATION_DIMENSION_UNAVAILABLE",
                "cross-disease validation is unavailable because no independent disease validation dimension is present",
            )

    if objective == "evidence_support" and not knowledge_available:
        if reason == "KNOWLEDGE_SOURCE_UNAVAILABLE":
            return (
                "KNOWLEDGE_SOURCE_UNAVAILABLE",
                "evidence support is unavailable because no knowledge source is configured",
            )

    return None


def resolve_objective_lifecycle(
    state: ScientificDecisionState,
    *,
    catalog: SchemaSemanticCatalog | None = None,
    availability_reasons: Mapping[str, str] | None = None,
    knowledge_available: bool = True,
    # Runtime supplies this proof from the real Observation list.  It is kept
    # as an argument rather than added to the policy State contract.
    dimension_attempts: Mapping[str, bool] | None = None,
    observations: Sequence[object] = (),
) -> ObjectiveResolutionSummary:
    """Resolve requested objectives into active, completed, or blocked.

    ``blocked`` means *not performed* and is never mapped to an analysis
    ``completed`` status.  It only affects the Runtime finish gate and final
    limitation reporting; Policy input continues to contain the six frozen
    blocks only.
    """

    reasons = availability_reasons or {}
    attempts = dimension_attempts or {}
    resolutions: list[ObjectiveResolution] = []
    for objective in state.task.objectives:
        section, field_name = _OBJECTIVE_STATUS_FIELDS[objective]
        if section == "analysis_state":
            status = getattr(getattr(state, section), field_name).status
        else:
            status = getattr(getattr(state, section), "status")
        action = _OBJECTIVE_ACTION[objective]
        if status == "completed":
            resolutions.append(ObjectiveResolution(
                objective=objective,
                status="completed",
                action=action,
            ))
            continue
        blocked = _terminal_block_reason(
            objective,
            state,
            catalog=catalog,
            availability_reasons=reasons,
            knowledge_available=knowledge_available,
            explicit_dimension_attempts=attempts,
            observations=observations,
        )
        if blocked is None:
            resolutions.append(ObjectiveResolution(
                objective=objective,
                status="active",
                action=action,
            ))
        else:
            reason_code, reason = blocked
            resolutions.append(ObjectiveResolution(
                objective=objective,
                status="blocked",
                reason_code=reason_code,
                reason=reason,
                action=action,
            ))
    return ObjectiveResolutionSummary(tuple(resolutions))


def active_remaining_objectives(
    state: ScientificDecisionState,
    resolution: ObjectiveResolutionSummary | None = None,
) -> list[ScientificObjective]:
    """Return only objectives that remain potentially satisfiable."""

    if resolution is None:
        return [
            objective for objective in state.task.objectives
            if objective in _unfinished_objectives(state)
        ]
    return list(resolution.active_objectives)


def _unfinished_objectives(state: ScientificDecisionState) -> set[ScientificObjective]:
    result: set[ScientificObjective] = set()
    for objective in state.task.objectives:
        section, field_name = _OBJECTIVE_STATUS_FIELDS[objective]
        status = (
            getattr(getattr(state, section), field_name).status
            if section == "analysis_state"
            else getattr(getattr(state, section), "status")
        )
        if status != "completed":
            result.add(objective)
    return result


def objective_resolution_from_dict(value: object) -> ObjectiveResolutionSummary | None:
    """Rehydrate the small Runtime snapshot when a checkpoint stores JSON."""

    if not isinstance(value, Mapping):
        return None
    entries = value.get("resolutions")
    if not isinstance(entries, list):
        return None
    resolutions: list[ObjectiveResolution] = []
    for item in entries:
        if not isinstance(item, Mapping):
            return None
        objective = item.get("objective")
        status = item.get("status")
        if objective not in _OBJECTIVE_ACTION or status not in {"active", "completed", "blocked"}:
            return None
        resolutions.append(ObjectiveResolution(
            objective=objective,
            status=status,
            reason_code=item.get("reason_code") if isinstance(item.get("reason_code"), str) else None,
            reason=item.get("reason") if isinstance(item.get("reason"), str) else None,
            action=item.get("action") if isinstance(item.get("action"), str) else _OBJECTIVE_ACTION[objective],
            resolved_at_step=item.get("resolved_at_step") if isinstance(item.get("resolved_at_step"), int) else None,
            limitation_code=item.get("limitation_code") if isinstance(item.get("limitation_code"), str) else None,
        ))
    return ObjectiveResolutionSummary(tuple(resolutions))


__all__ = [
    "ObjectiveResolution",
    "ObjectiveResolutionStatus",
    "ObjectiveResolutionSummary",
    "active_remaining_objectives",
    "completion_semantics_from_resolution",
    "limitation_code_for_resolution",
    "limitation_codes_from_resolution",
    "objective_resolution_from_dict",
    "persist_objective_resolution",
    "resolve_objective_lifecycle",
]
