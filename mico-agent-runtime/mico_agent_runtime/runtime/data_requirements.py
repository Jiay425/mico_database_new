"""Catalog-backed data requirements for Dynamic Scientific reads.

The Scientific Policy still chooses only a high-level Action.  This module is
the small Runtime-owned bridge between Task Understanding and the concrete
read shape needed by later objectives.  It never creates a SQL expression or
chooses a scientific route: it resolves only verified semantic Catalog IDs
and reports requirements that cannot be satisfied by the current Catalog.

The returned requirement set is intentionally kept outside the six-block
``ScientificDecisionState``.  The materializer receives its bounded
``requiredSemanticFields`` projection, while Runtime keeps the complete
diagnostic for trace/audit purposes.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Mapping
from itertools import combinations
from typing import Any

from mico_agent_runtime.contracts.decision_state import ScientificDecisionState
from mico_agent_runtime.contracts.materialization import QueryPlan, validate_query_plan_catalog
from mico_agent_runtime.contracts.schema_catalog import SchemaSemanticCatalog


_GROUP_ANALYSIS_OBJECTIVES = frozenset({
    "group_comparison",
    "stratified_analysis",
    "confounder_assessment",
    "cross_project_validation",
    "cross_disease_validation",
})


@dataclass(frozen=True)
class DataRequirementSet:
    """A bounded, JSON-safe requirement resolution result.

    ``required_fields`` contains only verified, non-sensitive semantic IDs.
    ``missing_fields`` contains the unresolved semantic request (for example
    ``age`` when no unique Catalog covariate matches it).  It is diagnostic
    data, not a field list that may be sent to Java or an LLM.
    """

    required_fields: tuple[str, ...] = ()
    required_group_field: str | None = None
    missing_fields: tuple[str, ...] = ()
    blocked_objectives: tuple[str, ...] = ()
    field_sources: tuple[tuple[str, tuple[str, ...]], ...] = ()
    # Runtime-only fields already proven to have zero usable coverage for the
    # current cohort.  They are never sent to the Policy input.
    unavailable_fields: tuple[str, ...] = ()

    def model_dump(self) -> dict[str, object]:
        return {
            "required_fields": list(self.required_fields),
            "required_group_field": self.required_group_field,
            "missing_fields": list(self.missing_fields),
            "blocked_objectives": list(self.blocked_objectives),
            "field_sources": {
                objective: list(fields)
                for objective, fields in self.field_sources
            },
            "unavailable_fields": list(self.unavailable_fields),
        }

    def without_fields(self, fields: Iterable[str]) -> "DataRequirementSet":
        excluded = set(fields)
        kept = tuple(field for field in self.required_fields if field not in excluded)
        group = self.required_group_field
        if group in excluded:
            group = None
        return DataRequirementSet(
            required_fields=kept,
            required_group_field=group,
            missing_fields=self.missing_fields,
            blocked_objectives=self.blocked_objectives,
            field_sources=self.field_sources,
            unavailable_fields=self.unavailable_fields,
        )


class DataRequirementError(ValueError):
    """A legal requirement cannot be added to the proposed QueryPlan shape."""


def _catalog_fields(
    catalog: SchemaSemanticCatalog | None,
) -> dict[str, Any]:
    if catalog is None:
        return {}
    result: dict[str, Any] = {}
    for entity in catalog.entities:
        entity_id = entity.entityId or entity.entityName
        for field in entity.fields:
            field_id = field.fieldId or f"{entity_id}.{field.name}"
            if field.semanticStatus != "verified" or field.sensitive:
                continue
            # A fieldId-less compatibility catalog is usable by the old
            # executor, but cannot safely cross the semantic requirement
            # boundary.  The synthesized fallback above is therefore only
            # retained when Java supplied a stable fieldId.
            if field.fieldId:
                result[field_id] = field
    return result


def _field_label(field_id: str) -> str:
    return field_id.rsplit(".", 1)[-1].lower()


def _resolve_field(
    requested: str,
    fields: Mapping[str, Any],
    *,
    capability: str | None = None,
) -> str | None:
    """Resolve an explicit short constraint to one verified Catalog ID.

    ``age`` therefore resolves to ``sample.age`` only when that is the unique
    Catalog match.  Ambiguous or absent requests are rejected instead of
    guessing a physical field or a similarly named scientific variable.
    """

    if not isinstance(requested, str) or not requested.strip():
        return None
    value = requested.strip()
    exact = fields.get(value)
    if exact is not None and (
        capability is None
        or capability in (getattr(exact, "scientificCapabilities", None) or [])
    ):
        return value
    lowered = value.casefold()
    candidates = [
        field_id for field_id, field in fields.items()
        if (
            field_id.casefold() == lowered
            or _field_label(field_id) == lowered
            or str(getattr(field, "name", "")).casefold() == lowered
        )
        and (
            capability is None
            or capability in (getattr(field, "scientificCapabilities", None) or [])
        )
    ]
    return candidates[0] if len(candidates) == 1 else None


def _choose_group_field(
    state: ScientificDecisionState,
    fields: Mapping[str, Any],
) -> str | None:
    current = state.data_state.group_state.group_field
    if isinstance(current, str) and current in fields:
        field = fields[current]
        if field.groupable and "dimension" in (field.scientificCapabilities or ()):
            return current

    constraints = state.task.constraints
    # Explicit disease labels are the strongest semantic signal for the
    # grouping dimension.  Prefer the sample-level disease field over a
    # dictionary display field when both are present.
    if constraints.disease_groups:
        disease_candidates = [
            field_id for field_id, field in fields.items()
            if field.groupable
            and "dimension" in (field.scientificCapabilities or ())
            and _field_label(field_id) in {"disease", "disease_name"}
        ]
        sample_disease = next(
            (field_id for field_id in disease_candidates if field_id == "sample.disease"),
            None,
        )
        if sample_disease is not None:
            return sample_disease
        if len(disease_candidates) == 1:
            return disease_candidates[0]

    candidates = [
        field_id for field_id, field in fields.items()
        if field.groupable
        and "dimension" in (field.scientificCapabilities or ())
        and _field_label(field_id) in {"disease", "disease_name", "project", "project_name"}
    ]
    sample_disease = next(
        (field_id for field_id in candidates if field_id == "sample.disease"),
        None,
    )
    return sample_disease or (candidates[0] if len(candidates) == 1 else None)


def _choose_outcome_field(
    state: ScientificDecisionState,
    fields: Mapping[str, Any],
) -> str | None:
    observed = [
        field_id for field_id in state.data_state.available_outcomes
        if field_id in fields
    ]
    if observed:
        return observed[0]
    candidates = [
        field_id for field_id, field in fields.items()
        if field.dataType in {"integer", "number"}
        and "outcome" in (field.scientificCapabilities or ())
    ]
    preferred = next(
        (field_id for field_id in candidates if field_id == "abundance.value"),
        None,
    )
    if preferred is not None:
        return preferred
    value_candidates = [field_id for field_id in candidates if _field_label(field_id) == "value"]
    return value_candidates[0] if len(value_candidates) == 1 else (
        candidates[0] if len(candidates) == 1 else None
    )


def _choose_feature_field(fields: Mapping[str, Any]) -> str | None:
    candidates = [
        field_id for field_id, field in fields.items()
        if field.groupable
        and (
            "dimension" in (field.scientificCapabilities or ())
            or "stratifier" in (field.scientificCapabilities or ())
        )
        and _field_label(field_id) in {"feature", "microbe_name_standard"}
    ]
    preferred = next(
        (field_id for field_id in candidates if field_id == "abundance.feature"),
        None,
    )
    return preferred or (candidates[0] if len(candidates) == 1 else None)


def _choose_project_field(fields: Mapping[str, Any]) -> str | None:
    candidates = [
        field_id for field_id, field in fields.items()
        if field.groupable
        and "dimension" in (field.scientificCapabilities or ())
        and _field_label(field_id) in {"project", "project_name"}
    ]
    preferred = next(
        (field_id for field_id in candidates if field_id == "metadata.project"),
        None,
    )
    return preferred or (candidates[0] if len(candidates) == 1 else None)


def derive_data_requirements(
    state: ScientificDecisionState,
    catalog: SchemaSemanticCatalog | None,
    *,
    unavailable_fields: Iterable[str] = (),
) -> DataRequirementSet:
    """Derive minimal Catalog-backed fields from objectives and constraints."""

    fields = _catalog_fields(catalog)
    unavailable = {
        value.strip() for value in unavailable_fields
        if isinstance(value, str) and value.strip()
    }
    objectives = tuple(state.task.objectives)
    required: list[str] = []
    missing: list[str] = []
    blocked: list[str] = []
    sources: dict[str, list[str]] = {}

    def add(field_id: str | None, objective: str, requested: str | None = None) -> None:
        if field_id is None:
            if requested and requested not in missing:
                missing.append(requested)
            if objective not in blocked:
                blocked.append(objective)
            return
        if field_id in unavailable:
            # A field can be present in the Catalog while its relation has
            # already been observed to have zero coverage for this cohort.
            # Preserve the objective as blocked, but never re-inject the
            # unavailable field into a later read plan.
            if objective not in blocked:
                blocked.append(objective)
            return
        if field_id not in required:
            required.append(field_id)
        sources.setdefault(field_id, []).append(objective)

    if _GROUP_ANALYSIS_OBJECTIVES.intersection(objectives):
        group_field = _choose_group_field(state, fields)
        outcome_field = _choose_outcome_field(state, fields)
        feature_field = _choose_feature_field(fields)
        for objective in objectives:
            if objective in _GROUP_ANALYSIS_OBJECTIVES:
                add(group_field, objective)
                add(outcome_field, objective)
                # The registered abundance operators are feature-aware and
                # sample-level.  Requiring the feature dimension prevents
                # pooled abundance rows from becoming pseudo-observations.
                if feature_field is not None and outcome_field == "abundance.value":
                    add(feature_field, objective)

    constraints = state.task.constraints
    if "confounder_assessment" in objectives:
        requested_covariates = list(constraints.focus_covariates)
        if not requested_covariates:
            requested_covariates = list(state.data_state.covariate_state.available_covariates)
        if requested_covariates:
            for requested in requested_covariates:
                add(
                    _resolve_field(requested, fields, capability="covariate"),
                    "confounder_assessment",
                    requested,
                )
        else:
            # An unconstrained confounder objective can discover all bounded,
            # Catalog-verified covariates on the first read.  If the Catalog
            # exposes none, preserve the objective as blocked rather than
            # fabricating a field or letting a later materializer guess one.
            catalog_covariates = [
                field_id for field_id, field in fields.items()
                if "covariate" in (field.scientificCapabilities or ())
            ]
            if catalog_covariates:
                for field_id in catalog_covariates[:8]:
                    add(field_id, "confounder_assessment")
            elif "confounder_assessment" not in blocked:
                blocked.append("confounder_assessment")

    if "stratified_analysis" in objectives:
        for requested in constraints.requested_stratifiers:
            add(
                _resolve_field(requested, fields, capability="stratifier"),
                "stratified_analysis",
                requested,
            )

    if "cross_project_validation" in objectives:
        add(_choose_project_field(fields), "cross_project_validation")

    # Cross-disease validation needs an independent validation dimension.  Do
    # not force a new multi-hop join into a first abundance read; if the
    # dimension is already observed, preserve it as a requirement for the
    # follow-up plan.  Otherwise the action-availability/lifecycle layer will
    # prove whether the capability can be attempted.
    if "cross_disease_validation" in objectives:
        observed = [
            field_id for field_id in state.data_state.available_dimensions
            if field_id in fields
            and _field_label(field_id) in {"disease", "disease_name"}
            and field_id != state.data_state.group_state.group_field
        ]
        if observed:
            add(observed[0], "cross_disease_validation")

    # Stable group field is used by the materializer's grouped coverage
    # contract.  Keep it separate from the ordinary field list when absent.
    required_group = _choose_group_field(state, fields)
    if required_group is not None and required_group not in required:
        # Only expose a group field when at least one objective needs a
        # grouped observation; an evidence-only task must not acquire data.
        if _GROUP_ANALYSIS_OBJECTIVES.intersection(objectives):
            required_group = required_group
        else:
            required_group = None

    # De-duplicate source labels while retaining deterministic insertion order.
    source_items = tuple(
        (field_id, tuple(dict.fromkeys(items)))
        for field_id, items in sources.items()
    )
    return DataRequirementSet(
        required_fields=tuple(required[:8]),
        required_group_field=required_group,
        missing_fields=tuple(missing[:8]),
        blocked_objectives=tuple(dict.fromkeys(blocked)),
        field_sources=source_items[:8],
        unavailable_fields=tuple(sorted(unavailable))[:8],
    )


def _active_entity_names(
    plan: QueryPlan,
    catalog: SchemaSemanticCatalog,
) -> tuple[set[str], set[str]]:
    entities = {
        (entity.entityId or entity.entityName): entity
        for entity in catalog.entities
    }
    root = entities.get(plan.root_entity)
    if root is None:
        raise DataRequirementError("query requirement root entity is not in catalog")
    active = {root.entityName}
    used = set(plan.relation_path)
    for relation_id in plan.relation_path:
        relation = next(
            (item for item in catalog.joins if item.relationId == relation_id),
            None,
        )
        if relation is None or relation.relationshipStatus != "verified":
            raise DataRequirementError("query requirement relation is not enabled")
        active.add(relation.leftEntity)
        active.add(relation.rightEntity)
    return active, used


def _connect_required_entity(
    *,
    target_entity_name: str,
    active_entities: set[str],
    relation_path: list[str],
    catalog: SchemaSemanticCatalog,
) -> None:
    if target_entity_name in active_entities:
        return
    # The Catalog relation graph is tiny and bounded.  Greedily append the
    # shortest verified edge from the active component; this never creates a
    # physical join or bypasses Java's final validator.
    while target_entity_name not in active_entities:
        candidates = sorted(
            (
                # Prefer an edge that reaches the required entity directly;
                # otherwise alphabetical relation order could add an
                # unrelated abundance/disease hop and exhaust the two-hop
                # QueryPlan bound before the requested field is reachable.
                0 if target_entity_name in {relation.leftEntity, relation.rightEntity} else 1,
                relation.relationId,
                relation,
            )
            for relation in catalog.joins
            if relation.relationId
            and relation.relationId not in relation_path
            and relation.relationshipStatus == "verified"
            and relation.cardinality not in {"many_to_many", "unknown"}
            and (
                (relation.leftEntity in active_entities and relation.rightEntity not in active_entities)
                or (relation.rightEntity in active_entities and relation.leftEntity not in active_entities)
            )
        )
        if not candidates:
            raise DataRequirementError(
                "query requirement field entity is not reachable from the QueryPlan root"
            )
        _priority, relation_id, relation = candidates[0]
        relation_path.append(relation_id)
        active_entities.add(relation.leftEntity)
        active_entities.add(relation.rightEntity)
        if len(relation_path) > 2:
            raise DataRequirementError("query requirement relation path exceeds the contract bound")


def augment_query_plan_with_fields(
    plan: QueryPlan,
    required_fields: Iterable[str],
    catalog: SchemaSemanticCatalog | None,
) -> QueryPlan:
    """Add legal missing raw-projection fields before the Java boundary.

    This is a technical projection binding analogous to the Runtime-owned
    opaque sample key.  It never changes filters, grouping, aggregation, or
    the selected high-level Action.  Grouped/aggregated plans are rejected
    rather than silently changing their scientific meaning.
    """

    if catalog is None:
        return plan
    required = list(dict.fromkeys(field for field in required_fields if isinstance(field, str)))
    selected = list(plan.select_fields)
    selected_or_aggregated = set(selected) | {item.field for item in plan.aggregations}
    missing = [field for field in required if field not in selected_or_aggregated]
    if not missing:
        return plan
    fields = _catalog_fields(catalog)
    unknown = [field for field in missing if field not in fields]
    if unknown:
        raise DataRequirementError(
            "query requirement field is not a verified Catalog field: " + ",".join(unknown)
        )
    if plan.aggregations or plan.group_by:
        raise DataRequirementError(
            "query requirement fields cannot be appended to an aggregated QueryPlan"
        )
    if len(selected) + len(missing) > 16:
        raise DataRequirementError("query requirement fields exceed QueryPlan select_fields bound")

    relation_path = list(plan.relation_path)
    active_entities, _used = _active_entity_names(plan, catalog)
    for field_id in missing:
        target_entity = field_id.split(".", 1)[0]
        _connect_required_entity(
            target_entity_name=next(
                (
                    entity.entityName
                    for entity in catalog.entities
                    if (entity.entityId or entity.entityName) == target_entity
                ),
                target_entity,
            ),
            active_entities=active_entities,
            relation_path=relation_path,
            catalog=catalog,
        )
        selected.append(field_id)
    augmented = plan.model_copy(update={
        "select_fields": selected,
        "relation_path": relation_path,
    })
    try:
        return validate_query_plan_catalog(augmented, catalog)
    except (TypeError, ValueError) as exc:
        raise DataRequirementError(str(exc)) from exc


def _catalog_entity_for_field(
    field_id: str,
    catalog: SchemaSemanticCatalog,
) -> str | None:
    entity_id = field_id.split(".", 1)[0]
    for entity in catalog.entities:
        if (entity.entityId or entity.entityName) == entity_id:
            return entity.entityName
    return None


def _minimal_relation_path(
    plan: QueryPlan,
    required_fields: Iterable[str],
    catalog: SchemaSemanticCatalog,
) -> list[str]:
    """Keep only the shortest subset of the model path needed by fields."""

    required_entities = {
        entity
        for field_id in required_fields
        if isinstance(field_id, str)
        for entity in [_catalog_entity_for_field(field_id, catalog)]
        if entity is not None
    }
    root_entity = next(
        (
            entity.entityName
            for entity in catalog.entities
            if (entity.entityId or entity.entityName) == plan.root_entity
        ),
        plan.root_entity,
    )
    required_entities.add(root_entity)
    original_path = list(plan.relation_path)
    for size in range(len(original_path) + 1):
        for subset in combinations(original_path, size):
            candidate = plan.model_copy(update={"relation_path": list(subset)})
            try:
                active, _ = _active_entity_names(candidate, catalog)
            except DataRequirementError:
                continue
            if required_entities.issubset(active):
                return list(subset)
    raise DataRequirementError(
        "query plan cannot connect all remaining selected fields after unavailable-field pruning"
    )


def prune_unavailable_query_plan(
    plan: QueryPlan,
    unavailable_fields: Iterable[str],
    catalog: SchemaSemanticCatalog | None,
) -> QueryPlan:
    """Remove proven-zero optional fields before the Java boundary.

    This is deliberately capability based.  It does not name a physical
    column or a particular metadata field.  If an unavailable field is used
    by a filter, grouping, aggregation, sample bound, or is the only field
    left in a plan, the plan fails closed instead of changing its meaning.
    Otherwise the field and any now-unnecessary relation edges are pruned.
    """

    if catalog is None:
        return plan
    unavailable = {
        value.strip() for value in unavailable_fields
        if isinstance(value, str) and value.strip()
    }
    if not unavailable:
        return plan
    selected = list(plan.select_fields)
    selected_unavailable = set(selected).intersection(unavailable)
    aggregate_unavailable = {
        item.field for item in plan.aggregations
    }.intersection(unavailable)
    grouped_unavailable = set(plan.group_by).intersection(unavailable)
    filtered_unavailable = {
        item.field for item in plan.filters
    }.intersection(unavailable)
    bound_unavailable = (
        {plan.sample_limit_group_field}
        if plan.sample_limit_group_field in unavailable
        else set()
    )
    if aggregate_unavailable or grouped_unavailable or filtered_unavailable or bound_unavailable:
        raise DataRequirementError(
            "unavailable query capability is required by the QueryPlan shape"
        )
    if not selected_unavailable:
        return plan
    remaining = [field_id for field_id in selected if field_id not in unavailable]
    if not remaining:
        raise DataRequirementError(
            "query plan contains only unavailable data capabilities"
        )
    relation_path = _minimal_relation_path(plan, remaining, catalog)
    pruned = plan.model_copy(update={
        "select_fields": remaining,
        "relation_path": relation_path,
    })
    try:
        return validate_query_plan_catalog(pruned, catalog)
    except (TypeError, ValueError) as exc:
        raise DataRequirementError(str(exc)) from exc


def unavailable_fields_from_state(state: Mapping[str, Any]) -> tuple[str, ...]:
    """Read the Runtime-only unavailable capability ledger."""

    values: list[str] = []
    for item in state.get("unavailableFields", ()):
        if isinstance(item, str) and item.strip():
            values.append(item.strip())
    capabilities = state.get("unavailableDataCapabilities", {})
    if isinstance(capabilities, Mapping):
        values.extend(
            item.strip() for item in capabilities
            if isinstance(item, str) and item.strip()
        )
    return tuple(dict.fromkeys(values))


_OPTIONAL_ZERO_COVERAGE_OBJECTIVES = frozenset({
    "cross_project_validation",
    "cross_disease_validation",
})


def infer_zero_coverage_fields(
    state: Mapping[str, Any],
    observation: object,
    query_plan: QueryPlan | None,
    catalog: SchemaSemanticCatalog | None,
) -> tuple[str, ...]:
    """Infer only objective-scoped optional fields from a zero-row join.

    A zero-row abundance read must not make the abundance outcome disappear.
    We therefore record a field only when the Catalog marks it as a
    non-outcome dimension and the Runtime requirement ledger shows that it
    was requested solely for a secondary validation objective.  The rule is
    generic over Catalog field IDs and relation names.
    """

    if catalog is None or query_plan is None or getattr(observation, "rowCount", None) != 0:
        return ()
    if not query_plan.relation_path:
        return ()
    fields = _catalog_fields(catalog)
    requirement_payload = state.get("dataRequirements", {})
    source_map = (
        requirement_payload.get("field_sources", {})
        if isinstance(requirement_payload, Mapping) else {}
    )
    if not isinstance(source_map, Mapping):
        source_map = {}
    try:
        active_entities, _ = _active_entity_names(query_plan, catalog)
    except DataRequirementError:
        return ()
    result: list[str] = []
    selected = list(query_plan.select_fields) + [
        item.field for item in query_plan.aggregations
    ]
    for field_id in selected:
        field = fields.get(field_id)
        if field is None or field_id.split(".", 1)[0] == query_plan.root_entity:
            continue
        entity = _catalog_entity_for_field(field_id, catalog)
        if entity is None or entity not in active_entities:
            continue
        capabilities = set(getattr(field, "scientificCapabilities", ()) or ())
        if "dimension" not in capabilities or "outcome" in capabilities:
            continue
        sources = source_map.get(field_id, ())
        if isinstance(sources, str):
            sources = [sources]
        source_set = {item for item in sources if isinstance(item, str)}
        if source_set and source_set.issubset(_OPTIONAL_ZERO_COVERAGE_OBJECTIVES):
            result.append(field_id)
    return tuple(dict.fromkeys(result))


def record_zero_coverage_capabilities(
    state: dict[str, Any],
    observation: object,
    query_plan: QueryPlan | None,
    catalog: SchemaSemanticCatalog | None,
) -> tuple[str, ...]:
    """Persist a de-identified Runtime capability fact for later planning."""

    fields = infer_zero_coverage_fields(state, observation, query_plan, catalog)
    if not fields:
        return ()
    unavailable = list(unavailable_fields_from_state(state))
    ledger = state.setdefault("unavailableDataCapabilities", {})
    if not isinstance(ledger, dict):
        ledger = {}
        state["unavailableDataCapabilities"] = ledger
    observation_id = getattr(observation, "observationId", None)
    relation_path = list(getattr(query_plan, "relation_path", ()) or ())
    for field_id in fields:
        if field_id not in unavailable:
            unavailable.append(field_id)
        ledger[field_id] = {
            "reason_code": "ZERO_COVERAGE_JOIN",
            "observed_coverage": 0,
            "observation_id": observation_id,
            "relation_path": relation_path,
        }
    state["unavailableFields"] = unavailable[:16]
    return fields


__all__ = [
    "DataRequirementError",
    "DataRequirementSet",
    "augment_query_plan_with_fields",
    "derive_data_requirements",
    "infer_zero_coverage_fields",
    "prune_unavailable_query_plan",
    "record_zero_coverage_capabilities",
    "unavailable_fields_from_state",
]
