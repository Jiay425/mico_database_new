"""Project validated observations into the compact Scientific Decision State.

This module is deliberately an observation adapter, not a planner.  It only
copies facts that are present in the Java read model, the bounded Python
result, or the literature retrieval result.  In particular, it never adds
``should_*``/``required_*`` hints and it never infers a next action.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from mico_agent_runtime.contracts.analysis import AnalysisResult
from mico_agent_runtime.contracts.decision_state import (
    ScientificDecisionState,
    ScientificObjective,
)
from mico_agent_runtime.contracts.evidence import LiteratureEvidenceItem
from mico_agent_runtime.contracts.materialization import QueryPlan
from mico_agent_runtime.contracts.research import Observation
from mico_agent_runtime.contracts.schema_catalog import SchemaSemanticCatalog


logger = logging.getLogger(__name__)


_ANALYSIS_STATUS = {
    "COMPLETED": "completed",
    "FAILED": "failed",
    "REJECTED": "failed",
    # The execution contract does not distinguish a partial result from an
    # insufficient-data result.  Keep the distinction out of the decision
    # state instead of pretending that PARTIAL has a scientific meaning.
    "PARTIAL": "failed",
}

# Compatibility labels for pre-v1 catalogs that omitted ``fieldId``.  These
# are the explicit semantic aliases already declared by Java's catalog
# builder; this is not a substring/column-name classifier.
_LEGACY_SEMANTIC_LABELS = {
    "project_name": "project",
    "abundance_value": "value",
    "microbe_name_standard": "feature",
    "disease_name": "name",
}


def _copy(state: ScientificDecisionState) -> ScientificDecisionState:
    return state.model_copy(deep=True)


def _field_id_map(
    catalog: SchemaSemanticCatalog | None,
    field_ids: Iterable[str],
) -> dict[str, Any]:
    """Return only catalog-declared, verified fields.

    A catalog without a stable ``fieldId`` is an old compatibility snapshot.
    It may still be used by the existing executor, but it is not safe to
    expose its physical ``name`` as a Decision State dimension, so such fields
    are intentionally left unmappable here.
    """

    wanted = set(field_ids)
    if catalog is None:
        return {}
    result: dict[str, Any] = {}
    for entity in catalog.entities:
        for field in entity.fields:
            field_id = field.fieldId
            if (
                field_id
                and field_id in wanted
                and field.semanticStatus == "verified"
                and not field.sensitive
            ):
                result[field_id] = field
    return result


def _field_label(field_id: str) -> str:
    """Return the terminal semantic component for internal classification."""

    label = field_id.rsplit(".", 1)[-1]
    return _LEGACY_SEMANTIC_LABELS.get(label, label)


def _has_capability(field: Any, capability: str) -> bool:
    """Read the closed Catalog capability list without guessing field names."""

    return capability in (getattr(field, "scientificCapabilities", None) or [])


def _aliases(field_id: str) -> tuple[str, ...]:
    entity, field = field_id.split(".", 1)
    return (
        f"a_{entity}_{field}",
        f"a_{entity}_{field}_count",
        f"a_{entity}_{field}_mean",
        f"a_{entity}_{field}_min",
        f"a_{entity}_{field}_max",
        f"a_{entity}_{field}_sum",
    )


def _payload_parts(payload: object) -> tuple[list[str], list[dict[str, object]]]:
    if not isinstance(payload, Mapping):
        return [], []
    columns = payload.get("columns")
    rows = payload.get("rows")
    safe_columns = [item for item in columns if isinstance(item, str)] if isinstance(columns, list) else []
    safe_rows = [item for item in rows if isinstance(item, Mapping)] if isinstance(rows, list) else []
    return safe_columns, [dict(item) for item in safe_rows]


def _returned_fields(
    observation: Observation,
    payload: object,
    query_plan: QueryPlan | None,
    catalog: SchemaSemanticCatalog | None,
) -> tuple[dict[str, Any], dict[str, str], list[dict[str, object]]]:
    columns, rows = _payload_parts(payload)
    candidate_ids = list(observation.queryPlanFields)
    if query_plan is not None:
        candidate_ids.extend(query_plan.select_fields)
        candidate_ids.extend(item.field for item in query_plan.aggregations)
    candidate_ids = list(dict.fromkeys(candidate_ids))
    field_map = _field_id_map(catalog, candidate_ids)
    alias_to_field: dict[str, str] = {}
    returned = set(columns)
    for field_id in field_map:
        for alias in _aliases(field_id):
            if alias in returned:
                alias_to_field[alias] = field_id
    return field_map, alias_to_field, rows


def _numeric(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _value_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, (dict, list, tuple, set)):
        return None
    text = str(value).strip()
    if not text or len(text) > 512 or any(ord(char) < 32 for char in text):
        return None
    return text


def _base_alias(field_id: str, alias_to_field: Mapping[str, str]) -> str | None:
    expected = f"a_{field_id.replace('.', '_')}"
    return expected if alias_to_field.get(expected) == field_id else None


def _count_alias(field_id: str, alias_to_field: Mapping[str, str]) -> str | None:
    expected = f"a_{field_id.replace('.', '_')}_count"
    return expected if alias_to_field.get(expected) == field_id else None


def _distinct_values(
    rows: Sequence[Mapping[str, object]],
    field_id: str,
    alias_to_field: Mapping[str, str],
) -> set[str]:
    alias = _base_alias(field_id, alias_to_field)
    if alias is None:
        return set()
    values: set[str] = set()
    for row in rows:
        value = _value_text(row.get(alias))
        if value is not None:
            values.add(value)
    return values


def _select_group_field(
    field_map: Mapping[str, Any],
    alias_to_field: Mapping[str, str],
    query_plan: QueryPlan | None,
) -> str | None:
    if query_plan is not None:
        grouped = [field_id for field_id in query_plan.group_by if field_id in field_map]
        if len(grouped) == 1 and any(value == grouped[0] for value in alias_to_field.values()):
            return grouped[0]
    semantic_candidates = [
        field_id for field_id, field in field_map.items()
        if _field_label(field_id) in {"disease", "project"}
        and (_has_capability(field, "dimension") or _has_capability(field, "stratifier"))
        and any(value == field_id for value in alias_to_field.values())
    ]
    if len(semantic_candidates) == 1:
        return semantic_candidates[0]
    # A raw observation can legitimately expose both dimensions without a
    # GROUP BY.  Disease is the stable cohort grouping dimension in the Java
    # catalog; project remains independently counted in project_state.  This
    # is a deterministic observation projection, not an action preference.
    for preferred in ("disease", "project"):
        candidate = next(
            (field_id for field_id in semantic_candidates if _field_label(field_id) == preferred),
            None,
        )
        if candidate is not None:
            return candidate
    return None


def update_from_query_result(
    state: ScientificDecisionState,
    observation: Observation,
    payload: object,
    *,
    query_plan: QueryPlan | None = None,
    catalog: SchemaSemanticCatalog | None = None,
    row_count: int | None = None,
    require_sample_level_outcome: bool = False,
) -> ScientificDecisionState:
    """Project one validated Java query/inspection observation."""

    updated = _copy(state)
    data = updated.data_state
    _columns, rows = _payload_parts(payload)
    authoritative_count = row_count if isinstance(row_count, int) and row_count >= 0 else observation.rowCount
    if authoritative_count != len(rows):
        logger.warning(
            "Java rowCount (%s) differs from returned row list length (%s); "
            "Decision State keeps the authoritative rowCount",
            authoritative_count,
            len(rows),
        )
    field_map, alias_to_field, rows = _returned_fields(observation, payload, query_plan, catalog)
    # A column name in an empty result is only a requested projection, not an
    # observed capability.  Promote a field into Decision State only when at
    # least one returned row contains a usable value for its verified alias;
    # this prevents a zero-row metadata join from falsely setting
    # ``has_project_field`` and advertising a zero-coverage dimension.
    visible_fields = [
        field_id for field_id in field_map
        if any(
            value == field_id
            and any(_value_text(row.get(alias)) is not None for row in rows)
            for alias, value in alias_to_field.items()
        )
    ]
    dimensions = [
        field_id
        for field_id in visible_fields
        if _has_capability(field_map[field_id], "dimension")
        or _has_capability(field_map[field_id], "stratifier")
    ]
    outcomes = [
        field_id
        for field_id in visible_fields
        if field_map[field_id].dataType in {"integer", "number"}
        and _has_capability(field_map[field_id], "outcome")
    ]
    if require_sample_level_outcome and outcomes:
        # Aggregate/grouped abundance rows are valid observations, but they
        # are not valid input for the registered sample-level statistical
        # operators.  Java adds this opaque runtime-only column only for a
        # raw sample×feature projection.  Do not promote an aggregate metric
        # to an analysis outcome merely because its semantic field is numeric.
        opaque_sample_key_present = "a_analysis_sample_key" in _columns
        if not opaque_sample_key_present or not rows:
            outcomes = []
    sample_level_result = (
        "a_analysis_sample_key" in _columns and bool(rows)
        if require_sample_level_outcome else True
    )
    current_has_tabular_data = bool(
        isinstance(payload, Mapping)
        and isinstance(payload.get("columns"), list)
        and isinstance(payload.get("rows"), list)
    )
    data.has_tabular_data = data.has_tabular_data or current_has_tabular_data
    data.row_count = authoritative_count
    # Java's runtime-only key is the only safe sample identity exposed to the
    # Python analysis layer.  Keep row/sample/feature counts separate; never
    # infer a patient count from abundance fan-out rows or opaque samples.
    if "a_analysis_sample_key" in _columns:
        data.sample_count = len({
            token
            for row in rows
            if (token := _value_text(row.get("a_analysis_sample_key"))) is not None
        })
    feature_alias = _base_alias("abundance.feature", alias_to_field)
    if feature_alias is not None:
        data.feature_count = len({
            token
            for row in rows
            if (token := _value_text(row.get(feature_alias))) is not None
        })
    data.available_dimensions = list(dict.fromkeys([
        *data.available_dimensions,
        *dimensions,
    ]))
    data.available_outcomes = list(dict.fromkeys([
        *data.available_outcomes,
        *outcomes,
    ]))

    group_field = _select_group_field(field_map, alias_to_field, query_plan)
    group_sizes: dict[str, int] = {}
    if group_field is not None and sample_level_result:
        base = _base_alias(group_field, alias_to_field)
        count_alias = _count_alias(group_field, alias_to_field)
        sample_alias = "a_analysis_sample_key"
        if require_sample_level_outcome and sample_alias in _columns and base is not None:
            grouped_samples: dict[str, set[str]] = {}
            for row in rows:
                label = _value_text(row.get(base))
                sample = _value_text(row.get(sample_alias))
                if label is None or sample is None:
                    continue
                grouped_samples.setdefault(label, set()).add(sample)
            group_sizes = {
                label: len(samples) for label, samples in grouped_samples.items()
            }
            if data.group_state.group_field == group_field:
                # Consecutive bounded reads may target different requested
                # groups.  Preserve the observed group counts across those
                # reads; each individual read is already de-duplicated by its
                # opaque sample key.
                group_sizes = {
                    **data.group_state.group_sizes,
                    **group_sizes,
                }
        elif base is not None:
            for row in rows:
                label = _value_text(row.get(base))
                if label is None:
                    continue
                count = _numeric(row.get(count_alias)) if count_alias else None
                group_sizes[label] = group_sizes.get(label, 0) + (int(count) if count is not None and count >= 0 else 1)
    if group_field is not None and sample_level_result:
        data.group_state.group_field = group_field
        data.group_state.group_count = len(group_sizes)
        data.group_state.group_sizes = group_sizes
        if require_sample_level_outcome:
            data.sample_count = sum(group_sizes.values())

    project_field = next(
        (field_id for field_id in visible_fields if _field_label(field_id) == "project"),
        None,
    )
    if project_field is not None and sample_level_result:
        data.project_state.has_project_field = True
        data.project_state.project_count = len(
            _distinct_values(rows, project_field, alias_to_field)
        )

    covariates = [
        field_id
        for field_id in visible_fields
        if _has_capability(field_map[field_id], "covariate")
    ]
    data.covariate_state.available_covariates = list(dict.fromkeys([
        *data.covariate_state.available_covariates,
        *covariates,
    ]))
    # No project-wide imbalance statistic or frozen threshold exists in the
    # current runtime.  Keep this empty rather than inventing a policy signal.
    data.covariate_state.imbalance = dict(data.covariate_state.imbalance)
    return updated


def _plan_field_values(plan: object | None, snake_name: str, legacy_name: str) -> list[str]:
    value = getattr(plan, snake_name, None)
    if value is None:
        value = getattr(plan, legacy_name, None)
    return [item for item in (value or []) if isinstance(item, str)]


def _source_rows(
    source_payloads: Mapping[str, object] | None,
    source_ids: Iterable[str],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    if not source_payloads:
        return result
    for observation_id in source_ids:
        _columns, rows = _payload_parts(source_payloads.get(observation_id))
        result.extend(rows)
    return result


def _distinct_catalog_dimension_count(
    source_payloads: Mapping[str, object] | None,
    catalog: SchemaSemanticCatalog | None,
    label: str,
) -> int:
    """Count a semantic dimension directly from source rows when needed."""

    rows = _source_rows(source_payloads, source_payloads.keys() if source_payloads else ())
    if not rows or catalog is None:
        return 0
    for entity in catalog.entities:
        for field in entity.fields:
            field_id = field.fieldId
            if not field_id or field.semanticStatus != "verified" or field.sensitive:
                continue
            if _field_label(field_id) != label:
                continue
            alias = f"a_{field_id.replace('.', '_')}"
            values = {
                value for row in rows
                if (value := _value_text(row.get(alias))) is not None
            }
            if values:
                return len(values)
    return 0


def update_from_analysis_result(
    state: ScientificDecisionState,
    result: AnalysisResult,
    *,
    analysis_plan: object | None = None,
    source_payloads: Mapping[str, object] | None = None,
    catalog: SchemaSemanticCatalog | None = None,
) -> ScientificDecisionState:
    """Project the real bounded Python metrics into the matching action block."""

    updated = _copy(state)
    status = _ANALYSIS_STATUS.get(result.status, "failed")
    metrics = result.metrics
    action = result.actionName

    if action == "compare_groups":
        updated.analysis_state.group_comparison.status = status
        # v2 typed operators expose the signed difference between the second
        # and first deterministic group labels.  Keep the v1 effect_size
        # mirror only for legacy/generated results.
        mean_difference = metrics.get("mean_difference")
        updated.analysis_state.group_comparison.mean_difference = mean_difference
        updated.analysis_state.group_comparison.effect_size = (
            metrics.get("effect_size") if mean_difference is None else mean_difference
        )
        updated.analysis_state.group_comparison.p_value = metrics.get("p_value")
        updated.analysis_state.group_comparison.confidence_interval_low = metrics.get(
            "confidence_interval_low"
        )
        updated.analysis_state.group_comparison.confidence_interval_high = metrics.get(
            "confidence_interval_high"
        )
    elif action == "adjust_confounders":
        updated.analysis_state.confounder_adjustment.status = status
        adjusted_effect = metrics.get("adjusted_group_effect")
        updated.analysis_state.confounder_adjustment.adjusted_group_effect = adjusted_effect
        # Retain the old field as a compatibility mirror; never substitute an
        # unadjusted group metric for an adjusted result.
        updated.analysis_state.confounder_adjustment.adjusted_effect_size = (
            metrics.get("adjusted_effect_size")
            if adjusted_effect is None else adjusted_effect
        )
        updated.analysis_state.confounder_adjustment.adjusted_p_value = metrics.get("adjusted_p_value")
        updated.analysis_state.confounder_adjustment.adjusted_covariates = list(dict.fromkeys(
            list(result.adjusted_covariates)
            or _plan_field_values(analysis_plan, "covariates", "confounders")
        ))
        updated.analysis_state.confounder_adjustment.used_row_count = (
            result.used_row_count or result.rowsAnalyzed
        )
        updated.analysis_state.confounder_adjustment.dropped_row_count = result.dropped_row_count
    elif action == "cross_project_validate":
        updated.analysis_state.cross_project_validation.status = status
        project_count = int(metrics.get("project_count", 0.0))
        if project_count == 0:
            project_count = len(result.validation_results)
        if project_count == 0:
            project_count = state.data_state.project_state.project_count
        if project_count == 0:
            project_count = _distinct_catalog_dimension_count(
                source_payloads, catalog, "project"
            )
        updated.analysis_state.cross_project_validation.project_count = project_count
        updated.analysis_state.cross_project_validation.positive_project_count = int(
            metrics.get("positive_project_count", 0.0)
        )
        updated.analysis_state.cross_project_validation.negative_project_count = int(
            metrics.get("negative_project_count", 0.0)
        )
        updated.analysis_state.cross_project_validation.neutral_project_count = int(
            metrics.get("neutral_project_count", 0.0)
        )
        updated.analysis_state.cross_project_validation.effect_min = metrics.get("effect_min")
        updated.analysis_state.cross_project_validation.effect_max = metrics.get("effect_max")
        # No formal heterogeneity metric/threshold is emitted by the current
        # typed operator, so the decision state remains explicitly unknown.
        updated.analysis_state.cross_project_validation.heterogeneity = "unknown"
    elif action == "cross_disease_validate":
        updated.analysis_state.cross_disease_validation.status = status
        disease_count = (
            state.data_state.group_state.group_count
            if (
                state.data_state.group_state.group_field == "disease.name"
                or (
                    state.data_state.group_state.group_field is not None
                    and state.data_state.group_state.group_field.endswith(".disease")
                )
            ) else 0
        )
        if disease_count == 0:
            disease_count = _distinct_catalog_dimension_count(
                source_payloads, catalog, "disease"
            )
        updated.analysis_state.cross_disease_validation.disease_count = disease_count
        updated.analysis_state.cross_disease_validation.heterogeneity = "unknown"
    elif action == "stratified_analysis":
        fields = _plan_field_values(analysis_plan, "stratify_by", "dimensions")
        updated.analysis_state.stratified_analysis.status = status
        updated.analysis_state.stratified_analysis.stratify_fields = list(dict.fromkeys(fields))
        rows = _source_rows(source_payloads, result.sourceObservationIds)
        stratum_count = len(result.stratum_results)
        if fields and rows:
            # Source payloads use Java aliases.  Match the semantic suffix in
            # the actual typed plan; no physical column-name guessing occurs.
            typed_fields = _plan_field_values(analysis_plan, "stratify_by", "dimensions")
            aliases: list[str] = []
            for field_id in typed_fields:
                base_alias = f"a_{field_id.replace('.', '_')}"
                if any(base_alias in row for row in rows):
                    aliases.append(base_alias)
            if aliases and not result.stratum_results:
                stratum_count = len({
                    tuple(_value_text(row.get(alias)) for alias in aliases)
                    for row in rows
                })
        updated.analysis_state.stratified_analysis.stratum_count = stratum_count
        updated.analysis_state.stratified_analysis.positive_stratum_count = int(
            metrics.get("positive_stratum_count", 0.0)
        )
        updated.analysis_state.stratified_analysis.negative_stratum_count = int(
            metrics.get("negative_stratum_count", 0.0)
        )
        updated.analysis_state.stratified_analysis.neutral_stratum_count = int(
            metrics.get("neutral_stratum_count", 0.0)
        )
        updated.analysis_state.stratified_analysis.effect_min = metrics.get("effect_min")
        updated.analysis_state.stratified_analysis.effect_max = metrics.get("effect_max")
        updated.analysis_state.stratified_analysis.heterogeneity = "unknown"
    elif action == "analyze_projection":
        updated.analysis_state.projection_analysis.status = status
        # The approved typed operator does not currently emit a projection
        # result cardinality.  Accept it only if a future/legacy bounded
        # result explicitly supplies ``result_count``; never reinterpret
        # rowsAnalyzed or topFeatures as that scientific quantity.
        projection_count = metrics.get("result_count")
        updated.analysis_state.projection_analysis.result_count = (
            int(projection_count)
            if projection_count is not None
            and projection_count >= 0
            and float(projection_count).is_integer()
            else 0
        )

    return updated


def update_from_evidence_result(
    state: ScientificDecisionState,
    evidence: Sequence[LiteratureEvidenceItem] | Sequence[object],
    *,
    observation_status: str = "VALIDATED",
) -> ScientificDecisionState:
    """Compress retrieval evidence by its existing direction labels."""

    updated = _copy(state)
    items = list(evidence)
    def direction(item: object) -> object:
        if isinstance(item, Mapping):
            return item.get("direction")
        return getattr(item, "direction", None)

    support = sum(1 for item in items if direction(item) == "supporting")
    conflict = sum(1 for item in items if direction(item) == "contrary")
    context = sum(1 for item in items if direction(item) == "context")
    updated.evidence_state.evidence_count = len(items)
    updated.evidence_state.support_count = support
    updated.evidence_state.conflict_count = conflict
    updated.evidence_state.context_count = context
    if observation_status in {"FAILED", "REJECTED"}:
        updated.evidence_state.status = "failed"
    elif not items:
        updated.evidence_state.status = "insufficient"
    else:
        updated.evidence_state.status = "completed"
    # The current retrieval layer has no deterministic consistency aggregator;
    # direction counts are retained while consistency stays unknown.
    updated.evidence_state.consistency = "unknown"
    return updated


def remaining_objectives(state: ScientificDecisionState) -> list[ScientificObjective]:
    """Compute required objectives from completed facts, never from policy hints."""

    completed = {
        "group_comparison": state.analysis_state.group_comparison.status == "completed",
        "projection_analysis": state.analysis_state.projection_analysis.status == "completed",
        "stratified_analysis": state.analysis_state.stratified_analysis.status == "completed",
        "confounder_assessment": state.analysis_state.confounder_adjustment.status == "completed",
        "cross_project_validation": state.analysis_state.cross_project_validation.status == "completed",
        "cross_disease_validation": state.analysis_state.cross_disease_validation.status == "completed",
        "evidence_support": state.evidence_state.status == "completed",
    }
    return [objective for objective in state.task.objectives if not completed.get(objective, False)]


def update_progress(
    state: ScientificDecisionState,
    action_name: str,
    *,
    successful: bool = True,
    completed: bool | None = None,
    remaining_objectives_override: Sequence[ScientificObjective] | None = None,
) -> ScientificDecisionState:
    """Record an executed action and recompute remaining objectives.

    ``successful`` counts a validated execution in the runtime history.  A
    ``PARTIAL`` observation is still an executed action, but is not added to
    ``completed_actions`` unless callers explicitly mark it completed.

    ``remaining_objectives_override`` is used only for a Runtime terminal
    decision.  It preserves the already-resolved active objective set (and,
    importantly, does not resurrect objectives that Runtime marked blocked).
    Ordinary observation updates continue to derive remaining objectives from
    the Decision State facts as before.
    """

    updated = _copy(state)
    if successful:
        counts = dict(updated.progress.action_counts)
        counts[action_name] = counts.get(action_name, 0) + 1
        updated.progress.action_counts = counts
        if completed is not False and action_name not in updated.progress.completed_actions:
            updated.progress.completed_actions.append(action_name)
        updated.progress.last_action = action_name
        updated.progress.action_count += 1
    if remaining_objectives_override is None:
        updated.progress.remaining_objectives = remaining_objectives(updated)
    else:
        updated.progress.remaining_objectives = list(remaining_objectives_override)
    return updated


__all__ = [
    "remaining_objectives",
    "update_from_analysis_result",
    "update_from_evidence_result",
    "update_from_query_result",
    "update_progress",
]
