"""Capability matching for typed AnalysisPlans.

The registry is deliberately narrower than the scientific policy.  It does
not choose an action or rank competing plans; it only validates a materialized
plan against the current semantic catalog and says which already-existing
execution channel may run it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from mico_agent_runtime.contracts.materialization import AnalysisPlan
from mico_agent_runtime.contracts.schema_catalog import SchemaSemanticCatalog


AnalysisCapabilityMode = Literal["SUPPORTED_TYPED", "SUPPORTED_GENERATED", "UNSUPPORTED"]


@dataclass(frozen=True)
class AnalysisCapabilityContext:
    """Runtime facts needed beyond the metadata-only Schema Catalog.

    The Java payload remains outside this contract.  Runtime supplies only
    opaque observation IDs, semantic field IDs proven to be present in those
    observations, and deterministic distinct counts for dimensions.
    """

    available_observation_ids: Sequence[str] = ()
    available_fields: Sequence[str] = ()
    # When supplied, this gives the stronger per-observation proof.  The
    # union in ``available_fields`` remains useful for legacy callers.
    observation_fields: Mapping[str, Sequence[str]] = field(default_factory=dict)
    distinct_counts: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalysisCapabilityMatch:
    mode: AnalysisCapabilityMode
    capability_code: str
    reason_code: str
    analysis_type: str

    @property
    def execution_mode(self) -> AnalysisCapabilityMode:
        """Alias used by trace/eval consumers without adding plan control."""

        return self.mode


def _unsupported(plan: AnalysisPlan, code: str, reason: str) -> AnalysisCapabilityMatch:
    return AnalysisCapabilityMatch(
        mode="UNSUPPORTED",
        capability_code=code,
        reason_code=reason,
        analysis_type=plan.analysis_type,
    )


def _generated(plan: AnalysisPlan, code: str, reason: str) -> AnalysisCapabilityMatch:
    return AnalysisCapabilityMatch(
        mode="SUPPORTED_GENERATED",
        capability_code=code,
        reason_code=reason,
        analysis_type=plan.analysis_type,
    )


def _typed(plan: AnalysisPlan, code: str, reason: str) -> AnalysisCapabilityMatch:
    return AnalysisCapabilityMatch(
        mode="SUPPORTED_TYPED",
        capability_code=code,
        reason_code=reason,
        analysis_type=plan.analysis_type,
    )


def _catalog_fields(
    catalog: SchemaSemanticCatalog | None,
) -> dict[str, tuple[str, frozenset[str], str, bool]]:
    """Return ``field_id -> (data_type, capabilities, status, sensitive)``."""

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


def _references(plan: AnalysisPlan) -> set[str]:
    return {
        value
        for value in (
            plan.outcome,
            plan.feature_field,
            plan.group_field,
            plan.validation_field,
            *plan.covariates,
            *plan.stratify_by,
        )
        if value is not None
    }


def _method_supported_for_action(plan: AnalysisPlan) -> bool:
    family = plan.method.family
    allowed: dict[str, set[str]] = {
        "group_comparison": {"auto", "parametric", "nonparametric", "bootstrap", "custom"},
        "confounder_adjustment": {"auto", "regression", "custom"},
        "stratified_comparison": {"auto", "stratified", "custom"},
        "cross_project_validation": {"auto", "validation", "custom"},
        "cross_disease_validation": {"auto", "validation", "custom"},
        "projection": {"auto", "custom"},
    }
    return family in allowed.get(plan.analysis_type, set())


def _validate_common(
    plan: AnalysisPlan,
    catalog: SchemaSemanticCatalog | None,
    context: AnalysisCapabilityContext,
) -> AnalysisCapabilityMatch | None:
    if catalog is None:
        return _unsupported(plan, "CATALOG_REQUIRED", "SCHEMA_CATALOG_REQUIRED")
    fields = _catalog_fields(catalog)
    if not fields:
        return _unsupported(plan, "FIELD_CATALOG_EMPTY", "SCIENTIFIC_FIELD_CATALOG_REQUIRED")
    if not _method_supported_for_action(plan):
        return _unsupported(
            plan,
            "METHOD_ACTION_MISMATCH",
            "ANALYSIS_METHOD_NOT_SUPPORTED_FOR_ACTION",
        )
    if context.available_observation_ids:
        unknown_observations = set(plan.source_observation_ids) - set(
            context.available_observation_ids
        )
        if unknown_observations:
            return _unsupported(
                plan,
                "SOURCE_OBSERVATION_MISSING",
                "ANALYSIS_SOURCE_OBSERVATION_NOT_AVAILABLE",
            )
    references = _references(plan)
    if context.observation_fields:
        for observation_id in plan.source_observation_ids:
            observed = context.observation_fields.get(observation_id)
            if observed is None or not references.issubset(set(observed)):
                return _unsupported(
                    plan,
                    "OBSERVATION_FIELD_MISSING",
                    "ANALYSIS_FIELD_NOT_PRESENT_IN_OBSERVATION",
                )
    elif context.available_fields:
        missing = references - set(context.available_fields)
        if missing:
            return _unsupported(
                plan,
                "OBSERVATION_FIELD_MISSING",
                "ANALYSIS_FIELD_NOT_PRESENT_IN_OBSERVATION",
            )
    for field_id in references:
        metadata = fields.get(field_id)
        if metadata is None:
            return _unsupported(plan, "FIELD_NOT_IN_CATALOG", "ANALYSIS_FIELD_NOT_IN_CATALOG")
        _data_type, _capabilities, status, sensitive = metadata
        if status != "verified" or sensitive:
            return _unsupported(
                plan,
                "FIELD_NOT_EXECUTABLE",
                "ANALYSIS_FIELD_NOT_VERIFIED_OR_IS_SENSITIVE",
            )
    return None


def _require_capability(
    plan: AnalysisPlan,
    fields: dict[str, tuple[str, frozenset[str], str, bool]],
    field_id: str | None,
    capability: str,
    code: str,
    reason: str,
) -> AnalysisCapabilityMatch | None:
    if field_id is None:
        return _unsupported(plan, code, reason)
    metadata = fields.get(field_id)
    if metadata is None or capability not in metadata[1]:
        return _unsupported(plan, code, reason)
    return None


def _require_numeric_outcome(
    plan: AnalysisPlan,
    fields: dict[str, tuple[str, frozenset[str], str, bool]],
) -> AnalysisCapabilityMatch | None:
    if plan.outcome is None:
        return _unsupported(plan, "OUTCOME_REQUIRED", "NUMERIC_OUTCOME_REQUIRED")
    metadata = fields.get(plan.outcome)
    if metadata is None:
        return _unsupported(plan, "OUTCOME_NOT_IN_CATALOG", "NUMERIC_OUTCOME_REQUIRED")
    data_type, capabilities, _status, _sensitive = metadata
    if "outcome" not in capabilities or data_type not in {"integer", "number"}:
        return _unsupported(
            plan,
            "OUTCOME_CAPABILITY_REQUIRED",
            "FIELD_IS_NOT_A_NUMERIC_SCIENTIFIC_OUTCOME",
        )
    return None


def _require_distinct(
    plan: AnalysisPlan,
    context: AnalysisCapabilityContext,
    field_id: str,
    code: str,
    reason: str,
) -> AnalysisCapabilityMatch | None:
    # Missing counts mean the caller has not supplied an observation summary;
    # catalog validation can still proceed.  A supplied count is authoritative.
    if field_id not in context.distinct_counts:
        return None
    count = context.distinct_counts[field_id]
    if isinstance(count, bool) or not isinstance(count, int) or count < 2:
        return _unsupported(plan, code, reason)
    return None


def _require_exact_distinct(
    plan: AnalysisPlan,
    context: AnalysisCapabilityContext,
    field_id: str,
    code: str,
    reason: str,
) -> AnalysisCapabilityMatch | None:
    if field_id not in context.distinct_counts:
        return None
    count = context.distinct_counts[field_id]
    if isinstance(count, bool) or not isinstance(count, int) or count != 2:
        return _unsupported(plan, code, reason)
    return None


def match_analysis_capability(
    plan: AnalysisPlan,
    catalog: SchemaSemanticCatalog | None,
    context: AnalysisCapabilityContext | Mapping[str, object] | None = None,
    *,
    available_observation_ids: Sequence[str] | None = None,
    available_fields: Sequence[str] | None = None,
    distinct_counts: Mapping[str, int] | None = None,
) -> AnalysisCapabilityMatch:
    """Classify a validated AnalysisPlan as typed, generated, or unsupported."""

    if isinstance(context, Mapping):
        context = AnalysisCapabilityContext(**context)
    runtime_context = context or AnalysisCapabilityContext()
    if any(value is not None for value in (
        available_observation_ids,
        available_fields,
        distinct_counts,
    )):
        runtime_context = AnalysisCapabilityContext(
            available_observation_ids=(
                runtime_context.available_observation_ids
                if available_observation_ids is None else available_observation_ids
            ),
            available_fields=(
                runtime_context.available_fields
                if available_fields is None else available_fields
            ),
            observation_fields=runtime_context.observation_fields,
            distinct_counts=(
                runtime_context.distinct_counts
                if distinct_counts is None else distinct_counts
            ),
        )

    common_error = _validate_common(plan, catalog, runtime_context)
    if common_error is not None:
        return common_error
    fields = _catalog_fields(catalog)

    outcome_error = _require_numeric_outcome(plan, fields)
    if outcome_error is not None:
        return outcome_error

    # Abundance outcomes are commonly returned at sample × feature grain. If
    # the catalog exposes a verified feature dimension, typed comparisons and
    # adjustments must bind it explicitly so species rows are never pooled as
    # independent observations. Legacy catalogs without such a dimension keep
    # the historical typed contract.
    feature_dimensions = [
        field_id for field_id, (_dtype, capabilities, status, sensitive) in fields.items()
        if status == "verified" and not sensitive
        and field_id.endswith(".feature")
    ]
    if plan.analysis_type in {"group_comparison", "confounder_adjustment"} \
            and plan.outcome == "abundance.value" \
            and feature_dimensions and plan.feature_field is None:
        return _unsupported(
            plan,
            "FEATURE_DIMENSION_REQUIRED",
            "ABUNDANCE_ANALYSIS_REQUIRES_FEATURE_FIELD",
        )
    if plan.feature_field is not None:
        feature_metadata = fields.get(plan.feature_field)
        if feature_metadata is None:
            return _unsupported(
                plan, "FEATURE_FIELD_NOT_IN_CATALOG", "FEATURE_FIELD_NOT_IN_CATALOG"
            )
        _dtype, capabilities, status, sensitive = feature_metadata
        if status != "verified" or sensitive or "dimension" not in capabilities:
            return _unsupported(
                plan,
                "FEATURE_FIELD_NOT_EXECUTABLE",
                "FEATURE_FIELD_MUST_BE_VERIFIED_DIMENSION",
            )

    if plan.analysis_type == "group_comparison":
        group_error = _require_capability(
            plan, fields, plan.group_field, "dimension",
            "GROUP_DIMENSION_REQUIRED", "GROUP_FIELD_MUST_BE_A_DIMENSION",
        )
        if group_error is not None:
            return group_error
        assert plan.group_field is not None
        coverage_error = _require_exact_distinct(
            plan, runtime_context, plan.group_field,
            "GROUP_COVERAGE_REQUIRED", "GROUP_FIELD_REQUIRES_EXACTLY_TWO_VALUES",
        )
        if coverage_error is not None:
            return coverage_error
        typed_metrics = {
            # ``effect`` is the v2 materializer spelling for the same
            # inferential quantity emitted by the approved typed operator as
            # ``effect_size``.  Keep both names in the capability contract so
            # a valid compare plan cannot be misclassified as generated-only.
            "count", "mean", "median", "effect_size", "effect", "p_value",
            "confidence_interval",
        }
        if plan.method.family == "auto" and plan.method.name is None \
                and not (set(plan.metrics) - typed_metrics):
            return _typed(
                plan,
                "TYPED_GROUP_COMPARISON",
                "MATCHED_STANDARD_GROUP_COMPARISON",
            )
        return _generated(
            plan,
            "GENERATED_GROUP_COMPARISON",
            "GROUP_COMPARISON_METHOD_OR_METRIC_REQUIRES_GENERATED_EXECUTION",
        )

    if plan.analysis_type == "confounder_adjustment":
        group_error = _require_capability(
            plan, fields, plan.group_field, "dimension",
            "GROUP_DIMENSION_REQUIRED", "GROUP_FIELD_MUST_BE_A_DIMENSION",
        )
        if group_error is not None:
            return group_error
        for covariate in plan.covariates:
            error = _require_capability(
                plan, fields, covariate, "covariate",
                "COVARIATE_CAPABILITY_REQUIRED", "COVARIATE_FIELD_REQUIRED",
            )
            if error is not None:
                return error
        group_coverage_error = _require_exact_distinct(
            plan, runtime_context, plan.group_field or "",
            "GROUP_COVERAGE_REQUIRED", "GROUP_FIELD_REQUIRES_EXACTLY_TWO_VALUES",
        )
        if group_coverage_error is not None:
            return group_coverage_error
        if plan.method.family in {"auto", "regression"}:
            return _typed(
                plan,
                "TYPED_CONFOUNDER_ADJUSTMENT",
                "MATCHED_STANDARD_LINEAR_ADJUSTMENT",
            )
        return _generated(
            plan,
            "GENERATED_CONFOUNDER_ADJUSTMENT",
            "CONFOUNDER_ADJUSTMENT_METHOD_REQUIRES_GENERATED_EXECUTION",
        )

    if plan.analysis_type == "stratified_comparison":
        group_error = _require_capability(
            plan, fields, plan.group_field, "dimension",
            "GROUP_DIMENSION_REQUIRED", "GROUP_FIELD_MUST_BE_A_DIMENSION",
        )
        if group_error is not None:
            return group_error
        numeric_stratifiers: list[str] = []
        for stratifier in plan.stratify_by:
            error = _require_capability(
                plan, fields, stratifier, "stratifier",
                "STRATIFIER_CAPABILITY_REQUIRED", "STRATIFY_FIELD_MUST_BE_A_STRATIFIER",
            )
            if error is not None:
                return error
            metadata = fields.get(stratifier)
            if metadata is not None and metadata[0] not in {"string", "boolean"}:
                numeric_stratifiers.append(stratifier)
        group_coverage_error = _require_exact_distinct(
            plan, runtime_context, plan.group_field or "",
            "GROUP_COVERAGE_REQUIRED", "GROUP_FIELD_REQUIRES_EXACTLY_TWO_VALUES",
        )
        if group_coverage_error is not None:
            return group_coverage_error
        if plan.numeric_stratification is not None and not numeric_stratifiers:
            return _unsupported(
                plan,
                "NUMERIC_STRATIFICATION_SPEC_REQUIRES_NUMERIC_FIELD",
                "NUMERIC_STRATIFICATION_FIELD_MUST_BE_NUMERIC",
            )
        if numeric_stratifiers:
            # Numeric binning is typed only when the materializer supplies the
            # closed NumericStratificationSpec.  Without it we retain the
            # historical generated compatibility route rather than silently
            # calling the old exact-scalar operator "typed".
            if len(numeric_stratifiers) != 1:
                return _unsupported(
                    plan,
                    "NUMERIC_STRATIFICATION_SHAPE_UNSUPPORTED",
                    "NUMERIC_STRATIFICATION_REQUIRES_ONE_FIELD",
                )
            spec = plan.numeric_stratification
            if spec is None:
                return _generated(
                    plan,
                    "GENERATED_STRATIFIED_COMPARISON",
                    "NON_CATEGORICAL_STRATIFIER_REQUIRES_GENERATED_EXECUTION",
                )
            if spec.stratifier != numeric_stratifiers[0]:
                return _unsupported(
                    plan,
                    "NUMERIC_STRATIFICATION_SPEC_MISMATCH",
                    "NUMERIC_STRATIFICATION_SPEC_FIELD_MISMATCH",
                )
            if plan.method.family in {"auto", "stratified"}:
                return _typed(
                    plan,
                    "TYPED_NUMERIC_STRATIFIED_COMPARISON",
                    "MATCHED_SAMPLE_LEVEL_NUMERIC_STRATIFICATION",
                )
            return _generated(
                plan,
                "GENERATED_STRATIFIED_COMPARISON",
                "STRATIFIED_METHOD_REQUIRES_GENERATED_EXECUTION",
            )
        if plan.method.family in {"auto", "stratified"}:
            return _typed(
                plan,
                "TYPED_STRATIFIED_COMPARISON",
                "MATCHED_CATEGORICAL_STRATIFIED_COMPARISON",
            )
        return _generated(
            plan,
            "GENERATED_STRATIFIED_COMPARISON",
            "STRATIFIED_METHOD_REQUIRES_GENERATED_EXECUTION",
        )

    if plan.analysis_type in {"cross_project_validation", "cross_disease_validation"}:
        group_error = _require_capability(
            plan, fields, plan.group_field, "dimension",
            "GROUP_DIMENSION_REQUIRED", "GROUP_FIELD_MUST_BE_A_DIMENSION",
        )
        if group_error is not None:
            return group_error
        validation_error = _require_capability(
            plan, fields, plan.validation_field, "dimension",
            "VALIDATION_DIMENSION_REQUIRED", "VALIDATION_FIELD_MUST_BE_A_DIMENSION",
        )
        if validation_error is not None:
            return validation_error
        assert plan.group_field is not None and plan.validation_field is not None
        group_coverage_error = _require_exact_distinct(
            plan, runtime_context, plan.group_field,
            "GROUP_COVERAGE_REQUIRED", "GROUP_FIELD_REQUIRES_EXACTLY_TWO_VALUES",
        )
        if group_coverage_error is not None:
            return group_coverage_error
        validation_coverage_error = _require_distinct(
            plan, runtime_context, plan.validation_field,
            "VALIDATION_COVERAGE_REQUIRED",
            "VALIDATION_FIELD_REQUIRES_AT_LEAST_TWO_VALUES",
        )
        if validation_coverage_error is not None:
            return validation_coverage_error
        if plan.analysis_type == "cross_project_validation" \
                and plan.method.family in {"auto", "validation"}:
            return _typed(
                plan,
                "TYPED_CROSS_PROJECT_VALIDATION",
                "MATCHED_PER_PROJECT_TWO_GROUP_COMPARISON",
            )
        return _generated(
            plan,
            "GENERATED_CROSS_VALIDATION",
            "CROSS_VALIDATION_METHOD_OR_TYPE_REQUIRES_GENERATED_EXECUTION",
        )

    if plan.analysis_type == "projection":
        return _generated(
            plan,
            "GENERATED_PROJECTION",
            "PROJECTION_SEMANTICS_NOT_REGISTERED_AS_TYPED",
        )

    # The Pydantic AnalysisType literal normally makes this unreachable.  Keep
    # the runtime branch fail-closed for model_construct/legacy callers.
    return _unsupported(plan, "ANALYSIS_TYPE_UNSUPPORTED", "ANALYSIS_TYPE_NOT_REGISTERED")


class AnalysisCapabilityRegistry:
    """Convenience object for callers that reuse one current catalog."""

    def __init__(
        self,
        catalog: SchemaSemanticCatalog | None,
        context: AnalysisCapabilityContext | None = None,
    ) -> None:
        self.catalog = catalog
        self.context = context

    def match(
        self,
        plan: AnalysisPlan,
        context: AnalysisCapabilityContext | None = None,
    ) -> AnalysisCapabilityMatch:
        return match_analysis_capability(
            plan,
            self.catalog,
            context if context is not None else self.context,
        )

    def match_analysis_capability(
        self,
        plan: AnalysisPlan,
        context: AnalysisCapabilityContext | None = None,
    ) -> AnalysisCapabilityMatch:
        """Named alias for callers that mirror the module-level API."""

        return self.match(plan, context)


__all__ = [
    "AnalysisCapabilityContext",
    "AnalysisCapabilityMatch",
    "AnalysisCapabilityMode",
    "AnalysisCapabilityRegistry",
    "match_analysis_capability",
]
