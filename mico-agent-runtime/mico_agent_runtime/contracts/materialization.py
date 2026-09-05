"""Typed, catalog-oriented plans produced by the Action Materializer.

These models deliberately contain semantic IDs rather than physical table or
column names.  Java owns the mapping from those IDs to SQL identifiers and
join keys.  The plans are transient execution inputs; they are not DPO/SFT
training records and must not be persisted with raw values.
"""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, StrictFloat, StrictInt, StrictStr, field_validator, model_validator

from .base import ClosedModel
from .schema_catalog import SchemaSemanticCatalog


SemanticFieldId = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z][a-z0-9_]{1,63}\.[a-z][a-z0-9_]{1,63}$", max_length=128),
]
RelationId = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z][a-z0-9_]{1,63}$", max_length=64),
]
EntityId = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z][a-z0-9_]{1,63}$", max_length=64),
]
ScalarValue = str | int | float | bool | None
ObservationId = Annotated[
    str,
    StringConstraints(pattern=r"^observation-[0-9a-f]{32}$", max_length=45),
]

# Analysis methods are intentionally a small, closed vocabulary.  The
# materializer can describe a method without being allowed to choose an
# execution path; that decision belongs to the Runtime capability registry.
AnalysisMethodFamily = Literal[
    "auto",
    "parametric",
    "nonparametric",
    "regression",
    "bootstrap",
    "stratified",
    "validation",
    "custom",
]
AnalysisMethodName = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,63}$", max_length=64),
]
_StrictJsonScalar = StrictStr | StrictInt | StrictFloat | bool | None
_MethodParameterValue = _StrictJsonScalar | list[_StrictJsonScalar]
_ALLOWED_METHOD_PARAMETERS = frozenset({
    "iterations",
    "confidence_level",
    "bin_count",
    "bin_boundaries",
    "reference_group",
})


class AnalysisMethod(ClosedModel):
    """Bounded description of *how* a plan may be analysed.

    ``parameters`` is deliberately a dictionary for wire compatibility, but
    its keys and scalar/list value shapes are closed here.  In particular it
    cannot carry ``execution_mode`` or arbitrary model-authored DSL fields.
    """

    family: AnalysisMethodFamily = "auto"
    name: AnalysisMethodName | None = None
    parameters: dict[
        Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,63}$", max_length=64)],
        _MethodParameterValue,
    ] = Field(default_factory=dict, max_length=8)

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, value: dict[str, object]) -> dict[str, object]:
        unknown = set(value) - _ALLOWED_METHOD_PARAMETERS
        if unknown:
            raise ValueError(
                "analysis method contains unknown parameter(s): "
                + ",".join(sorted(unknown))
            )
        iterations = value.get("iterations")
        if iterations is not None and (
            isinstance(iterations, bool)
            or not isinstance(iterations, int)
            or not 1 <= iterations <= 100_000
        ):
            raise ValueError("iterations must be an integer between 1 and 100000")
        confidence = value.get("confidence_level")
        if confidence is not None and (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0.0 < float(confidence) < 1.0
        ):
            raise ValueError("confidence_level must be strictly between 0 and 1")
        bin_count = value.get("bin_count")
        if bin_count is not None and (
            isinstance(bin_count, bool)
            or not isinstance(bin_count, int)
            or not 1 <= bin_count <= 100
        ):
            raise ValueError("bin_count must be an integer between 1 and 100")
        boundaries = value.get("bin_boundaries")
        if boundaries is not None:
            if not isinstance(boundaries, list) or not 2 <= len(boundaries) <= 101:
                raise ValueError("bin_boundaries must contain two to 101 values")
            if any(
                isinstance(item, bool) or not isinstance(item, (int, float))
                for item in boundaries
            ):
                raise ValueError("bin_boundaries must contain numeric values")
            if any(left >= right for left, right in zip(boundaries, boundaries[1:])):
                raise ValueError("bin_boundaries must be strictly increasing")
        reference_group = value.get("reference_group")
        if reference_group is not None and (
            not isinstance(reference_group, str)
            or not 1 <= len(reference_group) <= 128
        ):
            raise ValueError("reference_group must be a bounded non-empty string")
        return value


class QueryFilter(ClosedModel):
    field: SemanticFieldId
    operator: Literal["eq", "neq", "in", "is_null", "is_not_null"]
    value: ScalarValue | list[ScalarValue] | None = None

    @model_validator(mode="after")
    def validate_value_shape(self) -> "QueryFilter":
        if self.operator in {"is_null", "is_not_null"} and self.value is not None:
            raise ValueError("null operators cannot carry a filter value")
        if self.operator in {"eq", "neq"} and isinstance(self.value, list):
            raise ValueError("scalar operators require one scalar filter value")
        if self.operator in {"eq", "neq"} and self.value is None:
            raise ValueError("eq and neq require a non-null filter value")
        if self.operator == "in":
            if not isinstance(self.value, list) or not self.value or len(self.value) > 20:
                raise ValueError("in requires one to twenty scalar values")
            if any(item is None for item in self.value):
                raise ValueError("in filter values must be non-null")
        if isinstance(self.value, list) and any(isinstance(item, (dict, list, tuple)) for item in self.value):
            raise ValueError("filter values must remain scalar")
        return self


class QueryAggregation(ClosedModel):
    field: SemanticFieldId
    op: Literal["count", "mean", "min", "max", "sum"]


class QueryPlan(ClosedModel):
    schemaVersion: Literal["query-plan-v1"] = "query-plan-v1"
    root_entity: EntityId
    relation_path: list[RelationId] = Field(default_factory=list, max_length=2)
    select_fields: list[SemanticFieldId] = Field(min_length=1, max_length=16)
    aggregations: list[QueryAggregation] = Field(default_factory=list, max_length=8)
    filters: list[QueryFilter] = Field(default_factory=list, max_length=16)
    group_by: list[SemanticFieldId] = Field(default_factory=list, max_length=8)
    # ``limit`` is the bounded result-row cap.  Sample-bounded abundance
    # reads may legitimately return roughly ``samples × features`` rows, so
    # the typed read surface is wider than the legacy one-row SQL path while
    # remaining finite and Java-enforced.
    limit: int = Field(strict=True, ge=1, le=20_000)
    # Runtime-owned technical controls.  Materializers are not instructed to
    # emit these fields; the Dynamic Scientific Runtime attaches them to a
    # raw abundance projection after the Action/plan contract is validated.
    # They select unique root samples before the one-to-many abundance join;
    # they do not represent a scientific preference or a policy hint.
    sample_limit_per_group: int | None = Field(default=None, strict=True, ge=1, le=50)
    sample_limit_group_field: SemanticFieldId | None = None

    @field_validator("relation_path", "select_fields", "group_by")
    @classmethod
    def reject_duplicates(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("query plan path or field list contains duplicates")
        return value

    @model_validator(mode="after")
    def validate_grouping(self) -> "QueryPlan":
        selected = set(self.select_fields)
        if not set(self.group_by).issubset(selected):
            raise ValueError("group_by fields must be selected fields")
        if not self.aggregations and self.group_by:
            raise ValueError("group_by requires an aggregation")
        if self.aggregations and set(self.select_fields) != set(self.group_by):
            raise ValueError(
                "every selected non-aggregated field must be included in group_by"
            )
        if self.sample_limit_per_group is not None:
            if self.aggregations or self.group_by:
                raise ValueError(
                    "sample_limit_per_group requires a raw, ungrouped projection"
                )
            if self.sample_limit_group_field is None:
                raise ValueError(
                    "sample_limit_per_group requires sample_limit_group_field"
                )
        elif self.sample_limit_group_field is not None:
            raise ValueError(
                "sample_limit_group_field requires sample_limit_per_group"
            )
        return self


AnalysisType = Literal[
    "group_comparison",
    "stratified_comparison",
    "confounder_adjustment",
    "projection",
    "cross_project_validation",
    "cross_disease_validation",
]

AnalysisMetric = Literal[
    # ``effect_size`` is the metric emitted by the current typed operator;
    # ``effect`` is accepted by the v2 plan so newer materializers can name
    # the same scientific quantity without widening the operator contract.
    "count",
    "mean",
    "median",
    "effect_size",
    "effect",
    "p_value",
    "confidence_interval",
]
AnalysisGoal = Annotated[str, StringConstraints(min_length=1, max_length=512)]

NumericStratificationStrategy = Literal["quantile", "fixed_bins", "custom_cut_points"]
NumericMissingValuePolicy = Literal["drop", "separate_stratum"]
NumericMultipleTesting = Literal["benjamini_hochberg"]
NumericCutPoint = StrictInt | StrictFloat


class NumericStratificationSpec(ClosedModel):
    """Closed execution parameters for a typed numeric stratification.

    The materializer may select the semantic stratifier and a bounded binning
    strategy, but it cannot supply rows, labels, or code.  Runtime derives
    quantile cut points from the current sample-level observation and reports
    eligible/insufficient strata in the result.
    """

    stratifier: SemanticFieldId
    strategy: NumericStratificationStrategy = "quantile"
    bin_count: int | None = Field(default=None, strict=True, ge=2, le=8)
    cut_points: list[NumericCutPoint] = Field(default_factory=list, max_length=7)
    # The minimum is intentionally required rather than silently chosen by
    # Runtime.  It is an explicit plan parameter and insufficient strata are
    # reported when the observed sample counts do not meet it.
    min_samples_per_group: int = Field(strict=True, ge=2, le=1000)
    missing_value_policy: NumericMissingValuePolicy = "drop"
    multiple_testing: NumericMultipleTesting = "benjamini_hochberg"

    @field_validator("cut_points")
    @classmethod
    def validate_cut_points(cls, value: list[int | float]) -> list[int | float]:
        if any(not math.isfinite(float(item)) for item in value):
            raise ValueError("numeric stratification cut_points must be finite")
        if any(left >= right for left, right in zip(value, value[1:])):
            raise ValueError("numeric stratification cut_points must be strictly increasing")
        return value

    @model_validator(mode="after")
    def validate_binning_shape(self) -> "NumericStratificationSpec":
        if self.strategy == "quantile":
            if self.bin_count is None:
                raise ValueError("quantile numeric stratification requires bin_count")
            if self.cut_points:
                raise ValueError("quantile numeric stratification cannot provide cut_points")
        else:
            if self.bin_count is None:
                raise ValueError(f"{self.strategy} numeric stratification requires bin_count")
            if len(self.cut_points) != self.bin_count - 1:
                raise ValueError(
                    f"{self.strategy} numeric stratification requires bin_count - 1 cut_points"
                )
        return self


class AnalysisPlan(ClosedModel):
    schemaVersion: Literal["analysis-plan-v2"] = "analysis-plan-v2"
    analysis_type: AnalysisType
    source_observation_ids: list[ObservationId] = Field(min_length=1, max_length=8)
    outcome: SemanticFieldId | None = None
    # ``feature_field`` is required for feature-aware abundance analyses. It
    # prevents the typed operator from pooling multiple feature rows into one
    # pseudo-outcome while remaining optional for legacy non-feature plans.
    feature_field: SemanticFieldId | None = None
    group_field: SemanticFieldId | None = None
    covariates: list[SemanticFieldId] = Field(default_factory=list, max_length=8)
    stratify_by: list[SemanticFieldId] = Field(default_factory=list, max_length=8)
    # Numeric stratification is optional for backwards compatibility.  When a
    # numeric stratifier is used without this explicit spec, the Capability
    # Registry deliberately keeps the legacy generated compatibility route;
    # it must never be mislabeled as a typed result.
    numeric_stratification: NumericStratificationSpec | None = None
    validation_field: SemanticFieldId | None = None
    method: AnalysisMethod = Field(default_factory=AnalysisMethod)
    # Kept optional for compatibility with older deterministic test doubles;
    # all model-generated v2 plans are instructed to provide it.
    analysis_goal: AnalysisGoal = "unspecified analysis goal"
    metrics: list[AnalysisMetric] = Field(min_length=1, max_length=8)

    @field_validator("source_observation_ids", "covariates", "stratify_by", "metrics")
    @classmethod
    def reject_duplicates(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("analysis plan contains duplicate references")
        return value

    @model_validator(mode="after")
    def validate_action_shape(self) -> "AnalysisPlan":
        if self.analysis_type in {
            "group_comparison",
            "stratified_comparison",
            "confounder_adjustment",
            "projection",
            "cross_project_validation",
            "cross_disease_validation",
        } and not self.outcome:
            raise ValueError(f"{self.analysis_type} requires outcome")
        if self.analysis_type == "group_comparison" and not self.group_field:
            raise ValueError("group_comparison requires group_field")
        if self.analysis_type == "stratified_comparison":
            if not self.group_field:
                raise ValueError("stratified_comparison requires group_field")
            if not self.stratify_by:
                raise ValueError("stratified_comparison requires stratify_by")
            if self.numeric_stratification is not None:
                if self.numeric_stratification.stratifier not in self.stratify_by:
                    raise ValueError(
                        "numeric stratifier must be included in stratify_by"
                    )
                # The first typed numeric implementation has one numeric
                # partition axis.  Mixed/multi-axis shapes remain generated
                # or unsupported until they receive an explicit contract.
                if len(self.stratify_by) != 1:
                    raise ValueError(
                        "numeric stratification currently requires exactly one stratify_by field"
                    )
        elif self.numeric_stratification is not None:
            raise ValueError(
                "numeric_stratification is only valid for stratified_comparison"
            )
        if self.analysis_type == "confounder_adjustment":
            if not self.covariates:
                raise ValueError("confounder_adjustment requires covariates")
        if self.analysis_type in {"cross_project_validation", "cross_disease_validation"}:
            if not self.group_field:
                raise ValueError("cross validation requires group_field")
            if not self.validation_field:
                raise ValueError("cross validation requires validation_field")
            if self.group_field == self.validation_field:
                raise ValueError("cross validation group and validation fields must differ")
        return self


class RetrievalPlan(ClosedModel):
    schemaVersion: Literal["retrieval-plan-v1"] = "retrieval-plan-v1"
    topics: list[Annotated[str, StringConstraints(min_length=1, max_length=256)]] = Field(
        min_length=1, max_length=8
    )
    retrieval_mode: Literal["vector", "graph", "hybrid"] = "hybrid"
    top_k: int = Field(strict=True, ge=1, le=20)
    max_hops: int = Field(strict=True, ge=0, le=3)


def validate_query_plan_catalog(plan: QueryPlan, catalog: SchemaSemanticCatalog) -> QueryPlan:
    """Validate semantic IDs and ordered relation connectivity before Java.

    Java remains the authoritative compiler/validator. This early check gives
    the Materializer a bounded, opaque retry signal and prevents it from
    repeatedly emitting IDs that are absent from the current catalog.
    """

    if catalog is None:
        raise ValueError("query plan requires a schema catalog")
    entities_by_id = {
        (entity.entityId or entity.entityName): entity for entity in catalog.entities
    }
    entities_by_name = {entity.entityName: entity for entity in catalog.entities}
    root = entities_by_id.get(plan.root_entity)
    if root is None:
        raise ValueError("query plan root entity is not in catalog")
    active = {root.entityName}
    seen_relations: set[str] = set()
    for relation_id in plan.relation_path:
        if relation_id in seen_relations:
            raise ValueError("query plan relation path contains a duplicate")
        seen_relations.add(relation_id)
        relation = next(
            (item for item in catalog.joins if (item.relationId or "") == relation_id),
            None,
        )
        if relation is None or relation.relationshipStatus != "verified":
            raise ValueError("query plan relation is not enabled")
        left = entities_by_name.get(relation.leftEntity)
        right = entities_by_name.get(relation.rightEntity)
        if left is None or right is None:
            raise ValueError("query plan relation endpoint is unknown")
        left_active = left.entityName in active
        right_active = right.entityName in active
        if left_active == right_active:
            raise ValueError("query plan relation path is ambiguous or disconnected")
        if relation.cardinality in {"many_to_many", "unknown"}:
            raise ValueError("query plan relation cardinality is not safe")
        active.add(right.entityName if left_active else left.entityName)

    field_map: dict[str, tuple[object, object]] = {}
    for entity in catalog.entities:
        if entity.entityName not in active:
            continue
        entity_id = entity.entityId or entity.entityName
        for field in entity.fields:
            field_id = field.fieldId or f"{entity_id}.{field.name}"
            field_map[field_id] = (entity, field)
    for field_id in [
        *plan.select_fields,
        *plan.group_by,
        *(item.field for item in plan.aggregations),
        *(item.field for item in plan.filters),
        *([plan.sample_limit_group_field] if plan.sample_limit_group_field else []),
    ]:
        if field_id not in field_map:
            raise ValueError("query plan field is not in the active catalog")
    if plan.sample_limit_group_field is not None:
        if plan.sample_limit_group_field not in plan.select_fields:
            raise ValueError("sample limit group field must be selected")
        group_entity = plan.sample_limit_group_field.split(".", 1)[0]
        if group_entity != (root.entityId or root.entityName):
            raise ValueError("sample limit group field must belong to the query root")
    # Mirror Java's compiler capability gate before a plan crosses the HTTP
    # boundary.  A dimension/label such as ``sample.disease`` may be selected
    # or grouped, but it is not automatically a legal aggregation operand.
    # Non-count aggregations also require a numeric catalog field; this keeps
    # a model-generated coverage probe from treating a disease label as a
    # numeric measurement and gives the Materializer a precise repair signal.
    for aggregation in plan.aggregations:
        _entity, field = field_map[aggregation.field]
        if not field.aggregatable:
            raise ValueError("query plan aggregation field is not aggregatable")
        if aggregation.op != "count" and field.dataType not in {"integer", "number"}:
            raise ValueError("non-count query aggregation requires a numeric field")
    return plan


__all__ = [
    "AnalysisMethod",
    "AnalysisMethodFamily",
    "AnalysisMetric",
    "AnalysisPlan",
    "AnalysisType",
    "NumericCutPoint",
    "NumericMultipleTesting",
    "NumericMissingValuePolicy",
    "NumericStratificationSpec",
    "NumericStratificationStrategy",
    "QueryAggregation",
    "QueryFilter",
    "QueryPlan",
    "RetrievalPlan",
    "validate_query_plan_catalog",
]
