from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from mico_agent_runtime.contracts.materialization import AnalysisPlan
from mico_agent_runtime.contracts.schema_catalog import (
    SchemaEntitySemantics,
    SchemaFieldSemantics,
    SchemaSemanticCatalog,
)
from mico_agent_runtime.runtime.analysis_capability_registry import (
    AnalysisCapabilityContext,
    AnalysisCapabilityRegistry,
    match_analysis_capability,
)
from mico_agent_runtime.graph.generated_analysis import execute_typed_analysis


OBSERVATION = "observation-" + "a" * 32


def _catalog() -> SchemaSemanticCatalog:
    return SchemaSemanticCatalog(
        schemaVersion="schema-catalog-v1",
        source="java_schema_contract",
        generatedAt=datetime(2026, 8, 31, tzinfo=timezone.utc),
        entities=[
            SchemaEntitySemantics(
                entityName="sample",
                sourceTable="sample",
                fields=[
                    SchemaFieldSemantics(
                        fieldId="sample.disease",
                        name="disease",
                        dataType="string",
                        nullable=False,
                        semanticStatus="verified",
                        groupable=True,
                        scientificCapabilities=["dimension"],
                        description="Disease group",
                    ),
                    SchemaFieldSemantics(
                        fieldId="sample.age",
                        name="age",
                        dataType="integer",
                        nullable=True,
                        semanticStatus="verified",
                        groupable=True,
                        scientificCapabilities=["covariate", "stratifier"],
                        description="Age covariate",
                    ),
                    SchemaFieldSemantics(
                        fieldId="sample.sex",
                        name="sex",
                        dataType="string",
                        nullable=True,
                        semanticStatus="verified",
                        groupable=True,
                        scientificCapabilities=["stratifier"],
                        description="Sex stratum",
                    ),
                ],
            ),
            SchemaEntitySemantics(
                entityName="metadata",
                sourceTable="metadata",
                fields=[
                    SchemaFieldSemantics(
                        fieldId="metadata.project",
                        name="project",
                        dataType="string",
                        nullable=False,
                        semanticStatus="verified",
                        groupable=True,
                        scientificCapabilities=["dimension"],
                        description="Project dimension",
                    ),
                ],
            ),
            SchemaEntitySemantics(
                entityName="abundance",
                sourceTable="abundance",
                fields=[
                    SchemaFieldSemantics(
                        fieldId="abundance.value",
                        name="value",
                        dataType="number",
                        nullable=True,
                        semanticStatus="verified",
                        aggregatable=True,
                        scientificCapabilities=["outcome"],
                        description="Numeric abundance outcome",
                    ),
                ],
            ),
        ],
        queryRules=["select_or_with_only", "explicit_columns_only", "bounded_limit_required"],
    )


def _context(**counts: int) -> AnalysisCapabilityContext:
    fields = [
        "sample.disease",
        "sample.age",
        "sample.sex",
        "metadata.project",
        "abundance.value",
    ]
    return AnalysisCapabilityContext(
        available_observation_ids=[OBSERVATION],
        available_fields=fields,
        observation_fields={OBSERVATION: fields},
        distinct_counts=counts,
    )


def _plan(**updates: object) -> AnalysisPlan:
    payload = {
        "analysis_type": "group_comparison",
        "source_observation_ids": [OBSERVATION],
        "outcome": "abundance.value",
        "group_field": "sample.disease",
        "metrics": ["effect_size", "p_value"],
    }
    payload.update(updates)
    return AnalysisPlan.model_validate(payload)


def test_standard_two_group_numeric_comparison_is_typed() -> None:
    match = match_analysis_capability(
        _plan(),
        _catalog(),
        _context(**{"sample.disease": 2}),
    )
    assert match.mode == "SUPPORTED_TYPED"
    assert match.execution_mode == "SUPPORTED_TYPED"
    assert match.capability_code == "TYPED_GROUP_COMPARISON"


def test_typed_compare_plan_executes_typed_with_effect_metric() -> None:
    """The v2 ``effect`` spelling must stay on the approved typed route."""

    plan = _plan(metrics=["effect", "p_value"])
    match = match_analysis_capability(
        plan,
        _catalog(),
        _context(**{"sample.disease": 2}),
    )
    assert match.mode == "SUPPORTED_TYPED"
    result = execute_typed_analysis(
        plan,
        [
            {"sample.disease": "T2D", "abundance.value": 0.2},
            {"sample.disease": "T2D", "abundance.value": 0.3},
            {"sample.disease": "Healthy", "abundance.value": 0.8},
            {"sample.disease": "Healthy", "abundance.value": 0.9},
        ],
        4,
    )
    assert result.execution_mode == "typed"
    assert result.codeVersion == "typed-analysis-operator-v1"
    assert result.group_results
    assert result.metrics["effect"] == result.metrics["effect_size"]
    assert result.metrics["p_value"] >= 0.0


def test_typed_compare_never_silently_falls_back_to_generated() -> None:
    """A valid effect-bearing plan is not classified as generated-only."""

    plan = _plan(metrics=["effect", "p_value", "confidence_interval"])
    match = match_analysis_capability(
        plan,
        _catalog(),
        _context(**{"sample.disease": 2}),
    )
    assert match.mode == "SUPPORTED_TYPED"
    assert match.execution_mode == "SUPPORTED_TYPED"
    assert match.capability_code == "TYPED_GROUP_COMPARISON"


def test_analysis_plan_v2_round_trip_contains_method_and_goal() -> None:
    plan = _plan(
        analysis_goal="Compare abundance between disease groups with a bounded standard method",
        method={
            "family": "bootstrap",
            "name": "bootstrap_effect",
            "parameters": {"iterations": 1000, "confidence_level": 0.95},
        },
        metrics=["effect", "p_value", "confidence_interval"],
    )
    payload = plan.model_dump_json()
    restored = AnalysisPlan.model_validate_json(payload)
    assert restored.schemaVersion == "analysis-plan-v2"
    assert restored.validation_field is None
    assert restored.method.parameters["iterations"] == 1000
    assert restored.analysis_goal == plan.analysis_goal


def test_unsupported_method_uses_generated_path() -> None:
    match = AnalysisCapabilityRegistry(_catalog(), _context(**{"sample.disease": 2})).match(
        _plan(method={"family": "nonparametric", "name": "mann_whitney", "parameters": {}})
    )
    assert match.mode == "SUPPORTED_GENERATED"
    assert match.capability_code == "GENERATED_GROUP_COMPARISON"


def test_covariate_cannot_be_used_as_outcome() -> None:
    match = match_analysis_capability(
        _plan(outcome="sample.age"), _catalog(), _context(**{"sample.disease": 2})
    )
    assert match.mode == "UNSUPPORTED"
    assert match.reason_code == "FIELD_IS_NOT_A_NUMERIC_SCIENTIFIC_OUTCOME"


def test_outcome_cannot_be_used_as_group_dimension() -> None:
    match = match_analysis_capability(
        _plan(group_field="abundance.value"), _catalog(), _context(**{"abundance.value": 2})
    )
    assert match.mode == "UNSUPPORTED"
    assert match.reason_code == "GROUP_FIELD_MUST_BE_A_DIMENSION"


def test_standard_confounder_adjustment_is_typed() -> None:
    match = match_analysis_capability(
        _plan(
            analysis_type="confounder_adjustment",
            group_field="sample.disease",
            covariates=["sample.age"],
            metrics=["effect_size"],
        ),
        _catalog(),
        _context(**{"sample.disease": 2}),
    )
    assert match.mode == "SUPPORTED_TYPED"
    assert match.capability_code == "TYPED_CONFOUNDER_ADJUSTMENT"


def test_categorical_stratified_analysis_is_typed() -> None:
    match = match_analysis_capability(
        _plan(
            analysis_type="stratified_comparison",
            group_field="sample.disease",
            stratify_by=["sample.sex"],
            metrics=["effect_size"],
        ),
        _catalog(),
        _context(**{"sample.disease": 2}),
    )
    assert match.mode == "SUPPORTED_TYPED"
    assert match.capability_code == "TYPED_STRATIFIED_COMPARISON"


def test_cross_project_uses_group_and_validation_dimensions() -> None:
    plan = _plan(
        analysis_type="cross_project_validation",
        group_field="sample.disease",
        validation_field="metadata.project",
        metrics=["effect_size"],
    )
    match = match_analysis_capability(
        plan,
        _catalog(),
        _context(**{"sample.disease": 2, "metadata.project": 3}),
    )
    assert match.mode == "SUPPORTED_TYPED"
    assert match.reason_code == "MATCHED_PER_PROJECT_TWO_GROUP_COMPARISON"


def test_cross_validation_rejects_same_group_and_validation_field() -> None:
    with pytest.raises(ValidationError, match="must differ"):
        _plan(
            analysis_type="cross_project_validation",
            group_field="sample.disease",
            validation_field="sample.disease",
            metrics=["effect_size"],
        )


def test_cross_validation_rejects_non_dimension_validation_field() -> None:
    plan = _plan(
        analysis_type="cross_project_validation",
        group_field="sample.disease",
        validation_field="sample.age",
        metrics=["effect_size"],
    )
    match = match_analysis_capability(
        plan,
        _catalog(),
        _context(**{"sample.disease": 2, "sample.age": 5}),
    )
    assert match.mode == "UNSUPPORTED"
    assert match.capability_code == "VALIDATION_DIMENSION_REQUIRED"
    assert match.reason_code == "VALIDATION_FIELD_MUST_BE_A_DIMENSION"


def test_cross_validation_rejects_validation_field_missing_from_observation() -> None:
    plan = _plan(
        analysis_type="cross_project_validation",
        group_field="sample.disease",
        validation_field="metadata.project",
        metrics=["effect_size"],
    )
    fields = ["sample.disease", "abundance.value"]
    match = match_analysis_capability(
        plan,
        _catalog(),
        AnalysisCapabilityContext(
            available_observation_ids=[OBSERVATION],
            available_fields=fields,
            observation_fields={OBSERVATION: fields},
            distinct_counts={"sample.disease": 2, "metadata.project": 3},
        ),
    )
    assert match.mode == "UNSUPPORTED"
    assert match.capability_code == "OBSERVATION_FIELD_MISSING"


def test_numeric_stratification_with_explicit_spec_is_typed() -> None:
    plan = _plan(
        analysis_type="stratified_comparison",
        group_field="sample.disease",
        stratify_by=["sample.age"],
        numeric_stratification={
            "stratifier": "sample.age",
            "strategy": "quantile",
            "bin_count": 2,
            "cut_points": [],
            "min_samples_per_group": 2,
            "missing_value_policy": "drop",
            "multiple_testing": "benjamini_hochberg",
        },
        metrics=["effect_size", "p_value"],
    )
    match = match_analysis_capability(
        plan,
        _catalog(),
        _context(**{"sample.disease": 2, "sample.age": 6}),
    )
    assert match.mode == "SUPPORTED_TYPED"
    assert match.capability_code == "TYPED_NUMERIC_STRATIFIED_COMPARISON"


def test_cross_project_without_validation_field_is_rejected_for_new_shape() -> None:
    with pytest.raises(ValidationError):
        _plan(
            analysis_type="cross_project_validation",
            group_field="sample.disease",
            metrics=["effect_size"],
        )


def test_cross_validation_requires_two_validation_values() -> None:
    plan = _plan(
        analysis_type="cross_project_validation",
        group_field="sample.disease",
        validation_field="metadata.project",
        metrics=["effect_size"],
    )
    match = match_analysis_capability(
        plan,
        _catalog(),
        _context(**{"sample.disease": 2, "metadata.project": 1}),
    )
    assert match.mode == "UNSUPPORTED"
    assert match.capability_code == "VALIDATION_COVERAGE_REQUIRED"


@pytest.mark.parametrize(
    "payload",
    [
        {"method": {"family": "bootstrap", "parameters": {"unknown": 1}}},
        {"execution_mode": "typed"},
        {"unexpected": True},
    ],
)
def test_analysis_plan_rejects_unapproved_fields(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _plan(**payload)


def test_unsupported_method_action_combination_fails_closed() -> None:
    match = match_analysis_capability(
        _plan(
            analysis_type="cross_project_validation",
            group_field="sample.disease",
            validation_field="metadata.project",
            method={"family": "regression", "parameters": {}},
            metrics=["effect_size"],
        ),
        _catalog(),
        _context(**{"sample.disease": 2, "metadata.project": 2}),
    )
    assert match.mode == "UNSUPPORTED"
    assert match.reason_code == "ANALYSIS_METHOD_NOT_SUPPORTED_FOR_ACTION"


def test_abundance_comparison_requires_explicit_feature_dimension_when_catalog_has_one() -> None:
    catalog = _catalog()
    abundance = next(entity for entity in catalog.entities if entity.entityName == "abundance")
    abundance.fields.append(SchemaFieldSemantics(
        fieldId="abundance.feature",
        name="feature",
        dataType="string",
        nullable=False,
        semanticStatus="verified",
        groupable=True,
        scientificCapabilities=["dimension"],
        description="Feature dimension",
    ))
    context = _context(**{"sample.disease": 2})
    rejected = match_analysis_capability(_plan(), catalog, context)
    assert rejected.mode == "UNSUPPORTED"
    assert rejected.capability_code == "FEATURE_DIMENSION_REQUIRED"

    feature_fields = [*context.available_fields, "abundance.feature"]
    accepted_context = AnalysisCapabilityContext(
        available_observation_ids=[OBSERVATION],
        available_fields=feature_fields,
        observation_fields={OBSERVATION: feature_fields},
        distinct_counts={"sample.disease": 2},
    )
    accepted = match_analysis_capability(
        _plan(feature_field="abundance.feature"), catalog, accepted_context
    )
    assert accepted.mode == "SUPPORTED_TYPED"
