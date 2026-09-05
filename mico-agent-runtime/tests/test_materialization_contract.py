from datetime import datetime, timezone

import pytest

from mico_agent_runtime.contracts.materialization import (
    AnalysisPlan,
    QueryPlan,
    validate_query_plan_catalog,
)
from mico_agent_runtime.contracts.schema_catalog import (
    SchemaEntitySemantics,
    SchemaFieldSemantics,
    SchemaSemanticCatalog,
)
from mico_agent_runtime.graph.generated_analysis import execute_typed_analysis


def _aggregation_catalog() -> SchemaSemanticCatalog:
    return SchemaSemanticCatalog(
        schemaVersion="schema-catalog-v1",
        source="java_schema_contract",
        generatedAt=datetime.now(timezone.utc),
        entities=[SchemaEntitySemantics(
            entityId="sample",
            entityName="sample",
            sourceTable="sample",
            fields=[
                SchemaFieldSemantics(
                    fieldId="sample.disease",
                    name="disease",
                    dataType="string",
                    nullable=True,
                    semanticStatus="verified",
                    groupable=True,
                    displayable=True,
                    scientificCapabilities=["dimension"],
                    description="group label",
                ),
                SchemaFieldSemantics(
                    fieldId="sample.value",
                    name="value",
                    dataType="number",
                    nullable=True,
                    semanticStatus="verified",
                    aggregatable=True,
                    displayable=True,
                    scientificCapabilities=["outcome"],
                    description="numeric outcome",
                ),
            ],
        )],
        queryRules=[
            "select_or_with_only",
            "explicit_columns_only",
            "no_cross_database_reference",
            "bounded_limit_required",
            "java_final_validation",
        ],
    )


def test_query_plan_uses_semantic_ids_and_ordered_relations() -> None:
    plan = QueryPlan.model_validate({
        "root_entity": "sample",
        "relation_path": ["sample_to_abundance", "sample_to_project"],
        "select_fields": ["project.name"],
        "aggregations": [{"field": "abundance.value", "op": "mean"}],
        "filters": [{"field": "sample.disease", "operator": "eq", "value": "T2D"}],
        "group_by": ["project.name"],
        "limit": 100,
    })

    assert plan.schemaVersion == "query-plan-v1"
    assert plan.relation_path == ["sample_to_abundance", "sample_to_project"]


@pytest.mark.parametrize("field", ["sample", "Sample.project"])
def test_query_plan_rejects_invalid_field_ids(field: str) -> None:
    with pytest.raises(ValueError):
        QueryPlan.model_validate({
            "root_entity": "sample",
            "select_fields": [field],
            "limit": 10,
        })


def test_query_plan_rejects_unbounded_or_duplicate_relation_path() -> None:
    with pytest.raises(ValueError):
        QueryPlan.model_validate({
            "root_entity": "sample",
            "relation_path": ["sample_to_abundance", "sample_to_abundance"],
            "select_fields": ["sample.disease"],
            "limit": 1001,
        })


def test_catalog_rejects_non_aggregatable_dimension() -> None:
    plan = QueryPlan(
        root_entity="sample",
        select_fields=["sample.disease"],
        aggregations=[{"field": "sample.disease", "op": "count"}],
        group_by=["sample.disease"],
        limit=10,
    )
    with pytest.raises(ValueError, match="not aggregatable"):
        validate_query_plan_catalog(plan, _aggregation_catalog())


def test_catalog_allows_numeric_aggregation() -> None:
    plan = QueryPlan(
        root_entity="sample",
        select_fields=["sample.disease"],
        aggregations=[{"field": "sample.value", "op": "mean"}],
        group_by=["sample.disease"],
        limit=10,
    )
    assert validate_query_plan_catalog(plan, _aggregation_catalog()) == plan


def test_query_plan_accepts_runtime_owned_sample_bound_only_for_raw_reads() -> None:
    plan = QueryPlan.model_validate({
        "root_entity": "sample",
        "select_fields": ["sample.disease", "sample.value"],
        "limit": 20_000,
        "sample_limit_per_group": 50,
        "sample_limit_group_field": "sample.disease",
    })
    assert plan.sample_limit_per_group == 50
    with pytest.raises(ValueError, match="raw, ungrouped"):
        QueryPlan.model_validate({
            "root_entity": "sample",
            "select_fields": ["sample.disease"],
            "aggregations": [{"field": "sample.value", "op": "mean"}],
            "group_by": ["sample.disease"],
            "limit": 10,
            "sample_limit_per_group": 10,
            "sample_limit_group_field": "sample.disease",
        })


def test_analysis_plan_is_closed_by_analysis_type() -> None:
    plan = AnalysisPlan.model_validate({
        "analysis_type": "confounder_adjustment",
        "source_observation_ids": ["observation-" + "a" * 32],
        "outcome": "abundance.value",
        "group_field": "sample.disease",
        "covariates": ["sample.age", "sample.gender"],
        "metrics": ["effect_size", "confidence_interval"],
    })

    assert plan.analysis_type == "confounder_adjustment"

    with pytest.raises(ValueError):
        AnalysisPlan.model_validate({
            "analysis_type": "free_form_statistics",
            "source_observation_ids": ["observation-" + "a" * 32],
            "metrics": ["mean"],
        })


def test_numeric_stratification_spec_is_closed_and_round_trips() -> None:
    plan = AnalysisPlan.model_validate({
        "analysis_type": "stratified_comparison",
        "source_observation_ids": ["observation-" + "a" * 32],
        "outcome": "abundance.value",
        "group_field": "sample.disease",
        "stratify_by": ["sample.age"],
        "numeric_stratification": {
            "stratifier": "sample.age",
            "strategy": "quantile",
            "bin_count": 2,
            "cut_points": [],
            "min_samples_per_group": 2,
            "missing_value_policy": "drop",
            "multiple_testing": "benjamini_hochberg",
        },
        "metrics": ["effect_size", "p_value"],
    })
    restored = AnalysisPlan.model_validate_json(plan.model_dump_json())
    assert restored.numeric_stratification is not None
    assert restored.numeric_stratification.stratifier == "sample.age"
    assert restored.numeric_stratification.bin_count == 2


@pytest.mark.parametrize(
    "spec, message",
    [
        (
            {
                "stratifier": "sample.age",
                "strategy": "quantile",
                "cut_points": [],
                "min_samples_per_group": 2,
            },
            "requires bin_count",
        ),
        (
            {
                "stratifier": "sample.age",
                "strategy": "fixed_bins",
                "bin_count": 3,
                "cut_points": [30],
                "min_samples_per_group": 2,
            },
            "requires bin_count - 1 cut_points",
        ),
    ],
)
def test_numeric_stratification_spec_rejects_incomplete_binning(
    spec: dict[str, object], message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        AnalysisPlan.model_validate({
            "analysis_type": "stratified_comparison",
            "source_observation_ids": ["observation-" + "a" * 32],
            "outcome": "abundance.value",
            "group_field": "sample.disease",
            "stratify_by": ["sample.age"],
            "numeric_stratification": spec,
            "metrics": ["effect_size"],
        })


def test_numeric_stratification_spec_cannot_be_attached_to_another_analysis() -> None:
    with pytest.raises(ValueError, match="only valid for stratified_comparison"):
        AnalysisPlan.model_validate({
            "analysis_type": "group_comparison",
            "source_observation_ids": ["observation-" + "a" * 32],
            "outcome": "abundance.value",
            "group_field": "sample.disease",
            "numeric_stratification": {
                "stratifier": "sample.age",
                "strategy": "quantile",
                "bin_count": 2,
                "cut_points": [],
                "min_samples_per_group": 2,
            },
            "metrics": ["effect_size"],
        })


def test_typed_analysis_executes_only_approved_bounded_metrics() -> None:
    plan = AnalysisPlan.model_validate({
        "analysis_type": "group_comparison",
        "source_observation_ids": ["observation-" + "b" * 32],
        "outcome": "abundance.value",
        "group_field": "sample.gender",
        "metrics": ["count", "mean", "effect_size"],
    })
    result = execute_typed_analysis(
        plan,
        [
            {"a_sample_gender": "g1", "a_abundance_value": 1.0},
            {"a_sample_gender": "g2", "a_abundance_value": 3.0},
        ],
        2,
    )
    assert result.analysisType == "group_comparison"
    assert result.codeVersion == "typed-analysis-operator-v1"
    assert "typed_plan_executed_by_approved_operator" in result.limitations
    assert result.metrics["count"] == 2.0
    assert result.metrics["mean"] == 2.0
    assert result.metrics["effect_size"] == 2.0

    inferential = plan.model_copy(update={"metrics": ["p_value", "confidence_interval"]})
    inferential_result = execute_typed_analysis(
        inferential,
        [
            {"a_sample_gender": "g1", "a_abundance_value": 1.0},
            {"a_sample_gender": "g1", "a_abundance_value": 1.0},
            {"a_sample_gender": "g2", "a_abundance_value": 3.0},
            {"a_sample_gender": "g2", "a_abundance_value": 3.0},
        ],
        4,
    )
    assert 0.0 <= inferential_result.metrics["p_value"] <= 1.0
    assert inferential_result.metrics["confidence_interval"] >= 0.0


def test_cross_validation_requires_matching_dimension_and_bound_operator() -> None:
    with pytest.raises(ValueError, match="requires outcome"):
        AnalysisPlan.model_validate({
            "analysis_type": "cross_project_validation",
            "source_observation_ids": [
                "observation-" + "c" * 32,
                "observation-" + "d" * 32,
            ],
            "group_field": "metadata.project",
            "metrics": ["count"],
        })

    plan = AnalysisPlan.model_validate({
        "analysis_type": "cross_project_validation",
        "source_observation_ids": [
            "observation-" + "c" * 32,
            "observation-" + "d" * 32,
        ],
        "outcome": "abundance.value",
        "group_field": "sample.disease",
        "validation_field": "metadata.project",
        "metrics": ["effect_size"],
    })
    result = execute_typed_analysis(
        plan,
        [
            {"a_sample_disease": "d1", "a_metadata_project": "p1", "a_abundance_value": 1.0},
            {"a_sample_disease": "d2", "a_metadata_project": "p1", "a_abundance_value": 3.0},
            {"a_sample_disease": "d1", "a_metadata_project": "p2", "a_abundance_value": 1.0},
            {"a_sample_disease": "d2", "a_metadata_project": "p2", "a_abundance_value": 3.0},
        ],
        2,
    )
    assert result.metrics["effect_size"] == 2.0

    disease_dictionary_plan = AnalysisPlan.model_validate({
        "analysis_type": "cross_disease_validation",
        "source_observation_ids": [
            "observation-" + "e" * 32,
            "observation-" + "f" * 32,
        ],
        "outcome": "abundance.value",
        "group_field": "sample.project",
        "validation_field": "disease.name",
        "metrics": ["count"],
    })
    assert disease_dictionary_plan.validation_field == "disease.name"


def test_cross_validation_fails_closed_when_one_group_is_present() -> None:
    plan = AnalysisPlan.model_validate({
        "analysis_type": "cross_project_validation",
        "source_observation_ids": ["observation-" + "a" * 32],
        "outcome": "abundance.value",
        "group_field": "sample.disease",
        "validation_field": "metadata.project",
        "metrics": ["count"],
    })

    with pytest.raises(ValueError, match="GROUP_COVERAGE_REQUIRED"):
        execute_typed_analysis(
            plan,
            [
                {"a_sample_disease": "d1", "a_metadata_project": "p1", "a_abundance_value": 1.0},
                {"a_sample_disease": "d1", "a_metadata_project": "p1", "a_abundance_value": 3.0},
            ],
            2,
        )
