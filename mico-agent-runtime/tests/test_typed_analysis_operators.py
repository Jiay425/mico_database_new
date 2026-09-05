import pytest

from mico_agent_runtime.contracts.materialization import AnalysisPlan
from mico_agent_runtime.graph.generated_analysis import (
    GeneratedAnalysisError,
    execute_typed_analysis,
)


OBS = "observation-" + "b" * 32


def _plan(**updates: object) -> AnalysisPlan:
    payload: dict[str, object] = {
        "analysis_type": "group_comparison",
        "source_observation_ids": [OBS],
        "outcome": "abundance.value",
        "group_field": "sample.disease",
        "metrics": ["effect_size", "p_value", "confidence_interval"],
    }
    payload.update(updates)
    return AnalysisPlan.model_validate(payload)


def test_compare_groups_returns_two_group_summaries_and_mean_difference() -> None:
    result = execute_typed_analysis(
        _plan(),
        [
            {"a_sample_disease": "T2D", "a_abundance_value": 1.0},
            {"a_sample_disease": "T2D", "a_abundance_value": 3.0},
            {"a_sample_disease": "Healthy", "a_abundance_value": 5.0},
            {"a_sample_disease": "Healthy", "a_abundance_value": 7.0},
        ],
        4,
    )
    assert result.execution_mode == "typed"
    assert result.method_used == "two_group_normal_approximation"
    assert result.metrics["mean_difference"] == 4.0
    assert result.metrics["confidence_interval_low"] < 4.0
    assert result.metrics["confidence_interval_high"] > 4.0
    assert [item.group for item in result.group_results] == ["T2D", "Healthy"]
    assert [item.n for item in result.group_results] == [2, 2]
    assert [item.mean for item in result.group_results] == [2.0, 6.0]


def test_adjust_confounders_changes_a_confounded_unadjusted_difference() -> None:
    rows = [
        {"a_sample_disease": "A", "a_sample_age": 0, "a_abundance_value": 0.0},
        {"a_sample_disease": "A", "a_sample_age": 1, "a_abundance_value": 1.0},
        {"a_sample_disease": "B", "a_sample_age": 10, "a_abundance_value": 10.0},
        {"a_sample_disease": "B", "a_sample_age": 11, "a_abundance_value": 11.0},
    ]
    unadjusted = execute_typed_analysis(
        _plan(metrics=["effect_size"]), rows, len(rows)
    )
    adjusted = execute_typed_analysis(
        _plan(
            analysis_type="confounder_adjustment",
            covariates=["sample.age"],
            metrics=["effect_size", "p_value"],
        ),
        rows,
        len(rows),
    )
    assert unadjusted.metrics["mean_difference"] == 10.0
    assert abs(adjusted.metrics["adjusted_group_effect"]) < 1e-9
    assert adjusted.metrics["adjusted_p_value"] == 1.0
    assert adjusted.adjusted_covariates == ["sample.age"]
    assert adjusted.used_row_count == 4
    assert adjusted.dropped_row_count == 0


def test_adjust_confounders_reports_complete_case_counts() -> None:
    result = execute_typed_analysis(
        _plan(
            analysis_type="confounder_adjustment",
            covariates=["sample.age"],
            metrics=["effect_size"],
        ),
        [
            {"a_sample_disease": "A", "a_sample_age": 1, "a_abundance_value": 1.0},
            {"a_sample_disease": "A", "a_sample_age": None, "a_abundance_value": 2.0},
            {"a_sample_disease": "B", "a_sample_age": 3, "a_abundance_value": 4.0},
            {"a_sample_disease": "B", "a_sample_age": 4, "a_abundance_value": 5.0},
        ],
        4,
    )
    assert result.used_row_count == 3
    assert result.dropped_row_count == 1


def test_stratified_comparison_recomputes_effect_inside_each_stratum() -> None:
    result = execute_typed_analysis(
        _plan(
            analysis_type="stratified_comparison",
            stratify_by=["sample.sex"],
            metrics=["effect_size", "p_value"],
        ),
        [
            {"a_sample_sex": "young", "a_sample_disease": "A", "a_abundance_value": 1.0},
            {"a_sample_sex": "young", "a_sample_disease": "B", "a_abundance_value": 2.0},
            {"a_sample_sex": "old", "a_sample_disease": "A", "a_abundance_value": 10.0},
            {"a_sample_sex": "old", "a_sample_disease": "B", "a_abundance_value": 8.0},
        ],
        4,
    )
    assert len(result.stratum_results) == 2
    assert [item.stratum for item in result.stratum_results] == ["young", "old"]
    assert [item.mean_difference for item in result.stratum_results] == [1.0, -2.0]
    assert [item.direction for item in result.stratum_results] == ["positive", "negative"]
    assert result.metrics["positive_stratum_count"] == 1.0
    assert result.metrics["negative_stratum_count"] == 1.0


def test_numeric_stratification_is_sample_level_and_fdr_corrected() -> None:
    rows: list[dict[str, object]] = []
    # Six independent samples per disease group provide three samples in each
    # empirical median bin.  Each row also carries a feature so the operator
    # must retain the sample × feature unit rather than pooling rows.
    for index, age in enumerate([10, 20, 30, 40, 50, 60], start=1):
        rows.append({
            "a_analysis_sample_key": f"a{index}",
            "a_abundance_feature": "F1",
            "a_sample_disease": "A",
            "a_sample_age": age,
            "a_abundance_value": float(age),
        })
        rows.append({
            "a_analysis_sample_key": f"b{index}",
            "a_abundance_feature": "F1",
            "a_sample_disease": "B",
            "a_sample_age": age,
            "a_abundance_value": float(age + 2),
        })
    result = execute_typed_analysis(
        _plan(
            analysis_type="stratified_comparison",
            feature_field="abundance.feature",
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
        ),
        rows,
        len(rows),
    )
    assert result.execution_mode == "typed"
    assert result.method_used == "numeric_stratified_two_group_normal_approximation"
    assert result.metrics["sample_count"] == 12.0
    assert result.metrics["stratum_count"] == 2.0
    assert result.metrics["eligible_strata"] == 2.0
    assert result.metrics["insufficient_strata"] == 0.0
    assert result.ranking_method is not None
    assert len(result.stratum_results) == 2
    assert [item.group_a_n for item in result.stratum_results] == [3, 3]
    assert [item.group_b_n for item in result.stratum_results] == [3, 3]
    assert all(item.raw_p_value is not None for item in result.stratum_results)
    assert all(item.q_value is not None for item in result.stratum_results)
    assert all(item.adjusted_p_value == item.q_value for item in result.stratum_results)
    assert result.feature_results[0].stratum_results


def test_numeric_stratification_fails_closed_when_minimum_sample_is_not_met() -> None:
    with pytest.raises(
        GeneratedAnalysisError,
        match="ANALYSIS_TYPED_NUMERIC_STRATIFICATION_INSUFFICIENT_SAMPLE",
    ):
        execute_typed_analysis(
            _plan(
                analysis_type="stratified_comparison",
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
                metrics=["effect_size"],
            ),
            [
                {"a_sample_disease": "A", "a_sample_age": 10, "a_abundance_value": 1.0},
                {"a_sample_disease": "B", "a_sample_age": 20, "a_abundance_value": 2.0},
            ],
            2,
        )


def test_cross_project_comparison_runs_inside_each_project() -> None:
    result = execute_typed_analysis(
        _plan(
            analysis_type="cross_project_validation",
            validation_field="metadata.project",
            metrics=["effect_size", "p_value"],
        ),
        [
            {"a_metadata_project": "P1", "a_sample_disease": "A", "a_abundance_value": 1.0},
            {"a_metadata_project": "P1", "a_sample_disease": "B", "a_abundance_value": 2.0},
            {"a_metadata_project": "P2", "a_sample_disease": "A", "a_abundance_value": 3.0},
            {"a_metadata_project": "P2", "a_sample_disease": "B", "a_abundance_value": 1.0},
            {"a_metadata_project": "P3", "a_sample_disease": "A", "a_abundance_value": 4.0},
            {"a_metadata_project": "P3", "a_sample_disease": "B", "a_abundance_value": 4.0},
        ],
        6,
    )
    assert [item.validation_value for item in result.validation_results] == ["P1", "P2", "P3"]
    assert [item.direction for item in result.validation_results] == ["positive", "negative", "neutral"]
    assert result.metrics["project_count"] == 3.0
    assert result.metrics["positive_project_count"] == 1.0
    assert result.metrics["negative_project_count"] == 1.0
    assert result.metrics["neutral_project_count"] == 1.0


def test_cross_project_skips_project_without_both_groups() -> None:
    result = execute_typed_analysis(
        _plan(
            analysis_type="cross_project_validation",
            validation_field="metadata.project",
            metrics=["effect_size"],
        ),
        [
            {"a_metadata_project": "P1", "a_sample_disease": "A", "a_abundance_value": 1.0},
            {"a_metadata_project": "P1", "a_sample_disease": "B", "a_abundance_value": 2.0},
            {"a_metadata_project": "P2", "a_sample_disease": "A", "a_abundance_value": 3.0},
        ],
        3,
    )
    assert [item.validation_value for item in result.validation_results] == ["P1"]
    assert result.metrics["skipped_project_count"] == 1.0


def test_typed_operator_fails_when_no_complete_two_group_comparison_exists() -> None:
    with pytest.raises(GeneratedAnalysisError, match="ANALYSIS_TYPED_INSUFFICIENT_DATA"):
        execute_typed_analysis(
            _plan(
                analysis_type="cross_project_validation",
                validation_field="metadata.project",
                metrics=["effect_size"],
            ),
            [
                {"a_metadata_project": "P1", "a_sample_disease": "A", "a_abundance_value": 1.0},
            ],
            1,
        )


def test_feature_aware_comparison_collapses_duplicate_rows_to_sample_level() -> None:
    result = execute_typed_analysis(
        _plan(feature_field="abundance.feature", metrics=["effect_size", "p_value"]),
        [
            {"a_abundance_sample_id": "s1", "a_abundance_feature": "F1", "a_sample_disease": "A", "a_abundance_value": 1.0},
            {"a_abundance_sample_id": "s1", "a_abundance_feature": "F1", "a_sample_disease": "A", "a_abundance_value": 3.0},
            {"a_abundance_sample_id": "s2", "a_abundance_feature": "F1", "a_sample_disease": "B", "a_abundance_value": 5.0},
            {"a_abundance_sample_id": "s3", "a_abundance_feature": "F1", "a_sample_disease": "B", "a_abundance_value": 7.0},
        ],
        4,
    )
    assert result.metrics["sample_count"] == 3.0
    assert result.metrics["collapsed_row_count"] == 1.0
    assert [item.n for item in result.group_results] == [1, 2]
    assert result.metrics["mean_difference"] == 4.0


def test_multi_feature_comparison_returns_separate_feature_results_without_pooling() -> None:
    result = execute_typed_analysis(
        _plan(feature_field="abundance.feature", metrics=["effect_size", "p_value"]),
        [
            {"a_abundance_sample_id": "s1", "a_abundance_feature": "F1", "a_sample_disease": "A", "a_abundance_value": 1.0},
            {"a_abundance_sample_id": "s1", "a_abundance_feature": "F2", "a_sample_disease": "A", "a_abundance_value": 10.0},
            {"a_abundance_sample_id": "s2", "a_abundance_feature": "F1", "a_sample_disease": "B", "a_abundance_value": 5.0},
            {"a_abundance_sample_id": "s2", "a_abundance_feature": "F2", "a_sample_disease": "B", "a_abundance_value": 7.0},
            {"a_abundance_sample_id": "s3", "a_abundance_feature": "F1", "a_sample_disease": "B", "a_abundance_value": 7.0},
            {"a_abundance_sample_id": "s3", "a_abundance_feature": "F2", "a_sample_disease": "B", "a_abundance_value": 9.0},
        ],
        6,
    )
    assert [item.featureName for item in result.feature_results] == ["F1", "F2"]
    assert "mean_difference" not in result.metrics
    assert result.feature_results[0].metrics["mean_difference"] == 5.0
    assert result.feature_results[1].metrics["mean_difference"] == -2.0
    assert result.ranking_method == (
        "bh_fdr_adjusted_p_value_then_raw_p_value_then_abs_effect_desc_then_feature_name"
    )
    assert all("raw_p_value" in item.metrics for item in result.feature_results)
    assert all("adjusted_p_value" in item.metrics for item in result.feature_results)


def test_multi_feature_result_is_bounded_and_reports_truncation() -> None:
    rows = []
    for index in range(65):
        feature = f"F{index}"
        rows.extend([
            {
                "a_abundance_sample_id": f"a{index}",
                "a_abundance_feature": feature,
                "a_sample_disease": "A",
                "a_abundance_value": 1.0,
            },
            {
                "a_abundance_sample_id": f"b{index}",
                "a_abundance_feature": feature,
                "a_sample_disease": "B",
                "a_abundance_value": 2.0,
            },
        ])

    result = execute_typed_analysis(
        _plan(feature_field="abundance.feature", metrics=["effect_size", "p_value"]),
        rows,
        len(rows),
    )

    assert len(result.feature_results) == 64
    assert result.metrics["feature_results_total_count"] == 65.0
    assert result.metrics["feature_results_truncated_count"] == 1.0
    assert result.metrics["total_tested_features"] == 65.0
    assert result.metrics["eligible_features"] == 65.0
    assert result.metrics["returned_features"] == 64.0
    assert result.metrics["truncated_features"] == 1.0
    assert result.ranking_method is not None


def test_multi_feature_rows_fail_closed_without_feature_binding() -> None:
    with pytest.raises(GeneratedAnalysisError, match="ANALYSIS_TYPED_FEATURE_FIELD_REQUIRED"):
        execute_typed_analysis(
            _plan(),
            [
                {"a_abundance_sample_id": "s1", "a_abundance_feature": "F1", "a_sample_disease": "A", "a_abundance_value": 1.0},
                {"a_abundance_sample_id": "s2", "a_abundance_feature": "F2", "a_sample_disease": "B", "a_abundance_value": 2.0},
            ],
            2,
        )


def test_multi_feature_adjustment_keeps_insufficient_features_explicit() -> None:
    rows = [
        {"a_analysis_sample_key": "s1", "a_abundance_feature": "F1", "a_sample_disease": "A", "a_sample_age": 1, "a_abundance_value": 1.0},
        {"a_analysis_sample_key": "s2", "a_abundance_feature": "F1", "a_sample_disease": "A", "a_sample_age": 2, "a_abundance_value": 2.0},
        {"a_analysis_sample_key": "s3", "a_abundance_feature": "F1", "a_sample_disease": "B", "a_sample_age": 3, "a_abundance_value": 4.0},
        {"a_analysis_sample_key": "s4", "a_abundance_feature": "F1", "a_sample_disease": "B", "a_sample_age": 4, "a_abundance_value": 5.0},
        {"a_analysis_sample_key": "s5", "a_abundance_feature": "F2", "a_sample_disease": "A", "a_sample_age": None, "a_abundance_value": 1.0},
        {"a_analysis_sample_key": "s6", "a_abundance_feature": "F2", "a_sample_disease": "B", "a_sample_age": None, "a_abundance_value": 2.0},
    ]
    result = execute_typed_analysis(
        _plan(
            analysis_type="confounder_adjustment",
            feature_field="abundance.feature",
            covariates=["sample.age"],
            metrics=["effect_size", "p_value"],
        ),
        rows,
        len(rows),
    )
    by_feature = {item.featureName: item for item in result.feature_results}
    assert by_feature["F2"].status == "insufficient_data"
    assert by_feature["F2"].metrics["insufficient_sample"] == 1.0
    assert by_feature["F1"].status == "supported"
    assert "q_value" in by_feature["F1"].metrics


def test_aggregated_abundance_rows_fail_closed_for_sample_level_analysis() -> None:
    with pytest.raises(GeneratedAnalysisError, match="ANALYSIS_TYPED_SAMPLE_LEVEL_REQUIRED"):
        execute_typed_analysis(
            _plan(),
            [
                {"a_sample_disease": "A", "a_abundance_value_mean": 1.0},
                {"a_sample_disease": "B", "a_abundance_value_mean": 2.0},
            ],
            2,
        )
