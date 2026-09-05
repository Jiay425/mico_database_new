from scripts.run_data_semantics_probe import (
    _calibration_markdown,
    _hash_identifier,
    _java_limit_evidence,
    _percentile,
    _serialise_rows,
    _summarise_patient_sample_multiplicity,
)


def test_percentile_is_deterministic_for_rows_per_sample() -> None:
    assert _percentile([7, 96, 195, 316], 0.5) == 145.5
    assert _percentile([], 0.5) is None


def test_probe_redacts_sample_and_patient_identifiers() -> None:
    rows = _serialise_rows([
        {"patient_id": 42, "sample_id": "sample-42", "disease": "T2D"},
    ])
    assert rows[0]["patient_id"].startswith("sha256:")
    assert rows[0]["sample_id"].startswith("sha256:")
    assert rows[0]["disease"] == "T2D"
    assert _hash_identifier(42) == rows[0]["patient_id"]
    assert "sample-42" not in str(rows)


def test_calibration_markdown_distinguishes_rows_from_samples() -> None:
    result = {
        "requested_cohort_distribution": [
            {"disease": "T2D", "row_count": 887},
            {"disease": "healthy", "row_count": 5000},
        ],
        "abundance_requested_distribution": [
            {
                "disease": "T2D",
                "abundance_row_count": 171376,
                "distinct_sample_count": 887,
                "distinct_feature_count": 1147,
            },
            {
                "disease": "healthy",
                "abundance_row_count": 924044,
                "distinct_sample_count": 4997,
                "distinct_feature_count": 1898,
            },
        ],
        "rows_per_sample": {
            "T2D": {"rows_per_sample_median": 194},
            "healthy": {"rows_per_sample_median": 195},
        },
        "metadata_join": {
            "overall": [{"matched_patient_keys": 13897, "patient_keys": 24264}],
            "requested_cohort": [
                {"disease": "T2D", "matched_patient_keys": 0},
                {"disease": "healthy", "matched_patient_keys": 0},
            ],
        },
        "java_limit_evidence": {
            "limit_scope": "joined/result rows, not unique samples",
            "compiler_limit": {"found": True},
            "prepared_statement_bound": {"found": True},
        },
    }
    markdown = _calibration_markdown(result)
    assert "887" in markdown
    assert "171376 / 887 / 1147" in markdown
    assert "not unique samples" in markdown
    assert "patient keys" in markdown


def test_java_limit_evidence_points_to_compiler_and_jdbc_bound() -> None:
    evidence = _java_limit_evidence()
    assert evidence["limit_scope"] == "joined/result rows, not unique samples"
    assert evidence["compiler_limit"]["found"] is True
    assert evidence["prepared_statement_bound"]["found"] is True


def test_patient_sample_multiplicity_is_distinct_from_row_counts() -> None:
    summary = _summarise_patient_sample_multiplicity([
        {"disease": "T2D", "patient_id": 1, "sample_count": 1},
        {"disease": "T2D", "patient_id": 2, "sample_count": 2},
        {"disease": "healthy", "patient_id": 3, "sample_count": 1},
    ])
    assert summary["T2D"]["patient_count"] == 2
    assert summary["T2D"]["patients_with_multiple_samples"] == 1
    assert summary["T2D"]["sample_count_total"] == 3
    assert summary["healthy"]["samples_per_patient_median"] == 1
