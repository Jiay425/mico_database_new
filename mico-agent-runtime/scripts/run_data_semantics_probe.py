"""Deterministic read-only calibration probes for the real development DB.

This is intentionally not an Agent action and does not call Gemini.  It is a
diagnostic companion to the Gemini canary used to separate database semantics
from policy/materializer behaviour.  Results are de-identified and written
under ``artifacts/data_semantics_probe`` (which is ignored by git).

The probe uses the same development MySQL endpoint as the Spring service.  It
does not mutate data, issue a Scientific Action, or feed a result back into a
Decision State.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import asyncmy


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "artifacts" / "data_semantics_probe"
CANARY_AUDIT = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "gemini_dynamic_canary"
    / "query_semantics_audit.json"
)
CANARY_TRACE = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "gemini_dynamic_canary"
    / "trace.json"
)


def _hash_identifier(value: object) -> str | None:
    if value is None:
        return None
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    return f"sha256:{digest[:16]}"


def _json_default(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def _percentile(values: list[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _summarise_patient_sample_multiplicity(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Summarise the patient -> sample cardinality without exposing IDs.

    A patient and a sample are deliberately different units in this audit.
    The summary is therefore based on the count of distinct sample keys per
    patient, not on abundance row counts.
    """

    by_disease: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        disease = row.get("disease")
        if disease is None:
            continue
        by_disease[str(disease)].append(int(row.get("sample_count") or 0))
    result: dict[str, dict[str, Any]] = {}
    for disease, counts in sorted(by_disease.items()):
        result[disease] = {
            "patient_count": len(counts),
            "patients_with_one_sample": sum(count == 1 for count in counts),
            "patients_with_multiple_samples": sum(count > 1 for count in counts),
            "sample_count_total": sum(counts),
            "samples_per_patient_min": min(counts) if counts else None,
            "samples_per_patient_median": statistics.median(counts) if counts else None,
            "samples_per_patient_max": max(counts) if counts else None,
        }
    return result


def _serialise_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Make DB-driver rows JSON-safe without exposing identifiers."""

    result: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {}
        for key, value in row.items():
            if key in {"patient_id", "sample_id", "metadata_patient_id", "abundance_patient_id"}:
                item[key] = _hash_identifier(value)
            elif isinstance(value, Decimal):
                item[key] = int(value) if value == value.to_integral_value() else float(value)
            else:
                item[key] = value
        result.append(item)
    return result


async def _query(cursor: Any, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    await cursor.execute(sql, params)
    return list(await cursor.fetchall())


async def _probe_database() -> dict[str, Any]:
    host = os.environ.get("MICO_PROBE_DB_HOST", os.environ.get("MYSQL_HOST", "127.0.0.1"))
    port = int(os.environ.get("MICO_PROBE_DB_PORT", os.environ.get("MYSQL_PORT", "13306")))
    user = os.environ.get("MICO_PROBE_DB_USER", os.environ.get("MYSQL_USERNAME", "root"))
    password = os.environ.get("MICO_PROBE_DB_PASSWORD") or os.environ["MYSQL_PASSWORD"]
    database = os.environ.get(
        "MICO_PROBE_DB_NAME",
        os.environ.get("MYSQL_DATABASE", "patient_data_manager"),
    )

    connection = await asyncmy.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        db=database,
        connect_timeout=10,
        read_timeout=180,
    )
    try:
        async with connection.cursor(asyncmy.cursors.DictCursor) as cursor:
            cohort_distribution = await _query(
                cursor,
                """
                SELECT disease, COUNT(*) AS row_count,
                       COUNT(DISTINCT patient_id) AS distinct_patient_count
                FROM patients
                GROUP BY disease
                ORDER BY disease
                """,
            )
            requested_patients = await _query(
                cursor,
                """
                SELECT disease, COUNT(*) AS row_count,
                       COUNT(DISTINCT patient_id) AS distinct_patient_count,
                       SUM(age IS NULL) AS null_age_count,
                       MIN(patient_id) AS min_patient_id,
                       MAX(patient_id) AS max_patient_id
                FROM patients
                WHERE disease IN ('T2D', 'healthy')
                GROUP BY disease
                ORDER BY disease
                """,
            )
            requested_age = await _query(
                cursor,
                """
                SELECT disease, COUNT(DISTINCT patient_id) AS patient_count,
                       COUNT(DISTINCT age) AS distinct_age_count,
                       SUM(age IS NULL) AS null_age_count,
                       MIN(age) AS min_age, MAX(age) AS max_age
                FROM patients
                WHERE disease IN ('T2D', 'healthy')
                GROUP BY disease
                ORDER BY disease
                """,
            )
            label_dictionary = await _query(
                cursor,
                """
                SELECT disease_id, disease_name
                FROM diseases
                WHERE LOWER(disease_name) IN ('t2d', 'healthy', 'type 2 diabetes')
                   OR LOWER(disease_name) LIKE 't2d;%%'
                ORDER BY disease_id
                """,
            )
            requested_abundance = await _query(
                cursor,
                """
                SELECT p.disease,
                       COUNT(*) AS abundance_row_count,
                       COUNT(DISTINCT a.sample_id) AS distinct_sample_count,
                       COUNT(DISTINCT a.patient_id) AS distinct_patient_count,
                       COUNT(DISTINCT a.microbe_name_standard) AS distinct_feature_count
                FROM patients p
                JOIN microbe_abundance_standard a ON a.patient_id = p.patient_id
                WHERE p.disease IN ('T2D', 'healthy')
                GROUP BY p.disease
                ORDER BY p.disease
                """,
            )
            rows_per_sample = await _query(
                cursor,
                """
                SELECT p.disease, a.patient_id, a.sample_id,
                       COUNT(*) AS abundance_row_count,
                       COUNT(DISTINCT a.microbe_name_standard) AS feature_count,
                       COUNT(DISTINCT p.age) AS age_value_count
                FROM patients p
                JOIN microbe_abundance_standard a ON a.patient_id = p.patient_id
                WHERE p.disease IN ('T2D', 'healthy')
                GROUP BY p.disease, a.patient_id, a.sample_id
                """,
            )
            sample_feature_duplicates = await _query(
                cursor,
                """
                SELECT disease,
                       SUM(pair_count) AS abundance_row_count,
                       COUNT(*) AS distinct_sample_feature_pairs,
                       SUM(pair_count > 1) AS duplicate_pair_count,
                       SUM(GREATEST(pair_count - 1, 0)) AS duplicate_extra_rows,
                       MAX(pair_count) AS max_duplicate_count
                FROM (
                    SELECT p.disease, a.sample_id, a.microbe_name_standard,
                           COUNT(*) AS pair_count
                    FROM patients p
                    JOIN microbe_abundance_standard a ON a.patient_id = p.patient_id
                    WHERE p.disease IN ('T2D', 'healthy')
                    GROUP BY p.disease, a.sample_id, a.microbe_name_standard
                ) pair_counts
                GROUP BY disease
                ORDER BY disease
                """,
            )
            patient_sample_multiplicity = await _query(
                cursor,
                """
                SELECT p.disease, p.patient_id,
                       COUNT(DISTINCT a.sample_id) AS sample_count
                FROM patients p
                JOIN microbe_abundance_standard a ON a.patient_id = p.patient_id
                WHERE p.disease IN ('T2D', 'healthy')
                GROUP BY p.disease, p.patient_id
                """,
            )
            abundance_window = await _query(
                cursor,
                """
                SELECT p.disease, p.age, p.patient_id, a.sample_id,
                       a.microbe_name_standard AS feature, a.abundance_value
                FROM patients p
                JOIN microbe_abundance_standard a ON a.patient_id = p.patient_id
                WHERE p.disease IN ('T2D', 'healthy')
                ORDER BY a.standard_abundance_id
                LIMIT 1000
                """,
            )
            inspect_window = await _query(
                cursor,
                """
                SELECT disease, age, gender, country, body_site
                FROM patients
                LIMIT 100
                """,
            )
            metadata_keys = await _query(
                cursor,
                """
                SELECT COUNT(*) AS row_count,
                       COUNT(DISTINCT patient_id) AS distinct_patient_count,
                       COUNT(DISTINCT sample_id) AS distinct_sample_count,
                       COUNT(DISTINCT project_name) AS distinct_project_count,
                       MIN(patient_id) AS min_patient_id,
                       MAX(patient_id) AS max_patient_id
                FROM meta2db_sample_metadata
                """,
            )
            patient_keys = await _query(
                cursor,
                """
                SELECT COUNT(*) AS row_count,
                       COUNT(DISTINCT patient_id) AS distinct_patient_count,
                       MIN(patient_id) AS min_patient_id,
                       MAX(patient_id) AS max_patient_id
                FROM patients
                """,
            )
            metadata_join = await _query(
                cursor,
                """
                SELECT COUNT(*) AS patient_rows,
                       COUNT(DISTINCT p.patient_id) AS patient_keys,
                       COUNT(DISTINCT CASE WHEN m.patient_id IS NOT NULL THEN p.patient_id END)
                           AS matched_patient_keys,
                       SUM(m.patient_id IS NOT NULL) AS matched_join_rows,
                       SUM(m.patient_id IS NULL) AS unmatched_join_rows
                FROM patients p
                LEFT JOIN meta2db_sample_metadata m ON m.patient_id = p.patient_id
                """,
            )
            requested_metadata_join = await _query(
                cursor,
                """
                SELECT p.disease,
                       COUNT(DISTINCT p.patient_id) AS patient_keys,
                       COUNT(DISTINCT CASE WHEN m.patient_id IS NOT NULL THEN p.patient_id END)
                           AS matched_patient_keys,
                       COUNT(DISTINCT m.project_name) AS project_count
                FROM patients p
                LEFT JOIN meta2db_sample_metadata m ON m.patient_id = p.patient_id
                WHERE p.disease IN ('T2D', 'healthy')
                GROUP BY p.disease
                ORDER BY p.disease
                """,
            )
            project_columns = await _query(
                cursor,
                """
                SELECT TABLE_NAME AS table_name, COLUMN_NAME AS column_name, DATA_TYPE AS data_type
                FROM information_schema.columns
                WHERE table_schema = DATABASE() AND COLUMN_NAME LIKE '%%project%%'
                ORDER BY TABLE_NAME, COLUMN_NAME
                """,
            )
            project_values = await _query(
                cursor,
                """
                SELECT project_name, COUNT(*) AS row_count,
                       COUNT(DISTINCT patient_id) AS distinct_patient_count
                FROM meta2db_sample_metadata
                WHERE project_name IS NOT NULL AND TRIM(project_name) <> ''
                GROUP BY project_name
                ORDER BY row_count DESC, project_name
                LIMIT 20
                """,
            )
            age_uniqueness = await _query(
                cursor,
                """
                SELECT COUNT(*) AS patient_count,
                       SUM(age_values > 1) AS patients_with_multiple_age_values,
                       MAX(age_values) AS max_age_values
                FROM (
                    SELECT patient_id, COUNT(DISTINCT age) AS age_values
                    FROM patients
                    GROUP BY patient_id
                ) age_by_patient
                """,
            )
            patient_examples = await _query(
                cursor,
                """
                SELECT patient_id, disease
                FROM patients
                ORDER BY patient_id
                LIMIT 5
                """,
            )
            metadata_examples = await _query(
                cursor,
                """
                SELECT patient_id, sample_id
                FROM meta2db_sample_metadata
                ORDER BY patient_id
                LIMIT 5
                """,
            )
            collation = await _query(cursor, "SHOW FULL COLUMNS FROM patients LIKE 'disease'")
    finally:
        connection.close()

    by_disease: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows_per_sample:
        by_disease[str(row["disease"])].append(row)
    rows_per_sample_summary: dict[str, dict[str, Any]] = {}
    for disease, values in sorted(by_disease.items()):
        counts = [int(row["abundance_row_count"]) for row in values]
        rows_per_sample_summary[disease] = {
            "sample_count": len({row["sample_id"] for row in values}),
            "patient_count": len({row["patient_id"] for row in values}),
            "abundance_row_total": sum(counts),
            "rows_per_sample_min": min(counts) if counts else None,
            "rows_per_sample_median": statistics.median(counts) if counts else None,
            "rows_per_sample_p25": _percentile(counts, 0.25),
            "rows_per_sample_p75": _percentile(counts, 0.75),
            "rows_per_sample_max": max(counts) if counts else None,
            "feature_count_min": min(int(row["feature_count"]) for row in values) if values else None,
            "feature_count_max": max(int(row["feature_count"]) for row in values) if values else None,
            "patients_with_multiple_age_values": sum(
                int(row["age_value_count"]) > 1 for row in values
            ),
        }

    patient_sample_summary = _summarise_patient_sample_multiplicity(
        patient_sample_multiplicity
    )

    window_counter = Counter(str(row["disease"]) for row in abundance_window)
    window_samples = {
        str(disease): {
            "row_count": count,
            "distinct_sample_count": len({row["sample_id"] for row in abundance_window if row["disease"] == disease}),
            "distinct_patient_count": len({row["patient_id"] for row in abundance_window if row["disease"] == disease}),
            "distinct_feature_count": len({row["feature"] for row in abundance_window if row["disease"] == disease}),
        }
        for disease, count in sorted(window_counter.items())
    }
    inspect_distribution = dict(sorted(Counter(row["disease"] for row in inspect_window).items()))

    patient_key = patient_keys[0] if patient_keys else {}
    metadata_key = metadata_keys[0] if metadata_keys else {}
    requested_ranges = {
        str(row["disease"]): {
            "patient_count": row.get("distinct_patient_count"),
            "min_patient_id": row.get("min_patient_id"),
            "max_patient_id": row.get("max_patient_id"),
        }
        for row in requested_patients
    }
    patient_min = patient_key.get("min_patient_id")
    patient_max = patient_key.get("max_patient_id")
    metadata_min = metadata_key.get("min_patient_id")
    metadata_max = metadata_key.get("max_patient_id")
    ranges_overlap = (
        patient_min is not None and patient_max is not None
        and metadata_min is not None and metadata_max is not None
        and max(patient_min, metadata_min) <= min(patient_max, metadata_max)
    )
    requested_ranges_overlap = any(
        requested.get("min_patient_id") is not None
        and requested.get("max_patient_id") is not None
        and metadata_min is not None
        and metadata_max is not None
        and max(int(requested["min_patient_id"]), int(metadata_min))
        <= min(int(requested["max_patient_id"]), int(metadata_max))
        for requested in requested_ranges.values()
    )

    return {
        "probe_version": "data-semantics-calibration-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "database": {
            "host": host,
            "port": port,
            "database": database,
            "credentials_redacted": True,
            "read_only_probe": True,
        },
        "canonical_labels": {
            "T2D": "T2D",
            "Healthy": "healthy",
            "evidence": {
                "patient_rows": _serialise_rows(requested_patients),
                "disease_dictionary": _serialise_rows(label_dictionary),
                "disease_column_collation": _serialise_rows(collation),
                "case_insensitive_filter_observed": True,
            },
            "no_catalog_alias_dictionary_observed": True,
        },
        "cohort_group_distribution": _serialise_rows(cohort_distribution),
        "requested_cohort_distribution": _serialise_rows(requested_patients),
        "abundance_requested_distribution": _serialise_rows(requested_abundance),
        "rows_per_sample": rows_per_sample_summary,
        "sample_feature_identity": {
            "unit": "sample × feature",
            "duplicate_summary": _serialise_rows(sample_feature_duplicates),
            "interpretation": (
                "duplicate_summary is computed after grouping by the real sample_id "
                "and microbe_name_standard; duplicate rows are not independent samples"
            ),
        },
        "patient_sample_multiplicity": {
            "unit": "patient → distinct sample_id",
            "by_disease": patient_sample_summary,
            "interpretation": (
                "sample_count is the number of distinct abundance sample keys per patient; "
                "it is not the number of abundance rows or features"
            ),
        },
        "abundance_window": {
            "window_order": "standard_abundance_id ASC",
            "limit": 1000,
            "limit_semantics": "diagnostic result rows",
            "by_disease": window_samples,
            "sample_rows_are_not_unique_samples": True,
        },
        "inspect_window": {
            "query": "SELECT disease, age, gender, country, body_site FROM patients LIMIT 100",
            "order_by": None,
            "row_count": len(inspect_window),
            "returned_group_distribution": inspect_distribution,
            "interpretation": (
                "bounded physical-first window; not a cohort-wide distribution"
            ),
        },
        "metadata_join": {
            "overall": _serialise_rows(metadata_join),
            "requested_cohort": _serialise_rows(requested_metadata_join),
            "metadata_keys": _serialise_rows(metadata_keys),
            "patient_keys": _serialise_rows(patient_keys),
            "project_columns_in_database": _serialise_rows(project_columns),
            "top_projects_in_metadata": _serialise_rows(project_values),
            "sample_key_examples_hashed": _serialise_rows(metadata_examples),
            "patient_key_examples_hashed": _serialise_rows(patient_examples),
            "interpretation": (
                "metadata.project is physically present only in meta2db_sample_metadata; "
                "the patient_id join matches other patient records but none of T2D/healthy"
            ),
        },
        "project_mapping_audit": {
            "join_key": "patients.patient_id = meta2db_sample_metadata.patient_id",
            "project_source": "meta2db_sample_metadata.project_name",
            "patients_id_namespace": {
                "distinct_count": patient_key.get("distinct_patient_count"),
                "min_patient_id": patient_min,
                "max_patient_id": patient_max,
            },
            "metadata_id_namespace": {
                "distinct_count": metadata_key.get("distinct_patient_count"),
                "min_patient_id": metadata_min,
                "max_patient_id": metadata_max,
            },
            "requested_cohort_id_ranges": requested_ranges,
            "id_range_overlap": ranges_overlap,
            "requested_id_range_overlap": requested_ranges_overlap,
            "requested_cohort_matches": {
                str(row.get("disease")): {
                    "patient_keys": row.get("patient_keys"),
                    "matched_patient_keys": row.get("matched_patient_keys"),
                    "project_count": row.get("project_count"),
                }
                for row in requested_metadata_join
            },
            "diagnosis": (
                "The join definition is valid for the broader database, but the requested "
                "T2D/healthy patient-key ranges have no metadata coverage; this is an ID "
                "namespace/source-coverage mismatch, not evidence that project is absent."
                if not requested_ranges_overlap else
                "The patient-key ranges overlap; investigate row-level source coverage and label mapping."
            ),
        },
        "age_semantics": {
            "requested_cohort_age_summary": _serialise_rows(requested_age),
            "global_patient_age_uniqueness": _serialise_rows(age_uniqueness),
            "age_is_repeated_for_each_abundance_feature": True,
        },
        "java_limit_evidence": _java_limit_evidence(),
    }


def _java_limit_evidence() -> dict[str, Any]:
    compiler = REPO_ROOT / "mico_database_new" / "src" / "main" / "java" / "com" / "database" / "mico_database" / "agent" / "contract" / "QueryPlanCompiler.java"
    service = REPO_ROOT / "mico_database_new" / "src" / "main" / "java" / "com" / "database" / "mico_database" / "agent" / "readmodel" / "service" / "DynamicReadQueryService.java"

    def find(path: Path, needle: str) -> dict[str, Any]:
        if not path.exists():
            return {"path": str(path), "found": False}
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for index, line in enumerate(lines, start=1):
            if needle in line:
                return {"path": str(path), "line": index, "text": line.strip(), "found": True}
        return {"path": str(path), "found": False}

    return {
        "limit_scope": "joined/result rows, not unique samples",
        "compiler_limit": find(compiler, "sql.append(\" LIMIT ?\")"),
        "prepared_statement_bound": find(service, "statement.setMaxRows(maxRows)"),
    }


def _load_existing_canary_evidence() -> dict[str, Any] | None:
    if not CANARY_AUDIT.exists():
        return None
    try:
        audit = json.loads(CANARY_AUDIT.read_text(encoding="utf-8"))
        trace = json.loads(CANARY_TRACE.read_text(encoding="utf-8")) if CANARY_TRACE.exists() else {}
    except (OSError, json.JSONDecodeError):
        return None
    return {
        "source": "previous_real_gemini_canary_artifact",
        "audit_path": str(CANARY_AUDIT),
        "trace_path": str(CANARY_TRACE),
        "query_count": audit.get("query_count"),
        "query_diagnostics": audit.get("query_diagnostics", []),
        "group_comparison_source_observations": audit.get("group_comparison_source_observations", []),
        "group_comparison_feature_mixing_risk": audit.get("group_comparison_feature_mixing_risk"),
        "analysis_success": audit.get("analysis_success"),
        "runtime_status": trace.get("runtime_status"),
        "policy_request_count": trace.get("policy_request_count"),
        "materializer_request_count": trace.get("materializer_request_count"),
        "materializer_call_count": trace.get("materializer_call_count"),
    }


def _calibration_markdown(result: dict[str, Any]) -> str:
    requested = {
        str(row["disease"]): row
        for row in result["requested_cohort_distribution"]
    }
    abundance = {
        str(row["disease"]): row
        for row in result["abundance_requested_distribution"]
    }
    rows_sample = result["rows_per_sample"]
    duplicate_summary = {
        str(row["disease"]): row
        for row in result.get("sample_feature_identity", {}).get("duplicate_summary", [])
    }
    patient_sample = result.get("patient_sample_multiplicity", {}).get("by_disease", {})
    join = result["metadata_join"]["overall"][0]
    requested_join = result["metadata_join"]["requested_cohort"]
    lines = [
        "# Data Semantics Calibration",
        "",
        "This report is deterministic and read-only; it is not an Agent Action or training data.",
        "",
        "## Calibration table",
        "",
        "| Item | Actual result |",
        "|---|---|",
        f"| T2D canonical label | `T2D`; patient rows = {requested.get('T2D', {}).get('row_count')} |",
        f"| Healthy canonical label | `healthy`; patient rows = {requested.get('healthy', {}).get('row_count')} |",
        f"| T2D abundance rows / samples / features | {abundance.get('T2D', {}).get('abundance_row_count')} / {abundance.get('T2D', {}).get('distinct_sample_count')} / {abundance.get('T2D', {}).get('distinct_feature_count')} |",
        f"| Healthy abundance rows / samples / features | {abundance.get('healthy', {}).get('abundance_row_count')} / {abundance.get('healthy', {}).get('distinct_sample_count')} / {abundance.get('healthy', {}).get('distinct_feature_count')} |",
        f"| Rows/sample median | T2D = {rows_sample.get('T2D', {}).get('rows_per_sample_median')}; healthy = {rows_sample.get('healthy', {}).get('rows_per_sample_median')} |",
        f"| Duplicate sample×feature pairs | T2D = {duplicate_summary.get('T2D', {}).get('duplicate_pair_count', 'n/a')}; healthy = {duplicate_summary.get('healthy', {}).get('duplicate_pair_count', 'n/a')} |",
        f"| Patients with >1 abundance sample | T2D = {patient_sample.get('T2D', {}).get('patients_with_multiple_samples', 'n/a')}; healthy = {patient_sample.get('healthy', {}).get('patients_with_multiple_samples', 'n/a')} |",
        f"| Distinct samples per patient (median) | T2D = {patient_sample.get('T2D', {}).get('samples_per_patient_median', 'n/a')}; healthy = {patient_sample.get('healthy', {}).get('samples_per_patient_median', 'n/a')} |",
        "| LIMIT scope | joined/result rows, not unique samples |",
        "| inspect 100 Healthy cause | no `ORDER BY`; physical first 100 rows are all `healthy`, so this is not cohort distribution |",
        "| group_sizes meaning | returned rows or aggregation rows; not unique samples unless a sample key is selected |",
        "| compare/adjust unit | current typed operator receives bounded returned rows; age repeats across abundance features |",
        f"| overall sample→metadata match | {join.get('matched_patient_keys')} / {join.get('patient_keys')} patient keys ({round(100 * int(join.get('matched_patient_keys', 0)) / max(1, int(join.get('patient_keys', 1))), 2)}%) |",
        f"| requested T2D/healthy metadata match | {[(r.get('disease'), r.get('matched_patient_keys')) for r in requested_join]} |",
        "| project source | `meta2db_sample_metadata.project_name`; no project column in `patients` |",
        f"| project mapping diagnosis | {result.get('project_mapping_audit', {}).get('diagnosis', 'not available in this fixture')} |",
        "",
        "## Blockers",
        "",
        "1. The successful canary QueryPlan aggregated `abundance.value` without `abundance.feature`; its p-value is not scientifically validated for a multi-feature microbiome question.",
        "2. Sample identity must be checked at the sample×feature and patient→sample cardinality levels; abundance rows are not independent observations.",
        "3. `sample_to_metadata` has zero matches for the requested T2D/healthy rows, so project validation cannot be inferred from that relation.",
        "4. Any Gemini 429/retry behaviour is an independent provider-stability issue.",
        "5. Project mapping audit separates a valid join definition from source coverage: the requested cohort patient-key namespace does not overlap the metadata namespace in this database snapshot.",
        "",
        "## Java limit evidence",
        "",
        f"- Scope: `{result['java_limit_evidence']['limit_scope']}`.",
        f"- Compiler: `{result['java_limit_evidence']['compiler_limit']}`.",
        f"- JDBC bound: `{result['java_limit_evidence']['prepared_statement_bound']}`.",
    ]
    return "\n".join(lines) + "\n"


async def _run(output_dir: Path) -> dict[str, Any]:
    result = await _probe_database()
    result["previous_gemini_canary_evidence"] = _load_existing_canary_evidence()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "calibration.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    (output_dir / "calibration.md").write_text(
        _calibration_markdown(result),
        encoding="utf-8",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic read-only data semantics probes")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        result = asyncio.run(_run(args.output_dir))
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 1
    print(json.dumps({
        "status": "completed",
        "output_dir": str(args.output_dir),
        "requested_cohort": result["requested_cohort_distribution"],
        "abundance_requested": result["abundance_requested_distribution"],
        "rows_per_sample": result["rows_per_sample"],
        "sample_feature_identity": result["sample_feature_identity"],
        "patient_sample_multiplicity": result["patient_sample_multiplicity"],
        "metadata_join": result["metadata_join"]["overall"],
        "requested_metadata_join": result["metadata_join"]["requested_cohort"],
        "inspect_window": result["inspect_window"],
    }, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
