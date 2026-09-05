"""Attach complete Meta2DB sample metadata without touching abundance rows."""
from __future__ import annotations

import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import pymysql

ROOT = Path(__file__).resolve().parent / "meta2db_species"
SAMPLES = ROOT / "meta2db_species_samples.csv"
METADATA = Path(__file__).resolve().parents[2] / "data" / "meta2db" / "metadata" / "meta2db_metadata.csv"
SOURCE_BATCH = "meta2db-species-20260726"
MISSING = {"", "not available", "not applicable", "na", "n/a", "nan", "none", "null", "unknown"}
IDENTIFIER_FIELDS = ("filename_match", "run_acc", "experiment_acc", "biosample_id", "external_id", "source_mat_id")
DETAILED_DISEASE_FIELDS = (
    "cancer_disord", "cardiovascular_disord", "dermatology_disord", "endocrine_disord",
    "gastrointest_disord", "genitourinary_disord", "immune_disord", "liver_disord",
    "metabolic_disord", "muscoskeletal_disord", "neuro_disord",
    "nose_mouth_teeth_throat_disord", "other_disord", "pulmonary_disord", "birth_disord",
    "host_disease_stat", "disease_notes",
)


def usable(value: str | None) -> str | None:
    value = (value or "").strip()
    return value if value.lower() not in MISSING else None


def detailed_disease(record: dict[str, str] | None, fallback: str | None) -> str | None:
    """Prefer precise disorder fields while retaining the broad category as fallback.

    The raw disease_category column is intentionally kept in the metadata table;
    this value is only the patient-facing disease field.
    """
    if record:
        values: list[str] = []
        for field in DETAILED_DISEASE_FIELDS:
            value = usable(record.get(field))
            if not value:
                continue
            for item in re.split(r"[,;]", value):
                item = item.strip().replace("_", " ")
                if item and item.lower() not in MISSING and item.lower() not in {"control", "case", "disease"}:
                    if item.lower() not in {existing.lower() for existing in values}:
                        values.append(item)
        if values:
            return ";".join(values)
    return usable(fallback)


def number(value: str | None, low: float, high: float) -> float | None:
    value = usable(value)
    if not value:
        return None
    found = re.search(r"[-+]?\d+(?:\.\d+)?", value)
    if not found:
        return None
    parsed = float(found.group())
    return parsed if low <= parsed <= high else None


def read_metadata() -> dict[tuple[str, str], list[tuple[str, dict[str, str]]]]:
    index: dict[tuple[str, str], list[tuple[str, dict[str, str]]]] = defaultdict(list)
    with METADATA.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        for record in csv.DictReader(handle):
            project = usable(record.get("project_name"))
            if not project:
                continue
            for field in IDENTIFIER_FIELDS:
                identifier = usable(record.get(field))
                if identifier:
                    index[(project, identifier)].append((field, record))
    return index


def candidate_identifiers(profile_sample: str) -> list[str]:
    candidates = [profile_sample]
    if "_" in profile_sample:
        candidates.append(profile_sample.split("_", 1)[0])
    if "." in profile_sample:
        candidates.append(profile_sample.split(".", 1)[0])
    return list(dict.fromkeys(filter(None, candidates)))


def find_record(index, project: str, profile_sample: str):
    for identifier in candidate_identifiers(profile_sample):
        matches = index.get((project, identifier), [])
        unique = {json.dumps(record, sort_keys=True): (field, record) for field, record in matches}
        if len(unique) == 1:
            return next(iter(unique.values()))
    return None, None


def main() -> None:
    metadata_index = read_metadata()
    with SAMPLES.open("r", encoding="utf-8-sig", newline="") as handle:
        samples = list(csv.DictReader(handle))

    conn = pymysql.connect(
        host=os.getenv("MYSQL_HOST", "127.0.0.1"), port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USERNAME", "root"), password=os.environ["MYSQL_PASSWORD"],
        database=os.getenv("MYSQL_DATABASE", "patient_data_manager"), charset="utf8mb4", autocommit=False,
    )
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT patient_id, patient_name FROM patients WHERE patient_name LIKE %s", ("%|%",))
            patient_ids = {name: int(patient_id) for patient_id, name in cur.fetchall()}
            cur.execute("""
                CREATE TABLE IF NOT EXISTS meta2db_sample_metadata (
                    patient_id INT NOT NULL PRIMARY KEY,
                    sample_id VARCHAR(255) NOT NULL UNIQUE,
                    project_name VARCHAR(255) NOT NULL,
                    profile_sample VARCHAR(255) NOT NULL,
                    metadata_match_field VARCHAR(64) NULL,
                    health_disease_status VARCHAR(255) NULL,
                    disease_category VARCHAR(255) NULL,
                    body_product VARCHAR(255) NULL,
                    body_site_detail VARCHAR(255) NULL,
                    country VARCHAR(255) NULL,
                    sequencing_platform VARCHAR(255) NULL,
                    raw_metadata JSON NULL,
                    source_batch VARCHAR(128) NOT NULL,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    KEY idx_meta2db_project (project_name),
                    KEY idx_meta2db_disease (disease_category)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)

        patient_updates = []
        metadata_rows = []
        matched = 0
        field_counts = defaultdict(int)
        for sample in samples:
            sample_id = sample["sample_id"].strip()
            patient_id = patient_ids.get(sample_id)
            if patient_id is None:
                raise ValueError(f"Missing Meta2DB patient row: {sample_id}")
            project = sample["project_name"].strip()
            profile_sample = sample["profile_sample"].strip()
            match_field, record = find_record(metadata_index, project, profile_sample)
            if record is not None:
                matched += 1
            group = usable((record or {}).get("health_disease_stat")) or usable(sample.get("group"))
            disease = detailed_disease(record, usable(sample.get("disease")))
            body_product = usable((record or {}).get("host_body_product")) or usable(sample.get("body_site"))
            body_site_detail = usable((record or {}).get("host_body_site"))
            country = usable((record or {}).get("geo_loc_name"))
            platform = usable((record or {}).get("seq_meth"))
            age_value = number((record or {}).get("host_age"), 0, 150)
            age = int(age_value) if age_value is not None else None
            gender = usable((record or {}).get("host_sex"))
            bmi = number((record or {}).get("host_body_mass_index"), 5, 100)
            for name, value in (("group", group), ("disease", disease), ("body_site", body_product), ("age", age), ("gender", gender), ("country", country), ("platform", platform), ("bmi", bmi)):
                if value is not None:
                    field_counts[name] += 1
            patient_updates.append((group, body_product, disease, age, gender, country, platform, bmi, patient_id))
            metadata_rows.append((
                patient_id, sample_id, project, profile_sample, match_field, group, disease, body_product,
                body_site_detail, country, platform,
                json.dumps(record, ensure_ascii=False) if record is not None else None, SOURCE_BATCH,
            ))

        with conn.cursor() as cur:
            cur.executemany("""
                UPDATE patients SET
                    `Group` = %s,
                    body_site = %s,
                    disease = %s,
                    age = %s,
                    gender = %s,
                    country = %s,
                    sequencing_platform = %s,
                    BMI = %s
                WHERE patient_id = %s
            """, patient_updates)
            cur.executemany("""
                INSERT INTO meta2db_sample_metadata (
                    patient_id, sample_id, project_name, profile_sample, metadata_match_field,
                    health_disease_status, disease_category, body_product, body_site_detail,
                    country, sequencing_platform, raw_metadata, source_batch
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    project_name=VALUES(project_name), profile_sample=VALUES(profile_sample),
                    metadata_match_field=VALUES(metadata_match_field), health_disease_status=VALUES(health_disease_status),
                    disease_category=VALUES(disease_category), body_product=VALUES(body_product),
                    body_site_detail=VALUES(body_site_detail), country=VALUES(country),
                    sequencing_platform=VALUES(sequencing_platform), raw_metadata=VALUES(raw_metadata),
                    source_batch=VALUES(source_batch)
            """, metadata_rows)
        conn.commit()
        print(json.dumps({"samples": len(samples), "metadata_matched": matched, "metadata_unmatched": len(samples)-matched, "patient_fields_written": dict(field_counts)}, ensure_ascii=False))
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
