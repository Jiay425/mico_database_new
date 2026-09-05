"""Import the validated complete Meta2DB species batch into MySQL.

The process is deliberately two-stage: it first creates only missing patient
records, then loads the validated long-form CSV to an isolated staging table,
and finally performs one INSERT...SELECT into the standard abundance table.
This avoids duplicate data for existing sample IDs and makes a failed load
recoverable from the staging table without re-reading the raw profiles.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import pymysql


FEATURE_VERSION = "meta2db-species-v1"
SOURCE_BATCH = "meta2db-species-20260726"
STAGE_TABLE = "meta2db_species_stage_20260726"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent / "meta2db_species"
    parser = argparse.ArgumentParser(description="Import the complete species-level Meta2DB batch.")
    parser.add_argument("--samples", default=str(root / "meta2db_species_samples.csv"))
    parser.add_argument("--abundance", default=str(root / "meta2db_species_relative_abundance_long.csv"))
    parser.add_argument("--host", default=os.environ.get("MYSQL_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MYSQL_PORT", "3306")))
    parser.add_argument("--user", default=os.environ.get("MYSQL_USERNAME", "root"))
    parser.add_argument("--password", default=os.environ.get("MYSQL_PASSWORD"), required="MYSQL_PASSWORD" not in os.environ)
    parser.add_argument("--database", default=os.environ.get("MYSQL_DATABASE", "patient_data_manager"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-stage", action="store_true", help="Keep staging data after a successful import.")
    parser.add_argument("--resume-stage", action="store_true", help="Use an existing staging table after a recoverable final-insert failure.")
    return parser.parse_args()


def read_samples(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_id", "project_name", "disease", "body_site"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Samples file must contain {sorted(required)}")
    sample_ids = [row["sample_id"].strip() for row in rows]
    if not all(sample_ids) or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Samples file has empty or duplicate sample IDs")
    return rows


def connect(args: argparse.Namespace):
    return pymysql.connect(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        database=args.database,
        charset="utf8mb4",
        local_infile=True,
        autocommit=False,
    )


def main() -> None:
    args = parse_args()
    samples_path = Path(args.samples).resolve()
    abundance_path = Path(args.abundance).resolve()
    if not samples_path.exists() or not abundance_path.exists():
        raise FileNotFoundError("Samples or abundance file is missing")
    samples = read_samples(samples_path)
    connection = connect(args)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT patient_name FROM patients WHERE patient_name IS NOT NULL")
            existing = {row[0] for row in cursor.fetchall()}
            cursor.execute("SELECT COUNT(*) FROM microbe_abundance_standard WHERE source_batch = %s", (SOURCE_BATCH,))
            existing_batch_rows = int(cursor.fetchone()[0])
            cursor.execute("SHOW TABLES LIKE %s", (STAGE_TABLE,))
            stage_exists = cursor.fetchone() is not None

        incoming_ids = {row["sample_id"].strip() for row in samples}
        overlap = sorted(incoming_ids & existing)
        print(f"Samples in input: {len(samples)}")
        print(f"Existing sample IDs: {len(overlap)}")
        print(f"Existing rows for this source batch: {existing_batch_rows}")
        print(f"Existing staging table: {stage_exists}")
        # A prior run may have completed the patient phase but failed before the
        # staging load.  It is safe to resume only when *all* incoming IDs are
        # already present and this source batch/staging table is still absent.
        if overlap and len(overlap) != len(incoming_ids):
            raise ValueError(f"Refusing import because only some sample IDs already exist: {overlap[:10]}")
        if existing_batch_rows:
            raise ValueError("Refusing import because this batch or its staging table already exists")
        if stage_exists and not args.resume_stage:
            raise ValueError("Refusing import because this batch or its staging table already exists")
        if args.dry_run:
            return

        patient_rows = [
            (
                row["sample_id"].strip(),
                None,
                row.get("body_site", "").strip() or None,
                row.get("disease", "").strip() or None,
                None,
                row.get("project_name", "").strip() or None,
            )
            for row in samples
            if row["sample_id"].strip() not in existing
        ]
        if patient_rows and not args.resume_stage:
            with connection.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO patients (patient_name, `Group`, body_site, disease, country, sequencing_platform)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    patient_rows,
                )
            connection.commit()
        print(f"Inserted patients: {len(patient_rows)}")

        if not args.resume_stage:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    CREATE TABLE {STAGE_TABLE} (
                        sample_id VARCHAR(191) CHARACTER SET utf8mb4 NOT NULL,
                        microbe_name_standard VARCHAR(1024) CHARACTER SET utf8mb4 NOT NULL,
                        abundance_value DOUBLE NOT NULL,
                        source_taxonomy_rank VARCHAR(32) CHARACTER SET utf8mb4 NOT NULL,
                        KEY idx_stage_sample (sample_id)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                    """
                )
                local_path = str(abundance_path).replace("\\", "\\\\").replace("'", "\\'")
                cursor.execute(
                    f"""
                    LOAD DATA LOCAL INFILE '{local_path}'
                    INTO TABLE {STAGE_TABLE}
                    CHARACTER SET utf8mb4
                    FIELDS TERMINATED BY ',' ENCLOSED BY '"'
                    LINES TERMINATED BY '\\n'
                    IGNORE 1 LINES
                    (sample_id, microbe_name_standard, abundance_value, source_taxonomy_rank)
                    """
                )
            connection.commit()
            print("Loaded validated abundance CSV into staging table")

        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO microbe_abundance_standard (
                    patient_id, sample_id, sample_date, microbe_name_standard,
                    microbe_name_hash, abundance_value, abundance_unit,
                    feature_version, source_batch, normalization_method
                )
                SELECT
                    p.patient_id, s.sample_id, NULL, s.microbe_name_standard,
                    MD5(s.microbe_name_standard), s.abundance_value, 'relative_abundance',
                    %s, %s, 'per_sample_species_count'
                FROM {STAGE_TABLE} s
                INNER JOIN patients p
                    ON p.patient_name COLLATE utf8mb4_unicode_ci = s.sample_id COLLATE utf8mb4_unicode_ci
                """,
                (FEATURE_VERSION, SOURCE_BATCH),
            )
            inserted = cursor.rowcount
        connection.commit()
        print(f"Inserted abundance rows: {inserted}")

        if not args.keep_stage:
            with connection.cursor() as cursor:
                cursor.execute(f"DROP TABLE {STAGE_TABLE}")
            connection.commit()
            print("Dropped staging table")
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


if __name__ == "__main__":
    main()
