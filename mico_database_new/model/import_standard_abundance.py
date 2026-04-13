from __future__ import annotations

import argparse
import csv
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pymysql


DEFAULT_BATCH_SIZE = 5000


@dataclass
class PatientRef:
    patient_id: int
    patient_name: str


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Bulk import standardized abundance CSV into microbe_abundance_standard."
    )
    parser.add_argument(
        "--csv",
        default=str(root / "merged_abundance.csv"),
        help="Path to the standardized abundance matrix CSV.",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("MYSQL_HOST", "127.0.0.1"),
        help="MySQL host.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("MYSQL_PORT", "3306")),
        help="MySQL port.",
    )
    parser.add_argument(
        "--user",
        default=os.environ.get("MYSQL_USERNAME", "root"),
        help="MySQL username.",
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("MYSQL_PASSWORD", "ljy123"),
        help="MySQL password.",
    )
    parser.add_argument(
        "--database",
        default=os.environ.get("MYSQL_DATABASE", "patient_data_manager"),
        help="MySQL database name.",
    )
    parser.add_argument(
        "--table",
        default="microbe_abundance_standard",
        help="Target table name.",
    )
    parser.add_argument(
        "--match-field",
        choices=["patient_name", "patient_id"],
        default="patient_name",
        help="How CSV sample IDs are matched to the patients table.",
    )
    parser.add_argument(
        "--feature-version",
        default="v1",
        help="Feature version written into the target table.",
    )
    parser.add_argument(
        "--source-batch",
        default=None,
        help="Source batch label. Defaults to CSV filename.",
    )
    parser.add_argument(
        "--normalization-method",
        default="training_standardized",
        help="Normalization method label written into the target table.",
    )
    parser.add_argument(
        "--abundance-unit",
        default="relative_abundance",
        help="Abundance unit written into the target table.",
    )
    parser.add_argument(
        "--sample-date",
        default=None,
        help="Optional fixed sample_date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of rows per INSERT batch.",
    )
    parser.add_argument(
        "--skip-zero",
        action="store_true",
        help="Skip zero-abundance values to reduce table size.",
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Truncate the target table before import.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate without writing to MySQL.",
    )
    return parser.parse_args()


def connect_mysql(args: argparse.Namespace):
    return pymysql.connect(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        database=args.database,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.Cursor,
        autocommit=False,
    )


def load_patient_lookup(conn, match_field: str) -> Dict[str, PatientRef]:
    sql = f"SELECT patient_id, {match_field} FROM patients"
    result: Dict[str, PatientRef] = {}
    with conn.cursor() as cursor:
        cursor.execute(sql)
        for patient_id, match_value in cursor.fetchall():
            if match_value is None:
                continue
            result[str(match_value)] = PatientRef(patient_id=int(patient_id), patient_name=str(match_value))
    return result


def iter_matrix_rows(csv_path: Path) -> Iterable[Tuple[str, List[str], List[str]]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.reader(fp)
        header = next(reader)
        if len(header) < 2:
            raise ValueError("CSV header must contain tax column plus sample columns")
        sample_ids = header[1:]
        for row in reader:
            if not row:
                continue
            tax_name = row[0].strip()
            if not tax_name:
                continue
            values = row[1:]
            if len(values) != len(sample_ids):
                raise ValueError(
                    f"Row length mismatch for tax {tax_name}: expected {len(sample_ids)} values, got {len(values)}"
                )
            yield tax_name, sample_ids, values


def build_insert_sql(table: str) -> str:
    return f"""
        INSERT INTO {table} (
            patient_id,
            sample_id,
            sample_date,
            microbe_name_standard,
            microbe_name_hash,
            abundance_value,
            abundance_unit,
            feature_version,
            source_batch,
            normalization_method
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        ON DUPLICATE KEY UPDATE
            abundance_value = VALUES(abundance_value),
            abundance_unit = VALUES(abundance_unit),
            source_batch = VALUES(source_batch),
            normalization_method = VALUES(normalization_method)
    """.strip()


def flush_batch(conn, sql: str, batch: List[Tuple], dry_run: bool) -> int:
    if not batch:
        return 0
    if dry_run:
        count = len(batch)
        batch.clear()
        return count
    with conn.cursor() as cursor:
        cursor.executemany(sql, batch)
    conn.commit()
    count = len(batch)
    batch.clear()
    return count


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    source_batch = args.source_batch or csv_path.stem
    conn = connect_mysql(args)
    try:
        lookup = load_patient_lookup(conn, args.match_field)
        print(f"Loaded patients: {len(lookup)}")

        if args.truncate and not args.dry_run:
            with conn.cursor() as cursor:
                cursor.execute(f"TRUNCATE TABLE {args.table}")
            conn.commit()
            print(f"Truncated table: {args.table}")

        insert_sql = build_insert_sql(args.table)
        batch: List[Tuple] = []

        total_rows = 0
        inserted_rows = 0
        skipped_missing_patient = 0
        skipped_zero = 0
        missing_samples = set()

        for tax_name, sample_ids, values in iter_matrix_rows(csv_path):
            total_rows += 1
            if total_rows % 100 == 0:
                print(
                    f"Processed taxa: {total_rows}, inserted rows: {inserted_rows}, "
                    f"missing samples: {len(missing_samples)}",
                    end="\r",
                )

            for sample_id, raw_value in zip(sample_ids, values):
                patient_ref = lookup.get(sample_id)
                if patient_ref is None:
                    missing_samples.add(sample_id)
                    skipped_missing_patient += 1
                    continue

                try:
                    abundance_value = float(raw_value) if raw_value else 0.0
                except ValueError:
                    abundance_value = 0.0

                if args.skip_zero and abundance_value == 0.0:
                    skipped_zero += 1
                    continue

                batch.append(
                    (
                        patient_ref.patient_id,
                        sample_id,
                        args.sample_date,
                        tax_name,
                        hashlib.md5(tax_name.encode("utf-8")).hexdigest(),
                        abundance_value,
                        args.abundance_unit,
                        args.feature_version,
                        source_batch,
                        args.normalization_method,
                    )
                )

                if len(batch) >= args.batch_size:
                    inserted_rows += flush_batch(conn, insert_sql, batch, args.dry_run)

        inserted_rows += flush_batch(conn, insert_sql, batch, args.dry_run)

        print("\nImport finished")
        print(f"Taxa processed: {total_rows}")
        print(f"Rows inserted or updated: {inserted_rows}")
        print(f"Rows skipped because sample not found in patients: {skipped_missing_patient}")
        print(f"Rows skipped because abundance is zero: {skipped_zero}")

        if missing_samples:
            preview = sorted(missing_samples)[:20]
            print(f"Unmatched sample IDs ({len(missing_samples)}): {preview}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
