"""Low-disk importer for the complete Meta2DB species long table.

It streams one project at a time to a short-lived local CSV and loads it
directly into the target table.  The session disables binary logging, avoiding
the multi-gigabyte binlog growth that a staging import caused on this host.
"""
from __future__ import annotations
import csv, hashlib, os, re, shutil
from pathlib import Path
import pymysql

ROOT = Path(__file__).resolve().parent / "meta2db_species"
ABUNDANCE = ROOT / "meta2db_species_relative_abundance_long.csv"
TEMP = ROOT / "direct_load_tmp"
PREFIX = "meta2db-species-20260726"
MIN_FREE = 2 * 1024**3

def safe_project(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)

def load_project(conn, project: str, rows, patient_ids: dict[str, int]) -> int:
    TEMP.mkdir(exist_ok=True)
    path = TEMP / f"{safe_project(project)}.csv"
    count = 0
    with path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out, lineterminator="\n")
        for sample_id, microbe, value, _rank in rows:
            patient_id = patient_ids.get(sample_id)
            if patient_id is None:
                raise ValueError(f"No patient record for {sample_id}")
            writer.writerow((patient_id, sample_id, microbe, value))
            count += 1
    if shutil.disk_usage("D:\\").free < MIN_FREE:
        raise RuntimeError("D drive free space is below 2 GiB; stopping before the next project")
    batch = f"{PREFIX}:{project}"
    with conn.cursor() as cur:
        escaped = str(path).replace("\\", "\\\\").replace("'", "\\'")
        cur.execute(f"""
            LOAD DATA LOCAL INFILE '{escaped}' INTO TABLE microbe_abundance_standard
            CHARACTER SET utf8mb4 FIELDS TERMINATED BY ',' ENCLOSED BY '"' LINES TERMINATED BY '\\n'
            (patient_id, sample_id, @microbe, abundance_value)
            SET sample_date = NULL,
                microbe_name_standard = @microbe,
                microbe_name_hash = MD5(@microbe),
                abundance_unit = 'relative_abundance',
                feature_version = 'meta2db-species-v1',
                source_batch = '{batch}',
                normalization_method = 'per_sample_species_count'
        """)
        inserted = cur.rowcount
    conn.commit()
    path.unlink()
    free_bytes = shutil.disk_usage("D:\\").free
    print(f"Imported {project}: {inserted} rows; D free={free_bytes}", flush=True)
    return inserted

def main():
    conn = pymysql.connect(host=os.getenv("MYSQL_HOST", "127.0.0.1"), port=int(os.getenv("MYSQL_PORT", "3306")), user=os.getenv("MYSQL_USERNAME", "root"), password=os.environ["MYSQL_PASSWORD"], database=os.getenv("MYSQL_DATABASE", "patient_data_manager"), charset="utf8mb4", local_infile=True, autocommit=False)
    try:
        with conn.cursor() as cur:
            cur.execute("SET SESSION sql_log_bin = 0")
            cur.execute("SELECT patient_id, patient_name FROM patients WHERE patient_name IS NOT NULL")
            patient_ids = {name: int(pid) for pid, name in cur.fetchall()}
            cur.execute(
                "SELECT DISTINCT source_batch FROM microbe_abundance_standard "
                "WHERE source_batch LIKE %s",
                (f"{PREFIX}:%",),
            )
            completed_projects = {
                batch[len(PREFIX) + 1:]
                for (batch,) in cur.fetchall()
                if batch and batch.startswith(f"{PREFIX}:")
            }
        print(f"Resuming import: {len(completed_projects)} committed projects will be skipped", flush=True)
        total = 0
        with ABUNDANCE.open("r", encoding="utf-8", newline="") as source:
            reader = csv.reader(source); next(reader)
            current = None; rows = []
            for row in reader:
                project = row[0].split("|", 1)[0]
                if current is None: current = project
                if project != current:
                    if current in completed_projects:
                        print(f"Skipping committed project {current}", flush=True)
                    else:
                        total += load_project(conn, current, rows, patient_ids)
                    current, rows = project, []
                rows.append(row)
            if current is not None:
                if current in completed_projects:
                    print(f"Skipping committed project {current}", flush=True)
                else:
                    total += load_project(conn, current, rows, patient_ids)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM microbe_abundance_standard WHERE source_batch LIKE %s",
                (f"{PREFIX}:%",),
            )
            cumulative = int(cur.fetchone()[0])
        print(
            f"Completed direct Meta2DB import: {total} rows this run; "
            f"{cumulative} cumulative Meta2DB rows",
            flush=True,
        )
    finally:
        conn.close()

if __name__ == "__main__": main()
