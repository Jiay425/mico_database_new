"""Extract detailed Meta2DB disease fields into patients.disease.

Only samples with matched raw Meta2DB metadata are changed.  The raw JSON and
the original disease_category remain untouched for auditability.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import json
import os
import re
import shlex
from collections import OrderedDict

import paramiko
import pymysql

DETAIL_FIELDS = (
    "cancer_disord",
    "cardiovascular_disord",
    "dermatology_disord",
    "endocrine_disord",
    "gastrointest_disord",
    "genitourinary_disord",
    "immune_disord",
    "liver_disord",
    "metabolic_disord",
    "muscoskeletal_disord",
    "neuro_disord",
    "nose_mouth_teeth_throat_disord",
    "other_disord",
    "pulmonary_disord",
    "birth_disord",
    "host_disease_stat",
    "disease_notes",
)
MISSING = {
    "", "not available", "not applicable", "na", "n/a", "nan", "none",
    "null", "unknown", "false", "true", "control", "case", "disease",
    "healthy", "health", "healthy subject study",
}
CHUNK = 500
REMOTE_HOST = os.environ["REMOTE_SSH_HOST"]
REMOTE_USER = os.environ["REMOTE_SSH_USER"]
REMOTE_PASSWORD = os.environ["REMOTE_SSH_PASSWORD"]


def split_values(value):
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    result = []
    seen = set()
    for item in values:
        if not isinstance(item, str):
            continue
        for token in re.split(r"[,;]", item):
            token = re.sub(r"\s+", " ", token.replace("_", " ")).strip()
            key = token.lower()
            if token and key not in MISSING and key not in seen:
                seen.add(key)
                result.append(token.lower())
    return result


def derive_disease(category, raw_metadata):
    try:
        raw = json.loads(raw_metadata or "{}") if isinstance(raw_metadata, str) else (raw_metadata or {})
    except (TypeError, ValueError):
        raw = {}
    values = []
    seen = set()
    for field in DETAIL_FIELDS:
        for value in split_values(raw.get(field)):
            if value not in seen:
                seen.add(value)
                values.append(value)
    detailed = bool(values)
    if not values:
        values = split_values(category)
    return ";".join(values), values, detailed


def connect(port):
    return pymysql.connect(
        host=os.getenv("MYSQL_HOST", "127.0.0.1"),
        port=port,
        user=os.getenv("MYSQL_USERNAME", "root"),
        password=os.environ["MYSQL_PASSWORD"],
        database=os.getenv("MYSQL_DATABASE", "patient_data_manager"),
        charset="utf8mb4",
        autocommit=False,
        cursorclass=pymysql.cursors.DictCursor,
    )


def fetch_remote_metadata_over_ssh():
    """Fetch compressed raw JSON on the server, avoiding a huge uncompressed TCP transfer."""
    query = (
        "SELECT patient_id, disease_category, raw_metadata "
        "FROM meta2db_sample_metadata WHERE raw_metadata IS NOT NULL ORDER BY patient_id"
    )
    encoded = base64.b64encode(query.encode("utf-8")).decode("ascii")
    pipeline = (
        f"echo {encoded} | base64 -d | "
        "docker exec -i mico-mysql mysql --batch --raw --skip-column-names "
        "-uroot " + shlex.quote("-p" + os.environ["REMOTE_MYSQL_PASSWORD"]) + " patient_data_manager | gzip -c"
    )
    command = "sudo -S -p '' bash -lc " + shlex.quote(pipeline)
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(REMOTE_HOST, username=REMOTE_USER, password=REMOTE_PASSWORD,
                look_for_keys=False, allow_agent=False, timeout=30)
    stdin, stdout, stderr = ssh.exec_command(command)
    stdin.write(REMOTE_PASSWORD + "\n")
    stdin.flush()
    compressed = stdout.read()
    error = stderr.read().decode("utf-8", errors="replace")
    exit_status = stdout.channel.recv_exit_status()
    ssh.close()
    if exit_status != 0:
        raise RuntimeError(f"远程元数据读取失败(exit={exit_status}): {error}")
    if error and "password" not in error.lower() and "warning" not in error.lower():
        raise RuntimeError(error)
    data = gzip.decompress(compressed).decode("utf-8", errors="replace")
    result = OrderedDict()
    for line in data.splitlines():
        parts = line.split("\t", 2)
        if len(parts) < 3:
            continue
        patient_id, category, raw_metadata = parts
        disease, values, detailed = derive_disease(category, raw_metadata)
        result[int(patient_id)] = {"disease": disease, "values": values, "detailed": detailed}
    return result


def fetch_metadata(conn):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT patient_id, disease_category, raw_metadata "
            "FROM meta2db_sample_metadata WHERE raw_metadata IS NOT NULL"
        )
        rows = cur.fetchall()
    result = OrderedDict()
    for row in rows:
        disease, values, detailed = derive_disease(row["disease_category"], row["raw_metadata"])
        result[int(row["patient_id"])] = {
            "disease": disease,
            "values": values,
            "detailed": detailed,
        }
    return result


def fetch_patient_diseases(conn, patient_ids):
    result = {}
    with conn.cursor() as cur:
        for start in range(0, len(patient_ids), CHUNK):
            ids = patient_ids[start:start + CHUNK]
            marks = ",".join(["%s"] * len(ids))
            cur.execute(
                f"SELECT patient_id, disease FROM patients WHERE patient_id IN ({marks})",
                ids,
            )
            result.update({int(row["patient_id"]): row["disease"] for row in cur.fetchall()})
    return result


def apply_updates(conn, updates, label):
    patient_ids = list(updates.keys())
    current = fetch_patient_diseases(conn, patient_ids)
    if set(current) != set(patient_ids):
        raise RuntimeError(f"{label}: patients 表缺少匹配样本，已停止写入")

    disease_names = sorted({value for item in updates.values() for value in item["values"]})
    associations = [(patient_id, value) for patient_id, item in updates.items() for value in item["values"]]
    conn.begin()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "CREATE TEMPORARY TABLE tmp_meta2db_disease_refinement ("
                "patient_id INT NOT NULL PRIMARY KEY, "
                "disease VARCHAR(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL)"
            )
            cur.executemany(
                "INSERT INTO tmp_meta2db_disease_refinement (patient_id, disease) VALUES (%s, %s)",
                [(patient_id, item["disease"] or "") for patient_id, item in updates.items()],
            )
            if disease_names:
                print(f"{label}: inserting disease dictionary ({len(disease_names)})", flush=True)
                cur.executemany(
                    "INSERT INTO diseases (disease_name) VALUES (%s) "
                    "ON DUPLICATE KEY UPDATE disease_name=VALUES(disease_name)",
                    [(name,) for name in disease_names],
                )
            cur.execute(
                "UPDATE patients p JOIN tmp_meta2db_disease_refinement t "
                "ON t.patient_id=p.patient_id SET p.disease=t.disease"
            )
            print(f"{label}: patients updated", flush=True)
            cur.execute(
                "DELETE pd FROM patient_diseases pd "
                "JOIN tmp_meta2db_disease_refinement t ON t.patient_id=pd.patient_id"
            )
            print(f"{label}: old disease links removed", flush=True)

            disease_ids = {}
            for start in range(0, len(disease_names), CHUNK):
                names = disease_names[start:start + CHUNK]
                marks = ",".join(["%s"] * len(names))
                cur.execute(f"SELECT disease_id, disease_name FROM diseases WHERE disease_name IN ({marks})", names)
                disease_ids.update({row["disease_name"]: int(row["disease_id"]) for row in cur.fetchall()})

            links = [(patient_id, disease_ids[value]) for patient_id, value in associations]
            if links:
                print(f"{label}: inserting disease links ({len(links)})", flush=True)
                cur.executemany(
                    "INSERT IGNORE INTO patient_diseases "
                    "(patient_id, disease_id, notes) VALUES (%s, %s, %s)",
                    [(patient_id, disease_id, "Imported from Meta2DB detailed disease metadata")
                     for patient_id, disease_id in links],
                )
        conn.commit()
        print(f"{label}: transaction committed", flush=True)
    except Exception:
        conn.rollback()
        raise

    after = fetch_patient_diseases(conn, patient_ids)
    expected = {patient_id: item["disease"] or None for patient_id, item in updates.items()}
    if after != expected:
        raise RuntimeError(f"{label}: 写入后 patients.disease 校验不一致")
    return len(disease_names), len(links)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote-port", type=int, default=13306)
    parser.add_argument("--local-port", type=int, default=3306)
    parser.add_argument("--remote-only", action="store_true")
    parser.add_argument("--local-only", action="store_true")
    args = parser.parse_args()

    remote = connect(args.remote_port)
    local = None
    try:
        print("reading remote raw metadata", flush=True)
        remote_meta = fetch_remote_metadata_over_ssh()
        print(f"remote raw metadata loaded: {len(remote_meta)}", flush=True)
        if len(remote_meta) != 11811:
            raise RuntimeError(f"远程匹配元数据数量异常: {len(remote_meta)}")

        remote_current = fetch_patient_diseases(remote, list(remote_meta))
        print("remote patient rows checked", flush=True)
        for patient_id, item in remote_meta.items():
            if not item["values"]:
                item["disease"] = remote_current[patient_id]
                item["values"] = split_values(remote_current[patient_id])

        if not args.remote_only:
            local = connect(args.local_port)
            local_patient_ids = set(fetch_patient_diseases(local, list(remote_meta)))
            print("local patient rows checked", flush=True)
            if local_patient_ids != set(remote_meta):
                raise RuntimeError("本地 patients 表缺少远程匹配样本，已停止写入")
        detailed = sum(1 for item in remote_meta.values() if item["detailed"])
        fallback = len(remote_meta) - detailed
        print(f"matched={len(remote_meta)} detailed={detailed} fallback={fallback}")
        print(f"max disease length={max(len(item['disease'] or '') for item in remote_meta.values())}")

        print("updating remote", flush=True)
        if not args.local_only:
            names, links = apply_updates(remote, remote_meta, "remote")
            print(f"remote updated: disease_names={names} patient_disease_links={links}")
        if local is not None:
            print("updating local", flush=True)
            names, links = apply_updates(local, remote_meta, "local")
            print(f"local updated: disease_names={names} patient_disease_links={links}")
    finally:
        if local is not None:
            local.close()
        remote.close()


if __name__ == "__main__":
    main()
