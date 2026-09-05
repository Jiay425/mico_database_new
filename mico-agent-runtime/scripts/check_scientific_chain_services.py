"""No-quota preflight for the Scientific Agent chain.

The command intentionally makes no Gemini/DeepSeek/Qwen request.  It checks
the remote MySQL tunnel and the configured Java Agent Tool boundary, then
performs one bounded, read-only Java query.  The Gemini canary must call this
gate before Task Understanding so an unhealthy chain cannot consume model
quota.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import asyncmy

# Allow both ``python -m scripts.check_scientific_chain_services`` and the
# convenient direct ``python scripts/check_scientific_chain_services.py``.
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mico_agent_runtime.contracts.materialization import QueryPlan
from mico_agent_runtime.contracts.tools import (
    ExecuteReadQueryArguments,
    ExecuteReadQueryJavaToolCall,
)
from mico_agent_runtime.ports.java_agent import HttpJavaAgentToolPort
from mico_agent_runtime.ports.schema_catalog import JavaSchemaCatalogPort


def _env(name: str, fallback: str) -> str:
    value = os.environ.get(name, fallback)
    return value.strip() if isinstance(value, str) else fallback


def _ids() -> tuple[str, str, str]:
    seed = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    run_id = "preflight-" + hashlib.sha256(seed.encode()).hexdigest()[:24]
    task_id = "task-" + hashlib.sha256((seed + "task").encode()).hexdigest()[:32]
    call_id = "call-" + hashlib.sha256((seed + "call").encode()).hexdigest()[:32]
    return run_id, task_id, call_id


def _tcp_probe(host: str, port: int, timeout: float = 3.0) -> dict[str, Any]:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return {"status": "pass", "host": host, "port": port}
    except OSError as exc:
        return {"status": "fail", "host": host, "port": port, "error": type(exc).__name__}


async def _mysql_probe() -> dict[str, Any]:
    host = _env("MICO_PROBE_DB_HOST", _env("MYSQL_HOST", "127.0.0.1"))
    port = int(_env("MICO_PROBE_DB_PORT", _env("MYSQL_PORT", "13306")))
    user = _env("MICO_PROBE_DB_USER", _env("MYSQL_USERNAME", "root"))
    password = os.environ.get("MICO_PROBE_DB_PASSWORD") or os.environ["MYSQL_PASSWORD"]
    database = _env("MICO_PROBE_DB_NAME", _env("MYSQL_DATABASE", "patient_data_manager"))
    tcp = _tcp_probe(host, port)
    if tcp["status"] != "pass":
        return {"status": "fail", "tcp": tcp}
    connection = None
    try:
        connection = await asyncmy.connect(
            host=host, port=port, user=user, password=password, db=database,
            connect_timeout=5, read_timeout=10,
        )
        async with connection.cursor(asyncmy.cursors.DictCursor) as cursor:
            await cursor.execute("SELECT DATABASE() AS database_name, 1 AS probe")
            identity = (await cursor.fetchone()) or {}
            await cursor.execute("SELECT COUNT(*) AS patient_count FROM patients")
            patients = (await cursor.fetchone()) or {}
        return {
            "status": "pass",
            "tcp": tcp,
            "host": host,
            "port": port,
            "database": identity.get("database_name"),
            "patient_count": int(patients.get("patient_count") or 0),
            "read_only_probe": True,
        }
    except Exception as exc:
        return {"status": "fail", "tcp": tcp, "error": type(exc).__name__}
    finally:
        if connection is not None:
            connection.close()


def _java_probe() -> dict[str, Any]:
    base_url = _env("MICO_JAVA_AGENT_TOOL_BASE_URL", "")
    token = os.environ.get("MICO_AGENT_INTERNAL_TOKEN", "")
    if not base_url or not token:
        return {"status": "fail", "error": "JAVA_TOOL_CONFIGURATION_MISSING"}
    parsed = urlsplit(base_url)
    if parsed.hostname is None or parsed.port is None:
        return {"status": "fail", "error": "JAVA_TOOL_URL_INVALID"}
    tcp = _tcp_probe(parsed.hostname, parsed.port)
    if tcp["status"] != "pass":
        return {"status": "fail", "tcp": tcp}
    run_id, task_id, call_id = _ids()
    java = None
    try:
        java = HttpJavaAgentToolPort(base_url, token)
        catalog = JavaSchemaCatalogPort(java).load(run_id=run_id, task_id=task_id)
        field_ids = {
            field.fieldId
            for entity in catalog.entities
            for field in entity.fields
            if field.fieldId
        }
        plan = QueryPlan(root_entity="sample", select_fields=["sample.disease"], limit=1)
        call = ExecuteReadQueryJavaToolCall(
            toolName="execute_read_query",
            runId=run_id,
            toolCallId=call_id,
            arguments=ExecuteReadQueryArguments(queryPlan=plan),
        )
        response = java.execute(call)
        data = response.data if isinstance(response.data, dict) else {}
        columns = data.get("columns") if isinstance(data.get("columns"), list) else []
        rows = data.get("rows") if isinstance(data.get("rows"), list) else []
        required_opaque = "analysis.sample_key" in set(catalog.internalAnalysisFields)
        if response.status != "COMPLETED":
            return {"status": "fail", "error": "JAVA_READ_PROBE_NOT_COMPLETED", "response_status": response.status}
        if "sample.disease" not in field_ids:
            return {"status": "fail", "error": "JAVA_CATALOG_SAMPLE_DISEASE_MISSING"}
        if required_opaque is False:
            return {
                "status": "fail",
                "error": "JAVA_SERVICE_STALE_NO_OPAQUE_SAMPLE_KEY",
                "catalog_schema_version": catalog.schemaVersion,
                "catalog_generated_at": catalog.generatedAt.isoformat(),
                "catalog_entity_count": len(catalog.entities),
                "read_status": response.status,
                "read_row_count": response.rowCount,
            }
        return {
            "status": "pass",
            "tcp": tcp,
            "catalog_schema_version": catalog.schemaVersion,
            "catalog_entity_count": len(catalog.entities),
            "internal_analysis_fields": list(catalog.internalAnalysisFields),
            "read_status": response.status,
            "read_row_count": response.rowCount,
            "read_columns": columns,
            "read_rows_observed": len(rows),
            "opaque_sample_key_contract": "present",
        }
    except Exception as exc:
        return {"status": "fail", "tcp": tcp, "error": type(exc).__name__}
    finally:
        if java is not None:
            java.close()


async def run_preflight() -> dict[str, Any]:
    mysql, java = await asyncio.gather(_mysql_probe(), asyncio.to_thread(_java_probe))
    checks = {"remote_mysql": mysql, "java_agent_tool": java}
    return {
        "preflight_version": "scientific-chain-preflight-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gemini_calls_made": 0,
        "deepseek_calls_made": 0,
        "checks": checks,
        "status": "pass" if all(item.get("status") == "pass" for item in checks.values()) else "fail",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check remote MySQL and Java Agent Tool without model calls")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = asyncio.run(run_preflight())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
