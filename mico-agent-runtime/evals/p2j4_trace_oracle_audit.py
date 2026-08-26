"""Offline audit for the seven trace-backed result oracles.

This module reads only existing redacted checkpoint traces.  It performs no
model, Java, database, vector, graph, or network call and emits only closed
verification metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mico_agent_runtime.contracts.trace_eval import EvalTask, TraceProjection

from evals.p2j4_controlled_scenarios import verify_result_oracle
from evals.p2j4_result_oracles import BadCaseResultOracle, validate_result_oracle_set
from evals.p2j4_runner import load_task_set
from evals.p2j4_trace_assertions import closed_observation_for_trace


TRACE_DIR = Path(__file__).with_name("p2j4-bad-case-rerun-20260823")
TRACE_FILES = {
    # The current oracle accepts the repaired project -> country ordering;
    # use the latest redacted Gemini trace rather than the historical attempt
    # whose ordering preceded the task/oracle calibration.
    "p2j4-open-exploration-007": "../p2j4-gemini-retry-20260824/p2j4-open-exploration-007-retry2.json",
    "p2j4-open-exploration-008": "p2j4-open-exploration-008.json",
    "p2j4-open-exploration-010": "p2j4-open-exploration-010-attempt3.json",
    "p2j4-open-exploration-012": "p2j4-open-exploration-012.json",
    "p2j4-open-exploration-017": "p2j4-open-exploration-017.json",
    "p2j4-open-exploration-018": "p2j4-open-exploration-018-attempt2.json",
    "p2j4-open-exploration-020": "p2j4-open-exploration-020.json",
}


def _read_trace(path: Path) -> TraceProjection:
    payload = json.loads(path.read_text(encoding="utf-8"))
    traces = payload.get("traces") if isinstance(payload, dict) else None
    if not isinstance(traces, list) or len(traces) != 1:
        raise ValueError("TRACE_AUDIT_TRACE_COUNT_INVALID")
    return TraceProjection.model_validate(traces[0])


def audit_existing_traces(
    *,
    trace_dir: Path = TRACE_DIR,
    selected_files: dict[str, str] = TRACE_FILES,
) -> dict[str, Any]:
    tasks = {task.caseId: task for task in load_task_set()}
    oracle_records = validate_result_oracle_set(list(tasks.values()))["oracles"]
    oracles = {
        oracle.caseId: oracle
        for oracle in oracle_records
        if oracle.scenarioProvisioning == "TRACE_ORACLE_READY"
    }
    results: list[dict[str, Any]] = []
    for case_id, file_name in selected_files.items():
        task = tasks.get(case_id)
        oracle = oracles.get(case_id)
        if task is None or oracle is None:
            raise ValueError("TRACE_AUDIT_ORACLE_NOT_FOUND")
        trace = _read_trace(trace_dir / file_name)
        observation = closed_observation_for_trace(task, trace)
        verification = verify_result_oracle(oracle, observation)
        results.append({
            "caseId": case_id,
            "traceFile": file_name,
            "status": verification.status,
            "resultVerifierCode": verification.resultVerifierCode,
            "failureCodes": verification.failureCodes,
        })
    return {
        "schemaVersion": "p2j4-trace-oracle-adapter-audit-v1",
        "mode": "offline_redacted_trace_audit",
        "externalCalls": False,
        "caseCount": len(results),
        "passCount": sum(item["status"] == "PASS" for item in results),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit seven redacted trace-oracle cases")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = audit_existing_traces()
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(serialized, encoding="utf-8", newline="\n")
        print(json.dumps({"output": str(args.output), "status": "COMPLETED"}, ensure_ascii=False))
    else:
        print(serialized, end="")
    return 0 if payload["passCount"] == payload["caseCount"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
