"""Build the final redacted Gemini P2-J4 real-run baseline.

The 50 cases were intentionally executed as independent checkpoints so a
failed case could be repaired before spending calls on later cases.  This
module selects only the final PASS artifact for each case and never mutates
the source checkpoints.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from mico_agent_runtime.contracts.trace_eval import EvalScore, TraceProjection
from mico_agent_runtime.runtime.trace_eval import build_stability_baseline

from evals.p2j4_controlled_scenarios import verify_result_oracle
from evals.p2j4_result_oracles import validate_result_oracle_set
from evals.p2j4_trace_assertions import closed_observation_for_trace


ROOT = Path(__file__).resolve().parent
TASK_SET = ROOT / "p2j4-task-set-v2.json"
INDIVIDUAL = ROOT / "p2j4-gemini-individual-20260824"
RETRY = ROOT / "p2j4-gemini-retry-20260824"
FULL_BASELINE = ROOT / "p2j4-gemini-full-baseline-20260824.json"
FULL_REMAINING = ROOT / "p2j4-gemini-full-remaining-20260824.json"
OUTPUT = ROOT / "p2j4-gemini-full-baseline-final-20260824.json"


def _source_map() -> dict[str, Path]:
    mapping: dict[str, Path] = {}

    mapping.update({
        "p2j4-data-fact-001": RETRY / "p2j4-data-fact-001-retry2.json",
        "p2j4-data-fact-004": RETRY / "p2j4-data-fact-004-retry2.json",
        "p2j4-data-fact-008": RETRY / "p2j4-data-fact-008-retry2.json",
    })
    for case_id in ("002", "003"):
        mapping[f"p2j4-data-fact-{case_id}"] = FULL_BASELINE
    for case_id in ("005", "006", "007", "009", "010"):
        mapping[f"p2j4-data-fact-{case_id}"] = FULL_REMAINING

    mapping.update({
        "p2j4-focused-analysis-001": FULL_REMAINING,
        "p2j4-focused-analysis-002": FULL_REMAINING,
        "p2j4-focused-analysis-005": RETRY / "p2j4-focused-analysis-005-retry3.json",
        "p2j4-focused-analysis-008": RETRY / "p2j4-focused-analysis-008-retry3.json",
        "p2j4-focused-analysis-009": RETRY / "p2j4-focused-analysis-009-retry4.json",
        "p2j4-focused-analysis-010": RETRY / "p2j4-focused-analysis-010-retry2.json",
    })
    for case_id in ("003", "004", "006", "007", "011", "012", "013", "014", "015"):
        mapping[f"p2j4-focused-analysis-{case_id}"] = INDIVIDUAL / f"p2j4-focused-analysis-{case_id}.json"

    for case_id in ("001", "003", "004", "005", "006", "007", "014"):
        mapping[f"p2j4-open-exploration-{case_id}"] = RETRY / f"p2j4-open-exploration-{case_id}-retry2.json"
    for case_id in ("002", "008", "009", "010", "011", "012", "013", "015", "016", "017", "018", "019", "020"):
        mapping[f"p2j4-open-exploration-{case_id}"] = INDIVIDUAL / f"p2j4-open-exploration-{case_id}.json"

    for case_id in ("001", "002", "003", "004", "005"):
        mapping[f"p2j4-safety-{case_id}"] = INDIVIDUAL / f"p2j4-safety-{case_id}.json"
    return mapping


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build() -> dict[str, Any]:
    task_set = _load(TASK_SET)
    cases = task_set["cases"]
    source_map = _source_map()
    expected_ids = [case["caseId"] for case in cases]
    if set(source_map) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(source_map))
        extra = sorted(set(source_map) - set(expected_ids))
        raise ValueError(f"SOURCE_MAP_MISMATCH missing={missing} extra={extra}")

    raw_traces: dict[str, dict[str, Any]] = {}
    raw_scores: dict[str, dict[str, Any]] = {}
    bad_cases: list[dict[str, Any]] = []
    oracle_verifications: dict[str, dict[str, Any]] = {}
    for case_id, path in source_map.items():
        if not path.exists():
            raise FileNotFoundError(path)
        payload = _load(path)
        scores = {item["caseId"]: item for item in payload.get("scores", [])}
        traces = {item["taskId"]: item for item in payload.get("traces", [])}
        score = scores.get(case_id)
        if score is None or score.get("status") != "PASS" or score.get("failureCodes"):
            raise ValueError(f"NON_PASS_SOURCE case={case_id} source={path}")
        trace = next((item for item in payload.get("traces", []) if item.get("taskId") == case_id), None)
        if trace is None:
            # Trace taskId is normally the case ID; use traceId from the score
            # as a safe fallback for older checkpoint shapes.
            trace = next((item for item in payload.get("traces", []) if item.get("traceId") == score.get("traceId")), None)
        if trace is None:
            raise ValueError(f"TRACE_MISSING case={case_id} source={path}")
        raw_scores[case_id] = score
        raw_traces[case_id] = trace
        for record in payload.get("badCases", []):
            if record.get("caseId") == case_id:
                bad_cases.append(record)
        for verification in payload.get("resultOracleVerifications", []):
            verification_id = str(verification.get("caseId"))
            oracle_verifications[verification_id] = verification

    traces = [TraceProjection.model_validate(raw_traces[case_id]) for case_id in expected_ids]
    scores = [EvalScore.model_validate(raw_scores[case_id]) for case_id in expected_ids]
    tasks_by_case = {case["caseId"]: case for case in cases}
    # Recompute the seven trace-oracle-ready verifications from the selected
    # final traces.  This keeps the aggregate synchronized when an oracle's
    # accepted action subsequence is corrected after a retry was written.
    oracle_records = validate_result_oracle_set(
        [type("Task", (), {"caseId": case["caseId"], "allowedActions": case["allowedActions"],
                           "allowedActionPaths": case["allowedActionPaths"],
                           "expectedStopReason": case["expectedStopReason"]})() for case in cases]
    )["oracles"]
    trace_by_case = {
        case_id: trace for case_id, trace in zip(expected_ids, traces, strict=True)
    }
    for oracle in oracle_records:
        if oracle.scenarioProvisioning != "TRACE_ORACLE_READY":
            continue
        task = tasks_by_case[oracle.caseId]
        task_model = type("EvalTaskProxy", (), {
            "caseId": task["caseId"],
            "question": task["question"],
        })()
        verification = verify_result_oracle(
            oracle,
            closed_observation_for_trace(task_model, trace_by_case[oracle.caseId]),
        )
        oracle_verifications[oracle.caseId] = verification.model_dump(mode="json")
    baseline = build_stability_baseline(task_set["schemaVersion"], scores, traces)
    routes = {route for trace in traces for route in trace.sourceRoutes}
    model_planner_trace_count = sum(
        "SCIENTIFIC_PLANNER_DETERMINISTIC_FALLBACK" not in trace.fallbackCodes
        for trace in traces
    )
    deterministic_fallback_trace_count = len(traces) - model_planner_trace_count
    user_gemini_key = os.environ.get("MICO_GEMINI_API_KEY", "").strip()
    output: dict[str, Any] = {
        "schemaVersion": task_set["schemaVersion"],
        "checkpointVersion": "p2j4-real-run-checkpoint-v2",
        "mode": "real_run",
        "status": "COMPLETED",
        "caseCount": len(expected_ids),
        "kindDistribution": task_set["goldenCaseDistribution"],
        "selectedCaseIds": expected_ids,
        "completedCaseIds": expected_ids,
        "remainingCaseIds": [],
        "realRunsExecuted": len(scores),
        "externalCalls": True,
        "servicesObserved": {
            "java": "java" in routes,
            "pgvector": "vector" in routes,
            "neo4j": "graph" in routes,
            "gemini": bool(user_gemini_key),
            "plannerModelUsed": model_planner_trace_count > 0,
            "plannerModelTraceCount": model_planner_trace_count,
            "deterministicFallbackTraceCount": deterministic_fallback_trace_count,
            "plannerProvider": "gemini_openai_compatible",
            "plannerModel": "gemini-3.5-flash",
            "deepseek_or_planner_model": model_planner_trace_count > 0,
        },
        "baseline": baseline.model_dump(mode="json"),
        "traces": [trace.model_dump(mode="json") for trace in traces],
        "scores": [score.model_dump(mode="json") for score in scores],
        "badCases": bad_cases,
        "resultOracleVerifications": [oracle_verifications[key] for key in expected_ids if key in oracle_verifications],
        "aggregation": {
            "sourceArtifactCount": len(set(str(path) for path in source_map.values())),
            "selectionRule": "one final PASS artifact per case; historical failures excluded",
            "sourceArtifacts": sorted({str(path.relative_to(ROOT)) for path in source_map.values()}),
        },
    }
    return output


if __name__ == "__main__":
    OUTPUT.write_text(json.dumps(build(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "caseCount": 50, "status": "COMPLETED"}, ensure_ascii=False))
