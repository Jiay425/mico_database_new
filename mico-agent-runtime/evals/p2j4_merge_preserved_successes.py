"""Merge the preserved successful additive traces with repaired reruns.

This is intentionally a one-way audit artifact builder.  It never reruns a
case and refuses to overwrite a successful original case with a new trace.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mico_agent_runtime.contracts.trace_eval import EvalScore, TraceProjection
from evals.p2j4_runner import build_stability_baseline


ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT / "p2j4-gemini-expanded-50-20260824.json"
RERUNS = {
    "p2j4-data-fact-011": ROOT / "p2j4-rerun-data-fact-011-20260824-v3.json",
    "p2j4-data-fact-016": ROOT / "p2j4-rerun-data-fact-016-20260824.json",
    "p2j4-data-fact-019": ROOT / "p2j4-rerun-data-fact-019-20260824-v2.json",
    "p2j4-focused-analysis-016": ROOT / "p2j4-rerun-focused-analysis-016-20260824.json",
    "p2j4-focused-analysis-019": ROOT / "p2j4-rerun-focused-analysis-019-20260824.json",
    "p2j4-focused-analysis-027": ROOT / "p2j4-rerun-focused-analysis-027-20260824-v2.json",
    "p2j4-focused-analysis-029": ROOT / "p2j4-rerun-focused-analysis-029-20260824-v2.json",
    "p2j4-open-exploration-025": ROOT / "p2j4-rerun-open-exploration-025-20260824.json",
    "p2j4-open-exploration-033": ROOT / "p2j4-rerun-open-exploration-033-20260824.json",
    "p2j4-open-exploration-034": ROOT / "p2j4-rerun-open-exploration-034-20260824.json",
    "p2j4-open-exploration-035": ROOT / "p2j4-rerun-open-exploration-035-20260824.json",
    "p2j4-open-exploration-036": ROOT / "p2j4-rerun-open-exploration-036-20260824.json",
    "p2j4-open-exploration-037": ROOT / "p2j4-rerun-open-exploration-037-20260824-v3.json",
    "p2j4-open-exploration-038": ROOT / "p2j4-rerun-open-exploration-038-20260824-v2.json",
    "p2j4-open-exploration-039": ROOT / "p2j4-rerun-open-exploration-039-20260824-reviewed-rescore.json",
    "p2j4-open-exploration-040": ROOT / "p2j4-rerun-open-exploration-040-20260824.json",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _by_case(items: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in items:
        case_id = str(item.get("caseId", ""))
        if not case_id or case_id in result:
            raise ValueError(f"duplicate or missing {field} caseId")
        result[case_id] = item
    return result


def _traces_by_case(
    scores: dict[str, dict[str, Any]], traces: list[dict[str, Any]], label: str
) -> dict[str, dict[str, Any]]:
    by_trace_id = {str(item.get("traceId", "")): item for item in traces}
    if len(by_trace_id) != len(traces) or "" in by_trace_id:
        raise ValueError(f"duplicate or missing {label} traceId")
    result: dict[str, dict[str, Any]] = {}
    for case_id, score in scores.items():
        trace = by_trace_id.get(str(score.get("traceId", "")))
        if trace is None or case_id in result:
            raise ValueError(f"{label} trace does not match score")
        result[case_id] = trace
    return result


def build_merged_payload(output: Path) -> dict[str, Any]:
    original = _load(ORIGINAL)
    original_scores = _by_case(original.get("scores", []), "original score")
    original_traces = _traces_by_case(
        original_scores, original.get("traces", []), "original"
    )
    preserved_ids = [
        case_id for case_id in original.get("selectedCaseIds", [])
        if original_scores.get(case_id, {}).get("status") == "PASS"
    ]
    if len(preserved_ids) != 34:
        raise ValueError(f"expected 34 preserved successes, found {len(preserved_ids)}")

    rerun_scores: dict[str, dict[str, Any]] = {}
    rerun_traces: dict[str, dict[str, Any]] = {}
    rerun_bad_cases: list[dict[str, Any]] = []
    rerun_verifications: list[dict[str, Any]] = []
    for case_id, path in RERUNS.items():
        payload = _load(path)
        score_items = _by_case(payload.get("scores", []), f"{case_id} score")
        trace_items = _traces_by_case(
            score_items, payload.get("traces", []), f"{case_id}"
        )
        if set(score_items) != {case_id} or set(trace_items) != {case_id}:
            raise ValueError(f"rerun payload does not contain exactly {case_id}")
        if score_items[case_id].get("status") != "PASS":
            raise ValueError(f"rerun case is not PASS: {case_id}")
        if case_id in preserved_ids:
            raise ValueError(f"successful original case was scheduled for replacement: {case_id}")
        rerun_scores[case_id] = score_items[case_id]
        rerun_traces[case_id] = trace_items[case_id]
        rerun_bad_cases.extend(payload.get("badCases", []))
        rerun_verifications.extend(payload.get("resultOracleVerifications", []))

    selected_ids = list(original.get("selectedCaseIds", []))
    if len(selected_ids) != 50 or set(selected_ids) != set(preserved_ids) | set(RERUNS):
        raise ValueError("preserved and rerun case IDs do not form the executed 50-case set")
    if set(preserved_ids) & set(RERUNS):
        raise ValueError("preserved/rerun case sets overlap")

    scores = [
        original_scores[case_id] if case_id in preserved_ids else rerun_scores[case_id]
        for case_id in selected_ids
    ]
    traces = [
        original_traces[case_id] if case_id in preserved_ids else rerun_traces[case_id]
        for case_id in selected_ids
    ]
    if any(item.get("status") != "PASS" for item in scores):
        raise ValueError("merged score set contains a non-PASS case")
    if len({item["traceId"] for item in traces}) != 50:
        raise ValueError("merged trace IDs are not unique")

    trace_models = [TraceProjection.model_validate(item) for item in traces]
    score_models = [EvalScore.model_validate(item) for item in scores]
    baseline = build_stability_baseline(original["schemaVersion"], score_models, trace_models)
    routes = sorted({route for trace in traces for route in trace.get("sourceRoutes", [])})
    output_payload: dict[str, Any] = {
        "schemaVersion": original["schemaVersion"],
        "checkpointVersion": "p2j4-merged-real-run-v1",
        "mode": "real_run_merged",
        "status": "COMPLETED",
        "caseCount": 50,
        "selectedCaseIds": selected_ids,
        "completedCaseIds": selected_ids,
        "remainingCaseIds": [],
        "realRunsExecuted": 50,
        "externalCalls": True,
        "servicesObserved": {
            "java": "java" in routes,
            "pgvector": "vector" in routes,
            "neo4j": "graph" in routes,
            "gemini": True,
            "plannerModelUsed": True,
            "plannerProvider": "mixed_preserved_deepseek_and_rerun_gemini",
            "plannerModel": "gemini-3.5-flash",
            "plannerModels": ["deepseek-v4-flash", "gemini-3.5-flash"],
            "mixedProvider": True,
            "preservedSuccessfulCount": 34,
            "rerunCount": 16,
        },
        "baseline": baseline.model_dump(mode="json"),
        "traces": traces,
        "scores": scores,
        "badCases": rerun_bad_cases,
        "resultOracleVerifications": rerun_verifications,
        "mergeAudit": {
            "preservedSuccessfulCaseCount": 34,
            "repairedRerunCaseCount": 16,
            "successfulCasesWereNotRerun": True,
            "sourceOriginal": ORIGINAL.name,
            "sourceReruns": [path.name for path in RERUNS.values()],
        },
    }
    output.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "p2j4-gemini-expanded-50-final-20260824.json",
    )
    args = parser.parse_args()
    payload = build_merged_payload(args.output)
    print(json.dumps({
        "status": payload["status"],
        "output": str(args.output),
        "caseCount": payload["caseCount"],
        "passCount": sum(item["status"] == "PASS" for item in payload["scores"]),
        "preservedSuccessfulCount": payload["mergeAudit"]["preservedSuccessfulCaseCount"],
        "rerunCount": payload["mergeAudit"]["repairedRerunCaseCount"],
    }, ensure_ascii=False))
