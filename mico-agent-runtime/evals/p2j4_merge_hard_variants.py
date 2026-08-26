"""Merge Hard Variant runs while preserving every successful first attempt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mico_agent_runtime.contracts.trace_eval import EvalScore, TraceProjection
from mico_agent_runtime.runtime.trace_eval import build_stability_baseline


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("HARD_VARIANT_MERGE_PAYLOAD_INVALID")
    return payload


def merge(
    task_set: dict[str, Any],
    payloads: list[dict[str, Any]],
    source_names: list[str],
    *,
    forced_case_ids: set[str] | None = None,
) -> dict[str, Any]:
    case_ids = [item["caseId"] for item in task_set.get("cases", [])]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("HARD_VARIANT_MERGE_DUPLICATE_TASK_CASE")

    selected_scores: dict[str, EvalScore] = {}
    selected_traces: dict[str, TraceProjection] = {}
    first_pass_cases: set[str] = set()
    failed_before_repair: set[str] = set()
    forced_case_ids = forced_case_ids or set()
    attempts_by_case: dict[str, list[dict[str, Any]]] = {}
    all_bad_cases: list[dict[str, Any]] = []
    services_observed: dict[str, Any] = {}
    external_calls = False

    for payload, source_name in zip(payloads, source_names, strict=True):
        external_calls = external_calls or bool(payload.get("externalCalls"))
        for key, value in (payload.get("servicesObserved") or {}).items():
            if isinstance(value, bool):
                services_observed[key] = bool(services_observed.get(key, False) or value)
            elif value not in (None, ""):
                services_observed[key] = value

        traces = {
            trace.traceId: trace
            for trace in (TraceProjection.model_validate(item) for item in payload.get("traces", []))
        }
        for raw_score in payload.get("scores", []):
            score = EvalScore.model_validate(raw_score)
            attempts_by_case.setdefault(score.caseId, []).append({
                "source": source_name,
                "status": score.status,
                "traceId": score.traceId,
            })
            if score.status != "PASS":
                failed_before_repair.add(score.caseId)
                if score.caseId not in selected_scores:
                    selected_scores[score.caseId] = score
                    selected_traces[score.caseId] = traces[score.traceId]
                continue
            trace = traces.get(score.traceId)
            if trace is None:
                raise ValueError("HARD_VARIANT_MERGE_TRACE_MISSING:" + score.caseId)
            if score.caseId in first_pass_cases and score.caseId not in forced_case_ids:
                # A passing case is immutable after its first successful
                # attempt. This protects the no-rerun boundary explicitly.
                continue
            selected_scores[score.caseId] = score
            selected_traces[score.caseId] = trace
            first_pass_cases.add(score.caseId)

        all_bad_cases.extend(payload.get("badCases", []))

    missing = [case_id for case_id in case_ids if case_id not in first_pass_cases]
    if missing:
        raise ValueError("HARD_VARIANT_MERGE_CASE_NOT_PASS:" + ",".join(missing))

    ordered_scores = [selected_scores[case_id] for case_id in case_ids]
    ordered_traces = [selected_traces[case_id] for case_id in case_ids]
    baseline = build_stability_baseline(
        task_set["schemaVersion"], ordered_scores, ordered_traces
    )
    rerun_case_ids = sorted(
        (failed_before_repair | forced_case_ids) & first_pass_cases
    )
    first_pass_preserved = sorted(first_pass_cases - set(rerun_case_ids))
    attempt_count = sum(len(attempts) for attempts in attempts_by_case.values())
    final_pass_cases = set(first_pass_cases)
    final_bad_cases = [
        item for item in all_bad_cases
        if item.get("caseId") not in final_pass_cases
    ]
    result = {
        "schemaVersion": task_set["schemaVersion"],
        "name": task_set.get("name"),
        "reviewStatus": "REAL_HARD_VARIANT_COMPLETE",
        "sourceTaskSet": task_set.get("sourceTaskSet"),
        "caseCount": len(case_ids),
        "hardCaseDistribution": task_set.get("hardCaseDistribution", {}),
        "kindDistribution": task_set.get("kindDistribution", {}),
        "checkpointVersion": "p2j4-hard-variant-merged-v1",
        "mode": "hard_eval_merged",
        "status": "COMPLETED",
        "selectedCaseIds": case_ids,
        "completedCaseIds": case_ids,
        "remainingCaseIds": [],
        "realRunsExecuted": attempt_count,
        "externalCalls": external_calls,
        "servicesObserved": services_observed,
        "baseline": baseline.model_dump(mode="json"),
        "traces": [trace.model_dump(mode="json") for trace in ordered_traces],
        "scores": [score.model_dump(mode="json") for score in ordered_scores],
        "badCases": final_bad_cases,
        "mergeAudit": {
            "preservedSuccessfulCaseCount": len(first_pass_preserved),
            "repairedRerunCaseCount": len(rerun_case_ids),
            "successfulCasesWereNotRerun": True,
            "firstPassPreservedCaseIds": first_pass_preserved,
            "rerunCaseIds": rerun_case_ids,
            "attemptCount": attempt_count,
            "sourceRuns": source_names,
            "attemptsByCase": attempts_by_case,
        },
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge P2-J4 Hard Variant runs")
    parser.add_argument("--task-set", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--force-case", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        task_set = _read(args.task_set)
        payloads = [_read(path) for path in args.input]
        result = merge(
            task_set,
            payloads,
            [path.name for path in args.input],
            forced_case_ids=set(args.force_case),
        )
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "INVALID", "errorCode": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({
        "status": result["status"],
        "caseCount": result["caseCount"],
        "passCount": result["baseline"]["passCount"],
        "preservedSuccessfulCaseCount": result["mergeAudit"]["preservedSuccessfulCaseCount"],
        "repairedRerunCaseCount": result["mergeAudit"]["repairedRerunCaseCount"],
        "attemptCount": result["mergeAudit"]["attemptCount"],
        "output": str(args.output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
