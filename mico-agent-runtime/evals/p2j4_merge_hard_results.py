"""Merge Hard Eval results without rerunning successful cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mico_agent_runtime.contracts.trace_eval import EvalScore, TraceProjection
from mico_agent_runtime.runtime.trace_eval import build_stability_baseline


def _read(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("HARD_MERGE_PAYLOAD_INVALID")
    return payload


def merge(
    original: dict,
    repairs: list[dict],
    *,
    source_names: list[str],
    accepted_case_ids: set[str] | None = None,
) -> dict:
    traces = {item["traceId"]: TraceProjection.model_validate(item) for item in original.get("traces", [])}
    scores = {item["traceId"]: EvalScore.model_validate(item) for item in original.get("scores", [])}
    repaired_case_ids: list[str] = []
    repair_attempts: list[dict[str, object]] = []
    for payload, source_name in zip(repairs, source_names, strict=True):
        payload_scores = [EvalScore.model_validate(item) for item in payload.get("scores", [])]
        payload_traces = [TraceProjection.model_validate(item) for item in payload.get("traces", [])]
        if len(payload_scores) != len(payload_traces):
            raise ValueError("HARD_MERGE_REPAIR_TRACE_SCORE_MISMATCH")
        for score in payload_scores:
            if accepted_case_ids is not None and score.caseId not in accepted_case_ids:
                continue
            if score.status != "PASS":
                # A repair payload may contain an earlier failed attempt for
                # the same case. Only a passing accepted attempt can replace
                # the original trace; later repair payloads may supply it.
                continue
            trace = next(item for item in payload_traces if item.traceId == score.traceId)
            repaired_case_ids.append(score.caseId)
            traces[trace.traceId] = trace
            scores[score.traceId] = score
            repair_attempts.append({"caseId": score.caseId, "source": source_name})
    if accepted_case_ids is not None and not accepted_case_ids.issubset(set(repaired_case_ids)):
        missing = sorted(accepted_case_ids - set(repaired_case_ids))
        raise ValueError("HARD_MERGE_ACCEPTED_CASE_NOT_PASS:" + ",".join(missing))

    selected_case_ids = list(original.get("selectedCaseIds", []))
    score_by_case = {score.caseId: score for score in scores.values()}
    trace_by_id = {trace.traceId: trace for trace in traces.values()}
    ordered_scores = [score_by_case[case_id] for case_id in selected_case_ids]
    ordered_traces = [trace_by_id[score_by_case[case_id].traceId] for case_id in selected_case_ids]
    bad_cases = [item for item in original.get("badCases", []) if item.get("caseId") not in set(repaired_case_ids)]
    baseline = build_stability_baseline(
        original["schemaVersion"], ordered_scores, ordered_traces
    )
    result = dict(original)
    result.update({
        "checkpointVersion": "p2j4-hard-eval-merged-v1",
        "mode": "hard_eval_merged",
        "status": "COMPLETED",
        "completedCaseIds": selected_case_ids,
        "remainingCaseIds": [],
        "realRunsExecuted": len(ordered_scores),
        "baseline": baseline.model_dump(mode="json"),
        "traces": [trace.model_dump(mode="json") for trace in ordered_traces],
        "scores": [score.model_dump(mode="json") for score in ordered_scores],
        "badCases": bad_cases,
        "mergeAudit": {
            "preservedSuccessfulCaseCount": sum(1 for score in original.get("scores", []) if score.get("status") == "PASS"),
            "repairedRerunCaseCount": len(set(repaired_case_ids)),
            "successfulCasesWereNotRerun": True,
            "sourceOriginal": "p2j4-hard-real-20260824.json",
            "sourceRepairs": source_names,
            "repairAttempts": repair_attempts,
        },
    })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge P2-J4 Hard Eval results")
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--repair", type=Path, action="append", required=True)
    parser.add_argument("--accept-case", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repairs = [_read(path) for path in args.repair]
    result = merge(
        _read(args.original),
        repairs,
        source_names=[path.name for path in args.repair],
        accepted_case_ids=set(args.accept_case) if args.accept_case else None,
    )
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "caseCount": result["caseCount"],
        "preservedSuccessfulCaseCount": result["mergeAudit"]["preservedSuccessfulCaseCount"],
        "repairedRerunCaseCount": result["mergeAudit"]["repairedRerunCaseCount"],
        "output": str(args.output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
