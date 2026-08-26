"""Merge the policy-sensitive SFT run without rerunning any case.

The first pass is deliberately retained as an audit artifact.  If a case is
recovered by an explicit retry, this script uses the retry only after checking
that every selected case is unique and PASS.  It performs no network calls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mico_agent_runtime.contracts.trace_eval import EvalScore, TraceProjection
from mico_agent_runtime.runtime.trace_eval import build_stability_baseline

from evals.p2j4_runner import validate_task_set


ROOT = Path(__file__).resolve().parent
TASK_SET = ROOT / "p2j4-policy-sensitive-runtime-v1" / "task-set.json"
CANARY = ROOT / "p2j4-policy-sensitive-runtime-v1" / "sft-canary-compare-01.json"
REMAINING = ROOT / "p2j4-policy-sensitive-runtime-v1" / "sft-remaining-19.json"
RETRY = ROOT / "p2j4-policy-sensitive-runtime-v1" / "sft-retry-retrieve-03.json"
DEFAULT_OUTPUT = ROOT / "p2j4-policy-sensitive-runtime-v1" / "sft-20-policy-sensitive-aggregate.json"
RETRY_CASE_ID = "p2j4-policy-sensitive-retrieve-03"


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"payload is not an object: {path}")
    return payload


def build_aggregate() -> dict[str, Any]:
    validation = validate_task_set(TASK_SET)
    expected_ids = [task.caseId for task in validation["tasks"]]

    canary = _load(CANARY)
    remaining = _load(REMAINING)
    retry = _load(RETRY)
    payload_by_case: dict[str, tuple[dict[str, Any], int]] = {}

    for payload in (canary, remaining):
        for index, raw_score in enumerate(payload.get("scores", [])):
            score = EvalScore.model_validate(raw_score)
            if score.caseId == RETRY_CASE_ID:
                continue
            if score.caseId in payload_by_case:
                raise ValueError(f"duplicate case in primary payloads: {score.caseId}")
            payload_by_case[score.caseId] = (payload, index)

    retry_scores = retry.get("scores", [])
    if len(retry_scores) != 1:
        raise ValueError("retry payload must contain exactly one score")
    retry_score = EvalScore.model_validate(retry_scores[0])
    if retry_score.caseId != RETRY_CASE_ID or retry_score.status != "PASS":
        raise ValueError("retry payload is not the accepted PASS for retrieve-03")
    if RETRY_CASE_ID in payload_by_case:
        raise ValueError("retry case was not excluded from primary payloads")

    score_by_case: dict[str, EvalScore] = {RETRY_CASE_ID: retry_score}
    trace_by_case: dict[str, TraceProjection] = {
        RETRY_CASE_ID: TraceProjection.model_validate(retry["traces"][0])
    }
    bad_cases: list[dict[str, Any]] = []
    oracle_verifications: list[dict[str, Any]] = []
    services: dict[str, Any] = {}

    for payload in (canary, remaining):
        services.update(payload.get("servicesObserved", {}))
        bad_cases.extend(payload.get("badCases", []))
        oracle_verifications.extend(payload.get("resultOracleVerifications", []))
    services.update(retry.get("servicesObserved", {}))
    bad_cases.extend(retry.get("badCases", []))
    oracle_verifications.extend(retry.get("resultOracleVerifications", []))

    for case_id, (payload, index) in payload_by_case.items():
        score = EvalScore.model_validate(payload["scores"][index])
        trace = TraceProjection.model_validate(payload["traces"][index])
        if score.status != "PASS":
            raise ValueError(f"primary result is not PASS: {case_id}")
        score_by_case[case_id] = score
        trace_by_case[case_id] = trace

    if set(score_by_case) != set(expected_ids) or set(trace_by_case) != set(expected_ids):
        raise ValueError("aggregate case IDs do not exactly match the frozen task set")

    ordered_scores = [score_by_case[case_id] for case_id in expected_ids]
    ordered_traces = [trace_by_case[case_id] for case_id in expected_ids]
    baseline = build_stability_baseline(
        validation["schemaVersion"], ordered_scores, ordered_traces
    )
    return {
        "schemaVersion": validation["schemaVersion"],
        "checkpointVersion": "p2j4-real-run-checkpoint-v2",
        "mode": "real_run_aggregate",
        "status": "COMPLETED",
        "caseCount": len(expected_ids),
        "selectedCaseIds": expected_ids,
        "completedCaseIds": expected_ids,
        "realRunsExecuted": len(expected_ids),
        "externalCalls": True,
        "servicesObserved": services,
        "baseline": baseline.model_dump(mode="json"),
        "traces": [trace.model_dump(mode="json") for trace in ordered_traces],
        "scores": [score.model_dump(mode="json") for score in ordered_scores],
        "badCases": bad_cases,
        "resultOracleVerifications": oracle_verifications,
        "audit": {
            "primaryPassCases": len(payload_by_case),
            "acceptedRetryCase": RETRY_CASE_ID,
            "excludedFailedPrimaryArtifact": True,
            "networkCallsDuringMerge": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    payload = build_aggregate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "caseCount": payload["caseCount"], "status": payload["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
