"""Merge the completed Base policy-sensitive Runtime run without reruns."""

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
REMAINING = ROOT / "p2j4-policy-sensitive-runtime-v1" / "base-remaining-19.json"
RETRY = ROOT / "p2j4-policy-sensitive-runtime-v1" / "base-retry-canary-compare-01.json"
DEFAULT_OUTPUT = ROOT / "p2j4-policy-sensitive-runtime-v1" / "base-20-policy-sensitive-aggregate.json"
RETRY_CASE_ID = "p2j4-policy-sensitive-compare-01"


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"payload is not an object: {path}")
    return payload


def build_aggregate() -> dict[str, Any]:
    validation = validate_task_set(TASK_SET)
    expected_ids = [task.caseId for task in validation["tasks"]]
    remaining = _load(REMAINING)
    retry = _load(RETRY)
    score_by_case: dict[str, EvalScore] = {}
    trace_by_case: dict[str, TraceProjection] = {}

    for index, raw_score in enumerate(remaining.get("scores", [])):
        score = EvalScore.model_validate(raw_score)
        if score.caseId in score_by_case:
            raise ValueError(f"duplicate case: {score.caseId}")
        score_by_case[score.caseId] = score
        trace_by_case[score.caseId] = TraceProjection.model_validate(remaining["traces"][index])

    retry_scores = retry.get("scores", [])
    if len(retry_scores) != 1:
        raise ValueError("Base retry payload must contain exactly one score")
    retry_score = EvalScore.model_validate(retry_scores[0])
    if retry_score.caseId != RETRY_CASE_ID:
        raise ValueError("Base retry case ID mismatch")
    if RETRY_CASE_ID in score_by_case:
        raise ValueError("Base retry case already appears in remaining payload")
    score_by_case[RETRY_CASE_ID] = retry_score
    trace_by_case[RETRY_CASE_ID] = TraceProjection.model_validate(retry["traces"][0])

    if set(score_by_case) != set(expected_ids) or set(trace_by_case) != set(expected_ids):
        raise ValueError("Base aggregate IDs do not exactly match the frozen task set")

    ordered_scores = [score_by_case[case_id] for case_id in expected_ids]
    ordered_traces = [trace_by_case[case_id] for case_id in expected_ids]
    baseline = build_stability_baseline(
        validation["schemaVersion"], ordered_scores, ordered_traces
    )
    services = dict(remaining.get("servicesObserved", {}))
    services.update(retry.get("servicesObserved", {}))
    return {
        "schemaVersion": validation["schemaVersion"],
        "checkpointVersion": "p2j4-real-run-checkpoint-v2",
        "mode": "real_run_aggregate",
        "agentArm": "base",
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
        "badCases": list(remaining.get("badCases", [])) + list(retry.get("badCases", [])),
        "resultOracleVerifications": list(remaining.get("resultOracleVerifications", [])),
        "audit": {
            "primaryCaseCount": len(remaining.get("scores", [])),
            "acceptedRetryCase": RETRY_CASE_ID,
            "excludedBalanceFailureArtifact": True,
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
