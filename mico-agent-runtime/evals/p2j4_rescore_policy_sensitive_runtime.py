"""Offline re-score for the policy-sensitive Runtime aggregates.

The original aggregate was built from planner decision records.  This audit
preserves those records for SFT analysis, derives the actual ``execute_action``
path from the retained events, and scores the task oracle against that path.
No provider, database, SSH connection, or network call is used.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from mico_agent_runtime.contracts.trace_eval import EvalScore, TraceProjection
from mico_agent_runtime.runtime.trace_eval import (
    _canonical_actions_for_task,
    build_stability_baseline,
    score_trace,
)

from evals.p2j4_runner import validate_task_set


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"payload is not an object: {path}")
    return payload


def _upgrade_trace(raw: dict[str, Any]) -> TraceProjection:
    raw = dict(raw)
    if raw.get("executionActionBindingVersion") == "execution-action-v1":
        executed = list(raw.get("executedActions", []))
        counts = Counter(executed)
        raw["duplicateActionCount"] = sum(max(0, count - 1) for count in counts.values())
        raw["loopCount"] = sum(left == right for left, right in zip(executed, executed[1:]))
        raw["actionCount"] = max(int(raw.get("actionCount", 0) or 0), len(executed))
    else:
        # Do not manufacture an execution path from a legacy shared tool name.
        # score_trace will use explicit planner decisions for this artifact.
        raw["executedActions"] = []
    return TraceProjection.model_validate(raw)


def rescore(input_path: Path, output_path: Path) -> dict[str, Any]:
    payload = _load(input_path)
    task_path = input_path.parent / "task-set.json"
    validation = validate_task_set(task_path)
    tasks = {task.caseId: task for task in validation["tasks"]}
    raw_traces = payload.get("traces", [])
    if not isinstance(raw_traces, list):
        raise ValueError("aggregate traces must be a list")

    traces: list[TraceProjection] = []
    scores: list[EvalScore] = []
    audits: list[dict[str, Any]] = []
    previous_scores = {
        score["caseId"]: score
        for score in payload.get("scores", [])
        if isinstance(score, dict) and isinstance(score.get("caseId"), str)
    }
    case_by_trace_id = {
        score.get("traceId"): score.get("caseId")
        for score in payload.get("scores", [])
        if isinstance(score, dict)
        and isinstance(score.get("traceId"), str)
        and isinstance(score.get("caseId"), str)
    }
    for index, raw in enumerate(raw_traces):
        trace = _upgrade_trace(raw)
        case_id = case_by_trace_id.get(trace.traceId)
        if case_id is None and index < len(validation["tasks"]):
            # The aggregate writer orders traces by the frozen task set.  This
            # fallback keeps the audit usable for an older artifact that did
            # not retain a stable traceId-to-case mapping.
            case_id = validation["tasks"][index].caseId
        task = tasks.get(case_id)
        if task is None:
            raise ValueError(f"trace has no frozen case mapping: {trace.traceId}")
        score = score_trace(task, trace)
        traces.append(trace)
        scores.append(score)
        decision_path = [decision.chosenAction for decision in trace.decisions]
        execution_bound = trace.executionActionBindingVersion == "execution-action-v1"
        executed_path = list(trace.executedActions) if execution_bound else []
        canonical_path = _canonical_actions_for_task(task, executed_path)
        execute_event_count = sum(
            1 for event in trace.events if event.node == "execute_action"
        )
        cardinality_matches = execute_event_count == len(decision_path)
        audits.append({
            "caseId": task.caseId,
            "traceId": trace.traceId,
            "decisionPath": decision_path,
            "executedPath": executed_path,
            "canonicalExecutedPath": canonical_path,
            "pathAgreement": decision_path == canonical_path if execution_bound else None,
            "executionPathConfidence": (
                "observed" if execution_bound
                else "legacy-cardinality-inferred" if cardinality_matches
                else "legacy-unavailable"
            ),
            "executeActionEventCount": execute_event_count,
            "decisionCount": len(decision_path),
            "cardinalityMatches": cardinality_matches,
            "inferredExecutedPath": decision_path if not execution_bound and cardinality_matches else [],
            "previousStatus": previous_scores.get(task.caseId, {}).get("status"),
            "correctedStatus": score.status,
            "correctedFailureCodes": score.failureCodes,
        })

    baseline = build_stability_baseline(validation["schemaVersion"], scores, traces)
    observed_path_agreements = [
        item["pathAgreement"] for item in audits if item["pathAgreement"] is not None
    ]
    payload_out = {
        "schemaVersion": validation["schemaVersion"],
        "checkpointVersion": "p2j4-real-run-corrected-oracle-v1",
        "mode": "offline_corrected_oracle_rescore",
        "status": "COMPLETED",
        "caseCount": len(scores),
        "selectedCaseIds": [task.caseId for task in validation["tasks"]],
        "completedCaseIds": [task.caseId for task in validation["tasks"] if task.caseId in {score.caseId for score in scores}],
        "realRunsExecuted": 0,
        "externalCalls": False,
        "servicesObserved": payload.get("servicesObserved", {}),
        "baseline": baseline.model_dump(mode="json"),
        "traces": [trace.model_dump(mode="json") for trace in traces],
        "scores": [score.model_dump(mode="json") for score in scores],
        "badCases": payload.get("badCases", []),
        "resultOracleVerifications": payload.get("resultOracleVerifications", []),
        "audit": {
            "sourceArtifact": str(input_path),
            "networkCallsDuringRescore": False,
            "decisionPathAndExecutionPathAreSeparate": True,
            "pathAgreementCount": sum(observed_path_agreements),
            "pathAgreementRate": (
                sum(observed_path_agreements) / len(observed_path_agreements)
                if observed_path_agreements else None
            ),
            "observedExecutionBindingCaseCount": len(observed_path_agreements),
            "legacyCardinalityMatchCount": sum(item["cardinalityMatches"] for item in audits),
            "caseAudits": audits,
        },
    }
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload_out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = rescore(args.input, args.output)
    print(json.dumps({
        "output": str(args.output),
        "caseCount": payload["caseCount"],
        "passCount": payload["baseline"]["passCount"],
        "pathAgreementCount": payload["audit"]["pathAgreementCount"],
        "legacyCardinalityMatchCount": payload["audit"]["legacyCardinalityMatchCount"],
        "networkCallsDuringRescore": payload["audit"]["networkCallsDuringRescore"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
