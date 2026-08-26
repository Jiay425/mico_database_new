"""Rescore an immutable runner payload against a reviewed task asset.

No model, Java service, database, or network call is made.  The original
runner payload remains the source-of-record; this produces a separate audit
view when a task/oracle contract is corrected offline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evals.p2j4_runner import load_task_set
from mico_agent_runtime.contracts.trace_eval import TraceProjection
from mico_agent_runtime.runtime.trace_eval import build_bad_cases, score_trace


def rescore(input_path: Path, task_set_path: Path) -> dict:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    tasks = {task.caseId: task for task in load_task_set(task_set_path)}
    traces = [TraceProjection.model_validate(item) for item in payload.get("traces", [])]
    scores = []
    bad_cases = []
    for trace in traces:
        task = tasks.get(trace.taskId.replace("task-", "p2j4-", 1))
        if task is None:
            # The runner task ID is opaque; match through the original score's
            # trace ID instead when task IDs are not case IDs.
            original = next(
                (item for item in payload.get("scores", []) if item.get("traceId") == trace.traceId),
                None,
            )
            case_id = original.get("caseId") if original else None
            task = tasks.get(case_id)
        if task is None:
            raise ValueError("OFFLINE_RESCORE_TASK_MISSING")
        score = score_trace(task, trace)
        scores.append(score)
        bad_cases.extend(build_bad_cases(task, score))
    return {
        "schemaVersion": payload.get("schemaVersion"),
        "mode": "offline_rescore",
        "status": "COMPLETED",
        "sourceArtifact": input_path.name,
        # Preserve non-sensitive runner provenance so a subsequent merge can
        # prove which runtime/planner produced the immutable traces.
        "sourceMode": payload.get("mode"),
        "checkpointVersion": payload.get("checkpointVersion"),
        "selectedCaseIds": payload.get("selectedCaseIds"),
        "servicesObserved": payload.get("servicesObserved", {}),
        "scoringTaskSet": task_set_path.name,
        "externalCalls": False,
        "trainingStarted": False,
        "caseCount": len(scores),
        "traces": [item.model_dump(mode="json") for item in traces],
        "scores": [item.model_dump(mode="json") for item in scores],
        "badCases": [item.model_dump(mode="json") for item in bad_cases],
        "summary": {
            "pass": sum(item.status == "PASS" for item in scores),
            "fail": sum(item.status == "FAIL" for item in scores),
            "badCaseRecords": len(bad_cases),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline P2-J4 trace rescorer")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--task-set", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = rescore(args.input, args.task_set)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "READY", **result["summary"], "externalCalls": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
