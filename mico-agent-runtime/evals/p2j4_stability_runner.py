"""Bounded, per-attempt stability runner for P2-J4.

No external call is made unless the caller supplies all of:
``--real``, explicit ``--case-id`` values, ``--repeats`` and
``MICO_P2J4_REAL_RUNS=true``.  Every attempt is written to its own checkpoint
through the existing atomic runner before the next attempt starts.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

from mico_agent_runtime.contracts.trace_eval import BadCaseRecord, EvalScore, TraceProjection
from mico_agent_runtime.runtime.trace_eval import build_stability_baseline

from evals import p2j4_runner


def _aggregate(
    *,
    validation: dict[str, Any],
    selected_case_ids: list[str],
    repeats: int,
    attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    traces = [TraceProjection.model_validate(item) for attempt in attempts for item in attempt.get("traces", [])]
    scores = [EvalScore.model_validate(item) for attempt in attempts for item in attempt.get("scores", [])]
    bad_cases = [BadCaseRecord.model_validate(item) for attempt in attempts for item in attempt.get("badCases", [])]
    return {
        "schemaVersion": validation["schemaVersion"],
        "stabilitySchemaVersion": "p2j4-stability-v1",
        "mode": "stability_real_run",
        "status": "COMPLETED" if len(attempts) == len(selected_case_ids) * repeats else "PARTIAL",
        "selectedCaseIds": selected_case_ids,
        "repeatCount": repeats,
        "attemptCount": len(attempts),
        "expectedAttemptCount": len(selected_case_ids) * repeats,
        "externalCalls": any(attempt.get("externalCalls", False) for attempt in attempts),
        "trainingStarted": False,
        "baseline": build_stability_baseline(
            validation["schemaVersion"], scores, traces
        ).model_dump(mode="json"),
        "attempts": [
            {
                "caseId": attempt["caseId"],
                "repeatIndex": attempt["repeatIndex"],
                "status": attempt.get("status"),
                "output": attempt.get("output"),
                "pass": attempt.get("scores", [{}])[0].get("status") == "PASS",
            }
            for attempt in attempts
        ],
        "traces": [trace.model_dump(mode="json") for trace in traces],
        "scores": [score.model_dump(mode="json") for score in scores],
        "badCases": [record.model_dump(mode="json") for record in bad_cases],
    }


def run_stability(
    case_ids: list[str],
    repeats: int,
    output_dir: Path,
    *,
    env: Mapping[str, str],
    real: bool,
    task_set: Path = p2j4_runner.TASK_SET,
) -> dict[str, Any]:
    if not real:
        raise ValueError("STABILITY_REAL_FLAG_REQUIRED")
    if not case_ids:
        raise ValueError("STABILITY_EXPLICIT_CASES_REQUIRED")
    if not 1 <= repeats <= 5:
        raise ValueError("STABILITY_REPEAT_COUNT_OUT_OF_RANGE")
    if env.get("MICO_P2J4_REAL_RUNS", "").strip().lower() != "true":
        raise ValueError("REAL_RUN_DISABLED")
    missing = p2j4_runner._configuration_missing(env)
    if missing:
        raise ValueError("REAL_RUN_CONFIGURATION_MISSING:" + ",".join(missing))

    validation = p2j4_runner.select_tasks(
        p2j4_runner.validate_task_set(task_set), case_ids
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    task_by_id = {task.caseId: task for task in validation["tasks"]}
    attempts: list[dict[str, Any]] = []
    for repeat_index in range(1, repeats + 1):
        for case_id in case_ids:
            output = output_dir / f"{case_id}-repeat-{repeat_index:02d}.json"
            single_validation = p2j4_runner.select_tasks(validation, [case_id])
            payload = p2j4_runner._real_run(
                [task_by_id[case_id]],
                single_validation,
                env,
                output=output,
                resume=output.exists(),
            )
            attempts.append({
                "caseId": case_id,
                "repeatIndex": repeat_index,
                "status": payload.get("status"),
                "output": str(output),
                "externalCalls": payload.get("externalCalls", False),
                "traces": payload.get("traces", []),
                "scores": payload.get("scores", []),
                "badCases": payload.get("badCases", []),
            })
    return _aggregate(
        validation=validation,
        selected_case_ids=case_ids,
        repeats=repeats,
        attempts=attempts,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run bounded repeated P2-J4 stability cases")
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--case-id", action="append", dest="case_ids", required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-set", type=Path, default=p2j4_runner.TASK_SET)
    args = parser.parse_args(argv)
    try:
        payload = run_stability(
            list(dict.fromkeys(args.case_ids)),
            args.repeats,
            args.output_dir,
            env=os.environ,
            real=args.real,
            task_set=args.task_set,
        )
        p2j4_runner._atomic_write_payload(payload, args.output)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "INVALID", "errorCode": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({
        "status": payload["status"],
        "attemptCount": payload["attemptCount"],
        "expectedAttemptCount": payload["expectedAttemptCount"],
        "externalCalls": payload["externalCalls"],
        "trainingStarted": False,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
