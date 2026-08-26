"""Merge additive state-difference runs without rerunning successful cases."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from evals.p2j4_runner import _build_real_payload, load_task_set
from mico_agent_runtime.contracts.trace_eval import TraceProjection
from mico_agent_runtime.runtime.trace_eval import score_trace


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("mode") != "real_run":
        raise ValueError(f"STATE_DIFF_RUN_INVALID:{path}")
    return payload


def _index(payload: dict[str, Any]) -> dict[str, TraceProjection]:
    scores = payload.get("scores", [])
    traces = {
        item["traceId"]: TraceProjection.model_validate(item)
        for item in payload.get("traces", [])
    }
    result: dict[str, TraceProjection] = {}
    for score in scores:
        trace_id = score["traceId"]
        if trace_id not in traces:
            raise ValueError("STATE_DIFF_TRACE_SCORE_MISMATCH")
        case_id = score["caseId"]
        if case_id in result:
            raise ValueError(f"STATE_DIFF_DUPLICATE_CASE:{case_id}")
        result[case_id] = traces[trace_id]
    return result


def merge(
    inputs: list[Path],
    output: Path,
    task_set_path: Path = Path("evals/p2j4-decision-state-difference-task-set-v1.json"),
) -> dict[str, Any]:
    tasks = load_task_set(task_set_path)
    task_by_id = {task.caseId: task for task in tasks}
    # Later inputs are explicit failure repairs.  They replace only their own
    # failed case; no successful first-pass case is allowed to be overwritten.
    by_case: dict[str, TraceProjection] = {}
    source_by_case: dict[str, str] = {}
    first_pass_status: dict[str, str] = {}
    for path in inputs:
        payload = _load(path)
        for score in payload.get("scores", []):
            first_pass_status.setdefault(score["caseId"], score["status"])
        for case_id, trace in _index(payload).items():
            if case_id in by_case and first_pass_status.get(case_id) == "PASS":
                raise ValueError(f"STATE_DIFF_SUCCESSFUL_CASE_RERUN_REFUSED:{case_id}")
            by_case[case_id] = trace
            source_by_case[case_id] = str(path)

    missing = [task.caseId for task in tasks if task.caseId not in by_case]
    extra = [case_id for case_id in by_case if case_id not in task_by_id]
    if missing or extra:
        raise ValueError(f"STATE_DIFF_CASE_COVERAGE_INVALID:missing={missing}:extra={extra}")

    traces = [by_case[task.caseId] for task in tasks]
    scores = [score_trace(task, trace) for task, trace in zip(tasks, traces, strict=True)]
    if any(score.status != "PASS" for score in scores):
        failed = [score.caseId for score in scores if score.status != "PASS"]
        raise ValueError("STATE_DIFF_FINAL_NOT_ALL_PASS:" + ",".join(failed))

    env = dict(os.environ)
    env.update({
        "MICO_RESEARCH_PLANNER_BASE_URL": "https://generativelanguage.googleapis.com/v1beta/openai",
        "MICO_RESEARCH_PLANNER_MODEL": "gemini-3.5-flash",
        "MICO_GRAPH_RAG_GENERATOR_BASE_URL": "https://generativelanguage.googleapis.com/v1beta/openai",
        "MICO_GRAPH_RAG_GENERATOR_MODEL": "gemini-3.5-flash",
    })
    actual_external_invocation_count = sum(
        int(input_payload.get("realRunsExecuted", 0))
        for input_payload in (_load(path) for path in inputs)
    )
    payload = _build_real_payload(
        tasks,
        {
            "schemaVersion": tasks[0].schemaVersion,
        },
        env,
        traces,
        scores,
        [],
        [],
        status="COMPLETED",
    )
    payload["mergeAudit"] = {
        "successfulCasesWereNotRerun": True,
        "offlineRescoredWithCurrentOracle": True,
        "sourceArtifacts": [str(path) for path in inputs],
        "traceSourceByCase": source_by_case,
        "firstPassStatusByCase": first_pass_status,
        "actualExternalInvocationCount": actual_external_invocation_count,
        "inputRealRunCounts": {
            str(path): int(_load(path).get("realRunsExecuted", 0))
            for path in inputs
        },
        "uniqueCaseCount": len(tasks),
        "finalScoreRecomputedOffline": True,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "status": "COMPLETED",
        "caseCount": len(tasks),
        "passCount": len(scores),
        "externalInvocationCount": payload["mergeAudit"]["actualExternalInvocationCount"],
        "output": str(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--task-set",
        type=Path,
        default=Path("evals/p2j4-decision-state-difference-task-set-v1.json"),
    )
    parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps(merge(args.inputs, args.output, args.task_set), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
