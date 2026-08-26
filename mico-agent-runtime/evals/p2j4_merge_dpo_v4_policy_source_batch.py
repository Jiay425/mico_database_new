"""Freeze additive policy-source real-run checkpoints into one batch asset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evals.p2j4_runner import build_stability_baseline, load_task_set
from mico_agent_runtime.contracts.trace_eval import EvalScore, TraceProjection


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _services(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    value = payload.get("servicesObserved") or {}
    if not isinstance(value, dict):
        raise ValueError(f"POLICY_SOURCE_SERVICES_PROVENANCE_INVALID:{path.name}")
    return value


def merge(
    inputs: list[Path], task_set_path: Path, excluded_case_ids: set[str] | None = None
) -> dict[str, Any]:
    if len(inputs) < 2:
        raise ValueError("POLICY_SOURCE_MERGE_REQUIRES_MULTIPLE_INPUTS")

    excluded_case_ids = excluded_case_ids or set()
    scores_by_case: dict[str, dict[str, Any]] = {}
    traces_by_case: dict[str, dict[str, Any]] = {}
    services_by_source: dict[str, dict[str, Any]] = {}
    bad_cases: list[dict[str, Any]] = []
    oracle_verifications: list[dict[str, Any]] = []
    schema_versions: set[str] = set()
    for path in inputs:
        payload = _load(path)
        if payload.get("mode") != "real_run":
            raise ValueError(f"POLICY_SOURCE_INPUT_MUST_BE_REAL_RUN:{path.name}")
        schema_versions.add(str(payload.get("schemaVersion")))
        services_by_source[path.name] = _services(payload, path)
        scores = payload.get("scores", [])
        traces = payload.get("traces", [])
        traces_by_id = {str(item.get("traceId")): item for item in traces}
        if len(traces_by_id) != len(traces) or "None" in traces_by_id:
            raise ValueError(f"POLICY_SOURCE_TRACE_IDS_INVALID:{path.name}")
        for score in scores:
            case_id = str(score.get("caseId"))
            if case_id in excluded_case_ids and score.get("status") != "PASS":
                continue
            if case_id in scores_by_case:
                raise ValueError(f"POLICY_SOURCE_CASE_OVERLAP:{case_id}")
            if score.get("status") != "PASS":
                raise ValueError(f"POLICY_SOURCE_INPUT_NOT_ALL_PASS:{case_id}")
            trace_id = str(score.get("traceId"))
            trace = traces_by_id.get(trace_id)
            if trace is None:
                raise ValueError(f"POLICY_SOURCE_SCORE_TRACE_MISMATCH:{case_id}")
            scores_by_case[case_id] = score
            traces_by_case[case_id] = trace
        bad_cases.extend(
            item for item in payload.get("badCases", [])
            if str(item.get("traceId")) not in {
                str(score.get("traceId"))
                for score in scores
                if str(score.get("caseId")) in excluded_case_ids
            }
        )
        oracle_verifications.extend(payload.get("resultOracleVerifications", []))

    if len(schema_versions) != 1:
        raise ValueError("POLICY_SOURCE_SCHEMA_VERSION_MISMATCH")
    tasks = load_task_set(task_set_path)
    task_order = {task.caseId: index for index, task in enumerate(tasks)}
    if any(case_id not in task_order for case_id in scores_by_case):
        raise ValueError("POLICY_SOURCE_CASE_NOT_IN_TASK_SET")
    ordered_case_ids = sorted(scores_by_case, key=lambda case_id: task_order[case_id])
    ordered_scores = [scores_by_case[case_id] for case_id in ordered_case_ids]
    ordered_traces = [traces_by_case[case_id] for case_id in ordered_case_ids]

    observed: dict[str, Any] = {}
    for source_services in services_by_source.values():
        for key, value in source_services.items():
            if key not in observed:
                observed[key] = value
            elif key == "plannerModelUsed" and observed[key] != value:
                raise ValueError(f"POLICY_SOURCE_SERVICES_MISMATCH:{key}")
            elif isinstance(observed[key], bool) and isinstance(value, bool):
                # Route-dependent service observations are additive: a
                # canary may legitimately skip a service used by later cases.
                observed[key] = observed[key] or value
            elif observed[key] != value:
                raise ValueError(f"POLICY_SOURCE_SERVICES_MISMATCH:{key}")
    trace_models = [TraceProjection.model_validate(item) for item in ordered_traces]
    score_models = [EvalScore.model_validate(item) for item in ordered_scores]
    baseline = build_stability_baseline(schema_versions.pop(), score_models, trace_models)
    return {
        "schemaVersion": "p2j4-dpo-v4-policy-source-merge-v1",
        "sourceTaskSet": task_set_path.name,
        "checkpointVersion": "p2j4-policy-source-batch-merge-v1",
        "mode": "real_run_merged",
        "status": "FROZEN_ALL_PASS",
        "caseCount": len(ordered_case_ids),
        "selectedCaseIds": ordered_case_ids,
        "completedCaseIds": ordered_case_ids,
        "remainingCaseIds": [],
        "realRunsExecuted": len(ordered_case_ids),
        "externalCalls": True,
        "servicesObserved": observed,
        "sourceServicesObserved": services_by_source,
        "baseline": baseline.model_dump(mode="json"),
        "traces": ordered_traces,
        "scores": ordered_scores,
        "badCases": bad_cases,
        "resultOracleVerifications": oracle_verifications,
        "mergeAudit": {
            "sourceArtifacts": [path.name for path in inputs],
            "inputCount": len(inputs),
            "successfulCasesWereNotRerun": True,
            "caseSetsDisjoint": True,
            "replacedFailedCaseIds": sorted(excluded_case_ids),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--exclude-case-id", action="append", default=[])
    parser.add_argument("--task-set", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = merge(args.input, args.task_set, set(args.exclude_case_id))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "caseCount": result["caseCount"],
        "passCount": sum(item["status"] == "PASS" for item in result["scores"]),
        "sourceArtifacts": result["mergeAudit"]["sourceArtifacts"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
