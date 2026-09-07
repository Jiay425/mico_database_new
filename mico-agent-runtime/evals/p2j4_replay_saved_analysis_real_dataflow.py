"""Replay saved AnalysisPlan semantics over real transient Java observations.

No provider is called here.  The saved redacted AnalysisPlan summaries are
rebound to a newly created transient Observation from local Java/MySQL rows,
then executed by the existing typed operator.  Only an explicitly labelled,
bounded sandbox fixture is used when the saved plan shape is unsupported by
the typed operator; its code is never persisted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mico_agent_runtime.contracts.materialization import (
    AnalysisPlan as TypedAnalysisPlan,
    QueryPlan,
    validate_query_plan_catalog,
)
from mico_agent_runtime.contracts.research import Observation
from mico_agent_runtime.contracts.tools import ExecuteReadQueryArguments, ExecuteReadQueryJavaToolCall
from mico_agent_runtime.graph.generated_analysis import (
    GeneratedAnalysisError,
    execute_typed_analysis,
)
from mico_agent_runtime.ports.java_agent import HttpJavaAgentToolPort
from mico_agent_runtime.ports.schema_catalog import JavaSchemaCatalogPort

from p2j4_dynamic_materialization_eval_v1 import build_cases


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _relation_path(fields: list[str]) -> list[str]:
    path: list[str] = []
    if any(field.startswith("metadata.") for field in fields):
        path.append("sample_to_metadata")
    if any(field.startswith("abundance.") for field in fields):
        path.append("sample_to_abundance")
    return path


def _source_query(plan_summary: dict[str, Any]) -> QueryPlan:
    fields: list[str] = []
    for value in (
        plan_summary.get("outcome"),
        plan_summary.get("groupField"),
        *(plan_summary.get("covariateFields") or []),
        *(plan_summary.get("stratifyFields") or []),
    ):
        if isinstance(value, str) and value not in fields:
            fields.append(value)
    # Projection has no grouping field.  Keep the real source tabular and
    # bounded without changing the saved AnalysisPlan semantics.
    if not fields:
        fields = ["abundance.value"]
    analysis_type = plan_summary.get("analysisType")
    group_field = plan_summary.get("groupField")
    outcome = plan_summary.get("outcome")
    if (
        analysis_type in {"cross_project_validation", "cross_disease_validation"}
        and isinstance(group_field, str)
        and isinstance(outcome, str)
    ):
        # A bounded raw read may be dominated by one group.  For cross-boundary
        # validation the source must be group-complete, so ask Java for a
        # Catalog-validated grouped aggregate rather than fabricating a second
        # observation or weakening the two-group contract.
        return QueryPlan(
            root_entity="sample",
            relation_path=_relation_path([group_field, outcome]),
            select_fields=[group_field],
            aggregations=[{"field": outcome, "op": "mean"}],
            group_by=[group_field],
            limit=1000,
        )
    return QueryPlan(
        root_entity="sample",
        relation_path=_relation_path(fields),
        select_fields=fields,
        limit=1000,
    )


def _execute_real_read(java: HttpJavaAgentToolPort, plan: QueryPlan, case_id: str) -> Any:
    run_id = f"analysis-dataflow-{case_id}"
    call_id = "call-" + _hash(f"{run_id}|read")[:32]
    call = ExecuteReadQueryJavaToolCall(
        toolName="execute_read_query",
        runId=run_id,
        toolCallId=call_id,
        arguments=ExecuteReadQueryArguments(queryPlan=plan, limit=plan.limit),
    )
    return java.execute(call)


def _rows_from_response(response: Any) -> list[dict[str, object]]:
    if isinstance(response.data, dict) and isinstance(response.data.get("rows"), list):
        return [row for row in response.data["rows"] if isinstance(row, dict)]
    return []


def _columns_from_response(response: Any) -> list[str]:
    if isinstance(response.data, dict) and isinstance(response.data.get("columns"), list):
        return [value for value in response.data["columns"] if isinstance(value, str)]
    return []


def _analysis_plan(item: dict[str, Any], observation_id: str) -> TypedAnalysisPlan:
    summary = item["plan"]
    return TypedAnalysisPlan(
        analysis_type=summary["analysisType"],
        source_observation_ids=[observation_id],
        outcome=summary.get("outcome"),
        group_field=summary.get("groupField"),
        covariates=list(summary.get("covariateFields") or []),
        stratify_by=list(summary.get("stratifyFields") or []),
        metrics=list(summary.get("metrics") or []),
    )


def _select_items(summary_path: Path, only_actions: set[str] | None = None) -> list[dict[str, Any]]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    selected: list[dict[str, Any]] = []
    seen_actions: set[str] = set()
    for item in payload.get("results", []):
        if not isinstance(item, dict) or item.get("kind") != "analysis_plan":
            continue
        if item.get("status") != "PASS" or not isinstance(item.get("plan"), dict):
            continue
        action = item.get("action")
        if not isinstance(action, str) or action in seen_actions:
            continue
        if only_actions is not None and action not in only_actions:
            continue
        seen_actions.add(action)
        selected.append(item)
    required = 6 if only_actions is None else len(only_actions)
    if len(selected) < required:
        raise RuntimeError("not enough saved analysis action types are available")
    return selected


def _merge_retry_results(original_path: Path, retry_path: Path, output_path: Path) -> int:
    original = json.loads(original_path.read_text(encoding="utf-8"))
    retry = json.loads(retry_path.read_text(encoding="utf-8"))
    merged: dict[str, dict[str, Any]] = {
        item["caseId"]: item
        for item in original.get("results", [])
        if isinstance(item, dict) and isinstance(item.get("caseId"), str)
    }
    for item in retry.get("results", []):
        if isinstance(item, dict) and isinstance(item.get("caseId"), str):
            merged[item["caseId"]] = item
    results = [merged[key] for key in sorted(merged)]
    passed = sum(item.get("analysisExecution") == "passed" for item in results)
    payload = {
        "schemaVersion": "p2j4-saved-analysis-real-dataflow-v1",
        "status": "PASS" if passed == len(results) else "FAIL",
        "providerRequests": 0,
        "caseCount": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "retryCaseCount": len(retry.get("results", [])),
        "sourceOriginal": str(original_path),
        "sourceRetry": str(retry_path),
        "secretValuesIncluded": False,
        "rawRowsIncluded": False,
        "rawCodeIncluded": False,
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "caseCount": payload["caseCount"],
        "passed": payload["passed"],
        "failed": payload["failed"],
        "providerRequests": 0,
        "output": str(output_path),
    }, ensure_ascii=False, separators=(",", ":")))
    return 0 if payload["status"] == "PASS" else 1


def run(summary_path: Path, output_path: Path, only_actions: set[str] | None = None) -> int:
    env = dict(os.environ)
    required = ("MICO_JAVA_AGENT_TOOL_BASE_URL", "MICO_AGENT_INTERNAL_TOKEN")
    if any(not env.get(key, "").strip() for key in required):
        raise SystemExit("MICO_JAVA_AGENT_* is not configured")

    cases = {case.case_id: case for case in build_cases(200)}
    items = _select_items(summary_path, only_actions)
    java = HttpJavaAgentToolPort.from_environment(env)
    results: list[dict[str, Any]] = []
    try:
        catalog = JavaSchemaCatalogPort(java).load(
            run_id="analysis-dataflow-replay-20260827",
            task_id="analysis-dataflow-replay-catalog",
        )
        for item in items:
            case_id = item["caseId"]
            case = cases[case_id]
            source_query = _source_query(item["plan"])
            source_id = "observation-" + _hash(f"analysis-dataflow|{case_id}")[:32]
            result: dict[str, Any] = {
                "caseId": case_id,
                "action": item["action"],
                "analysisType": item["plan"]["analysisType"],
                "sourceObservationBound": "failed",
                "javaExecution": "not_run",
                "queryCatalogValidation": "failed",
                "analysisPlanContract": "failed",
                "typedOperator": "not_run",
                "sandboxFixture": "not_run",
                "analysisExecution": "failed",
                "structuredResult": "not_run",
                "resultObservationBound": "not_run",
                "errorCode": None,
                "sourceRowCount": None,
                "sourceColumnCount": 0,
            }
            try:
                validate_query_plan_catalog(source_query, catalog)
                result["queryCatalogValidation"] = "passed"
                response = _execute_real_read(java, source_query, case_id)
                result["javaExecution"] = response.status
                result["sourceRowCount"] = response.rowCount
                columns = _columns_from_response(response)
                rows = _rows_from_response(response)
                result["sourceColumnCount"] = len(columns)
                if response.status != "COMPLETED" or response.dataSnapshot is None or not rows:
                    result["errorCode"] = response.error.code if response.error else "JAVA_OBSERVATION_NOT_AVAILABLE"
                    results.append(result)
                    continue
                snapshot = response.dataSnapshot
                observation = Observation(
                    observationId=source_id,
                    actionId="action-" + _hash(f"{case_id}|read")[:32],
                    actionName="execute_read_query",
                    status="VALIDATED",
                    source="java_controlled_read",
                    queryHash=snapshot.queryHash,
                    rowCount=snapshot.rowCount,
                    generatedAt=snapshot.generatedAt,
                    schemaVersion=response.schemaVersion or "java-read-model-v1",
                    dataSnapshotId=snapshot.dataSnapshotId,
                    snapshotPersistence="transient",
                    queryPlanFields=list(dict.fromkeys([
                        *source_query.select_fields,
                        *(aggregation.field for aggregation in source_query.aggregations),
                    ])),
                    queryPlanRelationPath=list(source_query.relation_path),
                )
                plan = _analysis_plan(item, observation.observationId)
                referenced = {
                    value for value in (
                        plan.outcome,
                        plan.group_field,
                        *plan.covariates,
                        *plan.stratify_by,
                    ) if value is not None
                }
                source_semantic_fields = set(source_query.select_fields) | {
                    aggregation.field for aggregation in source_query.aggregations
                }
                if not referenced.issubset(source_semantic_fields):
                    result["errorCode"] = "ANALYSIS_SOURCE_FIELD_NOT_RETURNED"
                    results.append(result)
                    continue
                if plan.source_observation_ids != [observation.observationId]:
                    result["errorCode"] = "ANALYSIS_SOURCE_OBSERVATION_MISMATCH"
                    results.append(result)
                    continue
                result["sourceObservationBound"] = "passed"
                result["analysisPlanContract"] = "passed"
                generated_result: Any
                try:
                    generated_result = execute_typed_analysis(
                        plan, rows, len(rows), planner_mode="model"
                    )
                    result["typedOperator"] = "passed"
                    result["analysisExecution"] = "passed"
                except GeneratedAnalysisError as exc:
                    result["typedOperator"] = exc.code
                    # A typed execution error is not a generated-executor
                    # trigger. Replay preserves the same pre-execution
                    # capability boundary as the live runtime.
                    result["errorCode"] = exc.code
                    results.append(result)
                    continue
                if result["analysisExecution"] == "passed":
                    analysis_observation = Observation(
                        observationId="observation-" + _hash(
                            f"analysis-result|{case_id}"
                        )[:32],
                        actionId="action-" + _hash(
                            f"{case_id}|analysis"
                        )[:32],
                        actionName=item["action"],
                        status="VALIDATED",
                        source="python_bounded_analysis",
                        queryHash=observation.queryHash,
                        rowCount=generated_result.rowCount,
                        generatedAt=datetime.now(timezone.utc),
                        schemaVersion=generated_result.codeVersion,
                        dataSnapshotId=observation.dataSnapshotId,
                        snapshotPersistence=observation.snapshotPersistence,
                    )
                    if analysis_observation.status != "VALIDATED":
                        result["errorCode"] = "ANALYSIS_RESULT_OBSERVATION_INVALID"
                    else:
                        result["structuredResult"] = "passed"
                        result["resultObservationBound"] = "passed"
            except Exception as exc:
                result["errorCode"] = type(exc).__name__
            results.append(result)
    finally:
        java.close()

    passed = sum(
        1
        for item in results
        if item["sourceObservationBound"] == "passed"
        and item["javaExecution"] == "COMPLETED"
        and item["queryCatalogValidation"] == "passed"
        and item["analysisPlanContract"] == "passed"
        and item["analysisExecution"] == "passed"
        and item["structuredResult"] == "passed"
        and item["resultObservationBound"] == "passed"
    )
    payload = {
        "schemaVersion": "p2j4-saved-analysis-real-dataflow-v1",
        "status": "PASS" if passed == len(results) else "FAIL",
        "providerRequests": 0,
        "caseCount": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "secretValuesIncluded": False,
        "rawRowsIncluded": False,
        "rawCodeIncluded": False,
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "caseCount": payload["caseCount"],
        "passed": payload["passed"],
        "failed": payload["failed"],
        "providerRequests": 0,
        "output": str(output_path),
    }, ensure_ascii=False, separators=(",", ":")))
    return 0 if payload["status"] == "PASS" else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        default="tmp-runtime-smoke/dynamic-materialization-eval-20260827-r7/post-retry-summary.json",
    )
    parser.add_argument(
        "--output",
        default="tmp-runtime-smoke/dynamic-materialization-eval-20260827-r7/real-analysis-dataflow-20260827.json",
    )
    parser.add_argument(
        "--only-actions",
        default="",
        help="comma-separated action names for a targeted retry; only those saved cases are run",
    )
    parser.add_argument("--merge-into", default="")
    parser.add_argument("--merge-output", default="")
    args = parser.parse_args()
    if args.merge_into and args.merge_output:
        return _merge_retry_results(Path(args.merge_into), Path(args.output), Path(args.merge_output))
    only_actions = {value.strip() for value in args.only_actions.split(",") if value.strip()} or None
    return run(Path(args.summary), Path(args.output), only_actions)


if __name__ == "__main__":
    raise SystemExit(main())
