"""Replay saved Dynamic QueryPlans against the local Java/MySQL boundary.

This is deliberately a no-provider-call integration check.  It reuses only
the already frozen, redacted plans from the completed materializer pressure
evaluation and persists no SQL, filter values, rows, credentials, or model
output.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from mico_agent_runtime.contracts.materialization import QueryPlan, validate_query_plan_catalog
from mico_agent_runtime.ports.java_agent import HttpJavaAgentToolPort
from mico_agent_runtime.ports.schema_catalog import JavaSchemaCatalogPort

from p2j4_dynamic_materialization_eval_v1 import _execute_query_plan


def _safe_plan_shape(plan: QueryPlan) -> dict[str, Any]:
    return {
        "rootEntity": plan.root_entity,
        "relationHops": len(plan.relation_path),
        "hasAggregation": bool(plan.aggregations),
        "hasGroupBy": bool(plan.group_by),
        "hasFilters": bool(plan.filters),
        "limit": plan.limit,
    }


def _plan_from_redacted_summary(item: dict[str, Any]) -> QueryPlan:
    """Reconstruct only the executable, value-free portion of a saved plan."""
    summary = item["plan"]
    filter_fields = summary.get("filterFields") or []
    if filter_fields:
        # The pressure result intentionally redacts filter values.  Never
        # invent them just to make a replay executable.
        raise ValueError("QUERY_REPLAY_FILTER_VALUES_REDACTED")
    aggregation_fields = summary.get("aggregationFields") or []
    aggregation_ops = summary.get("aggregationOps") or []
    if len(aggregation_fields) != len(aggregation_ops):
        raise ValueError("QUERY_REPLAY_AGGREGATION_SHAPE_INVALID")
    return QueryPlan(
        root_entity=summary["rootEntity"],
        relation_path=list(summary.get("relationPath") or []),
        select_fields=list(summary["selectFields"]),
        aggregations=[
            {"field": field, "op": op}
            for field, op in zip(aggregation_fields, aggregation_ops)
        ],
        group_by=list(summary.get("groupBy") or []),
        limit=summary["limit"],
    )


def _select_unique_saved_plans(summary_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    selected: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for item in payload.get("results", []):
        if not isinstance(item, dict) or item.get("kind") != "query_plan":
            continue
        if item.get("status") != "PASS" or not isinstance(item.get("plan"), dict):
            continue
        plan_hash = item.get("planHash")
        if not isinstance(plan_hash, str) or plan_hash in seen_hashes:
            continue
        seen_hashes.add(plan_hash)
        selected.append(item)
    if not selected:
        raise RuntimeError("no saved PASS QueryPlan is available")
    return selected


def run(summary_path: Path, output_path: Path) -> int:
    env = dict(os.environ)
    required = (
        "MICO_JAVA_AGENT_TOOL_BASE_URL",
        "MICO_AGENT_INTERNAL_TOKEN",
    )
    if any(not env.get(key, "").strip() for key in required):
        raise SystemExit("MICO_JAVA_AGENT_* is not configured")

    saved = _select_unique_saved_plans(summary_path)
    java = HttpJavaAgentToolPort.from_environment(env)
    results: list[dict[str, Any]] = []
    try:
        catalog = JavaSchemaCatalogPort(java).load(
            run_id="dynamic-query-real-replay-20260827",
            task_id="dynamic-query-real-replay-catalog",
        )
        for index, item in enumerate(saved, start=1):
            plan: QueryPlan | None = None
            saved_summary = item["plan"]
            result: dict[str, Any] = {
                "caseId": item["caseId"],
                "sourcePlanHash": item["planHash"],
                "planShape": {
                    "rootEntity": saved_summary.get("rootEntity"),
                    "relationHops": len(saved_summary.get("relationPath") or []),
                    "hasAggregation": bool(saved_summary.get("aggregationFields") or []),
                    "hasGroupBy": bool(saved_summary.get("groupBy") or []),
                    "hasFilters": bool(saved_summary.get("filterFields") or []),
                    "limit": saved_summary.get("limit"),
                },
                "catalogValidation": "failed",
                "javaStatus": "not_run",
                "errorCode": None,
                "rowCount": None,
                "returnedColumnCount": 0,
            }
            try:
                plan = _plan_from_redacted_summary(item)
                result["planShape"] = _safe_plan_shape(plan)
                validate_query_plan_catalog(plan, catalog)
                result["catalogValidation"] = "passed"
            except ValueError as exc:
                result["errorCode"] = str(exc)
                results.append(result)
                continue
            except Exception:
                result["errorCode"] = "QUERY_PLAN_CATALOG_VALIDATION_FAILED"
                results.append(result)
                continue

            execution = _execute_query_plan(
                java,
                plan,
                f"replay-{index:03d}",
            )
            result["javaStatus"] = execution.get("status")
            result["errorCode"] = execution.get("errorCode")
            result["rowCount"] = execution.get("rowCount")
            result["returnedColumnCount"] = len(execution.get("columns") or [])
            results.append(result)
    finally:
        java.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    passed = sum(
        1
        for item in results
        if item["catalogValidation"] == "passed" and item["javaStatus"] == "COMPLETED"
    )
    payload = {
        "schemaVersion": "p2j4-saved-query-real-execution-v1",
        "status": "PASS" if passed == len(results) else "FAIL",
        "sourceSummary": str(summary_path),
        "sourceCaseCount": len(results),
        "uniquePlanCount": len(results),
        "catalogValidationPass": sum(item["catalogValidation"] == "passed" for item in results),
        "javaExecutionPass": passed,
        "javaExecutionFail": len(results) - passed,
        "secretValuesIncluded": False,
        "rawRowsIncluded": False,
        "rawSqlIncluded": False,
        "results": results,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "uniquePlanCount": payload["uniquePlanCount"],
        "catalogValidationPass": payload["catalogValidationPass"],
        "javaExecutionPass": payload["javaExecutionPass"],
        "javaExecutionFail": payload["javaExecutionFail"],
        "output": str(output_path),
    }, ensure_ascii=False, separators=(",", ":")))
    return 0 if payload["status"] == "PASS" else 1


def run_filter_probe(output_path: Path) -> int:
    """Exercise one transient equality filter without persisting its value."""
    env = dict(os.environ)
    required = ("MICO_JAVA_AGENT_TOOL_BASE_URL", "MICO_AGENT_INTERNAL_TOKEN")
    if any(not env.get(key, "").strip() for key in required):
        raise SystemExit("MICO_JAVA_AGENT_* is not configured")

    # The value exists only in this process and is never serialized.  A
    # zero-row COMPLETED response remains a valid execution/data result, not a
    # compiler failure.
    transient_value = "T2D"
    plan = QueryPlan(
        root_entity="sample",
        select_fields=["sample.gender"],
        filters=[{
            "field": "sample.disease",
            "operator": "eq",
            "value": transient_value,
        }],
        limit=10,
    )
    java = HttpJavaAgentToolPort.from_environment(env)
    try:
        catalog = JavaSchemaCatalogPort(java).load(
            run_id="dynamic-query-filter-probe-20260827",
            task_id="dynamic-query-filter-probe-catalog",
        )
        result: dict[str, Any] = {
            "probe": "eq_filter",
            "filterField": "sample.disease",
            "filterOperator": "eq",
            "catalogValidation": "failed",
            "javaStatus": "not_run",
            "executionClassification": "not_run",
            "errorCode": None,
            "rowCount": None,
            "filterValuePersisted": False,
            "rawSqlPersisted": False,
        }
        try:
            validate_query_plan_catalog(plan, catalog)
            result["catalogValidation"] = "passed"
            execution = _execute_query_plan(java, plan, "filter-probe")
            result["javaStatus"] = execution.get("status")
            result["errorCode"] = execution.get("errorCode")
            result["rowCount"] = execution.get("rowCount")
            if result["javaStatus"] == "COMPLETED":
                result["executionClassification"] = (
                    "OBSERVATION_EMPTY" if result["rowCount"] == 0 else "OBSERVATION_NONEMPTY"
                )
            else:
                result["executionClassification"] = "EXECUTION_FAILED"
        except Exception:
            result["errorCode"] = "QUERY_FILTER_PROBE_FAILED"
            result["executionClassification"] = "EXECUTION_FAILED"
    finally:
        java.close()

    passed = (
        result["catalogValidation"] == "passed"
        and result["javaStatus"] == "COMPLETED"
        and result["executionClassification"] in {"OBSERVATION_EMPTY", "OBSERVATION_NONEMPTY"}
        and result["filterValuePersisted"] is False
        and result["rawSqlPersisted"] is False
    )
    payload = {
        "schemaVersion": "p2j4-query-filter-real-execution-v1",
        "status": "PASS" if passed else "FAIL",
        "secretValuesIncluded": False,
        "rawRowsIncluded": False,
        "rawSqlIncluded": False,
        "result": result,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "catalogValidation": result["catalogValidation"],
        "javaStatus": result["javaStatus"],
        "executionClassification": result["executionClassification"],
        "filterValuePersisted": result["filterValuePersisted"],
        "rawSqlPersisted": result["rawSqlPersisted"],
        "output": str(output_path),
    }, ensure_ascii=False, separators=(",", ":")))
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter-probe", action="store_true")
    parser.add_argument(
        "--summary",
        default="tmp-runtime-smoke/dynamic-materialization-eval-20260827-r7/post-retry-summary.json",
    )
    parser.add_argument(
        "--output",
        default="tmp-runtime-smoke/dynamic-materialization-eval-20260827-r7/real-query-execution-replay-20260827.json",
    )
    args = parser.parse_args()
    if args.filter_probe:
        return run_filter_probe(Path(args.output))
    return run(Path(args.summary), Path(args.output))


if __name__ == "__main__":
    raise SystemExit(main())
