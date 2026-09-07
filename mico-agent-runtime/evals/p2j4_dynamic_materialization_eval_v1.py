"""Local DeepSeek Materializer + Harness evaluation.

This is deliberately not a policy evaluation and not a new scientific task
set.  Every case supplies a fixed high-level Action and a closed semantic
target.  The script measures whether the materializer can produce a valid
typed plan, whether bounded retry/normalization recovers it, and whether the
result can cross the real Java/Python execution boundary.

Only metadata is persisted.  Model responses, SQL, generated Python, filter
values, row values, credentials, and endpoint tokens are never written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mico_agent_runtime.contracts.generated_analysis import AnalysisPlannerContext
from mico_agent_runtime.contracts.materialization import QueryPlan, validate_query_plan_catalog
from mico_agent_runtime.contracts.research import ScientificPlannerContext
from mico_agent_runtime.contracts.retrieval import build_retrieval_plan
from mico_agent_runtime.contracts.tools import (
    ExecuteReadQueryArguments,
    ExecuteReadQueryJavaToolCall,
)
from mico_agent_runtime.graph.generated_analysis import (
    GeneratedAnalysisError,
    execute_typed_analysis,
)
from mico_agent_runtime.ports.java_agent import HttpJavaAgentToolPort
from mico_agent_runtime.ports.research_planner import HttpResearchPlannerPort
from mico_agent_runtime.ports.schema_catalog import JavaSchemaCatalogPort


EVAL_SCHEMA = "p2j4-dynamic-materialization-eval-v1"
DEFAULT_COUNT = 200
DEFAULT_WORKERS = 5
QUERY_COUNT = 80
ANALYSIS_COUNT = 120
_EXECUTION_CACHE: dict[str, dict[str, Any]] = {}
_EXECUTION_CACHE_LOCK = threading.Lock()


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    kind: str
    action: str
    question: str
    required_fields: tuple[str, ...]
    required_group: str | None = None
    expected_relation_path: tuple[str, ...] = ()
    analysis_type: str | None = None
    metrics: tuple[str, ...] = ("count", "mean")

    def manifest(self) -> dict[str, Any]:
        return {
            "caseId": self.case_id,
            "kind": self.kind,
            "action": self.action,
            "question": self.question,
            "requiredSemanticFields": list(self.required_fields),
            "requiredGroupField": self.required_group,
            "expectedRelationPath": list(self.expected_relation_path),
            "analysisType": self.analysis_type,
            "metrics": list(self.metrics),
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _catalog_hash(catalog: object) -> str:
    return _sha256_json(catalog.model_dump(mode="json"))


def _observation_id(index: int) -> str:
    return "observation-" + hashlib.sha256(f"materialization-eval|{index}".encode()).hexdigest()[:32]


def _query_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": "sample_projection",
            "question": "Read a bounded projection of verified sample fields for a descriptive observational summary.",
            "fields": ("sample.gender",),
            "relations": (),
        },
        {
            "name": "abundance_projection",
            "question": "Read a bounded projection of the verified abundance measurement for a descriptive summary.",
            "fields": ("abundance.value",),
            "relations": ("sample_to_abundance",),
        },
        {
            "name": "sample_abundance_rows",
            "question": "Read bounded sample and abundance fields needed for an observational comparison.",
            "fields": ("sample.gender", "abundance.value"),
            "relations": ("sample_to_abundance",),
        },
        {
            "name": "project_group_mean",
            "question": "Read a bounded project-grouped abundance summary for observational cross-project checking.",
            "fields": ("metadata.project", "abundance.value"),
            "group": "metadata.project",
            "relations": ("sample_to_metadata", "sample_to_abundance"),
        },
        {
            "name": "disease_group_mean",
            "question": "Read a bounded disease-grouped abundance summary for observational comparison.",
            "fields": ("sample.disease", "abundance.value"),
            "group": "sample.disease",
            "relations": ("sample_to_abundance",),
        },
        {
            "name": "country_abundance_rows",
            "question": "Read bounded country and abundance fields for a descriptive observational comparison.",
            "fields": ("sample.country", "abundance.value"),
            "relations": ("sample_to_abundance",),
        },
        {
            "name": "metadata_projection",
            "question": "Read a bounded project metadata projection after the approved catalog relation is selected.",
            "fields": ("metadata.project",),
            "relations": ("sample_to_metadata",),
        },
        {
            "name": "non_null_bounded_read",
            "question": "Read a bounded verified sample field while preserving the approved non-null observation boundary.",
            "fields": ("sample.gender",),
            "relations": (),
        },
    ]


def _analysis_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": "group_gender",
            "action": "compare_groups",
            "analysis_type": "group_comparison",
            "question": "Compare the observed abundance outcome across the approved sample groups.",
            "fields": ("abundance.value", "sample.gender"),
            "group": "sample.gender",
            "metrics": ("count", "mean", "effect_size"),
        },
        {
            "name": "group_project",
            "action": "compare_groups",
            "analysis_type": "group_comparison",
            "question": "Compare the observed abundance outcome across the approved project groups.",
            "fields": ("abundance.value", "metadata.project"),
            "group": "metadata.project",
            "metrics": ("count", "mean", "effect_size"),
        },
        {
            "name": "stratified_country",
            "action": "stratified_analysis",
            "analysis_type": "stratified_comparison",
            "question": "Check whether the bounded group comparison is consistent across the approved country stratum.",
            "fields": ("abundance.value", "sample.country"),
            "metrics": ("count", "mean"),
        },
        {
            "name": "confounder_age_gender",
            "action": "adjust_confounders",
            "analysis_type": "confounder_adjustment",
            "question": "Control the approved age and gender dimensions before interpreting the observed difference.",
            "fields": ("abundance.value", "sample.age", "sample.gender"),
            "metrics": ("count", "mean"),
        },
        {
            "name": "projection_abundance",
            "action": "analyze_projection",
            "analysis_type": "projection",
            "question": "Summarize the verified abundance outcome with bounded descriptive statistics.",
            "fields": ("abundance.value",),
            "metrics": ("count", "mean", "median"),
        },
        {
            "name": "cross_project",
            "action": "cross_project_validate",
            "analysis_type": "cross_project_validation",
            "question": "Validate whether the observed abundance pattern is stable across projects without causal claims.",
            "fields": ("metadata.project", "abundance.value"),
            "group": "metadata.project",
            "metrics": ("count", "mean", "effect_size"),
        },
        {
            "name": "cross_disease",
            "action": "cross_disease_validate",
            "analysis_type": "cross_disease_validation",
            "question": "Validate whether the observed abundance pattern is consistent across disease labels while remaining observational.",
            "fields": ("sample.disease", "abundance.value"),
            "group": "sample.disease",
            "metrics": ("count", "mean", "effect_size"),
        },
    ]


def build_cases(count: int = DEFAULT_COUNT) -> list[EvalCase]:
    if count < 2 or count > 500:
        raise ValueError("count must be between 2 and 500")
    cases: list[EvalCase] = []
    query_specs = _query_specs()
    analysis_specs = _analysis_specs()
    query_target = min(QUERY_COUNT, count // 2)
    analysis_target = count - query_target
    if count == DEFAULT_COUNT:
        query_target, analysis_target = QUERY_COUNT, ANALYSIS_COUNT

    for index in range(query_target):
        spec = query_specs[index % len(query_specs)]
        variant = index // len(query_specs) + 1
        cases.append(EvalCase(
            case_id=f"query-{index + 1:03d}",
            kind="query_plan",
            action="execute_read_query",
            question=f"{spec['question']} Evaluation variant {variant:02d}.",
            required_fields=tuple(spec["fields"]),
            required_group=spec.get("group"),
            expected_relation_path=tuple(spec["relations"]),
        ))
    for index in range(analysis_target):
        spec = analysis_specs[index % len(analysis_specs)]
        variant = index // len(analysis_specs) + 1
        cases.append(EvalCase(
            case_id=f"analysis-{index + 1:03d}",
            kind="analysis_plan",
            action=spec["action"],
            question=f"{spec['question']} Evaluation variant {variant:02d}.",
            required_fields=tuple(spec["fields"]),
            required_group=spec.get("group"),
            analysis_type=spec["analysis_type"],
            metrics=tuple(spec["metrics"]),
        ))
    return cases


def _safe_query_summary(plan: QueryPlan) -> dict[str, Any]:
    """Summarize a QueryPlan without persisting filter values."""

    return {
        "schemaVersion": plan.schemaVersion,
        "rootEntity": plan.root_entity,
        "relationPath": list(plan.relation_path),
        "selectFields": list(plan.select_fields),
        "aggregationFields": [item.field for item in plan.aggregations],
        "aggregationOps": [item.op for item in plan.aggregations],
        "filterFields": [item.field for item in plan.filters],
        "filterOperators": [item.operator for item in plan.filters],
        "groupBy": list(plan.group_by),
        "limit": plan.limit,
    }


def _safe_analysis_summary(plan: object) -> dict[str, Any]:
    return {
        "schemaVersion": plan.schemaVersion,
        "analysisType": plan.analysis_type,
        "sourceObservationCount": len(plan.source_observation_ids),
        "outcome": plan.outcome,
        "groupField": plan.group_field,
        "covariateFields": list(plan.covariates),
        "stratifyFields": list(plan.stratify_by),
        "metrics": list(plan.metrics),
    }


def _plan_hash(plan: object) -> str:
    return _sha256_json(plan.model_dump(mode="json"))


class _RequestCounterTransport:
    """Count provider requests while forwarding them to the real network."""

    # This helper is intentionally metadata-only: it never inspects request
    # bodies or response bodies.
    def __init__(self) -> None:
        import httpx

        self._inner = httpx.HTTPTransport()
        self._lock = threading.Lock()
        self.count = 0

    def handle_request(self, request: object) -> object:
        with self._lock:
            self.count += 1
        return self._inner.handle_request(request)

    def close(self) -> None:
        self._inner.close()


def _build_query_context(case: EvalCase, catalog: object) -> ScientificPlannerContext:
    return ScientificPlannerContext(
        questionSummary=case.question,
        intent="focused_comparison",
        approvedActions=[case.action],
        remainingActionBudget=4,
        observations=[],
        requiredSemanticFields=list(case.required_fields),
        requiredGroupField=case.required_group,
        schemaCatalog=catalog,
    )


def _fixture_source(case: EvalCase) -> dict[str, Any]:
    """Return a synthetic, redacted shape for the materializer pressure run."""

    fields = list(case.required_fields)
    columns = [f"a_{field.replace('.', '_')}" for field in fields]
    rows: list[dict[str, object]] = []
    for index in range(4):
        row: dict[str, object] = {}
        for field in fields:
            alias = f"a_{field.replace('.', '_')}"
            if field == "abundance.value":
                row[alias] = float(index + 1)
            elif field == "sample.age":
                row[alias] = 30 + index
            else:
                row[alias] = f"g{index % 2 + 1}"
        rows.append(row)
    return {"status": "COMPLETED", "errorCode": None, "rowCount": len(rows), "columns": columns, "rows": rows}


def _build_analysis_context(case: EvalCase, source: dict[str, Any] | None = None) -> AnalysisPlannerContext:
    source_id = _observation_id(int(case.case_id.split("-")[-1]) + 1000)
    source = source or {}
    source_columns = [value for value in source.get("columns", []) if isinstance(value, str)]
    source_rows = [value for value in source.get("rows", []) if isinstance(value, dict)]
    return AnalysisPlannerContext(
        questionSummary=case.question,
        workflow=case.action,
        actionName=case.action,
        sourceObservationIds=[source_id],
        # Keep the field vocabulary intentionally tight.  The model is tested
        # on parameterization, not on choosing unrelated catalog fields.
        availableSemanticFields=list(case.required_fields),
        requiredSemanticFields=list(case.required_fields),
        requiredGroupField=case.required_group,
        columns=source_columns[:64],
        previewRows=[
            {key: value for key, value in row.items() if key in set(source_columns[:64])}
            for row in source_rows[:20]
        ],
        executionFeedback=[],
    )


def _relation_path_for_fields(fields: tuple[str, ...]) -> list[str]:
    path: list[str] = []
    if any(field.startswith("metadata.") for field in fields):
        path.append("sample_to_metadata")
    if any(field.startswith("abundance.") for field in fields):
        path.append("sample_to_abundance")
    return path


def _execution_query_for_case(case: EvalCase) -> QueryPlan:
    relation_path = _relation_path_for_fields(case.required_fields)
    group = case.required_group
    outcome = next((field for field in case.required_fields if field == "abundance.value"), None)
    if case.kind == "analysis_plan" and outcome is not None and group is None:
        # Keep the real Java input bounded and fast for non-cross analyses.
        # The model is still evaluated against its own typed AnalysisPlan;
        # this extra grouping is only an execution fixture and is not part of
        # the persisted materializer result.
        group_candidates = [
            field for field in case.required_fields
            if field.startswith("sample.") and field in {"sample.age", "sample.gender", "sample.country"}
        ]
        group = group_candidates[0] if group_candidates else "sample.gender"
    if group and outcome:
        if group not in case.required_fields:
            relation_path = _relation_path_for_fields(tuple([*case.required_fields, group]))
        return QueryPlan(
            root_entity="sample",
            relation_path=relation_path,
            select_fields=[group],
            aggregations=[{"field": outcome, "op": "mean"}],
            group_by=[group],
            limit=1000,
        )
    return QueryPlan(
        root_entity="sample",
        relation_path=relation_path,
        select_fields=list(case.required_fields),
        limit=1000,
    )


def _execute_query_plan(java: HttpJavaAgentToolPort, plan: QueryPlan, case_id: str) -> dict[str, Any]:
    run_id = f"materialization-eval-{case_id}"
    call_id = "call-" + hashlib.sha256(f"{run_id}|query".encode()).hexdigest()[:32]
    call = ExecuteReadQueryJavaToolCall(
        toolName="execute_read_query",
        runId=run_id,
        toolCallId=call_id,
        arguments=ExecuteReadQueryArguments(queryPlan=plan, limit=plan.limit),
    )
    try:
        response = java.execute(call)
    except Exception:
        return {"status": "FAILED", "errorCode": "JAVA_TOOL_EXECUTION_FAILED", "rowCount": None, "rows": []}
    error_code = response.error.code if response.error else None
    rows: list[dict[str, Any]] = []
    if isinstance(response.data, dict) and isinstance(response.data.get("rows"), list):
        rows = [row for row in response.data["rows"] if isinstance(row, dict)]
    return {
        "status": response.status,
        "errorCode": error_code,
        "rowCount": response.rowCount,
        "columns": [
            value for value in (response.data.get("columns", []) if isinstance(response.data, dict) else [])
            if isinstance(value, str)
        ],
        "rows": rows,
    }


def _analysis_source_key(case: EvalCase) -> str:
    return _sha256_json({
        "fields": list(case.required_fields),
        "group": case.required_group,
    })


def _prepare_analysis_sources(
    cases: list[EvalCase],
    java: HttpJavaAgentToolPort,
) -> dict[str, dict[str, Any]]:
    """Prepare one bounded real Java snapshot per analysis input shape.

    Preparation is intentionally serial.  It avoids a thundering herd against
    MySQL while ensuring all repeated materializer variants see the same
    in-memory, transient input shape.
    """

    prepared: dict[str, dict[str, Any]] = {}
    representatives: dict[str, EvalCase] = {}
    for case in cases:
        if case.kind == "analysis_plan":
            representatives.setdefault(_analysis_source_key(case), case)
    for key, case in representatives.items():
        prepared[key] = _execute_query_plan(java, _execution_query_for_case(case), f"source-{key[7:23]}")
    return prepared


def _evaluate_query_case(case: EvalCase, catalog: object, planner: HttpResearchPlannerPort, java: HttpJavaAgentToolPort | None, counter: _RequestCounterTransport, execute_java: bool) -> dict[str, Any]:
    started = time.perf_counter()
    counter.count = 0
    result: dict[str, Any] = {
        "caseId": case.case_id,
        "kind": case.kind,
        "action": case.action,
        "status": "FAILED",
        "providerRequests": 0,
        "firstRequestModelPass": False,
        "modelAfterRetry": False,
        "materializerMode": None,
        "repairCodes": [],
        "fallbackCode": None,
        "fallbackReasonCode": None,
        "contractStatus": "not_run",
        "semanticStatus": "not_run",
        "executionStatus": "not_run",
        "executionErrorCode": None,
        "planHash": None,
        "plan": None,
    }
    try:
        planned = planner.plan_action(_build_query_context(case, catalog))
        result["providerRequests"] = counter.count
        result["materializerMode"] = planned.mode
        result["repairCodes"] = list(getattr(planned, "repairCodes", ()) or ())
        result["fallbackCode"] = planned.fallbackCode
        result["fallbackReasonCode"] = planned.fallbackReasonCode
        result["firstRequestModelPass"] = planned.mode == "model" and counter.count == 1
        result["modelAfterRetry"] = planned.mode == "model" and counter.count > 1
        if planned.mode != "model":
            # This eval measures real DeepSeek materialization.  A
            # deterministic plan may still be executable by the Runtime, but
            # it is not a materializer success and must never be counted as
            # one.
            result["contractStatus"] = "failed_deterministic"
            result["executionErrorCode"] = (
                planned.fallbackReasonCode
                or planned.fallbackCode
                or "DYNAMIC_MATERIALIZER_DETERMINISTIC_FALLBACK"
            )
            return result
        action = planned.action
        plan = getattr(action.arguments, "queryPlan", None)
        if not isinstance(plan, QueryPlan):
            result["contractStatus"] = "failed"
            result["executionErrorCode"] = "DYNAMIC_QUERY_PLAN_REQUIRED"
            return result
        result["contractStatus"] = "passed"
        result["planHash"] = _plan_hash(plan)
        result["plan"] = _safe_query_summary(plan)
        try:
            validate_query_plan_catalog(plan, catalog)
            result["semanticStatus"] = "passed"
            if not set(case.required_fields).issubset(set(plan.select_fields) | {item.field for item in plan.aggregations}):
                result["semanticStatus"] = "failed_required_fields"
            if case.required_group and case.required_group not in plan.group_by:
                result["semanticStatus"] = "failed_required_group"
            # The two catalog relations used for a project+abundance read are
            # independent branches from the sample root.  Their membership
            # must be exact, while Java retains the authoritative ordered
            # connectivity check.
            # A single-entity request may legitimately choose that entity as
            # the QueryPlan root (for example metadata.project) and therefore
            # needs no relation path. Multi-field cross-entity requests must
            # still include the complete catalog relation set.
            if (
                len(case.required_fields) > 1
                and case.expected_relation_path
                and not set(case.expected_relation_path).issubset(set(plan.relation_path))
            ):
                result["semanticStatus"] = "failed_relation_path"
        except Exception:
            result["semanticStatus"] = "failed_catalog"
        if result["semanticStatus"] != "passed":
            return result
        if execute_java:
            if java is None:
                result["executionStatus"] = "not_run"
                result["executionErrorCode"] = "JAVA_EXECUTOR_NOT_CONFIGURED"
            else:
                execution = _execute_query_plan(java, plan, case.case_id)
                result["executionStatus"] = "passed" if execution["status"] == "COMPLETED" else execution["status"]
                result["executionErrorCode"] = execution["errorCode"]
        result["status"] = "PASS" if result["semanticStatus"] == "passed" and (
            not execute_java or result["executionStatus"] == "passed"
        ) else "FAIL"
    except Exception as exc:
        result["executionErrorCode"] = type(exc).__name__
    finally:
        result["providerRequests"] = counter.count
        result["durationMs"] = max(0, int((time.perf_counter() - started) * 1000))
    return result


def _evaluate_analysis_case(case: EvalCase, source: dict[str, Any], planner: HttpResearchPlannerPort, counter: _RequestCounterTransport) -> dict[str, Any]:
    started = time.perf_counter()
    result: dict[str, Any] = {
        "caseId": case.case_id,
        "kind": case.kind,
        "action": case.action,
        "status": "FAILED",
        "providerRequests": 0,
        "firstRequestModelPass": False,
        "modelAfterRetry": False,
        "materializerMode": None,
        "repairCodes": [],
        "fallbackCode": None,
        "contractStatus": "not_run",
        "semanticStatus": "not_run",
        "executionStatus": "not_run",
        "executionErrorCode": None,
        "typedOperatorErrorCode": None,
        "typedOperatorStatus": "not_run",
        "sandboxFallbackStatus": "not_run",
        "sandboxFailureReasonCode": None,
        "planHash": None,
        "plan": None,
    }
    try:
        analysis_context = _build_analysis_context(case, source)
        planned = planner.generate_typed_analysis(analysis_context)
        result["providerRequests"] = counter.count
        result["materializerMode"] = planned.mode
        result["repairCodes"] = list(getattr(planned, "repairCodes", []) or [])
        result["fallbackCode"] = planned.fallbackCode
        result["firstRequestModelPass"] = planned.mode == "model" and counter.count == 1
        result["modelAfterRetry"] = planned.mode == "model" and counter.count > 1
        if planned.mode != "model":
            # The typed plan is useful for Runtime compatibility, but this
            # pressure eval must distinguish it from a model-origin plan.
            result["contractStatus"] = "failed_deterministic"
            result["executionErrorCode"] = (
                planned.fallbackCode
                or "DYNAMIC_MATERIALIZER_DETERMINISTIC_FALLBACK"
            )
            return result
        plan = planned.plan
        result["contractStatus"] = "passed" if planned.mode == "model" else "failed_deterministic"
        result["planHash"] = _plan_hash(plan)
        result["plan"] = _safe_analysis_summary(plan)
        expected_type = case.analysis_type
        referenced = {
            value for value in (plan.outcome, plan.group_field, *plan.covariates, *plan.stratify_by)
            if value is not None
        }
        if plan.analysis_type != expected_type:
            result["semanticStatus"] = "failed_analysis_type"
        elif not set(case.required_fields).issubset(referenced):
            result["semanticStatus"] = "failed_required_fields"
        elif case.required_group and plan.group_field != case.required_group:
            result["semanticStatus"] = "failed_required_group"
        else:
            result["semanticStatus"] = "passed"
        if result["semanticStatus"] != "passed":
            return result

        # The plan itself is still the DeepSeek output. The source is either a
        # bounded real Java snapshot (explicit --real-execution mode) or a
        # synthetic redacted shape for the high-volume materializer run.
        if source["status"] != "COMPLETED":
            result["executionStatus"] = "java_failed"
            result["executionErrorCode"] = source["errorCode"]
            return result
        source_id = plan.source_observation_ids[0]
        executable_plan = plan.model_copy(update={"source_observation_ids": [source_id]})
        try:
            execute_typed_analysis(executable_plan, source["rows"], int(source["rowCount"] or len(source["rows"])), planner_mode="model")
            result["typedOperatorStatus"] = "passed"
            result["executionStatus"] = "passed"
            result["status"] = "PASS"
        except GeneratedAnalysisError as exc:
            result["typedOperatorStatus"] = "failed"
            result["typedOperatorErrorCode"] = exc.code
            result["executionErrorCode"] = exc.code
            # Typed execution failures are never redirected to generated
            # code. Generated execution is valid only when the capability
            # registry selected it before the typed operator was invoked.
            result["executionStatus"] = "failed"
            result["status"] = "FAIL"
    except Exception as exc:
        result["executionErrorCode"] = type(exc).__name__
    finally:
        result["providerRequests"] = counter.count
        result["durationMs"] = max(0, int((time.perf_counter() - started) * 1000))
    return result


def _worker(case: EvalCase, catalog: object, env: dict[str, str], source: dict[str, Any] | None = None, execute_java: bool = False) -> dict[str, Any]:
    counter = _RequestCounterTransport()
    planner = HttpResearchPlannerPort(
        env["MICO_RESEARCH_PLANNER_BASE_URL"],
        env["MICO_RESEARCH_PLANNER_MODEL"],
        env["MICO_RESEARCH_PLANNER_TOKEN"],
        transport=counter,
    )
    java = HttpJavaAgentToolPort.from_environment(env) if case.kind == "query_plan" and execute_java else None
    try:
        if case.kind == "query_plan":
            return _evaluate_query_case(case, catalog, planner, java, counter, execute_java)
        return _evaluate_analysis_case(case, source or {}, planner, counter)
    finally:
        planner.close()
        if java is not None:
            java.close()
        counter.close()


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    def count(predicate) -> int:
        return sum(1 for item in results if predicate(item))

    by_kind: dict[str, dict[str, int]] = {}
    for kind in sorted({item["kind"] for item in results}):
        subset = [item for item in results if item["kind"] == kind]
        by_kind[kind] = {
            "cases": len(subset),
            "pass": count(lambda item, subset=subset: item in subset and item["status"] == "PASS"),
            "firstRequestModelPass": count(lambda item, subset=subset: item in subset and item["firstRequestModelPass"]),
            "modelAfterRetry": count(lambda item, subset=subset: item in subset and item["modelAfterRetry"]),
            "fallback": count(lambda item, subset=subset: item in subset and item["materializerMode"] != "model"),
            "contractPass": count(lambda item, subset=subset: item in subset and item["contractStatus"] == "passed"),
            "semanticPass": count(lambda item, subset=subset: item in subset and item["semanticStatus"] == "passed"),
            "executionPass": count(lambda item, subset=subset: item in subset and item["executionStatus"] == "passed"),
            "typedOperatorPass": count(lambda item, subset=subset: item in subset and item.get("typedOperatorStatus") == "passed"),
            "sandboxFallbackPass": count(lambda item, subset=subset: item in subset and item.get("sandboxFallbackStatus") == "passed"),
            "withRepairCodes": count(lambda item, subset=subset: item in subset and bool(item["repairCodes"])),
            "typedOperatorFailureCodes": {
                code: count(lambda item, code=code, subset=subset: item in subset and item.get("typedOperatorErrorCode") == code)
                for code in sorted({
                    item.get("typedOperatorErrorCode")
                    for item in subset
                    if item.get("typedOperatorErrorCode")
                })
            },
        }
    return {
        "cases": len(results),
        "pass": count(lambda item: item["status"] == "PASS"),
        "firstRequestModelPass": count(lambda item: item["firstRequestModelPass"]),
        "modelAfterRetry": count(lambda item: item["modelAfterRetry"]),
        "fallback": count(lambda item: item["materializerMode"] != "model"),
        "contractPass": count(lambda item: item["contractStatus"] == "passed"),
        "semanticPass": count(lambda item: item["semanticStatus"] == "passed"),
        "executionPass": count(lambda item: item["executionStatus"] == "passed"),
        "typedOperatorPass": count(lambda item: item.get("typedOperatorStatus") == "passed"),
        "sandboxFallbackPass": count(lambda item: item.get("sandboxFallbackStatus") == "passed"),
        "withRepairCodes": count(lambda item: bool(item["repairCodes"])),
        "typedOperatorFailureCodes": {
            code: count(lambda item, code=code: item.get("typedOperatorErrorCode") == code)
            for code in sorted({
                item.get("typedOperatorErrorCode")
                for item in results
                if item.get("typedOperatorErrorCode")
            })
        },
        "byKind": by_kind,
        "failureCodes": {
            code: count(lambda item, code=code: item.get("executionErrorCode") == code)
            for code in sorted({item.get("executionErrorCode") for item in results if item.get("executionErrorCode")})
        },
    }


def run_eval(*, count: int, workers: int, output_dir: Path, env: dict[str, str], execute_java: bool = False) -> Path:
    cases = build_cases(count)
    java = HttpJavaAgentToolPort.from_environment(env)
    try:
        catalog = JavaSchemaCatalogPort(java).load(
            run_id="dynamic-materialization-eval-20260827",
            task_id="dynamic-materialization-catalog",
        )
        analysis_sources = _prepare_analysis_sources(cases, java) if execute_java else {
            _analysis_source_key(case): _fixture_source(case)
            for case in cases
            if case.kind == "analysis_plan"
        }
    finally:
        java.close()
    cases_payload = [case.manifest() for case in cases]
    cases_hash = _sha256_json(cases_payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = output_dir / "cases.json"
    cases_path.write_text(json.dumps({
        "schemaVersion": EVAL_SCHEMA,
        "status": "FROZEN",
        "caseCount": len(cases),
        "casesHash": cases_hash,
        "cases": cases_payload,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    results_path = output_dir / "results.jsonl"
    existing: dict[str, dict[str, Any]] = {}
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and isinstance(item.get("caseId"), str):
                existing[item["caseId"]] = item
    pending_cases = [case for case in cases if case.case_id not in existing]
    results: list[dict[str, Any]] = list(existing.values())
    with results_path.open("a", encoding="utf-8") as result_file:
        with ThreadPoolExecutor(max_workers=max(1, min(workers, 8))) as executor:
            futures = {
                executor.submit(
                    _worker,
                    case,
                    catalog,
                    env,
                    analysis_sources.get(_analysis_source_key(case)) if case.kind == "analysis_plan" else None,
                    execute_java,
                ): case
                for case in pending_cases
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                result_file.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
                result_file.flush()
                print(json.dumps({
                    "caseId": result["caseId"],
                    "kind": result["kind"],
                    "status": result["status"],
                    "materializerMode": result["materializerMode"],
                    "providerRequests": result["providerRequests"],
                    "contractStatus": result["contractStatus"],
                    "semanticStatus": result["semanticStatus"],
                    "executionStatus": result["executionStatus"],
                }, ensure_ascii=False, separators=(",", ":")), flush=True)

    results.sort(key=lambda item: item["caseId"])
    summary = _aggregate(results)
    payload = {
        "schemaVersion": EVAL_SCHEMA,
        "status": "COMPLETED",
        "startedAt": _utc_now(),
        "caseCount": len(cases),
        "workerCount": max(1, min(workers, 8)),
        "provider": "DeepSeek",
        "model": env.get("MICO_RESEARCH_PLANNER_MODEL", ""),
        "materializerConfig": "evals/p2j4_dynamic_e2e_materializer_config_v1.json",
        "catalogHash": _catalog_hash(catalog),
        "casesHash": cases_hash,
        "secretValuesIncluded": False,
        "summary": summary,
        "results": results,
    }
    output_path = output_dir / "summary.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summaryPath": str(output_path), "summary": summary}, ensure_ascii=False, separators=(",", ":")))
    return output_path


def retry_failed_cases(
    *,
    count: int,
    workers: int,
    output_dir: Path,
    env: dict[str, str],
    execute_java: bool = False,
) -> Path:
    """Re-run only failed case IDs from an existing completed/partial run.

    The original ``results.jsonl`` and ``summary.json`` remain immutable audit
    history. Recovery results are append-only in a separate file and are
    merged only for a post-retry view; successful original cases are never
    submitted again.
    """

    cases = build_cases(count)
    results_path = output_dir / "results.jsonl"
    if not results_path.exists():
        raise FileNotFoundError(f"existing results.jsonl is required: {results_path}")
    existing: dict[str, dict[str, Any]] = {}
    for line in results_path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and isinstance(item.get("caseId"), str):
            existing[item["caseId"]] = item
    failed_ids = {
        case_id for case_id, item in existing.items()
        if item.get("status") != "PASS"
    }
    pending = [case for case in cases if case.case_id in failed_ids]
    if not pending:
        output_path = output_dir / "post-retry-summary.json"
        output_path.write_text(json.dumps({
            "schemaVersion": EVAL_SCHEMA,
            "status": "NO_FAILED_CASES",
            "secretValuesIncluded": False,
            "summary": _aggregate(list(existing.values())),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return output_path

    java = HttpJavaAgentToolPort.from_environment(env)
    try:
        catalog = JavaSchemaCatalogPort(java).load(
            run_id="dynamic-materialization-eval-20260827-retry",
            task_id="dynamic-materialization-retry-catalog",
        )
        analysis_sources = _prepare_analysis_sources(pending, java) if execute_java else {
            _analysis_source_key(case): _fixture_source(case)
            for case in pending
            if case.kind == "analysis_plan"
        }
    finally:
        java.close()

    retry_path = output_dir / "retry-results.jsonl"
    retry_existing: dict[str, dict[str, Any]] = {}
    if retry_path.exists():
        for line in retry_path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and isinstance(item.get("caseId"), str):
                retry_existing[item["caseId"]] = item
    retry_pending = [
        case for case in pending
        if retry_existing.get(case.case_id, {}).get("status") != "PASS"
    ]
    retry_results: dict[str, dict[str, Any]] = dict(retry_existing)
    with retry_path.open("a", encoding="utf-8") as result_file:
        with ThreadPoolExecutor(max_workers=max(1, min(workers, 8))) as executor:
            futures = {
                executor.submit(
                    _worker,
                    case,
                    catalog,
                    env,
                    analysis_sources.get(_analysis_source_key(case)) if case.kind == "analysis_plan" else None,
                    execute_java,
                ): case
                for case in retry_pending
            }
            for future in as_completed(futures):
                result = future.result()
                retry_results[result["caseId"]] = result
                result_file.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
                result_file.flush()
                print(json.dumps({
                    "retry": True,
                    "caseId": result["caseId"],
                    "kind": result["kind"],
                    "status": result["status"],
                    "materializerMode": result["materializerMode"],
                    "providerRequests": result["providerRequests"],
                    "contractStatus": result["contractStatus"],
                    "semanticStatus": result["semanticStatus"],
                    "executionStatus": result["executionStatus"],
                }, ensure_ascii=False, separators=(",", ":")), flush=True)

    merged = dict(existing)
    merged.update(retry_results)
    merged_results = sorted(merged.values(), key=lambda item: item["caseId"])
    cases_hash = _sha256_json([case.manifest() for case in cases])
    output_path = output_dir / "post-retry-summary.json"
    payload = {
        "schemaVersion": EVAL_SCHEMA,
        "status": "COMPLETED_WITH_RETRIES",
        "provider": "DeepSeek",
        "model": env.get("MICO_RESEARCH_PLANNER_MODEL", ""),
        "executionMode": "real_java_mysql" if execute_java else "redacted_in_memory_fixture",
        "caseCount": len(cases),
        "casesHash": cases_hash,
        "secretValuesIncluded": False,
        "baseResultPath": str(results_path),
        "retryResultPath": str(retry_path),
        "retriedCaseCount": len(pending),
        "recoveredCaseCount": sum(
            1 for case_id in failed_ids
            if retry_results.get(case_id, {}).get("status") == "PASS"
        ),
        "summary": _aggregate(merged_results),
        "results": merged_results,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summaryPath": str(output_path), "summary": payload["summary"]}, ensure_ascii=False, separators=(",", ":")))
    return output_path


def run_retrieval_smoke(output_dir: Path, env: dict[str, str]) -> Path:
    """Run deterministic retrieval-plan construction plus real local search."""

    # Import lazily so the main Materializer eval stays usable when the local
    # knowledge index is intentionally disabled.
    from mico_agent_runtime.knowledge.local_retriever import LocalKnowledgeSearchPort

    knowledge = LocalKnowledgeSearchPort.from_environment(env)
    cases = [
        ("retrieval-vector", "semantic_fact", "vector", ["vector"], 5),
        ("retrieval-graph", "relation", "graph", ["graph"], 5),
        ("retrieval-hybrid", "composite", "hybrid", ["vector", "sparse", "graph"], 5),
    ]
    results = []
    try:
        for case_id, query_type, mode, branches, top_k in cases:
            plan = build_retrieval_plan(
                query_summary="bounded observational evidence support",
                query_type=query_type,
                retrieval_mode=mode,
                retrieval_branches=branches,
                top_k=top_k,
            )
            status = "FAILED"
            error_code = None
            result_count = 0
            try:
                from mico_agent_runtime.contracts.evidence import EvidenceQuery

                evidence = knowledge.search_parallel(
                    EvidenceQuery(topic="bounded observational evidence support", direction="context", retrievalMode=mode, limit=top_k),
                    branches=tuple(branches),
                )
                result_count = len(evidence)
                status = "PASS" if result_count > 0 else "PARTIAL_EMPTY"
            except Exception:
                error_code = "KNOWLEDGE_SOURCE_FAILED"
            results.append({
                "caseId": case_id,
                "status": status,
                "errorCode": error_code,
                "planId": plan.planId,
                "planHash": _plan_hash(plan),
                "retrievalMode": plan.retrievalMode,
                "retrievalBranches": plan.retrievalBranches,
                "topK": plan.topK,
                "maxHops": plan.maxHops,
                "resultCount": result_count,
            })
    finally:
        close = getattr(knowledge, "close", None)
        if callable(close):
            close()
    output_path = output_dir / "retrieval-summary.json"
    output_path.write_text(json.dumps({
        "schemaVersion": EVAL_SCHEMA,
        "status": "COMPLETED",
        "secretValuesIncluded": False,
        "results": results,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--output-dir",
        default="tmp-runtime-smoke/dynamic-materialization-eval-20260827",
    )
    parser.add_argument(
        "--real-execution",
        action="store_true",
        help="also execute every QueryPlan against local Java/MySQL; otherwise use the in-memory materializer fixture",
    )
    parser.add_argument("--retrieval-smoke", action="store_true")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="re-run only non-PASS case IDs already present in results.jsonl",
    )
    args = parser.parse_args()
    env = dict(os.environ)
    required = ("MICO_RESEARCH_PLANNER_BASE_URL", "MICO_RESEARCH_PLANNER_MODEL", "MICO_RESEARCH_PLANNER_TOKEN")
    if any(not env.get(key, "").strip() for key in required):
        raise SystemExit("MICO_RESEARCH_PLANNER_* is not configured")
    output_dir = Path(args.output_dir)
    if args.retrieval_smoke:
        retrieval_env = dict(env)
        # Keep this process-local. Do not persist or rewrite the user's
        # environment configuration just to run the retrieval smoke.
        retrieval_env["MICO_LOCAL_KNOWLEDGE_ENABLED"] = "true"
        retrieval_env.setdefault("MICO_LOCAL_KNOWLEDGE_RETRIEVAL_BACKEND", "tfidf")
        print(run_retrieval_smoke(output_dir, retrieval_env))
        return 0
    if args.retry_failed:
        retry_failed_cases(
            count=args.count,
            workers=args.workers,
            output_dir=output_dir,
            env=env,
            execute_java=args.real_execution,
        )
    else:
        run_eval(
            count=args.count,
            workers=args.workers,
            output_dir=output_dir,
            env=env,
            execute_java=args.real_execution,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
