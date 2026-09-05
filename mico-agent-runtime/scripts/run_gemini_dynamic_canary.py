"""4D-1B Gemini-only Dynamic Scientific Agent canary.

The historical ``run_local_scientific_canary`` remains a deterministic
regression test.  This runner is deliberately separate: its Happy Path uses
Gemini for Task Understanding, Scientific Policy, and Action materialization,
while the Java read-model and typed Python operators remain real runtime
components.  No Qwen, DeepSeek, local policy stub, fixture Java port, or
deterministic materializer fallback is used on the Happy Path.

The command is intentionally a canary rather than a production entry point.
All generated artifacts are provenance-labelled and ``training_eligible`` is
always false because the route, provider, and data source are test-only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import threading
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

import httpx
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mico_agent_runtime.contracts.decision_state import ScientificDecisionState
from mico_agent_runtime.contracts.generated_analysis import GeneratedAnalysisPlan
from mico_agent_runtime.contracts.materialization import AnalysisPlan, QueryPlan, validate_query_plan_catalog
from mico_agent_runtime.contracts.research import (
    CompareGroupsAction,
    ProjectionAnalysisArguments,
    ResearchTask,
    ScientificObservationSummary,
    ScientificPlannerContext,
    validate_scientific_action,
)
from mico_agent_runtime.contracts.schema_catalog import SchemaSemanticCatalog
from mico_agent_runtime.contracts.tools import JavaToolCall, JavaToolResponse
from mico_agent_runtime.graph.generated_analysis import GeneratedAnalysisError, execute_typed_analysis
from mico_agent_runtime.ports.decision_policy import HttpDecisionSftPlannerPort
from mico_agent_runtime.ports.gemini_resilience import GeminiRequestBudget, GeminiResponseCache
from mico_agent_runtime.ports.java_agent import HttpJavaAgentToolPort
from mico_agent_runtime.ports.research_planner import (
    HttpResearchPlannerPort,
    HybridIntentPlannerPort,
    _canonicalize_scientific_action_shape,
    _canonicalize_query_aggregation_aliases,
    _canonicalize_query_plan_wrapper,
    _canonicalize_typed_analysis_shape,
)
from mico_agent_runtime.ports.schema_catalog import JavaSchemaCatalogPort
from mico_agent_runtime.ports.task_understanding import build_task_understanding_port
from mico_agent_runtime.runtime.scientific_service import ScientificRuntime
from mico_agent_runtime.runtime.objective_resolution import (
    completion_semantics_from_resolution,
)
from scripts.check_scientific_chain_services import run_preflight


CANARY_QUERY = (
    "比较 T2D 和 Healthy 的菌群差异，判断这种差异是否受年龄影响、"
    "在不同项目里是否稳定，并结合文献解释。"
)
ALL_ACTIONS = [
    "inspect_cohort",
    "execute_read_query",
    "compare_groups",
    "analyze_projection",
    "stratified_analysis",
    "adjust_confounders",
    "cross_project_validate",
    "cross_disease_validate",
    "retrieve_evidence",
    "finish",
]
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
GEMINI_MODEL = "gemini-3.5-flash-lite"
OUT_DIR = REPO_ROOT / "artifacts" / "gemini_dynamic_canary"
_ARTIFACT_PREFIXES = (
    "tu_request_",
    "tu_response_",
    "policy_request_",
    "policy_response_",
    "materializer_request_",
    "materializer_response_",
    "state_s",
)
_ARTIFACT_SINGLETONS = {
    "semantic_catalog.json",
    "trace.json",
    "failure_canary_report.json",
    "a100_readiness.json",
    "query_semantics_audit.json",
    "materializer_audit.json",
    "preflight.json",
    "objective_resolution.json",
    "completion_semantics.json",
}


_ACTION_ANALYSIS_TYPES = {
    "compare_groups": "group_comparison",
    "stratified_analysis": "stratified_comparison",
    "adjust_confounders": "confounder_adjustment",
    "cross_project_validate": "cross_project_validation",
    "cross_disease_validate": "cross_disease_validation",
    "analyze_projection": "projection",
}


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _clear_previous_artifacts(out_dir: Path) -> None:
    """Remove only files produced by this runner before starting a new run."""

    out_dir.mkdir(parents=True, exist_ok=True)
    for path in out_dir.iterdir():
        if not path.is_file():
            continue
        if path.name in _ARTIFACT_SINGLETONS or path.name.startswith(_ARTIFACT_PREFIXES):
            path.unlink()


def _decode_content(value: Any) -> Any:
    """Decode an OpenAI-compatible message content when it is JSON text."""

    if not isinstance(value, str):
        return value
    text = value.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _query_field_aliases(field_id: str) -> tuple[str, ...]:
    """Return Java's closed aliases for one semantic field.

    The Java read model deliberately exposes aliases instead of physical
    column names.  The canary may inspect those aliases after execution for
    an audit profile; this helper is not used to build or alter a QueryPlan.
    """

    base = "a_" + field_id.replace(".", "_")
    return (
        base,
        f"{base}_count",
        f"{base}_mean",
        f"{base}_min",
        f"{base}_max",
        f"{base}_sum",
    )


def _query_plan_parameters(plan: object | None) -> list[object]:
    """Mirror the Java compiler's bound parameter order for audit only."""

    if not isinstance(plan, QueryPlan):
        return []
    parameters: list[object] = []
    def append_filter_values() -> None:
        for item in plan.filters:
            if item.operator in {"eq", "neq"}:
                parameters.append(item.value)
            elif item.operator == "in" and isinstance(item.value, list):
                parameters.extend(item.value)

    # QueryPlanCompiler binds the outer WHERE values first.  A sample-bounded
    # raw projection then repeats the root filters inside the pre-join
    # subquery, followed by the per-group cap and the outer row LIMIT.
    append_filter_values()
    if plan.sample_limit_per_group is not None:
        append_filter_values()
        parameters.append(plan.sample_limit_per_group)
    parameters.append(plan.limit)
    return parameters


def _query_result_profile(
    payload: object,
    plan: object | None,
    *,
    authoritative_row_count: int | None = None,
    query_hash: str | None = None,
) -> dict[str, Any]:
    """Summarize real Java rows without copying them into the trace.

    ``row_count`` is the number of returned result rows.  It is intentionally
    kept separate from ``unique_sample_count``: dynamic raw projections may
    carry Java's run-scoped opaque analysis key, while aggregate projections
    do not.  When that key is absent, the latter is reported as unavailable
    instead of being guessed from fan-out rows.
    """

    data = payload if isinstance(payload, dict) else {}
    raw_rows = data.get("rows")
    rows = [row for row in raw_rows if isinstance(row, dict)] if isinstance(raw_rows, list) else []
    columns = [value for value in data.get("columns", []) if isinstance(value, str)] \
        if isinstance(data.get("columns"), list) else []

    def values(field_id: str) -> list[object]:
        aliases = _query_field_aliases(field_id)
        aliases_present = [alias for alias in aliases if alias in columns]
        if not aliases_present:
            return []
        alias = aliases_present[0]
        return [row.get(alias) for row in rows]

    def text_counts(field_id: str) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for value in values(field_id):
            if value is None:
                continue
            text = str(value).strip()
            if text:
                counts[text] += 1
        return dict(sorted(counts.items(), key=lambda item: item[0]))

    def distinct(field_id: str) -> list[str]:
        return sorted(text_counts(field_id))

    def null_count(field_id: str) -> int | None:
        aliases = _query_field_aliases(field_id)
        alias = next((item for item in aliases if item in columns), None)
        if alias is None:
            return None
        return sum(row.get(alias) is None for row in rows)

    sample_aliases = [
        alias for alias in columns
        if alias == "a_analysis_sample_key"
        or re.fullmatch(r"a_(?:sample|metadata|abundance)_(?:patient_id|sample_id)", alias)
    ]
    if sample_aliases:
        sample_alias = (
            "a_analysis_sample_key"
            if "a_analysis_sample_key" in sample_aliases
            else sample_aliases[0]
        )
        unique_sample_count: int | None = len({
            tuple(row.get(alias) for alias in sample_aliases)
            for row in rows
            if any(row.get(alias) is not None for alias in sample_aliases)
        })
        sample_count_source = (
            "returned_runtime_opaque_sample_key"
            if "a_analysis_sample_key" in sample_aliases
            else "returned_non_sensitive_sample_key"
        )
    else:
        unique_sample_count = None
        sample_count_source = "not_exposed_by_java_catalog_or_response"

    unique_samples_by_disease: dict[str, int] = {}
    if sample_aliases:
        disease_values = values("sample.disease")
        grouped_samples: dict[str, set[object]] = {}
        for index, disease in enumerate(disease_values):
            if disease is None or index >= len(rows):
                continue
            key = rows[index].get(sample_alias)
            if key is None:
                continue
            label = str(disease).strip()
            if label:
                grouped_samples.setdefault(label, set()).add(key)
        unique_samples_by_disease = {
            label: len(keys) for label, keys in sorted(grouped_samples.items())
        }

    plan_value = plan if isinstance(plan, QueryPlan) else None
    selected_or_grouped_fields = (
        set(plan_value.select_fields) | set(plan_value.group_by)
        if plan_value is not None else set()
    )
    has_abundance_aggregation = bool(
        plan_value is not None
        and any(item.field == "abundance.value" for item in plan_value.aggregations)
    )
    feature_dimension_selected = "abundance.feature" in selected_or_grouped_fields
    # An abundance outcome aggregated without the feature dimension is a
    # scientifically lossy projection: values for every feature are pooled
    # before a later group comparison can see them.  This is an audit fact,
    # never a replacement Action or an operator change.
    feature_pooling_risk = has_abundance_aggregation and not feature_dimension_selected
    profile = {
        "row_count": authoritative_row_count if authoritative_row_count is not None else len(rows),
        "returned_row_count": len(rows),
        "columns": columns,
        "distinct_disease_values": distinct("sample.disease"),
        "disease_value_counts": text_counts("sample.disease"),
        "distinct_disease_count": len(distinct("sample.disease")),
        "distinct_project_values": distinct("metadata.project"),
        "distinct_project_count": len(distinct("metadata.project")),
        "distinct_sample_count": unique_sample_count,
        "unique_samples_by_disease": unique_samples_by_disease,
        "sample_count_source": sample_count_source,
        # The Java response currently exposes joined rows but not a stable
        # non-sensitive sample key.  Never describe the row-derived counts as
        # independent samples; this label is consumed by the 4D-1C audit.
        "group_size_semantics": "returned_rows_or_aggregation_rows_not_unique_samples",
        "is_aggregation_result": bool(plan_value is not None and plan_value.aggregations),
        "aggregation_row_semantics": (
            "one_row_per_group_or_group_key"
            if plan_value is not None and plan_value.aggregations and plan_value.group_by
            else "joined_result_rows"
        ),
        "distinct_abundance_feature_count": len(distinct("abundance.feature")),
        "distinct_abundance_feature_values": distinct("abundance.feature")[:64],
        "feature_dimension_selected": feature_dimension_selected,
        "feature_dimension_grouped": bool(
            plan_value is not None and "abundance.feature" in plan_value.group_by
        ),
        "feature_pooling_risk": feature_pooling_risk,
        "sample_key_selected": bool(sample_aliases),
        "sample_level_analysis_ready": bool(feature_dimension_selected and sample_aliases),
        "null_counts": {
            "disease": null_count("sample.disease"),
            "project": null_count("metadata.project"),
            "abundance_value": (
                null_count("abundance.value")
                if any(alias.startswith("a_abundance_value") for alias in columns)
                else None
            ),
            "feature": null_count("abundance.feature"),
        },
        "limit": plan_value.limit if plan_value is not None else None,
        "sample_limit_per_group": (
            plan_value.sample_limit_per_group if plan_value is not None else None
        ),
        "sample_limit_group_field": (
            plan_value.sample_limit_group_field if plan_value is not None else None
        ),
        "sample_bound_applied": bool(
            plan_value is not None and plan_value.sample_limit_per_group is not None
        ),
        # QueryPlanCompiler emits SQL ``LIMIT ?`` and DynamicReadQueryService
        # additionally calls PreparedStatement.setMaxRows(maxRows). Both
        # bounds count result rows, not unique samples.
        "limit_semantics": "result_rows" if plan_value is not None else "legacy_sql_result_rows",
        "compiled_sql": "not_exposed_by_java_contract; Java compiler owns SQL",
        "compiled_sql_fingerprint": query_hash,
        "sql_parameters": _query_plan_parameters(plan_value),
        "filter_parameters": [
            {
                "field": item.field,
                "operator": item.operator,
                "value": item.value,
            }
            for item in (plan_value.filters if plan_value is not None else [])
        ],
        "group_by": list(plan_value.group_by) if plan_value is not None else [],
        "aggregation_fields": [item.field for item in plan_value.aggregations]
        if plan_value is not None else [],
    }
    return profile


class RecordingTransport(httpx.BaseTransport):
    """Record the real outbound Gemini HTTP boundary without bearer secrets."""

    def __init__(self, role: str) -> None:
        self.role = role
        self.records: list[dict[str, Any]] = []
        self._inner = httpx.HTTPTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        request_body: Any = None
        try:
            request_body = json.loads(request.content.decode("utf-8"))
        except Exception:
            request_body = {"_unparsed_body": True}
        record: dict[str, Any] = {
            "request_index": len(self.records) + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "role": self.role,
            "request": {
                "method": request.method,
                "url": str(request.url),
                "headers": {
                    "content-type": request.headers.get("content-type"),
                    "authorization": "[redacted]" if request.headers.get("authorization") else None,
                },
                "body": request_body,
            },
        }
        try:
            response = self._inner.handle_request(request)
            response.read()
            record["response_status"] = response.status_code
            try:
                record["response"] = response.json()
            except Exception:
                record["response"] = {"_unparsed_response": response.text[:4000]}
            self.records.append(record)
            return response
        except Exception as exc:
            record["error"] = type(exc).__name__
            self.records.append(record)
            raise

    def close(self) -> None:
        self._inner.close()


class RecordingJavaPort:
    """Thin recorder that delegates every call to the real Java HTTP port."""

    def __init__(self, inner: HttpJavaAgentToolPort) -> None:
        self.inner = inner
        self.calls: list[dict[str, Any]] = []

    def execute(self, call: JavaToolCall) -> JavaToolResponse:
        query_plan = getattr(call.arguments, "queryPlan", None)
        entry: dict[str, Any] = {
            "call_index": len(self.calls) + 1,
            "tool_name": call.toolName,
            "tool_call_id": call.toolCallId,
            # The Java contract accepts a semantic QueryPlan (or the legacy
            # SQL shape).  Persist only the typed, semantic arguments so the
            # canary can prove what crossed the Runtime→Java boundary
            # without exposing compiled SQL or credentials.
            "arguments": call.arguments.model_dump(mode="json", exclude_none=True),
        }
        try:
            response = self.inner.execute(call)
            data_snapshot = getattr(response, "dataSnapshot", None)
            entry.update({
                "status": response.status,
                "row_count": response.rowCount,
                "source": response.source,
                "schema_version": response.schemaVersion,
                "columns": (
                    response.data.get("columns", [])
                    if isinstance(response.data, dict)
                    else []
                ),
                "error_code": getattr(response.error, "code", None),
                "query_result_profile": _query_result_profile(
                    response.data,
                    query_plan,
                    authoritative_row_count=response.rowCount,
                    query_hash=getattr(data_snapshot, "queryHash", None),
                ),
            })
            self.calls.append(entry)
            return response
        except Exception as exc:
            entry.update({"status": "exception", "error_code": str(exc)})
            self.calls.append(entry)
            raise

    def close(self) -> None:
        self.inner.close()


def _gemini_token() -> str:
    token = (
        os.environ.get("MICO_GEMINI_API_KEY", "").strip()
        or os.environ.get("GEMINI_API_KEY", "").strip()
        or os.environ.get("GOOGLE_API_KEY", "").strip()
    )
    if not token:
        raise RuntimeError("GEMINI_API_KEY_REQUIRED")
    return token


def _task(*, max_actions: int = 8) -> ResearchTask:
    if max_actions < 1:
        raise ValueError("max_actions must be positive")
    return ResearchTask(
        runId="run-gemini-4d1b",
        taskId="task-gemini-4d1b",
        requesterId="canary",
        traceId="trace-gemini-4d1b",
        question=CANARY_QUERY,
        intent="scientific_exploration",
        requestedScopes=["mico:query:read", "mico:research:read", "mico:evidence:read"],
        allowedActions=ALL_ACTIONS,
        # Keep enough budget for the model to recover from a real empty/one
        # group read and still execute one analysis.  This is only the canary
        # budget; it does not change the production task contract.
        maxActions=max_actions,
        createdAt=datetime.now(timezone.utc),
    )


def _initial_graph_state(
    runtime: ScientificRuntime,
    request: ResearchTask,
    catalog: SchemaSemanticCatalog,
) -> dict[str, Any]:
    """Invoke the actual compiled LangGraph so the final state is inspectable."""

    return runtime._graph.invoke({
        "request": request,
        "auditEvents": [],
        "observations": [],
        "rawObservationPayloads": {},
        "analysisPlans": [],
        "analysisResults": [],
        "analysisEvidence": [],
        "actionHistory": [],
        "actionSignatures": [],
        "plannerFeedback": [],
        "dynamicMaterialization": True,
        "readReplanCount": 0,
        "fallbackCodes": [],
        "schemaCatalog": catalog,
    })


def _state_snapshot(state: Any) -> dict[str, Any] | None:
    if isinstance(state, ScientificDecisionState):
        return state.model_dump(mode="json")
    if isinstance(state, dict):
        return ScientificDecisionState.model_validate(state).model_dump(mode="json")
    return None


def _save_gemini_records(
    out_dir: Path,
    transport: RecordingTransport,
    prefix: str,
) -> None:
    for record in transport.records:
        index = int(record["request_index"])
        _dump(out_dir / f"{prefix}_request_{index:03d}.json", record["request"])
        response = {
            "response_status": record.get("response_status"),
            "response": record.get("response"),
            "error": record.get("error"),
        }
        _dump(out_dir / f"{prefix}_response_{index:03d}.json", response)


def _normalise_materializer_response(value: object) -> object:
    """Apply only the same lossless wrapper normalizations as Runtime."""

    if not isinstance(value, dict):
        return value
    value = dict(value)
    # Mirror the provider boundary's lossless whitespace repair so the audit
    # reports API-contract adherence after the exact normalization Runtime
    # applies, rather than counting a harmless JSON key typo as an HTTP retry.
    if " rationale" in value and "rationale" not in value:
        value["rationale"] = value.pop(" rationale")
    value = _canonicalize_query_plan_wrapper(value)
    value = _canonicalize_query_aggregation_aliases(value)
    # Scalar-to-list normalization belongs to the typed AnalysisPlan
    # contract.  Applying it to a ScientificAction envelope would silently
    # add ``covariates``/``stratify_by`` keys to a valid query action and make
    # this audit report a false schema failure that Runtime never saw.
    if "analysis_type" in value:
        value = _canonicalize_typed_analysis_shape(value)
    return value


def _materializer_failure_reason(
    *,
    status: int | None,
    context: object,
    response: object,
    catalog: SchemaSemanticCatalog | None = None,
) -> str | None:
    """Classify one model attempt without inventing a replacement plan."""

    if status != 200:
        return "http_failure"
    if not isinstance(response, dict):
        return "response_shape"
    approved = context.get("approvedActions", []) if isinstance(context, dict) else []
    action_name = response.get("actionName")
    arguments = response.get("arguments")
    # A numeric stratifier (for example sample.age) is intentionally classified
    # by Capability Registry as SUPPORTED_GENERATED.  The dynamic runtime then
    # invokes the historical sandbox-code planner after a valid v2 plan.  That
    # response is a different, explicitly bounded compatibility contract; do
    # not mislabel it as a missing ScientificAction wrapper in the audit.
    if {"language", "analysisType", "code"}.issubset(response):
        try:
            GeneratedAnalysisPlan.model_validate(response)
        except (ValidationError, TypeError, ValueError):
            return "generated_analysis_schema_validation"
        return None
    # Typed AnalysisPlan materialization has no ScientificAction envelope.
    if action_name is None and "analysis_type" in response:
        try:
            typed_plan = AnalysisPlan.model_validate(response)
        except (ValidationError, TypeError, ValueError):
            return "schema_validation"
        expected_type = _ACTION_ANALYSIS_TYPES.get(
            (context.get("actionName") or context.get("workflow"))
            if isinstance(context, dict) else None
        )
        if expected_type is not None and typed_plan.analysis_type != expected_type:
            return "analysis_type_mismatch"
        allowed_observations = set(
            context.get("sourceObservationIds", [])
            if isinstance(context, dict) else []
        )
        if allowed_observations and not set(typed_plan.source_observation_ids).issubset(
            allowed_observations
        ):
            return "unknown_observation"
        allowed_fields = set(
            context.get("availableSemanticFields", [])
            if isinstance(context, dict) else []
        )
        referenced_fields = {
            value for value in (
                typed_plan.outcome,
                typed_plan.feature_field,
                typed_plan.group_field,
                typed_plan.validation_field,
                *typed_plan.covariates,
                *typed_plan.stratify_by,
            ) if value is not None
        }
        if allowed_fields and not referenced_fields.issubset(allowed_fields):
            return "invalid_semantic_field_or_relation"
        required_fields = set(
            context.get("requiredSemanticFields", [])
            if isinstance(context, dict) else []
        )
        if required_fields and not required_fields.issubset(referenced_fields):
            return "required_field"
        return None
    if not isinstance(action_name, str):
        return "missing_action_name"
    if isinstance(approved, list) and action_name not in approved:
        return "action_mismatch"
    try:
        # Runtime owns the correlation identifier and replaces the model's
        # display value before contract validation.  Mirror that boundary in
        # the audit so a harmless model label is not misclassified as a plan
        # schema failure.
        action_for_validation = dict(response)
        action_for_validation["actionId"] = "action-" + "0" * 32
        validate_scientific_action(action_for_validation)
    except (ValidationError, TypeError, ValueError):
        return "schema_validation"
    if not isinstance(arguments, dict):
        return "required_field"
    if arguments.get("actionName") != action_name:
        return "action_mismatch"
    if action_name in {"execute_read_query", "inspect_cohort"}:
        plan = arguments.get("queryPlan")
        if not isinstance(plan, dict):
            return "required_field"
        try:
            parsed_plan = QueryPlan.model_validate(plan)
        except (ValidationError, TypeError, ValueError):
            return "schema_validation"
        if catalog is not None:
            try:
                validate_query_plan_catalog(parsed_plan, catalog)
            except (ValidationError, TypeError, ValueError):
                return "invalid_semantic_field_or_relation"
    elif action_name in {
        "compare_groups",
        "stratified_analysis",
        "adjust_confounders",
        "cross_project_validate",
        "cross_disease_validate",
    }:
        if not isinstance(arguments.get("observationIds"), list):
            return "required_field"
    return None


def _materializer_payload_records(
    transport: RecordingTransport,
    *,
    catalog: SchemaSemanticCatalog | None = None,
) -> list[dict[str, Any]]:
    """Return decoded model attempts and their closed-contract audit labels."""

    result: list[dict[str, Any]] = []
    call_number = 0
    attempt_by_call: dict[int, int] = {}
    for record in transport.records:
        body = record.get("request", {}).get("body", {})
        messages = body.get("messages", []) if isinstance(body, dict) else []
        context = _decode_content(messages[1].get("content")) if len(messages) > 1 else None
        # Every plan_action/generate_typed_analysis HTTP call starts with the
        # system + user pair. Subsequent messages are bounded contract-repair
        # attempts appended by the real provider adapter.
        if len(messages) <= 2:
            call_number += 1
            attempt_by_call[call_number] = 0
        attempt_by_call[call_number] = attempt_by_call.get(call_number, 0) + 1
        response_body = record.get("response", {})
        try:
            content = response_body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            content = None
        # A successful HTTP response is a model-response opportunity even if
        # its body is malformed or missing the expected choices/content.  Only
        # transport/status failures are excluded from first-model-response
        # contract scoring.
        model_response = record.get("response_status") == 200
        raw_response = _decode_content(content)
        normalized_response = _normalise_materializer_response(raw_response)
        if isinstance(context, dict) and isinstance(normalized_response, dict) \
                and "analysis_type" not in normalized_response:
            # The provider and Runtime share this lossless action-shape
            # normalization. Use a tiny context view here so first-pass audit
            # statistics reflect the actual closed contract (without
            # reconstructing or changing any scientific values).
            class _ActionContextView:
                questionSummary = context.get("questionSummary", "")

            normalized_response, _ = _canonicalize_scientific_action_shape(
                normalized_response,
                _ActionContextView(),  # type: ignore[arg-type]
            )
        context_action = (
            context.get("actionName") or context.get("workflow")
            if isinstance(context, dict) else None
        )
        response_action = (
            normalized_response.get("actionName")
            if isinstance(normalized_response, dict) else None
        )
        contract_kind = (
            "typed_analysis_plan"
            if isinstance(normalized_response, dict)
            and "analysis_type" in normalized_response
            and "source_observation_ids" in normalized_response
            else "generated_analysis_plan"
            if isinstance(normalized_response, dict)
            and {"language", "analysisType", "code"}.issubset(normalized_response)
            else "scientific_action"
            if isinstance(normalized_response, dict)
            and "actionName" in normalized_response
            else "unknown"
        )
        failure_reason = _materializer_failure_reason(
            status=record.get("response_status"),
            context=context,
            response=normalized_response,
            catalog=catalog,
        )
        result.append({
            "request_index": record.get("request_index"),
            "materializer_call": call_number,
            "attempt": attempt_by_call[call_number],
            "attempt_count_in_request": max(0, len(messages) - 1),
            "status": record.get("response_status"),
            # Keep provider transport from being scored as a model contract
            # failure.  A 200 response with malformed JSON is still a model
            # response and is therefore eligible for first-response contract
            # scoring; RemoteProtocolError/timeout/429 records are not.
            "model_response": model_response,
            "transport_error": record.get("error"),
            "repair_request": len(messages) > 2,
            "execution_feedback": (
                context.get("executionFeedback", [])
                if isinstance(context, dict) else []
            ),
            "action_name": context_action or response_action,
            "contract_kind": contract_kind,
            "planner_method": (
                "generate_typed_analysis"
                if contract_kind == "typed_analysis_plan"
                else "generate_analysis"
                if contract_kind == "generated_analysis_plan"
                else "plan_action"
                if contract_kind == "scientific_action"
                else None
            ),
            "expected_analysis_type": _ACTION_ANALYSIS_TYPES.get(context_action),
            "approved_actions": (
                context.get("approvedActions", [])
                if isinstance(context, dict) else []
            ),
            "context": context,
            "response": raw_response,
            "normalized_response": normalized_response,
            "failure_reason": failure_reason,
            "query_plan": (
                normalized_response.get("arguments", {}).get("queryPlan")
                if isinstance(normalized_response, dict)
                and isinstance(normalized_response.get("arguments"), dict)
                else None
            ),
            "analysis_plan": (
                normalized_response
                if isinstance(normalized_response, dict)
                and "analysis_type" in normalized_response
                else None
            ),
        })
    return result


def _materializer_call_audit(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate attempts by one provider call for first-pass statistics."""

    grouped: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(int(record.get("materializer_call") or 0), []).append(record)
    result: list[dict[str, Any]] = []
    for call_number in sorted(grouped):
        attempts = grouped[call_number]
        first = attempts[0]
        final = attempts[-1]
        # Older frozen canary fixtures predate the explicit transport/model
        # split.  Their status=200 records are model-response attempts; infer
        # that value only when the new field is absent so historical tests and
        # traces remain readable under the corrected accounting.
        def is_model_response(item: dict[str, Any]) -> bool:
            value = item.get("model_response")
            return bool(value) if value is not None else item.get("status") == 200

        def is_repair_request(item: dict[str, Any]) -> bool:
            value = item.get("repair_request")
            return bool(value) if value is not None else int(item.get("attempt") or 1) > 1

        model_responses = [item for item in attempts if is_model_response(item)]
        first_model = model_responses[0] if model_responses else None
        final_model = model_responses[-1] if model_responses else None
        transport_failures = [
            item for item in attempts if not is_model_response(item)
        ]
        result.append({
            "materializer_call": call_number,
            "action_name": first.get("action_name"),
            "contract_kind": first.get("contract_kind"),
            "planner_method": first.get("planner_method"),
            "final_contract_kind": final.get("contract_kind"),
            "final_planner_method": final.get("planner_method"),
            "first_model_contract_kind": (
                first_model.get("contract_kind") if first_model else None
            ),
            "first_model_planner_method": (
                first_model.get("planner_method") if first_model else None
            ),
            "final_model_contract_kind": (
                final_model.get("contract_kind") if final_model else None
            ),
            "final_model_planner_method": (
                final_model.get("planner_method") if final_model else None
            ),
            "expected_analysis_type": first.get("expected_analysis_type"),
            "approved_actions": first.get("approved_actions", []),
            "attempt_count": len(attempts),
            "repair_count": max(0, len(attempts) - 1),
            "transport_attempts": len(attempts),
            "transport_failure_count": len(transport_failures),
            "transport_failure_reasons": list(dict.fromkeys(
                str(item.get("transport_error") or item.get("failure_reason"))
                for item in transport_failures
            )),
            "model_response_count": len(model_responses),
            "model_responses": [
                {
                    "request_index": item.get("request_index"),
                    "attempt": item.get("attempt"),
                    "contract_kind": item.get("contract_kind"),
                    "failure_reason": item.get("failure_reason"),
                    "repair_request": item.get("repair_request", False),
                }
                for item in model_responses
            ],
            "contract_repair_count": sum(is_repair_request(item) for item in attempts),
            # These names are retained for consumers of the historical audit,
            # but now mean the first *successful model response*, never the
            # first transport attempt.
            "first_pass_success": (
                first_model is not None and first_model.get("failure_reason") is None
            ),
            "first_model_response_valid": (
                None
                if first_model is None
                else first_model.get("failure_reason") is None
            ),
            "first_model_response_failure_reason": (
                first_model.get("failure_reason") if first_model else None
            ),
            "first_failure_reason": (
                first_model.get("failure_reason") if first_model else None
            ),
            "attempts": [
                {
                    "request_index": item.get("request_index"),
                    "attempt": item.get("attempt"),
                    "status": item.get("status"),
                    "failure_reason": item.get("failure_reason"),
                    "model_response": is_model_response(item),
                    "transport_error": item.get("transport_error"),
                    "repair_request": is_repair_request(item),
                    "execution_feedback": item.get("execution_feedback", []),
                }
                for item in attempts
            ],
            "final_failure_reason": (
                final_model.get("failure_reason") if final_model else "no_model_response"
            ),
            "final_model_response_valid": (
                final_model is not None and final_model.get("failure_reason") is None
            ),
            "final_model_response_failure_reason": (
                final_model.get("failure_reason") if final_model else "no_model_response"
            ),
            "final_response": (
                final_model.get("normalized_response") if final_model else None
            ),
        })
    return result


def _policy_records(transport: RecordingTransport) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for record in transport.records:
        body = record.get("request", {}).get("body", {})
        messages = body.get("messages", []) if isinstance(body, dict) else []
        payload = _decode_content(messages[1].get("content")) if len(messages) > 1 else None
        response = record.get("response", {})
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            content = None
        result.append({
            "request_index": record.get("request_index"),
            "status": record.get("response_status"),
            "state": payload.get("state") if isinstance(payload, dict) else None,
            "response": _decode_content(content),
        })
    return result


def _query_plan_fingerprint(value: object) -> str | None:
    """Fingerprint a canonical QueryPlan, including its default schema version."""

    if not isinstance(value, dict):
        return None
    try:
        canonical = QueryPlan.model_validate(value).model_dump(mode="json")
    except (ValidationError, TypeError, ValueError):
        return None
    return "sha256:" + sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _query_plan_semantic_fingerprint(value: object) -> str | None:
    """Fingerprint a plan after removing Runtime-owned sample-bound fields."""

    if not isinstance(value, dict):
        return None
    semantic = dict(value)
    semantic.pop("sample_limit_per_group", None)
    semantic.pop("sample_limit_group_field", None)
    # The Runtime raises the model's row LIMIT to the approved sample-bound
    # cap for raw abundance projections.  LIMIT is still retained in the
    # execution audit, but it is not a stable key for joining the model plan
    # to the effective Java call.
    selected = set(semantic.get("select_fields", []) or [])
    if (
        not semantic.get("aggregations")
        and not semantic.get("group_by")
        and {"abundance.feature", "abundance.value"}.issubset(selected)
    ):
        semantic["limit"] = 20_000
    return _query_plan_fingerprint(semantic)


def _trace_rounds(final_state: dict[str, Any], records: list[Any]) -> list[dict[str, Any]]:
    observations = list(final_state.get("observations", []))
    snapshots = [record.state_snapshot for record in records if record.state_snapshot]
    final_snapshot = _state_snapshot(final_state.get("decisionState"))
    all_states = snapshots + ([final_snapshot] if final_snapshot else [])
    rounds: list[dict[str, Any]] = []
    observation_index = 0
    for index, record in enumerate(records):
        observation = None
        for candidate in observations[observation_index:]:
            if candidate.actionName == record.selected_action:
                observation = candidate
                observation_index = observations.index(candidate) + 1
                break
        after = all_states[index + 1] if index + 1 < len(all_states) else final_snapshot
        resolution_before = getattr(record, "objective_resolution", None)
        if index + 1 < len(records):
            resolution_after = getattr(records[index + 1], "objective_resolution", None)
        else:
            resolution_after = final_state.get("objectiveResolution")
        rounds.append({
            "turn": index + 1,
            "state_before": record.state_snapshot,
            "selected_action": record.selected_action,
            "decision_reason": record.decision_reason,
            "policy_origin": record.policy_origin,
            "materializer_origin": record.materializer_origin,
            "observation_summary": observation.model_dump(mode="json") if observation else None,
            "state_after": after,
            "objective_resolution_before": resolution_before,
            "objective_resolution_after": resolution_after,
        })
    return rounds


def _query_execution_audit(
    final_state: dict[str, Any],
    materializer_records: list[dict[str, Any]],
    java_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join each *executed* Java read with its semantic plan and observation.

    A Gemini materializer can receive bounded contract-repair retries.  The
    old implementation paired every raw response attempt with the next Java
    observation, which could report the metadata ``inspect_cohort`` result as
    the result of a later model QueryPlan.  The Java call list is the
    authoritative execution sequence; model plans are joined by their
    canonical fingerprint when available.
    """

    observations = [
        item for item in final_state.get("observations", [])
        if getattr(item, "source", None) == "java_controlled_read"
    ]
    calls = [
        item for item in java_calls
        if item.get("tool_name") == "execute_read_query"
    ]
    model_plans: dict[str, list[dict[str, Any]]] = {}
    model_semantic_plans: dict[str, list[dict[str, Any]]] = {}
    for record in materializer_records:
        action = record.get("normalized_response") or record.get("response")
        if not isinstance(action, dict) or action.get("actionName") != "execute_read_query":
            continue
        query_plan = record.get("query_plan")
        if not isinstance(query_plan, dict):
            continue
        fingerprint = _query_plan_fingerprint(query_plan)
        if fingerprint is None:
            continue
        model_plans.setdefault(fingerprint, []).append({
            "materializer_request_index": record.get("request_index"),
            "materializer_call": record.get("materializer_call"),
            "query_plan": query_plan,
        })
        semantic_fingerprint = _query_plan_semantic_fingerprint(query_plan)
        if semantic_fingerprint is not None:
            model_semantic_plans.setdefault(semantic_fingerprint, []).append({
                "materializer_request_index": record.get("request_index"),
                "materializer_call": record.get("materializer_call"),
                "query_plan": query_plan,
            })
    result: list[dict[str, Any]] = []
    for index, call in enumerate(calls):
        arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        query_plan = arguments.get("queryPlan")
        if not isinstance(query_plan, dict):
            # Dynamic Happy Path must never send the legacy SQL form.  Keep a
            # record of a malformed/legacy call for readiness diagnostics.
            result.append({
                "java_call_index": call.get("call_index"),
                "selected_action": (
                    observations[index].actionName
                    if index < len(observations) else None
                ),
                "query_plan": None,
                "materializer_request_index": None,
                "query_plan_fingerprint": None,
                "compiled_query_fingerprint": None,
                "compiled_sql_fingerprint": (
                    (call.get("query_result_profile") or {}).get("compiled_sql_fingerprint")
                ),
                "compiled_sql": "not_exposed_by_java_contract; Java compiler owns SQL",
                "sql_parameters": [],
                "query_plan_repair_count": None,
                "query_result_profile": call.get("query_result_profile"),
                "query_result_summary": (
                    observations[index].model_dump(mode="json")
                    if index < len(observations) else None
                ),
            })
            continue
        fingerprint = _query_plan_fingerprint(query_plan)
        model_match = model_plans.get(fingerprint, [])
        if not model_match:
            model_match = model_semantic_plans.get(
                _query_plan_semantic_fingerprint(query_plan) or "", []
            )
        model_record = model_match[0] if model_match else None
        model_call_records = [
            item for item in materializer_records
            if model_record is not None
            and item.get("materializer_call") == model_record.get("materializer_call")
        ]
        observation = observations[index] if index < len(observations) else None
        result.append({
            "java_call_index": call.get("call_index"),
            "materializer_request_index": (
                model_record.get("materializer_request_index")
                if model_record else None
            ),
            "selected_action": (
                observation.actionName if observation is not None
                else "unknown_java_read"
            ),
            "query_plan": query_plan,
            "model_query_plan": (
                model_record.get("query_plan") if model_record else None
            ),
            "runtime_sample_bound_applied": bool(
                query_plan.get("sample_limit_per_group") is not None
            ),
            "query_plan_fingerprint": fingerprint,
            # Java's DataSnapshot.queryHash is the hash of the compiled SQL;
            # keep it separate from the plan fingerprint used to join the
            # Gemini response to the executed call.
            "compiled_query_fingerprint": (
                (call.get("query_result_profile") or {}).get("compiled_sql_fingerprint")
            ),
            "compiled_sql_fingerprint": (
                (call.get("query_result_profile") or {}).get("compiled_sql_fingerprint")
            ),
            "compiled_sql": "not_exposed_by_java_contract; Java compiler owns SQL",
            "sql_parameters": call.get("query_result_profile", {}).get("sql_parameters", []),
            "query_plan_repair_count": (
                max(0, len(model_call_records) - 1) if model_call_records else None
            ),
            "query_result_profile": call.get("query_result_profile"),
            "query_result_summary": (
                observation.model_dump(mode="json") if observation is not None else None
            ),
            "plan_origin": "gemini_canary_model" if model_record else "runtime_or_unknown",
        })
    return result


def _analysis_execution_audit(
    final_state: dict[str, Any],
    materializer_records: list[dict[str, Any]],
    query_executions: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Expose the Gemini v2 AnalysisPlan alongside the real result.

    ``analysisPlans`` is the historical execution bookkeeping object.  The
    model-authored v2 plan crosses the Gemini materializer boundary directly,
    so it is recovered from the recorded typed-materializer response and
    joined with the resulting AnalysisResult here.
    """

    plans = [item.model_dump(mode="json") for item in final_state.get("analysisPlans", [])]
    results = [item.model_dump(mode="json") for item in final_state.get("analysisResults", [])]
    observations = {
        item.observationId: item
        for item in final_state.get("observations", [])
        if getattr(item, "observationId", None)
    }
    query_by_observation: dict[str, dict[str, Any]] = {}
    for query in query_executions or []:
        summary = query.get("query_result_summary")
        if isinstance(summary, dict) and isinstance(summary.get("observationId"), str):
            query_by_observation[summary["observationId"]] = query

    def source_audit(source_ids: object) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        if not isinstance(source_ids, list):
            return result
        payloads = final_state.get("rawObservationPayloads", {})
        for observation_id in source_ids:
            observation = observations.get(observation_id)
            if observation is None:
                result.append({
                    "observation_id": observation_id,
                    "status": "not_found",
                })
                continue
            payload = payloads.get(observation_id) if isinstance(payloads, dict) else None
            # Prefer the query execution audit's authoritative plan/profile.
            # Reconstructing from raw payload alone loses aggregation and
            # feature-dimension semantics, which would make a grouped
            # abundance query look like a raw row observation.
            query_record = query_by_observation.get(observation_id)
            profile = (
                query_record.get("query_result_profile")
                if isinstance(query_record, dict)
                and isinstance(query_record.get("query_result_profile"), dict)
                else _query_result_profile(
                    payload,
                    None,
                    authoritative_row_count=observation.rowCount,
                    query_hash=observation.queryHash,
                )
            )
            result.append({
                "observation_id": observation_id,
                "action_name": observation.actionName,
                "status": observation.status,
                "row_count": observation.rowCount,
                "query_plan_fields": list(observation.queryPlanFields),
                "query_plan_relation_path": list(observation.queryPlanRelationPath),
                "distinct_group_values": profile.get("distinct_disease_values", []),
                "group_value_counts": profile.get("disease_value_counts", {}),
                "distinct_project_count": profile.get("distinct_project_count", 0),
                "distinct_sample_count": profile.get("distinct_sample_count"),
                "distinct_feature_count": profile.get("distinct_abundance_feature_count", 0),
                "sample_count_source": profile.get("sample_count_source"),
                "group_size_semantics": profile.get("group_size_semantics"),
                "is_aggregation_result": profile.get("is_aggregation_result"),
                "feature_pooling_risk": profile.get("feature_pooling_risk"),
                "query_plan": (
                    query_record.get("query_plan")
                    if isinstance(query_record, dict) else None
                ),
            })
        return result

    def semantic_validity(plan: object, sources: list[dict[str, Any]]) -> str:
        """Mark analysis results pending when the source mixes features.

        This is deliberately an audit label, not a new Action or a change to
        the typed operator.  A single-outcome group/adjustment result over an
        abundance projection without a feature selector cannot be promoted
        to a scientific conclusion until its analysis unit is resolved.
        """

        if not isinstance(plan, dict):
            return "not_evaluated"
        if plan.get("analysis_type") not in {
            "group_comparison",
            "confounder_adjustment",
        }:
            return "not_evaluated"
        if any(bool(source.get("feature_pooling_risk")) for source in sources):
            return "pending"
        # Multiple features are safe when the AnalysisPlan explicitly binds
        # the feature dimension and the typed operator returns per-feature
        # results.  Only an unbound plan over a fan-out projection is pooled.
        if (
            not plan.get("feature_field")
            and any(int(source.get("distinct_feature_count") or 0) > 1 for source in sources)
        ):
            return "pending"
        return "validated"

    # One typed-materializer provider call may emit several HTTP requests while
    # repairing its closed AnalysisPlan contract.  Aggregate those attempts
    # so a retry is never mistaken for a second analysis execution.
    typed_groups: dict[int, list[dict[str, Any]]] = {}
    for record in materializer_records:
        response = record.get("normalized_response") or record.get("response")
        if not isinstance(response, dict) or "analysis_type" not in response:
            continue
        if "source_observation_ids" not in response:
            continue
        typed_groups.setdefault(int(record.get("materializer_call") or 0), []).append(record)
    if typed_groups:
        result_items: list[dict[str, Any]] = []
        result_index = 0
        for call_number in sorted(typed_groups):
            attempts = typed_groups[call_number]
            final_record = attempts[-1]
            successful_records = [
                item for item in attempts if item.get("failure_reason") is None
            ]
            selected_record = successful_records[-1] if successful_records else final_record
            plan = selected_record.get("normalized_response") or selected_record.get("response")
            plan = plan if isinstance(plan, dict) else None
            analysis_result = (
                results[result_index] if successful_records and result_index < len(results) else None
            )
            if successful_records:
                result_index += 1
            source_observations = source_audit(
                plan.get("source_observation_ids", []) if isinstance(plan, dict) else []
            )
            validity = semantic_validity(plan, source_observations)
            result_items.append({
                "materializer_call": call_number,
                "materializer_request_index": selected_record.get("request_index"),
                "action_name": selected_record.get("action_name"),
                "expected_analysis_type": selected_record.get("expected_analysis_type"),
                "materializer_attempt_count": len(attempts),
                "materializer_repair_count": max(0, len(attempts) - 1),
                "first_failure_reason": attempts[0].get("failure_reason"),
                "attempts": [
                    {
                        "request_index": item.get("request_index"),
                        "attempt": item.get("attempt"),
                        "failure_reason": item.get("failure_reason"),
                    }
                    for item in attempts
                ],
                "analysis_plan": plan,
                "source_observations": source_observations,
                "analysis_result": analysis_result,
                "semantic_validity": validity,
                "scientific_conclusion_eligible": validity == "validated",
                "capability_match": (
                    "typed" if analysis_result and analysis_result.get("codeVersion") == "typed-analysis-operator-v1"
                    else "generated" if analysis_result and analysis_result.get("codeVersion") == "sandbox-python-v1"
                    else "not_executed"
                ),
                "execution_mode": analysis_result.get("execution_mode") if analysis_result else None,
                "code_version": analysis_result.get("codeVersion") if analysis_result else None,
            })
        return result_items
    # Preserve visibility of an execution-layer plan even if a provider failed
    # before returning a v2 model plan.
    return [
        {
            "materializer_call": None,
            "materializer_request_index": None,
            "analysis_plan": plan,
            "source_observations": source_audit(plan.get("sourceObservationIds", [])),
            "analysis_result": results[index] if index < len(results) else None,
            "semantic_validity": semantic_validity(
                plan,
                source_audit(plan.get("sourceObservationIds", [])),
            ),
            "scientific_conclusion_eligible": semantic_validity(
                plan,
                source_audit(plan.get("sourceObservationIds", [])),
            ) == "validated",
            "capability_match": (
                "typed" if index < len(results)
                and results[index].get("codeVersion") == "typed-analysis-operator-v1"
                else "generated" if index < len(results)
                and results[index].get("codeVersion") == "sandbox-python-v1"
                else "not_executed"
            ),
            "execution_mode": results[index].get("execution_mode") if index < len(results) else None,
            "code_version": results[index].get("codeVersion") if index < len(results) else None,
        }
        for index, plan in enumerate(plans)
    ]


def _query_semantics_audit(
    query_executions: list[dict[str, Any]],
    analysis_executions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Produce the 4D-1C semantic diagnosis from real query profiles."""

    profiles = [
        item.get("query_result_profile")
        for item in query_executions
        if isinstance(item.get("query_result_profile"), dict)
    ]
    observed_diseases: set[str] = set()
    observed_projects: set[str] = set()
    for profile in profiles:
        observed_diseases.update(profile.get("distinct_disease_values", []))
        observed_projects.update(profile.get("distinct_project_values", []))
    user_labels = {"T2D", "Healthy"}
    casefold_labels = {value.casefold() for value in observed_diseases}
    # The relation path is retained in each query execution, not in the
    # profile's grouped fields.  Keep this diagnostic conservative and derive
    # the concrete join observation from the execution record below.
    metadata_filtered_empty = any(
        item.get("query_result_profile", {}).get("row_count") == 0
        and "sample_to_metadata" in item.get("query_plan", {}).get("relation_path", [])
        and bool(item.get("query_plan", {}).get("filters"))
        for item in query_executions
    )
    multi_feature = any(
        int(profile.get("distinct_abundance_feature_count") or 0) > 1
        for profile in profiles
    )
    feature_pooling = any(
        bool(profile.get("feature_pooling_risk"))
        for profile in profiles
    )
    # The profile cannot observe feature values when Java was asked to return
    # only an aggregate.  In that case the QueryPlan itself is authoritative:
    # an abundance.value aggregate without abundance.feature in the grouping
    # has already pooled the feature dimension.
    feature_pooling_plans = [
        item.get("query_plan")
        for item in query_executions
        if isinstance(item.get("query_plan"), dict)
        and any(
            isinstance(aggregation, dict)
            and aggregation.get("field") == "abundance.value"
            for aggregation in item["query_plan"].get("aggregations", [])
        )
        and "abundance.feature" not in set(item["query_plan"].get("select_fields", []))
        and "abundance.feature" not in set(item["query_plan"].get("group_by", []))
    ]
    group_plan = next(
        (
            item.get("analysis_plan")
            for item in analysis_executions
            if isinstance(item.get("analysis_plan"), dict)
            and item["analysis_plan"].get("analysis_type") == "group_comparison"
        ),
        None,
    )
    feature_aware_group_result = bool(
        group_plan
        and group_plan.get("feature_field")
        and any(
            isinstance(item.get("analysis_result"), dict)
            and bool(item["analysis_result"].get("feature_results"))
            for item in analysis_executions
            if isinstance(item.get("analysis_plan"), dict)
            and item["analysis_plan"].get("analysis_type") == "group_comparison"
        )
    )
    unresolved_feature_pooling = bool(
        feature_pooling
        or feature_pooling_plans
        or (multi_feature and not feature_aware_group_result)
    )
    query_diagnostics = [
        {
            "java_call_index": item.get("java_call_index"),
            "materializer_request_index": item.get("materializer_request_index"),
            "selected_action": item.get("selected_action"),
            "query_plan": item.get("query_plan"),
            "query_plan_fingerprint": item.get("query_plan_fingerprint"),
            "query_plan_repair_count": item.get("query_plan_repair_count"),
            "compiled_sql": item.get("compiled_sql"),
            "compiled_query_fingerprint": item.get("compiled_query_fingerprint"),
            "compiled_sql_fingerprint": item.get("compiled_sql_fingerprint"),
            "sql_parameters": item.get("sql_parameters", []),
            "query_result_profile": item.get("query_result_profile"),
        }
        for item in query_executions
    ]
    empty_due_to_metadata_join = any(
        item.get("query_result_profile", {}).get("row_count") == 0
        and "sample_to_metadata" in item.get("query_plan", {}).get("relation_path", [])
        and bool(item.get("query_plan", {}).get("filters"))
        for item in query_executions
    )
    observed_requested_labels = sorted(
        label for label in user_labels if label.casefold() in casefold_labels
    )
    root_causes: list[str] = []
    if empty_due_to_metadata_join:
        root_causes.append(
            "filtered_metadata_join_returned_zero_rows; verify join coverage and raw disease-label vocabulary"
        )
    if profiles and all(
        int(profile.get("distinct_disease_count", len(profile.get("distinct_disease_values", [])))) < 2
        for profile in profiles
    ):
        root_causes.append(
            "observed_query_profiles_have_fewer_than_two_disease_values; result-row LIMIT/order or label coverage may truncate groups"
        )
    if (
        not feature_aware_group_result
        and any(
            int(profile.get("distinct_abundance_feature_count") or 0) > 1
            for profile in profiles
        )
    ):
        root_causes.append(
            "abundance_feature_fanout_present; current typed compare_groups would pool feature values"
        )
    if feature_pooling_plans:
        root_causes.append(
            "abundance_value_aggregated_without_feature_dimension; feature values were pooled before compare_groups"
        )
    if not observed_requested_labels and profiles:
        root_causes.append(
            "requested_labels_not_observed_in_canary_profiles; catalog exposes no canonical-value/alias mapping"
        )
    return {
        "query_count": len(query_executions),
        "query_diagnostics": query_diagnostics,
        "observed_disease_values": sorted(observed_diseases),
        "observed_project_values": sorted(observed_projects),
        "requested_labels": sorted(user_labels),
        "requested_labels_casefold_present": observed_requested_labels,
        "limit_semantics": "result_rows",
        "sample_count_status": (
            profiles[0].get("sample_count_source")
            if profiles else "no_query_profile"
        ),
        "group_size_semantics": (
            profiles[0].get("group_size_semantics")
            if profiles else "no_query_profile"
        ),
        "metadata_join_filtered_empty_observation": metadata_filtered_empty,
        "metadata_filtered_empty_observation": metadata_filtered_empty,
        "root_causes": root_causes,
        "multi_feature_observation": multi_feature,
        "feature_pooling_from_query_plan": bool(feature_pooling_plans),
        "feature_pooling_query_plans": feature_pooling_plans,
        "feature_aware_group_result": feature_aware_group_result,
        "group_comparison_feature_mixing_risk": bool(group_plan and unresolved_feature_pooling),
        # This is an audit/provenance flag, not a Decision State field.  A
        # bounded p-value must not be treated as a scientific conclusion when
        # an abundance feature dimension was pooled or the analysis unit is
        # not sample-resolved.  Keep the raw AnalysisResult for debugging,
        # but mark its scientific eligibility explicitly.
        "analysis_semantics_validated": not bool(group_plan and unresolved_feature_pooling),
        "analysis_semantics_status": (
            "pending"
            if group_plan and unresolved_feature_pooling
            else "validated" if group_plan else "not_evaluated"
        ),
        "analysis_scientific_conclusion_eligible": bool(
            group_plan and not unresolved_feature_pooling
        ),
        "group_comparison_plan": group_plan,
        "group_comparison_source_observations": (
            next(
                (
                    item.get("source_observations", [])
                    for item in analysis_executions
                    if isinstance(item.get("analysis_plan"), dict)
                    and item["analysis_plan"].get("analysis_type") == "group_comparison"
                ),
                [],
            )
        ),
        "analysis_success": any(
            item.get("analysis_result") is not None for item in analysis_executions
        ),
    }


def run_happy(out_dir: Path, *, max_actions: int = 8) -> dict[str, Any]:
    """Run the real Gemini → Java → typed Python loop with no test doubles.

    ``max_actions`` is an explicit canary budget only.  The default preserves
    the existing eight-action run; a smaller value is useful for a short
    contract E2E after an offline materializer probe, without changing the
    production runtime policy or action semantics.
    """

    _clear_previous_artifacts(out_dir)
    # Hard gate: this performs only TCP/MySQL and Java read-model checks.  It
    # must complete before constructing any Gemini role or sending a model
    # request, otherwise an unhealthy chain would burn API quota.
    preflight = asyncio.run(run_preflight())
    _dump(out_dir / "preflight.json", preflight)
    if preflight.get("status") != "pass":
        trace = {
            "runtime_status": "FAILED",
            "error_code": "CHAIN_PREFLIGHT_FAILED",
            "training_eligible": False,
            "gemini_request_budget": {"total_used": 0},
            "policy_request_count": 0,
            "materializer_request_count": 0,
            "preflight": preflight,
        }
        _dump(out_dir / "trace.json", trace)
        return {
            "final_state": {},
            "catalog": None,
            "trace": trace,
            "task_understanding_records": [],
            "policy_transport": RecordingTransport("scientific_policy"),
            "materializer_transport": RecordingTransport("materializer"),
            "run_error": "CHAIN_PREFLIGHT_FAILED",
        }
    token = _gemini_token()
    model = os.environ.get("MICO_GEMINI_MODEL", GEMINI_MODEL).strip() or GEMINI_MODEL
    base_url = os.environ.get("MICO_GEMINI_BASE_URL", GEMINI_BASE_URL).strip() or GEMINI_BASE_URL

    tu_transport = RecordingTransport("task_understanding")
    policy_transport = RecordingTransport("scientific_policy")
    materializer_transport = RecordingTransport("materializer")
    gemini_budget = GeminiRequestBudget(
        # Keep the default below Gemini's documented free-tier 15 RPM budget
        # for a single canary burst; operators can raise it explicitly when
        # using a paid/quota-increased project.
        total_limit=int(os.environ.get("MICO_GEMINI_RUN_MAX_REQUESTS", "15")),
        role_limits={
            "task_understanding": int(os.environ.get("MICO_GEMINI_TU_MAX_REQUESTS", "1")),
            "policy": int(os.environ.get("MICO_GEMINI_POLICY_MAX_REQUESTS", "6")),
            "materializer": int(os.environ.get("MICO_GEMINI_MATERIALIZER_MAX_REQUESTS", "8")),
        },
    )
    gemini_cache = GeminiResponseCache(
        ttl_seconds=float(os.environ.get("MICO_GEMINI_RESPONSE_CACHE_TTL", "30")),
    )
    task_env = dict(os.environ)
    task_env.update({
        "MICO_TASK_UNDERSTANDING_ENABLED": "true",
        "MICO_TASK_UNDERSTANDING_BASE_URL": base_url,
        "MICO_TASK_UNDERSTANDING_MODEL": model,
        "MICO_TASK_UNDERSTANDING_TOKEN": token,
    })
    task_understanding = build_task_understanding_port(
        task_env,
        transport=tu_transport,
        request_budget=gemini_budget,
        response_cache=gemini_cache,
    )
    policy = HttpDecisionSftPlannerPort(
        base_url,
        model,
        token=token,
        transport=policy_transport,
        policy_origin="gemini_canary_model",
        request_budget=gemini_budget,
        response_cache=gemini_cache,
    )
    materializer = HttpResearchPlannerPort(
        base_url,
        model,
        token,
        transport=materializer_transport,
        materializer_origin="gemini_canary_model",
        request_budget=gemini_budget,
        response_cache=gemini_cache,
    )
    planner = HybridIntentPlannerPort(
        materializer,
        policy,
        allow_deterministic_materializer_fallback=False,
    )
    java = RecordingJavaPort(HttpJavaAgentToolPort.from_environment())
    request = _task(max_actions=max_actions)
    catalog: SchemaSemanticCatalog | None = None
    final_state: dict[str, Any] = {}
    run_error: str | None = None
    try:
        # This is a real Java metadata call; no catalog is synthesized by the
        # canary.  The same catalog is passed to the runtime/compiler path.
        catalog = JavaSchemaCatalogPort(java).load(run_id=request.runId, task_id=request.taskId)
        _dump(out_dir / "semantic_catalog.json", catalog)
        runtime = ScientificRuntime(
            java,
            planner,
            knowledge_port=None,
            schema_catalog=catalog,
            task_understanding_port=task_understanding,
        )
        final_state = _initial_graph_state(runtime, request, catalog)
    except Exception as exc:
        run_error = f"{type(exc).__name__}:{exc}"
    finally:
        try:
            planner.close()
        except Exception:
            pass
        close_tu = getattr(task_understanding, "close", None)
        if callable(close_tu):
            close_tu()
        java.close()

    _save_gemini_records(out_dir, tu_transport, "tu")
    _save_gemini_records(out_dir, policy_transport, "policy")
    _save_gemini_records(out_dir, materializer_transport, "materializer")
    policy_records = _policy_records(policy_transport)
    materializer_records = _materializer_payload_records(
        materializer_transport,
        catalog=catalog,
    )
    materializer_call_audit = _materializer_call_audit(materializer_records)
    records = list(final_state.get("decisionRecords", []))
    snapshots = [record.state_snapshot for record in records if record.state_snapshot]
    if final_state:
        final_snapshot = _state_snapshot(final_state.get("decisionState"))
        for index, snapshot in enumerate((snapshots + ([final_snapshot] if final_snapshot else []))[:8]):
            _dump(out_dir / f"state_s{index}.json", snapshot)
    rounds = _trace_rounds(final_state, records) if final_state else []
    query_executions = _query_execution_audit(
        final_state, materializer_records, java.calls,
    )
    analysis_executions = _analysis_execution_audit(
        final_state, materializer_records, query_executions,
    )
    query_semantics_audit = _query_semantics_audit(
        query_executions,
        analysis_executions,
    )
    objective_resolution = (
        final_state.get("objectiveResolution", {})
        if isinstance(final_state.get("objectiveResolution", {}), dict)
        else {}
    )
    decision_state = final_state.get("decisionState")
    remaining_objectives = (
        list(decision_state.progress.remaining_objectives)
        if decision_state is not None
        else []
    )
    completion_semantics = completion_semantics_from_resolution(
        objective_resolution,
        workflow_completed=final_state.get("status") == "COMPLETED",
        remaining_objectives=remaining_objectives,
    )
    _dump(out_dir / "objective_resolution.json", objective_resolution)
    _dump(out_dir / "completion_semantics.json", completion_semantics)
    _dump(out_dir / "materializer_audit.json", materializer_call_audit)
    _dump(out_dir / "query_semantics_audit.json", query_semantics_audit)
    trace = {
        "trace_id": request.traceId,
        "training_eligible": False,
        "task_understanding_origin": "gemini_model",
        "task_understanding_model": model,
        "policy_origin": "gemini_canary_model",
        "materializer_origin": "gemini_canary_model",
        "canary_max_actions": max_actions,
        "preflight": preflight,
        "gemini_request_budget": {
            "total_limit": gemini_budget.total_limit,
            "total_used": gemini_budget.total_used,
            "role_limits": dict(gemini_budget.role_limits),
            "role_used": dict(gemini_budget.role_used),
            "cache_hits": gemini_cache.hits,
        },
        "runtime_status": final_state.get("status") if final_state else "FAILED",
        "error_code": final_state.get("errorCode") if final_state else run_error,
        "planner_failure_detail": final_state.get("plannerFailureDetail") if final_state else None,
        "policy_request_count": len(policy_transport.records),
        "policy_decision_count": len(records),
        "materializer_request_count": len(materializer_transport.records),
        "materializer_call_count": len(materializer_call_audit),
        "materializer_call_audit": materializer_call_audit,
        "java_calls": java.calls,
        "policy_requests": policy_records,
        "materializer_requests": materializer_records,
        "action_history": final_state.get("actionHistory", []),
        "turns": [record.model_dump(mode="json") for record in records],
        "rounds": rounds,
        "query_executions": query_executions,
        "analysis_executions": analysis_executions,
        "objective_resolution": objective_resolution,
        "objectiveResolution": objective_resolution,
        "completion_semantics": completion_semantics,
        "remaining_objectives_invariant": completion_semantics[
            "remaining_objectives_invariant"
        ],
        "query_semantics_audit": query_semantics_audit,
        "analysis_results": [
            item.model_dump(mode="json") for item in final_state.get("analysisResults", [])
        ],
        "fallback_codes": final_state.get("fallbackCodes", []),
        "legacy_strategy_guard_used": any(
            "SEMANTIC" in code for code in final_state.get("fallbackCodes", [])
        ),
        "deterministic_scientific_policy_fallback_used": any(
            record.policy_origin != "gemini_canary_model" for record in records
        ),
        "deterministic_materializer_fallback_used": any(
            record.materializer_origin == "deterministic_fallback" for record in records
        ),
    }
    _dump(out_dir / "trace.json", trace)
    return {
        "final_state": final_state,
        "catalog": catalog,
        "trace": trace,
        "task_understanding_records": tu_transport.records,
        "policy_transport": policy_transport,
        "materializer_transport": materializer_transport,
        "run_error": run_error,
    }


class _FailureServer:
    """Small synthetic HTTP provider used only for fail-closed injections."""

    def __init__(self, responder: Callable[[dict[str, Any], int], tuple[int, dict[str, Any]]]) -> None:
        self.responder = responder
        self.requests: list[dict[str, Any]] = []
        self._server: ThreadingHTTPServer | None = None

    def start(self) -> str:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                owner.requests.append(body)
                status, response = owner.responder(body, len(owner.requests))
                encoded = json.dumps(response).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, *_args: Any) -> None:
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{self._server.server_port}/v1"

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()


def _policy_response(action: str, reason: str = "synthetic failure injection") -> dict[str, Any]:
    return {"choices": [{"message": {"content": json.dumps({
        "selected_action": action,
        "decision_reason": reason,
        "alternative_actions": [],
        "stop_reason": "EVIDENCE_SUFFICIENT" if action == "finish" else None,
    })}}]}


def _failure_state(actions: list[str]) -> ScientificDecisionState:
    return ScientificDecisionState(
        task={"query": CANARY_QUERY},
        action_space={"available_actions": actions},
    )


def _failure_context(catalog: SchemaSemanticCatalog | None) -> ScientificPlannerContext:
    return ScientificPlannerContext(
        questionSummary=CANARY_QUERY,
        intent="scientific_exploration",
        approvedActions=["adjust_confounders"],
        remainingActionBudget=1,
        observations=[ScientificObservationSummary(
            observationId="observation-" + "a" * 32,
            actionName="execute_read_query",
            status="VALIDATED",
            source="java_controlled_read",
            rowCount=2,
            queryPlanFields=["sample.disease", "sample.age", "abundance.value"],
        )],
        schemaCatalog=catalog,
    )


def run_failure_canaries(out_dir: Path, catalog: SchemaSemanticCatalog | None) -> dict[str, Any]:
    """Exercise closed failures with synthetic injection only (never Happy Path)."""

    results: dict[str, Any] = {}
    state = _failure_state(["execute_read_query"])

    cases: list[tuple[str, Callable[[dict[str, Any], int], tuple[int, dict[str, Any]]]]] = [
        ("F1_invalid_policy_action", lambda _body, _n: (200, _policy_response("fake_action"))),
        (
            "F2_unavailable_policy_action",
            lambda _body, n: (
                200,
                _policy_response("cross_project_validate" if n == 1 else "execute_read_query"),
            ),
        ),
        ("F3_policy_http_500", lambda _body, _n: (500, {"error": {"code": "injected_500"}})),
    ]
    for name, responder in cases:
        server = _FailureServer(responder)
        endpoint = server.start()
        port = HttpDecisionSftPlannerPort(endpoint, "synthetic-failure", token="synthetic")
        try:
            try:
                decision = port.select_action(state)
                outcome: dict[str, Any] = {
                    "status": "returned",
                    "selected_action": decision.selected_action,
                }
            except Exception as exc:
                outcome = {"status": "raised", "code": str(exc)}
        finally:
            port.close()
            server.close()
        results[name] = {"provider": "synthetic_failure_injection", "outcome": outcome}

    class MismatchMaterializer:
        def plan_action(self, _context: ScientificPlannerContext):
            action = CompareGroupsAction(
                actionId="action-" + "b" * 32,
                actionName="compare_groups",
                rationale="synthetic mismatch",
                arguments=ProjectionAnalysisArguments(
                    actionName="compare_groups",
                    observationIds=["observation-" + "a" * 32],
                    analysisGoal="synthetic mismatch",
                ),
            )
            from mico_agent_runtime.ports.scientific_planner import ScientificPlannerResult
            return ScientificPlannerResult(action=action, mode="model")

    server = _FailureServer(lambda _body, _n: (200, _policy_response("adjust_confounders")))
    endpoint = server.start()
    policy = HttpDecisionSftPlannerPort(endpoint, "synthetic-failure", token="synthetic", policy_origin="local_http_stub")
    hybrid = HybridIntentPlannerPort(MismatchMaterializer(), policy)
    try:
        try:
            hybrid.plan_action_with_state(_failure_context(catalog), _failure_state(["adjust_confounders"]))
            results["F4_materializer_action_mismatch"] = {"status": "unexpected_success"}
        except Exception as exc:
            results["F4_materializer_action_mismatch"] = {"status": "raised", "code": str(exc)}
    finally:
        hybrid.close()
        server.close()

    try:
        if catalog is None:
            raise RuntimeError("REAL_CATALOG_UNAVAILABLE")
        invalid = QueryPlan(root_entity="not_in_catalog", select_fields=["sample.disease"], limit=10)
        validate_query_plan_catalog(invalid, catalog)
        results["F5_invalid_query_plan"] = {"status": "unexpected_success"}
    except Exception as exc:
        results["F5_invalid_query_plan"] = {"status": "validator_rejected", "code": str(exc)}

    try:
        insufficient = AnalysisPlan(
            analysis_type="group_comparison",
            source_observation_ids=["observation-" + "c" * 32],
            outcome="abundance.value",
            group_field="sample.disease",
            metrics=["count", "effect_size", "p_value"],
        )
        execute_typed_analysis(
            insufficient,
            [{"a_sample_disease": "T2D", "a_abundance_value": 1.0}],
            1,
            planner_mode="model",
        )
        results["F6_typed_insufficient_data"] = {"status": "unexpected_success"}
    except (GeneratedAnalysisError, ValidationError, ValueError) as exc:
        results["F6_typed_insufficient_data"] = {"status": "operator_rejected", "code": str(exc)}

    _dump(out_dir / "failure_canary_report.json", results)
    return results


def _readiness(happy: dict[str, Any], failures: dict[str, Any]) -> dict[str, Any]:
    trace = happy.get("trace", {})
    final_state = happy.get("final_state", {})
    records = list(final_state.get("decisionRecords", []))
    actions = list(trace.get("action_history", []))
    analyses = list(trace.get("analysis_results", []))
    java_calls = list(trace.get("java_calls", []))
    policy_inputs = [item for item in trace.get("policy_requests", []) if item.get("status") == 200]
    query_execs = trace.get("query_executions", [])
    analysis_execs = trace.get("analysis_executions", [])
    query_semantics = trace.get("query_semantics_audit", {})
    checks = {
        # A provider/serialization smoke test is not sufficient for A100
        # readiness: the real canary must complete its runtime route without
        # a planner/materializer/executor failure.
        "happy_path_runtime_completed": trace.get("runtime_status") == "COMPLETED",
        "gemini_task_understanding": trace.get("task_understanding_origin") == "gemini_model"
        and bool(happy.get("task_understanding_records")),
        "gemini_scientific_policy": trace.get("policy_origin") == "gemini_canary_model"
        and bool(policy_inputs),
        "action_not_script_preset": bool(records) and all(
            record.policy_origin == "gemini_canary_model" for record in records
        ),
        "gemini_query_plan": any(
            item.get("plan_origin") == "gemini_canary_model"
            and item.get("materializer_request_index") is not None
            for item in query_execs
        ),
        "query_plan_not_script": any(
            item.get("plan_origin") == "gemini_canary_model"
            for item in query_execs
        ),
        "real_semantic_catalog": happy.get("catalog") is not None,
        "real_java_tool_api": any(item.get("tool_name") == "execute_read_query" for item in java_calls),
        "real_query_result": any(
            item.get("tool_name") == "execute_read_query" and item.get("status") == "COMPLETED"
            for item in java_calls
        ),
        "requested_group_labels_observed": set(
            query_semantics.get("requested_labels", [])
        ).issubset(set(query_semantics.get("requested_labels_casefold_present", []))),
        "query_group_size_semantics_explicit": all(
            isinstance(item.get("query_result_profile"), dict)
            and bool(item["query_result_profile"].get("group_size_semantics"))
            and bool(item["query_result_profile"].get("sample_count_source"))
            for item in query_execs
        ),
        "analysis_source_exactly_two_groups": any(
            len({
                str(group).casefold()
                for source in item.get("source_observations", [])
                for group in source.get("distinct_group_values", [])
            }) == 2
            for item in analysis_execs
            if item.get("source_observations")
        ),
        # A group-level aggregate over abundance.value without a feature key
        # is not a safe multi-feature comparison, even when its result happens
        # to contain exactly two disease rows.
        "no_group_comparison_feature_pooling": not bool(
            query_semantics.get("group_comparison_feature_mixing_risk", True)
        ),
        "observation_builder": bool(final_state.get("decisionState"))
        and bool(trace.get("rounds")),
        "next_policy_reads_new_state": len(policy_inputs) >= 2
        and policy_inputs[0].get("state") != policy_inputs[1].get("state"),
        "gemini_analysis_plan": any(
            item.get("materializer_request_index") is not None
            for item in analysis_execs
        ),
        "analysis_plan_not_script": any(
            item.get("materializer_request_index") is not None
            for item in analysis_execs
        ),
        "capability_registry": bool(analyses),
        "typed_or_generated_analysis": any(
            item.get("codeVersion") in {"typed-analysis-operator-v1", "sandbox-python-v1"}
            for item in analyses
        ),
        "at_least_three_policy_decisions": len(records) >= 3,
        "no_local_stub_in_happy_path": trace.get("policy_origin") != "local_http_stub",
        "no_fixture_java_in_happy_path": all(
            item.get("source") != "fixture_java_tool" for item in java_calls
        ),
        "no_hardcoded_query_plan": bool(query_execs),
        "no_hardcoded_analysis_plan": any(
            item.get("materializer_request_index") is not None
            for item in analysis_execs
        ),
        "no_deterministic_scientific_policy_fallback": not trace.get(
            "deterministic_scientific_policy_fallback_used", True
        ),
        "no_deterministic_materializer_fallback": not trace.get(
            "deterministic_materializer_fallback_used", True
        ),
        "no_legacy_semantic_strategy_guard": not trace.get("legacy_strategy_guard_used", True),
        "failure_canaries": len(failures) == 6,
    }
    return {
        "A100_READY": all(checks.values()),
        "checks": checks,
        "blockers": [name for name, passed in checks.items() if not passed],
        "happy_path": {
            "action_history": actions,
            "policy_decision_count": len(records),
            "query_execution_count": len(query_execs),
            "analysis_execution_count": len(analysis_execs),
            "runtime_status": trace.get("runtime_status"),
            "error_code": trace.get("error_code"),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Gemini-only 4D-1B dynamic scientific canary")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--max-actions",
        type=int,
        default=8,
        help="bounded action count for this canary run (default: 8)",
    )
    args = parser.parse_args(argv)
    if args.max_actions < 1:
        parser.error("--max-actions must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        happy = run_happy(args.output_dir, max_actions=args.max_actions)
        failures = run_failure_canaries(args.output_dir, happy.get("catalog"))
    except Exception as exc:
        happy = {"trace": {"runtime_status": "FAILED", "error_code": f"{type(exc).__name__}:{exc}"}}
        failures = {}
        _dump(args.output_dir / "failure_canary_report.json", {"runner_error": happy["trace"]["error_code"]})
    readiness = _readiness(happy, failures)
    _dump(args.output_dir / "a100_readiness.json", readiness)
    print(json.dumps(readiness, ensure_ascii=False, indent=2))
    return 0 if readiness["A100_READY"] else 1


if __name__ == "__main__":
    sys.exit(main())
