from __future__ import annotations

import json
import math
import os
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Protocol
from urllib.parse import urlsplit

import httpx

from mico_agent_runtime.contracts.generated_analysis import (
    AnalysisPlannerContext,
    GeneratedAnalysisPlan,
    GeneratedAnalysisPlannerResult,
    TypedAnalysisPlannerResult,
)
from mico_agent_runtime.contracts.materialization import (
    AnalysisPlan as TypedAnalysisPlan,
    QueryPlan,
    validate_query_plan_catalog,
)
from mico_agent_runtime.contracts.intent import IntentPlannerContext, IntentRoutePlan
from mico_agent_runtime.contracts.research import (
    FinishArguments,
    InspectCohortAction,
    InspectCohortArguments,
    ScientificPlannerContext,
    validate_scientific_action,
)
from mico_agent_runtime.contracts.scientific_policy import ScientificPolicyInput
from mico_agent_runtime.contracts.decision_state import ScientificDecisionState
from mico_agent_runtime.ports.gemini_resilience import (
    GeminiRequestBudget,
    GeminiRequestBudgetExceeded,
    GeminiResponseCache,
    parse_retry_delay_seconds,
    sleep_before_retry,
)
from mico_agent_runtime.runtime.data_requirements import (
    DataRequirementError,
    augment_query_plan_with_fields,
    derive_data_requirements,
    prune_unavailable_query_plan,
)
from mico_agent_runtime.ports.scientific_planner import (
    SCIENTIFIC_PLANNER_DETERMINISTIC_FALLBACK,
    ScientificPlannerResult,
    _deterministic_action,
    _finish_action,
)


PLANNER_FALLBACK_CODE = "RESEARCH_PLANNER_DETERMINISTIC_FALLBACK"
DETERMINISTIC_MODE_CODE = "RESEARCH_PLANNER_DETERMINISTIC_MODE"
ANALYSIS_PLANNER_FALLBACK_CODE = "ANALYSIS_PLANNER_DETERMINISTIC_FALLBACK"
MAX_MODEL_RETRIES = 3

# Kept outside the frozen prompt-literal assignments so the historical
# materializer contract fingerprint remains stable while adding the explicit
# feature/sample semantics required by 4D-1E.
_TYPED_FEATURE_GUIDANCE = (
    "When outcome=abundance.value and a verified abundance.feature field is available, include "
    "feature_field=abundance.feature. Never pool multiple feature rows into one outcome; the Runtime "
    "binds a run-scoped opaque sample identity and collapses duplicate sample×feature rows before "
    "statistics. Do not invent or select raw sample_id/patient_id. "
)
_RAW_FEATURE_GUIDANCE = (
    "When a verified abundance.feature field is available, include it in select_fields for feature-aware "
    "analysis; the Runtime binds the opaque sample identity needed to collapse duplicate sample×feature rows. "
)


class _PlannerResponseRejected(ValueError):
    def __init__(self, status_code: int, response: httpx.Response | None = None) -> None:
        super().__init__("planner response rejected")
        self.status_code = status_code
        self.response = response
        self.retry_delay_seconds = parse_retry_delay_seconds(response) if response is not None else None


def _action_fallback_reason(error: Exception) -> str:
    """Map the final planner failure to a safe, closed diagnostic code."""

    if isinstance(error, _PlannerResponseRejected):
        if error.status_code in {401, 403}:
            return "SCIENTIFIC_PLANNER_FALLBACK_REASON_AUTH_REJECTED"
        if error.status_code == 404:
            return "SCIENTIFIC_PLANNER_FALLBACK_REASON_ENDPOINT_NOT_FOUND"
        if error.status_code in {408, 425, 429}:
            return "SCIENTIFIC_PLANNER_FALLBACK_REASON_RATE_LIMITED"
        if error.status_code >= 500:
            return "SCIENTIFIC_PLANNER_FALLBACK_REASON_PROVIDER_SERVER_ERROR"
        return "SCIENTIFIC_PLANNER_FALLBACK_REASON_PROVIDER_RESPONSE_REJECTED"
    if isinstance(error, httpx.TimeoutException):
        return "SCIENTIFIC_PLANNER_FALLBACK_REASON_PROVIDER_TIMEOUT"
    if isinstance(error, httpx.HTTPError):
        return "SCIENTIFIC_PLANNER_FALLBACK_REASON_PROVIDER_NETWORK_ERROR"
    if isinstance(error, json.JSONDecodeError):
        return "SCIENTIFIC_PLANNER_FALLBACK_REASON_INVALID_JSON"
    error_type = type(error).__name__
    message = str(error).lower()
    if error_type == "ValidationError":
        return "SCIENTIFIC_PLANNER_FALLBACK_REASON_ACTION_SCHEMA_INVALID"
    if "not approved" in message:
        return "SCIENTIFIC_PLANNER_FALLBACK_REASON_ACTION_NOT_APPROVED"
    if "catalog" in message or "unsafe sql" in message:
        return "SCIENTIFIC_PLANNER_FALLBACK_REASON_CATALOG_VALIDATION_FAILED"
    if "json" in message or "trailing non-json" in message:
        return "SCIENTIFIC_PLANNER_FALLBACK_REASON_INVALID_JSON"
    if "choices" in message or "message" in message or "content" in message:
        return "SCIENTIFIC_PLANNER_FALLBACK_REASON_RESPONSE_SHAPE_INVALID"
    return "SCIENTIFIC_PLANNER_FALLBACK_REASON_ACTION_CONTRACT_REJECTED"

# Names-only context. It never contains values, credentials or a connection.
READ_QUERY_SCHEMA_GUIDE_V1 = """
Approved read-schema guide v1 (names only):
- patients(patient_id, disease, age, gender, country, body_site, sequencing_platform)
- meta2db_sample_metadata(patient_id, sample_id, project_name, profile_sample,
  health_disease_status, disease_category, raw_metadata)
- microbe_abundance_standard(patient_id, sample_id, microbe_name_standard,
  microbe_name_hash, abundance_value, abundance_unit, normalization_method,
  feature_version, source_batch, sample_date)
- diseases(disease_id, disease_name)
- patient_diseases(patient_id, disease_id)
Use only the internal patient_id relationship. It is an internal business
record key, not a Subject ID or a real-patient count. The model may propose a
bounded explicit-column SELECT/CTE shape; Java validates and executes it.
Never include sample values, locators, credentials, URLs, database names or
raw payloads in a plan.
""".strip()


@dataclass(frozen=True)
class IntentPlannerResult:
    plan: IntentRoutePlan
    mode: str
    fallbackCode: str | None = None


class IntentPlannerPort(Protocol):
    def route_intent(self, context: IntentPlannerContext) -> IntentPlannerResult:
        ...

    def generate_analysis(self, context: AnalysisPlannerContext) -> GeneratedAnalysisPlannerResult:
        ...

    def generate_typed_analysis(self, context: AnalysisPlannerContext) -> TypedAnalysisPlannerResult:
        ...

    def plan_action(self, context: ScientificPlannerContext) -> ScientificPlannerResult:
        ...


def _parse_model_json(content: object) -> object:
    if not isinstance(content, str):
        raise ValueError("planner content is not JSON")
    # DeepSeek-compatible endpoints sometimes return a reasoning wrapper or a
    # fenced JSON object even when response_format=json_object is requested.
    # Strip only transport formatting; never repair action fields or invent
    # values here, because those remain subject to the closed Pydantic model.
    text = re.sub(r"(?is)<think>.*?(?:</think>|$)", "", content).strip()
    fenced = re.search(r"(?is)```(?:json)?\s*(.*?)\s*```", text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Accept a single JSON object followed by harmless transport prose.
        # raw_decode already handles nested objects, but this scan also handles
        # a model prefix before the object and preserves strings containing
        # braces.  A malformed object still fails closed below.
        starts = [index for index, char in enumerate(text) if char in "{["]
        start = starts[0] if starts else -1
        if start < 0:
            raise
        opening = text[start]
        closing = "}" if opening == "{" else "]"
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == opening:
                depth += 1
            elif char == closing:
                depth -= 1
                if depth == 0:
                    candidate = text[start:index + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
        raise ValueError("planner content is not valid JSON")


def _canonicalize_query_plan_wrapper(payload: object) -> object:
    """Move a lossless flattened QueryPlan into ``arguments.queryPlan``.

    Gemini occasionally follows the semantic plan keys but places them next
    to ``arguments.actionName`` instead of under the required QueryPlan
    wrapper.  This is a transport-shape repair only: every plan value still
    comes from the model and is subsequently checked by the closed contract
    and Java catalog validator.
    """

    if not isinstance(payload, dict):
        return payload
    arguments = payload.get("arguments")
    if not isinstance(arguments, dict) or isinstance(arguments.get("queryPlan"), dict):
        return payload
    plan_keys = {
        "schemaVersion",
        "root_entity",
        "relation_path",
        "select_fields",
        "aggregations",
        "filters",
        "group_by",
        "limit",
    }
    if not plan_keys.intersection(arguments):
        return payload
    plan = {key: arguments[key] for key in plan_keys if key in arguments}
    normalized_arguments = {
        key: value for key, value in arguments.items() if key not in plan_keys
    }
    normalized_arguments["queryPlan"] = plan
    normalized_payload = dict(payload)
    normalized_payload["arguments"] = normalized_arguments
    return normalized_payload


def _canonicalize_query_aggregation_aliases(payload: object) -> object:
    """Normalize only lossless legacy spellings before closed validation.

    The Dynamic QueryPlan contract is still closed: Java receives only the
    canonical field names and bounded enums. A few DeepSeek responses may use
    older spellings despite the same bounded intent. Mapping those exact
    aliases is a lossless schema normalization; unknown values, extra keys,
    and invalid grouping shapes remain rejected by Pydantic and the catalog
    gate.
    """

    if not isinstance(payload, dict):
        return payload
    arguments = payload.get("arguments")
    if not isinstance(arguments, dict):
        return payload
    query_plan = arguments.get("queryPlan")
    if not isinstance(query_plan, dict):
        return payload
    aggregations = query_plan.get("aggregations")
    if aggregations is not None and not isinstance(aggregations, list):
        return payload
    if aggregations is None:
        aggregations = []
    allowed_ops = {"count", "mean", "min", "max", "sum"}
    op_aliases = {
        "average": "mean",
        "avg": "mean",
        "total": "sum",
        "minimum": "min",
        "maximum": "max",
    }
    changed = False
    normalized_arguments = dict(arguments)
    normalized_plan = dict(query_plan)
    # These are Runtime-owned explosion controls, never model-authored
    # scientific plan parameters.  Drop them if a provider emits them anyway;
    # the execution boundary will attach the bounded values after validation.
    for runtime_key in ("sample_limit_per_group", "sample_limit_group_field"):
        if runtime_key in normalized_plan:
            normalized_plan.pop(runtime_key)
            changed = True
    if "relation_path" not in normalized_plan and "ordered_relation_path" in normalized_plan:
        normalized_plan["relation_path"] = normalized_plan.pop("ordered_relation_path")
        changed = True
    if "select_fields" not in normalized_plan and "semantic_select_fields" in normalized_plan:
        normalized_plan["select_fields"] = normalized_plan.pop("semantic_select_fields")
        changed = True
    normalized_aggregations: list[object] = []
    for aggregation in aggregations:
        if not isinstance(aggregation, dict):
            normalized_aggregations.append(aggregation)
            continue
        normalized = dict(aggregation)
        if "alias" in normalized:
            # Java owns result aliases; an optional model alias cannot change
            # the query semantics and is not part of QueryAggregation v1.
            normalized.pop("alias")
            changed = True
        if "op" not in normalized and "function" in normalized:
            function = normalized.get("function")
            canonical_function = op_aliases.get(function, function)
            if canonical_function in allowed_ops:
                normalized["op"] = canonical_function
                normalized.pop("function")
                changed = True
        elif normalized.get("op") in op_aliases:
            normalized["op"] = op_aliases[normalized["op"]]
            changed = True
        normalized_aggregations.append(normalized)
    normalized_plan["aggregations"] = normalized_aggregations
    # DeepSeek commonly spells the list payload for an ``in`` filter as
    # ``values`` while the closed QueryPlan contract uses the single
    # ``value`` field for all operators.  This is a lossless wire alias: the
    # operator and exact user labels are preserved, and the closed model still
    # rejects any other unknown filter keys or shapes.
    filters = query_plan.get("filters")
    normalized_filters: list[object] = []
    if isinstance(filters, dict):
        # Some OpenAI-compatible responses collapse a one-filter list into a
        # single object and use ``op``/``values``.  Expand only that bounded
        # lossless representation; all resulting fields still cross the
        # closed QueryFilter model below.
        normalized_filter = dict(filters)
        if "operator" not in normalized_filter and "op" in normalized_filter:
            normalized_filter["operator"] = normalized_filter.pop("op")
            changed = True
        if (
            normalized_filter.get("operator") == "in"
            and "value" not in normalized_filter
            and "values" in normalized_filter
        ):
            normalized_filter["value"] = normalized_filter.pop("values")
            changed = True
        normalized_filters = [normalized_filter]
        changed = True
        normalized_plan["filters"] = normalized_filters
    elif isinstance(filters, list):
        for item in filters:
            if (
                isinstance(item, dict)
                and item.get("operator") == "in"
                and "value" not in item
                and "values" in item
            ):
                normalized_filter = dict(item)
                normalized_filter["value"] = normalized_filter.pop("values")
                normalized_filters.append(normalized_filter)
                changed = True
            else:
                normalized_filters.append(item)
        normalized_plan["filters"] = normalized_filters
    if not normalized_aggregations and normalized_plan.get("group_by"):
        # A group_by without an aggregate has no valid meaning in QueryPlan
        # v1. The raw projection remains useful to the typed analysis stage;
        # remove only the invalid, semantically inert grouping decoration.
        normalized_plan["group_by"] = []
        changed = True
    normalized_arguments["queryPlan"] = normalized_plan
    normalized_payload = dict(payload)
    normalized_payload["arguments"] = normalized_arguments
    rationale = normalized_payload.get("rationale")
    if rationale is None:
        normalized_payload["rationale"] = "execute the bounded catalog-approved read"
        changed = True
    elif isinstance(rationale, str) and len(rationale) > 256:
        normalized_payload["rationale"] = rationale[:256]
        changed = True
    if not changed:
        return payload
    return normalized_payload


def _canonicalize_typed_analysis_shape(payload: object) -> object:
    """Normalize scalar-to-list JSON shape without changing analysis meaning."""

    if not isinstance(payload, dict):
        return payload
    normalized = dict(payload)
    changed = False
    for key in ("source_observation_ids", "covariates", "stratify_by", "metrics"):
        value = normalized.get(key)
        if isinstance(value, str):
            normalized[key] = [value]
            changed = True
        elif value is None and key in {"covariates", "stratify_by"}:
            normalized[key] = []
            changed = True
    return normalized if changed else payload


def _canonicalize_scientific_action_shape(
    payload: object,
    context: ScientificPlannerContext,
) -> tuple[object, tuple[str, ...]]:
    """Repair only lossless projection-argument transport omissions.

    Gemini sometimes returns the selected Action and observation IDs but
    omits the common empty arrays, or uses ``covariates`` for the established
    ``confounders`` key.  These are wire-shape aliases, not a new scientific
    decision: the selected action, source observations and field values stay
    model-authored and all values still cross the closed contract afterward.
    """

    if not isinstance(payload, dict):
        return payload, ()
    # Gemini can occasionally preserve a leading space in a JSON object key
    # (for example ``" rationale"``). Trim only this known transport typo;
    # all scientific values and unknown keys remain closed-contract inputs.
    normalized_payload = dict(payload)
    wire_repairs: list[str] = []
    if " rationale" in normalized_payload and "rationale" not in normalized_payload:
        normalized_payload["rationale"] = normalized_payload.pop(" rationale")
        wire_repairs.append("DYNAMIC_MATERIALIZER_KEY_WHITESPACE_NORMALIZED")
    action_name = normalized_payload.get("actionName")
    arguments = normalized_payload.get("arguments")
    if action_name not in {
        "compare_groups",
        "stratified_analysis",
        "adjust_confounders",
        "cross_project_validate",
        "cross_disease_validate",
    } or not isinstance(arguments, dict):
        return (
            normalized_payload if wire_repairs else payload,
            tuple(wire_repairs),
        )
    normalized_arguments = dict(arguments)
    changed: list[str] = []
    if "analysisGoal" not in normalized_arguments and context.questionSummary:
        normalized_arguments["analysisGoal"] = context.questionSummary[:512]
        changed.append("DYNAMIC_ACTION_ANALYSIS_GOAL_DEFAULTED")
    # Some Gemini responses mirror the later AnalysisPlan vocabulary inside
    # the already-selected ScientificAction (groupField/outcomeField/
    # featureField).  The Action contract intentionally does not carry those
    # execution fields: preserve the group dimension as the Action's
    # dimensions, then let the typed AnalysisPlan materializer choose the
    # outcome/feature from the current observation and catalog.  Removing the
    # duplicate aliases avoids spending a repair request on a lossless wire
    # shape mismatch without allowing the Action role to author a plan.
    group_field = normalized_arguments.get("groupField")
    if "dimensions" not in normalized_arguments and isinstance(group_field, str):
        normalized_arguments["dimensions"] = [group_field]
        changed.append("DYNAMIC_ACTION_GROUP_FIELD_NORMALIZED")
    stratify_fields = normalized_arguments.get("stratifyFields")
    if "dimensions" not in normalized_arguments and isinstance(stratify_fields, (list, str)):
        normalized_arguments["dimensions"] = (
            [stratify_fields] if isinstance(stratify_fields, str) else stratify_fields
        )
        changed.append("DYNAMIC_ACTION_STRATIFY_FIELDS_NORMALIZED")
    for key in ("dimensions", "confounders"):
        if key not in normalized_arguments:
            normalized_arguments[key] = []
            changed.append(f"DYNAMIC_ACTION_{key.upper()}_DEFAULTED")
    if action_name == "adjust_confounders":
        legacy = normalized_arguments.get("covariates")
        if "confounders" not in arguments and isinstance(legacy, (list, str)):
            normalized_arguments["confounders"] = legacy
            changed.append("DYNAMIC_ACTION_COVARIATES_NORMALIZED")
        if "covariates" in normalized_arguments:
            normalized_arguments.pop("covariates")
    for key in ("groupField", "outcomeField", "featureField", "stratifyFields"):
        if key in normalized_arguments:
            normalized_arguments.pop(key)
            changed.append(f"DYNAMIC_ACTION_{key.upper()}_REMOVED")
    if not changed and not wire_repairs:
        return payload, ()
    normalized_payload["arguments"] = normalized_arguments
    return normalized_payload, tuple(dict.fromkeys([*wire_repairs, *changed]))


def _normalize_generated_analysis_code(value: str) -> str:
    """Normalize transport whitespace without destroying Python structure.

    Generated Python is parsed and AST-checked downstream.  Collapsing CR/LF
    into spaces turns valid ``if``/``for`` blocks into a single invalid Python
    statement, so only line-ending and indentation normalization is allowed.
    """

    return value.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ").strip()


def _canonicalize_cross_query_shape(
    payload: object,
    context: ScientificPlannerContext,
) -> object:
    """Normalize a cross-validation read to a group-complete projection.

    DeepSeek still chooses the semantic group/outcome fields.  This narrow
    normalization only closes the cross-validation execution shape when those
    chosen fields are already present: one group field in ``select_fields``
    and one non-count aggregation over the chosen numeric outcome.  It never
    introduces physical identifiers, filters, values, or a fixed SQL query.
    """

    group_field = context.requiredGroupField
    if group_field is None or not isinstance(payload, dict):
        return payload
    arguments = payload.get("arguments")
    if not isinstance(arguments, dict):
        return payload
    query_plan = arguments.get("queryPlan")
    if not isinstance(query_plan, dict):
        return payload
    selected = query_plan.get("select_fields")
    aggregations = query_plan.get("aggregations")
    if not isinstance(selected, list) or not isinstance(aggregations, list):
        return payload
    # A missing-group repair is a sample-level coverage read, not a
    # cross-validation aggregate.  The model must be allowed to return a
    # raw feature-aware projection so Java can expose the opaque sample key
    # and the Observation Builder can merge the missing group into the
    # existing state.  Without this guard the generic cross-query
    # normalizer turns that raw plan into ``GROUP BY disease, MEAN(value)``
    # before validation, destroying the sample-level evidence.
    coverage_feedback = {
        "DYNAMIC_GROUP_COVERAGE_REQUIRED",
        "DYNAMIC_REQUESTED_GROUP_COVERAGE_REQUIRED",
    }
    if coverage_feedback.intersection(context.executionFeedback) \
            and not query_plan.get("aggregations") \
            and not query_plan.get("group_by") \
            and {"abundance.feature", "abundance.value"}.issubset(set(selected)) \
            and "sample_to_metadata" not in set(query_plan.get("relation_path") or []):
        return payload
    aggregation_fields = {
        item.get("field") for item in aggregations
        if isinstance(item, dict) and isinstance(item.get("field"), str)
    }
    outcome_field = next(
        (
            field_id for field_id in context.requiredSemanticFields
            if field_id != group_field
            and (field_id in selected or field_id in aggregation_fields)
        ),
        None,
    )
    if group_field not in selected or outcome_field is None:
        return payload
    existing = next(
        (
            item for item in aggregations
            if isinstance(item, dict) and item.get("field") == outcome_field
        ),
        None,
    )
    op = existing.get("op") if isinstance(existing, dict) else None
    if op not in {"mean", "min", "max", "sum"}:
        op = "mean"
    normalized_plan = dict(query_plan)
    normalized_plan["select_fields"] = [group_field]
    normalized_plan["group_by"] = [group_field]
    normalized_plan["aggregations"] = [{"field": outcome_field, "op": op}]
    if (
        normalized_plan["select_fields"] == selected
        and normalized_plan.get("group_by") == query_plan.get("group_by")
        and normalized_plan["aggregations"] == aggregations
    ):
        return payload
    normalized_arguments = dict(arguments)
    normalized_arguments["queryPlan"] = normalized_plan
    normalized_payload = dict(payload)
    normalized_payload["arguments"] = normalized_arguments
    return normalized_payload


def _planner_error_feedback(error: Exception) -> str:
    """Return schema-only feedback safe to send into the next model attempt."""

    errors_method = getattr(error, "errors", None)
    if callable(errors_method):
        try:
            entries = errors_method(include_url=False)
        except TypeError:
            entries = errors_method()
        details = []
        for entry in entries[:12]:
            location = ".".join(str(item) for item in entry.get("loc", ())) or "root"
            details.append(f"{location}: {entry.get('msg', 'invalid value')}")
        if details:
            return "; ".join(details)[:1200]
    message = str(error).strip() or "closed contract validation failed"
    return f"{type(error).__name__}: {message[:800]}"


def _semantic_catalog_payload(catalog: object) -> dict[str, object]:
    """Project the Java catalog to IDs/semantics before model access.

    Physical source tables, physical column names, join keys, timestamps and
    descriptions are Java-only compiler metadata.  The Materializer needs
    only the stable vocabulary and relation/cardinality constraints.
    """

    if catalog is None:
        return {"schemaVersion": "none", "entities": [], "relations": []}
    entities = []
    for entity in catalog.entities:
        entity_id = entity.entityId or entity.entityName
        entities.append({
            "entityId": entity_id,
            "fields": [
                {
                    "fieldId": field.fieldId or f"{entity_id}.{field.name}",
                    "dataType": field.dataType,
                    "nullable": field.nullable,
                    "semanticStatus": field.semanticStatus,
                    "filterable": field.filterable,
                    "groupable": field.groupable,
                    "aggregatable": field.aggregatable,
                    "displayable": field.displayable,
                    # Scientific capabilities are semantic catalog metadata,
                    # not physical identifiers or values.  They let a
                    # materializer distinguish an outcome from a covariate
                    # (for example age) without Runtime choosing a plan.
                    "scientificCapabilities": list(field.scientificCapabilities),
                }
                for field in entity.fields
                if not field.sensitive
            ],
        })
    relations = [
        {
            "relationId": relation.relationId,
            "leftEntity": next(
                (entity.entityId or entity.entityName for entity in catalog.entities
                 if entity.entityName == relation.leftEntity),
                relation.leftEntity,
            ),
            "rightEntity": next(
                (entity.entityId or entity.entityName for entity in catalog.entities
                 if entity.entityName == relation.rightEntity),
                relation.rightEntity,
            ),
            "relationshipStatus": relation.relationshipStatus,
            "cardinality": relation.cardinality,
        }
        for relation in catalog.joins
        if relation.relationId
    ]
    return {
        "schemaVersion": catalog.schemaVersion,
        "source": "java_schema_contract",
        "entities": entities,
        "relations": relations,
        "queryRules": list(catalog.queryRules),
    }


def required_cross_validation_fields(
    catalog: object,
    action_name: str,
    question_summary: str,
) -> list[str]:
    """Return closed catalog IDs needed by a cross-validation read/analysis.

    This is intentionally semantic and catalog-driven.  It never invents a
    physical table/column, a filter value, or a statistical result.  The
    numeric outcome is ranked from verified aggregatable fields, preferring a
    ``*.value``/abundance-like field when the redacted question names that
    concept; otherwise the stable ``*.value`` convention is preferred.
    """

    if action_name not in {"cross_project_validate", "cross_disease_validate"}:
        return []
    if catalog is None:
        return []
    fields: list[tuple[str, object]] = []
    for entity in getattr(catalog, "entities", []):
        entity_id = entity.entityId or entity.entityName
        for field in entity.fields:
            field_id = field.fieldId or f"{entity_id}.{field.name}"
            if field.sensitive or field.semanticStatus != "verified":
                continue
            fields.append((field_id, field))

    if action_name == "cross_project_validate":
        group_candidates = [
            field_id for field_id, _field in fields
            if field_id.endswith(".project")
        ]
    else:
        group_candidates = [
            field_id for field_id, _field in fields
            if field_id.endswith(".disease") or field_id == "disease.name"
        ]
    if not group_candidates:
        return []

    question = question_summary.lower()
    numeric_candidates = [
        (field_id, field) for field_id, field in fields
        if field.dataType in {"integer", "number"} and field.aggregatable
    ]
    if not numeric_candidates:
        return group_candidates[:1]

    def score(item: tuple[str, object]) -> tuple[int, str]:
        field_id, _field = item
        lowered = field_id.lower()
        value_score = 0
        if lowered.endswith(".value"):
            value_score += 40
        if "abundance" in lowered and ("abundance" in question or "numeric" in question):
            value_score += 60
        if any(token in question for token in ("abundance", "丰度", "measurement", "outcome", "结果")) \
                and any(token in lowered for token in ("abundance", "value", "measurement", "outcome")):
            value_score += 30
        return (-value_score, field_id)

    numeric_candidates.sort(key=score)
    return list(dict.fromkeys([group_candidates[0], numeric_candidates[0][0]]))


def required_group_analysis_fields(
    catalog: object,
    action_name: str,
    question_summary: str,
) -> list[str]:
    """Select catalog-backed group/outcome requirements for an analysis read.

    This does not create a query or choose physical columns.  It only tells the
    model materializer which verified semantic fields must be present before a
    grouped analysis can execute.  Java still validates the relation path and
    compiles the final parameterized SQL.
    """

    if action_name not in {"compare_groups", "stratified_analysis", "adjust_confounders"}:
        return []
    if catalog is None:
        return []
    fields: list[tuple[str, object]] = []
    for entity in getattr(catalog, "entities", []):
        entity_id = entity.entityId or entity.entityName
        for field in entity.fields:
            field_id = field.fieldId or f"{entity_id}.{field.name}"
            if field.sensitive or field.semanticStatus != "verified":
                continue
            fields.append((field_id, field))

    group_candidates = [
        (field_id, field) for field_id, field in fields
        if field.groupable and field.displayable
    ]
    numeric_candidates = [
        (field_id, field) for field_id, field in fields
        if field.dataType in {"integer", "number"} and field.aggregatable
    ]
    if not group_candidates or not numeric_candidates:
        return []

    question = question_summary.lower()

    def group_score(item: tuple[str, object]) -> tuple[int, str]:
        field_id, _field = item
        lowered = field_id.lower()
        score = 0
        preferences = (
            (("project", "项目"), ".project", 100),
            (("disease", "疾病", "group", "组"), ".disease", 90),
            (("country", "国家"), ".country", 80),
            (("gender", "sex", "性别"), ".gender", 70),
            (("body site", "部位"), ".body_site", 60),
        )
        for tokens, suffix, weight in preferences:
            if lowered.endswith(suffix):
                score += weight if any(token in question for token in tokens) else weight // 3
        return (-score, field_id)

    def outcome_score(item: tuple[str, object]) -> tuple[int, str]:
        field_id, _field = item
        lowered = field_id.lower()
        score = 0
        if lowered.endswith(".value"):
            score += 100
        if "abundance" in lowered:
            score += 80
        if any(token in question for token in ("abundance", "丰度", "outcome", "结果")):
            if any(token in lowered for token in ("abundance", "value", "outcome")):
                score += 50
        return (-score, field_id)

    group_candidates.sort(key=group_score)
    numeric_candidates.sort(key=outcome_score)
    group_field = group_candidates[0][0]
    outcome_field = next(
        (field_id for field_id, _field in numeric_candidates if field_id != group_field),
        None,
    )
    if outcome_field is None:
        return []
    return [group_field, outcome_field]


def _model_scientific_context_payload(context: ScientificPlannerContext) -> dict[str, object]:
    """Build the Materializer request without physical catalog identifiers.

    In the dynamic path ``decisionState`` is a validated six-block snapshot
    supplied by the graph.  It is useful context for Gemini's concrete plan
    materialization, but it is still compressed state: raw observations and
    executor payloads remain outside this request.
    """

    payload = context.model_dump(mode="json", exclude_none=True)
    if context.schemaCatalog is not None:
        payload["schemaCatalog"] = _semantic_catalog_payload(context.schemaCatalog)
    return payload


def _validate_catalog_sql(sql: str, context: ScientificPlannerContext) -> None:
    """Reject model SQL that names a table outside Java's semantic catalog."""

    catalog = context.schemaCatalog
    if catalog is None:
        return
    allowed_tables = {entity.sourceTable.lower() for entity in catalog.entities}
    for match in re.finditer(r"(?is)\b(?:from|join)\s+([A-Za-z_][A-Za-z0-9_]*)", sql):
        table_name = match.group(1).lower()
        if table_name not in allowed_tables:
            raise ValueError(f"scientific planner SQL references unknown catalog table: {table_name}")


def deterministic_intent_route(context: IntentPlannerContext) -> IntentRoutePlan:
    """Choose a safe route without inventing SQL or evidence facts."""
    question = context.questionSummary.lower()
    if any(term in question for term in ("诊断", "治疗", "处方", "causal", "select ", "sql")):
        raise ValueError("unsafe intent")
    knowledge_terms = (
        "文献", "证据", "机制", "关联", "研究", "微生物", "微生物组", "菌",
        "疾病", "健康", "对照", "差异", "比较", "物种", "literature", "evidence",
        "microbiome", "microbe", "disease", "healthy", "pathway",
    )
    relation_terms = (
        "关系", "关联", "机制", "作用", "影响", "差别", "差异", "比较", "对比",
        "pathway", "graph", "difference", "compare", "versus", "relationship",
        "association", "mechanism", "link", " vs ",
    )
    multi_hop_terms = ("多跳", "链路", "路径", "multi-hop", "through", "from", "到")
    if "knowledge_retrieval" in context.allowedWorkflows and any(term in question for term in knowledge_terms):
        has_relation = any(term in question for term in relation_terms)
        has_multi_hop = any(term in question for term in multi_hop_terms)
        has_semantic = any(term in question for term in (
            "文献", "证据", "研究", "微生物", "微生物组", "菌", "疾病", "健康", "对照",
            "物种", "literature", "evidence", "microbiome", "microbe", "disease", "healthy",
        ))
        if has_multi_hop and (has_relation or has_semantic):
            query_type = "composite"
            retrieval_mode = "hybrid"
            signals = ["semantic", "relation", "multi_hop", "composite", "deterministic"]
            confidence = 0.94
        elif has_multi_hop:
            query_type = "multi_hop"
            retrieval_mode = "graph"
            signals = ["multi_hop", "deterministic"]
            confidence = 0.91
        elif has_relation and has_semantic:
            query_type = "composite"
            retrieval_mode = "hybrid"
            signals = ["semantic", "relation", "composite", "deterministic"]
            confidence = 0.93
        elif has_relation:
            query_type = "relation"
            retrieval_mode = "graph"
            signals = ["relation", "deterministic"]
            confidence = 0.9
        else:
            query_type = "semantic_fact"
            retrieval_mode = "vector"
            signals = ["semantic", "deterministic"]
            confidence = 0.88
        return IntentRoutePlan(
            workflow="knowledge_retrieval",
            responseMode="structured_evidence_review",
            safetyProfile="non_diagnostic",
            retrievalMode=retrieval_mode,
            queryType=query_type,
            routeConfidence=confidence,
            classificationSignals=signals,
        )
    if "dynamic_read_query" not in context.allowedWorkflows:
        if "knowledge_retrieval" in context.allowedWorkflows:
            return IntentRoutePlan(
                workflow="knowledge_retrieval",
                responseMode="structured_evidence_review",
                safetyProfile="non_diagnostic",
                retrievalMode="vector",
                queryType="semantic_fact",
                routeConfidence=0.7,
                classificationSignals=["semantic", "deterministic"],
            )
        raise ValueError("no approved workflow")
    return IntentRoutePlan(
        workflow="dynamic_read_query",
        responseMode="structured_analysis_job",
        safetyProfile="non_diagnostic",
        routeConfidence=0.7,
        classificationSignals=["deterministic"],
    )


def deterministic_analysis_plan() -> GeneratedAnalysisPlan:
    return GeneratedAnalysisPlan(
        language="python",
        analysisType="descriptive_summary",
        code="result = {'metrics': {'row_count': len(rows)}, 'topFeatures': []}",
    )


def deterministic_typed_analysis_plan(context: AnalysisPlannerContext) -> TypedAnalysisPlan:
    """Compatibility plan for non-dynamic callers; Dynamic Runtime rejects it."""

    action_name = context.actionName or context.workflow
    mapping = {
        "compare_groups": "group_comparison",
        "stratified_analysis": "stratified_comparison",
        "adjust_confounders": "confounder_adjustment",
        "cross_project_validate": "cross_project_validation",
        "cross_disease_validate": "cross_disease_validation",
        "analyze_projection": "projection",
    }
    analysis_type = mapping.get(action_name, "projection")
    fields = context.availableSemanticFields
    outcome = next((field for field in fields if field.endswith(".value")), None)
    feature = next((field for field in fields if field.endswith(".feature")), None)
    group = next((field for field in fields if field.endswith((".disease", ".project", ".gender"))), None)
    if analysis_type == "cross_project_validation":
        group = next((field for field in fields if field.endswith(".disease")), group)
    elif analysis_type == "cross_disease_validation":
        group = next((field for field in fields if field.endswith(".project")), group)
    validation = None
    if analysis_type == "cross_project_validation":
        validation = next((field for field in fields if field.endswith(".project")), None)
    elif analysis_type == "cross_disease_validation":
        validation = next((field for field in fields if field.endswith(".disease")), None)
    return TypedAnalysisPlan(
        analysis_type=analysis_type,
        source_observation_ids=context.sourceObservationIds,
        outcome=outcome,
        feature_field=feature,
        group_field=group if analysis_type in {
            "group_comparison",
            "stratified_comparison",
            "confounder_adjustment",
            "cross_project_validation",
            "cross_disease_validation",
        } else None,
        covariates=[field for field in fields if field.endswith((".age", ".gender"))][:2]
        if analysis_type == "confounder_adjustment" else [],
        stratify_by=[group] if analysis_type == "stratified_comparison" and group else [],
        validation_field=validation,
        analysis_goal=context.questionSummary,
        metrics=["count"],
    )


class DeterministicIntentPlanner:
    def route_intent(self, context: IntentPlannerContext) -> IntentPlannerResult:
        return IntentPlannerResult(
            plan=deterministic_intent_route(context),
            mode="deterministic",
            fallbackCode=DETERMINISTIC_MODE_CODE,
        )

    def generate_analysis(self, _context: AnalysisPlannerContext) -> GeneratedAnalysisPlannerResult:
        return GeneratedAnalysisPlannerResult(
            plan=deterministic_analysis_plan(),
            mode="deterministic",
            fallbackCode=DETERMINISTIC_MODE_CODE,
        )

    def generate_typed_analysis(self, context: AnalysisPlannerContext) -> TypedAnalysisPlannerResult:
        return TypedAnalysisPlannerResult(
            plan=deterministic_typed_analysis_plan(context),
            mode="deterministic",
            fallbackCode=DETERMINISTIC_MODE_CODE,
        )

    def plan_action(self, context: ScientificPlannerContext) -> ScientificPlannerResult:
        return ScientificPlannerResult(
            action=_deterministic_action(context),
            mode="deterministic",
            fallbackCode=SCIENTIFIC_PLANNER_DETERMINISTIC_FALLBACK,
        )


class HttpResearchPlannerPort:
    """OpenAI-compatible intent and Python planner with closed JSON outputs."""

    def __init__(self, base_url: str, model: str, token: str,
                 transport: httpx.BaseTransport | None = None,
                 materializer_origin: str = "model",
                 request_budget: GeminiRequestBudget | None = None,
                 response_cache: GeminiResponseCache | None = None,
                 sleep_fn: Callable[[float], None] = time.sleep,
                 max_retry_delay_seconds: float | None = None,
                 timeout_seconds: float = 300.0) -> None:
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username:
            raise ValueError("planner URL is invalid")
        if parsed.query or parsed.fragment or not model or not token.strip():
            raise ValueError("planner configuration is invalid")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("planner timeout is invalid")
        base = base_url.rstrip("/")
        provider_path = parsed.path.rstrip("/")
        if (
            (parsed.hostname or "").lower().endswith("deepseek.com")
            or provider_path.endswith("/v1")
            or provider_path.endswith("/openai")
        ):
            self._endpoint = base + "/chat/completions"
        else:
            self._endpoint = base + "/v1/chat/completions"
        self._model = model
        self._token = token
        # Provenance is kept separate from ``mode`` so a Gemini canary can be
        # distinguished from the production DeepSeek materializer without
        # changing the shared planner contract.  The default preserves the
        # historical ``model`` value for existing callers/tests.
        self.materializer_origin = materializer_origin
        self._request_budget = request_budget
        self._response_cache = response_cache
        self._sleep_fn = sleep_fn
        self._max_retry_delay_seconds = max_retry_delay_seconds
        self._timeout_seconds = timeout_seconds
        self.request_count = 0
        self.cache_hit_count = 0
        self._disable_reasoning = "deepseek.com" in (parsed.hostname or "").lower()
        # Real collection runs may have two independent cases in flight.  Give
        # the provider enough time to answer without turning a transient slow
        # response into a deterministic fallback; retries still remain bounded
        # by MAX_MODEL_RETRIES.
        self._client = httpx.Client(
            transport=transport,
            timeout=timeout_seconds,
            trust_env=False,
            limits=httpx.Limits(max_keepalive_connections=0, max_connections=10),
        )

    def _post_json(self, body: dict[str, object]) -> httpx.Response:
        """POST one materializer request with optional run budget/cache."""

        headers = {"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"}
        key = self._response_cache.key(self._endpoint, body) if self._response_cache else None
        request = self._client.build_request("POST", self._endpoint, headers=headers, json=body)
        if key is not None:
            cached = self._response_cache.get(key, request)
            if cached is not None:
                self.cache_hit_count += 1
                return cached
        if self._request_budget is not None:
            self._request_budget.reserve("materializer")
        self.request_count += 1
        response = self._client.send(request)
        if key is not None and response.status_code == 200:
            self._response_cache.put(key, response)
        return response

    def route_intent(self, context: IntentPlannerContext) -> IntentPlannerResult:
        system = (
            "Return only closed JSON with workflow, responseMode, safetyProfile, queryType, "
            "routeConfidence, retrievalMode, retrievalBranches, classificationSignals, and optional sqlDraft. "
            "workflow may be dynamic_read_query or knowledge_retrieval, but only if it appears "
            "in the approved workflow list. dynamic_read_query requires responseMode "
            "structured_analysis_job and a bounded SELECT/CTE sqlDraft. knowledge_retrieval "
            "requires responseMode structured_evidence_review and must not include sqlDraft. "
            "For knowledge_retrieval, retrievalMode must be vector for semantic literature search, "
            "graph for explicit relationship or multi-hop questions, or hybrid for combined questions. "
            "queryType must be semantic_fact, relation, multi_hop, or composite. "
            "retrievalBranches must be exactly [vector], [graph], or [vector, graph] to match retrievalMode. "
            "routeConfidence is a number from 0 to 1 and classificationSignals only contains the closed "
            "values semantic, relation, multi_hop, composite, model. "
            "safetyProfile must be non_diagnostic. Do not output tools, permissions, IDs, "
            "versions, sample values, locators, credentials, URLs, database names or payloads. "
            "sqlDraft is an untrusted bounded SELECT/CTE proposal for Java validation. It must "
            "use explicit columns, no SELECT *, no comments, no semicolon, no DDL/DML, no cross-"
            "database reference, and include LIMIT 1 through 1000.\n\n" + READ_QUERY_SCHEMA_GUIDE_V1
        )
        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": context.model_dump_json()},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": 1800,
            "temperature": 0,
        }
        if self._disable_reasoning:
            body["thinking"] = {"type": "disabled"}
        for attempt in range(MAX_MODEL_RETRIES + 1):
            try:
                response = self._post_json(body)
                if response.status_code in {400, 422}:
                    fallback_body = dict(body)
                    fallback_body.pop("response_format", None)
                    response = self._post_json(fallback_body)
                    if response.status_code in {400, 422} and "thinking" in fallback_body:
                        fallback_body.pop("thinking", None)
                        response = self._post_json(fallback_body)
                if response.status_code != 200:
                    raise _PlannerResponseRejected(response.status_code, response)
                payload = response.json()
                content = payload["choices"][0]["message"]["content"]
                parsed_payload = _parse_model_json(content)
                if isinstance(parsed_payload, dict) and isinstance(parsed_payload.get("sqlDraft"), str):
                    draft = parsed_payload["sqlDraft"]
                    if any(ord(char) < 32 and char not in "\r\n\t" for char in draft):
                        raise ValueError("unsafe SQL control character")
                    parsed_payload = dict(parsed_payload)
                    parsed_payload["sqlDraft"] = re.sub(r"[\r\n\t]+", " ", draft).strip()
                parsed = IntentRoutePlan.model_validate(parsed_payload)
                if parsed.workflow not in context.allowedWorkflows:
                    raise ValueError("planner route is not executable")
                if parsed.workflow == "knowledge_retrieval" and not parsed.retrievalBranches:
                    raise ValueError("knowledge route has no retrieval branch")
                if parsed.workflow == "dynamic_read_query" and not parsed.sqlDraft:
                    raise ValueError("dynamic planner route has no SQL draft")
                return IntentPlannerResult(plan=parsed, mode="model")
            except Exception as error:
                if isinstance(error, GeminiRequestBudgetExceeded):
                    raise
                if attempt < MAX_MODEL_RETRIES:
                    sleep_before_retry(
                        error,
                        sleep=self._sleep_fn,
                        max_delay_seconds=self._max_retry_delay_seconds,
                    )
                    body["messages"].append({
                        "role": "user",
                        "content": (
                            "The previous planner output was rejected by the closed contract. "
                            f"Validation feedback: {_planner_error_feedback(error)}. "
                            "Retry with one JSON object matching the approved workflow and do not include "
                            "any explanation outside JSON."
                        ),
                    })
        return IntentPlannerResult(
            plan=deterministic_intent_route(context),
            mode="deterministic",
            fallbackCode=PLANNER_FALLBACK_CODE,
        )

    def generate_analysis(self, context: AnalysisPlannerContext) -> GeneratedAnalysisPlannerResult:
        system = (
            "Return only JSON with language, analysisType, and code. language must be python. "
            "The code is untrusted and is AST-validated and sandboxed. It may read only list "
            "variable rows and safe builtins len, sum, min, max, sorted, round, float, int, "
            "abs, enumerate, range, and str. Do not import, access attributes, open files, "
            "network, SQL, eval, or exec. Assign result with metrics and topFeatures. Never "
            "output identifiers, locators, sample names, disease labels, URLs, payloads or credentials. "
            "Keep the code short: no more than 12 statements and no deeply nested comprehensions. "
            "Use only direct row[\"column_from_context\"] indexing, arithmetic, comparisons, "
            "if/for, and simple comprehensions; do not call methods such as get, items, map, "
            "or any function other than the listed safe builtins. Do not use keyword arguments. "
            "For a numeric summary, select the exact numeric column from context and use this "
            "shape: values = [float(row[\"<available numeric column>\"]) for row in rows "
            "if row[\"<available numeric column>\"] is not None]; then assign result with "
            "count and, when values is non-empty, a mean. Replace the placeholder with a real "
            "column from context; never emit the placeholder literally. "
            "If executionFeedback is present, it is an opaque sandbox rejection code for your prior "
            "plan: replace the unsafe expression with a simpler safe-builtin-only expression."
        )
        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": context.model_dump_json()},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": 1800,
            "temperature": 0,
        }
        if self._disable_reasoning:
            body["thinking"] = {"type": "disabled"}
        for attempt in range(MAX_MODEL_RETRIES + 1):
            try:
                response = self._post_json(body)
                if response.status_code != 200:
                    raise _PlannerResponseRejected(response.status_code, response)
                payload = response.json()
                plan_payload = _parse_model_json(payload["choices"][0]["message"]["content"])
                if isinstance(plan_payload, dict) and isinstance(plan_payload.get("code"), str):
                    plan_payload = dict(plan_payload)
                    plan_payload["code"] = _normalize_generated_analysis_code(plan_payload["code"])
                return GeneratedAnalysisPlannerResult(
                    plan=GeneratedAnalysisPlan.model_validate(plan_payload), mode="model"
                )
            except Exception as error:
                if isinstance(error, GeminiRequestBudgetExceeded):
                    raise
                if attempt < MAX_MODEL_RETRIES:
                    sleep_before_retry(
                        error,
                        sleep=self._sleep_fn,
                        max_delay_seconds=self._max_retry_delay_seconds,
                    )
                    body["messages"].append({
                        "role": "user",
                        "content": (
                            "The previous analysis plan was rejected by the closed contract. "
                            f"Validation feedback: {_planner_error_feedback(error)}. "
                            "Retry with language=python, a sandbox-safe code string, metrics and topFeatures, "
                            "and return JSON only."
                        ),
                    })
        return GeneratedAnalysisPlannerResult(
            plan=deterministic_analysis_plan(),
            mode="deterministic",
            fallbackCode=ANALYSIS_PLANNER_FALLBACK_CODE,
        )

    def generate_typed_analysis(self, context: AnalysisPlannerContext) -> TypedAnalysisPlannerResult:
        """Materialize an AnalysisPlan with semantic fields only.

        This is the Dynamic Scientific Runtime contract.  The older
        ``generate_analysis`` method remains for the historical intent path,
        but it is never accepted by the dynamic graph.
        """

        system = (
            "Return exactly one JSON object matching AnalysisPlan v2. The only permitted "
            "top-level keys are schemaVersion, analysis_type, source_observation_ids, "
            "outcome, feature_field, group_field, covariates, stratify_by, validation_field, "
            "numeric_stratification, "
            "method, analysis_goal, and metrics. Always set schemaVersion to analysis-plan-v2. "
            "Do not emit execution_mode, language, code, SQL, Python, values, labels, raw "
            "identifiers, credentials, or explanation outside the JSON object. Do not return a "
            "ScientificAction envelope or a generated-analysis/code envelope. The selected "
            "Action is a closed binding and must not be changed: compare_groups maps only to "
            "analysis_type=group_comparison; stratified_analysis maps only to "
            "analysis_type=stratified_comparison; adjust_confounders maps only to "
            "analysis_type=confounder_adjustment; cross_project_validate maps only to "
            "analysis_type=cross_project_validation; cross_disease_validate maps only to "
            "analysis_type=cross_disease_validation; analyze_projection maps only to "
            "analysis_type=projection. Never choose a different analysis family because it "
            "seems scientifically preferable. Copy source_observation_ids only from the "
            "supplied context. Copy field IDs only from availableSemanticFields; they are "
            "semantic IDs, never physical table or column names. The notation below is a "
            "closed shape: replace every angle-bracket placeholder with a value copied from "
            "context and never emit an angle-bracket placeholder literally. "
            "For compare_groups the required shape is {schemaVersion:'analysis-plan-v2', "
            "analysis_type:'group_comparison', source_observation_ids:[<observation-id>], "
            "outcome:<numeric-field>, feature_field:<feature-field-or-null>, "
            "group_field:<dimension>, covariates:[], stratify_by:[], validation_field:null, "
            "method:{family:'auto',name:null,parameters:{}}, analysis_goal:<text>, "
            "metrics:[<allowed-metric>]}. For stratified_analysis the required shape is "
            "{schemaVersion:'analysis-plan-v2', analysis_type:'stratified_comparison', "
            "source_observation_ids:[<observation-id>], outcome:<numeric-field>, "
            "feature_field:<feature-field-or-null>, group_field:<dimension>, covariates:[], "
            "stratify_by:[<stratifier-dimension>], validation_field:null, "
            "numeric_stratification:<numeric-spec-or-null>, "
            "method:{family:'auto',name:null,parameters:{}}, analysis_goal:<text>, "
            "metrics:[<allowed-metric>]}. For adjust_confounders the required shape is "
            "{schemaVersion:'analysis-plan-v2', analysis_type:'confounder_adjustment', "
            "source_observation_ids:[<observation-id>], outcome:<numeric-field>, "
            "feature_field:<feature-field-or-null>, group_field:<dimension>, "
            "covariates:[<covariate-dimension>], stratify_by:[], validation_field:null, "
            "method:{family:'auto',name:null,parameters:{}}, analysis_goal:<text>, "
            "metrics:[<allowed-metric>]}. For cross_project_validate or "
            "cross_disease_validate use the mapped analysis_type, a non-null group_field, "
            "a non-null validation_field, and empty covariates/stratify_by unless the "
            "contract explicitly requires otherwise. validation_field must be a separate "
            "semantic dimension from group_field (never the same field), present in the "
            "current availableSemanticFields, verified with dimension capability, and have "
            "at least two observed values; never substitute a covariate or outcome. If no "
            "independent validation dimension exists, return a contract failure rather than "
            "guessing one. For numeric stratify_by fields, numeric_stratification is required "
            "and must contain stratifier, strategy, bin_count, cut_points, "
            "min_samples_per_group, missing_value_policy, and multiple_testing. Use "
            "strategy=quantile with bin_count 2-8 and cut_points=[] for the standard typed "
            "route; Runtime derives cut points from sample-level observations. For categorical "
            "stratifiers set numeric_stratification=null. For analyze_projection use the mapped "
            "projection type and null group_field, validation_field with empty covariates "
            "and stratify_by. The method object must contain family, name, and parameters; "
            "family is one of auto, parametric, nonparametric, regression, bootstrap, "
            "stratified, validation, custom, and parameters may only use iterations, "
            "confidence_level, bin_count, bin_boundaries, reference_group. For the standard "
            "typed operator use method.family=auto and name=null. Metrics may only be count, "
            "mean, median, effect_size, effect, p_value, confidence_interval. When "
            "outcome=abundance.value and abundance.feature is available, feature_field must "
            "be abundance.feature; never pool multiple feature rows into one outcome. "
            "Return no code and no execution mode; Capability Registry owns execution choice."
            + _TYPED_FEATURE_GUIDANCE
        )
        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": context.model_dump_json()},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": 900,
            "temperature": 0,
        }
        if self._disable_reasoning:
            body["thinking"] = {"type": "disabled"}
        mapping = {
            "compare_groups": "group_comparison",
            "stratified_analysis": "stratified_comparison",
            "adjust_confounders": "confounder_adjustment",
            "cross_project_validate": "cross_project_validation",
            "cross_disease_validate": "cross_disease_validation",
            "analyze_projection": "projection",
        }
        expected_type = mapping.get(context.actionName or context.workflow)
        required_fields_by_action = {
            "compare_groups": "outcome, group_field",
            "stratified_analysis": "outcome, group_field, stratify_by (non-empty)",
            "adjust_confounders": "outcome, group_field, covariates (non-empty)",
            "cross_project_validate": "outcome, group_field, validation_field",
            "cross_disease_validate": "outcome, group_field, validation_field",
            "analyze_projection": "outcome",
        }
        repair_contract = (
            "The previous response did not conform to AnalysisPlan v2. "
            f"Keep selected Action={context.actionName or context.workflow!r}; "
            f"required analysis_type={expected_type!r}. Return only one JSON object with "
            "exactly the v2 keys schemaVersion, analysis_type, source_observation_ids, "
            "outcome, feature_field, group_field, covariates, stratify_by, "
            "numeric_stratification, validation_field, method, analysis_goal, metrics. "
            f"Required non-empty fields for this Action: "
            f"{required_fields_by_action.get(context.actionName or context.workflow, 'the action-specific fields')}. "
            "Do not return a ScientificAction wrapper, code, language, generated-analysis "
            "payload, execution mode, or explanation. Do not change the selected Action."
        )
        for attempt in range(MAX_MODEL_RETRIES + 1):
            try:
                response = self._post_json(body)
                if response.status_code != 200:
                    raise _PlannerResponseRejected(response.status_code, response)
                payload = response.json()
                plan_payload = _parse_model_json(payload["choices"][0]["message"]["content"])
                if not isinstance(plan_payload, dict):
                    raise ValueError("typed analysis plan is not an object")
                canonical_payload = _canonicalize_typed_analysis_shape(plan_payload)
                typed_repairs = (
                    ["DYNAMIC_MATERIALIZER_SCHEMA_NORMALIZED"]
                    if canonical_payload is not plan_payload else []
                )
                plan_payload = canonical_payload
                plan = TypedAnalysisPlan.model_validate(plan_payload)
                if expected_type is not None and plan.analysis_type != expected_type:
                    raise ValueError("typed analysis type does not match the selected action")
                allowed_observations = set(context.sourceObservationIds)
                if not set(plan.source_observation_ids).issubset(allowed_observations):
                    raise ValueError("typed analysis references an unknown observation")
                allowed_fields = set(context.availableSemanticFields)
                referenced_fields = {
                    value for value in (
                        plan.outcome,
                        plan.feature_field,
                        plan.group_field,
                        plan.validation_field,
                        *plan.covariates,
                        *plan.stratify_by,
                    ) if value is not None
                }
                if not referenced_fields.issubset(allowed_fields):
                    raise ValueError("typed analysis references a field outside the catalog")
                required_fields = set(context.requiredSemanticFields)
                if required_fields and not required_fields.issubset(referenced_fields):
                    raise ValueError(
                        "typed analysis omits required semantic fields: "
                        + ",".join(sorted(required_fields - referenced_fields))
                    )
                if context.requiredGroupField is not None \
                        and plan.group_field != context.requiredGroupField:
                    raise ValueError("typed analysis uses the wrong cross-validation group field")
                return TypedAnalysisPlannerResult(
                    plan=plan,
                    mode="model",
                    repairCodes=typed_repairs,
                )
            except Exception as error:
                if isinstance(error, GeminiRequestBudgetExceeded):
                    raise
                if attempt < MAX_MODEL_RETRIES:
                    sleep_before_retry(
                        error,
                        sleep=self._sleep_fn,
                        max_delay_seconds=self._max_retry_delay_seconds,
                    )
                    body["messages"].append({
                        "role": "user",
                        "content": (
                            repair_contract + " "
                            f"Validation feedback: {_planner_error_feedback(error)}."
                        ),
                    })
        return TypedAnalysisPlannerResult(
            plan=deterministic_typed_analysis_plan(context),
            mode="deterministic",
            fallbackCode=ANALYSIS_PLANNER_FALLBACK_CODE,
        )

    def plan_action(self, context: ScientificPlannerContext) -> ScientificPlannerResult:
        """Choose one generic capability from the current redacted state.

        The model may propose only typed semantic parameters inside the
        dedicated action contract.
        It cannot choose scopes, tools outside the approved action list, or
        inject raw Java observations into the prompt.
        """
        materializer_binding = ""
        if len(context.approvedActions) == 1:
            # HybridIntentPlannerPort passes a singleton allow-list after Qwen
            # has already selected the Action.  Make that ownership explicit
            # to DeepSeek: this call materializes arguments only and must not
            # perform a second scientific routing decision.
            selected_action = context.approvedActions[0]
            materializer_binding = (
                "Materialization-only mode: Runtime has already selected exactly one high-level "
                f"Action ({selected_action}). Emit that exact actionName at both the root and "
                "arguments.actionName positions. Never substitute, reorder, or invent another "
                "Scientific Action; if the arguments cannot be materialized, return a contract "
                "error rather than selecting a different capability.\n\n"
            )
        coverage_binding = ""
        if (
            len(context.approvedActions) == 1
            and context.approvedActions[0] == "execute_read_query"
            and {
                "DYNAMIC_GROUP_COVERAGE_REQUIRED",
                "DYNAMIC_REQUESTED_GROUP_COVERAGE_REQUIRED",
            }.intersection(context.executionFeedback)
        ):
            # These IDs are copied from the current Decision State/catalog;
            # they are not a fixed query template.  The explicit shape keeps
            # a model from answering a one-group coverage deficiency with a
            # second raw LIMIT over the abundance fan-out.
            group_field = context.requiredGroupField
            outcome_field = next(
                (
                    field_id for field_id in context.requiredSemanticFields
                    if field_id != group_field
                ),
                None,
            )
            decision_state = context.decisionState
            data_state = (
                decision_state.get("data_state", {})
                if isinstance(decision_state, dict) else {}
            )
            group_state = (
                data_state.get("group_state", {})
                if isinstance(data_state, dict) else {}
            )
            sample_count = data_state.get("sample_count") if isinstance(data_state, dict) else None
            observed_group_count = group_state.get("group_count") if isinstance(group_state, dict) else None
            if isinstance(sample_count, int) and sample_count > 0 and observed_group_count == 1:
                task_state = (
                    decision_state.get("task", {})
                    if isinstance(decision_state, dict) else {}
                )
                constraints = (
                    task_state.get("constraints", {})
                    if isinstance(task_state, dict) else {}
                )
                requested_groups = (
                    constraints.get("disease_groups", [])
                    if isinstance(constraints, dict) else []
                )
                observed_groups = {
                    str(value).casefold()
                    for value in (
                        group_state.get("group_sizes", {}).keys()
                        if isinstance(group_state, dict)
                        and isinstance(group_state.get("group_sizes"), dict)
                        else []
                    )
                }
                missing_groups = [
                    value for value in requested_groups
                    if isinstance(value, str) and value.casefold() not in observed_groups
                ]
                missing_filter = (
                    f"Use an exact in-filter containing only {missing_groups!r}. "
                    if missing_groups else
                    "Use the exact requested disease labels in the filter. "
                )
                coverage_binding = (
                    "Technical coverage repair: the current raw sample-level projection contains only "
                    "one observed disease group. Query the requested disease group labels that are "
                    "not present in decisionState.data_state.group_state.group_sizes, using only the "
                    "connected sample_to_abundance relation. Emit raw sample.disease, abundance.feature, "
                    "and abundance.value with empty aggregations and group_by; keep exact user labels in "
                    + missing_filter
                    + "do not include metadata.project. This is a technical coverage read, "
                    "not a change of Scientific Action.\n\n"
                )
            elif "DYNAMIC_METADATA_JOIN_EMPTY" in context.executionFeedback:
                coverage_binding = (
                    "Technical coverage repair: the previous sample_to_metadata join returned zero rows. "
                    "Emit a raw sample-level abundance projection using only the connected "
                    "sample_to_abundance relation. Include sample.disease, abundance.feature, and "
                    "abundance.value (plus explicitly requested sample covariates when available); "
                    "omit metadata.project, sample_to_metadata, aggregations, and group_by. "
                    "This is a technical data-shape repair and does not change the selected Action.\n\n"
                )
            elif group_field and outcome_field:
                coverage_binding = (
                    "Technical coverage contract for this materialization: the prior "
                    "validated read did not expose two groups. Emit a grouped coverage "
                    f"probe using requiredGroupField={group_field!r} and numeric "
                    f"outcome={outcome_field!r}: select_fields and group_by must contain "
                    f"only {group_field!r}, and aggregations must contain one non-count "
                    f"aggregation over {outcome_field!r}. Do not include abundance.feature "
                    "or another one-to-many dimension in this probe. This is a technical "
                    "shape requirement, not a change of Scientific Action.\n\n"
                )
        # Keep the legacy prompt assignment first for the frozen evaluator's
        # literal hash. The singleton branch below replaces it entirely and
        # is the only prompt used by the new Decision-State materializer.
        system = (
            "Return only one closed JSON ScientificAction object with actionId, actionName, rationale, and arguments. "
            "Allowed actionName values are execute_read_query, inspect_cohort, compare_groups, "
            "stratified_analysis, adjust_confounders, cross_project_validate, cross_disease_validate, "
            "retrieve_evidence, analyze_projection, finish, but the action must be in approvedActions. "
            "The root actionName and arguments.actionName are both required and must be identical. "
            "For every projection analysis action, emit all five argument keys exactly as named; "
            "There is no key named covariates in a ScientificAction: use confounders for the "
            "adjust_confounders Action, even when the scientific wording says covariates. "
            "Use [] for dimensions or confounders only when that field is not required. "
            "The response must use exactly that root/arguments nesting; do not output a wrapper such as "
            "selected_action, do not output placeholder IDs or ellipses, and copy every observationId "
            "only from the current context. "
            "For stratified_analysis provide a non-empty dimensions array; for adjust_confounders "
            "provide a non-empty confounders array; for either cross validation provide one or more observationIds "
            "that are present in the current context. A single VALIDATED Java observation is sufficient when its "
            "returned projection contains the requested project/disease group field and a numeric outcome with at "
            "least two observed group values. Do not force a second read merely to satisfy an observation-count "
            "rule; request another bounded read only when the available observation lacks the required fields or "
            "the task explicitly combines independent reads. Never fabricate an observationId. For stratified_analysis use a non-empty generic dimension such as country or age; "
            "for adjust_confounders use non-empty generic confounders such as age or gender. These arrays must "
            "contain field names only, never labels, values, IDs, or placeholders. "
            "Use these exact argument shapes: execute_read_query/inspect_cohort -> "
            "{actionName, queryPlan, limit}; compare_groups/stratified_analysis/adjust_confounders/"
            "cross_project_validate/cross_disease_validate -> {actionName, observationIds, analysisGoal, "
            "dimensions, confounders}; analyze_projection -> {actionName, observationId, analysisGoal}; "
            "For example, adjust_confounders MUST use arguments with "
            "actionName, observationIds, analysisGoal, dimensions, and confounders; never use a "
            "covariates key or add outcome/group_field keys. "
            "retrieve_evidence -> {actionName, topics, retrievalMode, topK, maxHops}; finish -> "
            "{actionName, reasonCode}. Copy observationId values only from the supplied context. "
            "Do not substitute groupA/groupB, comparisonType, query, or adjustment for the required keys. "
            "Do not repeat a cross_project_validate or cross_disease_validate action after that action already "
            "appears in observations; move to the next approved evidence or finish decision. "
            "Do not repeat inspect_cohort after a validated inspect_cohort observation; choose the next approved "
            "analysis/evidence action or finish. "
            "Do not repeat any already observed analysis or evidence action when a fresh approved action is available; "
            "in particular, do not repeat stratified_analysis, compare_groups, adjust_confounders, analyze_projection, "
            "retrieve_evidence, or a cross-validation action merely to spend the budget. "
            "Do not invent a disease-specific tool, cohort ID, sample ID, locator, permission, database, or tool. "
            "execute_read_query and inspect_cohort must emit queryPlan with root_entity, ordered relation_path, "
            "semantic select_fields, closed aggregations, filters, group_by, and limit. Never emit SQL, physical "
            "table/column names, join predicates, or free-form expressions; Java compiles and validates the plan. "
            "inspect_cohort is the metadata-first "
            "entry and may not be skipped when the state has no observation. The projection actions may reference "
            "only prior observationIds and use generic analysisGoal/dimensions/confounders; they cannot introduce "
            "SQL, versions, locators or raw labels. retrieve_evidence must use only redacted topics. "
            "Use finish when evidence is insufficient or no safe next step is justified. "
            "When the required validation action has been completed and the evidence obligation is satisfied, "
            "finish with the exact argument shape {actionName: finish, reasonCode: EVIDENCE_SUFFICIENT}; "
            "reasonCode is an enum token, not a sentence. "
            "Action semantics matter: validate project stability with cross_project_validate, "
            "country/age/sex strata with stratified_analysis, named confounders with "
            "adjust_confounders, disease specificity with cross_disease_validate, and a "
            "multi-feature combination with analyze_projection before evidence retrieval or "
            "finish when the question explicitly asks for that check. "
            "For finish, use NO_NEW_INFORMATION when the question explicitly says that data "
            "or evidence is insufficient; ACTION_BUDGET_EXHAUSTED is Runtime-owned and must "
            "not be selected by the model. "
            "For a scientific_exploration question that asks to discover or validate findings, do not finish "
            "after a Java data observation if retrieve_evidence is approved and has not been used, unless the "
            "question explicitly says that evidence is insufficient or cannot support a comparison. "
            "If executionFeedback is non-empty, a previous Java read returned only the listed opaque code. "
            "Re-plan once with a fresh, simpler catalog-bounded read: use one catalog root entity, explicit "
            "verified semantic fields, a valid ordered relation path, and no invented values or filters. Do not "
            "repeat the prior query. The feedback is "
            "not data or evidence and must not appear in the final answer. "
            "The following is the Java-provided semantic vocabulary. Use only field IDs and relation IDs. "
            "It intentionally omits physical tables, columns, join keys, values and permissions; Java owns "
            "the final compiler and validation.\n\n"
            + json.dumps(
                _semantic_catalog_payload(context.schemaCatalog),
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        if len(context.approvedActions) == 1:
            # Do not carry the legacy planner's metadata-first/order advice
            # into a singleton materialization call.  Those are strategy
            # rules, while this call owns only concrete arguments for the
            # already-selected Action.
            system = materializer_binding + coverage_binding + (
                "Return one closed JSON ScientificAction object with actionId, actionName, rationale, "
                "and arguments. The Runtime already selected the only allowed action in approvedActions; "
                "the root actionName and arguments.actionName must exactly equal it. Do not choose another "
                "capability, inspect/query first, or decide scientific ordering. For read actions emit a "
                "QueryPlan using exactly these keys: root_entity, relation_path, select_fields, "
                "aggregations, filters, group_by, limit. Use relation_path (an ordered list of relation IDs) "
                "to connect fields from more than one entity; never use keys named select, from, joins, "
                "or targetEntity. Every selected field must be a semantic field ID from the catalog. "
                "The QueryPlan contract requires 1 to 20000 for the bounded result-row limit and at most 2 relation IDs; never "
                "emit a larger limit or a three-hop path. Always nest the plan under arguments.queryPlan "
                "and never place root_entity, relation_path, select_fields, aggregations, filters, "
                "group_by, or limit directly beside arguments.actionName. Java applies the row LIMIT "
                "after the Runtime-owned unique-sample cap; never emit sample_limit_per_group or "
                "sample_limit_group_field because those technical controls are attached by Runtime. "
                "When the question asks for a numeric scientific measurement, include a catalog field marked "
                "with the outcome capability and its required verified relation path. Include groupable "
                "dimensions and covariates that the current task explicitly asks about when available. "
                "When requiredSemanticFields is non-empty, treat those verified semantic IDs as the "
                "Runtime-derived minimum projection for this read: include every reachable required "
                "field in select_fields for a raw ungrouped plan (or preserve it in a valid aggregation "
                "shape). Never substitute an unrelated field or omit a requirement. This is a technical "
                "data-acquisition binding, not a new Action or a scientific ordering decision. "
                "If the task constraints explicitly name disease groups, projects, or other labels, use a "
                "catalog filter (eq/in) with those exact user-provided labels; never invent or normalize a "
                "label and never put a label into a field list. "
                "For a raw-row read intended for a later typed analysis, leave aggregations and group_by "
                "as empty arrays and put the raw numeric outcome in select_fields. If you use an aggregation, "
                "select_fields must contain exactly the group_by fields (the aggregated field is not also a "
                "selected raw field). Never put a group_by field outside select_fields. "
                "Every aggregation.field must be a catalog field whose aggregatable capability is true. "
                "Never use a dimension, label, or metadata field such as sample.disease, metadata.project, "
                "sample.age, or sample.gender as the field of a count/mean/min/max/sum aggregation. "
                "A grouped count is represented by the returned grouped rows; it is not a count aggregation "
                "over the group field. Use a verified numeric outcome such as abundance.value for a numeric "
                "aggregation, and keep the group dimension only in select_fields/group_by. "
                "Dynamic replan guidance: a DYNAMIC_FRESH_QUERY_SHAPE_REQUIRED plan must be materially "
                "different from the prior read; use a groupable catalog dimension with count/mean "
                "aggregation when raw rows do not yet cover two groups. "
                "If the current observations contain a validated Java read with rowCount=0 or fewer "
                "than two observed group values, do not repeat that read's exact restrictive filter set; "
                "use a broader catalog-bounded projection or a groupable catalog aggregation to discover "
                "the real cohort coverage while retaining the numeric outcome when available. "
                "When DYNAMIC_GROUP_COVERAGE_REQUIRED follows a raw abundance read that exposed only one "
                "group, a valid coverage probe is a minimal aggregation grouped by the requested group "
                "dimension with count or mean of the numeric outcome; do not use another raw LIMIT-only "
                "projection that can be filled by one group's feature rows. "
                "For DYNAMIC_EMPTY_QUERY_REPLAN or DYNAMIC_METADATA_JOIN_EMPTY or DYNAMIC_GROUP_COVERAGE_REQUIRED, the replacement "
                "must change the data shape rather than merely changing the limit. Prefer the shortest "
                "catalog-verified relation path that still returns the requested numeric outcome and a "
                "groupable dimension; omit an optional one-to-many relation when it is not needed for the "
                "selected fields. This is a data-coverage repair, not permission to choose another Action. "
                "When DYNAMIC_METADATA_JOIN_EMPTY is present, do not include sample_to_metadata or metadata.project in the replacement raw abundance projection; project coverage is unknown and must not be inferred from an empty join. "
                "For DYNAMIC_REQUESTED_GROUP_COVERAGE_REQUIRED, if the task explicitly supplies disease "
                "groups, retain those exact labels as the filter while changing the relation path or "
                "projection; do not invent aliases, normalize labels, or replace the requested groups "
                "with every disease value in the database. "
                "A relation path is an ordered connected chain: every selected field's entity must be "
                "the root entity or a reachable endpoint of that chain. Do not combine disease.name "
                "with sample/abundance fields through a disconnected path. If a prior metadata join "
                "returned no rows, a valid fresh shape may use root_entity=sample with only the verified "
                "sample_to_abundance relation and omit metadata.project until a connected project read "
                "is needed. "
                 "Do not emit SQL or physical names. "
                 "For compare_groups, stratified_analysis, adjust_confounders, cross_project_validate, "
                "and cross_disease_validate, copy observationIds from context and use exactly the common "
                "analysis argument keys actionName, observationIds, analysisGoal, dimensions, and "
                "confounders. For analyze_projection, use a different exact shape: arguments must contain "
                "only actionName=analyze_projection, one singular observationId copied from context, and "
                "analysisGoal; never emit observationIds, dimensions, or confounders for this Action. "
                "Use only semantic fields available in context; do not invent observations, values, code, "
                 "or a different analysis family. For evidence use only the redacted research topic. For "
                 "finish use the closed reasonCode enum. Return JSON only.\n\n"
                 + _RAW_FEATURE_GUIDANCE
                 + "When decisionState is present in the materializer context, use its task, data_state, "
                "analysis_state, evidence_state, progress, and action_space as read-only compressed facts. "
                "It is not a source of raw rows and it does not authorize an action outside approvedActions. "
                "The Java-provided semantic vocabulary is:\n"
                + json.dumps(
                    _semantic_catalog_payload(context.schemaCatalog),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        else:
            # The multi-action compatibility path retains its historical
            # planner prompt and policy behavior.
            pass
        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(
                    _model_scientific_context_payload(context),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": 1800,
            "temperature": 0,
        }
        if self._disable_reasoning:
            body["thinking"] = {"type": "disabled"}
        for attempt in range(MAX_MODEL_RETRIES + 1):
            try:
                response = self._post_json(body)
                if response.status_code != 200:
                    raise _PlannerResponseRejected(response.status_code, response)
                payload = response.json()
                parsed_payload = _parse_model_json(payload["choices"][0]["message"]["content"])
                if not isinstance(parsed_payload, dict):
                    raise ValueError("scientific planner output is not an object")
                parsed_payload = _canonicalize_query_plan_wrapper(parsed_payload)
                canonical_payload = _canonicalize_query_aggregation_aliases(parsed_payload)
                cross_payload = _canonicalize_cross_query_shape(canonical_payload, context)
                action_payload, action_repairs = _canonicalize_scientific_action_shape(
                    cross_payload, context
                )
                materializer_repairs = tuple(dict.fromkeys([
                    *(("DYNAMIC_MATERIALIZER_SCHEMA_NORMALIZED",)
                      if canonical_payload is not parsed_payload else ()),
                    *(("DYNAMIC_CROSS_QUERY_GROUP_NORMALIZED",)
                      if cross_payload is not canonical_payload else ()),
                    *action_repairs,
                ]))
                parsed_payload = action_payload
                parsed_payload = dict(parsed_payload)
                # Action correlation is Runtime-owned. A model may choose the
                # closed action kind and its bounded arguments, but it must never
                # be able to forge or destabilize the audit identifier.
                parsed_payload["actionId"] = "action-" + sha256(
                    (context.questionSummary + "|" + str(context.remainingActionBudget)).encode("utf-8")
                ).hexdigest()[:32]
                action = validate_scientific_action(parsed_payload)
                if action.actionName in {"execute_read_query", "inspect_cohort"}:
                    if action.arguments.queryPlan is None:
                        raise ValueError("dynamic read materialization requires queryPlan")
                    # Requirements are technical, Catalog-derived projection
                    # bindings.  Add only verified missing fields to a raw
                    # plan before Java validation; never invent a field or
                    # alter a grouped scientific aggregation.
                    original_query_plan = action.arguments.queryPlan
                    try:
                        pruned_query_plan = prune_unavailable_query_plan(
                            original_query_plan,
                            context.unavailableSemanticFields,
                            context.schemaCatalog,
                        )
                        required_query_plan = augment_query_plan_with_fields(
                            pruned_query_plan,
                            context.requiredSemanticFields,
                            context.schemaCatalog,
                        )
                        required_query_plan = prune_unavailable_query_plan(
                            required_query_plan,
                            context.unavailableSemanticFields,
                            context.schemaCatalog,
                        )
                    except DataRequirementError as exc:
                        raise ValueError(str(exc)) from exc
                    if required_query_plan != original_query_plan:
                        action = action.model_copy(update={
                            "arguments": action.arguments.model_copy(update={
                                "queryPlan": required_query_plan,
                                "limit": required_query_plan.limit,
                            }),
                        })
                        repair_codes = [*materializer_repairs]
                        if pruned_query_plan != original_query_plan:
                            repair_codes.append("RUNTIME_UNAVAILABLE_FIELDS_PRUNED")
                        if required_query_plan != pruned_query_plan:
                            repair_codes.append("RUNTIME_DATA_REQUIREMENT_FIELDS_BOUND")
                        materializer_repairs = tuple(dict.fromkeys(repair_codes))
                    validate_query_plan_catalog(action.arguments.queryPlan, context.schemaCatalog)
                    required_fields = set(context.requiredSemanticFields)
                    selected_or_aggregated = set(action.arguments.queryPlan.select_fields) | {
                        item.field for item in action.arguments.queryPlan.aggregations
                    }
                    if required_fields and not required_fields.issubset(selected_or_aggregated):
                        missing = ",".join(sorted(required_fields - selected_or_aggregated))
                        raise ValueError(
                            "query plan omits required semantic fields: " + missing
                        )
                    # ``requiredGroupField`` has two distinct uses.  For a
                    # cross-validation action it is a true grouped-plan
                    # contract.  During a read-coverage repair it is only
                    # context for the materializer prompt; a raw,
                    # feature-aware projection with explicit requested-group
                    # filters is also a valid way to obtain sample-level
                    # observations.  Do not reject that projection with the
                    # misleading cross-validation error.
                    coverage_repair = bool({
                        "DYNAMIC_GROUP_COVERAGE_REQUIRED",
                        "DYNAMIC_REQUESTED_GROUP_COVERAGE_REQUIRED",
                    }.intersection(context.executionFeedback))
                    if context.requiredGroupField is not None and not coverage_repair:
                        query_plan = action.arguments.queryPlan
                        if context.requiredGroupField not in set(query_plan.group_by):
                            raise ValueError("cross validation query plan requires group_by")
                        required_outcomes = required_fields - {context.requiredGroupField}
                        if required_outcomes and not any(
                            item.field in required_outcomes and item.op != "count"
                            for item in query_plan.aggregations
                        ):
                            raise ValueError(
                                "cross validation query plan requires a numeric aggregation"
                            )
                    if {
                        "DYNAMIC_GROUP_COVERAGE_REQUIRED",
                        "DYNAMIC_REQUESTED_GROUP_COVERAGE_REQUIRED",
                    }.intersection(context.executionFeedback):
                        query_plan = action.arguments.queryPlan
                        required_group = context.requiredGroupField
                        raw_feature_coverage = (
                            not query_plan.aggregations
                            and not query_plan.group_by
                            and "abundance.feature" in set(query_plan.select_fields)
                            and "abundance.value" in set(query_plan.select_fields)
                            and "sample_to_metadata" not in set(query_plan.relation_path)
                        )
                        raw_metadata_coverage = (
                            not query_plan.aggregations
                            and not query_plan.group_by
                            and "metadata.project" in set(query_plan.select_fields)
                            and "sample.disease" in set(query_plan.select_fields)
                            and "sample_to_metadata" in set(query_plan.relation_path)
                        )
                        # Any feature-aware raw sample projection on the
                        # connected abundance relation is valid coverage
                        # repair, even when the compact feedback list no
                        # longer retains the older metadata-join code.
                        metadata_empty_raw = raw_feature_coverage or raw_metadata_coverage
                        if not metadata_empty_raw:
                            if required_group is None:
                                raise ValueError(
                                    "dynamic group coverage requires a verified group field"
                                )
                            if required_group not in set(query_plan.group_by):
                                raise ValueError(
                                    "dynamic group coverage query requires group_by"
                                )
                            if not any(
                                item.field != required_group
                                and item.op in {"count", "mean", "min", "max", "sum"}
                                for item in query_plan.aggregations
                            ):
                                raise ValueError(
                                    "dynamic group coverage query requires an aggregation"
                                )
                if action.actionName not in context.approvedActions:
                    raise ValueError("scientific action is not approved")
                return ScientificPlannerResult(
                    action=action,
                    mode="model",
                    repairCodes=materializer_repairs,
                )
            except Exception as error:
                if isinstance(error, GeminiRequestBudgetExceeded):
                    raise
                last_fallback_reason = _action_fallback_reason(error)
                if attempt < MAX_MODEL_RETRIES:
                    sleep_before_retry(
                        error,
                        sleep=self._sleep_fn,
                        max_delay_seconds=self._max_retry_delay_seconds,
                    )
                    body["messages"].append({
                        "role": "user",
                        "content": (
                            "The previous ScientificAction was rejected by the closed contract. "
                            f"Validation feedback: {_planner_error_feedback(error)}. "
                            "Retry with the exact argument shape above, including arguments.actionName "
                            "matching the root actionName, and return JSON only. For compare_groups, "
                            "stratified_analysis, adjust_confounders, cross_project_validate, and "
                            "cross_disease_validate, arguments must use exactly actionName, "
                            "observationIds, analysisGoal, dimensions, and confounders; for "
                            "adjust_confounders use confounders (not covariates) and provide a non-empty "
                            "list. For analyze_projection, arguments must use exactly actionName, "
                            "one singular observationId, and analysisGoal; never use observationIds, "
                            "dimensions, or confounders. Do not add outcome, groupField, group_field, "
                            "or other alternate keys."
                        ),
                    })
        try:
            fallback_action = _deterministic_action(context)
        except Exception:
            fallback_action = _finish_action(context)
        return ScientificPlannerResult(
            action=fallback_action,
            mode="deterministic",
            fallbackCode=SCIENTIFIC_PLANNER_DETERMINISTIC_FALLBACK,
            fallbackReasonCode=last_fallback_reason,
        )

    def close(self) -> None:
        self._client.close()


def _guard_dynamic_action(
    context: ScientificPlannerContext,
    selected_action: str,
) -> tuple[str, str | None]:
    """Repair only closed high-level obligations before materialization.

    The returned action is still sent to the model materializer.  This helper
    never creates a query, an analysis plan, or an execution result.
    """

    approved = set(context.approvedActions)
    # ``inspect_cohort`` is a Runtime-owned metadata probe.  It is backed by
    # a bounded Java read, but it is not the analyzable tabular observation
    # that a dynamic AnalysisPlan is allowed to consume.  Count only explicit
    # ``execute_read_query`` observations here so policy cannot jump from
    # metadata inspection directly into analysis.
    tabular_count = sum(
        item.actionName == "execute_read_query"
        and item.source == "java_controlled_read"
        and item.status in {"VALIDATED", "PARTIAL"}
        for item in context.observations
    )
    analysis_actions = {
        "compare_groups",
        "stratified_analysis",
        "adjust_confounders",
        "cross_project_validate",
        "cross_disease_validate",
        "analyze_projection",
    }
    if not context.observations and "inspect_cohort" in approved and selected_action != "inspect_cohort":
        return "inspect_cohort", "DYNAMIC_INITIAL_INSPECTION_REPAIRED"
    if selected_action in analysis_actions and tabular_count == 0 and "inspect_cohort" in approved:
        if "execute_read_query" in approved:
            return "execute_read_query", "DYNAMIC_ANALYSIS_READ_REQUIRED_REPAIRED"
        return "inspect_cohort", "DYNAMIC_ANALYSIS_WITHOUT_TABULAR_REPAIRED"
    latest_tabular = next(
        (
            item for item in reversed(context.observations)
            if item.actionName == "execute_read_query"
            and item.source == "java_controlled_read"
            and item.status in {"VALIDATED", "PARTIAL"}
        ),
        None,
    )
    if selected_action in analysis_actions and latest_tabular is not None \
            and "numeric_outcome_missing" in latest_tabular.qualityCodes \
            and "execute_read_query" in approved:
        return "execute_read_query", "DYNAMIC_ANALYSIS_NUMERIC_READ_REQUIRED_REPAIRED"
    if selected_action in {"cross_project_validate", "cross_disease_validate"}:
        # One grouped Java observation is a complete cross-validation input;
        # forcing a second read only creates duplicate QueryPlans and does not
        # create an independent cohort.  Require a fresh read only when the
        # available observation is absent or lacks the closed semantic fields
        # needed by this task.
        required_fields = set(required_cross_validation_fields(
            context.schemaCatalog,
            selected_action,
            context.questionSummary,
        ))
        latest_fields = set(latest_tabular.queryPlanFields) if latest_tabular else set()
        missing_required = bool(required_fields) and not required_fields.issubset(latest_fields)
        if latest_tabular is None or missing_required or (
            latest_tabular is not None
            and "numeric_outcome_missing" in latest_tabular.qualityCodes
        ):
            if "execute_read_query" in approved:
                return "execute_read_query", "DYNAMIC_CROSS_VALIDATION_REQUIRED_FIELDS_REPAIRED"
            if "inspect_cohort" in approved and tabular_count == 0:
                return "inspect_cohort", "DYNAMIC_CROSS_VALIDATION_INSPECTION_REQUIRED_REPAIRED"
    return selected_action, None


def _bounded_action_rationale(value: str) -> str:
    """Fit a policy reason into the closed ScientificAction rationale field.

    ``DecisionPolicyOutput.decision_reason`` is retained separately for the
    trace and allows up to 512 characters.  The executable action contract is
    deliberately narrower (256 characters), so this boundary must be
    explicit rather than relying on an unvalidated ``model_copy`` update.
    """

    text = str(value).strip()
    return (text or "execute the selected bounded scientific action")[:256]


def _runtime_owned_inspection_action(
    context: ScientificPlannerContext,
    *,
    rationale: str,
) -> InspectCohortAction:
    """Build the metadata-first inspection plan without a materializer call.

    ``inspect_cohort`` is intentionally Runtime-owned.  The model may select
    this high-level action, but it must not invent the first query shape before
    Java has supplied a semantic catalog.  The resulting typed plan contains
    only catalog semantic IDs; Java remains the final validator/compiler.
    """

    catalog = context.schemaCatalog
    if catalog is None or not catalog.entities:
        raise RuntimeError("DYNAMIC_INSPECTION_CATALOG_REQUIRED")
    entity = catalog.entities[0]
    entity_id = entity.entityId or entity.entityName
    fields = [
        field.fieldId or f"{entity_id}.{field.name}"
        for field in entity.fields
        if field.displayable
        and not field.sensitive
        and field.semanticStatus == "verified"
    ][:8]
    if not fields:
        fields = [
            field.fieldId or f"{entity_id}.{field.name}"
            for field in entity.fields
            if not field.sensitive and field.semanticStatus == "verified"
        ][:8]
    if not fields:
        raise RuntimeError("DYNAMIC_INSPECTION_CATALOG_FIELDS_REQUIRED")
    query_plan = QueryPlan(
        root_entity=entity_id,
        relation_path=[],
        select_fields=fields,
        limit=100,
    )
    validate_query_plan_catalog(query_plan, catalog)
    action_id = "action-" + sha256(
        (context.questionSummary + "|" + str(context.remainingActionBudget)
         + "|runtime-owned-inspect").encode("utf-8")
    ).hexdigest()[:32]
    return InspectCohortAction(
        actionId=action_id,
        actionName="inspect_cohort",
        # The policy contract permits a longer audit reason (512 chars),
        # while ScientificAction.rationale is intentionally bounded to 256
        # chars.  Keep the complete decision_reason in the trace metadata and
        # pass only the bounded action rationale to the execution contract.
        rationale=_bounded_action_rationale(rationale),
        arguments=InspectCohortArguments(
            actionName="inspect_cohort",
            queryPlan=query_plan,
            limit=query_plan.limit,
        ),
    )


class HybridIntentPlannerPort:
    """Combine a policy model with a dynamic research materializer.

    The SFT/DPO model chooses only the next high-level capability.  The base
    research model then materializes that exact capability into question- and
    schema-specific SQL or analysis arguments.  In the new Decision-State
    path a deterministic materializer is permitted only as an execution-plan
    fallback and only when it preserves the selected Action family; the
    historical legacy entry point keeps its stricter model-only contract.
    """

    def __init__(
        self,
        base: IntentPlannerPort,
        decision_planner: IntentPlannerPort,
        *,
        allow_deterministic_materializer_fallback: bool = True,
    ) -> None:
        self._base = base
        self._decision_planner = decision_planner
        # Production compatibility keeps the existing bounded fallback.  The
        # Gemini-only 4D-1B Happy Path disables it so a successful canary
        # proves that the materializer itself generated every plan.
        self.allow_deterministic_materializer_fallback = (
            allow_deterministic_materializer_fallback
        )
        self.materializer_origin = getattr(base, "materializer_origin", "model")

    prefer_policy_action = True
    dynamic_action_materialization = True

    def route_intent(self, context: IntentPlannerContext) -> IntentPlannerResult:
        return self._base.route_intent(context)

    def generate_analysis(self, context: AnalysisPlannerContext) -> GeneratedAnalysisPlannerResult:
        return self._base.generate_analysis(context)

    def generate_typed_analysis(self, context: AnalysisPlannerContext) -> TypedAnalysisPlannerResult:
        generator = getattr(self._base, "generate_typed_analysis", None)
        if not callable(generator):
            raise RuntimeError("DYNAMIC_TYPED_ANALYSIS_MATERIALIZER_NOT_CONFIGURED")
        return generator(context)

    @staticmethod
    def _merged_repair_codes(materialized: ScientificPlannerResult, repair_code: str | None) -> tuple[str, ...]:
        return tuple(dict.fromkeys([
            *getattr(materialized, "repairCodes", ()),
            *(([repair_code] if repair_code else [])),
        ]))

    @staticmethod
    def _materialization_context(
        context: ScientificPlannerContext,
        guarded_action: str,
        requested_action: str | None = None,
    ) -> ScientificPlannerContext:
        feedback = list(context.executionFeedback)
        if guarded_action == "execute_read_query":
            tabular_reads = [
                item for item in context.observations
                if item.source == "java_controlled_read"
            ]
            if any(item.queryPlanFields for item in tabular_reads):
                feedback.append("DYNAMIC_FRESH_QUERY_SHAPE_REQUIRED")
            if any(item.rowCount == 0 for item in tabular_reads):
                feedback.append("DYNAMIC_EMPTY_QUERY_REPLAN")
            if any(
                item.rowCount == 0
                and "sample_to_metadata" in item.queryPlanRelationPath
                for item in tabular_reads
            ):
                # A zero-row metadata join is an observed coverage fact, not
                # permission to invent a project mapping. Keep the next
                # materialization on the connected sample→abundance path.
                feedback.append("DYNAMIC_METADATA_JOIN_EMPTY")
            # The compressed Decision State is the only place where the
            # Runtime exposes deterministic group coverage.  This feedback
            # describes a technical data-shape deficiency; it does not rank
            # or replace the Scientific Action selected by the Policy.
            decision_state = context.decisionState
            data_state = (
                decision_state.get("data_state", {})
                if isinstance(decision_state, dict) else {}
            )
            group_state = (
                data_state.get("group_state", {})
                if isinstance(data_state, dict) else {}
            )
            group_count = group_state.get("group_count") if isinstance(group_state, dict) else None
            if tabular_reads and isinstance(group_count, int) and group_count < 2:
                feedback.append("DYNAMIC_GROUP_COVERAGE_REQUIRED")
            task_state = (
                decision_state.get("task", {})
                if isinstance(decision_state, dict) else {}
            )
            constraints = (
                task_state.get("constraints", {})
                if isinstance(task_state, dict) else {}
            )
            requested_groups = (
                constraints.get("disease_groups", [])
                if isinstance(constraints, dict) else []
            )
            if (
                tabular_reads
                and isinstance(group_count, int)
                and group_count != 2
                and isinstance(requested_groups, list)
                and requested_groups
            ):
                feedback.append("DYNAMIC_REQUESTED_GROUP_COVERAGE_REQUIRED")
        required_fields = list(context.requiredSemanticFields)
        required_group_field = context.requiredGroupField
        unavailable = set(context.unavailableSemanticFields)
        if unavailable:
            # Java has already proved these Catalog capabilities unavailable
            # for the current cohort. Preserve that observation as a
            # limitation and do not force them into the replacement read.
            required_fields = [
                field_id for field_id in required_fields
                if field_id not in unavailable
            ]
        coverage_feedback = {
            "DYNAMIC_GROUP_COVERAGE_REQUIRED",
            "DYNAMIC_REQUESTED_GROUP_COVERAGE_REQUIRED",
        }
        if guarded_action == "execute_read_query" and coverage_feedback.intersection(feedback):
            # A one-group/empty read is a technical data-shape deficiency.  If
            # the compressed observation state already identified the group
            # dimension and numeric outcome, carry those verified semantic IDs
            # into the next materializer contract.  This does not select a
            # Scientific Action; it only prevents another raw LIMIT projection
            # from being accepted as a purported coverage probe.
            decision_state = context.decisionState
            data_state = (
                decision_state.get("data_state", {})
                if isinstance(decision_state, dict) else {}
            )
            group_state = (
                data_state.get("group_state", {})
                if isinstance(data_state, dict) else {}
            )
            state_group_field = (
                group_state.get("group_field")
                if isinstance(group_state, dict) else None
            )
            if required_group_field is None and isinstance(state_group_field, str):
                required_group_field = state_group_field
            state_outcomes = (
                data_state.get("available_outcomes", [])
                if isinstance(data_state, dict) else []
            )
            if isinstance(state_outcomes, list):
                for outcome in state_outcomes:
                    if isinstance(outcome, str) and outcome not in required_fields:
                        required_fields.append(outcome)
                        break
            if required_group_field and required_group_field not in required_fields:
                required_fields.insert(0, required_group_field)
        if requested_action in {"cross_project_validate", "cross_disease_validate"}:
            required_fields = required_cross_validation_fields(
                context.schemaCatalog,
                requested_action,
                context.questionSummary,
            )
            required_group_field = required_fields[0] if required_fields else None
        elif guarded_action == "execute_read_query" and requested_action in {
            "compare_groups", "stratified_analysis", "adjust_confounders",
        }:
            required_fields = required_group_analysis_fields(
                context.schemaCatalog,
                requested_action,
                context.questionSummary,
            )
            required_group_field = required_fields[0] if required_fields else None
        return context.model_copy(update={
            "approvedActions": [guarded_action],
            "executionFeedback": list(dict.fromkeys(feedback))[-3:],
            "requiredSemanticFields": required_fields[:8],
            "requiredGroupField": required_group_field,
        })

    @staticmethod
    def _decision_result_metadata(decision: object) -> dict[str, object]:
        return {
            "decisionReason": getattr(decision, "decision_reason", None),
            "alternativeActions": tuple(getattr(decision, "alternative_actions", ())),
            "stopReason": getattr(decision, "stop_reason", None),
        }

    def _materialize_decision(
        self,
        context: ScientificPlannerContext,
        decision: object,
        *,
        apply_legacy_guard: bool,
    ) -> ScientificPlannerResult:
        """Materialize one selected action without changing it in the new path."""

        requested_action = decision.selected_action
        if apply_legacy_guard:
            guarded_action, repair_code = _guard_dynamic_action(context, requested_action)
        else:
            # New Decision State policy has already passed hard availability.
            # No scientific strategy rule may replace its selected action.
            guarded_action, repair_code = requested_action, None
        metadata = self._decision_result_metadata(decision)

        if requested_action == "finish" and guarded_action == "finish":
            action = _finish_action(context).model_copy(update={
                "rationale": _bounded_action_rationale(decision.decision_reason),
                "arguments": FinishArguments(
                    actionName="finish",
                    reasonCode=decision.stop_reason,
                ),
            })
            return ScientificPlannerResult(
                action=action,
                mode="sft_policy",
                policyOrigin=getattr(self._decision_planner, "policy_origin", "qwen_model"),
                materializerOrigin="runtime_owned",
                rawAction=requested_action,
                **metadata,
            )

        if guarded_action == "inspect_cohort":
            action = _runtime_owned_inspection_action(
                context,
                rationale=decision.decision_reason,
            )
            return ScientificPlannerResult(
                action=action,
                mode="sft_policy",
                policyOrigin=getattr(self._decision_planner, "policy_origin", "qwen_model"),
                materializerOrigin="runtime_owned",
                fallbackCode=repair_code,
                rawAction=requested_action,
                repairCodes=(repair_code,) if repair_code else (),
                runtimeOwned=True,
                **metadata,
            )

        materialization_context = self._materialization_context(
            context,
            guarded_action,
            requested_action,
        )
        materialized = self._base.plan_action(materialization_context)
        if materialized.mode not in {"model", "deterministic"}:
            raise RuntimeError("DYNAMIC_ACTION_MATERIALIZATION_FAILED")
        if (
            materialized.mode == "deterministic"
            and not self.allow_deterministic_materializer_fallback
        ):
            raise RuntimeError("DYNAMIC_ACTION_MATERIALIZATION_FAILED")
        if apply_legacy_guard and materialized.mode != "model":
            # Preserve the historical Hybrid contract for legacy callers.
            # Only the new Decision-State path may use a deterministic
            # execution-plan fallback, and it remains bound to the exact
            # Action selected by the policy.
            raise RuntimeError("DYNAMIC_ACTION_MATERIALIZATION_FAILED")
        if materialized.action.actionName != guarded_action:
            # A materializer may fail or fall back, but it may never change
            # the high-level Action selected by Qwen.  Fail closed before any
            # executor sees the mismatched family.
            raise RuntimeError("MATERIALIZATION_ACTION_MISMATCH")
        if materialized.action.actionName in {"execute_read_query", "inspect_cohort"} \
                and getattr(materialized.action.arguments, "queryPlan", None) is None:
            raise RuntimeError("DYNAMIC_QUERY_PLAN_REQUIRED")
        action = materialized.action.model_copy(update={
            "rationale": _bounded_action_rationale(decision.decision_reason),
        })
        return ScientificPlannerResult(
            action=action,
            mode="sft_policy",
            fallbackCode=repair_code or getattr(materialized, "fallbackCode", None),
            rawAction=requested_action,
            repairCodes=self._merged_repair_codes(materialized, repair_code),
            policyOrigin=getattr(self._decision_planner, "policy_origin", "qwen_model"),
            materializerOrigin=(
                getattr(self._base, "materializer_origin", "model")
                if materialized.mode == "model"
                else "deterministic_fallback"
            ),
            **metadata,
        )

    def plan_action(self, context: ScientificPlannerContext) -> ScientificPlannerResult:
        """Legacy context entry point retained for compatibility tests."""

        select_action = getattr(self._decision_planner, "select_action", None)
        if not callable(select_action):
            raise RuntimeError("DYNAMIC_ACTION_SELECTOR_NOT_CONFIGURED")
        decision = select_action(context)
        return self._materialize_decision(context, decision, apply_legacy_guard=True)

    def plan_action_with_state(
        self,
        context: ScientificPlannerContext,
        decision_state: ScientificDecisionState | ScientificPolicyInput,
    ) -> ScientificPlannerResult:
        """New policy entry point: Qwen sees only the six-block Decision State."""

        select_action = getattr(self._decision_planner, "select_action", None)
        if not callable(select_action):
            raise RuntimeError("DYNAMIC_ACTION_SELECTOR_NOT_CONFIGURED")
        if isinstance(decision_state, ScientificDecisionState):
            # Keep the materializer on the same validated State snapshot that
            # the policy consumed.  This is a compressed JSON view only; raw
            # Runtime observations remain behind the executor boundary.
            context = context.model_copy(update={
                "decisionState": decision_state.model_dump(mode="json"),
            })
        else:
            # ``ScientificPolicyInput`` is already the six-block, policy-safe
            # view.  Preserve it as a plain payload for materialization while
            # keeping the public planner context backwards compatible.
            context = context.model_copy(update={
                "decisionState": decision_state.model_dump(mode="json"),
            })
        if isinstance(decision_state, ScientificDecisionState) and not context.requiredSemanticFields:
            # Direct callers may invoke the new policy entry point without the
            # graph's precomputed context.  Derive the same Catalog-backed
            # requirements here so the serving boundary cannot silently lose
            # an explicitly requested covariate such as age.
            requirements = derive_data_requirements(
                decision_state,
                context.schemaCatalog,
                unavailable_fields=context.unavailableSemanticFields,
            )
            context = context.model_copy(update={
                "requiredSemanticFields": list(requirements.required_fields),
                # Keep raw reads ungrouped. The grouped field is carried in
                # requiredSemanticFields and is promoted to requiredGroupField
                # only for a later cross-validation/coverage plan.
                "requiredGroupField": context.requiredGroupField,
            })
        decision = select_action(decision_state)
        available = set(
            decision_state.action_space.available_actions
            if isinstance(decision_state, ScientificDecisionState)
            else decision_state.action_space.available_actions
        )
        if decision.selected_action not in available:
            raise RuntimeError("POLICY_ACTION_NOT_AVAILABLE")
        if not set(decision.alternative_actions).issubset(available):
            raise RuntimeError("POLICY_ACTION_NOT_AVAILABLE")
        return self._materialize_decision(context, decision, apply_legacy_guard=False)

    def close(self) -> None:
        for planner in (self._decision_planner, self._base):
            close = getattr(planner, "close", None)
            if callable(close):
                close()


def build_intent_planner(
    env: Mapping[str, str] | None = None,
    transport: httpx.BaseTransport | None = None,
) -> IntentPlannerPort:
    source = os.environ if env is None else env
    base_url = source.get("MICO_RESEARCH_PLANNER_BASE_URL", "").strip()
    model = source.get("MICO_RESEARCH_PLANNER_MODEL", "").strip()
    token = source.get("MICO_RESEARCH_PLANNER_TOKEN", "").strip()
    sft_enabled = source.get("MICO_SFT_POLICY_ENABLED", "false").strip().lower() == "true"
    if sft_enabled and (not base_url or not model or not token):
        raise ValueError("DYNAMIC_RESEARCH_PLANNER_CONFIG_REQUIRED")
    if not base_url or not model or not token:
        base_planner: IntentPlannerPort = DeterministicIntentPlanner()
    else:
        try:
            base_planner = HttpResearchPlannerPort(base_url, model, token, transport=transport)
        except ValueError:
            if sft_enabled:
                raise ValueError("DYNAMIC_RESEARCH_PLANNER_CONFIG_INVALID")
            base_planner = DeterministicIntentPlanner()

    sft_base_url = source.get("MICO_SFT_POLICY_BASE_URL", "").strip()
    sft_model = source.get("MICO_SFT_POLICY_MODEL", "").strip()
    if not sft_enabled:
        return base_planner
    if not sft_base_url or not sft_model:
        raise ValueError("SFT_POLICY_CONFIG_REQUIRED")
    try:
        from mico_agent_runtime.ports.decision_policy import HttpDecisionSftPlannerPort

        decision_planner = HttpDecisionSftPlannerPort(
            sft_base_url,
            sft_model,
            source.get("MICO_SFT_POLICY_TOKEN", "").strip(),
            transport=transport,
        )
        return HybridIntentPlannerPort(base_planner, decision_planner)
    except ValueError as exc:
        raise ValueError("SFT_POLICY_CONFIG_INVALID") from exc
