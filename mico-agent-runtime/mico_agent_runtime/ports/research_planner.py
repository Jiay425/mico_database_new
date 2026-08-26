from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Protocol
from urllib.parse import urlsplit

import httpx

from mico_agent_runtime.contracts.generated_analysis import (
    AnalysisPlannerContext,
    GeneratedAnalysisPlan,
    GeneratedAnalysisPlannerResult,
)
from mico_agent_runtime.contracts.intent import IntentPlannerContext, IntentRoutePlan
from mico_agent_runtime.contracts.research import (
    ScientificAction,
    ScientificPlannerContext,
    validate_scientific_action,
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


class _PlannerResponseRejected(ValueError):
    def __init__(self, status_code: int) -> None:
        super().__init__("planner response rejected")
        self.status_code = status_code


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

    def plan_action(self, context: ScientificPlannerContext) -> ScientificPlannerResult:
        return ScientificPlannerResult(
            action=_deterministic_action(context),
            mode="deterministic",
            fallbackCode=SCIENTIFIC_PLANNER_DETERMINISTIC_FALLBACK,
        )


class HttpResearchPlannerPort:
    """OpenAI-compatible intent and Python planner with closed JSON outputs."""

    def __init__(self, base_url: str, model: str, token: str,
                 transport: httpx.BaseTransport | None = None) -> None:
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username:
            raise ValueError("planner URL is invalid")
        if parsed.query or parsed.fragment or not model or not token.strip():
            raise ValueError("planner configuration is invalid")
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
        self._disable_reasoning = "deepseek.com" in (parsed.hostname or "").lower()
        # Real collection runs may have two independent cases in flight.  Give
        # the provider enough time to answer without turning a transient slow
        # response into a deterministic fallback; retries still remain bounded
        # by MAX_MODEL_RETRIES.
        self._client = httpx.Client(transport=transport, timeout=60.0)

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
                response = self._client.post(
                    self._endpoint,
                    headers={"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"},
                    json=body,
                )
                if response.status_code in {400, 422}:
                    fallback_body = dict(body)
                    fallback_body.pop("response_format", None)
                    response = self._client.post(
                        self._endpoint,
                        headers={"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"},
                        json=fallback_body,
                    )
                    if response.status_code in {400, 422} and "thinking" in fallback_body:
                        fallback_body.pop("thinking", None)
                        response = self._client.post(
                            self._endpoint,
                            headers={"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"},
                            json=fallback_body,
                        )
                if response.status_code != 200:
                    raise ValueError("planner response rejected")
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
                if attempt < MAX_MODEL_RETRIES:
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
        }
        if self._disable_reasoning:
            body["thinking"] = {"type": "disabled"}
        last_fallback_reason = "SCIENTIFIC_PLANNER_FALLBACK_REASON_UNKNOWN"
        for attempt in range(MAX_MODEL_RETRIES + 1):
            try:
                response = self._client.post(
                    self._endpoint,
                    headers={"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"},
                    json=body,
                )
                if response.status_code != 200:
                    raise ValueError("planner response rejected")
                payload = response.json()
                plan_payload = _parse_model_json(payload["choices"][0]["message"]["content"])
                if isinstance(plan_payload, dict) and isinstance(plan_payload.get("code"), str):
                    plan_payload = dict(plan_payload)
                    plan_payload["code"] = re.sub(r"[\r\n\t]+", " ", plan_payload["code"]).strip()
                return GeneratedAnalysisPlannerResult(
                    plan=GeneratedAnalysisPlan.model_validate(plan_payload), mode="model"
                )
            except Exception as error:
                if attempt < MAX_MODEL_RETRIES:
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

    def plan_action(self, context: ScientificPlannerContext) -> ScientificPlannerResult:
        """Choose one generic capability from the current redacted state.

        The model may propose SQL only inside the dedicated action contract.
        It cannot choose scopes, tools outside the approved action list, or
        inject raw Java observations into the prompt.
        """
        system = (
            "Return only one closed JSON ScientificAction object with actionId, actionName, rationale, and arguments. "
            "Allowed actionName values are execute_read_query, inspect_cohort, compare_groups, "
            "stratified_analysis, adjust_confounders, cross_project_validate, cross_disease_validate, "
            "retrieve_evidence, analyze_projection, finish, but the action must be in approvedActions. "
            "The root actionName and arguments.actionName are both required and must be identical. "
            "For every projection analysis action, emit all five argument keys exactly as named; "
            "use [] for dimensions or confounders only when that field is not required. "
            "The response must use exactly that root/arguments nesting; do not output a wrapper such as "
            "selected_action, do not output placeholder IDs or ellipses, and copy every observationId "
            "only from the current context. "
            "For stratified_analysis provide a non-empty dimensions array; for adjust_confounders "
            "provide a non-empty confounders array; for either cross validation provide at least two observationIds. "
            "Before cross_project_validate or cross_disease_validate, count the supplied observations: if fewer "
            "than two VALIDATED observations are present, do not choose cross validation yet; choose an approved "
            "bounded read such as execute_read_query to obtain the second Java observation first. Never fabricate "
            "an observationId. For stratified_analysis use a non-empty generic dimension such as country or age; "
            "for adjust_confounders use non-empty generic confounders such as age or gender. These arrays must "
            "contain field names only, never labels, values, IDs, or placeholders. "
            "Use these exact argument shapes: execute_read_query/inspect_cohort -> "
            "{actionName, sql, limit}; compare_groups/stratified_analysis/adjust_confounders/"
            "cross_project_validate/cross_disease_validate -> {actionName, observationIds, analysisGoal, "
            "dimensions, confounders}; analyze_projection -> {actionName, observationId, analysisGoal}; "
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
            "execute_read_query and inspect_cohort arguments.sql may contain only a bounded explicit-column SELECT "
            "or WITH proposal ending in a literal LIMIT 1 through 1000; Java will perform the final policy validation. "
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
            "Re-plan once with a fresh, simpler catalog-bounded read: use one catalog sourceTable, explicit "
            "verified fields, no invented joins or filters, and do not repeat the prior query. The feedback is "
            "not data or evidence and must not appear in the final answer. "
            "The following is the Java-provided metadata-only Schema Semantic Catalog. Use only its semantic "
            "field, sourceTable and join names when proposing SQL; every FROM/JOIN table must exactly match a "
            "catalog sourceTable, and it contains no sample values and is not permission to "
            "bypass Java validation.\n\n"
            + (context.schemaCatalog.model_dump_json() if context.schemaCatalog is not None
               else "No Schema Semantic Catalog is available; do not invent schema names.")
        )
        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": context.model_dump_json(exclude_none=True)},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": 1800,
            "temperature": 0,
        }
        if self._disable_reasoning:
            body["thinking"] = {"type": "disabled"}
        for attempt in range(MAX_MODEL_RETRIES + 1):
            try:
                response = self._client.post(
                    self._endpoint,
                    headers={"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"},
                    json=body,
                )
                if response.status_code != 200:
                    raise _PlannerResponseRejected(response.status_code)
                payload = response.json()
                parsed_payload = _parse_model_json(payload["choices"][0]["message"]["content"])
                if not isinstance(parsed_payload, dict):
                    raise ValueError("scientific planner output is not an object")
                parsed_payload = dict(parsed_payload)
                # Action correlation is Runtime-owned. A model may choose the
                # closed action kind and its bounded arguments, but it must never
                # be able to forge or destabilize the audit identifier.
                parsed_payload["actionId"] = "action-" + sha256(
                    (context.questionSummary + "|" + str(context.remainingActionBudget)).encode("utf-8")
                ).hexdigest()[:32]
                action = validate_scientific_action(parsed_payload)
                if action.actionName in {"execute_read_query", "inspect_cohort"}:
                    _validate_catalog_sql(action.arguments.sql, context)
                if action.actionName not in context.approvedActions:
                    raise ValueError("scientific action is not approved")
                return ScientificPlannerResult(action=action, mode="model")
            except Exception as error:
                last_fallback_reason = _action_fallback_reason(error)
                if attempt < MAX_MODEL_RETRIES:
                    body["messages"].append({
                        "role": "user",
                        "content": (
                            "The previous ScientificAction was rejected by the closed contract. "
                            f"Validation feedback: {_planner_error_feedback(error)}. "
                            "Retry with the exact argument shape above, including arguments.actionName "
                            "matching the root actionName, and return JSON only."
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


class HybridIntentPlannerPort:
    """Keep routing/analysis on the configured planner and optionally route
    only Scientific Decision policy through the SFT adapter.

    The SFT path is opt-in and owns its own safe fallback to ``base``.  This
    makes disabling the endpoint a configuration-only rollback to the exact
    pre-SFT planner behavior.
    """

    def __init__(self, base: IntentPlannerPort, decision_planner: IntentPlannerPort) -> None:
        self._base = base
        self._decision_planner = decision_planner

    # Scientific workflow normally uses a deterministic semantic path before
    # asking a planner.  When this explicit SFT wrapper is installed, ask the
    # bounded decision policy first and retain the semantic path as a guard.
    # This keeps Base Runtime behavior unchanged while making Runtime SFT
    # evaluation measure a real policy call rather than a no-op wrapper.
    prefer_policy_action = True

    def route_intent(self, context: IntentPlannerContext) -> IntentPlannerResult:
        return self._base.route_intent(context)

    def generate_analysis(self, context: AnalysisPlannerContext) -> GeneratedAnalysisPlannerResult:
        return self._base.generate_analysis(context)

    def plan_action(self, context: ScientificPlannerContext) -> ScientificPlannerResult:
        return self._decision_planner.plan_action(context)

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
    if not base_url or not model or not token:
        base_planner: IntentPlannerPort = DeterministicIntentPlanner()
    else:
        try:
            base_planner = HttpResearchPlannerPort(base_url, model, token, transport=transport)
        except ValueError:
            base_planner = DeterministicIntentPlanner()

    sft_enabled = source.get("MICO_SFT_POLICY_ENABLED", "false").strip().lower() == "true"
    sft_base_url = source.get("MICO_SFT_POLICY_BASE_URL", "").strip()
    sft_model = source.get("MICO_SFT_POLICY_MODEL", "").strip()
    if not sft_enabled or not sft_base_url or not sft_model:
        return base_planner
    try:
        from mico_agent_runtime.ports.decision_policy import HttpDecisionSftPlannerPort

        decision_planner = HttpDecisionSftPlannerPort(
            sft_base_url,
            sft_model,
            source.get("MICO_SFT_POLICY_TOKEN", "").strip(),
            fallback=base_planner,
            transport=transport,
        )
        return HybridIntentPlannerPort(base_planner, decision_planner)
    except ValueError:
        return base_planner
