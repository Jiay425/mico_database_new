from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlsplit

import httpx

from mico_agent_runtime.contracts.generated_analysis import (
    AnalysisPlannerContext,
    GeneratedAnalysisPlan,
    GeneratedAnalysisPlannerResult,
)
from mico_agent_runtime.contracts.intent import IntentPlannerContext, IntentRoutePlan


PLANNER_FALLBACK_CODE = "RESEARCH_PLANNER_DETERMINISTIC_FALLBACK"
DETERMINISTIC_MODE_CODE = "RESEARCH_PLANNER_DETERMINISTIC_MODE"
ANALYSIS_PLANNER_FALLBACK_CODE = "ANALYSIS_PLANNER_DETERMINISTIC_FALLBACK"

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


def _parse_model_json(content: object) -> object:
    if not isinstance(content, str):
        raise ValueError("planner content is not JSON")
    text = re.sub(r"(?is)<think>.*?</think>", "", content).strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) < 3:
            raise ValueError("planner JSON fence is empty")
        text = "\n".join(lines[1:-1]).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            raise
        parsed, end = json.JSONDecoder().raw_decode(text[start:])
        if text[start + end:].strip():
            raise ValueError("trailing non-json planner content")
        return parsed


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
        if (parsed.hostname or "").lower().endswith("deepseek.com") or parsed.path.rstrip("/").endswith("/v1"):
            self._endpoint = base + "/chat/completions"
        else:
            self._endpoint = base + "/v1/chat/completions"
        self._model = model
        self._token = token
        self._disable_reasoning = "deepseek.com" in (parsed.hostname or "").lower()
        self._client = httpx.Client(transport=transport, timeout=30.0)

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
        for attempt in range(2):
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
            except Exception:
                if attempt == 0:
                    body["messages"][0]["content"] += "\nReturn one valid closed JSON object matching the selected approved workflow."
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
            "output identifiers, locators, sample names, disease labels, URLs, payloads or credentials."
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
        except Exception:
            return GeneratedAnalysisPlannerResult(
                plan=deterministic_analysis_plan(),
                mode="deterministic",
                fallbackCode=ANALYSIS_PLANNER_FALLBACK_CODE,
            )

    def close(self) -> None:
        self._client.close()


def build_intent_planner(
    env: Mapping[str, str] | None = None,
    transport: httpx.BaseTransport | None = None,
) -> IntentPlannerPort:
    source = os.environ if env is None else env
    base_url = source.get("MICO_RESEARCH_PLANNER_BASE_URL", "").strip()
    model = source.get("MICO_RESEARCH_PLANNER_MODEL", "").strip()
    token = source.get("MICO_RESEARCH_PLANNER_TOKEN", "").strip()
    if not base_url or not model or not token:
        return DeterministicIntentPlanner()
    try:
        return HttpResearchPlannerPort(base_url, model, token, transport=transport)
    except ValueError:
        return DeterministicIntentPlanner()
