from __future__ import annotations

import json

import httpx

from mico_agent_runtime.contracts.evidence import EvidenceQuery
from mico_agent_runtime.contracts.intent import IntentPlannerContext
from mico_agent_runtime.contracts.research import ScientificPlannerContext
from mico_agent_runtime.ports.research_planner import deterministic_intent_route
from mico_agent_runtime.ports.research_planner import _PlannerResponseRejected
from mico_agent_runtime.ports.research_planner import _action_fallback_reason
from mico_agent_runtime.ports.research_planner import HttpResearchPlannerPort
from mico_agent_runtime.knowledge.local_retriever import (
    LocalKnowledgeIndexConfiguration,
    LocalKnowledgeSearchPort,
)

from tests.test_local_knowledge_retrieval import write_index


def _context(question: str):
    return IntentPlannerContext(
        questionSummary=question,
        allowedWorkflows=["knowledge_retrieval"],
    )


def test_structured_intent_classification_selects_vector_graph_or_hybrid() -> None:
    semantic = deterministic_intent_route(_context("文献证据：2型糖尿病与肠道微生物组"))
    assert semantic.queryType == "semantic_fact"
    assert semantic.retrievalBranches == ["vector"]
    assert semantic.retrievalMode == "vector"

    relation = deterministic_intent_route(_context("图谱中的节点关系机制"))
    assert relation.queryType == "relation"
    assert relation.retrievalBranches == ["graph"]

    composite = deterministic_intent_route(_context("文献证据中，糖尿病到微生物代谢通路的多跳关系"))
    assert composite.queryType == "composite"
    assert composite.retrievalMode == "hybrid"
    assert composite.retrievalBranches == ["vector", "graph"]
    assert 0 < composite.routeConfidence <= 1

    comparison = deterministic_intent_route(_context("健康人和某种疾病之间微生物的差别"))
    assert comparison.queryType == "composite"
    assert comparison.retrievalBranches == ["vector", "graph"]


def test_deepseek_planner_accepts_closed_structured_route() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps({
                "workflow": "knowledge_retrieval",
                "responseMode": "structured_evidence_review",
                "safetyProfile": "non_diagnostic",
                "queryType": "composite",
                "routeConfidence": 0.91,
                "retrievalMode": "hybrid",
                "retrievalBranches": ["vector", "graph"],
                "classificationSignals": ["semantic", "relation", "composite", "model"],
            })}}],
        })

    planner = HttpResearchPlannerPort(
        "https://api.deepseek.com",
        "deepseek-v4-flash",
        "test-token",
        transport=httpx.MockTransport(handler),
    )
    result = planner.route_intent(_context("健康人和疾病之间微生物的差别"))
    assert result.mode == "model"
    assert result.plan.queryType == "composite"
    assert result.plan.retrievalBranches == ["vector", "graph"]
    assert seen == ["https://api.deepseek.com/chat/completions"]


def test_gemini_openai_compatible_planner_uses_chat_completions_endpoint() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps({
                "workflow": "knowledge_retrieval",
                "responseMode": "structured_evidence_review",
                "safetyProfile": "non_diagnostic",
                "queryType": "semantic_fact",
                "routeConfidence": 0.9,
                "retrievalMode": "vector",
                "retrievalBranches": ["vector"],
                "classificationSignals": ["semantic", "model"],
            })}}],
        })

    planner = HttpResearchPlannerPort(
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "gemini-3.5-flash",
        "test-token",
        transport=httpx.MockTransport(handler),
    )
    result = planner.route_intent(_context("文献证据检索"))

    assert result.mode == "model"
    assert result.plan.retrievalBranches == ["vector"]
    assert seen == [
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    ]


def test_deepseek_route_retries_three_times_with_schema_feedback() -> None:
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = request.read()
        body = json.loads(payload)
        requests.append(body)
        if len(requests) < 4:
            content = "not-json"
        else:
            content = json.dumps({
                "workflow": "knowledge_retrieval",
                "responseMode": "structured_evidence_review",
                "safetyProfile": "non_diagnostic",
                "queryType": "semantic_fact",
                "routeConfidence": 0.8,
                "retrievalMode": "vector",
                "retrievalBranches": ["vector"],
                "classificationSignals": ["semantic", "model"],
            })
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    planner = HttpResearchPlannerPort(
        "https://api.deepseek.com",
        "deepseek-v4-flash",
        "test-token",
        transport=httpx.MockTransport(handler),
    )
    result = planner.route_intent(_context("文献证据检索"))

    assert result.mode == "model"
    assert len(requests) == 4
    assert any("Validation feedback:" in message["content"]
               for message in requests[-1]["messages"])


def test_deepseek_scientific_action_retries_three_times_with_schema_feedback() -> None:
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read())
        requests.append(body)
        if len(requests) < 4:
            content = "{\"actionName\":\"finish\"}"
        else:
            content = json.dumps({
                "actionId": "action-" + "0" * 32,
                "actionName": "finish",
                "rationale": "证据状态已足够，安全结束",
                "arguments": {
                    "actionName": "finish",
                    "reasonCode": "EVIDENCE_SUFFICIENT",
                },
            })
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    planner = HttpResearchPlannerPort(
        "https://api.deepseek.com",
        "deepseek-v4-flash",
        "test-token",
        transport=httpx.MockTransport(handler),
    )
    result = planner.plan_action(ScientificPlannerContext(
        questionSummary="当前证据是否足够",
        intent="data_fact",
        approvedActions=["finish"],
        remainingActionBudget=1,
        observations=[],
    ))

    assert result.mode == "model"
    assert result.action.actionName == "finish"
    assert len(requests) == 4
    assert any("Validation feedback:" in message["content"]
               for message in requests[-1]["messages"])


def test_scientific_action_fallback_records_redacted_reason_code() -> None:
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.read()))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "not-json"}}],
        })

    planner = HttpResearchPlannerPort(
        "https://api.deepseek.com",
        "deepseek-v4-flash",
        "test-token",
        transport=httpx.MockTransport(handler),
    )
    result = planner.plan_action(ScientificPlannerContext(
        questionSummary="当前证据是否足够",
        intent="data_fact",
        approvedActions=["finish"],
        remainingActionBudget=1,
        observations=[],
    ))

    assert result.mode == "deterministic"
    assert result.fallbackCode == "SCIENTIFIC_PLANNER_DETERMINISTIC_FALLBACK"
    assert result.fallbackReasonCode == "SCIENTIFIC_PLANNER_FALLBACK_REASON_INVALID_JSON"
    assert len(requests) == 4


def test_planner_rate_limit_is_classified_without_response_body() -> None:
    assert _action_fallback_reason(_PlannerResponseRejected(429)) == (
        "SCIENTIFIC_PLANNER_FALLBACK_REASON_RATE_LIMITED"
    )


def test_local_hybrid_runs_two_branches_and_unifies_source_labels(tmp_path) -> None:
    write_index(tmp_path)
    port = LocalKnowledgeSearchPort(LocalKnowledgeIndexConfiguration(tmp_path))
    results = port.search_parallel(EvidenceQuery(
        topic="type 2 diabetes gut microbiome",
        direction="context",
        retrievalMode="hybrid",
        limit=5,
    ))
    assert results
    assert all(item.retrievalRoute == "hybrid" for item in results)
    assert all(item.retrievalSources for item in results)
    assert all(item.rerankScore == item.retrievalScore for item in results)
    assert all(item.rerankScore >= 0 for item in results)


def test_candidate_taxon_path_is_speculative_not_supported(tmp_path) -> None:
    write_index(tmp_path)
    with (tmp_path / "medical_knowledge_graph.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"recordType":"node","nodeId":"candidate_taxon:x","nodeType":"candidate_taxon","label":"Bacteroides testus"}\n')
        handle.write('{"recordType":"edge","edgeId":"edge-x1","source":"term:type_2_diabetes","relation":"ASSOCIATED_WITH_CANDIDATE","target":"candidate_taxon:x","evidenceChunkId":"PMC1-A001"}\n')
        handle.write('{"recordType":"edge","edgeId":"edge-x2","source":"chunk:PMC1-A001","relation":"MENTIONS_CANDIDATE","target":"candidate_taxon:x","evidenceChunkId":"PMC1-A001"}\n')
    port = LocalKnowledgeSearchPort(LocalKnowledgeIndexConfiguration(tmp_path))
    results = port.search(EvidenceQuery(
        topic="type 2 diabetes",
        direction="context",
        retrievalMode="graph",
        limit=5,
    ))
    paths = [path for item in results for path in item.graphPaths]
    assert paths
    assert any(path.status == "speculative" for path in paths)
    assert any(step.supportStatus == "speculative" for path in paths for step in path.hops)


def test_explicit_conflict_edge_is_not_presented_as_supported(tmp_path) -> None:
    write_index(tmp_path)
    with (tmp_path / "medical_knowledge_graph.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"recordType":"edge","edgeId":"edge-conflict","source":"term:type_2_diabetes","relation":"CONTRADICTS","target":"chunk:PMC1-A001","evidenceChunkId":"PMC1-A001"}\n')
    port = LocalKnowledgeSearchPort(LocalKnowledgeIndexConfiguration(tmp_path))
    results = port.search(EvidenceQuery(
        topic="type 2 diabetes",
        direction="context",
        retrievalMode="graph",
        limit=5,
    ))
    paths = [path for item in results for path in item.graphPaths]
    assert any(path.status == "conflicted" for path in paths)
    assert any(step.supportStatus == "conflicted" for path in paths for step in path.hops)
