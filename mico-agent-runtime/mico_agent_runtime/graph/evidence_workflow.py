from __future__ import annotations

from datetime import datetime, timezone

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from mico_agent_runtime.contracts.audit import AuditEvent
from mico_agent_runtime.contracts.evidence import (
    EvidenceQuery,
    EvidenceReference,
    EvidenceReviewReport,
    EvidenceTaskRequest,
)
from mico_agent_runtime.contracts.retrieval import build_retrieval_plan
from mico_agent_runtime.contracts.graph_rag import ReasoningPath
from mico_agent_runtime.graph.evidence_state import EvidenceState
from mico_agent_runtime.ports.evidence import EvidenceSearchError, EvidenceSearchPort


def _identity(state: EvidenceState) -> tuple[str, str]:
    request = state.get("request")
    if isinstance(request, EvidenceTaskRequest):
        return request.traceId, request.runId
    return "invalid", "invalid"


def _audit(state: EvidenceState, node: str, status: str, error_code: str | None = None) -> None:
    trace_id, run_id = _identity(state)
    state.setdefault("auditEvents", []).append(AuditEvent(
        traceId=trace_id,
        runId=run_id,
        node=node,
        toolName="literature_evidence",
        status="COMPLETED" if status in {"COMPLETED", "INSUFFICIENT_EVIDENCE"} else status,
        errorCode=error_code,
        occurredAt=datetime.now(timezone.utc),
    ))


def _fail(state: EvidenceState, node: str, status: str, code: str) -> EvidenceState:
    state["status"] = status
    state["errorCode"] = code
    _audit(state, node, status, code)
    return state


def build_evidence_graph(search_port: EvidenceSearchPort):
    def validate_task(state: EvidenceState) -> EvidenceState:
        try:
            state["request"] = EvidenceTaskRequest.model_validate(state.get("request"))
        except ValidationError:
            return _fail(state, "validate_evidence_task", "REJECTED", "EVIDENCE_TASK_CONTRACT_INVALID")
        _audit(state, "validate_evidence_task", "COMPLETED")
        return state

    def prepare_queries(state: EvidenceState) -> EvidenceState:
        request = state["request"]
        if not isinstance(request, EvidenceTaskRequest):
            return _fail(state, "prepare_evidence_queries", "REJECTED", "EVIDENCE_TASK_CONTRACT_INVALID")
        taxons = request.taxonNames or [None]
        state["queries"] = [
            EvidenceQuery(
                topic=request.topic,
                taxonName=taxon,
                direction=direction,
                retrievalMode=request.retrievalMode,
                retrievalScope=request.retrievalScope,
                limit=request.limit,
            )
            for taxon in taxons
            for direction in request.requestedDirections
        ]
        _audit(state, "prepare_evidence_queries", "COMPLETED")
        return state

    def retrieve(state: EvidenceState) -> EvidenceState:
        found = []
        try:
            for query in state.get("queries", []):
                found.extend(search_port.search(query))
        except EvidenceSearchError:
            return _fail(state, "retrieve_evidence", "FAILED", "EVIDENCE_SOURCE_FAILED")
        except Exception:
            return _fail(state, "retrieve_evidence", "FAILED", "EVIDENCE_SOURCE_FAILED")
        unique = {item.evidenceId: item for item in found}
        state["evidence"] = list(unique.values())
        _audit(
            state,
            "retrieve_evidence",
            "COMPLETED" if unique else "INSUFFICIENT_EVIDENCE",
            "EVIDENCE_INSUFFICIENT" if not unique else None,
        )
        return state

    def review(state: EvidenceState) -> EvidenceState:
        request = state["request"]
        if not isinstance(request, EvidenceTaskRequest):
            return _fail(state, "scientific_review", "FAILED", "EVIDENCE_TASK_CONTRACT_INVALID")
        references = [EvidenceReference(
            evidenceId=item.evidenceId,
            source=item.source,
            externalId=item.externalId,
            title=item.title,
            journal=item.journal,
            publicationYear=item.publicationYear,
            direction=item.direction,
            summary=item.summary,
            evidenceTier=item.evidenceTier,
            retrievalRoute=item.retrievalRoute,
            retrievalModel=item.retrievalModel,
            sourceChunkId=item.sourceChunkId,
            retrievalScore=item.retrievalScore,
            sourceExcerpt=item.sourceExcerpt,
            vectorScore=item.vectorScore,
            sparseScore=item.sparseScore,
            graphScore=item.graphScore,
            rerankScore=item.rerankScore,
            rerankBreakdown=item.rerankBreakdown,
            graphPaths=item.graphPaths,
            reasoningPaths=item.reasoningPaths or [
                ReasoningPath.from_graph_path(path) for path in item.graphPaths
            ],
            retrievalSources=item.retrievalSources,
        ) for item in state.get("evidence", [])[:50]]
        status = "COMPLETED" if references else "INSUFFICIENT_EVIDENCE"
        state["status"] = status
        if status == "INSUFFICIENT_EVIDENCE":
            state["errorCode"] = "EVIDENCE_INSUFFICIENT"
        has_fulltext = any(item.evidenceTier == "fulltext" for item in state.get("evidence", []))
        limitations = [
            "literature_is_external_evidence",
            "retrieval_route_is_provenance_labeled",
            "candidate_taxon_edges_require_review",
            "graph_paths_are_source_bound",
            "multi_hop_claims_require_review",
            "external_evidence_does_not_override_internal_data",
            "evidence_is_not_causal_or_clinical_advice",
        ]
        limitations.insert(1, "fulltext_corpus_v1" if has_fulltext else "metadata_only_until_full_text_review")
        state["report"] = EvidenceReviewReport(
            status=status,
            topic=request.topic,
            queriesExecuted=len(state.get("queries", [])),
            references=references,
            retrievalPlan=build_retrieval_plan(
                query_summary=request.topic,
                query_type="multi_hop" if request.retrievalMode == "graph" else "composite" if request.retrievalMode == "hybrid" else "semantic_fact",
                retrieval_mode=("hybrid" if request.retrievalMode == "auto" else request.retrievalMode),
                retrieval_branches=(
                    ["vector", "graph"] if request.retrievalMode in {"auto", "hybrid"}
                    else [request.retrievalMode]
                ),
                top_k=request.limit,
                retrieval_scope=request.retrievalScope,
            ),
            limitations=limitations,
            nonDiagnostic="not_clinical_diagnostic_or_treatment_advice",
            generationMode="deterministic_grounded",
        )
        _audit(state, "scientific_review", status,
               "EVIDENCE_INSUFFICIENT" if status == "INSUFFICIENT_EVIDENCE" else None)
        return state

    def terminal(state: EvidenceState) -> EvidenceState:
        _audit(state, "terminal", state.get("status", "FAILED"), state.get("errorCode"))
        return state

    def route_after_validate(state: EvidenceState) -> str:
        return "terminal" if state.get("status") else "prepare_evidence_queries"

    def route_after_prepare(state: EvidenceState) -> str:
        return "terminal" if state.get("status") else "retrieve_evidence"

    def route_after_retrieve(state: EvidenceState) -> str:
        return "terminal" if state.get("status") == "FAILED" else "scientific_review"

    builder = StateGraph(EvidenceState)
    builder.add_node("validate_evidence_task", validate_task)
    builder.add_node("prepare_evidence_queries", prepare_queries)
    builder.add_node("retrieve_evidence", retrieve)
    builder.add_node("scientific_review", review)
    builder.add_node("terminal", terminal)
    builder.add_edge(START, "validate_evidence_task")
    builder.add_conditional_edges("validate_evidence_task", route_after_validate)
    builder.add_conditional_edges("prepare_evidence_queries", route_after_prepare)
    builder.add_conditional_edges("retrieve_evidence", route_after_retrieve)
    builder.add_edge("scientific_review", "terminal")
    builder.add_edge("terminal", END)
    return builder.compile()
