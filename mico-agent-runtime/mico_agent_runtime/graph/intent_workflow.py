from __future__ import annotations

import re
from datetime import datetime, timezone
from hashlib import sha256

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import ValidationError

from mico_agent_runtime.contracts.audit import AuditEvent
from mico_agent_runtime.contracts.evidence import EvidenceQuery
from mico_agent_runtime.contracts.evidence_report import EvidenceReference, EvidenceReviewReport
from mico_agent_runtime.contracts.generated_analysis import AnalysisPlannerContext
from mico_agent_runtime.contracts.graph_rag import (
    EvidenceSynthesisContext,
    EvidenceSynthesisResult,
    ReasoningPath,
    SynthesisEvidence,
)
from mico_agent_runtime.contracts.retrieval import build_retrieval_plan
from mico_agent_runtime.contracts.review import GraphReviewResumeCommand
from mico_agent_runtime.contracts.intent import (
    IntentPlannerContext,
    IntentQueryReport,
    IntentRoutePlan,
    IntentTaskRequest,
)
from mico_agent_runtime.contracts.tools import (
    ExecuteReadQueryArguments,
    ExecuteReadQueryJavaToolCall,
    JavaToolResponse,
)
from mico_agent_runtime.graph.generated_analysis import (
    GeneratedAnalysisError,
    build_analysis_preview,
    execute_generated_analysis,
)
from mico_agent_runtime.graph.intent_state import IntentState
from mico_agent_runtime.governance.guardrails import evaluate_input, evaluate_output, evaluate_workflow
from mico_agent_runtime.ports.java_agent import (
    JavaAgentToolPort,
    JavaPortContractError,
    JavaPortTransportError,
)
from mico_agent_runtime.ports.knowledge import KnowledgeSearchPort
from mico_agent_runtime.ports.research_planner import (
    DETERMINISTIC_MODE_CODE,
    IntentPlannerPort,
    deterministic_intent_route,
)
from mico_agent_runtime.knowledge.synthesis import (
    DeterministicGraphRagSynthesisPort,
    GraphRagSynthesisPort,
)
from mico_agent_runtime.workflow.catalog import get_workflow_tool_plan


_WORKFLOW_SCOPE = {
    "dynamic_read_query": {"mico:query:read"},
    "knowledge_retrieval": {"mico:evidence:read"},
}


def _identity(state: IntentState) -> tuple[str, str]:
    request = state.get("request")
    if isinstance(request, IntentTaskRequest):
        return request.traceId, request.runId
    return "invalid", "invalid"


def _audit(
    state: IntentState,
    node: str,
    status: str,
    *,
    tool_name: str | None = None,
    error_code: str | None = None,
    response: JavaToolResponse | None = None,
) -> None:
    trace_id, run_id = _identity(state)
    snapshot = response.dataSnapshot if response is not None else None
    event = AuditEvent(
        traceId=trace_id,
        runId=run_id,
        node=node,
        toolName=tool_name,
        toolCallId=response.toolCallId if response is not None else None,
        status=status,
        errorCode=error_code,
        dataSnapshotId=snapshot.dataSnapshotId if snapshot else None,
        snapshotPersistence=snapshot.snapshotPersistence if snapshot else None,
        occurredAt=datetime.now(timezone.utc),
    )
    state.setdefault("auditEvents", []).append(event)
    callback = state.get("progressCallback")
    if callable(callback):
        callback(event)


def _fail(state: IntentState, node: str, status: str, code: str) -> IntentState:
    state["status"] = status
    state["errorCode"] = code
    state.pop("report", None)
    state.pop("toolResult", None)
    _audit(state, node, status, error_code=code)
    return state


def redact_for_planner(question: str) -> str:
    value = re.sub(r"(?i)https?://\S+|file://\S+|Bearer\s+\S+", "[redacted]", question)
    value = re.sub(r"(?i)(sourceSampleId|internalRecordId|cohortCondition)\s*=\s*[^\s,;]+", "[redacted]", value)
    value = re.sub(r"(?i)\b(?:SRR|ERR|DRR)\d+\b", "[sample-id]", value)
    value = re.sub(r"(?i)\bMV_[A-Za-z0-9_-]+\b", "[sample-id]", value)
    value = re.sub(r"\b\d{4,}\b", "[number]", value)
    return value[:1200]


def _validate_task(state: IntentState) -> IntentState:
    try:
        state["request"] = IntentTaskRequest.model_validate(state.get("request"))
    except ValidationError:
        return _fail(state, "validate_user_task", "REJECTED", "INTENT_TASK_CONTRACT_INVALID")
    _audit(state, "validate_user_task", "COMPLETED")
    return state


def _policy_gate(state: IntentState) -> IntentState:
    request = state["request"]
    assert isinstance(request, IntentTaskRequest)
    decision = evaluate_input(request.question)
    if decision.verdict == "REJECT":
        return _fail(state, "policy_gate", "REJECTED", decision.code)
    _audit(state, "policy_gate", "COMPLETED")
    return state


def _plan_research(planner: IntentPlannerPort, state: IntentState) -> IntentState:
    request = state["request"]
    assert isinstance(request, IntentTaskRequest)
    context = IntentPlannerContext(
        questionSummary=redact_for_planner(request.question),
        allowedWorkflows=request.allowedWorkflows,
    )
    try:
        result = planner.route_intent(context)
    except Exception:
        try:
            plan = deterministic_intent_route(context)
        except Exception:
            return _fail(state, "recognize_intent", "REJECTED", "UNSUPPORTED_INTENT")
        state["routePlan"] = plan
        state["workflow"] = plan.workflow
        state["toolPlan"] = get_workflow_tool_plan(plan.workflow)
        state["plannerMode"] = "deterministic"
        _audit(state, "recognize_intent", "COMPLETED", error_code=DETERMINISTIC_MODE_CODE)
        return state

    state["routePlan"] = result.plan
    state["workflow"] = result.plan.workflow
    state["toolPlan"] = get_workflow_tool_plan(result.plan.workflow)
    state["plannerMode"] = result.mode
    _audit(state, "recognize_intent", "COMPLETED", error_code=result.fallbackCode)
    decision = evaluate_workflow(
        result.plan.workflow,
        request.allowedWorkflows,
        request.requestedScopes,
        _WORKFLOW_SCOPE[result.plan.workflow],
    )
    if decision.verdict == "REJECT":
        return _fail(state, "policy_gate_after_planner", "REJECTED", decision.code)
    if not state["toolPlan"].implemented:
        return _fail(state, "execute_selected_workflow", "NOT_IMPLEMENTED", "WORKFLOW_NOT_IMPLEMENTED")
    return state


def _safe_query_columns(columns: object) -> list[str]:
    if not isinstance(columns, list):
        return []
    sensitive = {
        "patient_id", "patient_name", "sample_id", "source_sample_id",
        "sourceSampleId", "internal_record_id", "internalRecordId",
        "raw_metadata", "cohortCondition",
    }
    result: list[str] = []
    for index, value in enumerate(columns[:128], start=1):
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", value):
            result.append(f"column_{index}")
        elif value.lower() in {item.lower() for item in sensitive}:
            result.append(f"column_{index}")
        else:
            result.append(value)
    return result


def _execute_knowledge_workflow(
    knowledge_port: KnowledgeSearchPort | None,
    synthesis_port: GraphRagSynthesisPort | None,
    state: IntentState,
) -> IntentState:
    request = state["request"]
    assert isinstance(request, IntentTaskRequest)
    if knowledge_port is None:
        return _fail(state, "execute_knowledge_retrieval", "FAILED", "KNOWLEDGE_SOURCE_NOT_CONFIGURED")
    route_plan = state.get("routePlan")
    retrieval_mode = (
        route_plan.retrievalMode
        if isinstance(route_plan, IntentRoutePlan)
        else "hybrid"
    )
    try:
        query = EvidenceQuery(
            topic=request.question,
            direction="context",
            retrievalMode=retrieval_mode,
            limit=10,
        )
        branches = (
            tuple(route_plan.retrievalBranches)
            if isinstance(route_plan, IntentRoutePlan) and route_plan.retrievalBranches
            else ("vector", "graph") if retrieval_mode == "hybrid" else (retrieval_mode,)
        )
        retrieval_plan = build_retrieval_plan(
            query_summary=redact_for_planner(request.question),
            query_type=(route_plan.queryType if isinstance(route_plan, IntentRoutePlan) else "semantic_fact"),
            retrieval_mode=retrieval_mode,
            retrieval_branches=list(branches),
            top_k=10,
        )
        parallel_search = getattr(knowledge_port, "search_parallel", None)
        if callable(parallel_search):
            try:
                evidence = parallel_search(query, branches=branches, plan=retrieval_plan)
            except TypeError:
                # Compatibility for older injected ports; production ports
                # implement the closed RetrievalPlan argument.
                evidence = parallel_search(query, branches=branches)
        else:
            # Compatibility for injected ports that implement the original
            # single-call protocol; the production local index uses the
            # explicit parallel branch method above.
            evidence = knowledge_port.search(query)
    except Exception:
        return _fail(state, "execute_knowledge_retrieval", "FAILED", "KNOWLEDGE_SOURCE_FAILED")

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
        graphScore=item.graphScore,
        rerankScore=item.rerankScore,
        rerankBreakdown=item.rerankBreakdown,
        graphPaths=item.graphPaths,
        reasoningPaths=item.reasoningPaths or [
            ReasoningPath.from_graph_path(path) for path in item.graphPaths
        ],
        retrievalSources=item.retrievalSources,
    ) for item in evidence[:10]]
    if references:
        synthesis_references = references[:4]
        if not any(reference.graphPaths for reference in synthesis_references):
            graph_reference = next((reference for reference in references if reference.graphPaths), None)
            if graph_reference is not None:
                synthesis_references = synthesis_references[:-1] + [graph_reference]
        synthesis = (synthesis_port or DeterministicGraphRagSynthesisPort()).synthesize(
            EvidenceSynthesisContext(
                questionSummary=redact_for_planner(request.question),
                evidence=[SynthesisEvidence(
                    evidenceId=reference.evidenceId,
                    title=reference.title,
                    summary=reference.summary,
                    graphPaths=reference.graphPaths,
                    reasoningPaths=reference.reasoningPaths,
                ) for reference in synthesis_references],
            )
        )
    else:
        synthesis = EvidenceSynthesisResult(
            claims=[],
            reasoningSteps=[],
            conclusion="No source-bound literature evidence was retrieved; no conclusion is formed.",
            mode="deterministic",
            fallbackCode="GRAPHRAG_SYNTHESIS_NO_EVIDENCE",
        )
    report = EvidenceReviewReport(
        status="COMPLETED" if references else "INSUFFICIENT_EVIDENCE",
        topic=redact_for_planner(request.question),
        queryType=(route_plan.queryType if isinstance(route_plan, IntentRoutePlan) else "semantic_fact"),
        routeConfidence=(route_plan.routeConfidence if isinstance(route_plan, IntentRoutePlan) else 0.0),
        retrievalBranches=(route_plan.retrievalBranches if isinstance(route_plan, IntentRoutePlan) else []),
        retrievalPlan=retrieval_plan,
        queriesExecuted=1,
        references=references,
        limitations=[
            "literature_is_external_evidence",
            "fulltext_corpus_v1",
            "retrieval_route_is_provenance_labeled",
            "candidate_taxon_edges_require_review",
            "graph_paths_are_source_bound",
            "multi_hop_claims_require_review",
            "external_evidence_does_not_override_internal_data",
            "evidence_is_not_causal_or_clinical_advice",
        ],
        nonDiagnostic="not_clinical_diagnostic_or_treatment_advice",
        generationMode="model_grounded" if synthesis.mode == "model" else "deterministic_grounded",
        groundedClaims=synthesis.claims,
        reasoningSteps=synthesis.reasoningSteps,
        conclusion=synthesis.conclusion,
    )
    review_paths = [
        path
        for reference in references
        for path in reference.graphPaths
        if path.status != "supported" or path.confidence < 0.75
    ]
    review_ids = [
        "review-" + sha256(f"evidence-path|{path.pathId}".encode("utf-8")).hexdigest()[:32]
        for path in review_paths[:8]
    ]
    state["reviewRequired"] = bool(review_paths)
    state["reviewPayload"] = {
        "type": "GRAPH_EVIDENCE_REVIEW_REQUIRED",
        "code": "GRAPH_REVIEW_REQUIRED",
        "reviewIds": review_ids,
        "pathIds": [path.pathId for path in review_paths[:8]],
        "pathStatuses": [path.status for path in review_paths[:8]],
        "replayable": False,
    }
    state["status"] = "COMPLETED"
    state["report"] = report
    _audit(state, "execute_knowledge_retrieval", "COMPLETED", tool_name="literature_evidence")
    return state


def _execute_selected_workflow(
    java_port: JavaAgentToolPort,
    planner: IntentPlannerPort,
    knowledge_port: KnowledgeSearchPort | None,
    synthesis_port: GraphRagSynthesisPort | None,
    state: IntentState,
) -> IntentState:
    request = state["request"]
    assert isinstance(request, IntentTaskRequest)
    route_plan = state.get("routePlan")
    if isinstance(route_plan, IntentRoutePlan) and route_plan.workflow == "knowledge_retrieval":
        return _execute_knowledge_workflow(knowledge_port, synthesis_port, state)
    if not isinstance(route_plan, IntentRoutePlan) or not route_plan.sqlDraft:
        return _fail(state, "execute_selected_workflow", "REJECTED", "QUERY_PLAN_REQUIRED")

    call_id = "call-" + sha256(
        f"{request.traceId}|{request.taskId}|execute_read_query".encode()
    ).hexdigest()[:32]
    call = ExecuteReadQueryJavaToolCall(
        toolName="execute_read_query",
        runId=request.runId,
        toolCallId=call_id,
        arguments=ExecuteReadQueryArguments(sql=route_plan.sqlDraft, limit=1000),
    )
    try:
        response = java_port.execute(call)
    except JavaPortContractError as exc:
        return _fail(state, "execute_selected_workflow", "REJECTED", exc.code)
    except JavaPortTransportError as exc:
        return _fail(state, "execute_selected_workflow", "FAILED", exc.code)
    except Exception:
        return _fail(state, "execute_selected_workflow", "FAILED", "JAVA_TOOL_EXECUTION_FAILED")

    if response.runId != call.runId or response.toolCallId != call.toolCallId:
        return _fail(state, "execute_selected_workflow", "FAILED", "JAVA_TOOL_RESPONSE_MISMATCH")
    if response.status != "COMPLETED" or response.dataSnapshot is None:
        return _fail(
            state,
            "execute_selected_workflow",
            response.status,
            response.error.code if response.error else response.status,
        )
    snapshot = response.dataSnapshot
    if snapshot.snapshotPersistence != "transient":
        return _fail(state, "execute_selected_workflow", "FAILED", "EVIDENCE_PERSISTENCE_UNSUPPORTED")

    try:
        preview_columns, preview_rows = build_analysis_preview(response.data)
        if not hasattr(planner, "generate_analysis"):
            return _fail(state, "generate_python_analysis", "FAILED", "ANALYSIS_PLANNER_NOT_CONFIGURED")
        analysis_result = planner.generate_analysis(AnalysisPlannerContext(
            questionSummary=redact_for_planner(request.question),
            workflow="dynamic_read_query",
            columns=preview_columns,
            previewRows=preview_rows,
        ))
        generated_analysis = execute_generated_analysis(
            analysis_result.plan,
            preview_rows,
            response.rowCount if response.rowCount is not None else snapshot.rowCount,
            planner_mode=analysis_result.mode,
        )
    except GeneratedAnalysisError:
        return _fail(state, "generate_python_analysis", "FAILED", "ANALYSIS_CODE_REJECTED")
    except (ValidationError, TypeError, ValueError):
        return _fail(state, "generate_python_analysis", "FAILED", "ANALYSIS_SCHEMA_INVALID")

    raw_columns = response.data.get("columns", []) if isinstance(response.data, dict) else []
    report = IntentQueryReport(
        status="COMPLETED",
        workflow="dynamic_read_query",
        source=response.source or "java_agent_read_model",
        rowCount=response.rowCount if response.rowCount is not None else snapshot.rowCount,
        columns=_safe_query_columns(raw_columns),
        dataSnapshotId=snapshot.dataSnapshotId,
        snapshotPersistence=snapshot.snapshotPersistence,
        queryHash=snapshot.queryHash,
        generatedAt=snapshot.generatedAt,
        schemaVersion=response.schemaVersion or "v1",
        limitations=[
            "query_is_model_proposed_and_java_validated",
            "snapshot_is_transient_and_not_replayable",
            "query_result_is_not_a_clinical_conclusion",
            "bounded_result_projection_only",
            "dynamic_python_analysis_is_bounded_and_sandboxed",
        ],
        nonDiagnostic="not_clinical_diagnostic_or_treatment_advice",
        generatedAnalysis=generated_analysis,
    )
    if evaluate_output(report.model_dump(mode="json")).verdict == "REJECT":
        return _fail(state, "output_guardrail", "REJECTED", "OUTPUT_SAFETY_BLOCKED")
    state["status"] = "COMPLETED"
    state["report"] = report
    _audit(state, "execute_selected_workflow", "COMPLETED", tool_name="execute_read_query", response=response)
    return state


def _route_after_validation(state: IntentState) -> str:
    return "policy_gate" if state.get("status") is None else "terminal"


def _route_after_policy(state: IntentState) -> str:
    return "recognize_intent" if state.get("status") is None else "terminal"


def _route_after_plan(state: IntentState) -> str:
    return "execute_selected_workflow" if state.get("status") is None else "terminal"


def _route_after_execution(state: IntentState, interrupt_on_review: bool) -> str:
    if (
        interrupt_on_review
        and state.get("status") == "COMPLETED"
        and state.get("reviewRequired")
    ):
        return "prepare_human_review"
    return "terminal"


def _prepare_human_review(state: IntentState) -> IntentState:
    """Commit the waiting state before the interrupt control signal fires."""

    state["status"] = "WAITING_APPROVAL"
    state["errorCode"] = "GRAPH_REVIEW_REQUIRED"
    _audit(state, "human_review", "WAITING_APPROVAL", error_code="GRAPH_REVIEW_REQUIRED")
    return state


def _human_review(state: IntentState, *, checkpoint_enabled: bool) -> IntentState:
    if not checkpoint_enabled:
        return _fail(state, "human_review", "FAILED", "GRAPH_REVIEW_CHECKPOINT_REQUIRED")
    # Do not catch the LangGraph interrupt exception: it is the control signal
    # that persists the checkpoint and returns WAITING_APPROVAL to the caller.
    resumed = interrupt(state.get("reviewPayload") or {
        "type": "GRAPH_EVIDENCE_REVIEW_REQUIRED",
        "code": "GRAPH_REVIEW_REQUIRED",
        "replayable": False,
    })
    try:
        command = GraphReviewResumeCommand.model_validate(resumed)
        expected_ids = set((state.get("reviewPayload") or {}).get("reviewIds", []))
        if expected_ids and command.reviewId not in expected_ids:
            return _fail(state, "human_review", "REJECTED", "GRAPH_REVIEW_DECISION_INVALID")
    except ValidationError:
        return _fail(state, "human_review", "REJECTED", "GRAPH_REVIEW_DECISION_INVALID")
    state["reviewDecision"] = command
    if command.decision == "REJECTED":
        return _fail(state, "human_review", "REJECTED", "GRAPH_REVIEW_REJECTED")
    state["status"] = "COMPLETED"
    # LangGraph state updates are additive; assigning None is required to
    # clear the waiting code when the checkpoint resumes.
    state["errorCode"] = None
    _audit(state, "human_review", "COMPLETED", error_code="GRAPH_REVIEW_APPROVED")
    return state


def _terminal(state: IntentState) -> IntentState:
    _audit(state, "terminal", state.get("status", "FAILED"), error_code=state.get("errorCode"))
    return state


def build_intent_graph(
    java_port: JavaAgentToolPort,
    planner: IntentPlannerPort,
    checkpointer=None,
    knowledge_port: KnowledgeSearchPort | None = None,
    synthesis_port: GraphRagSynthesisPort | None = None,
    interrupt_on_review: bool = False,
):
    checkpoint_enabled = checkpointer is not None
    builder = StateGraph(IntentState)
    builder.add_node("validate_user_task", _validate_task)
    builder.add_node("policy_gate", _policy_gate)
    builder.add_node("recognize_intent", lambda state: _plan_research(planner, state))
    builder.add_node(
        "execute_selected_workflow",
        lambda state: _execute_selected_workflow(
            java_port, planner, knowledge_port, synthesis_port, state
        ),
    )
    builder.add_node(
        "prepare_human_review",
        _prepare_human_review,
    )
    builder.add_node(
        "human_review",
        lambda state: _human_review(state, checkpoint_enabled=checkpoint_enabled),
    )
    builder.add_node("terminal", _terminal)
    builder.add_edge(START, "validate_user_task")
    builder.add_conditional_edges("validate_user_task", _route_after_validation)
    builder.add_conditional_edges("policy_gate", _route_after_policy)
    builder.add_conditional_edges("recognize_intent", _route_after_plan)
    builder.add_conditional_edges(
        "execute_selected_workflow",
        lambda state: _route_after_execution(state, interrupt_on_review),
    )
    builder.add_edge("prepare_human_review", "human_review")
    builder.add_edge("human_review", "terminal")
    builder.add_edge("terminal", END)
    return builder.compile(checkpointer=checkpointer)
