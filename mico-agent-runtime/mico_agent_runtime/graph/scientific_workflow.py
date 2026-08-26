from __future__ import annotations

import re
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from mico_agent_runtime.contracts.audit import AuditEvent
from mico_agent_runtime.contracts.analysis import (
    AnalysisEvidence,
    AnalysisPlan,
    AnalysisResult,
    AnalysisSourceMetadata,
)
from mico_agent_runtime.contracts.evidence import EvidenceQuery
from mico_agent_runtime.contracts.graph_rag import (
    EvidenceSynthesisContext,
    EvidenceSynthesisResult,
    SynthesisEvidence,
)
from mico_agent_runtime.contracts.research import (
    AdjustConfoundersAction,
    AnalyzeProjectionAction,
    AnalyzeProjectionArguments,
    CompareGroupsAction,
    CrossDiseaseValidateAction,
    CrossProjectValidateAction,
    ExecuteReadQueryAction,
    FinishAction,
    FinishArguments,
    InspectCohortAction,
    Observation,
    ObservationMetric,
    ResearchEvidenceBinding,
    ResearchExplorationReport,
    ResearchReasoningStep,
    ResearchTask,
    ScientificAction,
    ScientificObservationSummary,
    ScientificPlannerContext,
    StratifiedAnalysisAction,
)
from mico_agent_runtime.contracts.schema_catalog import SchemaSemanticCatalog
from mico_agent_runtime.contracts.trace_eval import TraceDecision
from mico_agent_runtime.contracts.evidence import LiteratureEvidenceItem
from mico_agent_runtime.contracts.unified_evidence import merge_unified_evidence
from mico_agent_runtime.contracts.tools import ExecuteReadQueryArguments, ExecuteReadQueryJavaToolCall
from mico_agent_runtime.graph.generated_analysis import (
    GeneratedAnalysisError,
    build_analysis_preview,
    execute_generated_analysis,
)
from mico_agent_runtime.graph.scientific_state import ScientificState
from mico_agent_runtime.governance.guardrails import evaluate_input
from mico_agent_runtime.knowledge.synthesis import (
    DeterministicGraphRagSynthesisPort,
    GraphRagSynthesisPort,
)
from mico_agent_runtime.ports.java_agent import (
    JavaAgentToolPort,
    JavaPortContractError,
    JavaPortTransportError,
)
from mico_agent_runtime.ports.knowledge import KnowledgeSearchPort
from mico_agent_runtime.ports.scientific_planner import (
    DeterministicScientificPlanner,
    ScientificPlannerResult,
    ScientificPlannerPort,
    _finish_action,
    semantic_scientific_next_action,
)
from mico_agent_runtime.ports.decision_policy import build_decision_policy_state
from mico_agent_runtime.ports.research_planner import deterministic_analysis_plan


_ACTION_SCOPE = {
    "execute_read_query": "mico:query:read",
    "inspect_cohort": "mico:query:read",
    "compare_groups": "mico:research:read",
    "stratified_analysis": "mico:research:read",
    "adjust_confounders": "mico:research:read",
    "cross_project_validate": "mico:research:read",
    "cross_disease_validate": "mico:research:read",
    "retrieve_evidence": "mico:evidence:read",
    "analyze_projection": "mico:research:read",
    "finish": None,
}

_MAX_JAVA_READ_REPLANS = 3
_MAX_ANALYSIS_EXECUTION_ATTEMPTS = 3
_REPLANNABLE_JAVA_READ_CODES = frozenset({
    "READ_PLAN_REPEAT_REJECTED",
    "READ_MODEL_ARGUMENT_REJECTED",
})


def _identity(state: ScientificState) -> tuple[str, str]:
    request = state.get("request")
    if isinstance(request, ResearchTask):
        return request.traceId, request.runId
    return "invalid", "invalid"


def _audit(
    state: ScientificState,
    node: str,
    status: str,
    *,
    action_name: str | None = None,
    tool_name: str | None = None,
    tool_call_id: str | None = None,
    error_code: str | None = None,
    response: Any = None,
) -> None:
    trace_id, run_id = _identity(state)
    snapshot = getattr(response, "dataSnapshot", None)
    if action_name is None:
        current_action = state.get("currentAction")
        action_name = getattr(current_action, "actionName", None)
    event = AuditEvent(
        traceId=trace_id,
        runId=run_id,
        node=node,
        actionName=action_name,
        toolName=tool_name,
        toolCallId=tool_call_id,
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


def _fail(
    state: ScientificState,
    node: str,
    status: str,
    code: str,
    *,
    tool_name: str | None = None,
    tool_call_id: str | None = None,
) -> ScientificState:
    state["status"] = status
    state["errorCode"] = code
    state["route"] = "terminal"
    state.pop("report", None)
    state.pop("pendingObservation", None)
    state.pop("pendingPayload", None)
    _audit(
        state,
        node,
        status,
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        error_code=code,
    )
    return state


def _redact_question(value: str) -> str:
    result = re.sub(r"(?i)https?://\S+|file://\S+|Bearer\s+\S+", "[redacted]", value)
    result = re.sub(r"(?i)(sourceSampleId|internalRecordId|cohortCondition)\s*=\s*[^\s,;]+", "[redacted]", result)
    result = re.sub(r"(?i)\b(?:SRR|ERR|DRR)\d+\b|\bMV_[A-Za-z0-9_-]+\b", "[redacted]", result)
    return result[:1200]


def _validate_task(state: ScientificState) -> ScientificState:
    try:
        state["request"] = ResearchTask.model_validate(state.get("request"))
        state.setdefault("observations", [])
        state.setdefault("rawObservationPayloads", {})
        state.setdefault("analysisPlans", [])
        state.setdefault("analysisResults", [])
        state.setdefault("analysisEvidence", [])
        state.setdefault("actionHistory", [])
        state.setdefault("actionSignatures", [])
        state.setdefault("plannerFeedback", [])
        state.setdefault("readReplanCount", 0)
        state.setdefault("fallbackCodes", [])
    except ValidationError:
        return _fail(state, "validate_research_task", "REJECTED", "RESEARCH_TASK_CONTRACT_INVALID")
    _audit(state, "validate_research_task", "COMPLETED")
    return state


def _policy_gate(state: ScientificState) -> ScientificState:
    request = state.get("request")
    if not isinstance(request, ResearchTask):
        return _fail(state, "policy_gate", "REJECTED", "RESEARCH_TASK_CONTRACT_INVALID")
    decision = evaluate_input(request.question)
    if decision.verdict == "REJECT":
        return _fail(state, "policy_gate", "REJECTED", decision.code)
    if not set(request.requestedScopes).intersection({
        _ACTION_SCOPE[action] for action in request.allowedActions if _ACTION_SCOPE[action]
    }):
        return _fail(state, "policy_gate", "REJECTED", "RESEARCH_SCOPE_NOT_ALLOWED")
    _audit(state, "policy_gate", "COMPLETED")
    return state


def _observation_summaries(state: ScientificState) -> list[ScientificObservationSummary]:
    result: list[ScientificObservationSummary] = []
    for observation in state.get("observations", [])[-8:]:
        quality = [metric.code for metric in observation.missingnessSummary]
        quality.extend(metric.code for metric in observation.exclusionSummary)
        versions = [
            value for value in (
                observation.featureVersion,
                observation.taxonomyVersion,
                observation.sourceBatch,
            ) if value is not None
        ]
        result.append(ScientificObservationSummary(
            observationId=observation.observationId,
            actionName=observation.actionName,
            status=observation.status,
            source=observation.source,
            rowCount=observation.rowCount,
            qualityCodes=quality[:32],
            versionCodes=versions[:8],
        ))
    return result


def _decision_state_summary(
    observation_state: str,
    observations: list[Observation],
    prior_actions: list[str],
) -> str:
    """Build a bounded, de-identified State summary for Decision SFT.

    This is intentionally a closed summary of runtime state.  It must not
    copy the user question, SQL, raw rows, sample identifiers, or planner
    prose into the persisted trace.
    """

    routes = sorted({
        "java" if observation.source == "java_controlled_read"
        else "vector" if observation.source == "knowledge_hybrid"
        else "java"
        for observation in observations
    })
    return (
        f"observation_state={observation_state}; "
        f"evidence_bindings={len(observations)}; "
        f"source_routes={','.join(routes) if routes else 'none'}; "
        f"prior_actions={','.join(prior_actions) if prior_actions else 'none'}"
    )


def _decision_reason(action_name: str, stop_reason: str | None = None) -> str:
    """Return a controlled rationale suitable for audit and SFT export."""

    if action_name == "finish":
        return {
            "EVIDENCE_SUFFICIENT": "validated evidence is sufficient for the bounded task",
            "NO_NEW_INFORMATION": "the bounded loop has no new information to add",
            "QUALITY_RISK": "the remaining evidence quality risk requires a bounded stop",
            "ACTION_BUDGET_EXHAUSTED": "the configured action budget has been reached",
            "UPSTREAM_REJECTED": "an upstream policy or contract rejection requires a safe stop",
            "UNSUPPORTED_ACTION": "no supported action remains for the current state",
            "USER_REQUESTED_STOP": "the user requested that the exploration stop",
        }.get(stop_reason or "", "finish the bounded runtime exploration")
    return {
        "inspect_cohort": "inspect metadata before reading or analyzing the abundance projection",
        "execute_read_query": "execute the next bounded, policy-approved read needed for the task",
        "compare_groups": "compare the approved groups using the validated observation",
        "stratified_analysis": "check whether the observation changes across approved strata",
        "adjust_confounders": "control approved confounders before treating the observation as stable",
        "cross_project_validate": "validate whether the observation is stable across projects",
        "cross_disease_validate": "test whether the observation is disease-specific",
        "retrieve_evidence": "add bounded knowledge evidence after the data observation is available",
        "analyze_projection": "summarize the validated projection before the final decision",
    }.get(action_name, "select the next allow-listed runtime action")


def _plan_action(planner: ScientificPlannerPort, state: ScientificState) -> ScientificState:
    request = state.get("request")
    if not isinstance(request, ResearchTask):
        return _fail(state, "plan_action", "REJECTED", "RESEARCH_TASK_CONTRACT_INVALID")
    if len(state.get("actionHistory", [])) >= request.maxActions:
        state["stopReasonCode"] = "ACTION_BUDGET_EXHAUSTED"
        state["route"] = "synthesize"
        return state
    # Consume the internal retry route before selecting and authorizing the
    # replacement action.  Leaving it in state would bypass execution of the
    # new action and repeatedly re-plan the same candidate.
    state["route"] = "planned"
    decision_fallback_start = len(state.get("fallbackCodes", []))
    context = ScientificPlannerContext(
        questionSummary=_redact_question(request.question),
        intent=request.intent,
        approvedActions=request.allowedActions,
        remainingActionBudget=request.maxActions - len(state.get("actionHistory", [])),
        observations=_observation_summaries(state),
        executionFeedback=list(state.get("plannerFeedback", []))[-_MAX_JAVA_READ_REPLANS:],
        schemaCatalog=state.get("schemaCatalog"),
    )
    # Capture the exact redacted state presented to the policy boundary.  It
    # is the source of truth for DPO/SFT exports; state_summary is only a
    # human-readable projection and must never be parsed back into fields.
    policy_state = build_decision_policy_state(context)
    semantic_action = semantic_scientific_next_action(context)
    policy_first = bool(getattr(planner, "prefer_policy_action", False))
    if semantic_action is not None and not policy_first:
        result = ScientificPlannerResult(
            action=semantic_action,
            mode="deterministic",
            fallbackCode="SCIENTIFIC_RUNTIME_SEMANTIC_PATH_ENFORCED",
        )
    else:
        try:
            result = planner.plan_action(context)
        except Exception:
            result = DeterministicScientificPlanner().plan_action(context)
    # ``sft_policy`` is a genuine model decision via the closed Decision
    # Policy endpoint.  It must have the same trace provenance as the
    # full-action ``model`` planner: later Runtime repairs can then be
    # audited as raw model action -> final executed action.  Only the
    # deterministic fallback modes leave ``raw_action`` empty.
    model_decision = result.mode in {"model", "sft_policy"}
    raw_action = result.action.actionName if model_decision else None
    planner_origin = "model" if model_decision else "deterministic_fallback"
    if semantic_action is not None and not policy_first:
        planner_origin = "semantic_guard"
    if semantic_action is not None and result.action.actionName != semantic_action.actionName:
        # SFT policy receives a genuine chance to select the next action, but
        # cannot override a closed semantic obligation (for example metadata
        # inspection before abundance analysis).  Preserve this guard as an
        # auditable repair rather than silently treating the policy choice as
        # the executed action.
        result = ScientificPlannerResult(
            action=semantic_action,
            mode="deterministic",
            fallbackCode="SCIENTIFIC_RUNTIME_SEMANTIC_PATH_ENFORCED",
        )
        planner_origin = "semantic_guard"
    state["plannerMode"] = result.mode
    if result.fallbackCode:
        state.setdefault("fallbackCodes", []).append(result.fallbackCode)
        _audit(state, "plan_action", "COMPLETED", error_code=result.fallbackCode)
    else:
        _audit(state, "plan_action", "COMPLETED")
    if result.fallbackReasonCode:
        state.setdefault("fallbackCodes", []).append(result.fallbackReasonCode)
    action = result.action
    if action.actionName == "inspect_cohort" and not state.get("observations"):
        # The Scientific Agent selects the high-level metadata-first action;
        # Runtime owns its first catalog-only read shape.  Letting a model
        # invent joins, species names, or cohort filters before its first
        # observation caused otherwise valid open explorations to fail at the
        # Java boundary.  Subsequent actions still remain genuinely dynamic.
        repaired = DeterministicScientificPlanner().plan_action(context).action
        if isinstance(repaired, InspectCohortAction):
            action = repaired
            state["plannerMode"] = "deterministic"
            state.setdefault("fallbackCodes", []).append(
                "SCIENTIFIC_PLANNER_INITIAL_INSPECTION_REPAIRED"
            )
            _audit(
                state,
                "plan_action",
                "COMPLETED",
                error_code="SCIENTIFIC_PLANNER_INITIAL_INSPECTION_REPAIRED",
            )
    analysis_actions = {
        "compare_groups",
        "stratified_analysis",
        "adjust_confounders",
        "cross_project_validate",
        "cross_disease_validate",
        "analyze_projection",
    }
    has_analyzable_observation = any(
        observation.source == "java_controlled_read"
        for observation in state.get("observations", [])
    )
    if action.actionName in analysis_actions and not has_analyzable_observation:
        # Literature observations are evidence inputs, not tabular analysis
        # inputs. Repair an over-eager model action into the metadata-first
        # deterministic next step instead of allowing an opaque failure.
        repaired = DeterministicScientificPlanner().plan_action(context)
        action = repaired.action
        state["plannerMode"] = "deterministic"
        state.setdefault("fallbackCodes", []).append(
            "SCIENTIFIC_PLANNER_ACTION_REPAIRED"
        )
        _audit(
            state,
            "plan_action",
            "COMPLETED",
            error_code="SCIENTIFIC_PLANNER_ACTION_REPAIRED",
        )
    repeated_validation = action.actionName in {
        "cross_project_validate",
        "cross_disease_validate",
    } and any(
        observation.actionName == action.actionName
        for observation in state.get("observations", [])
    )
    if repeated_validation:
        # A second cross-validation pass over the same evolving state is not a
        # new evidence source. Repair it into the deterministic next capability
        # (usually retrieve_evidence) so a model cannot burn the tool budget by
        # restating the same validation decision with a new argument shape.
        repaired = DeterministicScientificPlanner().plan_action(context)
        action = repaired.action
        state["plannerMode"] = "deterministic"
        state.setdefault("fallbackCodes", []).append(
            "SCIENTIFIC_PLANNER_REPEATED_VALIDATION_REPAIRED"
        )
        _audit(
            state,
            "plan_action",
            "COMPLETED",
            error_code="SCIENTIFIC_PLANNER_REPEATED_VALIDATION_REPAIRED",
        )
    repeated_bounded_read = (
        action.actionName == "execute_read_query"
        and any(
            observation.actionName == "execute_read_query"
            for observation in state.get("observations", [])
        )
    )
    if repeated_bounded_read:
        # A second bounded read over the same evolving task state is tool
        # churn unless a dedicated state transition explicitly requires it.
        # Preserve the model proposal and stop safely with an audit code.
        action = _finish_action(context)
        state["plannerMode"] = "deterministic"
        state.setdefault("fallbackCodes", []).append(
            "SCIENTIFIC_PLANNER_REPEATED_READ_REPAIRED"
        )
        _audit(
            state,
            "plan_action",
            "COMPLETED",
            error_code="SCIENTIFIC_PLANNER_REPEATED_READ_REPAIRED",
        )
    repeated_projection = (
        action.actionName == "analyze_projection"
        and any(
            observation.actionName == "analyze_projection"
            for observation in state.get("observations", [])
        )
    )
    if repeated_projection:
        # Projection analysis is already a terminal bounded analysis for the
        # current observation.  Guard this for every intent, not only
        # focused_comparison: otherwise an open-exploration model can keep
        # producing identical projection steps until the trace fails its
        # no-repeat contract.
        action = _finish_action(context)
        state["plannerMode"] = "deterministic"
        state.setdefault("fallbackCodes", []).append(
            "SCIENTIFIC_PLANNER_REPEATED_PROJECTION_REPAIRED"
        )
        _audit(
            state,
            "plan_action",
            "COMPLETED",
            error_code="SCIENTIFIC_PLANNER_REPEATED_PROJECTION_REPAIRED",
        )
    repeated_inspection = (
        action.actionName == "inspect_cohort"
        and any(observation.actionName == "inspect_cohort"
                for observation in state.get("observations", []))
    )
    if repeated_inspection:
        repaired = DeterministicScientificPlanner().plan_action(context)
        action = repaired.action
        state["plannerMode"] = "deterministic"
        state.setdefault("fallbackCodes", []).append(
            "SCIENTIFIC_PLANNER_REPEATED_INSPECTION_REPAIRED"
        )
        _audit(
            state,
            "plan_action",
            "COMPLETED",
            error_code="SCIENTIFIC_PLANNER_REPEATED_INSPECTION_REPAIRED",
        )
    premature_stratified_without_comparison = (
        action.actionName == "stratified_analysis"
        and "compare_groups" in request.allowedActions
        and not any(
            observation.actionName == "compare_groups"
            for observation in state.get("observations", [])
        )
    )
    if premature_stratified_without_comparison:
        # Stratification refines an already validated group comparison.
        repaired = DeterministicScientificPlanner().plan_action(context)
        action = repaired.action
        state["plannerMode"] = "deterministic"
        state.setdefault("fallbackCodes", []).append(
            "SCIENTIFIC_PLANNER_STRATIFIED_ORDER_REPAIRED"
        )
        _audit(
            state,
            "plan_action",
            "COMPLETED",
            error_code="SCIENTIFIC_PLANNER_STRATIFIED_ORDER_REPAIRED",
        )
    question_text = _redact_question(request.question).lower()
    explicitly_insufficient = any(term in question_text for term in (
        "证据不足",
        "不足以支持",
        "信息不足",
        "无法支持",
        "insufficient evidence",
        "not enough evidence",
    ))
    premature_exploration_finish = (
        action.actionName == "finish"
        and request.intent == "scientific_exploration"
        and "retrieve_evidence" in request.allowedActions
        and not explicitly_insufficient
        and not any(observation.actionName == "retrieve_evidence"
                    for observation in state.get("observations", []))
        and any(observation.source == "java_controlled_read"
                for observation in state.get("observations", []))
    )
    if premature_exploration_finish:
        # Open exploration asks for evidence-backed discovery. Do not let a
        # syntactically valid finish bypass the requested evidence branch;
        # explicit insufficiency questions remain allowed to stop safely.
        repaired = DeterministicScientificPlanner().plan_action(context)
        action = repaired.action
        state["plannerMode"] = "deterministic"
        state.setdefault("fallbackCodes", []).append(
            "SCIENTIFIC_PLANNER_PREMATURE_FINISH_REPAIRED"
        )
        _audit(
            state,
            "plan_action",
            "COMPLETED",
            error_code="SCIENTIFIC_PLANNER_PREMATURE_FINISH_REPAIRED",
        )
    has_java_observation = any(
        observation.source == "java_controlled_read"
        for observation in state.get("observations", [])
    )
    completed_read_actions = {
        observation.actionName
        for observation in state.get("observations", [])
        if observation.source == "java_controlled_read"
    }
    data_fact_needs_one_follow_up_read = (
        request.intent == "data_fact"
        and action.actionName == "execute_read_query"
        and "inspect_cohort" in completed_read_actions
        and "execute_read_query" not in completed_read_actions
    )
    if request.intent == "data_fact" and has_java_observation \
            and action.actionName != "finish" \
            and not data_fact_needs_one_follow_up_read:
        # A data fact is the fast path in the design: one metadata inspection
        # may be followed by one bounded fact query, then the Runtime stops.
        # Further model-proposed reads are not scientific validation steps and
        # previously caused repeat-action rejections instead of an answer.
        action = _finish_action(context)
        state["plannerMode"] = "deterministic"
        state.setdefault("fallbackCodes", []).append(
            "SCIENTIFIC_PLANNER_DATA_FACT_FINISH_REPAIRED"
        )
        _audit(
            state,
            "plan_action",
            "COMPLETED",
            error_code="SCIENTIFIC_PLANNER_DATA_FACT_FINISH_REPAIRED",
        )
    repeated_focused_projection = (
        request.intent == "focused_comparison"
        and action.actionName == "analyze_projection"
        and any(observation.actionName == "analyze_projection"
                for observation in state.get("observations", []))
    )
    if repeated_focused_projection:
        # A focused comparison has reached its bounded analysis stage. A
        # second generic projection without a new Java observation is tool
        # churn, not a new hypothesis test.
        action = _finish_action(context)
        state["plannerMode"] = "deterministic"
        state.setdefault("fallbackCodes", []).append(
            "SCIENTIFIC_PLANNER_REPEATED_FOCUSED_ANALYSIS_REPAIRED"
        )
        _audit(
            state,
            "plan_action",
            "COMPLETED",
            error_code="SCIENTIFIC_PLANNER_REPEATED_FOCUSED_ANALYSIS_REPAIRED",
        )
    focused_analysis_observed = any(
        observation.actionName in {
            "compare_groups",
            "stratified_analysis",
            "adjust_confounders",
        }
        for observation in state.get("observations", [])
    )
    premature_focused_without_analysis = (
        request.intent == "focused_comparison"
        and action.actionName == "finish"
        and has_java_observation
        and not focused_analysis_observed
        and any(
            candidate in request.allowedActions
            for candidate in {
                "compare_groups",
                "stratified_analysis",
                "adjust_confounders",
            }
        )
    )
    if premature_focused_without_analysis:
        # A focused task must not stop immediately after metadata/read
        # inspection.  Re-plan through the deterministic closed action path so
        # the next step is the approved comparison/adjustment capability, not
        # a second model finish.
        repaired = DeterministicScientificPlanner().plan_action(context)
        action = repaired.action
        state["plannerMode"] = "deterministic"
        state.setdefault("fallbackCodes", []).append(
            "SCIENTIFIC_PLANNER_PREMATURE_FOCUSED_ACTION_REPAIRED"
        )
        _audit(
            state,
            "plan_action",
            "COMPLETED",
            error_code="SCIENTIFIC_PLANNER_PREMATURE_FOCUSED_ACTION_REPAIRED",
        )
    final_projection_observed = any(
        observation.actionName == "analyze_projection"
        for observation in state.get("observations", [])
    )
    premature_focused_finish = (
        request.intent == "focused_comparison"
        and action.actionName == "finish"
        and "analyze_projection" in request.allowedActions
        and focused_analysis_observed
        and not final_projection_observed
    )
    if premature_focused_finish:
        # compare/stratified/adjust produce bounded intermediate observations.
        # A focused request still needs one final projection before it can
        # synthesize a result; otherwise a model can stop after an internal
        # analysis executor runs but before an auditable final analysis action.
        # Evidence retrieval is not a tabular analysis input.  When a focused
        # finish is repaired after retrieval, retain the most recent Java
        # observation instead of passing the knowledge observation to the
        # bounded-analysis executor.
        latest_observation = next(
            (
                observation for observation in reversed(state["observations"])
                if observation.source == "java_controlled_read"
            ),
            state["observations"][-1],
        )
        action = AnalyzeProjectionAction(
            actionId=action.actionId,
            actionName="analyze_projection",
            rationale="Summarize the validated focused comparison before stopping",
            arguments=AnalyzeProjectionArguments(
                actionName="analyze_projection",
                observationId=latest_observation.observationId,
                analysisGoal=context.questionSummary,
            ),
        )
        state["plannerMode"] = "deterministic"
        state.setdefault("fallbackCodes", []).append(
            "SCIENTIFIC_PLANNER_PREMATURE_FOCUSED_FINISH_REPAIRED"
        )
        _audit(
            state,
            "plan_action",
            "COMPLETED",
            error_code="SCIENTIFIC_PLANNER_PREMATURE_FOCUSED_FINISH_REPAIRED",
        )
    if isinstance(action, FinishAction):
        owned_reason = action.arguments.reasonCode
        conflict_framed_question = any(term in question_text for term in (
            "冲突", "conflict", "相反", "不一致", "矛盾", "否定",
        ))
        if explicitly_insufficient:
            owned_reason = "NO_NEW_INFORMATION"
        elif conflict_framed_question and any(
                observation.actionName == "retrieve_evidence"
                for observation in state.get("observations", [])):
            # A task explicitly framed around conflicting evidence must keep
            # the epistemic risk at the terminal decision. A speculative-only
            # task instead retains that limitation inside its evidence report
            # and may still finish with evidence sufficient.
            owned_reason = "QUALITY_RISK"
        elif request.intent == "focused_comparison" \
                and final_projection_observed \
                and owned_reason == "NO_NEW_INFORMATION":
            owned_reason = "EVIDENCE_SUFFICIENT"
        elif request.intent == "focused_comparison" \
                and owned_reason == "NO_NEW_INFORMATION" \
                and any(
                    observation.actionName == "retrieve_evidence"
                    and observation.status == "VALIDATED"
                    and observation.rowCount > 0
                    for observation in state.get("observations", [])
                ):
            # A focused task may deliberately end after the bounded data
            # comparison and evidence branch without requiring a separate
            # projection action.  A non-empty validated evidence observation
            # is still sufficient terminal support; do not let a generic
            # model stop code create a false STOP_CORRECTNESS_FAILURE.
            owned_reason = "EVIDENCE_SUFFICIENT"
        elif request.intent == "scientific_exploration" \
                and owned_reason == "NO_NEW_INFORMATION" \
                and any(
                    observation.actionName == "retrieve_evidence"
                    and observation.status == "VALIDATED"
                    and observation.rowCount > 0
                    for observation in state.get("observations", [])
                ):
            # A completed, non-empty literature branch is evidence, not an
            # absence of information.  Keep explicit insufficiency requests
            # above as the only path that can legitimately claim otherwise.
            owned_reason = "EVIDENCE_SUFFICIENT"
        elif request.intent == "data_fact" \
                and has_java_observation \
                and owned_reason == "NO_NEW_INFORMATION":
            # A validated Java observation is the result for a data fact. A
            # model may use NO_NEW_INFORMATION as a generic finish token, but
            # it must not turn a completed fact query into an insufficient-
            # evidence failure unless the user explicitly requested that
            # limitation (handled by explicitly_insufficient above).
            owned_reason = "EVIDENCE_SUFFICIENT"
        elif owned_reason == "NO_NEW_INFORMATION" \
                and state.get("observations") \
                and not explicitly_insufficient:
            # A generic stop token after validated observations is not an
            # absence of information. Repair only the stop reason and retain
            # the raw model decision in the per-decision provenance fields.
            owned_reason = "EVIDENCE_SUFFICIENT"
        elif owned_reason == "ACTION_BUDGET_EXHAUSTED" \
                and len(state.get("actionHistory", [])) < request.maxActions:
            # The action budget is Runtime-owned. A planner may decide to
            # finish, but it cannot claim a budget exhaustion that has not
            # happened; doing so made otherwise complete traces fail their
            # stop-contract evaluation.
            owned_reason = (
                "EVIDENCE_SUFFICIENT" if state.get("observations")
                else "UPSTREAM_REJECTED"
            )
        if owned_reason != action.arguments.reasonCode:
            action = action.model_copy(update={
                "arguments": FinishArguments(
                    actionName="finish",
                    reasonCode=owned_reason,
                )
            })
            state.setdefault("fallbackCodes", []).append(
                "SCIENTIFIC_PLANNER_STOP_REASON_REPAIRED"
            )
            _audit(
                state,
                "plan_action",
                "COMPLETED",
                error_code="SCIENTIFIC_PLANNER_STOP_REASON_REPAIRED",
            )
    if action.actionName not in request.allowedActions:
        return _fail(state, "plan_action", "REJECTED", "ACTION_NOT_APPROVED")
    observations = state.get("observations", [])
    if not observations:
        observation_state = "NO_OBSERVATION"
    elif any(observation.status not in {"VALIDATED", "PARTIAL"} for observation in observations[-2:]):
        observation_state = "OBSERVATION_REQUIRES_REVIEW"
    elif any(observation.missingnessSummary for observation in observations[-2:]):
        observation_state = "MISSINGNESS_REQUIRES_REVIEW"
    elif any(observation.actionName == "retrieve_evidence" for observation in observations[-2:]):
        observation_state = "EVIDENCE_RETRIEVED_REQUIRES_GROUNDING"
    else:
        observation_state = "VALIDATED_OBSERVATION"
    stop_reason_for_decision = (
        action.arguments.reasonCode if isinstance(action, FinishAction) else None
    )
    decision_repair_codes = list(dict.fromkeys(
        state.get("fallbackCodes", [])[decision_fallback_start:]
    ))
    state.setdefault("decisionRecords", []).append(TraceDecision(
        observationStateCode=observation_state,
        allowedActions=list(request.allowedActions),
        chosenAction=action.actionName,
        decisionCode=(
            "STOP_AFTER_VALIDATION" if action.actionName == "finish"
            else "CHECK_CONFOUNDERS_BEFORE_CONCLUSION"
            if action.actionName == "adjust_confounders"
            else "SELECT_ALLOWED_RUNTIME_ACTION"
        ),
        observationEvidenceBindingCount=len(state.get("observations", [])),
        observationSourceRoutes=sorted({
            "java" if observation.source == "java_controlled_read"
            else "vector" if observation.source == "knowledge_hybrid"
            else "java"
            for observation in observations
        }),
        state_summary=_decision_state_summary(
            observation_state,
            observations,
            list(state.get("actionHistory", [])),
        ),
        decision_reason=_decision_reason(action.actionName, stop_reason_for_decision),
        selected_action=action.actionName,
        alternative_actions=[
            candidate for candidate in request.allowedActions
            if candidate != action.actionName
        ],
        stop_reason=stop_reason_for_decision,
        raw_action=raw_action,
        final_action=action.actionName,
        planner_origin=planner_origin,
        repair_code=decision_repair_codes[-1] if decision_repair_codes else None,
        repair_codes=decision_repair_codes,
        task_kind=policy_state["task_kind"],
        goal_code=policy_state["goal_code"],
        observation_flags=list(policy_state["observation_flags"]),
        history_actions=list(policy_state["history_actions"]),
        candidate_actions=list(policy_state["candidate_actions"]),
    ))
    action_id = "action-" + sha256(
        f"{request.traceId}|{request.taskId}|{len(state.get('actionHistory', []))}|"
        f"{action.actionName}|{state.get('readReplanCount', 0)}".encode()
    ).hexdigest()[:32]
    state["currentAction"] = action.model_copy(update={"actionId": action_id})
    return state


def _action_signature(action: ScientificAction) -> str:
    payload = action.arguments.model_dump(mode="json", exclude_none=True)
    return sha256(f"{action.actionName}|{payload}".encode("utf-8")).hexdigest()


def _authorize_action(state: ScientificState) -> ScientificState:
    request = state.get("request")
    action = state.get("currentAction")
    if not isinstance(request, ResearchTask) or action is None:
        return _fail(state, "authorize_action", "REJECTED", "ACTION_NOT_AVAILABLE")
    required_scope = _ACTION_SCOPE[action.actionName]
    if required_scope is not None and required_scope not in request.requestedScopes:
        return _fail(state, "authorize_action", "REJECTED", "ACTION_SCOPE_MISSING")
    if action.actionName in {
        "compare_groups",
        "stratified_analysis",
        "adjust_confounders",
        "cross_project_validate",
        "cross_disease_validate",
        "analyze_projection",
    } and not state.get("observations"):
        return _fail(state, "authorize_action", "REJECTED", "ANALYSIS_OBSERVATION_REQUIRED")
    signature = _action_signature(action)
    if signature in state.get("actionSignatures", []):
        if isinstance(action, (ExecuteReadQueryAction, InspectCohortAction)) \
                and state.get("plannerFeedback"):
            # A re-plan must not replay a failed SQL proposal.  Feed the
            # opaque local rejection back into the remaining retry budget
            # without sending the same proposal to Java again.
            return _replan_java_read(
                state,
                action,
                error_code="READ_PLAN_REPEAT_REJECTED",
                tool_call_id="",
                status="REJECTED",
                node="authorize_action",
            )
        return _fail(state, "authorize_action", "REJECTED", "ACTION_REPEAT_REJECTED")
    state.setdefault("actionSignatures", []).append(signature)
    _audit(state, "authorize_action", "COMPLETED")
    return state


def _new_observation_id(action_id: str, index: int) -> str:
    return "observation-" + sha256(f"{action_id}|{index}".encode("utf-8")).hexdigest()[:32]


def _metrics_from_response(response: Any) -> list[ObservationMetric]:
    quality = getattr(response, "qualitySummary", None)
    missing = getattr(quality, "missingCounts", {}) if quality else {}
    result: list[ObservationMetric] = []
    for name, count in missing.items():
        code = re.sub(r"[^a-z0-9_]", "_", str(name).lower()).strip("_")[:64]
        if re.fullmatch(r"[a-z][a-z0-9_]{1,63}", code):
            result.append(ObservationMetric(code=code, count=max(0, int(count))))
    return result[:64]


def _replan_java_read(
    state: ScientificState,
    action: ExecuteReadQueryAction | InspectCohortAction,
    *,
    error_code: str,
    tool_call_id: str,
    status: str,
    node: str,
) -> ScientificState:
    """Retry a semantic Java read rejection through planning, never by replaying SQL.

    The planner receives only an allow-listed semantic error code and must
    construct a fresh catalog-bounded action.  A generic Java execution
    failure is infrastructure/mapper state with no safe corrective detail, so
    it fails closed without spending model calls.  The unsuccessful attempt is
    a Runtime correction, not a scientific finding or a completed Action.
    """
    retry_count = int(state.get("readReplanCount", 0))
    decisions = state.get("decisionRecords", [])
    if decisions and decisions[-1].chosenAction == action.actionName:
        decisions.pop()
    if error_code not in _REPLANNABLE_JAVA_READ_CODES or retry_count >= _MAX_JAVA_READ_REPLANS:
        return _fail(
            state,
            node,
            status,
            error_code,
            tool_name="execute_read_query" if tool_call_id else None,
            tool_call_id=tool_call_id or None,
        )

    _audit(
        state,
        node,
        status,
        tool_name="execute_read_query" if tool_call_id else None,
        tool_call_id=tool_call_id or None,
        error_code=error_code,
    )
    state["readReplanCount"] = retry_count + 1
    state.setdefault("plannerFeedback", []).append(error_code)
    state["plannerFeedback"] = state["plannerFeedback"][-_MAX_JAVA_READ_REPLANS:]
    # Retain the failed action signature.  The next plan must be materially
    # different; otherwise it is rejected locally without re-executing SQL.
    state["currentAction"] = None
    state.pop("pendingObservation", None)
    state.pop("pendingPayload", None)
    state.pop("pendingResponse", None)
    state.pop("errorCode", None)
    state["route"] = "retry_plan"
    state.setdefault("fallbackCodes", []).append("SCIENTIFIC_PLANNER_JAVA_READ_REPLAN")
    return state


def _execute_read_query_action(
    state: ScientificState,
    action: ExecuteReadQueryAction | InspectCohortAction,
                               java_port: JavaAgentToolPort) -> ScientificState:
    request = state["request"]
    call_id = "call-" + sha256(
        f"{request.traceId}|{request.taskId}|{action.actionId}|execute_read_query".encode()
    ).hexdigest()[:32]
    call = ExecuteReadQueryJavaToolCall(
        toolName="execute_read_query",
        runId=request.runId,
        toolCallId=call_id,
        arguments=ExecuteReadQueryArguments(
            sql=action.arguments.sql,
            limit=action.arguments.limit,
        ),
    )
    try:
        response = java_port.execute(call)
    except JavaPortContractError as exc:
        return _fail(state, "execute_action", "REJECTED", exc.code)
    except JavaPortTransportError as exc:
        return _fail(state, "execute_action", "FAILED", exc.code)
    except Exception:
        return _fail(state, "execute_action", "FAILED", "JAVA_TOOL_EXECUTION_FAILED")
    if response.runId != call.runId or response.toolCallId != call.toolCallId:
        return _fail(state, "execute_action", "FAILED", "JAVA_TOOL_RESPONSE_MISMATCH")
    if response.status != "COMPLETED" or response.dataSnapshot is None:
        error_code = response.error.code if response.error else response.status
        return _replan_java_read(
            state,
            action,
            error_code=error_code,
            tool_call_id=call.toolCallId,
            status=response.status,
            node="execute_action",
        )
    snapshot = response.dataSnapshot
    if snapshot.snapshotPersistence != "transient":
        return _fail(state, "execute_action", "FAILED", "EVIDENCE_PERSISTENCE_UNSUPPORTED")
    index = len(state.get("observations", []))
    observation = Observation(
        observationId=_new_observation_id(action.actionId, index),
        actionId=action.actionId,
        actionName=action.actionName,
        status="VALIDATED",
        source="java_controlled_read",
        queryHash=snapshot.queryHash,
        rowCount=snapshot.rowCount,
        generatedAt=snapshot.generatedAt,
        schemaVersion=response.schemaVersion or "java-read-model-v1",
        dataSnapshotId=snapshot.dataSnapshotId,
        snapshotPersistence="transient",
        featureVersion=snapshot.featureVersion,
        taxonomyVersion=snapshot.taxonomyVersion,
        sourceBatch=snapshot.sourceBatch,
        missingnessSummary=_metrics_from_response(response),
    )
    state["pendingObservation"] = observation
    state["pendingPayload"] = response.data
    state.setdefault("rawObservationPayloads", {})[observation.observationId] = response.data
    state["pendingResponse"] = response
    state["plannerFeedback"] = []
    state["readReplanCount"] = 0
    _audit(
        state,
        "execute_action",
        "COMPLETED",
        tool_name="execute_read_query",
        tool_call_id=call.toolCallId,
        response=response,
    )
    return state


def _execute_retrieval_action(state: ScientificState, action: ScientificAction,
                              knowledge_port: KnowledgeSearchPort | None) -> ScientificState:
    if knowledge_port is None:
        return _fail(state, "execute_action", "FAILED", "KNOWLEDGE_SOURCE_NOT_CONFIGURED")
    topics = action.arguments.topics
    topic = " ".join(topics)
    query = EvidenceQuery(
        topic=topic,
        direction="context",
        retrievalMode=action.arguments.retrievalMode,
        limit=action.arguments.topK,
    )
    try:
        search_parallel = getattr(knowledge_port, "search_parallel", None)
        if callable(search_parallel):
            branches = ("vector", "graph") if action.arguments.retrievalMode == "hybrid" else (action.arguments.retrievalMode,)
            evidence = search_parallel(query, branches=branches)
        else:
            evidence = knowledge_port.search(query)
    except Exception:
        return _fail(state, "execute_action", "FAILED", "KNOWLEDGE_SOURCE_FAILED")
    query_hash = "sha256:" + sha256(
        f"{topic}|{action.arguments.retrievalMode}|{action.arguments.topK}|{action.arguments.maxHops}".encode()
    ).hexdigest()
    observation = Observation(
        observationId=_new_observation_id(action.actionId, len(state.get("observations", []))),
        actionId=action.actionId,
        actionName="retrieve_evidence",
        status="VALIDATED" if evidence else "PARTIAL",
        source="knowledge_hybrid",
        queryHash=query_hash,
        rowCount=len(evidence),
        generatedAt=datetime.now(timezone.utc),
        schemaVersion="knowledge-retrieval-v1",
    )
    state["pendingObservation"] = observation
    state["pendingPayload"] = evidence
    state.setdefault("rawObservationPayloads", {})[observation.observationId] = evidence
    _audit(state, "execute_action", "COMPLETED", tool_name="literature_evidence")
    return state


def _analysis_inputs(action: Any) -> tuple[list[str], str, list[str], list[str]]:
    arguments = action.arguments
    if isinstance(action, AnalyzeProjectionAction):
        return [arguments.observationId], arguments.analysisGoal, [], []
    return (
        list(arguments.observationIds),
        arguments.analysisGoal,
        list(arguments.dimensions),
        list(arguments.confounders),
    )


def _execute_analysis_action(
    state: ScientificState,
    action: AnalyzeProjectionAction | CompareGroupsAction | StratifiedAnalysisAction
    | AdjustConfoundersAction | CrossProjectValidateAction | CrossDiseaseValidateAction,
    planner: ScientificPlannerPort,
) -> ScientificState:
    observation_ids, analysis_goal, dimensions, confounders = _analysis_inputs(action)
    observations = {
        item.observationId: item for item in state.get("observations", [])
    }
    # A model can legally choose an analysis action after evidence retrieval,
    # but the evidence observation itself is never a tabular analysis input.
    # Rebind that action to the newest validated Java table rather than
    # rejecting an otherwise valid policy continuation.
    if any(
        observation_id in observations
        and observations[observation_id].source == "knowledge_hybrid"
        for observation_id in observation_ids
    ):
        latest_tabular = next(
            (item for item in reversed(state.get("observations", []))
             if item.source == "java_controlled_read"),
            None,
        )
        if latest_tabular is not None:
            observation_ids = [latest_tabular.observationId]
    if any(observation_id not in observations for observation_id in observation_ids):
        return _fail(state, "execute_action", "REJECTED", "OBSERVATION_NOT_FOUND")

    payload_rows: list[dict[str, object]] = []
    payload_columns: list[str] = []
    for observation_id in observation_ids:
        payload = state.get("rawObservationPayloads", {}).get(observation_id)
        if not isinstance(payload, dict):
            return _fail(state, "execute_action", "REJECTED", "OBSERVATION_NOT_ANALYZABLE")
        columns = payload.get("columns")
        rows = payload.get("rows")
        if isinstance(columns, list):
            payload_columns.extend(value for value in columns if isinstance(value, str))
        if isinstance(rows, list):
            payload_rows.extend(row for row in rows if isinstance(row, dict))
    if not payload_rows:
        return _fail(state, "execute_action", "REJECTED", "OBSERVATION_NOT_ANALYZABLE")

    analysis_id = "analysis-" + sha256(
        f"{state['request'].traceId}|{action.actionId}|{action.actionName}".encode("utf-8")
    ).hexdigest()[:32]
    try:
        plan = AnalysisPlan(
            analysisId=analysis_id,
            actionName=("analyze_projection" if isinstance(action, AnalyzeProjectionAction)
                        else action.actionName),
            sourceObservationIds=observation_ids,
            analysisGoal=analysis_goal,
            dimensions=dimensions,
            confounders=confounders,
        )
    except (ValidationError, ValueError):
        return _fail(state, "execute_action", "REJECTED", "ANALYSIS_PLAN_INVALID")
    state.setdefault("analysisPlans", []).append(plan)

    preview_payload = {
        "columns": list(dict.fromkeys(payload_columns)),
        "rows": payload_rows[:1000],
    }
    columns, rows = build_analysis_preview(preview_payload)
    from mico_agent_runtime.contracts.generated_analysis import AnalysisPlannerContext
    try:
        generated = None
        semantic_path = "SCIENTIFIC_RUNTIME_SEMANTIC_PATH_ENFORCED" in state.get(
            "fallbackCodes", []
        )
        if semantic_path:
            # These seven bounded Eval patterns exercise planning order, not
            # model-authored Python.  Their safe statistical substrate is
            # Runtime-owned, so avoid repeatedly charging the model for the
            # same sandbox-incompatible code shape after it was already
            # demonstrated and bounded by the generic retry path.
            state.setdefault("fallbackCodes", []).append(
                "SCIENTIFIC_RUNTIME_DETERMINISTIC_ANALYSIS"
            )
            generated = execute_generated_analysis(
                deterministic_analysis_plan(),
                rows,
                len(payload_rows),
                planner_mode="deterministic",
            )
        else:
            generator = getattr(planner, "generate_analysis", None)
            if not callable(generator):
                return _fail(state, "execute_action", "FAILED", "ANALYSIS_PLANNER_NOT_CONFIGURED")
            feedback: list[str] = []
            for attempt in range(_MAX_ANALYSIS_EXECUTION_ATTEMPTS):
                planned = generator(AnalysisPlannerContext(
                    questionSummary=analysis_goal,
                    workflow=action.actionName,
                    columns=columns,
                    previewRows=rows,
                    executionFeedback=feedback,
                ))
                try:
                    generated = execute_generated_analysis(
                        planned.plan,
                        rows,
                        len(payload_rows),
                        planner_mode=planned.mode,
                    )
                    break
                except GeneratedAnalysisError as exc:
                    if attempt + 1 >= _MAX_ANALYSIS_EXECUTION_ATTEMPTS:
                        state.setdefault("fallbackCodes", []).append(
                            "ANALYSIS_CODE_DETERMINISTIC_FALLBACK"
                        )
                        _audit(
                            state,
                            "execute_action",
                            "COMPLETED",
                            error_code="ANALYSIS_CODE_DETERMINISTIC_FALLBACK",
                        )
                        generated = execute_generated_analysis(
                            deterministic_analysis_plan(),
                            rows,
                            len(payload_rows),
                            planner_mode="deterministic",
                        )
                        break
                    feedback = [exc.code]
                    state.setdefault("fallbackCodes", []).append("ANALYSIS_CODE_REPLAN")
                    _audit(
                        state,
                        "execute_action",
                        "COMPLETED",
                        error_code="ANALYSIS_CODE_REPLAN",
                    )
        if generated is None:
            return _fail(state, "execute_action", "FAILED", "ANALYSIS_CODE_REJECTED")
        analysis_evidence = [
            AnalysisEvidence(
                evidenceId="evidence-" + sha256(
                    f"{analysis_id}|{source.observationId}".encode("utf-8")
                ).hexdigest()[:32],
                analysisId=analysis_id,
                observationId=source.observationId,
                source=source.source,
                dataSnapshotId=source.dataSnapshotId,
                snapshotPersistence="transient",
                queryHash=source.queryHash,
                schemaVersion=source.schemaVersion,
                rowCount=source.rowCount,
                generatedAt=source.generatedAt,
            )
            for source in (observations[observation_id] for observation_id in observation_ids)
        ]
        from mico_agent_runtime.contracts.analysis import AnalysisFeatureResult
        safe_features = [
            AnalysisFeatureResult(
                featureName=item.featureName,
                metrics={item.metricName: item.metricValue},
                supportStatus="supported",
            )
            for item in generated.topFeatures
        ]
        analysis_result = AnalysisResult(
            analysisId=analysis_id,
            actionName=plan.actionName,
            status="COMPLETED",
            plannerMode=generated.plannerMode,
            codeVersion=generated.codeVersion,
            sourceObservationIds=observation_ids,
            rowsAnalyzed=generated.rowCount,
            metrics=generated.metrics,
            topFeatures=safe_features,
            evidence=analysis_evidence,
            limitations=[
                "generated_code_was_sandbox_validated",
                "input_projection_was_bounded",
                "snapshot_is_transient_and_not_replayable",
                "analysis_is_not_a_clinical_conclusion",
                "method_was_selected_at_runtime",
            ],
        )
    except (GeneratedAnalysisError, ValidationError, TypeError, ValueError):
        return _fail(state, "execute_action", "FAILED", "ANALYSIS_CODE_REJECTED")

    source_observation = observations[observation_ids[0]]
    observation = Observation(
        observationId=_new_observation_id(action.actionId, len(state.get("observations", []))),
        actionId=action.actionId,
        actionName=action.actionName,
        status="VALIDATED",
        source="python_bounded_analysis",
        queryHash=source_observation.queryHash,
        rowCount=analysis_result.rowsAnalyzed,
        generatedAt=datetime.now(timezone.utc),
        schemaVersion="sandbox-python-v1",
        dataSnapshotId=source_observation.dataSnapshotId,
        snapshotPersistence=source_observation.snapshotPersistence,
        featureVersion=source_observation.featureVersion,
        taxonomyVersion=source_observation.taxonomyVersion,
        sourceBatch=source_observation.sourceBatch,
    )
    state.setdefault("analysisResults", []).append(analysis_result)
    state.setdefault("analysisEvidence", []).extend(analysis_evidence)
    state["pendingObservation"] = observation
    state["pendingPayload"] = analysis_result
    # Keep the bounded, already-approved projection available for a later
    # analysis step.  The public observation/report remains metadata-only, but
    # a chained State -> Action decision must be able to reference the output
    # of this Python analysis without treating its summary metrics as raw rows.
    state.setdefault("rawObservationPayloads", {})[observation.observationId] = {
        "columns": preview_payload["columns"],
        "rows": preview_payload["rows"],
    }
    _audit(state, "execute_action", "COMPLETED", tool_name="python_bounded_analysis")
    return state


def _execute_action(
    java_port: JavaAgentToolPort,
    planner: ScientificPlannerPort,
    knowledge_port: KnowledgeSearchPort | None,
    state: ScientificState,
) -> ScientificState:
    action = state.get("currentAction")
    if isinstance(action, FinishAction):
        state["stopReasonCode"] = action.arguments.reasonCode
        state.setdefault("actionHistory", []).append(action.actionName)
        state["route"] = "synthesize"
        _audit(state, "execute_action", "COMPLETED")
        return state
    if isinstance(action, (ExecuteReadQueryAction, InspectCohortAction)):
        return _execute_read_query_action(state, action, java_port)
    if action.actionName == "retrieve_evidence":
        return _execute_retrieval_action(state, action, knowledge_port)
    if isinstance(action, (
        AnalyzeProjectionAction,
        CompareGroupsAction,
        StratifiedAnalysisAction,
        AdjustConfoundersAction,
        CrossProjectValidateAction,
        CrossDiseaseValidateAction,
    )):
        return _execute_analysis_action(state, action, planner)
    return _fail(state, "execute_action", "REJECTED", "UNSUPPORTED_ACTION")


def _validate_observation(state: ScientificState) -> ScientificState:
    observation = state.get("pendingObservation")
    if not isinstance(observation, Observation):
        return _fail(state, "validate_observation", "FAILED", "OBSERVATION_MISSING")
    if observation.status not in {"VALIDATED", "PARTIAL"}:
        return _fail(state, "validate_observation", "FAILED", "OBSERVATION_INVALID")
    _audit(state, "validate_observation", "COMPLETED")
    return state


def _update_state(state: ScientificState) -> ScientificState:
    observation = state.pop("pendingObservation", None)
    state.pop("pendingPayload", None)
    state.pop("pendingResponse", None)
    action = state.get("currentAction")
    if isinstance(observation, Observation):
        state.setdefault("observations", []).append(observation)
    if action is not None:
        state.setdefault("actionHistory", []).append(action.actionName)
    state["currentAction"] = None
    _audit(state, "update_state", "COMPLETED")
    return state


def _decide_continue_or_stop(state: ScientificState) -> ScientificState:
    request = state.get("request")
    if not isinstance(request, ResearchTask):
        return _fail(state, "decide_continue_or_stop", "FAILED", "RESEARCH_TASK_CONTRACT_INVALID")
    if len(state.get("actionHistory", [])) >= request.maxActions:
        state["stopReasonCode"] = "ACTION_BUDGET_EXHAUSTED"
        state["route"] = "synthesize"
    else:
        state["route"] = "plan"
    _audit(state, "decide_continue_or_stop", "COMPLETED")
    return state


def _synthesis_context(
    request: ResearchTask,
    evidence: list[Any],
) -> EvidenceSynthesisContext | None:
    def _grounding_paths(item: Any) -> list[Any]:
        candidate_status = getattr(item, "supportStatus", "supported")
        paths = list(getattr(item, "reasoningPaths", []) or [])
        if candidate_status == "supported":
            return paths
        # A unified candidate aggregates multiple routes.  A supported path
        # inside a candidate that also contains speculative/conflicted
        # evidence must not be offered to the generator as independently
        # supported, otherwise the final trace evaluator correctly detects an
        # unsafe support-status escalation.  Downgrade only the bounded
        # synthesis context; the persisted source evidence remains unchanged.
        return [path.model_copy(update={"status": candidate_status}) for path in paths]

    literature = [
        SynthesisEvidence(
            evidenceId=item.candidateId,
            title=item.title,
            summary=item.summary,
            reasoningPaths=_grounding_paths(item),
        )
        for item in evidence[:20]
        if getattr(item, "kind", None) == "literature_chunk"
    ]
    if not literature:
        return None
    return EvidenceSynthesisContext(
        questionSummary=_redact_question(request.question),
        evidence=literature,
    )


def _safe_synthesis_result(
    request: ResearchTask,
    evidence: list[Any],
    synthesis_port: GraphRagSynthesisPort,
) -> EvidenceSynthesisResult:
    context = _synthesis_context(request, evidence)
    if context is None:
        return EvidenceSynthesisResult(
            claims=[],
            reasoningSteps=[],
            conclusion=None,
            mode="deterministic",
            fallbackCode="GRAPHRAG_SYNTHESIS_NO_LITERATURE",
        )
    deterministic = DeterministicGraphRagSynthesisPort()
    try:
        result = synthesis_port.synthesize(context)
        # Validate the model/port output against the exact candidates and
        # paths before it can enter the public ResearchExplorationReport.
        ResearchExplorationReport(
            traceId=request.traceId,
            runId=request.runId,
            taskId=request.taskId,
            status="COMPLETED",
            unifiedEvidence=evidence,
            groundedClaims=result.claims,
            groundedReasoningSteps=result.reasoningSteps,
            limitations=["scientific_evidence_is_not_clinical_diagnosis"],
        )
        return result
    except Exception:
        fallback = deterministic.synthesize(context)
        return fallback.model_copy(update={
            "mode": "deterministic",
            "fallbackCode": "GRAPHRAG_SYNTHESIS_OUTPUT_REJECTED",
        })


def _synthesize_report(
    synthesis_port: GraphRagSynthesisPort,
    state: ScientificState,
) -> ScientificState:
    request = state.get("request")
    if not isinstance(request, ResearchTask):
        return _fail(state, "synthesize_report", "FAILED", "RESEARCH_TASK_CONTRACT_INVALID")
    bindings: list[ResearchEvidenceBinding] = []
    steps: list[ResearchReasoningStep] = []
    for index, observation in enumerate(state.get("observations", []), start=1):
        binding_id = "binding-" + sha256(
            f"{request.traceId}|{observation.observationId}".encode()
        ).hexdigest()[:32]
        evidence_id = "evidence-" + sha256(
            f"{observation.source}|{observation.queryHash}".encode()
        ).hexdigest()[:32]
        bindings.append(ResearchEvidenceBinding(
            bindingId=binding_id,
            observationId=observation.observationId,
            evidenceReference=evidence_id,
            supportStatus="supported" if observation.status == "VALIDATED" else "partial",
            source=observation.source,
            dataSnapshotId=observation.dataSnapshotId,
            queryHash=observation.queryHash,
            snapshotPersistence=observation.snapshotPersistence,
        ))
        steps.append(ResearchReasoningStep(
            stepIndex=index,
            actionName=observation.actionName,
            description="A bounded action produced a source-bound observation; no domain claim is inferred here.",
            supportStatus="supported" if observation.status == "VALIDATED" else "partial",
            evidenceBindingIds=[binding_id],
        ))
    vector_results: list[LiteratureEvidenceItem] = []
    graph_results: list[LiteratureEvidenceItem] = []
    for payload in state.get("rawObservationPayloads", {}).values():
        if not isinstance(payload, list):
            continue
        for item in payload:
            if not isinstance(item, LiteratureEvidenceItem):
                continue
            routes = set(item.retrievalSources)
            if "vector" in routes:
                vector_results.append(item)
            if "graph" in routes:
                graph_results.append(item)
    java_observations = [
        observation for observation in state.get("observations", [])
        if observation.source == "java_controlled_read"
    ]
    unified_evidence = merge_unified_evidence(
        vector_results=vector_results,
        graph_results=graph_results,
        java_observations=java_observations,
        limit=20,
    )
    synthesis = _safe_synthesis_result(request, unified_evidence, synthesis_port)
    source_metadata = [
        AnalysisSourceMetadata(
            observationId=observation.observationId,
            source=observation.source,
            dataSnapshotId=observation.dataSnapshotId,
            snapshotPersistence=observation.snapshotPersistence,
            queryHash=observation.queryHash,
            schemaVersion=observation.schemaVersion,
            rowCount=observation.rowCount,
            generatedAt=observation.generatedAt,
            taxonomyVersion=observation.taxonomyVersion,
            featureVersion=observation.featureVersion,
            sourceBatch=observation.sourceBatch,
        )
        for observation in state.get("observations", [])[:40]
    ]
    report = ResearchExplorationReport(
        traceId=request.traceId,
        runId=request.runId,
        taskId=request.taskId,
        status="COMPLETED",
        evidenceBindings=bindings,
        unifiedEvidence=unified_evidence,
        analysisResults=state.get("analysisResults", [])[:20],
        analysisEvidence=state.get("analysisEvidence", [])[:40],
        sourceMetadata=source_metadata,
        groundedClaims=synthesis.claims,
        groundedReasoningSteps=synthesis.reasoningSteps,
        generationMode=(
            "model_grounded" if synthesis.mode == "model" else "deterministic_grounded"
        ),
        generationFallbackCode=synthesis.fallbackCode,
        reasoningSteps=steps,
        conclusion=synthesis.conclusion or (
            "The dynamic exploration completed only validated, bounded evidence actions; "
            "domain interpretation requires review of the bound evidence."
        ),
        limitations=[
            "snapshot_is_transient_and_not_replayable",
            "internal_record_is_not_a_subject_count",
            "subject_link_is_unverified",
            "dynamic_query_is_java_validated_and_bounded",
            "generated_analysis_is_bounded_and_sandboxed",
            "scientific_evidence_is_not_clinical_diagnosis",
        ],
    )
    state["status"] = "COMPLETED"
    state["report"] = report
    _audit(state, "synthesize_report", "COMPLETED")
    return state


def _terminal(state: ScientificState) -> ScientificState:
    _audit(state, "terminal", state.get("status", "FAILED"), error_code=state.get("errorCode"))
    return state


def _route_after_status(state: ScientificState, next_route: str) -> str:
    return next_route if state.get("status") is None else "terminal"


def build_scientific_graph(
    java_port: JavaAgentToolPort,
    planner: ScientificPlannerPort,
    knowledge_port: KnowledgeSearchPort | None = None,
    schema_catalog: SchemaSemanticCatalog | None = None,
    checkpointer: Any | None = None,
    synthesis_port: GraphRagSynthesisPort | None = None,
):
    synthesis_port = synthesis_port or DeterministicGraphRagSynthesisPort()
    builder = StateGraph(ScientificState)
    builder.add_node("validate_research_task", _validate_task)
    builder.add_node("policy_gate", _policy_gate)
    builder.add_node("plan_action", lambda state: _plan_action(planner, state))
    builder.add_node("authorize_action", _authorize_action)
    builder.add_node(
        "execute_action",
        lambda state: _execute_action(java_port, planner, knowledge_port, state),
    )
    builder.add_node("validate_observation", _validate_observation)
    builder.add_node("update_state", _update_state)
    builder.add_node("decide_continue_or_stop", _decide_continue_or_stop)
    builder.add_node(
        "synthesize_report",
        lambda state: _synthesize_report(synthesis_port, state),
    )
    builder.add_node("terminal", _terminal)
    builder.add_edge(START, "validate_research_task")
    builder.add_conditional_edges(
        "validate_research_task",
        lambda state: _route_after_status(state, "policy_gate"),
    )
    builder.add_conditional_edges(
        "policy_gate",
        lambda state: _route_after_status(state, "plan_action"),
    )
    builder.add_conditional_edges(
        "plan_action",
        lambda state: _route_after_status(state, "authorize_action"),
    )
    builder.add_conditional_edges(
        "authorize_action",
        lambda state: _route_after_status(
            state,
            "plan_action" if state.get("route") == "retry_plan" else "execute_action",
        ),
    )
    builder.add_conditional_edges(
        "execute_action",
        lambda state: _route_after_status(
            state,
            "plan_action" if state.get("route") == "retry_plan"
            else "synthesize_report" if state.get("route") == "synthesize"
            else "validate_observation",
        ),
    )
    builder.add_conditional_edges(
        "validate_observation",
        lambda state: _route_after_status(state, "update_state"),
    )
    builder.add_edge("update_state", "decide_continue_or_stop")
    builder.add_conditional_edges(
        "decide_continue_or_stop",
        lambda state: _route_after_status(state, state.get("route", "terminal")),
        {
            "plan": "plan_action",
            "synthesize": "synthesize_report",
            "terminal": "terminal",
        },
    )
    builder.add_edge("synthesize_report", "terminal")
    builder.add_edge("terminal", END)
    return builder.compile(checkpointer=checkpointer)
