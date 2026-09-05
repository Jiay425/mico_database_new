from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import replace
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
from mico_agent_runtime.contracts.decision_state import (
    ScientificActionSpaceState,
    ScientificDecisionState,
    ScientificTaskState,
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
from mico_agent_runtime.contracts.task_understanding import TaskUnderstandingOutput
from mico_agent_runtime.contracts.scientific_policy import (
    ScientificPolicyInput,
    build_scientific_policy_input,
)
from mico_agent_runtime.contracts.evidence import LiteratureEvidenceItem
from mico_agent_runtime.contracts.unified_evidence import merge_unified_evidence
from mico_agent_runtime.contracts.tools import ExecuteReadQueryArguments, ExecuteReadQueryJavaToolCall
from mico_agent_runtime.graph.generated_analysis import (
    GeneratedAnalysisError,
    TYPED_ANALYSIS_SANDBOX_FALLBACK_CODES,
    build_analysis_preview,
    execute_generated_analysis,
    execute_typed_analysis,
)
from mico_agent_runtime.graph.scientific_state import ScientificState
from mico_agent_runtime.runtime.decision_state_builder import (
    update_from_analysis_result,
    update_from_evidence_result,
    update_from_query_result,
    update_progress,
)
from mico_agent_runtime.runtime.analysis_capability_registry import (
    AnalysisCapabilityContext,
    match_analysis_capability,
)
from mico_agent_runtime.runtime.action_availability import (
    ActionAvailabilityContext,
    evaluate_action_availability,
)
from mico_agent_runtime.runtime.objective_resolution import (
    active_remaining_objectives,
    limitation_codes_from_resolution,
    persist_objective_resolution,
    resolve_objective_lifecycle,
)
from mico_agent_runtime.runtime.data_requirements import (
    DataRequirementError,
    augment_query_plan_with_fields,
    derive_data_requirements,
    prune_unavailable_query_plan,
    record_zero_coverage_capabilities,
    unavailable_fields_from_state,
)
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
from mico_agent_runtime.ports.research_planner import required_cross_validation_fields
from mico_agent_runtime.ports.task_understanding import (
    DeterministicTaskUnderstandingPort,
    TaskUnderstandingError,
    TaskUnderstandingPort,
)


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
# Runtime-owned technical controls for raw one-to-many abundance reads.  The
# model still chooses semantic fields and filters; it never chooses these
# explosion guards.
_SAMPLE_LIMIT_PER_GROUP = 50
_SAMPLE_BOUNDED_ROW_LIMIT = 20_000
_REPLANNABLE_JAVA_READ_CODES = frozenset({
    "READ_PLAN_REPEAT_REJECTED",
    "READ_MODEL_ARGUMENT_REJECTED",
})

# A DeepSeek AnalysisPlan is an execution materialization of one already
# selected high-level Action.  Keep this mapping closed at the Runtime
# boundary so a malformed/misbehaving materializer cannot silently switch
# scientific action families.
_ACTION_ANALYSIS_TYPE = {
    "compare_groups": "group_comparison",
    "stratified_analysis": "stratified_comparison",
    "adjust_confounders": "confounder_adjustment",
    "cross_project_validate": "cross_project_validation",
    "cross_disease_validate": "cross_disease_validation",
    "analyze_projection": "projection",
}


def _identity(state: ScientificState) -> tuple[str, str]:
    request = state.get("request")
    if isinstance(request, ResearchTask):
        return request.traceId, request.runId
    return "invalid", "invalid"


def _sha256_model(value: Any) -> str:
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + sha256(serialized.encode("utf-8")).hexdigest()


def _annotate_last_decision(state: ScientificState, **updates: object) -> None:
    decisions = state.get("decisionRecords", [])
    if not decisions:
        return
    for key, value in updates.items():
        setattr(decisions[-1], key, value)


def _record_runtime_repair(state: ScientificState, code: str) -> None:
    """Attach an execution-time binding repair to the current decision."""

    state.setdefault("fallbackCodes", []).append(code)
    decisions = state.get("decisionRecords", [])
    if not decisions:
        return
    current = decisions[-1]
    repair_codes = list(dict.fromkeys([*current.repair_codes, code]))
    current.repair_codes = repair_codes
    current.repair_code = repair_codes[-1]


def _sample_bounded_query_plan(
    query_plan: object,
    catalog: SchemaSemanticCatalog | None,
) -> object:
    """Attach the Runtime-owned sample cap to a raw abundance projection.

    This is deliberately a narrow technical binding.  It only applies when
    the model already selected a raw sample/feature/value projection and a
    verified root dimension is present.  The model's scientific Action,
    outcome, feature, relation path and filters are preserved verbatim.
    """

    if not hasattr(query_plan, "model_copy") or catalog is None:
        return query_plan
    if getattr(query_plan, "aggregations", None) or getattr(query_plan, "group_by", None):
        return query_plan
    selected = list(getattr(query_plan, "select_fields", []) or [])
    if not {"abundance.feature", "abundance.value"}.issubset(selected):
        return query_plan
    if "sample_to_abundance" not in set(getattr(query_plan, "relation_path", []) or []):
        return query_plan
    # The first selected verified root dimension is the technical partition
    # key.  In the canonical microbiome projection this is sample.disease;
    # using Catalog capabilities keeps the binding generic for another
    # dimension without inventing a field or changing the model's plan.
    selected_dimensions: list[str] = []
    for field_id in selected:
        if not field_id.startswith("sample."):
            continue
        entity_id, _, field_name = field_id.partition(".")
        if entity_id != "sample":
            continue
        entity = next((item for item in catalog.entities if item.entityId == entity_id), None)
        field = next((item for item in entity.fields if item.fieldId == field_id), None) if entity else None
        if field is not None and field.groupable and not field.sensitive:
            selected_dimensions.append(field_id)
    if not selected_dimensions:
        return query_plan
    # A requested group label filter is the strongest deterministic signal for
    # the cohort partition (for the canary this is sample.disease in {T2D,
    # healthy}); prefer it over another selected root dimension such as age.
    filtered_dimensions = [
        field_id for field_id in selected_dimensions
        if any(
            getattr(item, "field", None) == field_id
            for item in (getattr(query_plan, "filters", []) or [])
        )
    ]
    group_field = (filtered_dimensions or selected_dimensions)[0]
    current_limit = int(getattr(query_plan, "limit", 0) or 0)
    current_sample_limit = getattr(query_plan, "sample_limit_per_group", None)
    current_group_field = getattr(query_plan, "sample_limit_group_field", None)
    if (
        current_sample_limit == _SAMPLE_LIMIT_PER_GROUP
        and current_group_field == group_field
        and current_limit >= _SAMPLE_BOUNDED_ROW_LIMIT
    ):
        return query_plan
    return query_plan.model_copy(update={
        "limit": max(current_limit, _SAMPLE_BOUNDED_ROW_LIMIT),
        "sample_limit_per_group": _SAMPLE_LIMIT_PER_GROUP,
        "sample_limit_group_field": group_field,
    })


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


def _refresh_action_availability(state: ScientificState) -> ScientificState:
    """Refresh hard-executable actions without influencing the old planner."""

    request = state.get("request")
    decision_state = state.get("decisionState")
    if not isinstance(request, ResearchTask) or not isinstance(
        decision_state, ScientificDecisionState
    ):
        return state
    rows: list[dict[str, object]] = []
    for payload in state.get("rawObservationPayloads", {}).values():
        if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
            continue
        rows.extend(row for row in payload["rows"] if isinstance(row, dict))
    distinct_counts = _semantic_distinct_counts(
        state,
        rows,
        decision_state.data_state.available_dimensions,
    ) if rows else {}
    context = ActionAvailabilityContext(
        allowed_actions=list(request.allowedActions),
        requested_scopes=list(request.requestedScopes),
        catalog=state.get("schemaCatalog"),
        knowledge_available=bool(state.get("knowledgeAvailable", True)),
        max_actions=request.maxActions,
        action_count=decision_state.progress.action_count,
        distinct_counts=distinct_counts,
    )
    # First compute plain hard availability.  Objective lifecycle then uses
    # those technical reasons plus the real Observation query shapes to prove
    # whether a missing capability is terminal for this task.  A second pass
    # only changes the finish gate; it never adds or ranks a scientific Action.
    preliminary = evaluate_action_availability(decision_state, context)
    resolution = resolve_objective_lifecycle(
        decision_state,
        catalog=state.get("schemaCatalog"),
        availability_reasons=preliminary.blocked_reasons,
        knowledge_available=bool(state.get("knowledgeAvailable", True)),
        observations=state.get("observations", []),
    )
    updated = decision_state.model_copy(deep=True)
    # The policy sees only satisfiable obligations.  Blocked objectives remain
    # in Runtime objectiveResolution and in the final limitation report; they
    # are not mislabeled as completed.
    updated.progress.remaining_objectives = active_remaining_objectives(
        decision_state,
        resolution,
    )
    state["decisionState"] = updated
    state["objectiveResolution"] = persist_objective_resolution(
        resolution,
        previous=state.get("objectiveResolution"),
        resolved_at_step=updated.progress.action_count,
    )
    result = evaluate_action_availability(
        updated,
        replace(context, blocked_objectives=resolution.blocked_objectives),
    )
    updated.action_space.available_actions = list(result.available_actions)
    state["decisionState"] = updated
    # Blocked reasons are runtime/debug data only.  They are intentionally
    # not embedded in ScientificDecisionState and therefore cannot leak into
    # a future Qwen policy payload.
    state["actionAvailabilityReasons"] = dict(result.blocked_reasons)
    return state


def _understand_task(
    task_understanding_port: TaskUnderstandingPort | None,
    state: ScientificState,
) -> ScientificState:
    """Parse the original user question into objectives and explicit constraints.

    This is the only entry-point model boundary for Task Understanding.  It
    intentionally receives no observations, catalog, action history, or
    planner context.  The original question remains owned by Runtime and is
    never replaced by model output.
    """

    request = state.get("request")
    decision_state = state.get("decisionState")
    if not isinstance(request, ResearchTask) or not isinstance(
        decision_state, ScientificDecisionState
    ):
        return _fail(state, "understand_task", "REJECTED", "RESEARCH_TASK_CONTRACT_INVALID")

    port = task_understanding_port or DeterministicTaskUnderstandingPort()
    try:
        result = port.understand(request.question)
        output = result.output
        if not isinstance(output, TaskUnderstandingOutput):
            raise TaskUnderstandingError("TASK_UNDERSTANDING_FAILED")
    except TaskUnderstandingError as exc:
        return _fail(state, "understand_task", "FAILED", exc.code)
    except Exception:
        return _fail(state, "understand_task", "FAILED", "TASK_UNDERSTANDING_FAILED")

    updated = decision_state.model_copy(deep=True)
    # Runtime preserves the exact user question.  The model is only allowed
    # to fill the closed objective and explicit-constraint fields.
    updated.task = ScientificTaskState(
        query=request.question,
        objectives=list(output.objectives),
        constraints=output.constraints.model_copy(deep=True),
    )
    state["decisionState"] = updated
    state["taskUnderstandingMode"] = result.mode
    state["taskUnderstandingModel"] = result.model
    if result.fallbackCode:
        fallback_codes = state.setdefault("fallbackCodes", [])
        if result.fallbackCode not in fallback_codes:
            fallback_codes.append(result.fallbackCode)
    # Availability is recomputed only after Task Understanding has populated
    # the required objectives.  It remains a technical capability filter;
    # it does not select or rank the next scientific action.
    _refresh_action_availability(state)
    _audit(
        state,
        "understand_task",
        "COMPLETED",
        error_code=result.fallbackCode,
    )
    return state


def _validate_task(
    state: ScientificState,
    *,
    knowledge_available: bool = True,
) -> ScientificState:
    try:
        state["request"] = ResearchTask.model_validate(state.get("request"))
        request = state["request"]
        assert isinstance(request, ResearchTask)
        # Initialize the policy-facing contract. Task Understanding fills the
        # objectives/constraints and computes hard availability in the next
        # graph node; no policy-facing decision is made at validation time.
        state["decisionState"] = ScientificDecisionState(
            task=ScientificTaskState(query=request.question),
            action_space=ScientificActionSpaceState(
                available_actions=[],
            ),
        )
        state["knowledgeAvailable"] = knowledge_available
        state.setdefault("observations", [])
        state.setdefault("rawObservationPayloads", {})
        state.setdefault("analysisPlans", [])
        state.setdefault("analysisResults", [])
        state.setdefault("analysisEvidence", [])
        state.setdefault("actionHistory", [])
        state.setdefault("actionSignatures", [])
        state.setdefault("objectiveResolution", {})
        state.setdefault("plannerFeedback", [])
        state.setdefault("unavailableFields", [])
        state.setdefault("unavailableDataCapabilities", {})
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
        if observation.actionName == "execute_read_query" \
                and observation.source == "java_controlled_read":
            quality.extend(_dynamic_read_capability_codes(state, observation))
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
            queryPlanFields=list(observation.queryPlanFields),
            queryPlanRelationPath=list(observation.queryPlanRelationPath),
        ))
    return result


def _catalog_field_aliases(field_id: str) -> list[str]:
    """Return only Java's closed aliases for one semantic Catalog field."""

    entity, field = field_id.split(".", 1)
    return [
        f"a_{entity}_{field}",
        f"a_{entity}_{field}_count",
        f"a_{entity}_{field}_mean",
        f"a_{entity}_{field}_min",
        f"a_{entity}_{field}_max",
        f"a_{entity}_{field}_sum",
    ]


def _catalog_field_metadata(
    state: ScientificState,
) -> dict[str, tuple[str, bool, bool]]:
    """Map verified semantic IDs to (data type, sensitive, verified) metadata."""

    catalog = state.get("schemaCatalog")
    if catalog is None:
        return {}
    metadata: dict[str, tuple[str, bool, bool]] = {}
    for entity in catalog.entities:
        entity_id = entity.entityId or entity.entityName
        for field in entity.fields:
            field_id = field.fieldId or f"{entity_id}.{field.name}"
            metadata[field_id] = (
                field.dataType,
                field.sensitive,
                field.semanticStatus == "verified",
            )
    return metadata


def _payload_semantic_fields(
    state: ScientificState,
    payload: object,
) -> tuple[list[str], set[str]]:
    """Resolve returned Java aliases to safe semantic IDs and preview keys."""

    if not isinstance(payload, dict):
        return [], set()
    columns = payload.get("columns")
    if not isinstance(columns, list):
        return [], set()
    returned = {
        value for value in columns
        if isinstance(value, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", value)
    }
    metadata = _catalog_field_metadata(state)
    fields: list[str] = []
    allowed_columns: set[str] = set()
    for field_id, (_data_type, sensitive, verified) in metadata.items():
        if sensitive or not verified:
            continue
        matched = returned.intersection(_catalog_field_aliases(field_id))
        if matched:
            fields.append(field_id)
            allowed_columns.update(matched)
    return list(dict.fromkeys(fields)), allowed_columns


def _dynamic_read_capability_codes(
    state: ScientificState,
    observation: Observation,
) -> list[str]:
    """Describe only bounded analysis capability, never values or raw columns."""

    payload = state.get("rawObservationPayloads", {}).get(observation.observationId)
    fields, _ = _payload_semantic_fields(state, payload)
    metadata = _catalog_field_metadata(state)
    numeric_fields = [
        field_id for field_id in fields
        if metadata.get(field_id, ("unknown", True, False))[0] in {"integer", "number"}
    ]
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return ["numeric_outcome_missing"]
    for field_id in numeric_fields:
        aliases = set(_catalog_field_aliases(field_id))
        for row in rows[:1000]:
            if not isinstance(row, dict):
                continue
            if any(
                isinstance(row.get(alias), (int, float))
                and not isinstance(row.get(alias), bool)
                for alias in aliases
            ):
                return ["numeric_outcome_available"]
    return ["numeric_outcome_missing"]


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
    terminal_finish_turn = bool(state.pop("terminalFinishTurnGranted", False))
    budget_exhausted = len(state.get("actionHistory", [])) >= request.maxActions
    if terminal_finish_turn is False and budget_exhausted:
        state["stopReasonCode"] = "ACTION_BUDGET_EXHAUSTED"
        state["route"] = "synthesize"
        return state
    # Consume the internal retry route before selecting and authorizing the
    # replacement action.  Leaving it in state would bypass execution of the
    # new action and repeatedly re-plan the same candidate.
    state["route"] = "planned"
    decision_fallback_start = len(state.get("fallbackCodes", []))
    dynamic_materialization = bool(getattr(planner, "dynamic_action_materialization", False))
    planning_feedback = list(state.get("plannerFeedback", []))[-_MAX_JAVA_READ_REPLANS:]
    if dynamic_materialization:
        latest_tabular = next(
            (
                item for item in reversed(state.get("observations", []))
                if item.actionName == "execute_read_query"
                and item.source == "java_controlled_read"
                and item.status in {"VALIDATED", "PARTIAL"}
            ),
            None,
        )
        if latest_tabular is not None and "numeric_outcome_missing" in _dynamic_read_capability_codes(
            state,
            latest_tabular,
        ):
            planning_feedback.append("ANALYSIS_NUMERIC_OUTCOME_REQUIRED")
    decision_state = state.get("decisionState")
    data_requirements = None
    unavailable_fields = unavailable_fields_from_state(state)
    if dynamic_materialization and isinstance(decision_state, ScientificDecisionState):
        # Derive a bounded semantic requirement set before the Policy-selected
        # Action is materialized.  This is Runtime data acquisition context,
        # not a recommendation to the Policy and never enters its six-block
        # input.
        data_requirements = derive_data_requirements(
            decision_state,
            state.get("schemaCatalog"),
            unavailable_fields=unavailable_fields,
        )
        state["dataRequirements"] = data_requirements.model_dump()
    planning_budget = max(
        1,
        request.maxActions - len(state.get("actionHistory", [])),
    )
    context = ScientificPlannerContext(
        questionSummary=_redact_question(request.question),
        intent=request.intent,
        approvedActions=request.allowedActions,
        remainingActionBudget=planning_budget,
        observations=_observation_summaries(state),
        executionFeedback=list(dict.fromkeys(planning_feedback))[-_MAX_JAVA_READ_REPLANS:],
        requiredSemanticFields=(
            list(data_requirements.required_fields)
            if data_requirements is not None else []
        ),
        unavailableSemanticFields=list(unavailable_fields),
        # ``requiredGroupField`` is reserved for the grouped
        # cross-validation/coverage contract.  Ordinary raw reads still carry
        # the group semantic ID in requiredSemanticFields, but must remain
        # ungrouped so sample-level operators can consume their rows.
        requiredGroupField=None,
        schemaCatalog=state.get("schemaCatalog"),
    )
    new_policy_path = (
        dynamic_materialization
        and isinstance(decision_state, ScientificDecisionState)
        and callable(getattr(planner, "plan_action_with_state", None))
    )
    if new_policy_path and isinstance(decision_state, ScientificDecisionState):
        # Materialization is a separate Gemini role from Scientific Policy,
        # but it still needs the same validated, compressed State to turn the
        # selected Action into concrete semantic arguments.  Keep the view
        # JSON-only and exclude all Runtime payloads/raw rows by construction.
        context = context.model_copy(update={
            "decisionState": decision_state.model_dump(mode="json"),
        })
    # Legacy planners retain their historical policy payload temporarily.  A
    # dynamic HybridIntentPlannerPort uses only the explicit six-block input;
    # this branch never calls build_decision_policy_state.
    policy_state = None if new_policy_path else build_decision_policy_state(context)
    policy_input = (
        build_scientific_policy_input(decision_state)
        if new_policy_path and isinstance(decision_state, ScientificDecisionState)
        else None
    )
    # A dynamic policy/materializer pair owns the complete State -> Action ->
    # arguments decision.  It remains constrained by allowedActions and the
    # Java schema contract, but no semantic helper may replace its query with
    # a catalog template before Java has validated the model proposal.
    semantic_action = None if dynamic_materialization else semantic_scientific_next_action(context)
    policy_first = bool(getattr(planner, "prefer_policy_action", False))
    if semantic_action is not None and not policy_first:
        result = ScientificPlannerResult(
            action=semantic_action,
            mode="deterministic",
            fallbackCode="SCIENTIFIC_RUNTIME_SEMANTIC_PATH_ENFORCED",
        )
    else:
        try:
            if new_policy_path and policy_input is not None:
                result = planner.plan_action_with_state(context, policy_input)
            else:
                result = planner.plan_action(context)
        except Exception as exc:
            # Keep a bounded runtime-only diagnostic for canary/audit output;
            # never copy provider details into the six-block policy state.
            state["plannerFailureDetail"] = (
                f"{type(exc).__name__}:{str(exc).strip()[:240]}"
            )
            if new_policy_path and str(exc) in {
                "POLICY_ACTION_NOT_AVAILABLE",
                "POLICY_DECISION_FAILED",
                "MATERIALIZATION_ACTION_MISMATCH",
            }:
                # A new-state policy/materialization contract failure is
                # terminal for this planning turn. Runtime must not select a
                # different scientific Action as a recovery strategy.
                return _fail(state, "plan_action", "REJECTED", str(exc))
            if dynamic_materialization:
                return _fail(state, "plan_action", "FAILED", "DYNAMIC_ACTION_PLANNER_FAILED")
            result = DeterministicScientificPlanner().plan_action(context)
    # ``sft_policy`` is a genuine model decision via the closed Decision
    # Policy endpoint.  It must have the same trace provenance as the
    # full-action ``model`` planner: later Runtime repairs can then be
    # audited as raw model action -> final executed action.  Only the
    # deterministic fallback modes leave ``raw_action`` empty.
    model_decision = result.mode in {"model", "sft_policy"}
    raw_action = (
        (getattr(result, "rawAction", None) or result.action.actionName)
        if model_decision else None
    )
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
    for repair_code in getattr(result, "repairCodes", ()):
        if repair_code not in state.setdefault("fallbackCodes", []):
            state["fallbackCodes"].append(repair_code)
            _audit(state, "plan_action", "COMPLETED", error_code=repair_code)
    action = result.action
    if action.actionName == "inspect_cohort" and not state.get("observations") \
            and not dynamic_materialization:
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
    dynamic_materialization = bool(getattr(planner, "dynamic_action_materialization", False))
    has_analyzable_observation = any(
        observation.source == "java_controlled_read"
        for observation in state.get("observations", [])
    )
    if action.actionName in analysis_actions and not has_analyzable_observation:
        if dynamic_materialization:
            return _fail(state, "plan_action", "REJECTED", "ANALYSIS_REQUIRES_TABULAR_OBSERVATION")
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
    if repeated_validation and not dynamic_materialization:
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
    if repeated_bounded_read and not dynamic_materialization:
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
    if repeated_projection and not dynamic_materialization:
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
    if repeated_inspection and not dynamic_materialization:
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
    if premature_stratified_without_comparison and not dynamic_materialization:
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
    if premature_exploration_finish and not dynamic_materialization:
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
    if not dynamic_materialization and request.intent == "data_fact" and has_java_observation \
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
    if repeated_focused_projection and not dynamic_materialization:
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
    if premature_focused_without_analysis and not dynamic_materialization:
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
    if premature_focused_finish and not dynamic_materialization:
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
    if isinstance(action, FinishAction) and not dynamic_materialization:
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
    if new_policy_path and isinstance(policy_input, ScientificPolicyInput) \
            and action.actionName not in policy_input.action_space.available_actions:
        # New-state policy decisions are bounded by Hard Availability.  A
        # mismatch is a technical policy error; Runtime never substitutes a
        # different scientific action.
        return _fail(state, "plan_action", "REJECTED", "POLICY_ACTION_NOT_AVAILABLE")
    if terminal_finish_turn and action.actionName != "finish":
        # A terminal allowance only opens one Policy decision. Runtime must
        # not replace a non-terminal proposal with finish itself.
        return _fail(state, "plan_action", "REJECTED", "POLICY_ACTION_NOT_AVAILABLE")
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
    policy_available_actions = (
        list(policy_input.action_space.available_actions)
        if isinstance(policy_input, ScientificPolicyInput)
        else list(request.allowedActions)
    )
    trace_decision_reason = (
        getattr(result, "decisionReason", None)
        or _decision_reason(action.actionName, stop_reason_for_decision)
    )
    trace_alternative_actions = list(getattr(result, "alternativeActions", ()))
    if not trace_alternative_actions:
        trace_alternative_actions = [
            candidate for candidate in policy_available_actions
            if candidate != action.actionName
        ]
    legacy_policy_fields = policy_state or {}
    state.setdefault("decisionRecords", []).append(TraceDecision(
        observationStateCode=observation_state,
        allowedActions=policy_available_actions,
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
        decision_reason=trace_decision_reason,
        selected_action=action.actionName,
        alternative_actions=trace_alternative_actions,
        stop_reason=stop_reason_for_decision,
        raw_action=raw_action,
        final_action=action.actionName,
        planner_origin=planner_origin,
        repair_code=decision_repair_codes[-1] if decision_repair_codes else None,
        repair_codes=decision_repair_codes,
        task_kind=legacy_policy_fields.get("task_kind"),
        goal_code=legacy_policy_fields.get("goal_code"),
        observation_flags=list(legacy_policy_fields.get("observation_flags", [])),
        history_actions=list(legacy_policy_fields.get("history_actions", [])),
        candidate_actions=list(legacy_policy_fields.get("candidate_actions", [])),
        policy_input_version=("scientific-decision-state-v1" if policy_input is not None else None),
        state_snapshot=(policy_input.model_dump(mode="json") if policy_input is not None else None),
        available_actions=policy_available_actions,
        objective_resolution=(
            deepcopy(state.get("objectiveResolution"))
            if isinstance(state.get("objectiveResolution"), dict)
            else None
        ),
        materializer_origin=(
            getattr(result, "materializerOrigin", "unknown")
            if getattr(result, "materializerOrigin", "unknown") != "unknown"
            else (
                "runtime_owned" if dynamic_materialization and getattr(result, "runtimeOwned", False)
                else "model" if dynamic_materialization and model_decision
                else "model" if result.mode == "model"
                else "deterministic"
            )
        ),
        policy_origin=(
            getattr(result, "policyOrigin", "unknown")
            if getattr(result, "policyOrigin", "unknown") != "unknown"
            else (
                "qwen_model" if new_policy_path and model_decision
                else "legacy_model" if model_decision
                else "deterministic_fallback"
            )
        ),
        catalog_hash=(
            _sha256_model(state["schemaCatalog"])
            if dynamic_materialization and state.get("schemaCatalog") is not None else None
        ),
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
                and (
                    state.get("plannerFeedback")
                    or state.get("dynamicMaterialization", False)
                ):
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
    planner: ScientificPlannerPort,
    java_port: JavaAgentToolPort) -> ScientificState:
    request = state["request"]
    query_plan = action.arguments.queryPlan
    effective_query_plan = query_plan
    include_analysis_sample_key = False
    if getattr(planner, "dynamic_action_materialization", False):
        # Dynamic Scientific Runtime never sends the legacy raw-SQL shape to
        # Java.  The normal planner path already rejects it; this second
        # boundary protects direct/test planner injections as well.
        if query_plan is None:
            return _fail(state, "execute_action", "REJECTED", "DYNAMIC_QUERY_PLAN_REQUIRED")
        # Sample identity is intentionally sensitive and therefore omitted
        # from the Materializer's catalog view.  For a raw abundance
        # projection, request Java's run-scoped opaque key at the Runtime
        # boundary so the typed operator can collapse duplicate
        # sample×feature rows.  This is a technical projection repair, not a
        # scientific Action or a hard-coded QueryPlan.
        catalog = state.get("schemaCatalog")
        if (
            query_plan is not None
            and not query_plan.aggregations
            and not query_plan.group_by
            and "abundance.value" in query_plan.select_fields
            and "abundance.feature" in query_plan.select_fields
            and catalog is not None
        ):
            internal_fields = set(getattr(catalog, "internalAnalysisFields", []) or [])
            if "analysis.sample_key" in internal_fields:
                include_analysis_sample_key = True
                _record_runtime_repair(state, "RUNTIME_SAMPLE_KEY_BOUND")
                effective_query_plan = _sample_bounded_query_plan(query_plan, catalog)
                if effective_query_plan is not query_plan:
                    _record_runtime_repair(state, "RUNTIME_SAMPLE_BOUND_APPLIED")
        # Bind the minimum semantic projection required by the task before it
        # crosses the Java boundary.  The requirement set is derived from the
        # validated Task Understanding output and the Java Catalog; it never
        # contains a physical column, raw label, or a Policy recommendation.
        if action.actionName == "execute_read_query":
            decision_state = state.get("decisionState")
            if isinstance(decision_state, ScientificDecisionState):
                requirements = derive_data_requirements(
                    decision_state,
                    state.get("schemaCatalog"),
                    unavailable_fields=unavailable_fields_from_state(state),
                )
                try:
                    effective_query_plan = prune_unavailable_query_plan(
                        effective_query_plan,
                        unavailable_fields_from_state(state),
                        state.get("schemaCatalog"),
                    )
                    augmented = augment_query_plan_with_fields(
                        effective_query_plan,
                        requirements.required_fields,
                        state.get("schemaCatalog"),
                    )
                    effective_query_plan = prune_unavailable_query_plan(
                        augmented,
                        unavailable_fields_from_state(state),
                        state.get("schemaCatalog"),
                    )
                except DataRequirementError:
                    return _fail(
                        state,
                        "execute_action",
                        "REJECTED",
                        "QUERY_DATA_REQUIREMENTS_UNSATISFIABLE",
                    )
                if effective_query_plan != query_plan:
                    _record_runtime_repair(state, "RUNTIME_DATA_REQUIREMENT_FIELDS_BOUND")
        # Keep the effective technical projection on the current action.  The
        # following Observation Builder therefore records exactly what Java
        # received, including any requirement-bound covariate fields.
        if effective_query_plan is not query_plan:
            arguments = action.arguments.model_copy(update={
                "queryPlan": effective_query_plan,
                "limit": effective_query_plan.limit,
            })
            action = action.model_copy(update={"arguments": arguments})
            state["currentAction"] = action
            query_plan = effective_query_plan
    call_id = "call-" + sha256(
        f"{request.traceId}|{request.taskId}|{action.actionId}|execute_read_query".encode()
    ).hexdigest()[:32]
    call = ExecuteReadQueryJavaToolCall(
        toolName="execute_read_query",
        runId=request.runId,
        toolCallId=call_id,
        arguments=ExecuteReadQueryArguments(
            queryPlan=effective_query_plan,
            sql=action.arguments.sql,
            # A model-authored action limit is a row safety hint.  Once the
            # Runtime attaches a sample-bounded projection it owns the final
            # finite row cap so N unique samples can survive the abundance
            # fan-out; Java still validates the bound and applies LIMIT.
            limit=(
                effective_query_plan.limit
                if effective_query_plan is not None
                else action.arguments.limit
            ),
            includeAnalysisSampleKey=include_analysis_sample_key,
        ),
    )
    if effective_query_plan is not None:
        _annotate_last_decision(
            state,
            query_plan_hash=_sha256_model(effective_query_plan),
            query_plan_validation_status="failed",
        )
    else:
        _annotate_last_decision(state, query_plan_validation_status="legacy")
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
    _annotate_last_decision(state, query_plan_validation_status="passed")
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
        queryPlanFields=(
            list(dict.fromkeys([
                *query_plan.select_fields,
                *(item.field for item in query_plan.aggregations),
            ]))
            if action.arguments.queryPlan is not None else []
        ),
        queryPlanRelationPath=(
            list(query_plan.relation_path)
            if query_plan is not None else []
        ),
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


def _available_analysis_fields(state: ScientificState) -> list[str]:
    """Expose verified semantic field IDs, never physical catalog names."""

    catalog = state.get("schemaCatalog")
    if catalog is None:
        return []
    fields: list[str] = []
    for entity in catalog.entities:
        entity_id = entity.entityId or entity.entityName
        for field in entity.fields:
            if field.sensitive or field.semanticStatus != "verified":
                continue
            field_id = field.fieldId or f"{entity_id}.{field.name}"
            fields.append(field_id)
    return list(dict.fromkeys(fields))[:64]


def _semantic_distinct_counts(
    state: ScientificState,
    rows: list[dict[str, object]],
    field_ids: list[str],
) -> dict[str, int]:
    """Compute deterministic dimension coverage from the Java observation.

    The registry consumes semantic IDs and counts only.  Alias resolution is
    kept here, where the Runtime already owns the Catalog-to-payload proof;
    the Registry never guesses physical column names.
    """

    counts: dict[str, int] = {}
    for field_id in field_ids:
        values: set[str] = set()
        aliases = _catalog_field_aliases(field_id)
        # The Java typed read is bounded at 20,000 rows.  Counting only the
        # first 1,000 can make an otherwise complete sample-bounded cohort
        # look single-group when rows happen to be ordered by one disease;
        # that would incorrectly reject the typed capability.  This remains a
        # finite Runtime-owned bound and exposes counts only, never payloads.
        for row in rows[:_SAMPLE_BOUNDED_ROW_LIMIT]:
            if not isinstance(row, dict):
                continue
            value = next((row[alias] for alias in aliases if alias in row), None)
            if value is not None:
                values.add(str(value))
        counts[field_id] = len(values)
    return counts


def _execute_analysis_action(
    state: ScientificState,
    action: AnalyzeProjectionAction | CompareGroupsAction | StratifiedAnalysisAction
    | AdjustConfoundersAction | CrossProjectValidateAction | CrossDiseaseValidateAction,
    planner: ScientificPlannerPort,
) -> ScientificState:
    dynamic_materialization = bool(getattr(planner, "dynamic_action_materialization", False))
    observation_ids, analysis_goal, dimensions, confounders = _analysis_inputs(action)
    observations = {
        item.observationId: item for item in state.get("observations", [])
    }
    # A model can legally choose an analysis action after metadata inspection
    # or evidence retrieval, but those observations are not tabular analysis
    # inputs. Dynamic analysis consumes only Java execute_read_query outputs;
    # Runtime-owned inspect_cohort is a metadata probe and knowledge_hybrid is
    # literature evidence. Preserve the repair as provenance rather than
    # silently mixing payloads from different observation types.
    latest_tabular = next(
        (item for item in reversed(state.get("observations", []))
         if item.actionName == "execute_read_query"
         and item.source == "java_controlled_read"
         and item.status in {"VALIDATED", "PARTIAL"}),
        None,
    )

    def _has_sample_level_abundance(observation_id: str) -> bool:
        payload = state.get("rawObservationPayloads", {}).get(observation_id)
        if not isinstance(payload, dict):
            return False
        columns = {
            value for value in payload.get("columns", [])
            if isinstance(value, str)
        }
        rows = payload.get("rows")
        return (
            bool(rows)
            and "a_analysis_sample_key" in columns
            and "a_abundance_feature" in columns
            and "a_abundance_value" in columns
        )

    if dynamic_materialization:
        tabular_ids = [
            observation_id for observation_id in observation_ids
            if observation_id in observations
            and observations[observation_id].actionName == "execute_read_query"
            and observations[observation_id].source == "java_controlled_read"
        ]
        if tabular_ids:
            if tabular_ids != observation_ids:
                observation_ids = tabular_ids
                _record_runtime_repair(
                    state,
                    "DYNAMIC_ANALYSIS_NON_TABULAR_OBSERVATION_FILTERED",
                )
        elif latest_tabular is not None:
            observation_ids = [latest_tabular.observationId]
            _record_runtime_repair(
                state,
                "DYNAMIC_ANALYSIS_INSPECTION_REBOUND",
            )
        if action.actionName in {
            "compare_groups",
            "stratified_analysis",
            "adjust_confounders",
            "cross_project_validate",
            "cross_disease_validate",
        }:
            sample_level_ids = [
                observation_id for observation_id in observation_ids
                if _has_sample_level_abundance(observation_id)
            ]
            if sample_level_ids and sample_level_ids != observation_ids:
                # Aggregate/grouped rows are valid coverage observations but
                # cannot be independent inputs to a sample-level operator.
                # Keep only the Java projection carrying the opaque key; this
                # is a technical input-shape guard, not a scientific choice.
                observation_ids = sample_level_ids
                _record_runtime_repair(
                    state,
                    "DYNAMIC_ANALYSIS_AGGREGATE_OBSERVATION_FILTERED",
                )
    elif any(
        observation_id in observations
        and observations[observation_id].source == "knowledge_hybrid"
        for observation_id in observation_ids
    ) and latest_tabular is not None:
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
    available_payload_fields: list[str] = []
    observation_semantic_fields: dict[str, list[str]] = {}
    allowed_preview_columns: set[str] = set()
    for observation_id in observation_ids:
        payload = state.get("rawObservationPayloads", {}).get(observation_id)
        semantic_fields, safe_columns = _payload_semantic_fields(state, payload)
        observation_semantic_fields[observation_id] = semantic_fields
        available_payload_fields.extend(semantic_fields)
        allowed_preview_columns.update(safe_columns)
    columns, rows = build_analysis_preview(
        preview_payload,
        allowed_columns=allowed_preview_columns or None,
    )
    from mico_agent_runtime.contracts.generated_analysis import AnalysisPlannerContext
    try:
        generated = None
        typed_analysis_plan = None
        semantic_path = "SCIENTIFIC_RUNTIME_SEMANTIC_PATH_ENFORCED" in state.get(
            "fallbackCodes", []
        )
        if dynamic_materialization:
            generator = getattr(planner, "generate_typed_analysis", None)
            if not callable(generator):
                return _fail(
                    state,
                    "execute_action",
                    "FAILED",
                    "ANALYSIS_TYPED_PLAN_MATERIALIZER_NOT_CONFIGURED",
                )
            typed_context = AnalysisPlannerContext(
                questionSummary=analysis_goal,
                workflow=action.actionName,
                actionName=action.actionName,
                sourceObservationIds=observation_ids,
                availableSemanticFields=list(dict.fromkeys(available_payload_fields))
                or _available_analysis_fields(state),
                requiredSemanticFields=(
                    required_cross_validation_fields(
                        state.get("schemaCatalog"),
                        action.actionName,
                        analysis_goal,
                    )
                    if action.actionName in {
                        "cross_project_validate",
                        "cross_disease_validate",
                    }
                    else []
                ),
                # Cross-validation has two dimensions in the v2 typed plan:
                # group_field is the within-group comparison and
                # validation_field is the repeated project/disease dimension.
                # Do not force the latter into group_field.
                requiredGroupField=None,
                columns=columns,
                previewRows=rows,
                executionFeedback=[],
            )
            feedback: list[str] = []
            for attempt in range(_MAX_ANALYSIS_EXECUTION_ATTEMPTS):
                try:
                    typed_context = typed_context.model_copy(update={"executionFeedback": feedback})
                    planned = generator(typed_context)
                    if planned.mode not in {"model", "deterministic"}:
                        raise GeneratedAnalysisError("ANALYSIS_TYPED_PLAN_MODE_INVALID")
                    expected_analysis_type = _ACTION_ANALYSIS_TYPE.get(action.actionName)
                    if (
                        expected_analysis_type is not None
                        and planned.plan.analysis_type != expected_analysis_type
                    ):
                        # Materialization cannot change the Action family.
                        return _fail(
                            state,
                            "execute_action",
                            "REJECTED",
                            "MATERIALIZATION_ACTION_MISMATCH",
                        )
                    if planned.mode == "deterministic":
                        if not getattr(
                            planner,
                            "allow_deterministic_materializer_fallback",
                            True,
                        ):
                            # The Gemini-only 4D-1B Happy Path must prove
                            # that Gemini authored the AnalysisPlan.  Do not
                            # silently replace a failed provider response
                            # with a deterministic plan.
                            return _fail(
                                state,
                                "execute_action",
                                "FAILED",
                                "DYNAMIC_ACTION_MATERIALIZATION_FAILED",
                            )
                        # This is an execution-plan fallback, not a new
                        # scientific decision. Keep the selected Action and
                        # expose the fallback provenance in the trace.
                        fallback_code = getattr(
                            planned,
                            "fallbackCode",
                            None,
                        ) or "ANALYSIS_TYPED_PLAN_DETERMINISTIC_FALLBACK"
                        _record_runtime_repair(state, fallback_code)
                        _annotate_last_decision(
                            state,
                            materializer_origin="deterministic_fallback",
                        )
                    else:
                        _annotate_last_decision(
                            state,
                            materializer_origin=getattr(
                                planner,
                                "materializer_origin",
                                getattr(
                                    getattr(planner, "_base", None),
                                    "materializer_origin",
                                    "model",
                                ),
                            ),
                        )
                    for repair_code in getattr(planned, "repairCodes", []):
                        _record_runtime_repair(state, repair_code)
                    if set(planned.plan.source_observation_ids) != set(observation_ids):
                        raise GeneratedAnalysisError("ANALYSIS_TYPED_PLAN_SOURCE_MISMATCH")
                    available_fields = set(typed_context.availableSemanticFields)
                    referenced_fields = {
                        value for value in (
                            planned.plan.outcome,
                            planned.plan.feature_field,
                            planned.plan.group_field,
                            *planned.plan.covariates,
                            *planned.plan.stratify_by,
                        ) if value is not None
                    }
                    if not available_fields or not referenced_fields.issubset(available_fields):
                        raise GeneratedAnalysisError("ANALYSIS_TYPED_PLAN_FIELD_REJECTED")
                    # The registry is a capability gate, not a policy
                    # decision.  Older test/demonstration catalogs may not
                    # yet carry the 3B-1 scientific capability bridge; keep
                    # their historical execution path until Java supplies
                    # that metadata, while all capability-aware catalogs are
                    # fail-closed here.
                    catalog = state.get("schemaCatalog")
                    catalog_has_capabilities = bool(
                        catalog is not None
                        and any(
                            field.scientificCapabilities
                            for entity in catalog.entities
                            for field in entity.fields
                        )
                    )
                    if catalog_has_capabilities:
                        capability_match = match_analysis_capability(
                            planned.plan,
                            catalog,
                            AnalysisCapabilityContext(
                                available_observation_ids=observation_ids,
                                available_fields=typed_context.availableSemanticFields,
                                observation_fields=observation_semantic_fields,
                                distinct_counts=_semantic_distinct_counts(
                                    state,
                                    payload_rows,
                                    list(referenced_fields),
                                ),
                            ),
                        )
                        if capability_match.mode == "UNSUPPORTED":
                            return _fail(
                                state,
                                "execute_action",
                                "REJECTED",
                                "ANALYSIS_CAPABILITY_UNSUPPORTED",
                            )
                        if capability_match.mode == "SUPPORTED_GENERATED":
                            # A v2 typed AnalysisPlan is never silently
                            # converted into a generated compatibility run.
                            # If the registry cannot prove this concrete
                            # shape is covered by an approved typed
                            # operator, fail closed and preserve the selected
                            # Action for Runtime/audit handling.
                            return _fail(
                                state,
                                "execute_action",
                                "REJECTED",
                                "ANALYSIS_CAPABILITY_UNSUPPORTED",
                            )
                    _annotate_last_decision(
                        state,
                        analysis_plan_hash=_sha256_model(planned.plan),
                        analysis_execution_status="failed",
                    )
                    generated = execute_typed_analysis(
                        planned.plan,
                        payload_rows,
                        len(payload_rows),
                        planner_mode=planned.mode,
                    )
                    typed_analysis_plan = planned.plan
                    _annotate_last_decision(state, analysis_execution_status="passed")
                    break
                except GeneratedAnalysisError as exc:
                    if exc.code in {
                        "ANALYSIS_TYPED_FEATURE_FIELD_REQUIRED",
                        "ANALYSIS_TYPED_SAMPLE_KEY_REQUIRED",
                        "ANALYSIS_TYPED_SAMPLE_LEVEL_REQUIRED",
                        "ANALYSIS_TYPED_FEATURE_AWARE_SHAPE_REQUIRED",
                    }:
                        # These are scientific-semantics safety failures, not
                        # malformed model output. Retrying the same plan or
                        # switching to generated Python could reintroduce
                        # feature pooling/pseudo-replication, so fail closed
                        # with the stable diagnostic code.
                        return _fail(state, "execute_action", "REJECTED", exc.code)
                    if exc.code in TYPED_ANALYSIS_SANDBOX_FALLBACK_CODES:
                        code_generator = getattr(planner, "generate_analysis", None)
                        if not callable(code_generator):
                            return _fail(
                                state,
                                "execute_action",
                                "FAILED",
                                "ANALYSIS_GENERATION_FAILED",
                            )
                        code_feedback: list[str] = []
                        for code_attempt in range(_MAX_ANALYSIS_EXECUTION_ATTEMPTS):
                            try:
                                code_context = typed_context.model_copy(update={
                                    "executionFeedback": code_feedback,
                                })
                                code_planned = code_generator(code_context)
                                if code_planned.mode != "model":
                                    raise GeneratedAnalysisError(
                                        "ANALYSIS_CODE_MODEL_REQUIRED"
                                    )
                                generated = execute_generated_analysis(
                                    code_planned.plan,
                                    rows,
                                    len(payload_rows),
                                    planner_mode=code_planned.mode,
                                )
                                _annotate_last_decision(
                                    state,
                                    analysis_execution_status="passed",
                                )
                                break
                            except GeneratedAnalysisError as code_exc:
                                if code_attempt + 1 >= _MAX_ANALYSIS_EXECUTION_ATTEMPTS:
                                    return _fail(
                                        state,
                                        "execute_action",
                                        "FAILED",
                                        "ANALYSIS_GENERATION_FAILED",
                                    )
                                code_feedback = [
                                    getattr(code_exc, "reasonCode", None)
                                    or code_exc.code
                                ]
                                state.setdefault("fallbackCodes", []).append(
                                    "ANALYSIS_CODE_REPLAN"
                                )
                                _audit(
                                    state,
                                    "execute_action",
                                    "COMPLETED",
                                    error_code="ANALYSIS_CODE_REPLAN",
                                )
                            except Exception:
                                if code_attempt + 1 >= _MAX_ANALYSIS_EXECUTION_ATTEMPTS:
                                    return _fail(
                                        state,
                                        "execute_action",
                                        "FAILED",
                                        "ANALYSIS_GENERATION_FAILED",
                                    )
                                code_feedback = ["ANALYSIS_CODE_REJECTED"]
                        if generated is None:
                            return _fail(
                                state,
                                "execute_action",
                                "FAILED",
                                "ANALYSIS_GENERATION_FAILED",
                            )
                        break
                    if attempt + 1 >= _MAX_ANALYSIS_EXECUTION_ATTEMPTS:
                        return _fail(
                            state,
                            "execute_action",
                            "FAILED",
                            "ANALYSIS_GENERATION_FAILED",
                        )
                    feedback = [exc.code]
                    state.setdefault("fallbackCodes", []).append("ANALYSIS_TYPED_PLAN_REPLAN")
                    state.setdefault("fallbackCodes", []).append(
                        f"ANALYSIS_TYPED_PLAN_REPLAN_{exc.code}"
                    )
                    _audit(
                        state,
                        "execute_action",
                        "COMPLETED",
                        error_code=exc.code,
                    )
                except Exception:
                    if attempt + 1 >= _MAX_ANALYSIS_EXECUTION_ATTEMPTS:
                        return _fail(
                            state,
                            "execute_action",
                            "FAILED",
                            "ANALYSIS_GENERATION_FAILED",
                        )
                    feedback = ["ANALYSIS_TYPED_PLAN_REJECTED"]
                    state.setdefault("fallbackCodes", []).append("ANALYSIS_TYPED_PLAN_REPLAN")
            if generated is None:
                return _fail(state, "execute_action", "FAILED", "ANALYSIS_GENERATION_FAILED")
        elif semantic_path:
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
                        if dynamic_materialization:
                            return _fail(
                                state,
                                "execute_action",
                                "FAILED",
                                "ANALYSIS_GENERATION_FAILED",
                            )
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
        if not dynamic_materialization:
            _annotate_last_decision(state, analysis_execution_status="legacy")
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
        feature_results = [
            AnalysisFeatureResult(
                featureName=item.featureName,
                metrics=item.metrics,
                status=item.status,
                supportStatus=(
                    "partial" if item.status == "insufficient_data" else "supported"
                ),
            )
            for item in generated.feature_results
        ]
        analysis_result = AnalysisResult(
            analysisId=analysis_id,
            actionName=plan.actionName,
            analysis_type=generated.analysisType,
            execution_mode=generated.execution_mode,
            method_used=generated.method_used,
            status="COMPLETED",
            plannerMode=generated.plannerMode,
            codeVersion=generated.codeVersion,
            sourceObservationIds=observation_ids,
            rowsAnalyzed=generated.rowCount,
            metrics=generated.metrics,
            group_results=generated.group_results,
            stratum_results=generated.stratum_results,
            validation_results=generated.validation_results,
            feature_results=feature_results,
            ranking_method=generated.ranking_method,
            adjusted_covariates=generated.adjusted_covariates,
            used_row_count=generated.used_row_count,
            dropped_row_count=generated.dropped_row_count,
            topFeatures=safe_features,
            evidence=analysis_evidence,
            limitations=[
                (
                    "typed_plan_executed_by_approved_operator"
                    if generated.codeVersion == "typed-analysis-operator-v1"
                    else "generated_code_was_sandbox_validated"
                ),
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
        schemaVersion=generated.codeVersion,
        dataSnapshotId=source_observation.dataSnapshotId,
        snapshotPersistence=source_observation.snapshotPersistence,
        featureVersion=source_observation.featureVersion,
        taxonomyVersion=source_observation.taxonomyVersion,
        sourceBatch=source_observation.sourceBatch,
    )
    state.setdefault("analysisResults", []).append(analysis_result)
    state.setdefault("analysisEvidence", []).extend(analysis_evidence)
    state["pendingTypedAnalysisPlan"] = typed_analysis_plan
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
        decision_state = state.get("decisionState")
        if isinstance(decision_state, ScientificDecisionState):
            # A terminal decision closes the workflow; it must not recompute
            # required objectives from analysis status and thereby resurrect a
            # Runtime-blocked objective.  The lifecycle resolution is owned by
            # Runtime and remains unchanged through the finish turn.
            resolution = state.get("objectiveResolution")
            if isinstance(resolution, dict) and isinstance(
                resolution.get("active_objectives"), list
            ):
                remaining = list(resolution["active_objectives"])
            else:
                remaining = list(decision_state.progress.remaining_objectives)
            state["decisionState"] = update_progress(
                decision_state,
                action.actionName,
                successful=True,
                remaining_objectives_override=remaining,
            )
        state["route"] = "synthesize"
        _audit(state, "execute_action", "COMPLETED")
        return state
    if isinstance(action, (ExecuteReadQueryAction, InspectCohortAction)):
        return _execute_read_query_action(state, action, planner, java_port)
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
    pending_payload = state.pop("pendingPayload", None)
    pending_response = state.pop("pendingResponse", None)
    pending_typed_plan = state.pop("pendingTypedAnalysisPlan", None)
    action = state.get("currentAction")
    if isinstance(observation, Observation):
        state.setdefault("observations", []).append(observation)
        decision_state = state.get("decisionState")
        if isinstance(decision_state, ScientificDecisionState):
            if observation.source == "java_controlled_read":
                query_plan = getattr(getattr(action, "arguments", None), "queryPlan", None)
                # A zero-row joined read can prove that an objective-scoped
                # secondary dimension has no coverage for this cohort. Record
                # that fact before the next planning turn so requirements and
                # QueryPlan enrichment cannot reintroduce the same relation.
                record_zero_coverage_capabilities(
                    state,
                    observation,
                    query_plan,
                    state.get("schemaCatalog"),
                )
                response_row_count = getattr(pending_response, "rowCount", None)
                decision_state = update_from_query_result(
                    decision_state,
                    observation,
                    pending_payload,
                    query_plan=query_plan,
                    catalog=state.get("schemaCatalog"),
                    row_count=response_row_count,
                    require_sample_level_outcome=bool(
                        state.get("dynamicMaterialization", False)
                    ),
                )
            elif observation.source == "python_bounded_analysis" \
                    and isinstance(pending_payload, AnalysisResult):
                analysis_plan = pending_typed_plan
                if analysis_plan is None:
                    plans = state.get("analysisPlans", [])
                    analysis_plan = plans[-1] if plans else None
                decision_state = update_from_analysis_result(
                    decision_state,
                    pending_payload,
                    analysis_plan=analysis_plan,
                    source_payloads=state.get("rawObservationPayloads"),
                    catalog=state.get("schemaCatalog"),
                )
            elif observation.source == "knowledge_hybrid":
                evidence = pending_payload if isinstance(pending_payload, list) else []
                decision_state = update_from_evidence_result(
                    decision_state,
                    evidence,
                    observation_status=observation.status,
                )
            state["decisionState"] = update_progress(
                decision_state,
                observation.actionName,
                successful=True,
                completed=observation.status == "VALIDATED",
            )
    if action is not None:
        state.setdefault("actionHistory", []).append(action.actionName)
    state["currentAction"] = None
    _refresh_action_availability(state)
    _audit(state, "update_state", "COMPLETED")
    return state


def _decide_continue_or_stop(state: ScientificState) -> ScientificState:
    """Continue until budget, with one policy-owned terminal finish turn."""
    request = state.get("request")
    if not isinstance(request, ResearchTask):
        return _fail(state, "decide_continue_or_stop", "FAILED", "RESEARCH_TASK_CONTRACT_INVALID")
    if len(state.get("actionHistory", [])) >= request.maxActions:
        decision_state = state.get("decisionState")
        available = (
            list(decision_state.action_space.available_actions)
            if isinstance(decision_state, ScientificDecisionState)
            else []
        )
        finish_only = "finish" in available and not any(
            action != "finish" for action in available
        )
        if (
            finish_only
            and state.get("dynamicMaterialization", False)
            and not state.get("terminalFinishTurnGranted", False)
        ):
            state["terminalFinishTurnGranted"] = True
            state.pop("stopReasonCode", None)
            state["route"] = "plan"
        else:
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
    objective_resolution = state.get("objectiveResolution", {})
    blocked_entries = [
        item for item in objective_resolution.get("resolutions", [])
        if isinstance(item, dict) and item.get("status") == "blocked"
    ] if isinstance(objective_resolution, dict) else []
    blocked_objectives = [
        str(item.get("objective"))
        for item in blocked_entries
        if isinstance(item.get("objective"), str)
    ]
    objective_limitation = bool(blocked_entries)
    conclusion = synthesis.conclusion or (
        "The dynamic exploration completed only validated, bounded evidence actions; "
        "domain interpretation requires review of the bound evidence."
    )
    if objective_limitation:
        conclusion = (
            f"{conclusion.rstrip()} "
            "Some requested objectives were not executable with the observed data "
            f"({', '.join(blocked_objectives)}); they are reported with explicit "
            "data or environment limitation provenance."
        )[:1600]
    limitations = [
        "snapshot_is_transient_and_not_replayable",
        "internal_record_is_not_a_subject_count",
        "subject_link_is_unverified",
        "dynamic_query_is_java_validated_and_bounded",
        "generated_analysis_is_bounded_and_sandboxed",
        "scientific_evidence_is_not_clinical_diagnosis",
    ]
    # Preserve the provenance of each blocked objective.  A missing project
    # dimension is a data limitation, while an unavailable knowledge/embedding
    # backend is an environment limitation.  These are Runtime-derived report
    # metadata and do not affect Policy action selection.
    for limitation_code in limitation_codes_from_resolution(objective_resolution):
        if limitation_code not in limitations:
            limitations.append(limitation_code)
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
        conclusion=conclusion,
        limitations=limitations,
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
    task_understanding_port: TaskUnderstandingPort | None = None,
):
    synthesis_port = synthesis_port or DeterministicGraphRagSynthesisPort()
    builder = StateGraph(ScientificState)
    builder.add_node(
        "validate_research_task",
        lambda state: _validate_task(
            state,
            knowledge_available=knowledge_port is not None,
        ),
    )
    builder.add_node(
        "understand_task",
        lambda state: _understand_task(task_understanding_port, state),
    )
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
        lambda state: _route_after_status(state, "understand_task"),
    )
    builder.add_conditional_edges(
        "understand_task",
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
