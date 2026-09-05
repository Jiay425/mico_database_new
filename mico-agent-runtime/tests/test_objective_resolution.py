from types import SimpleNamespace

from mico_agent_runtime.contracts.decision_state import (
    ConfounderAdjustmentState,
    GroupComparisonState,
    ScientificAnalysisState,
    ScientificDataState,
    ScientificDecisionState,
    ScientificTaskState,
)
from mico_agent_runtime.contracts.schema_catalog import (
    SchemaEntitySemantics,
    SchemaFieldSemantics,
    SchemaSemanticCatalog,
)
from mico_agent_runtime.contracts.research import FinishAction, FinishArguments, ResearchTask
from mico_agent_runtime.runtime.action_availability import (
    ActionAvailabilityContext,
    evaluate_action_availability,
)
from mico_agent_runtime.runtime.objective_resolution import (
    active_remaining_objectives,
    completion_semantics_from_resolution,
    limitation_code_for_resolution,
    limitation_codes_from_resolution,
    resolve_objective_lifecycle,
)
from mico_agent_runtime.graph.scientific_workflow import (
    _decide_continue_or_stop,
    _execute_action,
    _plan_action,
    _refresh_action_availability,
    _synthesize_report,
)
from mico_agent_runtime.contracts.trace_eval import TraceDecision
from mico_agent_runtime.knowledge.synthesis import DeterministicGraphRagSynthesisPort
from mico_agent_runtime.ports.scientific_planner import ScientificPlannerResult


def _catalog(*, include_project: bool = True) -> SchemaSemanticCatalog:
    fields = [
        SchemaFieldSemantics(
            fieldId="sample.disease",
            name="disease",
            dataType="string",
            nullable=False,
            semanticStatus="verified",
            groupable=True,
            scientificCapabilities=["dimension"],
            description="Disease dimension",
        ),
        SchemaFieldSemantics(
            fieldId="abundance.value",
            name="value",
            dataType="number",
            nullable=False,
            semanticStatus="verified",
            aggregatable=True,
            scientificCapabilities=["outcome"],
            description="Numeric outcome",
        ),
    ]
    entities = [
        SchemaEntitySemantics(
            entityName="sample",
            sourceTable="sample",
            fields=[fields[0]],
        ),
        SchemaEntitySemantics(
            entityName="abundance",
            sourceTable="abundance",
            fields=[fields[1]],
        ),
    ]
    if include_project:
        entities.append(SchemaEntitySemantics(
            entityName="metadata",
            sourceTable="metadata",
            fields=[SchemaFieldSemantics(
                fieldId="metadata.project",
                name="project",
                dataType="string",
                nullable=True,
                semanticStatus="verified",
                groupable=True,
                scientificCapabilities=["dimension"],
                description="Project dimension",
            )],
        ))
    return SchemaSemanticCatalog(
        schemaVersion="schema-catalog-v1",
        source="java_schema_contract",
        generatedAt="2026-09-03T00:00:00Z",
        entities=entities,
        queryRules=[
            "select_or_with_only",
            "explicit_columns_only",
            "bounded_limit_required",
        ],
    )


def _state(*, objectives: list[str], include_project: bool = True) -> ScientificDecisionState:
    return ScientificDecisionState(
        task=ScientificTaskState(query="objective lifecycle test", objectives=objectives),
        data_state=ScientificDataState(
            has_tabular_data=True,
            row_count=100,
            available_dimensions=["sample.disease"],
            available_outcomes=["abundance.value"],
            group_state={
                "group_field": "sample.disease",
                "group_count": 2,
                "group_sizes": {"A": 50, "B": 50},
            },
            project_state={
                "has_project_field": False,
                "project_count": 0,
            },
            covariate_state={
                "available_covariates": ["sample.age"],
            },
        ),
        analysis_state=ScientificAnalysisState(
            group_comparison=GroupComparisonState(status="completed"),
            confounder_adjustment=ConfounderAdjustmentState(status="completed"),
        ),
    )


def _availability(state: ScientificDecisionState, catalog: SchemaSemanticCatalog):
    return evaluate_action_availability(
        state,
        ActionAvailabilityContext(catalog=catalog),
    )


def test_blocked_project_objective_allows_finish_with_limitation() -> None:
    state = _state(
        objectives=["cross_project_validation"],
        include_project=False,
    )
    catalog = _catalog(include_project=False)
    preliminary = _availability(state, catalog)
    resolution = resolve_objective_lifecycle(
        state,
        catalog=catalog,
        availability_reasons=preliminary.blocked_reasons,
    )

    assert resolution.blocked_objectives == ("cross_project_validation",)
    assert resolution.completed_objectives == ()
    assert resolution.active_objectives == ()
    assert resolution.all_requested_objectives_completed is False
    assert resolution.limitations_present is True
    assert active_remaining_objectives(state, resolution) == []

    finished = _availability(state, catalog)
    # The first pass intentionally has no lifecycle proof.  Supplying the
    # Runtime resolution is what opens finish; no scientific Action is added.
    finished = evaluate_action_availability(
        state,
        ActionAvailabilityContext(
            catalog=catalog,
            blocked_objectives=resolution.blocked_objectives,
        ),
    )
    assert "cross_project_validate" not in finished.available_actions
    assert "finish" in finished.available_actions


def test_missing_project_field_is_not_marked_blocked_before_a_bounded_attempt() -> None:
    state = _state(
        objectives=["cross_project_validation"],
        include_project=True,
    )
    catalog = _catalog(include_project=True)
    preliminary = _availability(state, catalog)
    resolution = resolve_objective_lifecycle(
        state,
        catalog=catalog,
        availability_reasons=preliminary.blocked_reasons,
    )

    assert resolution.active_objectives == ("cross_project_validation",)
    assert resolution.blocked_objectives == ()


def test_explicit_project_attempt_with_zero_values_is_blocked() -> None:
    state = _state(
        objectives=["cross_project_validation"],
        include_project=True,
    )
    catalog = _catalog(include_project=True)
    observation = SimpleNamespace(
        queryPlanFields=["sample.disease", "metadata.project", "abundance.value"],
        queryPlanRelationPath=["sample_to_metadata"],
    )
    preliminary = _availability(state, catalog)
    resolution = resolve_objective_lifecycle(
        state,
        catalog=catalog,
        availability_reasons=preliminary.blocked_reasons,
        observations=[observation],
    )

    assert resolution.blocked_objectives == ("cross_project_validation",)
    assert resolution.resolutions[0].reason_code == "PROJECT_DIMENSION_UNAVAILABLE"


def test_task_d_uses_same_generic_blocked_lifecycle_for_missing_dimension() -> None:
    state = _state(
        objectives=["cross_project_validation"],
        include_project=False,
    )
    catalog = _catalog(include_project=False)
    resolution = resolve_objective_lifecycle(
        state,
        catalog=catalog,
        availability_reasons={
            "cross_project_validate": "PROJECT_DIMENSION_NOT_IN_OBSERVATION",
        },
    )

    assert resolution.blocked_objectives == ("cross_project_validation",)
    assert resolution.resolutions[0].status == "blocked"
    assert resolution.resolutions[0].reason_code == "PROJECT_DIMENSION_UNAVAILABLE"


def test_blocked_objective_does_not_change_frozen_policy_schema() -> None:
    state = _state(objectives=["cross_project_validation"], include_project=False)
    serialized = state.model_dump(mode="json")
    assert set(serialized) == {
        "task",
        "data_state",
        "analysis_state",
        "evidence_state",
        "progress",
        "action_space",
    }


def test_final_report_preserves_blocked_objective_as_limitation() -> None:
    state = {
        "request": ResearchTask.model_validate({
            "runId": "run-" + "a" * 32,
            "taskId": "task-" + "b" * 32,
            "requesterId": "principal-" + "c" * 32,
            "traceId": "trace-" + "d" * 32,
            "question": "validate the result across projects",
            "intent": "scientific_exploration",
            "requestedScopes": ["mico:query:read", "mico:research:read"],
            "allowedActions": ["finish"],
            "maxActions": 6,
            "createdAt": "2026-09-03T00:00:00Z",
        }),
        "observations": [],
        "rawObservationPayloads": {},
        "analysisResults": [],
        "analysisEvidence": [],
        "objectiveResolution": {
            "resolutions": [{
                "objective": "cross_project_validation",
                "status": "blocked",
                "reason_code": "PROJECT_DIMENSION_UNAVAILABLE",
                "reason": "no valid project dimension",
                "action": "cross_project_validate",
            }],
        },
    }
    result = _synthesize_report(DeterministicGraphRagSynthesisPort(), state)
    assert result["report"] is not None
    assert "objective_unavailable_due_to_data" in result["report"].limitations
    assert "cross_project_validation" in (result["report"].conclusion or "")


def test_graph_refresh_exposes_active_only_and_opens_finish_after_other_goals() -> None:
    decision_state = _state(
        objectives=["cross_project_validation", "evidence_support"],
        include_project=False,
    )
    request = ResearchTask.model_validate({
        "runId": "run-" + "e" * 32,
        "taskId": "task-" + "f" * 32,
        "requesterId": "principal-" + "1" * 32,
        "traceId": "trace-" + "2" * 32,
        "question": "validate the result across projects",
        "intent": "scientific_exploration",
        "requestedScopes": ["mico:query:read", "mico:research:read", "mico:evidence:read"],
        "allowedActions": [
            "execute_read_query", "compare_groups", "cross_project_validate",
            "retrieve_evidence", "finish",
        ],
        "maxActions": 6,
        "createdAt": "2026-09-03T00:00:00Z",
    })
    runtime_state = {
        "request": request,
        "decisionState": decision_state,
        "schemaCatalog": _catalog(include_project=False),
        "knowledgeAvailable": True,
        "rawObservationPayloads": {},
        "observations": [],
    }
    _refresh_action_availability(runtime_state)
    assert runtime_state["decisionState"].progress.remaining_objectives == [
        "evidence_support",
    ]
    assert runtime_state["objectiveResolution"]["blocked_objectives"] == [
        "cross_project_validation",
    ]
    assert "finish" not in runtime_state["decisionState"].action_space.available_actions

    runtime_state["decisionState"].evidence_state.status = "completed"
    _refresh_action_availability(runtime_state)
    assert runtime_state["decisionState"].progress.remaining_objectives == []
    assert "finish" in runtime_state["decisionState"].action_space.available_actions
    assert runtime_state["objectiveResolution"]["all_requested_objectives_completed"] is False


def _runtime_request_for_lifecycle(*, max_actions: int = 6) -> ResearchTask:
    return ResearchTask.model_validate({
        "runId": "run-" + "9" * 32,
        "taskId": "task-" + "8" * 32,
        "requesterId": "principal-" + "7" * 32,
        "traceId": "trace-" + "6" * 32,
        "question": "compare the observed groups and validate project stability",
        "intent": "scientific_exploration",
        "requestedScopes": ["mico:query:read", "mico:research:read", "mico:evidence:read"],
        "allowedActions": [
            "execute_read_query", "compare_groups", "adjust_confounders",
            "cross_project_validate", "retrieve_evidence", "finish",
        ],
        "maxActions": max_actions,
        "createdAt": "2026-09-03T00:00:00Z",
    })


def _zero_project_observation() -> SimpleNamespace:
    return SimpleNamespace(
        observationId="observation-" + "0" * 32,
        actionName="execute_read_query",
        source="java_controlled_read",
        status="VALIDATED",
        rowCount=0,
        missingnessSummary=[],
        exclusionSummary=[],
        featureVersion=None,
        taxonomyVersion=None,
        sourceBatch=None,
        queryPlanFields=[
            "sample.disease", "sample.age", "abundance.feature",
            "abundance.value", "metadata.project",
        ],
        queryPlanRelationPath=["sample_to_abundance", "sample_to_metadata"],
    )


def _lifecycle_runtime_state(
    decision_state: ScientificDecisionState,
    *,
    max_actions: int = 6,
) -> dict[str, object]:
    return {
        "request": _runtime_request_for_lifecycle(max_actions=max_actions),
        "decisionState": decision_state,
        "schemaCatalog": _catalog(include_project=True),
        "knowledgeAvailable": True,
        "rawObservationPayloads": {},
        "observations": [_zero_project_observation()],
        "actionHistory": [],
    }


def test_zero_project_coverage_blocks_objective_on_next_state_refresh() -> None:
    state = _lifecycle_runtime_state(
        _state(
            objectives=["cross_project_validation", "evidence_support"],
            include_project=True,
        ),
    )

    _refresh_action_availability(state)

    assert state["objectiveResolution"]["blocked_objectives"] == [
        "cross_project_validation",
    ]
    assert state["decisionState"].progress.remaining_objectives == [
        "evidence_support",
    ]
    assert "cross_project_validate" not in (
        state["decisionState"].action_space.available_actions
    )
    blocked_entry = state["objectiveResolution"]["resolutions"][0]
    assert blocked_entry["resolved_at_step"] == state["decisionState"].progress.action_count
    assert blocked_entry["limitation_code"] == "objective_unavailable_due_to_data"


def test_blocked_objective_removed_from_remaining_objectives() -> None:
    state = _lifecycle_runtime_state(
        _state(objectives=["cross_project_validation"], include_project=True),
    )

    _refresh_action_availability(state)

    assert state["decisionState"].progress.remaining_objectives == []
    assert state["objectiveResolution"]["workflow_can_finish"] is True
    assert "finish" in state["decisionState"].action_space.available_actions


def test_finish_available_before_budget_dead_end() -> None:
    decision_state = _state(
        objectives=["cross_project_validation", "evidence_support"],
        include_project=True,
    ).model_copy(deep=True)
    decision_state.evidence_state.status = "completed"
    decision_state.progress.action_count = 1
    state = _lifecycle_runtime_state(decision_state, max_actions=1)
    state["actionHistory"] = ["compare_groups"]

    _refresh_action_availability(state)

    assert state["decisionState"].progress.remaining_objectives == []
    assert state["decisionState"].action_space.available_actions == ["finish"]


def test_terminal_finish_turn_budget_semantics() -> None:
    decision_state = _state(
        objectives=["cross_project_validation"],
        include_project=True,
    ).model_copy(deep=True)
    decision_state.progress.action_count = 1
    decision_state.action_space.available_actions = ["finish"]
    state = _lifecycle_runtime_state(decision_state, max_actions=1)
    state["actionHistory"] = ["compare_groups"]
    state["dynamicMaterialization"] = True

    continued = _decide_continue_or_stop(state)
    assert continued["route"] == "plan"
    assert continued["terminalFinishTurnGranted"] is True

    class FinishPlanner:
        dynamic_action_materialization = True
        prefer_policy_action = True

        def plan_action_with_state(self, _context, _policy_input):
            return ScientificPlannerResult(
                action=FinishAction(
                    actionId="action-" + "1" * 32,
                    actionName="finish",
                    rationale="the resolved state permits a terminal decision",
                    arguments=FinishArguments(
                        actionName="finish",
                        reasonCode="EVIDENCE_SUFFICIENT",
                    ),
                ),
                mode="model",
                policyOrigin="qwen_model",
            )

    planned = _plan_action(FinishPlanner(), continued)
    assert planned.get("errorCode") is None
    assert planned["currentAction"].actionName == "finish"
    assert "terminalFinishTurnGranted" not in planned


def _blocked_project_resolution() -> dict[str, object]:
    return {
        "resolutions": [
            {
                "objective": "group_comparison",
                "status": "completed",
                "reason_code": None,
                "reason": None,
                "action": "compare_groups",
                "resolved_at_step": 4,
                "limitation_code": None,
            },
            {
                "objective": "cross_project_validation",
                "status": "blocked",
                "reason_code": "PROJECT_DIMENSION_UNAVAILABLE",
                "reason": "no valid project dimension",
                "action": "cross_project_validate",
                "resolved_at_step": 5,
                "limitation_code": "objective_unavailable_due_to_data",
            },
        ],
        "active_objectives": [],
        "completed_objectives": ["group_comparison"],
        "blocked_objectives": ["cross_project_validation"],
        "workflow_can_finish": True,
        "all_requested_objectives_completed": False,
        "limitations_present": True,
    }


def _finish_action() -> FinishAction:
    return FinishAction(
        actionId="action-" + "f" * 32,
        actionName="finish",
        rationale="close the resolved workflow",
        arguments=FinishArguments(
            actionName="finish",
            reasonCode="EVIDENCE_SUFFICIENT",
        ),
    )


def test_blocked_objective_survives_finish() -> None:
    decision_state = _state(
        objectives=["group_comparison", "cross_project_validation"],
        include_project=True,
    ).model_copy(deep=True)
    decision_state.progress.remaining_objectives = []
    decision_state.action_space.available_actions = ["finish"]
    runtime_state = {
        "decisionState": decision_state,
        "objectiveResolution": _blocked_project_resolution(),
        "currentAction": _finish_action(),
        "actionHistory": [],
    }

    finished = _execute_action(None, None, None, runtime_state)

    assert finished["decisionState"].progress.remaining_objectives == []
    assert finished["objectiveResolution"]["blocked_objectives"] == [
        "cross_project_validation"
    ]
    assert finished["objectiveResolution"]["completed_objectives"] == [
        "group_comparison"
    ]


def test_completed_objective_survives_finish() -> None:
    resolution = _blocked_project_resolution()
    decision_state = _state(
        objectives=["group_comparison", "cross_project_validation"],
        include_project=True,
    ).model_copy(deep=True)
    decision_state.progress.remaining_objectives = []
    runtime_state = {
        "decisionState": decision_state,
        "objectiveResolution": resolution,
        "currentAction": _finish_action(),
        "actionHistory": [],
    }

    finished = _execute_action(None, None, None, runtime_state)

    assert finished["objectiveResolution"]["resolutions"][0]["status"] == "completed"
    assert finished["objectiveResolution"]["resolutions"][1]["status"] == "blocked"


def test_finish_does_not_reinitialize_remaining_objectives() -> None:
    decision_state = _state(
        objectives=["cross_project_validation"],
        include_project=True,
    ).model_copy(deep=True)
    decision_state.progress.remaining_objectives = []
    runtime_state = {
        "decisionState": decision_state,
        "objectiveResolution": _blocked_project_resolution(),
        "currentAction": _finish_action(),
        "actionHistory": [],
    }

    finished = _execute_action(None, None, None, runtime_state)

    assert finished["decisionState"].progress.remaining_objectives == []


def test_objective_resolution_is_serialized_to_trace() -> None:
    record = TraceDecision(
        observationStateCode="OBSERVATION_AVAILABLE",
        allowedActions=["finish"],
        chosenAction="finish",
        selected_action="finish",
        stop_reason="EVIDENCE_SUFFICIENT",
        objective_resolution=_blocked_project_resolution(),
    )

    serialized = record.model_dump(mode="json")

    assert serialized["objective_resolution"]["blocked_objectives"] == [
        "cross_project_validation"
    ]
    assert serialized["objective_resolution"]["resolutions"][1]["resolved_at_step"] == 5


def test_remaining_objectives_equals_active_objectives() -> None:
    decision_state = _state(
        objectives=["cross_project_validation", "evidence_support"],
        include_project=False,
    )
    runtime_state = {
        "request": _runtime_request_for_lifecycle(),
        "decisionState": decision_state,
        "schemaCatalog": _catalog(include_project=False),
        "knowledgeAvailable": True,
        "rawObservationPayloads": {},
        "observations": [],
    }

    _refresh_action_availability(runtime_state)

    assert runtime_state["decisionState"].progress.remaining_objectives == runtime_state[
        "objectiveResolution"
    ]["active_objectives"]


def test_final_completion_semantics_preserve_limitation() -> None:
    semantics = completion_semantics_from_resolution(
        _blocked_project_resolution(),
        workflow_completed=True,
        remaining_objectives=[],
    )

    assert semantics["workflow_completed"] is True
    assert semantics["all_requested_objectives_completed"] is False
    assert semantics["limitations_present"] is True
    assert semantics["blocked_objectives"] == ["cross_project_validation"]
    assert semantics["limitation_code"] == "objective_unavailable_due_to_data"
    assert semantics["remaining_objectives_invariant"] is True


def test_environment_blocked_objective_gets_environment_limitation_code() -> None:
    assert limitation_code_for_resolution(
        "blocked",
        reason_code="KNOWLEDGE_SOURCE_UNAVAILABLE",
    ) == "objective_unavailable_due_to_environment"
    assert limitation_code_for_resolution(
        "blocked",
        reason_code="EMBEDDING_CONFIGURATION_ERROR",
    ) == "objective_unavailable_due_to_environment"
    assert limitation_code_for_resolution(
        "blocked",
        reason_code="PROJECT_DIMENSION_UNAVAILABLE",
    ) == "objective_unavailable_due_to_data"


def test_completion_projection_corrects_stale_environment_limitation_code() -> None:
    resolution = {
        "resolutions": [
            {
                "objective": "cross_project_validation",
                "status": "blocked",
                "reason_code": "PROJECT_DIMENSION_UNAVAILABLE",
                "limitation_code": "objective_unavailable_due_to_data",
            },
            {
                "objective": "evidence_support",
                "status": "blocked",
                "reason_code": "KNOWLEDGE_SOURCE_UNAVAILABLE",
                # Historical traces used the data code for every block.
                "limitation_code": "objective_unavailable_due_to_data",
            },
        ],
        "active_objectives": [],
        "completed_objectives": [],
        "blocked_objectives": [
            "cross_project_validation",
            "evidence_support",
        ],
        "all_requested_objectives_completed": False,
        "limitations_present": True,
    }

    assert limitation_codes_from_resolution(resolution) == [
        "objective_unavailable_due_to_data",
        "objective_unavailable_due_to_environment",
    ]
    semantics = completion_semantics_from_resolution(
        resolution,
        workflow_completed=True,
        remaining_objectives=[],
    )
    assert semantics["limitation_codes"] == [
        "objective_unavailable_due_to_data",
        "objective_unavailable_due_to_environment",
    ]
