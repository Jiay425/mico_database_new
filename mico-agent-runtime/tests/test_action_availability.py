from datetime import datetime, timezone

from mico_agent_runtime.contracts.decision_state import (
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
from mico_agent_runtime.contracts.research import ResearchTask
from mico_agent_runtime.graph.scientific_workflow import _understand_task, _validate_task
from mico_agent_runtime.ports.task_understanding import DeterministicTaskUnderstandingPort
from mico_agent_runtime.runtime.action_availability import (
    ActionAvailabilityContext,
    compute_available_actions,
    evaluate_action_availability,
)


def _catalog() -> SchemaSemanticCatalog:
    return SchemaSemanticCatalog(
        schemaVersion="schema-catalog-v1",
        source="java_schema_contract",
        generatedAt=datetime(2026, 8, 31, tzinfo=timezone.utc),
        entities=[
            SchemaEntitySemantics(
                entityName="sample",
                sourceTable="sample",
                fields=[
                    SchemaFieldSemantics(
                        fieldId="sample.disease",
                        name="disease",
                        dataType="string",
                        nullable=False,
                        semanticStatus="verified",
                        groupable=True,
                        scientificCapabilities=["dimension"],
                        description="Disease group",
                    ),
                    SchemaFieldSemantics(
                        fieldId="sample.age",
                        name="age",
                        dataType="integer",
                        nullable=True,
                        semanticStatus="verified",
                        groupable=True,
                        scientificCapabilities=["covariate", "stratifier"],
                        description="Age covariate",
                    ),
                    SchemaFieldSemantics(
                        fieldId="sample.sex",
                        name="sex",
                        dataType="string",
                        nullable=True,
                        semanticStatus="verified",
                        groupable=True,
                        scientificCapabilities=["covariate", "stratifier"],
                        description="Sex covariate",
                    ),
                ],
            ),
            SchemaEntitySemantics(
                entityName="metadata",
                sourceTable="metadata",
                fields=[
                    SchemaFieldSemantics(
                        fieldId="metadata.project",
                        name="project",
                        dataType="string",
                        nullable=False,
                        semanticStatus="verified",
                        groupable=True,
                        scientificCapabilities=["dimension"],
                        description="Project dimension",
                    ),
                ],
            ),
            SchemaEntitySemantics(
                entityName="abundance",
                sourceTable="abundance",
                fields=[
                    SchemaFieldSemantics(
                        fieldId="abundance.value",
                        name="value",
                        dataType="number",
                        nullable=True,
                        semanticStatus="verified",
                        aggregatable=True,
                        scientificCapabilities=["outcome"],
                        description="Numeric abundance outcome",
                    ),
                ],
            ),
        ],
        queryRules=["select_or_with_only", "explicit_columns_only", "bounded_limit_required"],
    )


def _state(
    *,
    project_count: int = 7,
    covariates: list[str] | None = None,
    objectives: list[str] | None = None,
    group_status: str = "not_started",
) -> ScientificDecisionState:
    return ScientificDecisionState(
        task=ScientificTaskState(
            query="compare disease groups",
            objectives=objectives or ["group_comparison"],
        ),
        data_state=ScientificDataState(
            has_tabular_data=True,
            row_count=326,
            available_dimensions=[
                "sample.disease",
                "sample.age",
                "sample.sex",
                "metadata.project",
            ],
            available_outcomes=["abundance.value"],
            group_state={
                "group_field": "sample.disease",
                "group_count": 2,
                "group_sizes": {"T2D": 172, "Healthy": 154},
            },
            project_state={"has_project_field": True, "project_count": project_count},
            covariate_state={
                "available_covariates": covariates if covariates is not None else [
                    "sample.age", "sample.sex"
                ],
                "imbalance": {"sample.age": "high"},
            },
        ),
        analysis_state=ScientificAnalysisState(
            group_comparison=GroupComparisonState(status=group_status),
        ),
    )


def _context(**updates: object) -> ActionAvailabilityContext:
    payload: dict[str, object] = {
        "catalog": _catalog(),
        "knowledge_available": True,
        "max_actions": 6,
        "action_count": 0,
    }
    payload.update(updates)
    return ActionAvailabilityContext(**payload)


def test_complete_observed_state_keeps_all_independent_actions_available() -> None:
    available = set(compute_available_actions(_state(), _context()))
    assert {
        "compare_groups",
        "adjust_confounders",
        "stratified_analysis",
        "cross_project_validate",
        "retrieve_evidence",
    }.issubset(available)


def test_completed_comparison_does_not_filter_auxiliary_actions() -> None:
    available = set(compute_available_actions(
        _state(group_status="completed"),
        _context(),
    ))
    assert {
        "adjust_confounders",
        "stratified_analysis",
        "cross_project_validate",
        "retrieve_evidence",
    }.issubset(available)


def test_one_project_blocks_cross_project_but_not_read_query() -> None:
    result = evaluate_action_availability(_state(project_count=1), _context())
    assert "cross_project_validate" not in result.available_actions
    assert result.blocked_reasons["cross_project_validate"] == "INSUFFICIENT_PROJECT_COUNT"
    assert "execute_read_query" in result.available_actions


def test_no_covariates_blocks_adjustment_but_keeps_cross_project() -> None:
    available = set(compute_available_actions(_state(covariates=[]), _context()))
    assert "adjust_confounders" not in available
    assert "cross_project_validate" in available


def test_project_count_without_observed_project_field_blocks_cross_project() -> None:
    state = _state().model_copy(deep=True)
    state.data_state.project_state.has_project_field = False
    result = evaluate_action_availability(state, _context())
    assert "cross_project_validate" not in result.available_actions
    assert result.blocked_reasons["cross_project_validate"] == "PROJECT_DIMENSION_NOT_IN_OBSERVATION"


def test_project_dimension_requires_verified_catalog_dimension_capability() -> None:
    catalog = _catalog().model_copy(deep=True)
    metadata = next(entity for entity in catalog.entities if entity.entityName == "metadata")
    project = next(field for field in metadata.fields if field.fieldId == "metadata.project")
    project.semanticStatus = "unverified"
    result = evaluate_action_availability(
        _state(),
        _context(catalog=catalog),
    )
    assert "cross_project_validate" not in result.available_actions
    assert result.blocked_reasons["cross_project_validate"] == "PROJECT_DIMENSION_NOT_IN_OBSERVATION"


def test_project_dimension_requires_catalog_proof_even_when_name_is_present() -> None:
    state = _state().model_copy(deep=True)
    state.data_state.available_dimensions = [
        value for value in state.data_state.available_dimensions
        if value != "metadata.project"
    ]
    result = evaluate_action_availability(state, _context())
    assert "cross_project_validate" not in result.available_actions
    assert result.blocked_reasons["cross_project_validate"] == "PROJECT_DIMENSION_NOT_IN_OBSERVATION"


def test_cross_disease_requires_an_independent_validation_dimension() -> None:
    state = _state(objectives=["cross_disease_validation"])
    result = evaluate_action_availability(state, _context())
    assert "cross_disease_validate" not in result.available_actions
    assert result.blocked_reasons["cross_disease_validate"] == "DISEASE_DIMENSION_UNAVAILABLE"


def test_finish_is_available_only_after_required_objective_is_completed() -> None:
    completed = compute_available_actions(
        _state(objectives=["group_comparison"], group_status="completed"),
        _context(),
    )
    pending = evaluate_action_availability(_state(), _context())
    assert "finish" in completed
    assert "finish" not in pending.available_actions
    assert pending.blocked_reasons["finish"] == "REQUIRED_OBJECTIVE_INCOMPLETE"
    assert "compare_groups" in pending.available_actions


def test_blocked_reasons_are_not_part_of_decision_state() -> None:
    state = _state(project_count=1)
    result = evaluate_action_availability(state, _context())
    assert "blocked_reasons" not in state.model_dump()
    assert result.blocked_reasons["cross_project_validate"] == "INSUFFICIENT_PROJECT_COUNT"


def test_missing_knowledge_port_only_blocks_retrieval() -> None:
    available = set(compute_available_actions(
        _state(),
        _context(knowledge_available=False),
    ))
    assert "retrieve_evidence" not in available
    assert "compare_groups" in available


def test_workflow_initialization_refreshes_hard_actions_without_data() -> None:
    request = ResearchTask(
        runId="run-availability",
        taskId="task-availability",
        requesterId="requester-availability",
        traceId="trace-availability",
        question="compare disease groups",
        requestedScopes=["mico:query:read", "mico:research:read"],
        allowedActions=["inspect_cohort", "execute_read_query", "retrieve_evidence", "finish"],
        maxActions=6,
        createdAt=datetime(2026, 8, 31, tzinfo=timezone.utc),
    )
    state = _validate_task(
        {"request": request, "schemaCatalog": _catalog()},
        knowledge_available=False,
    )
    state = _understand_task(DeterministicTaskUnderstandingPort(), state)
    assert state["decisionState"].action_space.available_actions == [
        "inspect_cohort", "execute_read_query", "finish",
    ]
    assert state["actionAvailabilityReasons"]["retrieve_evidence"] == "MISSING_REQUIRED_SCOPE"
