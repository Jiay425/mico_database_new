from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx
import pytest

from mico_agent_runtime.contracts.decision_state import (
    ScientificActionSpaceState,
    ScientificDecisionState,
    ScientificTaskConstraints,
    ScientificTaskState,
)
from mico_agent_runtime.contracts.materialization import QueryPlan
from mico_agent_runtime.contracts.research import (
    ExecuteReadQueryAction,
    ExecuteReadQueryArguments,
    Observation,
    ResearchTask,
    ScientificPlannerContext,
)
from mico_agent_runtime.contracts.schema_catalog import (
    SchemaEntitySemantics,
    SchemaFieldSemantics,
    SchemaJoinSemantics,
    SchemaSemanticCatalog,
)
from mico_agent_runtime.graph.scientific_workflow import (
    _action_signature,
    _authorize_action,
    _execute_read_query_action,
    _plan_action,
    _update_state,
)
from mico_agent_runtime.ports.research_planner import HttpResearchPlannerPort
from mico_agent_runtime.ports.research_planner import ScientificPlannerResult
from mico_agent_runtime.runtime.data_requirements import (
    DataRequirementError,
    augment_query_plan_with_fields,
    derive_data_requirements,
    prune_unavailable_query_plan,
    record_zero_coverage_capabilities,
)
from tests.conftest import completed_response


NOW = datetime(2026, 9, 3, tzinfo=timezone.utc)


def _catalog(
    *,
    include_age: bool = True,
    include_gender: bool = True,
    include_project: bool = True,
) -> SchemaSemanticCatalog:
    sample_fields = [
        SchemaFieldSemantics(
            fieldId="sample.disease", name="disease", dataType="string",
            nullable=True, semanticStatus="verified", filterable=True,
            groupable=True, displayable=True,
            scientificCapabilities=["dimension", "stratifier"],
            description="Disease dimension",
        ),
    ]
    if include_age:
        sample_fields.append(SchemaFieldSemantics(
            fieldId="sample.age", name="age", dataType="integer",
            nullable=True, semanticStatus="verified", filterable=True,
            groupable=True, aggregatable=True, displayable=True,
            scientificCapabilities=["covariate", "stratifier"],
            description="Age covariate",
        ))
    if include_gender:
        sample_fields.append(SchemaFieldSemantics(
            fieldId="sample.gender", name="gender", dataType="string",
            nullable=True, semanticStatus="verified", filterable=True,
            groupable=True, aggregatable=True, displayable=True,
            scientificCapabilities=["covariate", "dimension", "stratifier"],
            description="Gender covariate",
        ))
    entities = [
        SchemaEntitySemantics(
            entityId="sample", entityName="patient_record", sourceTable="patients",
            fields=sample_fields,
        ),
        SchemaEntitySemantics(
            entityId="abundance", entityName="standard_abundance",
            sourceTable="microbe_abundance_standard", fields=[
                SchemaFieldSemantics(
                    fieldId="abundance.feature", name="feature", dataType="string",
                    nullable=True, semanticStatus="verified", groupable=True,
                    displayable=True, scientificCapabilities=["dimension", "stratifier"],
                    description="Feature dimension",
                ),
                SchemaFieldSemantics(
                    fieldId="abundance.value", name="value", dataType="number",
                    nullable=True, semanticStatus="verified", aggregatable=True,
                    displayable=True, scientificCapabilities=["outcome"],
                    description="Numeric outcome",
                ),
            ],
        ),
    ]
    joins = [SchemaJoinSemantics(
        relationId="sample_to_abundance", leftEntity="patient_record",
        leftField="patient_id", rightEntity="standard_abundance",
        rightField="patient_id", relationshipStatus="verified",
        cardinality="one_to_many", description="sample abundance",
    )]
    if include_project:
        entities.append(SchemaEntitySemantics(
            entityId="metadata", entityName="sample_metadata",
            sourceTable="meta2db_sample_metadata", fields=[
                SchemaFieldSemantics(
                    fieldId="metadata.project", name="project_name", dataType="string",
                    nullable=True, semanticStatus="verified", groupable=True,
                    displayable=True, scientificCapabilities=["dimension"],
                    description="Project dimension",
                ),
            ],
        ))
        joins.append(SchemaJoinSemantics(
            relationId="sample_to_metadata", leftEntity="patient_record",
            leftField="patient_id", rightEntity="sample_metadata",
            rightField="patient_id", relationshipStatus="verified",
            cardinality="one_to_many", description="sample metadata",
        ))
    return SchemaSemanticCatalog(
        schemaVersion="schema-catalog-v1", source="java_schema_contract",
        generatedAt=NOW, entities=entities, joins=joins,
        internalAnalysisFields=["analysis.sample_key"],
        queryRules=["select_or_with_only", "explicit_columns_only", "bounded_limit_required"],
    )


def _state(*objectives: str, focus: list[str] | None = None, stratifiers: list[str] | None = None) -> ScientificDecisionState:
    return ScientificDecisionState(task=ScientificTaskState(
        query="比较 T2D 和 Healthy，并判断年龄影响",
        objectives=list(objectives),
        constraints=ScientificTaskConstraints(
            disease_groups=["T2D", "Healthy"],
            focus_covariates=focus or [],
            requested_stratifiers=stratifiers or [],
        ),
    ))


def test_confounder_objective_propagates_short_age_to_catalog_id() -> None:
    requirements = derive_data_requirements(
        _state("group_comparison", "confounder_assessment", focus=["age"]),
        _catalog(),
    )

    assert "sample.age" in requirements.required_fields
    assert "sample.disease" in requirements.required_fields
    assert "abundance.value" in requirements.required_fields
    assert "abundance.feature" in requirements.required_fields
    assert requirements.missing_fields == ()
    assert requirements.blocked_objectives == ()


def test_missing_requested_covariate_is_blocked_without_fabrication() -> None:
    requirements = derive_data_requirements(
        _state("group_comparison", "confounder_assessment", focus=["age"]),
        _catalog(include_age=False),
    )

    assert "age" in requirements.missing_fields
    assert "sample.age" not in requirements.required_fields
    assert "confounder_assessment" in requirements.blocked_objectives


def test_confounder_objective_without_catalog_covariate_is_blocked() -> None:
    requirements = derive_data_requirements(
        _state("group_comparison", "confounder_assessment"),
        _catalog(include_age=False, include_gender=False),
    )

    assert "confounder_assessment" in requirements.blocked_objectives
    assert all(field not in requirements.required_fields for field in (
        "sample.age", "sample.gender",
    ))


def test_stratifier_requirement_is_catalog_backed() -> None:
    requirements = derive_data_requirements(
        _state("group_comparison", "stratified_analysis", stratifiers=["age"]),
        _catalog(),
    )

    assert "sample.age" in requirements.required_fields
    assert requirements.missing_fields == ()


def test_ordinary_comparison_does_not_pull_all_covariates_or_project() -> None:
    requirements = derive_data_requirements(
        _state("group_comparison"),
        _catalog(),
    )

    assert requirements.required_fields == (
        "sample.disease", "abundance.value", "abundance.feature",
    )
    assert "sample.age" not in requirements.required_fields
    assert "sample.gender" not in requirements.required_fields
    assert "metadata.project" not in requirements.required_fields


def test_query_plan_augmentation_preserves_raw_shape_and_adds_age() -> None:
    catalog = _catalog()
    plan = QueryPlan(
        root_entity="sample",
        relation_path=["sample_to_abundance"],
        select_fields=["sample.disease", "abundance.feature", "abundance.value"],
        limit=500,
    )

    augmented = augment_query_plan_with_fields(plan, ["sample.age"], catalog)

    assert augmented.select_fields == [
        "sample.disease", "abundance.feature", "abundance.value", "sample.age",
    ]
    assert augmented.relation_path == ["sample_to_abundance"]
    assert augmented.aggregations == []
    assert augmented.group_by == []
    assert augmented.limit == 500


def test_query_plan_augmentation_connects_a_catalog_project_relation() -> None:
    catalog = _catalog()
    plan = QueryPlan(root_entity="sample", select_fields=["sample.disease"], limit=100)

    augmented = augment_query_plan_with_fields(plan, ["metadata.project"], catalog)

    assert "metadata.project" in augmented.select_fields
    assert augmented.relation_path == ["sample_to_metadata"]


def test_http_materializer_binds_required_age_before_java_validation() -> None:
    catalog = _catalog()
    body = {
        "actionName": "execute_read_query",
        "rationale": "read the bounded observation",
        "arguments": {
            "actionName": "execute_read_query",
            "queryPlan": {
                "root_entity": "sample",
                "relation_path": ["sample_to_abundance"],
                "select_fields": ["sample.disease", "abundance.feature", "abundance.value"],
                "aggregations": [], "filters": [], "group_by": [], "limit": 500,
            },
            "limit": 500,
        },
    }

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps(body)}}],
        })

    planner = HttpResearchPlannerPort(
        "http://materializer.local/v1", "gemini-3.5-flash-lite", "test-token",
        transport=httpx.MockTransport(handler),
    )
    try:
        result = planner.plan_action(ScientificPlannerContext(
            questionSummary="比较疾病组并评估年龄混杂",
            intent="scientific_exploration",
            approvedActions=["execute_read_query"],
            remainingActionBudget=4,
            requiredSemanticFields=[
                "sample.disease", "abundance.feature", "abundance.value", "sample.age",
            ],
            schemaCatalog=catalog,
        ))
    finally:
        planner.close()

    query_plan = result.action.arguments.queryPlan
    assert query_plan is not None
    assert "sample.age" in query_plan.select_fields
    assert "RUNTIME_DATA_REQUIREMENT_FIELDS_BOUND" in result.repairCodes


def test_runtime_read_call_contains_age_and_keeps_opaque_sample_bound() -> None:
    catalog = _catalog(include_project=False)
    decision_state = _state(
        "group_comparison", "confounder_assessment", focus=["age"],
    )
    task = ResearchTask(
        runId="run-" + "a" * 32,
        taskId="task-" + "b" * 32,
        requesterId="principal-" + "c" * 32,
        traceId="trace-" + "d" * 32,
        question=decision_state.task.query,
        requestedScopes=["mico:query:read", "mico:research:read"],
        allowedActions=["execute_read_query", "compare_groups", "adjust_confounders"],
        maxActions=6,
        createdAt=NOW,
    )
    plan = QueryPlan(
        root_entity="sample",
        relation_path=["sample_to_abundance"],
        select_fields=["sample.disease", "abundance.feature", "abundance.value"],
        limit=500,
    )
    action = ExecuteReadQueryAction(
        actionId="action-" + "e" * 32,
        actionName="execute_read_query",
        rationale="read the bounded sample projection",
        arguments=ExecuteReadQueryArguments(
            actionName="execute_read_query", queryPlan=plan, limit=500,
        ),
    )

    class Java:
        def __init__(self) -> None:
            self.call = None

        def execute(self, call):
            self.call = call
            response = completed_response(call)
            snapshot = response.dataSnapshot.model_copy(update={"rowCount": 2})
            return response.model_copy(update={
                "rowCount": 2,
                "dataSnapshot": snapshot,
                "data": {
                    "columns": [
                        "a_sample_disease", "a_sample_age", "a_abundance_feature",
                        "a_abundance_value", "a_analysis_sample_key",
                    ],
                    "rows": [
                        {
                            "a_sample_disease": "T2D", "a_sample_age": 60,
                            "a_abundance_feature": "f1", "a_abundance_value": 0.2,
                            "a_analysis_sample_key": "opaque-1",
                        },
                        {
                            "a_sample_disease": "Healthy", "a_sample_age": 40,
                            "a_abundance_feature": "f1", "a_abundance_value": 0.1,
                            "a_analysis_sample_key": "opaque-2",
                        },
                    ],
                },
            })

    java = Java()
    runtime_state = {
        "request": task,
        "decisionState": decision_state,
        "schemaCatalog": catalog,
        "dynamicMaterialization": True,
        "plannerFeedback": [],
        "fallbackCodes": [],
        "decisionRecords": [],
        "observations": [],
        "rawObservationPayloads": {},
    }

    class Planner:
        dynamic_action_materialization = True

    result = _execute_read_query_action(runtime_state, action, Planner(), java)

    assert result.get("errorCode") is None
    assert java.call is not None
    sent_plan = java.call.arguments.queryPlan
    assert sent_plan is not None
    assert "sample.age" in sent_plan.select_fields
    assert sent_plan.sample_limit_per_group == 50
    assert sent_plan.sample_limit_group_field == "sample.disease"
    assert java.call.arguments.includeAnalysisSampleKey is True
    observation = result["pendingObservation"]
    assert "sample.age" in observation.queryPlanFields
    assert "a_sample_patient_id" not in result["pendingPayload"]["columns"]


def test_graph_plan_action_propagates_objective_requirements_to_materializer_context() -> None:
    catalog = _catalog(include_project=False)
    decision_state = _state(
        "group_comparison", "confounder_assessment", focus=["age"],
    ).model_copy(update={
        "action_space": ScientificActionSpaceState(
            available_actions=["execute_read_query"],
        ),
    })
    task = ResearchTask(
        runId="run-" + "f" * 32,
        taskId="task-" + "g" * 32,
        requesterId="principal-" + "h" * 32,
        traceId="trace-" + "i" * 32,
        question=decision_state.task.query,
        requestedScopes=["mico:query:read"],
        allowedActions=["execute_read_query"],
        maxActions=6,
        createdAt=NOW,
    )
    plan = QueryPlan(
        root_entity="sample",
        relation_path=["sample_to_abundance"],
        select_fields=["sample.disease", "abundance.feature", "abundance.value"],
        limit=500,
    )

    class Planner:
        dynamic_action_materialization = True
        prefer_policy_action = True

        def __init__(self) -> None:
            self.context = None

        def plan_action_with_state(self, context, _policy_input):
            self.context = context
            return ScientificPlannerResult(
                action=ExecuteReadQueryAction(
                    actionId="action-" + "a" * 32,
                    actionName="execute_read_query",
                    rationale="read the bounded projection",
                    arguments=ExecuteReadQueryArguments(
                        actionName="execute_read_query",
                        queryPlan=plan,
                        limit=plan.limit,
                    ),
                ),
                mode="model",
                policyOrigin="qwen_model",
                materializerOrigin="gemini_canary_model",
            )

    planner = Planner()
    result = _plan_action(
        planner,
        {
            "request": task,
            "decisionState": decision_state,
            "schemaCatalog": catalog,
            "observations": [],
            "actionHistory": [],
            "fallbackCodes": [],
            "decisionRecords": [],
            "plannerFeedback": [],
        },
    )

    assert result.get("errorCode") is None, result.get("plannerFailureDetail")
    assert planner.context is not None
    assert planner.context.requiredSemanticFields == [
        "sample.disease",
        "abundance.value",
        "abundance.feature",
        "sample.age",
    ]
    # Raw reads remain ungrouped; group_field is a required projection, not a
    # grouped QueryPlan instruction.
    assert planner.context.requiredGroupField is None
    assert result["dataRequirements"]["required_fields"][-1] == "sample.age"


def _dynamic_task_and_state(catalog: SchemaSemanticCatalog):
    decision_state = _state(
        "group_comparison", "confounder_assessment", "cross_project_validation",
        focus=["age"],
    ).model_copy(update={
        "action_space": ScientificActionSpaceState(
            available_actions=["execute_read_query", "compare_groups", "adjust_confounders", "finish"],
        ),
    })
    task = ResearchTask(
        runId="run-" + "1" * 32,
        taskId="task-" + "2" * 32,
        requesterId="principal-" + "3" * 32,
        traceId="trace-" + "4" * 32,
        question=decision_state.task.query,
        requestedScopes=["mico:query:read", "mico:research:read"],
        allowedActions=["execute_read_query", "compare_groups", "adjust_confounders", "finish"],
        maxActions=6,
        createdAt=NOW,
    )
    state = {
        "request": task,
        "decisionState": decision_state,
        "schemaCatalog": catalog,
        "dynamicMaterialization": True,
        "plannerFeedback": [],
        "fallbackCodes": [],
        "decisionRecords": [],
        "observations": [],
        "rawObservationPayloads": {},
        "actionHistory": [],
        "actionSignatures": [],
        "objectiveResolution": {},
        "knowledgeAvailable": True,
        "unavailableFields": [],
        "unavailableDataCapabilities": {},
    }
    state["dataRequirements"] = derive_data_requirements(
        decision_state, catalog,
    ).model_dump()
    return task, decision_state, state


def _project_join_plan() -> QueryPlan:
    return QueryPlan(
        root_entity="sample",
        relation_path=["sample_to_abundance", "sample_to_metadata"],
        select_fields=[
            "sample.disease", "sample.age", "abundance.feature",
            "abundance.value", "metadata.project",
        ],
        filters=[],
        aggregations=[],
        group_by=[],
        limit=20_000,
    )


def _read_action(plan: QueryPlan, suffix: str = "a") -> ExecuteReadQueryAction:
    return ExecuteReadQueryAction(
        actionId="action-" + suffix * 32,
        actionName="execute_read_query",
        rationale="bounded read",
        arguments=ExecuteReadQueryArguments(
            actionName="execute_read_query", queryPlan=plan, limit=plan.limit,
        ),
    )


def _zero_join_response(call):
    response = completed_response(call)
    snapshot = response.dataSnapshot.model_copy(update={"rowCount": 0})
    return response.model_copy(update={
        "rowCount": 0,
        "dataSnapshot": snapshot,
        "data": {
            "columns": [
                "a_sample_disease", "a_sample_age", "a_abundance_feature",
                "a_abundance_value", "a_metadata_project", "a_analysis_sample_key",
            ],
            "rows": [],
        },
    })


def test_zero_coverage_dimension_not_reinjected_into_query() -> None:
    catalog = _catalog()
    _task, decision_state, state = _dynamic_task_and_state(catalog)
    observation = Observation(
        observationId="observation-" + "a" * 32,
        actionId="action-" + "b" * 32,
        actionName="execute_read_query",
        status="VALIDATED",
        source="java_controlled_read",
        queryHash="sha256:" + "c" * 64,
        rowCount=0,
        generatedAt=NOW,
        schemaVersion="java-read-model-v1",
        dataSnapshotId="transient-550e8400-e29b-41d4-a716-446655440000",
        snapshotPersistence="transient",
        queryPlanFields=_project_join_plan().select_fields,
        queryPlanRelationPath=_project_join_plan().relation_path,
    )
    fields = record_zero_coverage_capabilities(
        state, observation, _project_join_plan(), catalog,
    )
    assert fields == ("metadata.project",)
    requirements = derive_data_requirements(
        decision_state,
        catalog,
        unavailable_fields=state["unavailableFields"],
    )
    assert "metadata.project" not in requirements.required_fields
    assert "cross_project_validation" in requirements.blocked_objectives
    assert state["unavailableDataCapabilities"]["metadata.project"]["observed_coverage"] == 0


def test_optional_unavailable_dimension_is_pruned_before_java() -> None:
    catalog = _catalog()
    _task, _decision_state, state = _dynamic_task_and_state(catalog)
    state["unavailableFields"] = ["metadata.project"]
    state["unavailableDataCapabilities"] = {
        "metadata.project": {"reason_code": "ZERO_COVERAGE_JOIN", "observed_coverage": 0},
    }

    class Java:
        def __init__(self) -> None:
            self.call = None

        def execute(self, call):
            self.call = call
            response = completed_response(call)
            snapshot = response.dataSnapshot.model_copy(update={"rowCount": 1})
            return response.model_copy(update={
                "rowCount": 1,
                "dataSnapshot": snapshot,
                "data": {
                    "columns": [
                        "a_sample_disease", "a_sample_age", "a_abundance_feature",
                        "a_abundance_value", "a_analysis_sample_key",
                    ],
                    "rows": [{
                        "a_sample_disease": "T2D", "a_sample_age": 60,
                        "a_abundance_feature": "f1", "a_abundance_value": 0.2,
                        "a_analysis_sample_key": "opaque-1",
                    }],
                },
            })

    java = Java()
    result = _execute_read_query_action(
        state, _read_action(_project_join_plan(), "d"),
        type("Planner", (), {"dynamic_action_materialization": True})(), java,
    )
    assert result.get("errorCode") is None
    sent = java.call.arguments.queryPlan
    assert sent is not None
    assert "metadata.project" not in sent.select_fields
    assert sent.relation_path == ["sample_to_abundance"]


def test_required_unavailable_dimension_fails_closed() -> None:
    catalog = _catalog()
    plan = QueryPlan(
        root_entity="sample",
        relation_path=["sample_to_metadata"],
        select_fields=["metadata.project"],
        limit=100,
    )
    with pytest.raises(DataRequirementError, match="unavailable query capability|only unavailable"):
        prune_unavailable_query_plan(plan, ["metadata.project"], catalog)


def test_project_blocking_and_query_capability_are_consistent() -> None:
    catalog = _catalog()
    _task, _decision_state, state = _dynamic_task_and_state(catalog)
    state["currentAction"] = _read_action(_project_join_plan(), "e")
    state["pendingObservation"] = Observation(
        observationId="observation-" + "d" * 32,
        actionId=state["currentAction"].actionId,
        actionName="execute_read_query",
        status="VALIDATED",
        source="java_controlled_read",
        queryHash="sha256:" + "e" * 64,
        rowCount=0,
        generatedAt=NOW,
        schemaVersion="java-read-model-v1",
        dataSnapshotId="transient-550e8400-e29b-41d4-a716-446655440000",
        snapshotPersistence="transient",
        queryPlanFields=_project_join_plan().select_fields,
        queryPlanRelationPath=_project_join_plan().relation_path,
    )
    state["pendingPayload"] = {
        "columns": [
            "a_sample_disease", "a_sample_age", "a_abundance_feature",
            "a_abundance_value", "a_metadata_project", "a_analysis_sample_key",
        ],
        "rows": [],
    }
    state["pendingResponse"] = type("Response", (), {"rowCount": 0})()
    updated = _update_state(state)
    assert updated["decisionState"].data_state.project_state.has_project_field is False
    assert updated["objectiveResolution"]["blocked_objectives"] == ["cross_project_validation"]
    assert "metadata.project" in updated["unavailableFields"]


def test_materially_different_read_plan_not_treated_as_repeat() -> None:
    catalog = _catalog(include_project=False)
    _task, _decision_state, state = _dynamic_task_and_state(catalog)
    first = _read_action(QueryPlan(
        root_entity="sample", relation_path=["sample_to_abundance"],
        select_fields=["sample.disease", "abundance.feature", "abundance.value"],
        limit=100,
    ), "f")
    second = _read_action(QueryPlan(
        root_entity="sample", relation_path=["sample_to_abundance"],
        select_fields=["sample.disease", "sample.age", "abundance.feature", "abundance.value"],
        limit=100,
    ), "a")
    state["actionSignatures"] = [_action_signature(first)]
    state["currentAction"] = second
    state["plannerFeedback"] = ["DYNAMIC_EMPTY_QUERY_REPLAN"]
    authorized = _authorize_action(state)
    assert authorized["currentAction"] == second
    assert authorized["actionSignatures"][-1] == _action_signature(second)


def test_identical_read_plan_still_rejected() -> None:
    catalog = _catalog(include_project=False)
    _task, _decision_state, state = _dynamic_task_and_state(catalog)
    action = _read_action(QueryPlan(
        root_entity="sample", relation_path=["sample_to_abundance"],
        select_fields=["sample.disease", "abundance.feature", "abundance.value"],
        limit=100,
    ), "b")
    state["actionSignatures"] = [_action_signature(action)]
    state["currentAction"] = action
    state["plannerFeedback"] = ["DYNAMIC_EMPTY_QUERY_REPLAN"]
    state["readReplanCount"] = 3
    rejected = _authorize_action(state)
    assert rejected["errorCode"] == "READ_PLAN_REPEAT_REJECTED"
