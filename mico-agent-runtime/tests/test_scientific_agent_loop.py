from __future__ import annotations

from datetime import datetime, timezone

from mico_agent_runtime.contracts.generated_analysis import (
    AnalysisPlannerContext,
    GeneratedAnalysisPlan,
    GeneratedAnalysisPlannerResult,
)
from mico_agent_runtime.contracts.graph_rag import EvidenceSynthesisResult, GroundedClaim
from mico_agent_runtime.contracts.research import (
    AnalyzeProjectionArguments,
    AnalyzeProjectionAction,
    CompareGroupsAction,
    ExecuteReadQueryArguments,
    ExecuteReadQueryAction,
    FinishAction,
    FinishArguments,
    InspectCohortAction,
    InspectCohortArguments,
    ProjectionAnalysisArguments,
    ResearchTask,
    Observation,
    ScientificObservationSummary,
    ScientificPlannerContext,
)
from mico_agent_runtime.contracts.unified_evidence import merge_unified_evidence
from mico_agent_runtime.contracts.tools import JavaToolError, JavaToolResponse
from mico_agent_runtime.contracts.schema_catalog import (
    SchemaEntitySemantics,
    SchemaFieldSemantics,
    SchemaSemanticCatalog,
)
from mico_agent_runtime.graph.scientific_workflow import (
    _plan_action,
    _safe_synthesis_result,
    _synthesis_context,
)
from mico_agent_runtime.ports.research_planner import ScientificPlannerResult
from mico_agent_runtime.ports.scientific_planner import (
    DeterministicScientificPlanner,
    semantic_scientific_next_action,
)
from mico_agent_runtime.ports.java_agent import JavaPortTransportError
from mico_agent_runtime.ports.schema_catalog import JavaSchemaCatalogPort
from mico_agent_runtime.runtime.scientific_service import ScientificRuntime
from tests.conftest import FakeJavaPort, completed_response
from tests.test_unified_evidence import _item, _path


NOW = datetime(2026, 8, 23, 12, 0, tzinfo=timezone.utc)


def _semantic_catalog() -> SchemaSemanticCatalog:
    return SchemaSemanticCatalog(
        schemaVersion="schema-catalog-v1",
        source="java_schema_contract",
        generatedAt=NOW,
        entities=[SchemaEntitySemantics(
            entityName="sample_metadata",
            sourceTable="meta2db_sample_metadata",
            fields=[
                SchemaFieldSemantics(
                    name="country", dataType="string", nullable=True,
                    semanticStatus="verified", filterable=True, groupable=True,
                    displayable=True, description="Cohort country label",
                ),
                SchemaFieldSemantics(
                    name="age", dataType="integer", nullable=True,
                    semanticStatus="verified", filterable=True, groupable=True,
                    displayable=True, description="Cohort age dimension",
                ),
                SchemaFieldSemantics(
                    name="gender", dataType="string", nullable=True,
                    semanticStatus="verified", filterable=True, groupable=True,
                    displayable=True, description="Cohort gender dimension",
                ),
                SchemaFieldSemantics(
                    name="project_name", dataType="string", nullable=True,
                    semanticStatus="verified", filterable=True, groupable=True,
                    displayable=True, description="Cohort project label",
                ),
                SchemaFieldSemantics(
                    name="disease", dataType="string", nullable=True,
                    semanticStatus="verified", filterable=True, groupable=True,
                    displayable=True, description="Cohort disease label",
                ),
            ],
        )],
        queryRules=["select_or_with_only", "explicit_columns_only", "bounded_limit_required"],
    )


def _semantic_observation(index: int, action_name: str, source: str = "java_controlled_read") -> ScientificObservationSummary:
    return ScientificObservationSummary(
        observationId="observation-" + f"{index:032x}",
        actionName=action_name,
        status="VALIDATED",
        source=source,
        rowCount=2,
    )


def _semantic_context(question: str, actions: list[str], observations: list[ScientificObservationSummary]) -> ScientificPlannerContext:
    return ScientificPlannerContext(
        questionSummary=question,
        intent="scientific_exploration",
        approvedActions=actions,
        remainingActionBudget=8,
        observations=observations,
        schemaCatalog=_semantic_catalog(),
    )


def _task(max_actions: int = 4) -> ResearchTask:
    return ResearchTask(
        runId="run-scientific-00000000000000000000000000000001",
        taskId="task-scientific-00000000000000000000000000000001",
        requesterId="principal-00000000000000000000000000000001",
        traceId="trace-scientific-00000000000000000000000000000001",
        question="探索当前数据中值得进一步验证的微生物现象",
        intent="scientific_exploration",
        requestedScopes=["mico:query:read", "mico:research:read"],
        allowedActions=["execute_read_query", "analyze_projection", "finish"],
        maxActions=max_actions,
        createdAt=NOW,
    )


class FakeScientificPlanner:
    def __init__(self) -> None:
        self.calls = 0
        self.analysis_calls = 0

    def plan_action(self, context):
        self.calls += 1
        action_id = "action-" + f"{self.calls:032x}"
        if self.calls == 1:
            return ScientificPlannerResult(
                action=ExecuteReadQueryAction(
                    actionId=action_id,
                    actionName="execute_read_query",
                    rationale="先获取当前问题所需的有界数据观察",
                    arguments=ExecuteReadQueryArguments(
                        actionName="execute_read_query",
                        sql="SELECT feature_name, abundance_value FROM approved_projection LIMIT 10",
                        limit=10,
                    ),
                ),
                mode="model",
            )
        if self.calls == 2:
            observation_id = context.observations[-1].observationId
            return ScientificPlannerResult(
                action=AnalyzeProjectionAction(
                    actionId=action_id,
                    actionName="analyze_projection",
                    rationale="根据已验证观察生成当前问题所需的受限分析",
                    arguments=AnalyzeProjectionArguments(
                        actionName="analyze_projection",
                        observationId=observation_id,
                        analysisGoal="总结有界结果的数量和范围",
                    ),
                ),
                mode="model",
            )
        return ScientificPlannerResult(
            action=FinishAction(
                actionId=action_id,
                actionName="finish",
                rationale="已有观察和分析结果，停止继续调用",
                arguments=FinishArguments(actionName="finish", reasonCode="EVIDENCE_SUFFICIENT"),
            ),
            mode="model",
        )

    def generate_analysis(self, context: AnalysisPlannerContext) -> GeneratedAnalysisPlannerResult:
        self.analysis_calls += 1
        return GeneratedAnalysisPlannerResult(
            plan=GeneratedAnalysisPlan(
                language="python",
                analysisType="descriptive_summary",
                code="result = {'metrics': {'row_count': len(rows)}, 'topFeatures': []}",
            ),
            mode="model",
        )


def test_scientific_loop_runs_dynamic_query_analysis_and_finish() -> None:
    port = FakeJavaPort()

    def response(call):
        value = completed_response(call)
        value.data = {
            "columns": ["feature_name", "abundance_value"],
            "rows": [
                {"feature_name": "feature_a", "abundance_value": 0.75},
                {"feature_name": "feature_b", "abundance_value": 0.25},
            ],
        }
        value.rowCount = 2
        value.dataSnapshot.rowCount = 2
        return value

    port.response_factory = response
    planner = FakeScientificPlanner()
    result = ScientificRuntime(port, planner).run(_task())

    assert result["status"] == "COMPLETED"
    assert result["actionCount"] == 3
    assert port.calls == 1
    assert planner.analysis_calls == 1
    assert result["report"] is not None
    report_text = result["report"].model_dump_json()
    assert "feature_a" not in report_text
    assert "sourceSampleId" not in report_text
    assert "internalRecordId" not in report_text


def test_scientific_loop_repairs_repeated_projection_without_raw_summary_rows() -> None:
    port = FakeJavaPort()

    def response(call):
        value = completed_response(call)
        value.data = {
            "columns": ["feature_name", "abundance_value"],
            "rows": [
                {"feature_name": "feature_a", "abundance_value": 0.75},
                {"feature_name": "feature_b", "abundance_value": 0.25},
            ],
        }
        value.rowCount = 2
        value.dataSnapshot.rowCount = 2
        return value

    class ChainedPlanner(FakeScientificPlanner):
        def plan_action(self, context):
            self.calls += 1
            action_id = "action-" + f"{self.calls:032x}"
            if self.calls == 1:
                return ScientificPlannerResult(
                    action=ExecuteReadQueryAction(
                        actionId=action_id,
                        actionName="execute_read_query",
                        rationale="先获取有界观察",
                        arguments=ExecuteReadQueryArguments(
                            actionName="execute_read_query",
                            sql="SELECT feature_name, abundance_value FROM approved_projection LIMIT 10",
                            limit=10,
                        ),
                    ),
                    mode="model",
                )
            if self.calls in {2, 3}:
                return ScientificPlannerResult(
                    action=AnalyzeProjectionAction(
                        actionId=action_id,
                        actionName="analyze_projection",
                        rationale="基于上一阶段的有界投影继续分析",
                        arguments=AnalyzeProjectionArguments(
                            actionName="analyze_projection",
                            observationId=context.observations[-1].observationId,
                            analysisGoal="总结当前有界投影的变化",
                        ),
                    ),
                    mode="model",
                )
            return ScientificPlannerResult(
                action=FinishAction(
                    actionId=action_id,
                    actionName="finish",
                    rationale="已有连续分析结果，停止继续调用",
                    arguments=FinishArguments(actionName="finish", reasonCode="EVIDENCE_SUFFICIENT"),
                ),
                mode="model",
            )

    port.response_factory = response
    planner = ChainedPlanner()
    result = ScientificRuntime(port, planner).run(_task(max_actions=4))

    assert result["status"] == "COMPLETED"
    assert result["actionCount"] == 3
    assert planner.analysis_calls == 1
    assert "SCIENTIFIC_PLANNER_REPEATED_PROJECTION_REPAIRED" in result["fallbackCodes"]
    assert "OBSERVATION_NOT_ANALYZABLE" not in result["fallbackCodes"]


def test_scientific_loop_rejects_mismatched_java_response_without_report() -> None:
    port = FakeJavaPort()

    def mismatched(call):
        response = completed_response(call)
        response.toolCallId = "call-" + "f" * 32
        return response

    port.response_factory = mismatched
    result = ScientificRuntime(port, FakeScientificPlanner()).run(_task())

    assert result["status"] == "FAILED"
    assert result["errorCode"] == "JAVA_TOOL_RESPONSE_MISMATCH"
    assert result["report"] is None
    assert all(event.dataSnapshotId is None for event in result["auditEvents"])


def test_scientific_loop_replans_semantic_java_read_rejection_without_counting_failed_action() -> None:
    class ReadRetryPlanner:
        def __init__(self) -> None:
            self.calls = 0
            self.feedbacks = []

        def plan_action(self, context):
            self.calls += 1
            self.feedbacks.append(list(context.executionFeedback))
            action_id = "action-" + f"{self.calls:032x}"
            if not context.observations:
                suffix = "alternate_value" if context.executionFeedback else "value"
                return ScientificPlannerResult(
                    action=ExecuteReadQueryAction(
                        actionId=action_id,
                        actionName="execute_read_query",
                        rationale="以新的受限投影重试 Java 已拒绝的读取",
                        arguments=ExecuteReadQueryArguments(
                            actionName="execute_read_query",
                            sql=f"SELECT {suffix} FROM approved_projection LIMIT 10",
                            limit=10,
                        ),
                    ),
                    mode="model",
                )
            return ScientificPlannerResult(
                action=FinishAction(
                    actionId=action_id,
                    actionName="finish",
                    rationale="已获得一项经过 Java 验证的有界观察",
                    arguments=FinishArguments(actionName="finish", reasonCode="EVIDENCE_SUFFICIENT"),
                ),
                mode="model",
            )

    port = FakeJavaPort()

    def fail_once(call):
        if port.calls == 1:
            return JavaToolResponse(
                toolCallId=call.toolCallId,
                runId=call.runId,
                status="FAILED",
                source="java_agent_read_model",
                generatedAt=NOW,
                error=JavaToolError(
                    code="READ_MODEL_ARGUMENT_REJECTED",
                    message="Read-model rejected the supplied query boundary",
                ),
            )
        return completed_response(call)

    port.response_factory = fail_once
    planner = ReadRetryPlanner()
    task = _task(max_actions=3).model_copy(update={
        "allowedActions": ["execute_read_query", "finish"],
    })
    result = ScientificRuntime(port, planner).run(task)

    assert result["status"] == "COMPLETED"
    assert result["actionCount"] == 2
    assert port.calls == 2
    assert planner.feedbacks == [[], ["READ_MODEL_ARGUMENT_REJECTED"], []]
    assert [item.chosenAction for item in result["decisionRecords"]] == [
        "execute_read_query", "finish"
    ]
    assert result["fallbackCodes"].count("SCIENTIFIC_PLANNER_JAVA_READ_REPLAN") == 1
    assert [event.errorCode for event in result["auditEvents"]].count(
        "READ_MODEL_ARGUMENT_REJECTED"
    ) == 1


def test_scientific_loop_blocks_repeated_semantic_read_plan_before_java_execution() -> None:
    class DuplicateThenFreshPlanner:
        def __init__(self) -> None:
            self.calls = 0
            self.feedbacks = []

        def plan_action(self, context):
            self.calls += 1
            self.feedbacks.append(list(context.executionFeedback))
            action_id = "action-" + f"{self.calls:032x}"
            if not context.observations:
                column = "value" if self.calls < 3 else "alternate_value"
                return ScientificPlannerResult(
                    action=ExecuteReadQueryAction(
                        actionId=action_id,
                        actionName="execute_read_query",
                        rationale="生成 Java 目录允许的有界读取",
                        arguments=ExecuteReadQueryArguments(
                            actionName="execute_read_query",
                            sql=f"SELECT {column} FROM approved_projection LIMIT 10",
                            limit=10,
                        ),
                    ),
                    mode="model",
                )
            return ScientificPlannerResult(
                action=FinishAction(
                    actionId=action_id,
                    actionName="finish",
                    rationale="已完成经过验证的读取",
                    arguments=FinishArguments(actionName="finish", reasonCode="EVIDENCE_SUFFICIENT"),
                ),
                mode="model",
            )

    port = FakeJavaPort()

    def fail_first_only(call):
        if port.calls == 1:
            return JavaToolResponse(
                toolCallId=call.toolCallId,
                runId=call.runId,
                status="FAILED",
                source="java_agent_read_model",
                generatedAt=NOW,
                error=JavaToolError(
                    code="READ_MODEL_ARGUMENT_REJECTED",
                    message="Read-model rejected the supplied query boundary",
                ),
            )
        return completed_response(call)

    port.response_factory = fail_first_only
    planner = DuplicateThenFreshPlanner()
    task = _task(max_actions=3).model_copy(update={
        "allowedActions": ["execute_read_query", "finish"],
    })
    result = ScientificRuntime(port, planner).run(task)

    assert result["status"] == "COMPLETED"
    assert port.calls == 2
    assert planner.feedbacks[:3] == [
        [],
        ["READ_MODEL_ARGUMENT_REJECTED"],
        ["READ_MODEL_ARGUMENT_REJECTED", "READ_PLAN_REPEAT_REJECTED"],
    ]
    assert [item.chosenAction for item in result["decisionRecords"]] == [
        "execute_read_query", "finish"
    ]


def test_scientific_loop_safely_fails_after_three_semantic_java_read_replans() -> None:
    class AlwaysFreshReadPlanner:
        def __init__(self) -> None:
            self.calls = 0

        def plan_action(self, context):
            self.calls += 1
            return ScientificPlannerResult(
                action=ExecuteReadQueryAction(
                    actionId="action-" + f"{self.calls:032x}",
                    actionName="execute_read_query",
                    rationale="使用不同的受限列重新规划读取",
                    arguments=ExecuteReadQueryArguments(
                        actionName="execute_read_query",
                        sql=(
                            f"SELECT value_{self.calls} FROM approved_projection LIMIT 10"
                        ),
                        limit=10,
                    ),
                ),
                mode="model",
            )

    def always_fail(call):
        return JavaToolResponse(
            toolCallId=call.toolCallId,
            runId=call.runId,
            status="FAILED",
            source="java_agent_read_model",
            generatedAt=NOW,
            error=JavaToolError(
                code="READ_MODEL_ARGUMENT_REJECTED",
                message="Read-model rejected the supplied query boundary",
            ),
        )

    port = FakeJavaPort(response_factory=always_fail)
    task = _task(max_actions=2).model_copy(update={
        "allowedActions": ["execute_read_query", "finish"],
    })
    result = ScientificRuntime(port, AlwaysFreshReadPlanner()).run(task)

    assert result["status"] == "FAILED"
    assert result["errorCode"] == "READ_MODEL_ARGUMENT_REJECTED"
    assert result["report"] is None
    assert result["actionCount"] == 0
    assert port.calls == 4
    assert len([
        event for event in result["auditEvents"]
        if event.toolName == "execute_read_query"
    ]) == 4


def test_scientific_loop_fails_closed_on_java_execution_failure_without_replanning() -> None:
    def execution_failure(call):
        return JavaToolResponse(
            toolCallId=call.toolCallId,
            runId=call.runId,
            status="FAILED",
            source="java_agent_read_model",
            generatedAt=NOW,
            error=JavaToolError(
                code="READ_MODEL_EXECUTION_FAILED",
                message="Read-model execution did not complete",
                retryable=True,
            ),
        )

    port = FakeJavaPort(response_factory=execution_failure)
    planner = FakeScientificPlanner()
    task = _task(max_actions=3).model_copy(update={
        "allowedActions": ["execute_read_query", "finish"],
    })
    result = ScientificRuntime(port, planner).run(task)

    assert result["status"] == "FAILED"
    assert result["errorCode"] == "READ_MODEL_EXECUTION_FAILED"
    assert result["report"] is None
    assert port.calls == 1
    assert planner.calls == 1


def test_scientific_loop_budget_stops_without_fixed_business_route() -> None:
    planner = FakeScientificPlanner()
    result = ScientificRuntime(FakeJavaPort(), planner).run(_task(max_actions=1))

    assert result["status"] == "COMPLETED"
    assert result["actionCount"] == 1
    assert result["report"] is not None


def test_data_fact_repairs_a_second_read_into_a_finish() -> None:
    class RepeatingDataFactPlanner(FakeScientificPlanner):
        def plan_action(self, context):
            self.calls += 1
            if self.calls in {1, 2}:
                return ScientificPlannerResult(
                    action=ExecuteReadQueryAction(
                        actionId="action-" + f"{self.calls:032x}",
                        actionName="execute_read_query",
                        rationale="读取当前数据事实所需的有界投影",
                        arguments=ExecuteReadQueryArguments(
                            actionName="execute_read_query",
                            sql=(
                                "SELECT value FROM approved_projection LIMIT 10"
                                if self.calls == 1
                                else "SELECT other_value FROM approved_projection LIMIT 10"
                            ),
                            limit=10,
                        ),
                    ),
                    mode="model",
                )
            raise AssertionError("data fact must finish after its first validated read")

    task = _task(max_actions=3).model_copy(update={
        "intent": "data_fact",
        "allowedActions": ["execute_read_query", "finish"],
    })
    planner = RepeatingDataFactPlanner()
    result = ScientificRuntime(FakeJavaPort(), planner).run(task)

    assert result["status"] == "COMPLETED"
    assert [item.chosenAction for item in result["decisionRecords"]] == [
        "execute_read_query", "finish"
    ]
    assert result["actionCount"] == 2
    assert planner.calls == 2


def test_data_fact_allows_one_fact_query_after_metadata_inspection() -> None:
    class InspectThenFactPlanner(FakeScientificPlanner):
        def plan_action(self, context):
            self.calls += 1
            action_id = "action-" + f"{self.calls:032x}"
            if self.calls == 1:
                return ScientificPlannerResult(
                    action=InspectCohortAction(
                        actionId=action_id,
                        actionName="inspect_cohort",
                        rationale="先检查元数据投影",
                        arguments=InspectCohortArguments(
                            actionName="inspect_cohort",
                            sql="SELECT value FROM approved_projection LIMIT 10",
                            limit=10,
                        ),
                    ),
                    mode="model",
                )
            if self.calls == 2:
                return ScientificPlannerResult(
                    action=ExecuteReadQueryAction(
                        actionId=action_id,
                        actionName="execute_read_query",
                        rationale="执行一个受限的数据事实查询",
                        arguments=ExecuteReadQueryArguments(
                            actionName="execute_read_query",
                            sql="SELECT value FROM approved_projection LIMIT 10",
                            limit=10,
                        ),
                    ),
                    mode="model",
                )
            return ScientificPlannerResult(
                action=FinishAction(
                    actionId=action_id,
                    actionName="finish",
                    rationale="已取得所需的元数据与事实投影",
                    arguments=FinishArguments(
                        actionName="finish",
                        reasonCode="EVIDENCE_SUFFICIENT",
                    ),
                ),
                mode="model",
            )

    task = _task(max_actions=4).model_copy(update={
        "intent": "data_fact",
        "allowedActions": ["inspect_cohort", "execute_read_query", "finish"],
    })
    planner = InspectThenFactPlanner()
    port = FakeJavaPort()
    result = ScientificRuntime(port, planner).run(task)

    assert result["status"] == "COMPLETED"
    assert [item.chosenAction for item in result["decisionRecords"]] == [
        "inspect_cohort", "execute_read_query", "finish"
    ]
    assert port.calls == 2


def test_data_fact_repairs_generic_no_new_finish_after_validated_inspection() -> None:
    class GenericNoNewPlanner:
        def __init__(self) -> None:
            self.calls = 0

        def plan_action(self, _context):
            self.calls += 1
            action_id = "action-" + f"{self.calls:032x}"
            if self.calls == 1:
                return ScientificPlannerResult(
                    action=InspectCohortAction(
                        actionId=action_id,
                        actionName="inspect_cohort",
                        rationale="先检查元数据投影",
                        arguments=InspectCohortArguments(
                            actionName="inspect_cohort",
                            sql="SELECT value FROM approved_projection LIMIT 10",
                            limit=10,
                        ),
                    ),
                    mode="model",
                )
            return ScientificPlannerResult(
                action=FinishAction(
                    actionId=action_id,
                    actionName="finish",
                    rationale="使用通用停止码结束数据事实查询",
                    arguments=FinishArguments(
                        actionName="finish",
                        reasonCode="NO_NEW_INFORMATION",
                    ),
                ),
                mode="model",
            )

        def generate_analysis(self, _context):
            return FakeScientificPlanner().generate_analysis(_context)

    task = _task(max_actions=3).model_copy(update={
        "intent": "data_fact",
        "question": "统计当前元数据投影中的样本记录数",
        "allowedActions": ["inspect_cohort", "finish"],
    })
    result = ScientificRuntime(FakeJavaPort(), GenericNoNewPlanner()).run(task)

    assert result["status"] == "COMPLETED"
    assert result["stopReasonCode"] == "EVIDENCE_SUFFICIENT"
    assert "SCIENTIFIC_PLANNER_STOP_REASON_REPAIRED" in result["fallbackCodes"]


def test_initial_inspection_uses_runtime_catalog_query_not_model_sql() -> None:
    class UnsafeInspectionPlanner:
        def plan_action(self, _context):
            return ScientificPlannerResult(
                action=InspectCohortAction(
                    actionId="action-" + "1" * 32,
                    actionName="inspect_cohort",
                    rationale="模型提出一个不应作为首步的复杂读取",
                    arguments=InspectCohortArguments(
                        actionName="inspect_cohort",
                        sql="SELECT disease FROM non_catalog_source LIMIT 10",
                        limit=10,
                    ),
                ),
                mode="model",
            )

    catalog = SchemaSemanticCatalog(
        schemaVersion="schema-catalog-v1",
        source="java_schema_contract",
        generatedAt=NOW,
        entities=[SchemaEntitySemantics(
            entityName="patient_record",
            sourceTable="patients",
            fields=[SchemaFieldSemantics(
                name="disease",
                dataType="string",
                nullable=True,
                semanticStatus="verified",
                filterable=True,
                groupable=True,
                displayable=True,
                sensitive=False,
                description="Verified non-sensitive disease label",
            )],
            primaryKeyFields=[],
        )],
        joins=[],
        queryRules=[
            "select_or_with_only",
            "explicit_columns_only",
            "no_cross_database_reference",
            "bounded_limit_required",
            "java_final_validation",
        ],
    )
    task = _task(max_actions=3).model_copy(update={
        "allowedActions": ["inspect_cohort", "finish"],
    })
    state = {
        "request": task,
        "schemaCatalog": catalog,
        "observations": [],
        "actionHistory": [],
        "actionSignatures": [],
        "fallbackCodes": [],
        "auditEvents": [],
    }

    planned = _plan_action(UnsafeInspectionPlanner(), state)

    assert planned["currentAction"].actionName == "inspect_cohort"
    assert planned["currentAction"].arguments.sql == "SELECT disease FROM patients LIMIT 100"
    assert "SCIENTIFIC_PLANNER_INITIAL_INSPECTION_REPAIRED" in planned["fallbackCodes"]


def test_dynamic_materialization_preserves_model_inspection_sql() -> None:
    class DynamicInspectionPlanner:
        dynamic_action_materialization = True

        def plan_action(self, _context):
            return ScientificPlannerResult(
                action=InspectCohortAction(
                    actionId="action-" + "2" * 32,
                    actionName="inspect_cohort",
                    rationale="Inspect the requested disease distribution before analysis.",
                    arguments=InspectCohortArguments(
                        actionName="inspect_cohort",
                        sql="SELECT disease FROM patients LIMIT 17",
                        limit=17,
                    ),
                ),
                mode="model",
            )

    task = _task(max_actions=3).model_copy(update={
        "allowedActions": ["inspect_cohort", "finish"],
    })
    state = {
        "request": task,
        "schemaCatalog": _semantic_catalog(),
        "observations": [],
        "actionHistory": [],
        "actionSignatures": [],
        "fallbackCodes": [],
        "auditEvents": [],
    }

    planned = _plan_action(DynamicInspectionPlanner(), state)

    assert planned["currentAction"].actionName == "inspect_cohort"
    assert planned["currentAction"].arguments.sql == "SELECT disease FROM patients LIMIT 17"
    assert planned["decisionRecords"][-1].planner_origin == "model"
    assert "SCIENTIFIC_PLANNER_INITIAL_INSPECTION_REPAIRED" not in planned["fallbackCodes"]


def test_focused_comparison_repairs_a_repeated_projection_into_a_finish() -> None:
    class RepeatingFocusedPlanner(FakeScientificPlanner):
        def plan_action(self, context):
            self.calls += 1
            action_id = "action-" + f"{self.calls:032x}"
            if self.calls == 1:
                return ScientificPlannerResult(
                    action=ExecuteReadQueryAction(
                        actionId=action_id,
                        actionName="execute_read_query",
                        rationale="读取比较所需的有界投影",
                        arguments=ExecuteReadQueryArguments(
                            actionName="execute_read_query",
                            sql="SELECT value FROM approved_projection LIMIT 10",
                            limit=10,
                        ),
                    ),
                    mode="model",
                )
            return ScientificPlannerResult(
                action=AnalyzeProjectionAction(
                    actionId=action_id,
                    actionName="analyze_projection",
                    rationale="执行当前受限投影分析",
                    arguments=AnalyzeProjectionArguments(
                        actionName="analyze_projection",
                        observationId=context.observations[-1].observationId,
                        analysisGoal="比较当前有界投影",
                    ),
                ),
                mode="model",
            )

    task = _task(max_actions=4).model_copy(update={
        "intent": "focused_comparison",
        "allowedActions": ["execute_read_query", "analyze_projection", "finish"],
    })
    planner = RepeatingFocusedPlanner()
    result = ScientificRuntime(FakeJavaPort(), planner).run(task)

    assert result["status"] == "COMPLETED"
    assert [item.chosenAction for item in result["decisionRecords"]] == [
        "execute_read_query", "analyze_projection", "finish"
    ]
    assert planner.analysis_calls == 1


def test_focused_comparison_repairs_a_premature_finish_into_final_projection() -> None:
    class PrematureFinishPlanner(FakeScientificPlanner):
        def plan_action(self, context):
            self.calls += 1
            action_id = "action-" + f"{self.calls:032x}"
            if self.calls == 1:
                return ScientificPlannerResult(
                    action=ExecuteReadQueryAction(
                        actionId=action_id,
                        actionName="execute_read_query",
                        rationale="读取比较所需的有界投影",
                        arguments=ExecuteReadQueryArguments(
                            actionName="execute_read_query",
                            sql="SELECT value FROM approved_projection LIMIT 10",
                            limit=10,
                        ),
                    ),
                    mode="model",
                )
            if self.calls == 2:
                return ScientificPlannerResult(
                    action=CompareGroupsAction(
                        actionId=action_id,
                        actionName="compare_groups",
                        rationale="比较已验证投影中的两个研究组",
                        arguments=ProjectionAnalysisArguments(
                            actionName="compare_groups",
                            observationIds=[context.observations[-1].observationId],
                            analysisGoal="比较当前有界投影",
                        ),
                    ),
                    mode="model",
                )
            return ScientificPlannerResult(
                action=FinishAction(
                    actionId=action_id,
                    actionName="finish",
                    rationale="过早停止",
                    arguments=FinishArguments(
                        actionName="finish",
                        reasonCode="NO_NEW_INFORMATION",
                    ),
                ),
                mode="model",
            )

    task = _task(max_actions=5).model_copy(update={
        "intent": "focused_comparison",
        "allowedActions": ["execute_read_query", "compare_groups", "analyze_projection", "finish"],
    })
    planner = PrematureFinishPlanner()
    result = ScientificRuntime(FakeJavaPort(), planner).run(task)

    assert result["status"] == "COMPLETED"
    assert [item.chosenAction for item in result["decisionRecords"]] == [
        "execute_read_query", "compare_groups", "analyze_projection", "finish"
    ]
    assert result["stopReasonCode"] == "EVIDENCE_SUFFICIENT"


def test_runtime_owns_premature_budget_and_insufficient_stop_reasons() -> None:
    class BudgetFinishPlanner:
        def __init__(self) -> None:
            self.calls = 0

        def plan_action(self, _context):
            self.calls += 1
            return FinishAction(
                actionId="action-" + f"{self.calls:032x}",
                actionName="finish",
                rationale="停止当前探索",
                arguments=FinishArguments(
                    actionName="finish",
                    reasonCode="ACTION_BUDGET_EXHAUSTED",
                ),
            )

        def generate_analysis(self, _context):
            return FakeScientificPlanner().generate_analysis(_context)

    class PlannerPort(BudgetFinishPlanner):
        def plan_action(self, context):
            return ScientificPlannerResult(action=super().plan_action(context), mode="model")

    normal_task = _task(max_actions=3).model_copy(update={
        "allowedActions": ["execute_read_query", "finish"],
    })
    normal = ScientificRuntime(FakeJavaPort(), PlannerPort()).run(normal_task)
    assert normal["stopReasonCode"] == "UPSTREAM_REJECTED"

    insufficient_task = normal_task.model_copy(update={
        "question": "当前证据不足以支持比较，应安全结束。",
    })
    insufficient = ScientificRuntime(FakeJavaPort(), PlannerPort()).run(insufficient_task)
    assert insufficient["stopReasonCode"] == "NO_NEW_INFORMATION"


def test_scientific_exploration_owns_no_new_stop_after_nonempty_evidence_retrieval() -> None:
    class NoNewFinishPlanner:
        def plan_action(self, _context):
            return ScientificPlannerResult(
                action=FinishAction(
                    actionId="action-" + "1" * 32,
                    actionName="finish",
                    rationale="错误地把已检索到的证据视为没有新信息",
                    arguments=FinishArguments(actionName="finish", reasonCode="NO_NEW_INFORMATION"),
                ),
                mode="model",
            )

    task = _task(max_actions=5).model_copy(update={
        "allowedActions": ["execute_read_query", "retrieve_evidence", "finish"],
    })
    state = {
        "request": task,
        "observations": [Observation(
            observationId="observation-" + "1" * 32,
            actionId="action-" + "2" * 32,
            actionName="retrieve_evidence",
            status="VALIDATED",
            source="knowledge_hybrid",
            queryHash="sha256:" + "a" * 64,
            rowCount=2,
            generatedAt=NOW,
            schemaVersion="knowledge-retrieval-v1",
        )],
        "actionHistory": ["execute_read_query", "retrieve_evidence"],
        "actionSignatures": [],
        "fallbackCodes": [],
        "auditEvents": [],
    }

    planned = _plan_action(NoNewFinishPlanner(), state)

    assert planned["currentAction"].actionName == "finish"
    assert planned["currentAction"].arguments.reasonCode == "EVIDENCE_SUFFICIENT"
    assert "SCIENTIFIC_PLANNER_STOP_REASON_REPAIRED" in planned["fallbackCodes"]


def test_focused_comparison_owns_no_new_stop_after_nonempty_evidence_retrieval() -> None:
    class NoNewFinishPlanner:
        def plan_action(self, _context):
            return ScientificPlannerResult(
                action=FinishAction(
                    actionId="action-" + "1" * 32,
                    actionName="finish",
                    rationale="错误地把已检索到的证据视为没有新信息",
                    arguments=FinishArguments(
                        actionName="finish", reasonCode="NO_NEW_INFORMATION"
                    ),
                ),
                mode="model",
            )

    task = _task(max_actions=5).model_copy(update={
        "intent": "focused_comparison",
        "allowedActions": ["execute_read_query", "retrieve_evidence", "finish"],
    })
    state = {
        "request": task,
        "observations": [Observation(
            observationId="observation-" + "1" * 32,
            actionId="action-" + "2" * 32,
            actionName="retrieve_evidence",
            status="VALIDATED",
            source="knowledge_hybrid",
            queryHash="sha256:" + "a" * 64,
            rowCount=2,
            generatedAt=NOW,
            schemaVersion="knowledge-retrieval-v1",
        )],
        "actionHistory": ["execute_read_query", "retrieve_evidence"],
        "actionSignatures": [],
        "fallbackCodes": [],
        "auditEvents": [],
    }

    planned = _plan_action(NoNewFinishPlanner(), state)

    assert planned["currentAction"].actionName == "finish"
    assert planned["currentAction"].arguments.reasonCode == "EVIDENCE_SUFFICIENT"
    assert "SCIENTIFIC_PLANNER_STOP_REASON_REPAIRED" in planned["fallbackCodes"]


def test_semantic_path_enforces_country_project_stability_before_evidence() -> None:
    actions = [
        "inspect_cohort", "stratified_analysis", "cross_project_validate",
        "retrieve_evidence", "finish",
    ]
    question = "Explore candidate stability by project and country with coverage limitations."
    first = semantic_scientific_next_action(
        _semantic_context(question, actions, [_semantic_observation(1, "inspect_cohort")])
    )
    assert first is not None
    assert first.actionName == "stratified_analysis"
    assert first.arguments.dimensions == ["country", "project_name"]

    second = semantic_scientific_next_action(
        _semantic_context(question, actions, [
            _semantic_observation(1, "inspect_cohort"),
            _semantic_observation(2, "stratified_analysis", "python_bounded_analysis"),
        ])
    )
    assert second is not None
    assert second.actionName == "cross_project_validate"
    assert len(second.arguments.observationIds) == 2


def test_semantic_path_enforces_age_stratification_before_projection() -> None:
    actions = ["inspect_cohort", "stratified_analysis", "analyze_projection", "finish"]
    question = "按年龄层分层比较两个研究组，并报告每层纳入数量。"
    stratified = semantic_scientific_next_action(_semantic_context(
        question,
        actions,
        [_semantic_observation(1, "inspect_cohort")],
    ))

    assert stratified is not None
    assert stratified.actionName == "stratified_analysis"
    assert stratified.arguments.dimensions == ["age"]


def test_semantic_path_does_not_construct_empty_stratification_dimensions() -> None:
    actions = ["inspect_cohort", "stratified_analysis", "analyze_projection", "finish"]
    catalog = _semantic_catalog().model_copy(update={
        "entities": [SchemaEntitySemantics(
            entityName="sample_metadata",
            sourceTable="meta2db_sample_metadata",
            fields=[SchemaFieldSemantics(
                name="unrelated_field", dataType="string", nullable=True,
                semanticStatus="verified", filterable=True, groupable=True,
                displayable=True, description="Unrelated verified field",
            )],
        )],
    })
    context = ScientificPlannerContext(
        questionSummary="按年龄区间分层比较两个研究组的微生态摘要。",
        intent="scientific_exploration",
        approvedActions=actions,
        remainingActionBudget=8,
        observations=[_semantic_observation(1, "inspect_cohort")],
        schemaCatalog=catalog,
    )

    assert semantic_scientific_next_action(context) is None


def test_data_fact_planner_uses_catalog_count_adapter_without_model_sql() -> None:
    abundance = SchemaEntitySemantics(
        entityName="standard_abundance",
        sourceTable="meta2db_standard_abundance",
        fields=[
            SchemaFieldSemantics(
                name="sample_id", dataType="string", nullable=False,
                semanticStatus="verified", filterable=True, groupable=True,
                displayable=True, description="Verified sample key",
            ),
            SchemaFieldSemantics(
                name="patient_id", dataType="string", nullable=True,
                semanticStatus="verified", filterable=True, groupable=True,
                displayable=True, description="Verified patient key",
            ),
        ],
    )
    catalog = _semantic_catalog().model_copy(update={
        "entities": [_semantic_catalog().entities[0], abundance],
    })
    context = _semantic_context(
        "分别统计样本键唯一数、元数据记录数和丰度存储记录数，并明确三者口径。",
        ["execute_read_query", "finish"],
        [],
    ).model_copy(update={"intent": "data_fact", "schemaCatalog": catalog})

    action = semantic_scientific_next_action(context)

    assert isinstance(action, ExecuteReadQueryAction)
    assert "COUNT(DISTINCT sample_id)" in action.arguments.sql
    assert "metadata_record_count" in action.arguments.sql
    assert "abundance_storage_row_count" in action.arguments.sql


def test_semantic_path_validates_multi_feature_stability_before_evidence() -> None:
    context = _semantic_context(
        "寻找多个研究组共同变化的 feature 组合，验证稳定性后再检索证据。",
        ["inspect_cohort", "compare_groups", "cross_project_validate", "retrieve_evidence", "finish"],
        [
            _semantic_observation(1, "inspect_cohort"),
            _semantic_observation(2, "compare_groups", "python_bounded_analysis"),
        ],
    )

    action = semantic_scientific_next_action(context)

    assert action is not None
    assert action.actionName == "cross_project_validate"


def test_semantic_path_controls_confounders_before_project_validation() -> None:
    context = _semantic_context(
        "探索年龄和性别是否解释疾病相关的微生态差异，再进行跨 project 验证。",
        ["inspect_cohort", "compare_groups", "adjust_confounders", "cross_project_validate", "retrieve_evidence", "finish"],
        [
            _semantic_observation(1, "inspect_cohort"),
            _semantic_observation(2, "compare_groups", "python_bounded_analysis"),
        ],
    )

    action = semantic_scientific_next_action(context)

    assert action is not None
    assert action.actionName == "adjust_confounders"
    assert "age" in action.arguments.confounders
    assert "gender" in action.arguments.confounders


def test_semantic_path_does_not_misclassify_country_adjustment_as_stability() -> None:
    context = _semantic_context(
        "比较两个 project 的微生物特征时控制 country 差异后重新分析。",
        ["inspect_cohort", "compare_groups", "adjust_confounders", "analyze_projection", "finish"],
        [
            _semantic_observation(1, "inspect_cohort"),
            _semantic_observation(2, "compare_groups", "python_bounded_analysis"),
        ],
    )

    action = semantic_scientific_next_action(context)

    assert action is not None
    assert action.actionName == "adjust_confounders"
    assert action.arguments.confounders == ["country", "project_name"]


def test_semantic_path_recognizes_cross_disease_specificity_wording() -> None:
    context = _semantic_context(
        "探索一个候选 feature 是否跨 project 稳定、跨 disease 特异，并用知识图谱补证。",
        ["inspect_cohort", "compare_groups", "cross_project_validate", "cross_disease_validate", "retrieve_evidence", "finish"],
        [
            _semantic_observation(1, "inspect_cohort"),
            _semantic_observation(2, "compare_groups", "python_bounded_analysis"),
            _semantic_observation(3, "cross_project_validate", "python_bounded_analysis"),
        ],
    )

    action = semantic_scientific_next_action(context)

    assert action is not None
    assert action.actionName == "cross_disease_validate"


def test_focused_finish_after_read_is_repaired_into_group_comparison() -> None:
    class FinishAfterReadPlanner:
        def plan_action(self, _context):
            return ScientificPlannerResult(
                action=FinishAction(
                    actionId="action-" + "3" * 32,
                    actionName="finish",
                    rationale="错误地在读取后停止",
                    arguments=FinishArguments(
                        actionName="finish", reasonCode="NO_NEW_INFORMATION"
                    ),
                ),
                mode="model",
            )

    task = _task(max_actions=4).model_copy(update={
        "intent": "focused_comparison",
        "allowedActions": ["inspect_cohort", "compare_groups", "analyze_projection", "finish"],
    })
    state = {
        "request": task,
        "observations": [Observation(
            observationId="observation-" + "1" * 32,
            actionId="action-" + "2" * 32,
            actionName="inspect_cohort",
            status="VALIDATED",
            source="java_controlled_read",
            queryHash="sha256:" + "a" * 64,
            rowCount=2,
            generatedAt=NOW,
            schemaVersion="scientific-observation-v1",
        )],
        "schemaCatalog": _semantic_catalog(),
        "actionHistory": ["inspect_cohort"],
        "actionSignatures": [],
        "fallbackCodes": [],
        "auditEvents": [],
    }

    planned = _plan_action(FinishAfterReadPlanner(), state)

    assert planned["currentAction"].actionName == "compare_groups"
    assert "SCIENTIFIC_PLANNER_PREMATURE_FOCUSED_ACTION_REPAIRED" in planned["fallbackCodes"]


def test_semantic_path_enforces_comparability_combination_and_cross_disease_order() -> None:
    comparability_actions = ["inspect_cohort", "compare_groups", "cross_project_validate", "finish"]
    comparability_question = "Identify projects with matched case-control cohorts for comparability."
    compare = semantic_scientific_next_action(_semantic_context(
        comparability_question, comparability_actions, [_semantic_observation(1, "inspect_cohort")]
    ))
    assert compare is not None and compare.actionName == "compare_groups"
    project = semantic_scientific_next_action(_semantic_context(
        comparability_question, comparability_actions, [
            _semantic_observation(1, "inspect_cohort"),
            _semantic_observation(2, "compare_groups", "python_bounded_analysis"),
        ]
    ))
    assert project is not None and project.actionName == "cross_project_validate"

    combination_actions = ["inspect_cohort", "compare_groups", "analyze_projection", "retrieve_evidence", "finish"]
    combination_question = "Explore whether multiple microbes form a co-change combination."
    combination = semantic_scientific_next_action(_semantic_context(
        combination_question, combination_actions, [
            _semantic_observation(1, "inspect_cohort"),
            _semantic_observation(2, "compare_groups", "python_bounded_analysis"),
        ]
    ))
    assert combination is not None and combination.actionName == "analyze_projection"

    cross_disease_actions = [
        "inspect_cohort", "compare_groups", "cross_project_validate",
        "cross_disease_validate", "analyze_projection", "retrieve_evidence", "finish",
    ]
    cross_disease_question = "Explore a combination across multiple diseases and multiple projects."
    disease = semantic_scientific_next_action(_semantic_context(
        cross_disease_question, cross_disease_actions, [
            _semantic_observation(1, "inspect_cohort"),
            _semantic_observation(2, "compare_groups", "python_bounded_analysis"),
            _semantic_observation(3, "cross_project_validate", "python_bounded_analysis"),
        ]
    ))
    assert disease is not None and disease.actionName == "cross_disease_validate"


def test_semantic_path_enforces_confounder_adjustment_before_projection() -> None:
    actions = [
        "inspect_cohort", "compare_groups", "adjust_confounders",
        "analyze_projection", "finish",
    ]
    question = "比较两个研究组时控制年龄与性别混杂，并输出调整前后差异。"
    compare = semantic_scientific_next_action(_semantic_context(
        question,
        actions,
        [_semantic_observation(1, "inspect_cohort")],
    ))
    assert compare is not None
    assert compare.actionName == "compare_groups"

    adjusted = semantic_scientific_next_action(_semantic_context(
        question,
        actions,
        [
            _semantic_observation(1, "inspect_cohort"),
            _semantic_observation(2, "compare_groups", "python_bounded_analysis"),
        ],
    ))

    assert adjusted is not None
    assert adjusted.actionName == "adjust_confounders"
    assert adjusted.arguments.confounders == ["age", "gender"]


def test_confounder_adjustment_takes_priority_over_optional_stratification() -> None:
    actions = [
        "inspect_cohort", "compare_groups", "stratified_analysis", "adjust_confounders",
        "analyze_projection", "finish",
    ]
    question = "发现候选 species 后，检查年龄和性别差异是否解释了原始组间结果。"
    compare = semantic_scientific_next_action(_semantic_context(
        question,
        actions,
        [_semantic_observation(1, "inspect_cohort")],
    ))

    assert compare is not None
    assert compare.actionName == "compare_groups"

    adjusted = semantic_scientific_next_action(_semantic_context(
        question,
        actions,
        [
            _semantic_observation(1, "inspect_cohort"),
            _semantic_observation(2, "compare_groups", "python_bounded_analysis"),
        ],
    ))

    assert adjusted is not None
    assert adjusted.actionName == "adjust_confounders"


def test_semantic_path_enforces_project_confounder_adjustment() -> None:
    actions = [
        "inspect_cohort", "compare_groups", "adjust_confounders",
        "analyze_projection", "finish",
    ]
    question = "比较两个 project 的微生物特征，并控制 project 或批次差异后重新分析。"
    adjusted = semantic_scientific_next_action(_semantic_context(
        question,
        actions,
        [
            _semantic_observation(1, "inspect_cohort"),
            _semantic_observation(2, "compare_groups", "python_bounded_analysis"),
        ],
    ))

    assert adjusted is not None
    assert adjusted.actionName == "adjust_confounders"
    assert adjusted.arguments.confounders == ["project_name"]


def test_semantic_path_binds_missingness_stratification_to_catalog_dimensions() -> None:
    actions = ["inspect_cohort", "stratified_analysis", "analyze_projection", "finish"]
    question = "比较两个研究组的缺失模式，并按缺失状态分层描述分析结果。"
    stratified = semantic_scientific_next_action(_semantic_context(
        question,
        actions,
        [_semantic_observation(1, "inspect_cohort")],
    ))

    assert stratified is not None
    assert stratified.actionName == "stratified_analysis"
    assert stratified.arguments.dimensions
    assert set(stratified.arguments.dimensions).issubset(
        {"age", "gender", "country", "project_name", "disease"}
    )


def test_semantic_path_requires_cross_project_validation_before_evidence() -> None:
    actions = [
        "inspect_cohort", "compare_groups", "cross_project_validate",
        "retrieve_evidence", "finish",
    ]
    question = "探索 T2D 值得关注的微生态特征，并验证发现是否跨 project 稳定。"
    cross_project = semantic_scientific_next_action(_semantic_context(
        question,
        actions,
        [
            _semantic_observation(1, "inspect_cohort"),
            _semantic_observation(2, "compare_groups", "python_bounded_analysis"),
        ],
    ))

    assert cross_project is not None
    assert cross_project.actionName == "cross_project_validate"


def test_project_imbalance_continues_to_requested_evidence() -> None:
    actions = [
        "inspect_cohort", "compare_groups", "adjust_confounders",
        "cross_project_validate", "retrieve_evidence", "finish",
    ]
    question = (
        "探索 ASD 的稳定微生态特征，年龄和国家不平衡；"
        "请先做混杂校正，再验证跨 project 稳定性，并补充文献证据后再停止。"
    )
    evidence = semantic_scientific_next_action(_semantic_context(
        question,
        actions,
        [
            _semantic_observation(1, "inspect_cohort"),
            _semantic_observation(2, "compare_groups", "python_bounded_analysis"),
            _semantic_observation(3, "adjust_confounders", "python_bounded_analysis"),
            _semantic_observation(4, "cross_project_validate", "python_bounded_analysis"),
        ],
    ))

    assert evidence is not None
    assert evidence.actionName == "retrieve_evidence"


def test_semantic_path_recognizes_project_level_result_comparison() -> None:
    actions = [
        "inspect_cohort", "compare_groups", "cross_project_validate",
        "retrieve_evidence", "finish",
    ]
    question = "系统寻找一个疾病相关的稳定微生物特征，并比较总体结果与各 project 结果。"
    cross_project = semantic_scientific_next_action(_semantic_context(
        question,
        actions,
        [
            _semantic_observation(1, "inspect_cohort"),
            _semantic_observation(2, "compare_groups", "python_bounded_analysis"),
        ],
    ))

    assert cross_project is not None
    assert cross_project.actionName == "cross_project_validate"


def test_semantic_path_adds_country_project_stratification_after_validation() -> None:
    actions = [
        "inspect_cohort", "compare_groups", "cross_project_validate",
        "stratified_analysis", "retrieve_evidence", "finish",
    ]
    question = "探索某候选特征在 project 和 country 两个维度上的稳定性，并保留缺失信息。"
    stratified = semantic_scientific_next_action(_semantic_context(
        question,
        actions,
        [
            _semantic_observation(1, "inspect_cohort"),
            _semantic_observation(2, "compare_groups", "python_bounded_analysis"),
            _semantic_observation(3, "cross_project_validate", "python_bounded_analysis"),
        ],
    ))

    assert stratified is not None
    assert stratified.actionName == "stratified_analysis"
    assert stratified.arguments.dimensions == ["country", "project_name"]


def test_semantic_path_composes_project_confounder_and_disease_validation() -> None:
    actions = [
        "inspect_cohort", "compare_groups", "cross_project_validate",
        "adjust_confounders", "cross_disease_validate", "retrieve_evidence", "finish",
    ]
    question = "探索一个候选 species 是否同时满足跨 project 稳定、混杂控制后仍成立、且具有疾病特异性。"
    observations = [_semantic_observation(1, "inspect_cohort")]
    first = semantic_scientific_next_action(_semantic_context(question, actions, observations))
    assert first is not None and first.actionName == "compare_groups"

    observations.append(_semantic_observation(2, "compare_groups", "python_bounded_analysis"))
    second = semantic_scientific_next_action(_semantic_context(question, actions, observations))
    assert second is not None and second.actionName == "cross_project_validate"

    observations.append(_semantic_observation(3, "cross_project_validate", "python_bounded_analysis"))
    third = semantic_scientific_next_action(_semantic_context(question, actions, observations))
    assert third is not None and third.actionName == "adjust_confounders"

    observations.append(_semantic_observation(4, "adjust_confounders", "python_bounded_analysis"))
    fourth = semantic_scientific_next_action(_semantic_context(question, actions, observations))
    assert fourth is not None and fourth.actionName == "cross_disease_validate"


def test_semantic_path_requires_disease_specificity_validation_before_evidence() -> None:
    actions = [
        "inspect_cohort", "compare_groups", "cross_disease_validate",
        "retrieve_evidence", "finish",
    ]
    question = "探索 Obesity 相关的微生态特征，并检查它是否可能只是代谢相关而非疾病特异。"
    cross_disease = semantic_scientific_next_action(_semantic_context(
        question,
        actions,
        [
            _semantic_observation(1, "inspect_cohort"),
            _semantic_observation(2, "compare_groups", "python_bounded_analysis"),
        ],
    ))

    assert cross_disease is not None
    assert cross_disease.actionName == "cross_disease_validate"


def test_semantic_path_recognizes_other_disease_direction_validation() -> None:
    actions = [
        "inspect_cohort", "compare_groups", "cross_disease_validate",
        "retrieve_evidence", "finish",
    ]
    question = "寻找某疾病的特异性微生物，验证它在其他疾病中是否也发生同方向变化。"
    cross_disease = semantic_scientific_next_action(_semantic_context(
        question,
        actions,
        [
            _semantic_observation(1, "inspect_cohort"),
            _semantic_observation(2, "compare_groups", "python_bounded_analysis"),
        ],
    ))

    assert cross_disease is not None
    assert cross_disease.actionName == "cross_disease_validate"


def test_conflict_or_speculation_finishes_with_quality_risk_after_retrieval() -> None:
    class EvidenceFinishPlanner:
        def plan_action(self, _context):
            return ScientificPlannerResult(
                action=FinishAction(
                    actionId="action-" + "1" * 32,
                    actionName="finish",
                    rationale="停止当前探索",
                    arguments=FinishArguments(actionName="finish", reasonCode="EVIDENCE_SUFFICIENT"),
                ),
                mode="model",
            )

    task = _task(max_actions=5).model_copy(update={
        "question": "When data and literature conflict or remain speculative, downgrade the conclusion.",
        "allowedActions": ["inspect_cohort", "compare_groups", "retrieve_evidence", "finish"],
    })
    state = {
        "request": task,
        "observations": [Observation(
            observationId="observation-" + "1" * 32,
            actionId="action-" + "2" * 32,
            actionName="retrieve_evidence",
            status="VALIDATED",
            source="knowledge_hybrid",
            queryHash="sha256:" + "a" * 64,
            rowCount=1,
            generatedAt=NOW,
            schemaVersion="knowledge-retrieval-v1",
        )],
        "actionHistory": ["inspect_cohort", "compare_groups", "retrieve_evidence"],
        "actionSignatures": [],
        "fallbackCodes": [],
        "auditEvents": [],
    }
    planned = _plan_action(EvidenceFinishPlanner(), state)
    assert planned["currentAction"].arguments.reasonCode == "QUALITY_RISK"

    speculative_task = task.model_copy(update={
        "question": "Retain speculative functional evidence without presenting a causal conclusion.",
    })
    speculative_state = {
        **state,
        "request": speculative_task,
        "observations": list(state["observations"]),
        "actionHistory": list(state["actionHistory"]),
        "actionSignatures": [],
        "fallbackCodes": [],
        "auditEvents": [],
    }
    speculative = _plan_action(EvidenceFinishPlanner(), speculative_state)
    assert speculative["currentAction"].arguments.reasonCode == "EVIDENCE_SUFFICIENT"


def test_semantic_eval_path_uses_runtime_analysis_without_model_code_generation() -> None:
    class NoModelCodePlanner:
        def __init__(self) -> None:
            self.plan_calls = 0
            self.analysis_calls = 0

        def plan_action(self, _context):
            self.plan_calls += 1
            raise AssertionError("semantic path should select the next action before model planning")

        def generate_analysis(self, _context):
            self.analysis_calls += 1
            raise AssertionError("semantic Eval path must not request model-generated Python")

    task = _task(max_actions=4).model_copy(update={
        "question": "Identify projects with matched case-control cohorts for comparability.",
        "allowedActions": ["inspect_cohort", "compare_groups", "cross_project_validate", "finish"],
    })
    planner = NoModelCodePlanner()
    result = ScientificRuntime(
        FakeJavaPort(),
        planner,
        schema_catalog=_semantic_catalog(),
    ).run(task)

    assert result.status == "COMPLETED"
    assert [item.chosenAction for item in result.decisionRecords] == [
        "inspect_cohort", "compare_groups", "cross_project_validate", "finish",
    ]
    assert planner.plan_calls == 0
    assert planner.analysis_calls == 0
    assert "SCIENTIFIC_RUNTIME_DETERMINISTIC_ANALYSIS" in result.fallbackCodes


def test_sft_policy_first_planner_is_called_before_semantic_guard() -> None:
    class PolicyFirstPlanner:
        prefer_policy_action = True

        def __init__(self) -> None:
            self.plan_calls = 0

        def plan_action(self, context):
            self.plan_calls += 1
            return DeterministicScientificPlanner().plan_action(context)

        def generate_analysis(self, _context):
            return GeneratedAnalysisPlannerResult(
                plan=GeneratedAnalysisPlan(
                    language="python",
                    analysisType="descriptive_summary",
                    code="result = {'metrics': {'row_count': len(rows)}, 'topFeatures': []}",
                ),
                mode="deterministic",
            )

    task = _task(max_actions=4).model_copy(update={
        "question": "Identify projects with matched case-control cohorts for comparability.",
        "allowedActions": ["inspect_cohort", "compare_groups", "cross_project_validate", "finish"],
    })
    planner = PolicyFirstPlanner()
    result = ScientificRuntime(
        FakeJavaPort(),
        planner,
        schema_catalog=_semantic_catalog(),
    ).run(task)

    assert result.status == "COMPLETED"
    assert planner.plan_calls == 4
    assert [item.chosenAction for item in result.decisionRecords] == [
        "inspect_cohort", "compare_groups", "cross_project_validate", "finish",
    ]


def test_sft_policy_decisions_are_recorded_as_model_origin() -> None:
    class PolicyFirstPlanner:
        prefer_policy_action = True

        def plan_action(self, context):
            planned = DeterministicScientificPlanner().plan_action(context)
            return ScientificPlannerResult(action=planned.action, mode="sft_policy")

        def generate_analysis(self, _context):
            return GeneratedAnalysisPlannerResult(
                plan=GeneratedAnalysisPlan(
                    language="python",
                    analysisType="descriptive_summary",
                    code="result = {'metrics': {'row_count': len(rows)}, 'topFeatures': []}",
                ),
                mode="deterministic",
            )

    task = _task(max_actions=4).model_copy(update={
        "question": "Identify projects with matched case-control cohorts for comparability.",
        "allowedActions": ["inspect_cohort", "compare_groups", "cross_project_validate", "finish"],
    })
    result = ScientificRuntime(
        FakeJavaPort(), PolicyFirstPlanner(), schema_catalog=_semantic_catalog(),
    ).run(task)

    assert result.status == "COMPLETED"
    assert [record.planner_origin for record in result.decisionRecords] == [
        "model", "model", "semantic_guard", "model",
    ]
    assert all(record.raw_action is not None for record in result.decisionRecords)


def test_scientific_runtime_loads_java_schema_catalog_before_planning() -> None:
    from tests.test_schema_catalog import _catalog
    from mico_agent_runtime.contracts.tools import JavaToolResponse

    catalog = _catalog()
    calls = []

    class SchemaAwareJava:
        def execute(self, call):
            calls.append(call.toolName)
            return JavaToolResponse(
                toolCallId=call.toolCallId,
                runId=call.runId,
                status="COMPLETED",
                source="java_schema_contract",
                rowCount=2,
                schemaVersion="p1a-agent-tool-contract-v1",
                generatedAt=NOW,
                data=catalog.model_dump(mode="json"),
            )

    class CatalogAwarePlanner:
        def __init__(self):
            self.catalog = None

        def plan_action(self, context):
            self.catalog = context.schemaCatalog
            return ScientificPlannerResult(
                action=FinishAction(
                    actionId="action-" + "1" * 32,
                    actionName="finish",
                    rationale="停止并返回当前证据状态",
                    arguments=FinishArguments(actionName="finish", reasonCode="EVIDENCE_SUFFICIENT"),
                ),
                mode="deterministic",
            )

    planner = CatalogAwarePlanner()
    java = SchemaAwareJava()
    result = ScientificRuntime(
        java,
        planner,
        schema_catalog_port=JavaSchemaCatalogPort(java),
    ).run(_task(max_actions=1))

    assert result["status"] == "COMPLETED"
    assert planner.catalog is not None
    assert planner.catalog.source == "java_schema_contract"
    assert calls == ["describe_read_schema"]


def test_scientific_runtime_fails_closed_when_schema_catalog_is_unavailable() -> None:
    class BrokenCatalog:
        def load(self, *, run_id, task_id):
            raise JavaPortTransportError("JAVA_SCHEMA_CATALOG_UNAVAILABLE")

    result = ScientificRuntime(
        FakeJavaPort(),
        FakeScientificPlanner(),
        schema_catalog_port=BrokenCatalog(),
    ).run(_task())

    assert result["status"] == "FAILED"
    assert result["errorCode"] == "JAVA_SCHEMA_CATALOG_UNAVAILABLE"
    assert result["report"] is None


def test_untrusted_synthesis_output_is_replaced_by_deterministic_grounding() -> None:
    candidate = merge_unified_evidence(
        vector_results=[],
        graph_results=[_item("graph", 0.8, _path())],
        java_observations=[],
        limit=5,
    )[0]

    class OverreachingSynthesis:
        def synthesize(self, _context):
            return EvidenceSynthesisResult(
                claims=[GroundedClaim(
                    claimId="claim-" + "1" * 32,
                    statement="This claim is bound to an unknown evidence ID.",
                    supportStatus="supported",
                    evidenceIds=["evidence-" + "f" * 32],
                    reasoningPathIds=[],
                )],
                reasoningSteps=[],
                conclusion="An unsafe unbound conclusion.",
                mode="model",
            )

    result = _safe_synthesis_result(_task(), [candidate], OverreachingSynthesis())
    assert result.mode == "deterministic"
    assert result.fallbackCode == "GRAPHRAG_SYNTHESIS_OUTPUT_REJECTED"
    assert result.claims
    assert result.claims[0].evidenceIds == [candidate.candidateId]


def test_synthesis_context_downgrades_paths_for_uncertain_candidates() -> None:
    candidate = merge_unified_evidence(
        vector_results=[],
        graph_results=[_item("graph", 0.8, _path())],
        java_observations=[],
        limit=5,
    )[0].model_copy(update={"supportStatus": "speculative"})

    context = _synthesis_context(_task(), [candidate])

    assert context is not None
    assert context.evidence[0].reasoningPaths[0].status == "speculative"
