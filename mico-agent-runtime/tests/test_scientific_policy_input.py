from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx
import pytest

from mico_agent_runtime.contracts.decision_state import (
    ScientificDataState,
    ScientificDecisionState,
    ScientificTaskState,
)
from mico_agent_runtime.contracts.materialization import QueryPlan
from mico_agent_runtime.contracts.research import (
    AdjustConfoundersAction,
    CompareGroupsAction,
    CrossProjectValidateAction,
    ExecuteReadQueryAction,
    ExecuteReadQueryArguments,
    InspectCohortAction,
    InspectCohortArguments,
    ProjectionAnalysisArguments,
    RetrieveEvidenceAction,
    RetrieveEvidenceArguments,
    ResearchTask,
    ScientificPlannerContext,
    StratifiedAnalysisAction,
)
from mico_agent_runtime.contracts.scientific_policy import (
    ScientificPolicyInput,
    build_scientific_policy_input,
)
from mico_agent_runtime.graph.scientific_workflow import _plan_action, _understand_task, _validate_task
from mico_agent_runtime.ports.decision_policy import (
    DecisionPolicyOutput,
    HttpDecisionSftPlannerPort,
)
from mico_agent_runtime.ports.scientific_planner import ScientificPlannerResult
from mico_agent_runtime.ports.task_understanding import TaskUnderstandingResult
from mico_agent_runtime.contracts.task_understanding import TaskUnderstandingOutput


def _state(*, available: list[str] | None = None) -> ScientificDecisionState:
    return ScientificDecisionState(
        task=ScientificTaskState(
            query="比较 T2D 和 Healthy 的菌群差异",
            objectives=["group_comparison", "evidence_support"],
            constraints={"disease_groups": ["T2D", "Healthy"]},
        ),
        data_state=ScientificDataState(
            has_tabular_data=True,
            row_count=326,
            available_dimensions=["disease", "project", "age"],
            available_outcomes=["abundance.value"],
            group_state={
                "group_field": "disease",
                "group_count": 2,
                "group_sizes": {"T2D": 172, "Healthy": 154},
            },
            project_state={"has_project_field": True, "project_count": 7},
            covariate_state={
                "available_covariates": ["age"],
                "imbalance": {"age": "high"},
            },
        ),
        action_space={
            "available_actions": available or [
                "execute_read_query",
                "adjust_confounders",
                "cross_project_validate",
                "retrieve_evidence",
            ],
        },
    )


def test_policy_input_contains_only_six_decision_blocks() -> None:
    policy_input = build_scientific_policy_input(_state())
    payload = policy_input.model_dump(mode="json")
    assert isinstance(policy_input, ScientificPolicyInput)
    assert set(payload) == {
        "task", "data_state", "analysis_state", "evidence_state", "progress", "action_space",
    }
    serialized = json.dumps(payload, ensure_ascii=False)
    for forbidden in (
        "task_kind", "goal_code", "observation_flags", "candidate_actions",
        "history_actions", "state_summary", "METADATA_FIRST", "CROSS_PROJECT_REQUIRED",
    ):
        assert forbidden not in serialized
    assert "T2D" in serialized  # Policy sees the explicit task constraint, not raw rows.
    assert "rows" not in payload


def test_policy_input_changes_when_observation_facts_change() -> None:
    high = build_scientific_policy_input(_state())
    no_covariate = _state()
    no_covariate.data_state.covariate_state.available_covariates = []
    no_covariate.data_state.covariate_state.imbalance = {}
    changed = build_scientific_policy_input(no_covariate)
    assert high.model_dump(mode="json") != changed.model_dump(mode="json")
    assert high.data_state.covariate_state.imbalance == {"age": "high"}
    assert changed.data_state.covariate_state.imbalance == {}


def test_gemini_policy_receives_new_state_not_legacy_flags() -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps({
                "selected_action": "adjust_confounders",
                "decision_reason": "the covariate is available for adjustment",
                "alternative_actions": ["cross_project_validate", "retrieve_evidence"],
                "stop_reason": None,
            })}}],
        })

    planner = HttpDecisionSftPlannerPort(
        "http://qwen.local/v1",
        "qwen3-8b-decision-sft-v5",
        transport=httpx.MockTransport(handler),
    )
    result = planner.select_action(_state())

    assert result.selected_action == "adjust_confounders"
    assert len(seen) == 1
    user_payload = json.loads(seen[0]["messages"][1]["content"])
    assert user_payload["decision_type"] == "scientific_action"
    assert set(user_payload["state"]) == {
        "task", "data_state", "analysis_state", "evidence_state", "progress", "action_space",
    }
    request_text = json.dumps(seen[0], ensure_ascii=False)
    assert "objective names are not Scientific Action names" in request_text
    assert "copied exactly, character-for-character" in request_text
    for forbidden in (
        "task_kind", "goal_code", "observation_flags", "candidate_actions",
        "METADATA_FIRST", "CROSS_PROJECT_REQUIRED",
    ):
        assert forbidden not in request_text
    planner.close()


def test_unavailable_policy_action_is_repaired_without_runtime_substitution() -> None:
    calls: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps({
                "selected_action": "cross_project_validate",
                "decision_reason": "invalid action test",
                "alternative_actions": [],
                "stop_reason": None,
            })}}],
        })

    planner = HttpDecisionSftPlannerPort(
        "http://qwen.local/v1",
        "qwen3-8b-decision-sft-v5",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(RuntimeError, match="POLICY_ACTION_NOT_AVAILABLE"):
        planner.select_action(_state(available=["adjust_confounders", "retrieve_evidence"]))
    assert len(calls) == 2
    repair = calls[1]["messages"][-1]["content"]
    assert "Choose exactly one action copied character-for-character" in repair
    assert "Objective names are not action names" in repair
    assert "current available_actions list" in repair
    assert "adjust_confounders" in repair
    assert "retrieve_evidence" in repair
    planner.close()


class _FakePolicy:
    def select_action(self, state: ScientificPolicyInput) -> DecisionPolicyOutput:
        assert set(state.model_dump(mode="json")) == {
            "task", "data_state", "analysis_state", "evidence_state", "progress", "action_space",
        }
        return DecisionPolicyOutput(
            selected_action="execute_read_query",
            decision_reason="the available read capability is sufficient",
            alternative_actions=[],
            stop_reason=None,
        )


class _FakeMaterializer:
    def __init__(self) -> None:
        self.contexts: list[ScientificPlannerContext] = []

    def plan_action(self, context: ScientificPlannerContext) -> ScientificPlannerResult:
        self.contexts.append(context)
        return ScientificPlannerResult(
            action=ExecuteReadQueryAction(
                actionId="action-" + "a" * 32,
                actionName="execute_read_query",
                rationale="bounded read",
                arguments=ExecuteReadQueryArguments(
                    actionName="execute_read_query",
                    queryPlan=QueryPlan(
                        root_entity="sample",
                        select_fields=["sample.disease"],
                        limit=10,
                    ),
                    limit=10,
                ),
            ),
            mode="model",
        )


def test_new_policy_path_does_not_apply_legacy_strategy_guard() -> None:
    from mico_agent_runtime.ports.research_planner import HybridIntentPlannerPort

    materializer = _FakeMaterializer()
    hybrid = HybridIntentPlannerPort(materializer, _FakePolicy())
    decision_state = _state(available=["execute_read_query", "inspect_cohort", "finish"])
    context = ScientificPlannerContext(
        questionSummary="比较当前数据",
        intent="scientific_exploration",
        approvedActions=["execute_read_query", "inspect_cohort", "finish"],
        remainingActionBudget=4,
        observations=[],
    )
    result = hybrid.plan_action_with_state(
        context,
        decision_state,
    )
    assert result.action.actionName == "execute_read_query"
    assert materializer.contexts[0].approvedActions == ["execute_read_query"]
    assert materializer.contexts[0].decisionState == decision_state.model_dump(mode="json")
    assert "rawObservationPayloads" not in materializer.contexts[0].decisionState


def test_new_policy_trace_contains_state_snapshot_and_no_legacy_fields() -> None:
    class DynamicPlanner:
        dynamic_action_materialization = True
        prefer_policy_action = True

        def plan_action_with_state(
            self,
            _context: ScientificPlannerContext,
            policy_input: ScientificPolicyInput,
        ) -> ScientificPlannerResult:
            assert policy_input.task.query == "原始问题，不应被模型改写"
            return ScientificPlannerResult(
                action=InspectCohortAction(
                    actionId="action-" + "b" * 32,
                    actionName="inspect_cohort",
                    rationale="bounded read",
                    arguments=InspectCohortArguments(
                        actionName="inspect_cohort",
                        queryPlan=QueryPlan(
                            root_entity="sample",
                            select_fields=["sample.disease"],
                            limit=10,
                        ),
                        limit=10,
                    ),
                ),
                mode="sft_policy",
                rawAction="inspect_cohort",
                decisionReason="the read capability is currently executable",
            )

        def plan_action(self, _context: ScientificPlannerContext) -> ScientificPlannerResult:
            raise AssertionError("new policy path must not use legacy plan_action")

    task = ResearchTask(
        runId="run-policy-trace",
        taskId="task-policy-trace",
        requesterId="requester-policy-trace",
        traceId="trace-policy-trace",
        question="原始问题，不应被模型改写",
        requestedScopes=["mico:query:read", "mico:research:read"],
        allowedActions=["inspect_cohort", "finish"],
        maxActions=4,
        createdAt=datetime(2026, 8, 31, tzinfo=timezone.utc),
    )
    state = _validate_task({"request": task}, knowledge_available=False)
    state = _understand_task(
        type("TaskPort", (), {
            "understand": lambda self, _query: TaskUnderstandingResult(
                output=TaskUnderstandingOutput(objectives=["group_comparison"]),
                mode="model",
                model="gemini-3.5-flash-lite",
            )
        })(),
        state,
    )
    state["schemaCatalog"] = None
    state = _plan_action(DynamicPlanner(), state)

    decision = state["decisionRecords"][-1]
    assert decision.policy_input_version == "scientific-decision-state-v1"
    assert set(decision.state_snapshot or {}) == {
        "task", "data_state", "analysis_state", "evidence_state", "progress", "action_space",
    }
    assert decision.available_actions == ["inspect_cohort"]
    assert decision.task_kind is None
    assert decision.goal_code is None
    assert decision.observation_flags == []
    assert decision.candidate_actions == []
    assert decision.decision_reason == "the read capability is currently executable"


def test_two_round_policy_trace_rebuilds_input_after_state_change() -> None:
    snapshots: list[dict[str, object]] = []

    class TwoRoundPlanner:
        dynamic_action_materialization = True
        prefer_policy_action = True

        def plan_action_with_state(
            self,
            _context: ScientificPlannerContext,
            policy_input: ScientificPolicyInput,
        ) -> ScientificPlannerResult:
            snapshots.append(policy_input.model_dump(mode="json"))
            return ScientificPlannerResult(
                action=ExecuteReadQueryAction(
                    actionId="action-" + "c" * 32,
                    actionName="execute_read_query",
                    rationale="bounded read",
                    arguments=ExecuteReadQueryArguments(
                        actionName="execute_read_query",
                        queryPlan=QueryPlan(
                            root_entity="sample",
                            select_fields=["sample.disease"],
                            limit=10,
                        ),
                        limit=10,
                    ),
                ),
                mode="sft_policy",
                rawAction="execute_read_query",
                decisionReason="re-evaluate the changed state",
            )

        def plan_action(self, _context: ScientificPlannerContext) -> ScientificPlannerResult:
            raise AssertionError("new policy path must not use legacy plan_action")

    task = ResearchTask(
        runId="run-two-round",
        taskId="task-two-round",
        requesterId="requester-two-round",
        traceId="trace-two-round",
        question="比较两组并观察协变量状态变化",
        requestedScopes=["mico:query:read", "mico:research:read"],
        allowedActions=["execute_read_query", "finish"],
        maxActions=4,
        createdAt=datetime(2026, 8, 31, tzinfo=timezone.utc),
    )
    state = _validate_task({"request": task}, knowledge_available=False)
    state = _understand_task(
        type("TaskPort", (), {
            "understand": lambda self, _query: TaskUnderstandingResult(
                output=TaskUnderstandingOutput(objectives=["group_comparison"]),
                mode="model",
                model="gemini-3.5-flash-lite",
            )
        })(),
        state,
    )
    # Keep this controlled boundary test independent of Java; the second
    # round represents a newly observed covariate fact in Decision State.
    state["decisionState"].action_space.available_actions = ["execute_read_query"]
    state["decisionState"].data_state.covariate_state.available_covariates = ["age"]
    state["decisionState"].data_state.covariate_state.imbalance = {"age": "high"}
    state = _plan_action(TwoRoundPlanner(), state)
    state["decisionState"].data_state.covariate_state.imbalance = {"age": "low"}
    state = _plan_action(TwoRoundPlanner(), state)

    assert len(snapshots) == 2
    assert snapshots[0] != snapshots[1]
    assert snapshots[0]["data_state"]["covariate_state"]["imbalance"] == {"age": "high"}
    assert snapshots[1]["data_state"]["covariate_state"]["imbalance"] == {"age": "low"}
    assert len(state["decisionRecords"]) == 2
    assert all(
        item.policy_input_version == "scientific-decision-state-v1"
        for item in state["decisionRecords"]
    )


class _SelectedActionPolicy:
    def __init__(self, selected: str) -> None:
        self.selected = selected

    def select_action(self, _state: ScientificPolicyInput) -> DecisionPolicyOutput:
        return DecisionPolicyOutput(
            selected_action=self.selected,
            decision_reason=f"select {self.selected} for the current observation",
            alternative_actions=[],
            stop_reason=None,
        )


class _SelectedActionMaterializer:
    def __init__(self, selected: str, *, mode: str = "model") -> None:
        self.selected = selected
        self.mode = mode

    def plan_action(self, _context: ScientificPlannerContext) -> ScientificPlannerResult:
        action_id = "action-" + "d" * 32
        observation_id = "observation-" + "e" * 32
        if self.selected == "retrieve_evidence":
            action = RetrieveEvidenceAction(
                actionId=action_id,
                actionName="retrieve_evidence",
                rationale="retrieve the selected evidence branch",
                arguments=RetrieveEvidenceArguments(
                    actionName="retrieve_evidence",
                    topics=["the current research question"],
                    retrievalMode="hybrid",
                    topK=5,
                    maxHops=1,
                ),
            )
        else:
            arguments = ProjectionAnalysisArguments(
                actionName=self.selected,
                observationIds=[observation_id],
                analysisGoal="evaluate the selected scientific capability",
                dimensions=["sample_metadata.age"]
                if self.selected == "stratified_analysis" else [],
                confounders=["sample_metadata.age"]
                if self.selected == "adjust_confounders" else [],
            )
            action_type = {
                "stratified_analysis": StratifiedAnalysisAction,
                "adjust_confounders": AdjustConfoundersAction,
                "cross_project_validate": CrossProjectValidateAction,
                "compare_groups": CompareGroupsAction,
            }[self.selected]
            action = action_type(
                actionId=action_id,
                actionName=self.selected,
                rationale="materialize the selected scientific capability",
                arguments=arguments,
            )
        return ScientificPlannerResult(action=action, mode=self.mode)


def test_new_policy_can_select_retrieve_without_metadata_first_substitution() -> None:
    from mico_agent_runtime.ports.research_planner import HybridIntentPlannerPort

    planner = HybridIntentPlannerPort(
        _SelectedActionMaterializer("retrieve_evidence"),
        _SelectedActionPolicy("retrieve_evidence"),
    )
    context = ScientificPlannerContext(
        questionSummary="解释当前研究问题",
        intent="scientific_exploration",
        approvedActions=["inspect_cohort", "execute_read_query", "retrieve_evidence"],
        remainingActionBudget=4,
        observations=[],
    )
    result = planner.plan_action_with_state(
        context,
        _state(available=["inspect_cohort", "execute_read_query", "retrieve_evidence"]),
    )
    assert result.action.actionName == "retrieve_evidence"
    assert result.policyOrigin == "qwen_model"
    assert result.materializerOrigin == "model"


def test_new_policy_can_select_stratified_before_comparison() -> None:
    from mico_agent_runtime.ports.research_planner import HybridIntentPlannerPort

    planner = HybridIntentPlannerPort(
        _SelectedActionMaterializer("stratified_analysis"),
        _SelectedActionPolicy("stratified_analysis"),
    )
    context = ScientificPlannerContext(
        questionSummary="检查年龄分层差异",
        intent="focused_comparison",
        approvedActions=["compare_groups", "stratified_analysis"],
        remainingActionBudget=3,
        observations=[],
    )
    result = planner.plan_action_with_state(
        context,
        _state(available=["compare_groups", "stratified_analysis"]),
    )
    assert result.action.actionName == "stratified_analysis"


def test_new_policy_can_select_cross_project_without_forced_adjustment() -> None:
    from mico_agent_runtime.ports.research_planner import HybridIntentPlannerPort

    planner = HybridIntentPlannerPort(
        _SelectedActionMaterializer("cross_project_validate"),
        _SelectedActionPolicy("cross_project_validate"),
    )
    context = ScientificPlannerContext(
        questionSummary="验证项目间稳定性",
        intent="focused_comparison",
        approvedActions=["adjust_confounders", "cross_project_validate"],
        remainingActionBudget=3,
        observations=[],
    )
    result = planner.plan_action_with_state(
        context,
        _state(available=["adjust_confounders", "cross_project_validate"]),
    )
    assert result.action.actionName == "cross_project_validate"


def test_materializer_action_mismatch_fails_closed() -> None:
    from mico_agent_runtime.ports.research_planner import HybridIntentPlannerPort

    planner = HybridIntentPlannerPort(
        _SelectedActionMaterializer("compare_groups"),
        _SelectedActionPolicy("adjust_confounders"),
    )
    context = ScientificPlannerContext(
        questionSummary="控制协变量后评估差异",
        intent="focused_comparison",
        approvedActions=["adjust_confounders", "compare_groups"],
        remainingActionBudget=3,
        observations=[],
    )
    with pytest.raises(RuntimeError, match="MATERIALIZATION_ACTION_MISMATCH"):
        planner.plan_action_with_state(
            context,
            _state(available=["adjust_confounders", "compare_groups"]),
        )


def test_new_policy_keeps_action_when_materializer_uses_deterministic_fallback() -> None:
    from mico_agent_runtime.ports.research_planner import HybridIntentPlannerPort

    planner = HybridIntentPlannerPort(
        _SelectedActionMaterializer("retrieve_evidence", mode="deterministic"),
        _SelectedActionPolicy("retrieve_evidence"),
    )
    context = ScientificPlannerContext(
        questionSummary="补充证据",
        intent="scientific_exploration",
        approvedActions=["retrieve_evidence"],
        remainingActionBudget=2,
        observations=[],
    )
    result = planner.plan_action_with_state(
        context,
        _state(available=["retrieve_evidence"]),
    )
    assert result.action.actionName == "retrieve_evidence"
    assert result.policyOrigin == "qwen_model"
    assert result.materializerOrigin == "deterministic_fallback"


def test_new_policy_provider_failure_does_not_choose_a_deterministic_action() -> None:
    calls: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(503, json={"error": "unavailable"})

    planner = HttpDecisionSftPlannerPort(
        "http://qwen.local/v1",
        "qwen3-8b-decision-sft-v5",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(RuntimeError, match="POLICY_DECISION_FAILED"):
        planner.select_action(_state(available=["retrieve_evidence"]))
    assert len(calls) == 2
    planner.close()


def test_singleton_materializer_prompt_has_no_legacy_strategy_rules() -> None:
    calls: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps({
                "actionId": "action-" + "f" * 32,
                "actionName": "retrieve_evidence",
                "rationale": "retrieve the selected evidence branch",
                "arguments": {
                    "actionName": "retrieve_evidence",
                    "topics": ["the current research question"],
                    "retrievalMode": "hybrid",
                    "topK": 5,
                    "maxHops": 1,
                },
            })}}],
        })

    from mico_agent_runtime.ports.research_planner import HttpResearchPlannerPort

    planner = HttpResearchPlannerPort(
        "http://deepseek.local/v1",
        "deepseek-materializer",
        "test-token",
        transport=httpx.MockTransport(handler),
    )
    try:
        planner.plan_action(ScientificPlannerContext(
            questionSummary="补充证据",
            intent="scientific_exploration",
            approvedActions=["retrieve_evidence"],
            remainingActionBudget=2,
            observations=[],
        ))
    finally:
        planner.close()
    system_prompt = calls[0]["messages"][0]["content"]
    assert "Materialization-only mode" in system_prompt
    assert "metadata-first" not in system_prompt
    assert "Do not repeat a cross_project_validate" not in system_prompt


def test_singleton_query_materializer_prompt_restricts_aggregation_fields() -> None:
    calls: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps({
                "actionId": "action-" + "f" * 32,
                "actionName": "execute_read_query",
                "rationale": "read the requested numeric outcome",
                "arguments": {
                    "actionName": "execute_read_query",
                    "queryPlan": {
                        "root_entity": "sample",
                        "relation_path": [],
                        "select_fields": ["sample.disease"],
                        "aggregations": [],
                        "filters": [],
                        "group_by": [],
                        "limit": 10,
                    },
                    "limit": 10,
                },
            })}}],
        })

    from mico_agent_runtime.ports.research_planner import HttpResearchPlannerPort

    planner = HttpResearchPlannerPort(
        "http://gemini.local/v1",
        "gemini-canary",
        "test-token",
        transport=httpx.MockTransport(handler),
    )
    try:
        planner.plan_action(ScientificPlannerContext(
            questionSummary="覆盖两个疾病组",
            intent="scientific_exploration",
            approvedActions=["execute_read_query"],
            remainingActionBudget=2,
            observations=[],
            executionFeedback=["DYNAMIC_GROUP_COVERAGE_REQUIRED"],
            requiredSemanticFields=["sample.disease", "sample.value"],
            requiredGroupField="sample.disease",
        ))
    finally:
        planner.close()
    system_prompt = calls[0]["messages"][0]["content"]
    assert "aggregatable capability is true" in system_prompt
    assert "Never use a dimension, label, or metadata field" in system_prompt
    assert "A grouped count is represented by the returned grouped rows" in system_prompt
    assert "Technical coverage contract for this materialization" in system_prompt
    assert "requiredGroupField='sample.disease'" in system_prompt
