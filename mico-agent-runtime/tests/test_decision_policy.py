import json

import httpx
import pytest

from mico_agent_runtime.contracts.generated_analysis import AnalysisPlannerContext
from mico_agent_runtime.contracts.research import (
    ExecuteReadQueryAction,
    ExecuteReadQueryArguments,
    ScientificObservationSummary,
    ScientificPlannerContext,
)
from mico_agent_runtime.contracts.materialization import QueryPlan
from mico_agent_runtime.ports.decision_policy import (
    DecisionPolicyOutput,
    HttpDecisionSftPlannerPort,
    build_decision_policy_state,
)
from mico_agent_runtime.ports.research_planner import (
    HybridIntentPlannerPort,
    _canonicalize_query_aggregation_aliases,
    HttpResearchPlannerPort,
    build_intent_planner,
)
from mico_agent_runtime.ports.scientific_planner import ScientificPlannerResult


def _context(*, approved_actions: list[str], question: str = "探索当前研究证据") -> ScientificPlannerContext:
    return ScientificPlannerContext(
        questionSummary=question,
        intent="scientific_exploration",
        approvedActions=approved_actions,
        remainingActionBudget=4,
        observations=[],
    )


def _observation(action: str, index: int) -> ScientificObservationSummary:
    return ScientificObservationSummary(
        observationId="observation-" + f"{index:032x}",
        actionName=action,
        status="VALIDATED",
        source="java_controlled_read",
        rowCount=1,
    )


def test_materializer_query_aggregation_alias_is_canonicalized_without_widening_contract() -> None:
    payload = {
        "actionName": "execute_read_query",
        "rationale": "read a bounded projection",
        "arguments": {
            "actionName": "execute_read_query",
            "queryPlan": {
                "root_entity": "sample",
                "select_fields": ["sample.gender"],
                "aggregations": [{"field": "sample.age", "function": "mean"}],
                "group_by": ["sample.gender"],
                "limit": 10,
            },
        },
    }
    normalized = _canonicalize_query_aggregation_aliases(payload)
    assert normalized["arguments"]["queryPlan"]["aggregations"] == [
        {"field": "sample.age", "op": "mean"},
    ]


def test_stratified_typed_prompt_binds_one_closed_v2_schema() -> None:
    requests: list[dict] = []
    content = json.dumps({
        "schemaVersion": "analysis-plan-v2",
        "analysis_type": "stratified_comparison",
        "source_observation_ids": ["observation-" + "c" * 32],
        "outcome": "abundance.value",
        "feature_field": "abundance.feature",
        "group_field": "sample.disease",
        "covariates": [],
        "stratify_by": ["sample.age"],
        "validation_field": None,
        "method": {"family": "auto", "name": None, "parameters": {}},
        "analysis_goal": "compare age strata",
        "metrics": ["effect_size"],
    })

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    planner = HttpResearchPlannerPort(
        "http://materializer.local:9000/v1",
        "gemini-3.5-flash-lite",
        "test-token",
        transport=httpx.MockTransport(handler),
    )
    try:
        result = planner.generate_typed_analysis(AnalysisPlannerContext(
            questionSummary="比较不同年龄层的菌群差异",
            workflow="stratified_analysis",
            actionName="stratified_analysis",
            sourceObservationIds=["observation-" + "c" * 32],
            availableSemanticFields=[
                "sample.disease", "sample.age", "abundance.feature", "abundance.value",
            ],
            columns=["a_sample_disease", "a_sample_age", "a_abundance_feature", "a_abundance_value"],
            previewRows=[],
        ))
    finally:
        planner.close()

    assert result.mode == "model"
    assert result.plan.analysis_type == "stratified_comparison"
    assert result.plan.stratify_by == ["sample.age"]
    system = requests[0]["messages"][0]["content"]
    assert "schemaVersion, analysis_type, source_observation_ids" in system
    assert "stratified_analysis maps only to analysis_type=stratified_comparison" in system
    assert "stratify_by:[<stratifier-dimension>]" in system
    assert "Capability Registry owns execution choice" in system


def test_stratified_typed_repair_prompt_is_schema_only() -> None:
    responses = [
        json.dumps({
            "language": "python",
            "analysisType": "stratified_analysis",
            "code": "result = {'metrics': {'count': len(rows)}, 'topFeatures': []}",
        }),
        json.dumps({
            "schemaVersion": "analysis-plan-v2",
            "analysis_type": "stratified_comparison",
            "source_observation_ids": ["observation-" + "d" * 32],
            "outcome": "abundance.value",
            "feature_field": "abundance.feature",
            "group_field": "sample.disease",
            "covariates": [],
            "stratify_by": ["sample.age"],
            "validation_field": None,
            "method": {"family": "auto", "name": None, "parameters": {}},
            "analysis_goal": "compare age strata",
            "metrics": ["effect_size"],
        }),
    ]
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": responses.pop(0)}}]})

    planner = HttpResearchPlannerPort(
        "http://materializer.local:9000/v1",
        "gemini-3.5-flash-lite",
        "test-token",
        transport=httpx.MockTransport(handler),
    )
    try:
        result = planner.generate_typed_analysis(AnalysisPlannerContext(
            questionSummary="比较不同年龄层的菌群差异",
            workflow="stratified_analysis",
            actionName="stratified_analysis",
            sourceObservationIds=["observation-" + "d" * 32],
            availableSemanticFields=[
                "sample.disease", "sample.age", "abundance.feature", "abundance.value",
            ],
            columns=["a_sample_disease", "a_sample_age", "a_abundance_feature", "a_abundance_value"],
            previewRows=[],
        ))
    finally:
        planner.close()

    assert result.mode == "model"
    assert len(requests) == 2
    repair = requests[1]["messages"][-1]["content"]
    assert "AnalysisPlan v2" in repair
    assert "required analysis_type='stratified_comparison'" in repair
    assert "Do not return a ScientificAction wrapper, code" in repair
    assert "change the selected Action" in repair


def test_projection_action_repair_prompt_uses_singular_observation_id() -> None:
    observation_id = "observation-" + "e" * 32
    responses = [
        json.dumps({
            "actionId": "action-" + "f" * 32,
            "actionName": "analyze_projection",
            "rationale": "summarize the validated projection",
            "arguments": {
                "actionName": "analyze_projection",
                "observationIds": [observation_id],
            },
        }),
        json.dumps({
            "actionId": "action-" + "f" * 32,
            "actionName": "analyze_projection",
            "rationale": "summarize the validated projection",
            "arguments": {
                "actionName": "analyze_projection",
                "observationId": observation_id,
                "analysisGoal": "summarize the validated projection",
            },
        }),
    ]
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": responses.pop(0)}}]})

    planner = HttpResearchPlannerPort(
        "http://materializer.local:9000/v1",
        "gemini-3.5-flash-lite",
        "test-token",
        transport=httpx.MockTransport(handler),
    )
    try:
        result = planner.plan_action(ScientificPlannerContext(
            questionSummary="分析已验证的菌群投影",
            intent="scientific_exploration",
            approvedActions=["analyze_projection"],
            remainingActionBudget=2,
            observations=[_observation("execute_read_query", 6)],
        ))
    finally:
        planner.close()

    assert result.mode == "model"
    assert result.action.actionName == "analyze_projection"
    assert result.action.arguments.observationId == observation_id
    assert len(requests) == 2
    repair = requests[1]["messages"][-1]["content"]
    assert "one singular observationId" in repair
    assert "never use observationIds" in repair


def test_policy_state_is_deidentified_and_closed() -> None:
    context = _context(
        approved_actions=["inspect_cohort", "retrieve_evidence", "finish"],
        question="T2D 原始问题不应跨越 policy endpoint",
    )
    state = build_decision_policy_state(context)
    assert state["task_kind"] == "open_exploration"
    assert state["goal_code"] == "scientific_exploration"
    assert state["observation_flags"] == ["NO_OBSERVATION", "METADATA_FIRST"]
    serialized = json.dumps(state, ensure_ascii=False)
    assert "T2D" not in serialized
    assert set(state) == {
        "task_kind",
        "goal_code",
        "observation_flags",
        "history_actions",
        "candidate_actions",
        "state_summary",
    }


def test_policy_state_exposes_next_obligation_after_validated_observation() -> None:
    read_state = build_decision_policy_state(ScientificPlannerContext(
        questionSummary="统计当前项目的样本覆盖记录",
        intent="data_fact",
        approvedActions=["inspect_cohort", "execute_read_query", "finish"],
        remainingActionBudget=3,
        observations=[_observation("inspect_cohort", 1)],
    ))
    assert read_state["goal_code"] == "cohort_fact"
    assert "BOUNDED_READ_REQUIRED" in read_state["observation_flags"]

    analysis_state = build_decision_policy_state(ScientificPlannerContext(
        questionSummary="比较两组的微生物差异",
        intent="focused_comparison",
        approvedActions=["compare_groups", "analyze_projection", "finish"],
        remainingActionBudget=3,
        observations=[_observation("inspect_cohort", 2), _observation("compare_groups", 3)],
    ))
    assert "ANALYSIS_REQUIRED" in analysis_state["observation_flags"]

    project_state = build_decision_policy_state(ScientificPlannerContext(
        questionSummary="验证跨 project 的稳定性",
        intent="scientific_exploration",
        approvedActions=["cross_project_validate", "retrieve_evidence", "finish"],
        remainingActionBudget=3,
        observations=[_observation("inspect_cohort", 4), _observation("compare_groups", 5)],
    ))
    assert "CROSS_PROJECT_REQUIRED" in project_state["observation_flags"]

    adjusted_open_state = build_decision_policy_state(ScientificPlannerContext(
        questionSummary="探索混杂控制后的稳定性",
        intent="scientific_exploration",
        approvedActions=["cross_project_validate", "retrieve_evidence", "finish"],
        remainingActionBudget=3,
        observations=[
            _observation("inspect_cohort", 40),
            _observation("compare_groups", 41),
            _observation("adjust_confounders", 42),
        ],
    ))
    assert "CROSS_PROJECT_REQUIRED" in adjusted_open_state["observation_flags"]

    replicated_adjusted_open_state = build_decision_policy_state(ScientificPlannerContext(
        questionSummary="探索混杂控制后的稳定性",
        intent="scientific_exploration",
        approvedActions=["retrieve_evidence", "finish"],
        remainingActionBudget=2,
        observations=[
            _observation("inspect_cohort", 43),
            _observation("compare_groups", 44),
            _observation("adjust_confounders", 45),
            _observation("cross_project_validate", 46),
        ],
    ))
    assert "EVIDENCE_REQUIRED" in replicated_adjusted_open_state["observation_flags"]

    disease_validated_state = build_decision_policy_state(ScientificPlannerContext(
        questionSummary="验证疾病特异性并补充证据",
        intent="scientific_exploration",
        approvedActions=["retrieve_evidence", "finish"],
        remainingActionBudget=2,
        observations=[
            _observation("inspect_cohort", 47),
            _observation("compare_groups", 48),
            _observation("analyze_projection", 49),
            _observation("cross_disease_validate", 50),
        ],
    ))
    assert "EVIDENCE_REQUIRED" in disease_validated_state["observation_flags"]

    focused_adjusted_state = build_decision_policy_state(ScientificPlannerContext(
        questionSummary="控制混杂因素后进行确定性比较",
        intent="focused_comparison",
        approvedActions=["cross_project_validate", "analyze_projection", "finish"],
        remainingActionBudget=3,
        observations=[
            _observation("inspect_cohort", 51),
            _observation("compare_groups", 52),
            _observation("adjust_confounders", 53),
        ],
    ))
    assert "CROSS_PROJECT_REQUIRED" not in focused_adjusted_state["observation_flags"]

    focused_completed_state = build_decision_policy_state(ScientificPlannerContext(
        questionSummary="比较两组的微生物差异",
        intent="focused_comparison",
        approvedActions=["retrieve_evidence", "finish"],
        remainingActionBudget=2,
        observations=[
            _observation("inspect_cohort", 54),
            _observation("compare_groups", 55),
            _observation("analyze_projection", 56),
        ],
    ))
    assert focused_completed_state["task_kind"] == "focused_analysis"
    assert focused_completed_state["goal_code"] == "group_comparison"
    assert "EVIDENCE_REQUIRED" not in focused_completed_state["observation_flags"]
    assert "ANALYSIS_REQUIRED" not in focused_completed_state["observation_flags"]

    completed_state = build_decision_policy_state(ScientificPlannerContext(
        questionSummary="检索并核实当前分析证据",
        intent="literature_or_relationship",
        approvedActions=["retrieve_evidence", "finish"],
        remainingActionBudget=2,
        observations=[
            _observation("analyze_projection", 6),
            ScientificObservationSummary(
                observationId="observation-" + "7" * 32,
                actionName="retrieve_evidence",
                status="VALIDATED",
                source="knowledge_hybrid",
                rowCount=1,
            ),
        ],
    ))
    assert "ANALYSIS_COMPLETE" in completed_state["observation_flags"]
    assert "EVIDENCE_GROUNDED" in completed_state["observation_flags"]


def test_sft_policy_accepts_closed_finish_output() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read())
        seen.append(body)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps({
                "selected_action": "finish",
                "decision_reason": "the unresolved evidence quality risk requires a bounded stop",
                "alternative_actions": [],
                "stop_reason": "QUALITY_RISK",
            })}}],
        })

    planner = HttpDecisionSftPlannerPort(
        "http://sft.local:9000/v1",
        "qwen3-8b-decision-sft-v2",
        transport=httpx.MockTransport(handler),
    )
    result = planner.select_action(_context(approved_actions=["finish"]))
    assert result.selected_action == "finish"
    assert result.stop_reason == "QUALITY_RISK"
    assert len(seen) == 1
    assert "T2D" not in json.dumps(seen[0], ensure_ascii=False)


def test_sft_policy_retries_three_times_with_schema_feedback() -> None:
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read())
        requests.append(body)
        content = "not-json" if len(requests) < 4 else json.dumps({
            "selected_action": "retrieve_evidence",
            "decision_reason": "add bounded evidence after the observation state",
            "alternative_actions": ["finish"],
            "stop_reason": None,
        })
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    planner = HttpDecisionSftPlannerPort(
        "http://sft.local:9000",
        "qwen3-8b-decision-sft-v2",
        transport=httpx.MockTransport(handler),
    )
    result = planner.select_action(_context(approved_actions=["retrieve_evidence", "finish"]))
    assert result.selected_action == "retrieve_evidence"
    assert len(requests) == 4
    assert any("Schema-only feedback:" in message["content"] for message in requests[-1]["messages"])


def test_sft_policy_rejects_invalid_output_without_deterministic_materialization() -> None:
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(200, json={"choices": [{"message": {"content": "not-json"}}]})

    planner = HttpDecisionSftPlannerPort(
        "http://sft.local:9000",
        "qwen3-8b-decision-sft-v2",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(RuntimeError, match="DYNAMIC_ACTION_SELECTION_FAILED"):
        planner.select_action(_context(approved_actions=["finish"]))
    assert attempts == 4


def test_hybrid_materializes_qwen_read_choice_with_dynamic_research_planner() -> None:
    class DynamicReadMaterializer:
        def __init__(self) -> None:
            self.contexts: list[ScientificPlannerContext] = []

        def plan_action(self, context: ScientificPlannerContext) -> ScientificPlannerResult:
            self.contexts.append(context)
            return ScientificPlannerResult(
                action=ExecuteReadQueryAction(
                    actionId="action-" + "a" * 32,
                    actionName="execute_read_query",
                    rationale="dynamic question-specific bounded read",
                    arguments=ExecuteReadQueryArguments(
                        actionName="execute_read_query",
                        queryPlan=QueryPlan(
                            root_entity="sample",
                            select_fields=["sample.disease"],
                            limit=25,
                        ),
                        limit=25,
                    ),
                ),
                mode="model",
            )

    def policy_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps({
                "selected_action": "execute_read_query",
                "decision_reason": "retrieve the project-specific bounded data required next",
                "alternative_actions": ["finish"],
                "stop_reason": None,
            })}}],
        })

    materializer = DynamicReadMaterializer()
    policy = HttpDecisionSftPlannerPort(
        "http://sft.local:9000/v1",
        "qwen3-8b-decision-sft-v5-dpo-v4-controlled",
        transport=httpx.MockTransport(policy_handler),
    )
    planner = HybridIntentPlannerPort(materializer, policy)
    result = planner.plan_action(_context(approved_actions=["execute_read_query", "finish"]))

    assert result.mode == "sft_policy"
    assert result.action.actionName == "execute_read_query"
    assert result.action.arguments.queryPlan.limit == 25
    assert materializer.contexts[0].approvedActions == ["execute_read_query"]


def test_hybrid_rejects_a_static_materializer_result() -> None:
    class StaticMaterializer:
        def plan_action(self, _context: ScientificPlannerContext) -> ScientificPlannerResult:
            return ScientificPlannerResult(
                action=ExecuteReadQueryAction(
                    actionId="action-" + "b" * 32,
                    actionName="execute_read_query",
                    rationale="static catalog template",
                    arguments=ExecuteReadQueryArguments(
                        actionName="execute_read_query",
                        sql="SELECT project_name FROM meta2db_sample_metadata LIMIT 25",
                        limit=25,
                    ),
                ),
                mode="deterministic",
            )

    class FixedPolicy:
        def select_action(self, _context: ScientificPlannerContext) -> DecisionPolicyOutput:
            return DecisionPolicyOutput(
                selected_action="execute_read_query",
                decision_reason="read the next bounded observation",
                alternative_actions=["finish"],
                stop_reason=None,
            )

    planner = HybridIntentPlannerPort(StaticMaterializer(), FixedPolicy())
    with pytest.raises(RuntimeError, match="DYNAMIC_ACTION_MATERIALIZATION_FAILED"):
        planner.plan_action(_context(approved_actions=["execute_read_query", "finish"]))


def test_research_materializer_emits_closed_typed_analysis_plan() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps({
                "analysis_type": "group_comparison",
                "source_observation_ids": ["observation-" + "c" * 32],
                "outcome": "abundance.value",
                "group_field": "sample.gender",
                "covariates": [],
                "stratify_by": [],
                "metrics": ["count", "mean", "effect_size"],
            })}}],
        })

    from mico_agent_runtime.ports.research_planner import HttpResearchPlannerPort

    planner = HttpResearchPlannerPort(
        "http://materializer.local:9000/v1",
        "deepseek-v4-flash",
        "test-token",
        transport=httpx.MockTransport(handler),
    )
    result = planner.generate_typed_analysis(AnalysisPlannerContext(
        questionSummary="比较已验证分组的有界结果",
        workflow="compare_groups",
        actionName="compare_groups",
        sourceObservationIds=["observation-" + "c" * 32],
        availableSemanticFields=["abundance.value", "sample.gender"],
        columns=["a_abundance_value", "a_sample_gender"],
        previewRows=[],
    ))
    assert result.mode == "model"
    assert result.plan.analysis_type == "group_comparison"
    assert result.plan.outcome == "abundance.value"


def test_research_materializer_context_does_not_send_physical_catalog_names() -> None:
    captured: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request.content.decode("utf-8"))
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({
            "actionId": "action-" + "d" * 32,
            "actionName": "inspect_cohort",
            "rationale": "inspect semantic fields",
            "arguments": {
                "actionName": "inspect_cohort",
                "queryPlan": {
                    "root_entity": "sample_metadata",
                    "select_fields": ["sample_metadata.gender"],
                    "limit": 10,
                },
                "limit": 10,
            },
        })}}]})

    from mico_agent_runtime.ports.research_planner import HttpResearchPlannerPort
    from tests.test_scientific_agent_loop import _semantic_catalog

    planner = HttpResearchPlannerPort(
        "http://materializer.local:9000/v1",
        "deepseek-v4-flash",
        "test-token",
        transport=httpx.MockTransport(handler),
    )
    context = _context(approved_actions=["inspect_cohort"]).model_copy(update={
        "schemaCatalog": _semantic_catalog(),
    })
    planner.plan_action(context)

    assert captured
    assert "sourceTable" not in captured[0]
    assert "meta2db_sample_metadata" not in captured[0]
    assert "fieldId" in captured[0]


def test_sft_planner_is_opt_in_and_wraps_existing_planner() -> None:
    planner = build_intent_planner({
        "MICO_RESEARCH_PLANNER_BASE_URL": "http://base.local/v1",
        "MICO_RESEARCH_PLANNER_MODEL": "base-model",
        "MICO_RESEARCH_PLANNER_TOKEN": "base-token",
        "MICO_SFT_POLICY_ENABLED": "true",
        "MICO_SFT_POLICY_BASE_URL": "http://sft.local:9000/v1",
        "MICO_SFT_POLICY_MODEL": "qwen3-8b-decision-sft-v2",
    })
    assert isinstance(planner, HybridIntentPlannerPort)


def test_sft_policy_requires_dynamic_research_planner_configuration() -> None:
    with pytest.raises(ValueError, match="DYNAMIC_RESEARCH_PLANNER_CONFIG_REQUIRED"):
        build_intent_planner({
            "MICO_SFT_POLICY_ENABLED": "true",
            "MICO_SFT_POLICY_BASE_URL": "http://sft.local:9000/v1",
            "MICO_SFT_POLICY_MODEL": "qwen3-8b-decision-sft-v5-dpo-v4-controlled",
        })
