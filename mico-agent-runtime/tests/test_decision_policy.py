import json

import httpx

from mico_agent_runtime.contracts.research import (
    ScientificObservationSummary,
    ScientificPlannerContext,
)
from mico_agent_runtime.ports.decision_policy import (
    HttpDecisionSftPlannerPort,
    build_decision_policy_state,
)
from mico_agent_runtime.ports.research_planner import (
    HybridIntentPlannerPort,
    build_intent_planner,
)
from mico_agent_runtime.ports.scientific_planner import DeterministicScientificPlanner


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
        fallback=DeterministicScientificPlanner(),
    )
    result = planner.plan_action(_context(approved_actions=["finish"]))
    assert result.mode == "sft_policy"
    assert result.action.actionName == "finish"
    assert result.action.arguments.reasonCode == "QUALITY_RISK"
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
        fallback=DeterministicScientificPlanner(),
    )
    result = planner.plan_action(_context(approved_actions=["retrieve_evidence", "finish"]))
    assert result.mode == "sft_policy"
    assert result.action.actionName == "retrieve_evidence"
    assert len(requests) == 4
    assert any("Schema-only feedback:" in message["content"] for message in requests[-1]["messages"])


def test_sft_policy_rejects_invalid_output_and_uses_safe_fallback() -> None:
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(200, json={"choices": [{"message": {"content": "not-json"}}]})

    planner = HttpDecisionSftPlannerPort(
        "http://sft.local:9000",
        "qwen3-8b-decision-sft-v2",
        transport=httpx.MockTransport(handler),
        fallback=DeterministicScientificPlanner(),
    )
    result = planner.plan_action(_context(approved_actions=["finish"]))
    assert attempts == 4
    assert result.action.actionName == "finish"
    assert result.mode == "deterministic"
    assert "SFT_POLICY_SAFE_FALLBACK" in (result.fallbackCode or "")


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
