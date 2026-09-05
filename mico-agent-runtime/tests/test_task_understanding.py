from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx
import pytest
from pydantic import ValidationError

from mico_agent_runtime.contracts.decision_state import ScientificTaskConstraints
from mico_agent_runtime.contracts.research import ResearchTask
from mico_agent_runtime.contracts.task_understanding import TaskUnderstandingOutput
from mico_agent_runtime.graph.scientific_workflow import _understand_task, _validate_task
from mico_agent_runtime.ports.task_understanding import (
    DeterministicTaskUnderstandingPort,
    HttpTaskUnderstandingPort,
    TaskUnderstandingError,
    TaskUnderstandingResult,
    build_task_understanding_port,
)


def _task(question: str = "比较 T2D 和 Healthy 的菌群差异") -> ResearchTask:
    return ResearchTask(
        runId="run-task-understanding",
        taskId="task-task-understanding",
        requesterId="requester-task-understanding",
        traceId="trace-task-understanding",
        question=question,
        requestedScopes=["mico:query:read", "mico:research:read", "mico:evidence:read"],
        allowedActions=[
            "inspect_cohort", "execute_read_query", "compare_groups",
            "adjust_confounders", "cross_project_validate", "stratified_analysis",
            "retrieve_evidence", "finish",
        ],
        maxActions=6,
        createdAt=datetime(2026, 8, 31, tzinfo=timezone.utc),
    )


def test_task_understanding_output_round_trip_is_closed() -> None:
    output = TaskUnderstandingOutput(
        objectives=["group_comparison", "confounder_assessment"],
        constraints=ScientificTaskConstraints(
            disease_groups=["T2D", "Healthy"],
            focus_covariates=["age", "gender"],
        ),
    )
    restored = TaskUnderstandingOutput.model_validate_json(output.model_dump_json())
    assert restored == output


def test_task_understanding_rejects_action_hints_and_unknown_objectives() -> None:
    with pytest.raises(ValidationError):
        TaskUnderstandingOutput.model_validate({
            "objectives": ["group_comparison"],
            "constraints": {},
            "next_action": "compare_groups",
        })
    with pytest.raises(ValidationError):
        TaskUnderstandingOutput.model_validate({
            "objectives": ["execute_read_query"],
            "constraints": {},
        })


def test_deterministic_fallback_never_infers_from_keywords() -> None:
    result = DeterministicTaskUnderstandingPort().understand(
        "比较 T2D 和 Healthy，判断年龄是否影响，并查文献"
    )
    assert result.mode == "deterministic"
    assert result.output.objectives == []
    assert result.output.constraints == ScientificTaskConstraints()
    assert result.fallbackCode == "TASK_UNDERSTANDING_DETERMINISTIC_FALLBACK"


def test_gemini_adapter_uses_original_query_and_fixed_model() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        payload = json.loads(request.content)
        assert request.url == (
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        )
        assert payload["model"] == "gemini-3.5-flash-lite"
        assert payload["messages"][1]["content"] == "原始问题：比较 T2D 和 Healthy"
        assert "group_comparison" in payload["messages"][0]["content"]
        assert "Never output an action name" in payload["messages"][0]["content"]
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps({
                "objectives": ["group_comparison"],
                "constraints": {"disease_groups": ["T2D", "Healthy"]},
            })}}],
        })

    port = HttpTaskUnderstandingPort(
        token="test-token",
        transport=httpx.MockTransport(handler),
    )
    result = port.understand("原始问题：比较 T2D 和 Healthy")

    assert result.mode == "model"
    assert result.model == "gemini-3.5-flash-lite"
    assert result.output.objectives == ["group_comparison"]
    assert result.output.constraints.disease_groups == ["T2D", "Healthy"]
    assert len(calls) == 1
    port.close()


def test_gemini_adapter_retries_once_with_contract_feedback() -> None:
    calls: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(payload)
        if len(calls) == 1:
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "not json"}}],
            })
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps({
                "objectives": ["evidence_support"],
                "constraints": {},
            })}}],
        })

    port = HttpTaskUnderstandingPort(
        token="test-token",
        transport=httpx.MockTransport(handler),
    )
    result = port.understand("查找相关文献")

    assert result.output.objectives == ["evidence_support"]
    assert len(calls) == 2
    assert "Retry once" in calls[1]["messages"][-1]["content"]
    assert "Validation feedback:" in calls[1]["messages"][-1]["content"]
    port.close()


def test_deepseek_task_understanding_uses_native_endpoint_and_disables_reasoning() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        payload = json.loads(request.content)
        assert request.url == "https://api.deepseek.com/chat/completions"
        assert payload["model"] == "deepseek-v4-flash"
        assert payload["thinking"] == {"type": "disabled"}
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps({
                "objectives": ["group_comparison"], "constraints": {},
            })}}],
        })

    port = HttpTaskUnderstandingPort(
        base_url="https://api.deepseek.com",
        model="deepseek-v4-flash",
        token="test-token",
        transport=httpx.MockTransport(handler),
    )
    result = port.understand("compare groups")

    assert result.mode == "model"
    assert result.output.objectives == ["group_comparison"]
    assert len(calls) == 1
    port.close()


def test_gemini_adapter_fails_after_bounded_retry() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "still not json"}}],
        })

    port = HttpTaskUnderstandingPort(
        token="test-token",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(TaskUnderstandingError) as error:
        port.understand("无效输出测试")
    assert error.value.code == "TASK_UNDERSTANDING_FAILED"
    port.close()


def test_build_port_defaults_to_explicit_deterministic_fallback() -> None:
    port = build_task_understanding_port({})
    assert isinstance(port, DeterministicTaskUnderstandingPort)


def test_build_port_enables_gemini_flash_lite_from_environment() -> None:
    port = build_task_understanding_port({
        "MICO_TASK_UNDERSTANDING_ENABLED": "true",
        "MICO_GEMINI_API_KEY": "test-token",
    })
    assert isinstance(port, HttpTaskUnderstandingPort)
    assert port.model == "gemini-3.5-flash-lite"
    port.close()


def test_build_port_accepts_extended_timeout_from_environment() -> None:
    port = build_task_understanding_port({
        "MICO_TASK_UNDERSTANDING_ENABLED": "true",
        "MICO_GEMINI_API_KEY": "test-token",
        "MICO_TASK_UNDERSTANDING_TIMEOUT_SECONDS": "300",
    })
    assert isinstance(port, HttpTaskUnderstandingPort)
    assert port._client.timeout.read == 300.0
    port.close()


class _StubTaskUnderstandingPort:
    def understand(self, _query: str) -> TaskUnderstandingResult:
        return TaskUnderstandingResult(
            output=TaskUnderstandingOutput(
                objectives=[
                    "group_comparison",
                    "confounder_assessment",
                    "cross_project_validation",
                    "evidence_support",
                ],
                constraints=ScientificTaskConstraints(
                    disease_groups=["T2D", "Healthy"],
                    focus_covariates=["age"],
                ),
            ),
            mode="model",
            model="gemini-3.5-flash-lite",
        )


def test_understand_task_preserves_query_and_refreshes_availability() -> None:
    original = "比较 T2D 和 Healthy 的菌群差异，判断年龄是否影响，并查文献"
    state = _validate_task({"request": _task(original)}, knowledge_available=True)
    assert state["decisionState"].task.objectives == []

    updated = _understand_task(_StubTaskUnderstandingPort(), state)

    decision_state = updated["decisionState"]
    assert decision_state.task.query == original
    assert decision_state.task.objectives == [
        "group_comparison",
        "confounder_assessment",
        "cross_project_validation",
        "evidence_support",
    ]
    assert decision_state.task.constraints.focus_covariates == ["age"]
    assert "finish" not in decision_state.action_space.available_actions
    assert updated["taskUnderstandingMode"] == "model"
    assert updated["taskUnderstandingModel"] == "gemini-3.5-flash-lite"


@pytest.mark.parametrize(
    ("query", "objectives", "constraints"),
    [
        (
            "比较 T2D 和 Healthy",
            ["group_comparison"],
            {"disease_groups": ["T2D", "Healthy"]},
        ),
        (
            "比较两组并控制 age 和 gender",
            ["group_comparison", "confounder_assessment"],
            {"focus_covariates": ["age", "gender"]},
        ),
        (
            "按 gender 分层比较",
            ["group_comparison", "stratified_analysis"],
            {"requested_stratifiers": ["gender"]},
        ),
        (
            "判断不同项目是否稳定",
            ["cross_project_validation"],
            {},
        ),
        (
            "查 Bacteroides 与 T2D 的文献",
            ["evidence_support"],
            {"target_features": ["Bacteroides"], "disease_groups": ["T2D"]},
        ),
        (
            "比较两组并结合文献解释",
            ["group_comparison", "evidence_support"],
            {},
        ),
    ],
)
def test_standard_task_understanding_examples_are_closed(
    query: str,
    objectives: list[str],
    constraints: dict[str, list[str]],
) -> None:
    output = TaskUnderstandingOutput.model_validate({
        "objectives": objectives,
        "constraints": constraints,
    })
    assert output.objectives == objectives
    assert output.model_dump(mode="json")["constraints"] | {}  # serializable contract
    assert query
