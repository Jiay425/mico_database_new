import httpx
import pytest
import json

from mico_agent_runtime.ports.gemini_resilience import (
    GeminiRequestBudget,
    GeminiRequestBudgetExceeded,
    GeminiResponseCache,
    parse_retry_delay_seconds,
    sleep_before_retry,
)
from mico_agent_runtime.ports.task_understanding import HttpTaskUnderstandingPort


def _quota_response(delay: str = "49s") -> httpx.Response:
    return httpx.Response(
        429,
        json={
            "error": {
                "status": "RESOURCE_EXHAUSTED",
                "details": [{
                    "@type": "type.googleapis.com/google.rpc.RetryInfo",
                    "retryDelay": delay,
                }],
            }
        },
    )


def test_parse_gemini_retry_info_delay() -> None:
    assert parse_retry_delay_seconds(_quota_response("49s")) == 49.0


def test_sleep_before_retry_honors_delay_and_optional_cap() -> None:
    calls: list[float] = []

    class Error(RuntimeError):
        response = _quota_response("49s")
        retry_delay_seconds = 49.0

    assert sleep_before_retry(Error(), sleep=calls.append, max_delay_seconds=2.5) == 2.5
    assert calls == [2.5]


def test_request_budget_enforces_role_and_total_limits() -> None:
    budget = GeminiRequestBudget(total_limit=2, role_limits={"policy": 1})
    budget.reserve("policy")
    with pytest.raises(GeminiRequestBudgetExceeded):
        budget.reserve("policy")
    budget.reserve("materializer")
    with pytest.raises(GeminiRequestBudgetExceeded):
        budget.reserve("task_understanding")


def test_response_cache_only_reuses_successful_identical_payload() -> None:
    cache = GeminiResponseCache(ttl_seconds=30)
    endpoint = "https://example.test/v1/chat/completions"
    body = {"model": "gemini", "messages": [{"role": "user", "content": "x"}]}
    request = httpx.Request("POST", endpoint)
    key = cache.key(endpoint, body)
    assert cache.get(key, request) is None
    cache.put(key, httpx.Response(429, json={"error": {"status": "RESOURCE_EXHAUSTED"}}))
    assert cache.get(key, request) is None
    cache.put(key, httpx.Response(200, json={"choices": []}))
    cached = cache.get(key, request)
    assert cached is not None
    assert cached.status_code == 200
    assert cached.json() == {"choices": []}


def test_task_understanding_waits_for_quota_retry_delay_before_retrying() -> None:
    calls = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _quota_response("3s")
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps({
                "objectives": ["group_comparison"], "constraints": {},
            })}}],
        })

    port = HttpTaskUnderstandingPort(
        token="test-token",
        transport=httpx.MockTransport(handler),
        sleep_fn=sleeps.append,
    )
    result = port.understand("compare groups")
    assert result.output.objectives == ["group_comparison"]
    assert calls == 2
    assert sleeps == [3.0]
    port.close()
