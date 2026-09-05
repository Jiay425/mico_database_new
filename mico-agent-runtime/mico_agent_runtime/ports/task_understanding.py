"""Strong-model Task Understanding boundary.

The adapter consumes only the original user question and returns a closed
objective/constraint object.  It deliberately has no access to observations,
action history, catalogs, SQL, or analysis methods.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable, Literal, Protocol
from urllib.parse import urlsplit

import httpx

from mico_agent_runtime.contracts.task_understanding import TaskUnderstandingOutput
from mico_agent_runtime.ports.gemini_resilience import (
    GeminiRequestBudget,
    GeminiRequestBudgetExceeded,
    GeminiResponseCache,
    parse_retry_delay_seconds,
    sleep_before_retry,
)


TASK_UNDERSTANDING_MODEL = "gemini-3.5-flash-lite"
TASK_UNDERSTANDING_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
MAX_TASK_UNDERSTANDING_RETRIES = 1


class TaskUnderstandingError(ValueError):
    """Stable error after the bounded model/contract retry is exhausted."""

    def __init__(
        self,
        code: str = "TASK_UNDERSTANDING_FAILED",
        *,
        response: httpx.Response | None = None,
    ) -> None:
        super().__init__(code)
        self.code = code
        self.response = response
        self.retry_delay_seconds = parse_retry_delay_seconds(response) if response is not None else None


@dataclass(frozen=True)
class TaskUnderstandingResult:
    output: TaskUnderstandingOutput
    mode: Literal["model", "deterministic"]
    model: str
    fallbackCode: str | None = None


class TaskUnderstandingPort(Protocol):
    def understand(self, query: str) -> TaskUnderstandingResult:
        ...


def _parse_json(content: object) -> object:
    if not isinstance(content, str):
        raise ValueError("task understanding content is not text")
    text = re.sub(r"(?is)<think>.*?(?:</think>|$)", "", content).strip()
    fenced = re.search(r"(?is)```(?:json)?\s*(.*?)\s*```", text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char not in "{[":
                continue
            try:
                value, _ = decoder.raw_decode(text[index:])
                return value
            except json.JSONDecodeError:
                continue
        raise ValueError("task understanding content is not valid JSON")


TASK_UNDERSTANDING_SYSTEM_PROMPT = """
You are the Mico scientific task-understanding model. Return exactly one JSON
object with only these keys:
{
  "objectives": [],
  "constraints": {
    "disease_groups": [],
    "target_features": [],
    "focus_covariates": [],
    "requested_projects": [],
    "requested_stratifiers": []
  }
}

Choose objectives only from this closed catalog:
- group_comparison
- projection_analysis
- stratified_analysis
- confounder_assessment
- cross_project_validation
- cross_disease_validation
- evidence_support

Record a constraint only when the user explicitly states it. Preserve the
user's concrete labels (for example T2D, Healthy, Bacteroides, age, gender,
or a project name). Do not infer a constraint from domain knowledge or from
the wording alone. The original question is not an output field; the Runtime
will preserve it verbatim.

This is not execution planning. Never output an action name, action sequence,
allowed_actions, candidate_actions, next_action, analysis_method, SQL, Python,
retrieval_route, should_adjust_confounders, or any other key. Do not summarize
or rewrite the question. Return empty arrays when the user did not explicitly
request a corresponding goal or constraint.
""".strip()


class DeterministicTaskUnderstandingPort:
    """Explicit no-model fallback; it never infers objectives by keywords."""

    def understand(self, _query: str) -> TaskUnderstandingResult:
        return TaskUnderstandingResult(
            output=TaskUnderstandingOutput(),
            mode="deterministic",
            model="deterministic-empty-v1",
            fallbackCode="TASK_UNDERSTANDING_DETERMINISTIC_FALLBACK",
        )


class HttpTaskUnderstandingPort:
    """OpenAI-compatible Gemini Task Understanding adapter."""

    def __init__(
        self,
        base_url: str = TASK_UNDERSTANDING_BASE_URL,
        model: str = TASK_UNDERSTANDING_MODEL,
        token: str = "",
        transport: httpx.BaseTransport | None = None,
        request_budget: GeminiRequestBudget | None = None,
        response_cache: GeminiResponseCache | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        max_retry_delay_seconds: float | None = None,
        timeout_seconds: float = 300.0,
        max_retries: int = MAX_TASK_UNDERSTANDING_RETRIES,
    ) -> None:
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username:
            raise ValueError("task understanding URL is invalid")
        if parsed.query or parsed.fragment or not model or not token.strip():
            raise ValueError("task understanding configuration is invalid")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("task understanding timeout is invalid")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError("task understanding retry count is invalid")
        base = base_url.rstrip("/")
        provider_path = parsed.path.rstrip("/")
        provider_host = (parsed.hostname or "").lower()
        if (
            provider_host.endswith("deepseek.com")
            or provider_path.endswith("/v1")
            or provider_path.endswith("/openai")
        ):
            self._endpoint = base + "/chat/completions"
        else:
            self._endpoint = base + "/v1/chat/completions"
        self._model = model
        self._token = token
        self._request_budget = request_budget
        self._response_cache = response_cache
        self._sleep_fn = sleep_fn
        self._max_retry_delay_seconds = max_retry_delay_seconds
        self._disable_reasoning = provider_host.endswith("deepseek.com")
        self._max_retries = max_retries
        self.request_count = 0
        self.cache_hit_count = 0
        # Gemini's OpenAI-compatible front door occasionally closes an idle
        # keep-alive connection between role calls.  A canary retry should
        # establish a fresh connection rather than surfacing that as a
        # provider-contract failure; disabling the pool does not add any
        # model requests and keeps the request budget authoritative.
        self._client = httpx.Client(
            transport=transport,
            timeout=timeout_seconds,
            trust_env=False,
            limits=httpx.Limits(max_keepalive_connections=0, max_connections=10),
        )

    @property
    def model(self) -> str:
        return self._model

    def _post_json(self, body: dict[str, object]) -> httpx.Response:
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }
        key = self._response_cache.key(self._endpoint, body) if self._response_cache else None
        request = self._client.build_request("POST", self._endpoint, headers=headers, json=body)
        if key is not None:
            cached = self._response_cache.get(key, request)
            if cached is not None:
                self.cache_hit_count += 1
                return cached
        if self._request_budget is not None:
            self._request_budget.reserve("task_understanding")
        self.request_count += 1
        response = self._client.send(request)
        if key is not None and response.status_code == 200:
            self._response_cache.put(key, response)
        return response

    def understand(self, query: str) -> TaskUnderstandingResult:
        if not isinstance(query, str) or not query.strip():
            raise TaskUnderstandingError("TASK_UNDERSTANDING_QUERY_INVALID")
        messages: list[dict[str, str]] = [
            {"role": "system", "content": TASK_UNDERSTANDING_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        body: dict[str, object] = {
            "model": self._model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "max_tokens": 800,
            "temperature": 0,
        }
        # DeepSeek's reasoning output can consume the whole bounded completion
        # budget before emitting the required JSON.  The materializer already
        # disables reasoning for this provider; keep Task Understanding on the
        # same contract path so it cannot fail merely because of hidden-chain
        # token exhaustion.
        if self._disable_reasoning:
            body["thinking"] = {"type": "disabled"}
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._post_json(body)
                if response.status_code != 200:
                    raise TaskUnderstandingError(
                        "TASK_UNDERSTANDING_PROVIDER_REJECTED",
                        response=response,
                    )
                payload = response.json()
                content = payload["choices"][0]["message"]["content"]
                parsed = TaskUnderstandingOutput.model_validate(_parse_json(content))
                return TaskUnderstandingResult(
                    output=parsed,
                    mode="model",
                    model=self._model,
                )
            except Exception as error:
                if isinstance(error, GeminiRequestBudgetExceeded):
                    raise
                last_error = error
                if attempt < self._max_retries:
                    sleep_before_retry(
                        error,
                        sleep=self._sleep_fn,
                        max_delay_seconds=self._max_retry_delay_seconds,
                    )
                    detail = str(error).strip().replace("\n", " ")[:1000]
                    body["messages"] = [
                        *messages,
                        {
                            "role": "user",
                            "content": (
                                "The previous output failed the closed Task Understanding contract. "
                                "Retry once with exactly the required JSON keys, only closed objective "
                                "values, and only explicitly stated constraints. "
                                f"Validation feedback: {detail}"
                            ),
                        },
                    ]
        if isinstance(last_error, TaskUnderstandingError) and last_error.code.startswith(
            "TASK_UNDERSTANDING_QUERY_"
        ):
            raise last_error
        raise TaskUnderstandingError("TASK_UNDERSTANDING_FAILED") from last_error

    def close(self) -> None:
        self._client.close()


def build_task_understanding_port(
    env: Mapping[str, str] | None = None,
    transport: httpx.BaseTransport | None = None,
    request_budget: GeminiRequestBudget | None = None,
    response_cache: GeminiResponseCache | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    max_retry_delay_seconds: float | None = None,
) -> TaskUnderstandingPort:
    source = os.environ if env is None else env
    enabled = source.get("MICO_TASK_UNDERSTANDING_ENABLED", "false").strip().lower() == "true"
    if not enabled:
        return DeterministicTaskUnderstandingPort()
    base_url = source.get(
        "MICO_TASK_UNDERSTANDING_BASE_URL", TASK_UNDERSTANDING_BASE_URL
    ).strip()
    model = source.get(
        "MICO_TASK_UNDERSTANDING_MODEL", TASK_UNDERSTANDING_MODEL
    ).strip()
    token = (
        source.get("MICO_TASK_UNDERSTANDING_TOKEN", "").strip()
        or source.get("MICO_GEMINI_API_KEY", "").strip()
        or source.get("GEMINI_API_KEY", "").strip()
        or source.get("GOOGLE_API_KEY", "").strip()
    )
    if not token:
        raise ValueError("TASK_UNDERSTANDING_TOKEN_REQUIRED")
    try:
        return HttpTaskUnderstandingPort(
            base_url,
            model,
            token,
            transport=transport,
            request_budget=request_budget,
            response_cache=response_cache,
            sleep_fn=sleep_fn,
            max_retry_delay_seconds=max_retry_delay_seconds,
            timeout_seconds=float(source.get("MICO_TASK_UNDERSTANDING_TIMEOUT_SECONDS", "300")),
            max_retries=int(source.get(
                "MICO_TASK_UNDERSTANDING_MAX_RETRIES",
                str(MAX_TASK_UNDERSTANDING_RETRIES),
            )),
        )
    except ValueError as exc:
        raise ValueError("TASK_UNDERSTANDING_CONFIG_INVALID") from exc


__all__ = [
    "DeterministicTaskUnderstandingPort",
    "HttpTaskUnderstandingPort",
    "MAX_TASK_UNDERSTANDING_RETRIES",
    "TASK_UNDERSTANDING_BASE_URL",
    "TASK_UNDERSTANDING_MODEL",
    "TaskUnderstandingError",
    "TaskUnderstandingPort",
    "TaskUnderstandingResult",
    "build_task_understanding_port",
]
