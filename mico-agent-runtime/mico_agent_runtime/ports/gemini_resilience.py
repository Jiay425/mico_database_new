"""Small, provider-neutral resilience helpers for Gemini canary calls.

The Gemini OpenAI-compatible endpoint returns quota failures as a structured
``RESOURCE_EXHAUSTED`` response.  These helpers deliberately keep the
provider details at the HTTP boundary: callers can wait for the provider's
``RetryInfo`` delay, deduplicate successful identical requests, and enforce a
per-run request budget without changing scientific planning semantics.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx


class GeminiRequestBudgetExceeded(RuntimeError):
    """Raised before an outbound call would exceed the configured run budget."""

    code = "GEMINI_REQUEST_BUDGET_EXCEEDED"

    def __init__(self, role: str, used: int, limit: int) -> None:
        super().__init__(f"{self.code}:{role}:{used}/{limit}")
        self.role = role
        self.used = used
        self.limit = limit


@dataclass
class GeminiRequestBudget:
    """Optional total/per-role request budget shared by one Agent Run."""

    total_limit: int | None = None
    role_limits: Mapping[str, int] = field(default_factory=dict)
    total_used: int = 0
    role_used: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.total_limit is not None and (
            isinstance(self.total_limit, bool)
            or not isinstance(self.total_limit, int)
            or self.total_limit < 1
        ):
            raise ValueError("total Gemini request limit must be a positive integer")
        if any(
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
            for limit in self.role_limits.values()
        ):
            raise ValueError("role Gemini request limits must be positive integers")

    def reserve(self, role: str) -> None:
        role_limit = self.role_limits.get(role)
        if role_limit is not None and self.role_used.get(role, 0) >= role_limit:
            raise GeminiRequestBudgetExceeded(role, self.role_used.get(role, 0), role_limit)
        if self.total_limit is not None and self.total_used >= self.total_limit:
            raise GeminiRequestBudgetExceeded("total", self.total_used, self.total_limit)
        self.total_used += 1
        self.role_used[role] = self.role_used.get(role, 0) + 1


def parse_retry_delay_seconds(response: httpx.Response) -> float | None:
    """Extract Gemini RetryInfo/Retry-After without assuming one wire shape."""

    header = response.headers.get("retry-after")
    if header:
        try:
            return max(0.0, float(header.strip()))
        except ValueError:
            pass
    try:
        payload = response.json()
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    details = payload.get("error", {}).get("details", []) if isinstance(payload.get("error"), Mapping) else []
    if isinstance(details, list):
        for detail in details:
            if not isinstance(detail, Mapping):
                continue
            type_name = str(detail.get("@type", ""))
            if not type_name.endswith("RetryInfo"):
                continue
            retry = detail.get("retryDelay")
            if isinstance(retry, (int, float)) and not isinstance(retry, bool):
                return max(0.0, float(retry))
            if isinstance(retry, str):
                match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)s\s*", retry)
                if match:
                    return max(0.0, float(match.group(1)))
    return None


def is_quota_response(response: httpx.Response) -> bool:
    """Return true only for an HTTP 429/RESOURCE_EXHAUSTED quota response."""

    if response.status_code == 429:
        return True
    try:
        payload = response.json()
    except (ValueError, json.JSONDecodeError):
        return False
    error = payload.get("error") if isinstance(payload, Mapping) else None
    return isinstance(error, Mapping) and str(error.get("status", "")) == "RESOURCE_EXHAUSTED"


def sleep_before_retry(
    error: BaseException,
    *,
    sleep: Callable[[float], None] = time.sleep,
    max_delay_seconds: float | None = None,
) -> float | None:
    """Honor a provider retry delay and return the applied delay."""

    response = getattr(error, "response", None)
    if not isinstance(response, httpx.Response) or not is_quota_response(response):
        return None
    delay = getattr(error, "retry_delay_seconds", None)
    if delay is None:
        delay = parse_retry_delay_seconds(response)
    if delay is None:
        # A provider can close a TLS/HTTP connection without returning a
        # response (httpx RemoteProtocolError/ConnectError).  Immediate
        # retries tend to hit the same transient edge again, especially when
        # the endpoint is under load.  Keep this short and bounded; it is not
        # a quota wait and never changes the request budget.
        if isinstance(error, httpx.TransportError):
            applied = 2.0
            if max_delay_seconds is not None:
                applied = min(applied, max(0.0, max_delay_seconds))
            sleep(applied)
            return applied
        return None
    applied = max(0.0, float(delay))
    if max_delay_seconds is not None:
        applied = min(applied, max(0.0, max_delay_seconds))
    sleep(applied)
    return applied


class GeminiResponseCache:
    """Short-lived cache for successful identical request payloads only."""

    def __init__(self, ttl_seconds: float = 30.0) -> None:
        if ttl_seconds < 0:
            raise ValueError("cache TTL must be non-negative")
        self.ttl_seconds = float(ttl_seconds)
        self._entries: dict[str, tuple[float, int, dict[str, Any], dict[str, str]]] = {}
        self.hits = 0

    @staticmethod
    def key(endpoint: str, body: object) -> str:
        encoded = json.dumps(
            {"endpoint": endpoint, "body": body},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        import hashlib
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def get(self, key: str, request: httpx.Request) -> httpx.Response | None:
        if self.ttl_seconds <= 0:
            return None
        entry = self._entries.get(key)
        if entry is None:
            return None
        created, status, payload, headers = entry
        if time.monotonic() - created > self.ttl_seconds:
            self._entries.pop(key, None)
            return None
        self.hits += 1
        return httpx.Response(status, json=payload, headers=headers, request=request)

    def put(self, key: str, response: httpx.Response) -> None:
        if response.status_code != 200:
            return
        try:
            payload = response.json()
        except (ValueError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        self._entries[key] = (
            time.monotonic(),
            response.status_code,
            payload,
            {key: value for key, value in response.headers.items() if key.lower() in {"content-type"}},
        )


__all__ = [
    "GeminiRequestBudget",
    "GeminiRequestBudgetExceeded",
    "GeminiResponseCache",
    "is_quota_response",
    "parse_retry_delay_seconds",
    "sleep_before_retry",
]
