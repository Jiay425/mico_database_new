from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Protocol
from urllib.parse import urlsplit

import httpx

from mico_agent_runtime.contracts.tools import JavaToolCall, JavaToolResponse, validate_java_tool_call


class JavaPortConfigurationError(RuntimeError):
    """Raised without opening a socket when Java port configuration is absent."""


class JavaPortTransportError(RuntimeError):
    """Safe transport error; response bodies and connection details are withheld."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class JavaPortContractError(RuntimeError):
    """Raised before transport when a call is not in the closed Java contract."""

    def __init__(self, code: str = "JAVA_TOOL_CALL_INVALID") -> None:
        super().__init__(code)
        self.code = code


class JavaAgentToolPort(Protocol):
    def execute(self, call: JavaToolCall) -> JavaToolResponse:
        ...


class HttpJavaAgentToolPort:
    """HTTP adapter for the Java internal API; it never discovers another endpoint."""

    endpoint_suffix = "/internal/agent/tools/execute"

    def __init__(self, base_url: str, token: str, transport: httpx.BaseTransport | None = None) -> None:
        if not base_url or not token or not token.strip():
            raise JavaPortConfigurationError("Java Agent Tool port is not configured")
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username:
            raise JavaPortConfigurationError("Java Agent Tool base URL is invalid")
        if parsed.query or parsed.fragment:
            raise JavaPortConfigurationError("Java Agent Tool base URL is invalid")
        self._endpoint = base_url.rstrip("/") + self.endpoint_suffix
        self._token = token
        # Aggregate read-model operations are bounded by the Java tool contract,
        # but the remote standard-abundance scan is materially slower than a
        # point sample read. Keep a finite transport cap long enough for that
        # controlled operation; this does not widen the tool or SQL surface.
        self._client = httpx.Client(transport=transport, timeout=120.0)

    @classmethod
    def from_environment(
        cls,
        env: Mapping[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> "HttpJavaAgentToolPort":
        source = os.environ if env is None else env
        base_url = source.get("MICO_JAVA_AGENT_TOOL_BASE_URL")
        token = source.get("MICO_AGENT_INTERNAL_TOKEN")
        if not base_url or not token or not token.strip():
            raise JavaPortConfigurationError("Java Agent Tool port is not configured")
        return cls(base_url, token, transport=transport)

    def execute(self, call: JavaToolCall) -> JavaToolResponse:
        try:
            validated_call = validate_java_tool_call(call)
        except (TypeError, ValueError) as exc:
            raise JavaPortContractError() from exc
        try:
            response = self._client.post(
                self._endpoint,
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Content-Type": "application/json",
                },
                json=validated_call.to_java_payload(),
            )
        except httpx.HTTPError as exc:
            raise JavaPortTransportError("JAVA_TOOL_NETWORK_FAILED") from exc
        if response.status_code != 200:
            raise JavaPortTransportError("JAVA_TOOL_HTTP_REJECTED")
        try:
            parsed = JavaToolResponse.model_validate(response.json())
        except (ValueError, TypeError) as exc:
            raise JavaPortTransportError("JAVA_TOOL_RESPONSE_INVALID") from exc
        if parsed.runId != validated_call.runId or parsed.toolCallId != validated_call.toolCallId:
            raise JavaPortTransportError("JAVA_TOOL_RESPONSE_MISMATCH")
        return parsed

    def close(self) -> None:
        self._client.close()
