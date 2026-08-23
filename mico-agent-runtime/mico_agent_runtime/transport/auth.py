from __future__ import annotations

import os
import secrets
from collections.abc import Mapping


class RuntimeAuth:
    """Runtime service-token guard; the token is never exposed to callers."""

    def __init__(self, token: str | None) -> None:
        self._token = token if token and token.strip() else None

    @classmethod
    def from_environment(cls, env: Mapping[str, str] | None = None) -> "RuntimeAuth":
        source = os.environ if env is None else env
        return cls(source.get("MICO_RUNTIME_INTERNAL_TOKEN"))

    @property
    def enabled(self) -> bool:
        return self._token is not None

    def matches(self, authorization: str | None) -> bool:
        if self._token is None or authorization is None or not authorization.startswith("Bearer "):
            return False
        presented = authorization[len("Bearer "):]
        if not presented or not presented.strip():
            return False
        return secrets.compare_digest(self._token, presented)
