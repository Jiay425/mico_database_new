from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping
from urllib.parse import urlsplit

from .crypto import RuntimeStateCipher, RuntimeStateConfigurationError


class RuntimeStorageConfigurationError(RuntimeError):
    """Safe MySQL runtime-storage configuration failure with no URL details."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class RuntimeStorageConfiguration:
    database_url: str
    database_name: str
    state_cipher: RuntimeStateCipher

    @classmethod
    def from_environment(cls, env: Mapping[str, str] | None = None) -> "RuntimeStorageConfiguration":
        source = os.environ if env is None else env
        enabled = source.get("MICO_AGENT_RUNTIME_MYSQL_ENABLED", "").strip().lower()
        if enabled != "true":
            raise RuntimeStorageConfigurationError("RUNTIME_MYSQL_DISABLED")

        database_url = source.get("MICO_AGENT_RUNTIME_DATABASE_URL", "").strip()
        if not database_url:
            raise RuntimeStorageConfigurationError("RUNTIME_DATABASE_URL_INVALID")
        parsed = urlsplit(database_url)
        if parsed.scheme not in {"mysql", "mysql+asyncmy"} or not parsed.netloc:
            raise RuntimeStorageConfigurationError("RUNTIME_DATABASE_URL_INVALID")
        if parsed.query or parsed.fragment:
            raise RuntimeStorageConfigurationError("RUNTIME_DATABASE_URL_INVALID")
        database_name = parsed.path.rstrip("/").rsplit("/", 1)[-1]
        if database_name != "mico_agent_runtime":
            raise RuntimeStorageConfigurationError("RUNTIME_DATABASE_URL_INVALID")
        try:
            cipher = RuntimeStateCipher.from_environment(source)
        except RuntimeStateConfigurationError as exc:
            raise RuntimeStorageConfigurationError(exc.code) from None
        normalized_url = database_url
        if parsed.scheme == "mysql":
            normalized_url = database_url.replace("mysql://", "mysql+asyncmy://", 1)
        return cls(database_url=normalized_url, database_name=database_name, state_cipher=cipher)
