from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Mapping
from urllib.parse import urlsplit


class MigrationStorageConfigurationError(RuntimeError):
    """Safe migration-only configuration error without URL or credential details."""

    def __init__(self, code: str = "RUNTIME_MIGRATION_DATABASE_URL_INVALID") -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class MigrationStorageConfiguration:
    database_url: str
    database_name: str = "mico_agent_runtime"

    @classmethod
    def from_environment(
        cls, env: Mapping[str, str] | None = None
    ) -> "MigrationStorageConfiguration":
        source = os.environ if env is None else env
        database_url = source.get("MICO_AGENT_RUNTIME_MIGRATION_DATABASE_URL", "").strip()
        if not database_url:
            raise MigrationStorageConfigurationError()
        parsed = urlsplit(database_url)
        if parsed.scheme not in {"mysql", "mysql+asyncmy"} or not parsed.hostname:
            raise MigrationStorageConfigurationError()
        if parsed.query or parsed.fragment:
            raise MigrationStorageConfigurationError()
        database_name = parsed.path.rstrip("/").rsplit("/", 1)[-1]
        if database_name != "mico_agent_runtime":
            raise MigrationStorageConfigurationError()
        normalized_url = database_url
        if parsed.scheme == "mysql":
            normalized_url = database_url.replace("mysql://", "mysql+asyncmy://", 1)
        return cls(database_url=normalized_url)
