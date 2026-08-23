from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.dialects.mysql import JSON

from mico_agent_runtime.storage.configuration import (
    RuntimeStorageConfiguration,
    RuntimeStorageConfigurationError,
)
from mico_agent_runtime.storage.mysql_store import (
    MySqlRuntimeStore,
    RuntimeOrmBase,
    _utc_aware,
    _utc_naive,
)
from mico_agent_runtime.storage.migration_configuration import (
    MigrationStorageConfiguration,
    MigrationStorageConfigurationError,
)
from mico_agent_runtime.storage.ports import DuplicateRecordError


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "alembic" / "versions" / "0001_agent_runtime_state.py"
MODELS = ROOT / "mico_agent_runtime" / "storage" / "models.py"


def key_text() -> str:
    import base64

    return base64.urlsafe_b64encode(bytes(range(32))).rstrip(b"=").decode("ascii")


def mysql_env(url: str = "mysql+asyncmy://runtime@127.0.0.1/mico_agent_runtime") -> dict[str, str]:
    return {
        "MICO_AGENT_RUNTIME_MYSQL_ENABLED": "true",
        "MICO_AGENT_RUNTIME_DATABASE_URL": url,
        "MICO_RUNTIME_STATE_ENCRYPTION_KEY": key_text(),
        "MICO_RUNTIME_STATE_ENCRYPTION_KEY_ID": "key-runtime-v1",
    }


def test_orm_metadata_has_only_the_five_runtime_tables_and_mysql_options() -> None:
    assert set(RuntimeOrmBase.metadata.tables) == {
        "agent_run", "agent_step", "agent_artifact", "approval_ticket", "tool_audit"
    }
    for table in RuntimeOrmBase.metadata.tables.values():
        assert table.dialect_options["mysql"]["engine"] == "InnoDB"
        assert table.dialect_options["mysql"]["charset"] == "utf8mb4"
        assert table.dialect_options["mysql"]["collate"] == "utf8mb4_bin"
        assert "patient_data_manager" not in table.name
    assert isinstance(RuntimeOrmBase.metadata.tables["agent_step"].c.snapshot_metadata.type, JSON)
    assert isinstance(RuntimeOrmBase.metadata.tables["tool_audit"].c.snapshot_metadata.type, JSON)
    assert all(column.type.timezone is False for table in RuntimeOrmBase.metadata.tables.values() for column in table.columns if column.name.endswith("_at"))


def test_migration_contains_mysql_constraints_indexes_and_no_business_tables() -> None:
    text = MIGRATION.read_text(encoding="utf-8")
    for table in ("agent_run", "agent_step", "agent_artifact", "approval_ticket", "tool_audit"):
        assert f'"{table}"' in text
    for index in ("ix_agent_run_status_updated_at", "ix_agent_step_run_started_at", "ix_tool_audit_run_created_at"):
        assert index in text
    for constraint in ("ck_agent_run_status", "ck_agent_step_status", "ck_approval_ticket_status", "ck_tool_audit_status"):
        assert constraint in text
    for constraint in (
        "ck_agent_run_id",
        "ck_agent_run_encryption_key_id",
        "ck_agent_step_id",
        "ck_agent_artifact_id",
        "ck_agent_artifact_type",
        "ck_agent_artifact_storage_ref_binding",
        "ck_approval_ticket_requested_by",
        "ck_tool_audit_call_id",
    ):
        assert constraint in text
    for column in ("safe_input_code", "safe_output_code", "artifact_storage_ref"):
        assert column in text
    assert "safe_input_summary" not in text
    assert "safe_output_summary" not in text
    assert "external_location_ref" not in text
    assert "sha256:[0-9a-f]{64}" in text
    assert "transient-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}" in text
    assert "REGEXP" in text
    assert "mysql.JSON" in text
    assert "InnoDB" in text and "utf8mb4" in text
    assert "JSONB" not in text
    assert "asyncpg" not in text
    assert " ~ " not in text
    assert "?:" not in text
    for forbidden in (
        "patient_data_manager", "patients", "microbe_abundance",
        "microbe_abundance_standard", "meta2db_sample_metadata",
    ):
        assert forbidden not in text


def test_storage_model_has_no_business_value_blacklist() -> None:
    text = MODELS.read_text(encoding="utf-8")
    assert "_FORBIDDEN_VALUE_MARKERS" not in text
    assert "SRR1518476" not in text
    assert "2015_Castro-NallarE" not in text
    assert "2015_castro-nallare" not in text


def test_mysql_configuration_fails_closed_without_enable_url_or_key() -> None:
    with pytest.raises(RuntimeStorageConfigurationError) as error:
        RuntimeStorageConfiguration.from_environment({})
    assert error.value.code == "RUNTIME_MYSQL_DISABLED"

    base = {"MICO_AGENT_RUNTIME_MYSQL_ENABLED": "true"}
    with pytest.raises(RuntimeStorageConfigurationError) as error:
        RuntimeStorageConfiguration.from_environment(base)
    assert error.value.code == "RUNTIME_DATABASE_URL_INVALID"

    for url in (
        "postgresql+asyncpg://runtime@127.0.0.1/mico_agent_runtime",
        "mysql+asyncmy://runtime@127.0.0.1/patient_data_manager",
        "mysql+asyncmy://runtime@127.0.0.1/other_runtime",
        "mysql+asyncmy://runtime@127.0.0.1/",
        "sqlite+aiosqlite:///mico_agent_runtime.db",
    ):
        with pytest.raises(RuntimeStorageConfigurationError) as error:
            RuntimeStorageConfiguration.from_environment({**base, "MICO_AGENT_RUNTIME_DATABASE_URL": url})
        assert error.value.code == "RUNTIME_DATABASE_URL_INVALID"

    missing_key = mysql_env()
    del missing_key["MICO_RUNTIME_STATE_ENCRYPTION_KEY"]
    with pytest.raises(RuntimeStorageConfigurationError) as error:
        RuntimeStorageConfiguration.from_environment(missing_key)
    assert error.value.code == "RUNTIME_STATE_ENCRYPTION_KEY_INVALID"

    missing_key_id = mysql_env()
    del missing_key_id["MICO_RUNTIME_STATE_ENCRYPTION_KEY_ID"]
    with pytest.raises(RuntimeStorageConfigurationError) as error:
        RuntimeStorageConfiguration.from_environment(missing_key_id)
    assert error.value.code == "RUNTIME_STATE_ENCRYPTION_KEY_ID_INVALID"


def test_mysql_configuration_normalizes_only_mysql_scheme_without_connecting() -> None:
    config = RuntimeStorageConfiguration.from_environment(
        mysql_env("mysql://runtime@127.0.0.1/mico_agent_runtime")
    )
    assert config.database_url.startswith("mysql+asyncmy://")
    assert config.database_name == "mico_agent_runtime"
    assert config.state_cipher.key_id == "key-runtime-v1"


def test_migration_only_configuration_does_not_require_runtime_state_key() -> None:
    config = MigrationStorageConfiguration.from_environment({
        "MICO_AGENT_RUNTIME_MIGRATION_DATABASE_URL":
            "mysql://runtime@127.0.0.1/mico_agent_runtime",
    })
    assert config.database_url.startswith("mysql+asyncmy://")
    assert config.database_name == "mico_agent_runtime"


@pytest.mark.parametrize("url", [
    "",
    "postgresql://runtime@127.0.0.1/mico_agent_runtime",
    "mariadb://runtime@127.0.0.1/mico_agent_runtime",
    "mysql+asyncmy://runtime@127.0.0.1/patient_data_manager",
    "mysql+asyncmy://runtime@127.0.0.1/other_runtime",
    "mysql+asyncmy://runtime@127.0.0.1/",
    "mysql+asyncmy://runtime@127.0.0.1/mico_agent_runtime?ssl=false",
    "mysql+asyncmy://runtime@127.0.0.1/mico_agent_runtime#fragment",
])
def test_migration_only_configuration_rejects_unsafe_urls(url: str) -> None:
    with pytest.raises(MigrationStorageConfigurationError) as error:
        MigrationStorageConfiguration.from_environment({
            "MICO_AGENT_RUNTIME_MIGRATION_DATABASE_URL": url,
        })
    assert error.value.code == "RUNTIME_MIGRATION_DATABASE_URL_INVALID"


def test_runtime_store_configuration_still_requires_state_key_and_key_id() -> None:
    with pytest.raises(RuntimeStorageConfigurationError) as error:
        RuntimeStorageConfiguration.from_environment({
            "MICO_AGENT_RUNTIME_MYSQL_ENABLED": "true",
            "MICO_AGENT_RUNTIME_DATABASE_URL":
                "mysql+asyncmy://runtime@127.0.0.1/mico_agent_runtime",
        })
    assert error.value.code == "RUNTIME_STATE_ENCRYPTION_KEY_INVALID"


def test_mysql_store_is_not_claimed_recoverable_and_maps_conflicts_safely() -> None:
    assert MySqlRuntimeStore.persistent is True
    assert MySqlRuntimeStore.recoverable is False
    error = MySqlRuntimeStore._map_integrity_error("RUNTIME_RUN_ALREADY_EXISTS")
    assert isinstance(error, DuplicateRecordError)
    assert str(error) == "RUNTIME_RUN_ALREADY_EXISTS"
    assert "SQL" not in str(error)


def test_mysql_datetime_boundary_normalizes_aware_values_to_utc() -> None:
    from datetime import datetime, timezone, timedelta

    source = datetime(2026, 8, 21, 20, 0, tzinfo=timezone(timedelta(hours=8)))
    stored = _utc_naive(source)
    assert stored.tzinfo is None
    assert stored == datetime(2026, 8, 21, 12, 0)
    loaded = _utc_aware(stored)
    assert loaded.tzinfo == timezone.utc
    assert loaded == datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)


def test_mysql_access_is_confined_to_storage_package() -> None:
    package_root = ROOT / "mico_agent_runtime"
    for path in package_root.rglob("*.py"):
        if "storage" in path.parts:
            continue
        text = path.read_text(encoding="utf-8").lower()
        assert "asyncmy" not in text
        assert "create_async_engine" not in text
        assert "mysqlconnector" not in text
