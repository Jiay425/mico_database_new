"""Alembic environment for the independent Runtime MySQL database.

This module is intentionally inert until an operator explicitly runs Alembic
with a migration-only database URL configured. Runtime state encryption is
validated only when the Runtime Store itself is enabled.
"""

from __future__ import annotations

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from mico_agent_runtime.storage.migration_configuration import MigrationStorageConfiguration
from mico_agent_runtime.storage.mysql_store import RuntimeOrmBase


config = context.config
target_metadata = RuntimeOrmBase.metadata


def _runtime_database_url() -> str:
    return MigrationStorageConfiguration.from_environment().database_url


def run_migrations_offline() -> None:
    context.configure(
        url=_runtime_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _runtime_database_url()
    connectable = async_engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(
            lambda sync_connection: context.configure(
                connection=sync_connection,
                target_metadata=target_metadata,
            )
        )
        async with connection.begin():
            await connection.run_sync(lambda _: context.run_migrations())
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    import asyncio

    asyncio.run(run_migrations_online())
