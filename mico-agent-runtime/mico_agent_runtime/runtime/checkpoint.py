"""Encrypted LangGraph checkpoint adapter for the independent Runtime store.

This module is deliberately narrow.  LangGraph owns the checkpoint shape in
memory, while the Runtime store persists only an encrypted envelope.  No
checkpoint channel value, request locator, Java payload, or model output is
projected into a normal Runtime table column.
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    PendingWrite,
)
from pydantic import TypeAdapter

from mico_agent_runtime.storage.crypto import (
    RuntimeStateCipher,
    RuntimeStateDecryptError,
    StateBindingContext,
)
from mico_agent_runtime.storage.ports import RuntimeStore, RuntimeStoreError
from mico_agent_runtime.storage.types import RunId, TaskId, TraceId


class RuntimeCheckpointError(RuntimeError):
    """Safe checkpoint error without state, SQL, DSN, or payload details."""

    def __init__(self, code: str = "RUNTIME_CHECKPOINT_FAILED") -> None:
        super().__init__(code)
        self.code = code


_RUN_ID = TypeAdapter(RunId)
_TASK_ID = TypeAdapter(TaskId)
_TRACE_ID = TypeAdapter(TraceId)


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _unb64(value: str) -> bytes:
    if not isinstance(value, str) or not value or len(value) % 4 == 1:
        raise RuntimeCheckpointError("RUNTIME_CHECKPOINT_INVALID")
    try:
        return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except Exception:
        raise RuntimeCheckpointError("RUNTIME_CHECKPOINT_INVALID") from None


def _jsonable(value: Any) -> Any:
    """Serialize only the small, non-channel envelope metadata."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        if len(value) > 256:
            raise RuntimeCheckpointError("RUNTIME_CHECKPOINT_INVALID")
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > 512:
            raise RuntimeCheckpointError("RUNTIME_CHECKPOINT_INVALID")
        return [_jsonable(child) for child in value]
    raise RuntimeCheckpointError("RUNTIME_CHECKPOINT_INVALID")


class EncryptedLangGraphCheckpointSaver(BaseCheckpointSaver[Any]):
    """A single-latest-checkpoint saver backed by an encrypted Runtime row.

    The adapter intentionally does not implement time-travel, branch copying,
    or checkpoint deletion.  A Runtime run has one recoverable execution
    lineage.  The underlying store remains a separate concern and never
    becomes a general-purpose JSON or SQL interface.
    """

    def __init__(self, store: RuntimeStore, cipher: RuntimeStateCipher) -> None:
        super().__init__()
        self._store = store
        self._cipher = cipher

    @property
    def recoverable(self) -> bool:
        return True

    @staticmethod
    def _binding(config: RunnableConfig) -> tuple[RunId, TaskId, TraceId]:
        try:
            configurable = config.get("configurable", {})
            if not isinstance(configurable, Mapping):
                raise ValueError
            return (
                _RUN_ID.validate_python(configurable["thread_id"]),
                _TASK_ID.validate_python(configurable["runtime_task_id"]),
                _TRACE_ID.validate_python(configurable["runtime_trace_id"]),
            )
        except Exception:
            raise RuntimeCheckpointError("RUNTIME_CHECKPOINT_BINDING_INVALID") from None

    @staticmethod
    def _context(run_id: RunId, task_id: TaskId, trace_id: TraceId) -> StateBindingContext:
        return StateBindingContext(
            runId=run_id,
            taskId=task_id,
            traceId=trace_id,
            dataContractVersion="v1",
        )

    @staticmethod
    def _run_sync(awaitable: Any) -> Any:
        """Bridge the synchronous LangGraph API to the async Store safely."""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            try:
                return asyncio.run(awaitable)
            except RuntimeCheckpointError:
                raise
            except Exception:
                raise RuntimeCheckpointError() from None
        raise RuntimeCheckpointError("RUNTIME_CHECKPOINT_SYNC_CONTEXT_INVALID")

    async def _load(self, config: RunnableConfig) -> CheckpointTuple | None:
        run_id, task_id, trace_id = self._binding(config)
        try:
            run = await self._store.load_run(run_id)
            if run is None:
                return None
            if run.taskId != task_id or run.traceId != trace_id:
                raise RuntimeCheckpointError("RUNTIME_CHECKPOINT_BINDING_INVALID")
            context = self._context(run_id, task_id, trace_id)
            value = self._cipher.decrypt(run.encryptedStatePayload, context)
            # A newly created run contains only the encrypted request until
            # LangGraph writes its first checkpoint.  That is a valid empty
            # checkpoint history, not corruption.
            if "checkpointVersion" not in value:
                return None
            if value.get("checkpointVersion") != "langgraph-v1":
                raise RuntimeCheckpointError("RUNTIME_CHECKPOINT_INVALID")
            checkpoint = self.serde.loads_typed((
                value["checkpointType"],
                _unb64(value["checkpointData"]),
            ))
            if not isinstance(checkpoint, dict):
                raise RuntimeCheckpointError("RUNTIME_CHECKPOINT_INVALID")
            metadata = value.get("metadata") or {}
            if not isinstance(metadata, dict):
                raise RuntimeCheckpointError("RUNTIME_CHECKPOINT_INVALID")
            parent_config = value.get("parentConfig")
            if parent_config is not None and not isinstance(parent_config, dict):
                raise RuntimeCheckpointError("RUNTIME_CHECKPOINT_INVALID")
            return CheckpointTuple(
                config={"configurable": {
                    "thread_id": run_id,
                    "runtime_task_id": task_id,
                    "runtime_trace_id": trace_id,
                    "checkpoint_id": checkpoint.get("id"),
                }},
                checkpoint=checkpoint,
                metadata=metadata,
                parent_config=parent_config,
                pending_writes=None,
            )
        except RuntimeCheckpointError:
            raise
        except RuntimeStateDecryptError:
            raise RuntimeCheckpointError("RUNTIME_CHECKPOINT_DECRYPT_FAILED") from None
        except (RuntimeStoreError, KeyError, TypeError, ValueError, AttributeError):
            raise RuntimeCheckpointError() from None

    async def _save(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
    ) -> RunnableConfig:
        run_id, task_id, trace_id = self._binding(config)
        try:
            checkpoint_type, checkpoint_bytes = self.serde.dumps_typed(checkpoint)
            payload = {
                "checkpointVersion": "langgraph-v1",
                "checkpointType": checkpoint_type,
                "checkpointData": _b64(checkpoint_bytes),
                "metadata": _jsonable(metadata),
                # LangGraph adds process-local runtime objects to RunnableConfig.
                # Persist only the opaque binding and checkpoint ID needed for
                # a later read; never serialize callbacks or runtime objects.
                "parentConfig": {"configurable": {
                    key: config.get("configurable", {}).get(key)
                    for key in (
                        "thread_id",
                        "runtime_task_id",
                        "runtime_trace_id",
                        "checkpoint_id",
                    )
                    if config.get("configurable", {}).get(key) is not None
                }},
            }
            envelope = self._cipher.encrypt(payload, self._context(run_id, task_id, trace_id))
            await self._store.update_encrypted_state(
                run_id,
                envelope,
                updated_at=datetime.now(timezone.utc),
            )
            return {
                "configurable": {
                    **dict(config.get("configurable", {})),
                    "checkpoint_id": checkpoint["id"],
                }
            }
        except RuntimeCheckpointError:
            raise
        except (RuntimeStoreError, TypeError, ValueError, KeyError):
            raise RuntimeCheckpointError() from None

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return self._run_sync(self._load(config))

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        del new_versions
        return self._run_sync(self._save(config, checkpoint, metadata))

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        del filter, before
        if config is None:
            return
        item = self.get_tuple(config)
        if item is not None and (limit is None or limit > 0):
            yield item

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        # Writes are transient until LangGraph commits the next full checkpoint.
        # They are intentionally not projected into Runtime audit/storage rows.
        del config, writes, task_id, task_path

    def delete_thread(self, thread_id: str) -> None:
        # The RuntimeStore has no destructive delete operation by design.
        del thread_id

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return await self._load(config)

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        del new_versions
        return await self._save(config, checkpoint, metadata)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Any:
        del filter, before
        if config is None:
            return
        item = await self._load(config)
        if item is not None and (limit is None or limit > 0):
            yield item

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        del config, writes, task_id, task_path
