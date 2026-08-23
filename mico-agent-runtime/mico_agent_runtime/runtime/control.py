"""Fail-closed run cancellation and recovery boundary."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from pydantic import TypeAdapter

from mico_agent_runtime.contracts.control import RunControlResponse
from mico_agent_runtime.storage.models import RuntimeStatus
from mico_agent_runtime.storage.ports import RuntimeStore, RuntimeStoreError
from mico_agent_runtime.storage.types import RunId


class RunControlError(RuntimeError):
    """Safe control error without SQL, state or connection details."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class RunControlCoordinator:
    """Control only states whose execution semantics are actually supported."""

    def __init__(self, store: RuntimeStore, *, checkpoint_saver=None,
                 resume_handler: Callable[[str], Awaitable[RunControlResponse]] | None = None) -> None:
        self._store = store
        self._checkpoint_saver = checkpoint_saver
        self._resume_handler = resume_handler

    @property
    def recoverable(self) -> bool:
        return self._checkpoint_saver is not None and self._resume_handler is not None

    def bind_resume_handler(
        self,
        handler: Callable[[str], Awaitable[RunControlResponse]],
    ) -> None:
        """Bind the graph owner only after its checkpoint saver is ready."""
        self._resume_handler = handler

    async def cancel(self, run_id: str) -> RunControlResponse:
        try:
            typed_id = TypeAdapter(RunId).validate_python(run_id)
        except Exception:
            raise RunControlError("RUN_NOT_FOUND") from None
        try:
            run = await self._store.load_run(typed_id)
            if run is None:
                raise RunControlError("RUN_NOT_FOUND")
            if run.status in {
                RuntimeStatus.COMPLETED,
                RuntimeStatus.FAILED,
                RuntimeStatus.CANCELLED,
            }:
                raise RunControlError("RUN_ALREADY_TERMINAL")
            if run.status == RuntimeStatus.RUNNING:
                # The current synchronous graph has no cooperative cancellation
                # token; marking it cancelled would be a false success.
                raise RunControlError("RUN_CANCEL_NOT_SUPPORTED_FOR_ACTIVE_RUN")
            updated = await self._store.mark_run_status(
                typed_id, RuntimeStatus.CANCELLED
            )
            return RunControlResponse(
                runId=updated.runId,
                taskId=updated.taskId,
                status=updated.status.value,
                controlCode="RUN_CANCELLED",
            )
        except RunControlError:
            raise
        except RuntimeStoreError:
            raise RunControlError("RUNTIME_PERSISTENCE_FAILED") from None

    async def resume(self, run_id: str) -> RunControlResponse:
        try:
            typed_id = TypeAdapter(RunId).validate_python(run_id)
            run = await self._store.load_run(typed_id)
        except Exception:
            raise RunControlError("RUN_NOT_FOUND") from None
        if run is None:
            raise RunControlError("RUN_NOT_FOUND")
        if run.status in {RuntimeStatus.COMPLETED, RuntimeStatus.CANCELLED}:
            raise RunControlError("RUN_ALREADY_TERMINAL")
        if self._checkpoint_saver is None:
            raise RunControlError("RUNTIME_RECOVERY_NOT_ENABLED")
        if self._resume_handler is None:
            raise RunControlError("RUNTIME_RESUME_NOT_IMPLEMENTED")
        try:
            return await self._resume_handler(typed_id)
        except RunControlError:
            raise
        except Exception:
            raise RunControlError("RUNTIME_RESUME_FAILED") from None

    async def pause(self, run_id: str) -> RunControlResponse:
        try:
            typed_id = TypeAdapter(RunId).validate_python(run_id)
            run = await self._store.load_run(typed_id)
        except Exception:
            raise RunControlError("RUN_NOT_FOUND") from None
        if run is None:
            raise RunControlError("RUN_NOT_FOUND")
        if run.status in {
            RuntimeStatus.COMPLETED,
            RuntimeStatus.FAILED,
            RuntimeStatus.CANCELLED,
        }:
            raise RunControlError("RUN_ALREADY_TERMINAL")
        raise RunControlError("RUN_PAUSE_NOT_IMPLEMENTED")
