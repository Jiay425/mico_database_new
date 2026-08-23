"""Closed Runtime run-control projections."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .base import ClosedModel


RunControlStatus = Literal[
    "QUEUED",
    "RUNNING",
    "WAITING_APPROVAL",
    "WAITING_JOB",
    "COMPLETED",
    "FAILED",
    "CANCELLED",
]
RunControlCode = Literal[
    "RUN_CANCELLED",
    "RUN_ALREADY_TERMINAL",
    "RUN_CANCEL_NOT_SUPPORTED_FOR_ACTIVE_RUN",
    "RUN_PAUSE_NOT_IMPLEMENTED",
    "RUN_RESUMED",
    "RUN_NOT_FOUND",
    "RUNTIME_RECOVERY_NOT_ENABLED",
    "RUNTIME_RESUME_NOT_IMPLEMENTED",
    "RUNTIME_RESUME_FAILED",
]


class RunControlResponse(ClosedModel):
    runId: str = Field(pattern=r"^run-[0-9a-f]{32}$")
    taskId: str = Field(pattern=r"^task-[0-9a-f]{32}$")
    status: RunControlStatus
    controlCode: RunControlCode
    dataContractVersion: Literal["v1"] = "v1"
