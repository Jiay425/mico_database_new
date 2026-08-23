from __future__ import annotations

from typing import Literal

from pydantic import Field

from .base import ClosedModel


class RuntimeMetrics(ClosedModel):
    """Safe, bounded execution metrics for the internal progress projection.

    These values describe this Runtime invocation only.  They intentionally do
    not include tokens, prompts, payloads, SQL, locator values or cost data.
    """

    durationMs: int | None = Field(default=None, strict=True, ge=0, le=86_400_000)
    plannerMode: Literal["deterministic", "model"] | None = None
    modelCallCount: int = Field(default=0, strict=True, ge=0, le=10)
    javaToolCallCount: int = Field(default=0, strict=True, ge=0, le=10)
    snapshotCount: int = Field(default=0, strict=True, ge=0, le=10)
    costStatus: Literal["not_measured"] = "not_measured"
