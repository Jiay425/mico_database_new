"""Closed output contract for the entry-point Task Understanding step."""

from __future__ import annotations

from pydantic import Field, model_validator

from .base import ClosedModel
from .decision_state import ScientificObjective, ScientificTaskConstraints


class TaskUnderstandingOutput(ClosedModel):
    """User goals and explicit constraints, never an execution plan."""

    objectives: list[ScientificObjective] = Field(default_factory=list)
    constraints: ScientificTaskConstraints = Field(default_factory=ScientificTaskConstraints)

    @model_validator(mode="after")
    def reject_duplicate_objectives(self) -> "TaskUnderstandingOutput":
        if len(self.objectives) != len(set(self.objectives)):
            raise ValueError("task understanding objectives must be unique")
        return self


__all__ = ["TaskUnderstandingOutput"]
