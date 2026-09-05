"""Stable input contract for the Scientific Action Decision Policy."""

from __future__ import annotations

from typing import Any

from .base import ClosedModel
from .decision_state import (
    ScientificActionSpaceState,
    ScientificAnalysisState,
    ScientificDataState,
    ScientificDecisionState,
    ScientificEvidenceState,
    ScientificProgressState,
    ScientificTaskState,
)


class ScientificPolicyInput(ClosedModel):
    """The six policy-facing blocks; runtime/debug fields are excluded."""

    task: ScientificTaskState
    data_state: ScientificDataState
    analysis_state: ScientificAnalysisState
    evidence_state: ScientificEvidenceState
    progress: ScientificProgressState
    action_space: ScientificActionSpaceState


def build_scientific_policy_input(
    state: ScientificDecisionState | ScientificPolicyInput,
) -> ScientificPolicyInput:
    """Create a stable, JSON-safe policy payload from Decision State."""

    if isinstance(state, ScientificPolicyInput):
        return state.model_copy(deep=True)
    if not isinstance(state, ScientificDecisionState):
        raise TypeError("scientific policy input requires ScientificDecisionState")
    # Round-tripping through JSON mode prevents future runtime-only fields on
    # ScientificDecisionState from crossing this explicit policy boundary.
    payload: dict[str, Any] = state.model_dump(mode="json")
    return ScientificPolicyInput.model_validate(payload)


build_policy_input = build_scientific_policy_input


__all__ = [
    "ScientificPolicyInput",
    "build_policy_input",
    "build_scientific_policy_input",
]
