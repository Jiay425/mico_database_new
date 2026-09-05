"""Versioned Full Dynamic E2E preparation contracts.

The E2E task set is deliberately kept outside the six-block policy State.  It
describes what a canary run must cover and how a missing capability is reported;
it never supplies a Scientific Action, QueryPlan, AnalysisPlan, or trajectory.
"""

from .decision_sft_v1_full_dynamic import (
    ALL_SCIENTIFIC_ACTIONS,
    E2E_TASK_SET_SCHEMA_VERSION,
    FULL_DYNAMIC_E2E_VERSION,
    TaskSetValidationError,
    load_task_set,
    prepare_task_set,
    validate_task_set,
)

__all__ = [
    "ALL_SCIENTIFIC_ACTIONS",
    "E2E_TASK_SET_SCHEMA_VERSION",
    "FULL_DYNAMIC_E2E_VERSION",
    "TaskSetValidationError",
    "load_task_set",
    "prepare_task_set",
    "validate_task_set",
]
