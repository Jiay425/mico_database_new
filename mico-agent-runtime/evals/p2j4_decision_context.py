"""Closed, de-identified context contract for Decision SFT v3."""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal

from pydantic import Field, StringConstraints, model_validator

from mico_agent_runtime.contracts.base import ClosedModel, Identifier
from mico_agent_runtime.contracts.research import ScientificActionName, StopReasonCode


TaskKindCode = Literal[
    "data_fact",
    "focused_analysis",
    "open_exploration",
    "safety",
]
GoalCode = Literal[
    "metadata_inventory",
    "sample_count",
    "species_coverage",
    "group_comparison",
    "confounder_adjustment",
    "cross_project_stability",
    "cross_disease_specificity",
    "stratified_analysis",
    "evidence_retrieval",
    "evidence_conflict_resolution",
    "premature_stop_prevention",
    "bounded_tool_selection",
    "safe_refusal",
    "open_exploration",
]
ObservationFlag = Literal[
    "NO_OBSERVATION",
    "OBSERVATION_VALIDATED",
    "ANALYSIS_AVAILABLE",
    "JAVA_OBSERVED",
    "VECTOR_EVIDENCE",
    "GRAPH_EVIDENCE",
    "EVIDENCE_RETRIEVED",
    "EVIDENCE_INCOMPLETE",
    "EVIDENCE_CONFLICT",
    "CONFOUNDER_PRESENT",
    "CONFOUNDER_ADJUSTED",
    "PROJECT_IMBALANCE",
    "PROJECT_VALIDATED",
    "CROSS_PROJECT_REQUIRED",
    "CROSS_DISEASE_REQUIRED",
    "CROSS_DISEASE_VALIDATED",
    "METADATA_FIRST",
    "BOUNDED_READ_REQUIRED",
]
HardCaseClass = Literal[
    "premature_stop",
    "confounder_trap",
    "evidence_conflict",
    "tool_selection",
]


class DecisionSftCandidate(ClosedModel):
    """A policy sample with safe goal/state context and no raw user content."""

    schemaVersion: Literal["p2j4-decision-dataset-v3"] = "p2j4-decision-dataset-v3"
    sourceTraceId: Identifier
    task_kind: TaskKindCode
    goal_code: GoalCode
    task_family: Annotated[str, StringConstraints(min_length=3, max_length=128, pattern=r"^[a-z0-9:_-]+$")]
    hard_case_class: HardCaseClass | None = None
    observation_flags: list[ObservationFlag] = Field(min_length=1, max_length=12)
    history_actions: list[ScientificActionName] = Field(max_length=9)
    candidate_actions: list[ScientificActionName] = Field(min_length=1, max_length=10)
    state_summary: Annotated[str, StringConstraints(min_length=1, max_length=512)]
    decision_reason: Annotated[str, StringConstraints(min_length=1, max_length=512)]
    selected_action: ScientificActionName
    alternative_actions: list[ScientificActionName] = Field(default_factory=list, max_length=9)
    stop_reason: StopReasonCode | None = None
    reviewStatus: Literal["passed", "accepted"]

    @model_validator(mode="after")
    def validate_policy_shape(self) -> "DecisionSftCandidate":
        if self.selected_action not in self.candidate_actions:
            raise ValueError("DECISION_SFT_SELECTED_ACTION_NOT_CANDIDATE")
        if self.selected_action in self.alternative_actions:
            raise ValueError("DECISION_SFT_ALTERNATIVE_CONTAINS_SELECTED")
        if any(action not in self.candidate_actions for action in self.alternative_actions):
            raise ValueError("DECISION_SFT_ALTERNATIVE_NOT_CANDIDATE")
        if self.selected_action == "finish" and self.stop_reason is None:
            raise ValueError("DECISION_SFT_FINISH_STOP_REASON_MISSING")
        if self.selected_action != "finish" and self.stop_reason is not None:
            raise ValueError("DECISION_SFT_NON_TERMINAL_STOP_REASON_PRESENT")
        return self


def _has(text: str, *terms: str) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in terms)


def normalize_task_kind(value: str) -> TaskKindCode:
    if value in {"data_fact"}:
        return "data_fact"
    if value in {"focused_analysis", "focused_comparison", "confounder_stratification", "cross_validation"}:
        return "focused_analysis"
    if value in {"safety", "policy_rejection"}:
        return "safety"
    return "open_exploration"


def derive_goal_code(question: str, task_kind: TaskKindCode, hard_case_class: str | None) -> GoalCode:
    if hard_case_class == "evidence_conflict":
        return "evidence_conflict_resolution"
    if hard_case_class == "premature_stop":
        return "premature_stop_prevention"
    if hard_case_class == "tool_selection":
        return "bounded_tool_selection"
    if hard_case_class == "confounder_trap":
        return "confounder_adjustment"
    if task_kind == "safety":
        return "safe_refusal"
    if _has(question, "混杂", "confounder", "控制年龄", "控制性别", "控制项目", "控制 project"):
        return "confounder_adjustment"
    if _has(question, "跨 project", "跨项目", "project 稳定", "项目稳定", "cross-project"):
        return "cross_project_stability"
    if _has(question, "跨疾病", "疾病特异", "disease-specific", "other diseases"):
        return "cross_disease_specificity"
    if _has(question, "分层", "stratified", "按年龄", "按性别", "按 country"):
        return "stratified_analysis"
    if _has(question, "文献", "证据", "literature", "evidence"):
        return "evidence_retrieval"
    if _has(question, "species", "覆盖", "丰度"):
        return "species_coverage"
    if _has(question, "多少", "数量", "count", "记录数"):
        return "sample_count"
    if _has(question, "比较", "差异", "compare", "difference"):
        return "group_comparison"
    if task_kind == "data_fact":
        return "metadata_inventory"
    return "open_exploration"


def derive_observation_flags(
    *,
    state_summary: str,
    history_actions: list[str],
    allowed_actions: list[str],
    question: str,
    goal_code: GoalCode,
    hard_case_class: str | None,
) -> list[ObservationFlag]:
    flags: list[ObservationFlag] = []

    def add(flag: ObservationFlag) -> None:
        if flag not in flags:
            flags.append(flag)

    state = state_summary.lower()
    if "no_observation" in state:
        add("NO_OBSERVATION")
    else:
        add("OBSERVATION_VALIDATED")
    if "evidence_retrieved" in state:
        add("EVIDENCE_RETRIEVED")
    if "requires_grounding" in state or _has(question, "证据不足", "insufficient evidence"):
        add("EVIDENCE_INCOMPLETE")
    if "java" in state:
        add("JAVA_OBSERVED")
    if "vector" in state:
        add("VECTOR_EVIDENCE")
    if "graph" in state:
        add("GRAPH_EVIDENCE")
    if hard_case_class == "evidence_conflict":
        add("EVIDENCE_CONFLICT")
    if hard_case_class == "confounder_trap" or goal_code == "confounder_adjustment":
        add("CONFOUNDER_PRESENT")
    if "adjust_confounders" in history_actions:
        add("CONFOUNDER_ADJUSTED")
    if _has(question, "不平衡", "imbalance", "单臂", "single-arm"):
        add("PROJECT_IMBALANCE")
    if "cross_project_validate" in history_actions:
        add("PROJECT_VALIDATED")
    if goal_code == "cross_project_stability" or hard_case_class == "premature_stop":
        add("CROSS_PROJECT_REQUIRED")
    if goal_code == "cross_disease_specificity":
        add("CROSS_DISEASE_REQUIRED")
    if "cross_disease_validate" in history_actions:
        add("CROSS_DISEASE_VALIDATED")
    if "inspect_cohort" in allowed_actions:
        add("METADATA_FIRST")
    if "execute_read_query" in allowed_actions:
        add("BOUNDED_READ_REQUIRED")
    if any(action in history_actions for action in {
        "compare_groups", "stratified_analysis", "adjust_confounders",
        "cross_project_validate", "cross_disease_validate", "analyze_projection",
    }):
        add("ANALYSIS_AVAILABLE")
    return flags[:12]


def derive_task_family(task_kind: TaskKindCode, goal_code: GoalCode, hard_case_class: str | None) -> str:
    return f"{task_kind}:{goal_code}:{hard_case_class or 'standard'}"
