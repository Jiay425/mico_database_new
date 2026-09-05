"""Closed, serializable state contract exposed to the scientific Decision Policy.

``ScientificState`` remains the LangGraph execution state.  This module is a
deliberately smaller public contract for the future Qwen decision boundary;
raw observations and execution payloads do not belong here.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StringConstraints, field_validator

from .base import ClosedModel
from .research import ScientificActionName


ScientificObjective = Literal[
    "group_comparison",
    "projection_analysis",
    "stratified_analysis",
    "confounder_assessment",
    "cross_project_validation",
    "cross_disease_validation",
    "evidence_support",
]
AnalysisStatus = Literal[
    "not_started",
    "running",
    "completed",
    "failed",
    "insufficient_data",
]
ImbalanceLevel = Literal["unknown", "low", "medium", "high"]
MissingnessLevel = Literal["unknown", "low", "medium", "high"]
SampleSizeLevel = Literal["unknown", "insufficient", "limited", "sufficient"]
HeterogeneityLevel = Literal["unknown", "low", "medium", "high"]
EvidenceStatus = Literal["not_started", "completed", "failed", "insufficient"]
EvidenceConsistency = Literal[
    "unknown",
    "supportive",
    "mostly_supportive",
    "mixed",
    "conflicted",
    "insufficient",
]

_StateText = Annotated[str, StringConstraints(min_length=1, max_length=512)]
_NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]


def _reject_duplicates(value: list[str]) -> list[str]:
    if len(value) != len(set(value)):
        raise ValueError("decision state list values must be unique")
    return value


class ScientificTaskConstraints(ClosedModel):
    disease_groups: list[_StateText] = Field(default_factory=list)
    target_features: list[_StateText] = Field(default_factory=list)
    focus_covariates: list[_StateText] = Field(default_factory=list)
    requested_projects: list[_StateText] = Field(default_factory=list)
    requested_stratifiers: list[_StateText] = Field(default_factory=list)

    @field_validator(
        "disease_groups",
        "target_features",
        "focus_covariates",
        "requested_projects",
        "requested_stratifiers",
    )
    @classmethod
    def reject_duplicate_constraints(cls, value: list[str]) -> list[str]:
        return _reject_duplicates(value)


class ScientificTaskState(ClosedModel):
    query: Annotated[str, StringConstraints(min_length=1, max_length=4096)]
    objectives: list[ScientificObjective] = Field(default_factory=list)
    constraints: ScientificTaskConstraints = Field(default_factory=ScientificTaskConstraints)

    @field_validator("objectives")
    @classmethod
    def reject_duplicate_objectives(cls, value: list[ScientificObjective]) -> list[ScientificObjective]:
        return _reject_duplicates(value)


class GroupDataState(ClosedModel):
    group_field: _StateText | None = None
    group_count: _NonNegativeInt = 0
    group_sizes: dict[_StateText, _NonNegativeInt] = Field(default_factory=dict)


class ProjectDataState(ClosedModel):
    has_project_field: bool = False
    project_count: _NonNegativeInt = 0


class CovariateDataState(ClosedModel):
    available_covariates: list[_StateText] = Field(default_factory=list)
    imbalance: dict[_StateText, ImbalanceLevel] = Field(default_factory=dict)

    @field_validator("available_covariates")
    @classmethod
    def reject_duplicate_covariates(cls, value: list[str]) -> list[str]:
        return _reject_duplicates(value)


class DataQualityState(ClosedModel):
    missingness_level: MissingnessLevel = "unknown"
    sample_size_level: SampleSizeLevel = "unknown"


class ScientificDataState(ClosedModel):
    has_tabular_data: bool = False
    row_count: _NonNegativeInt = 0
    # These counts are deliberately distinct from joined result rows.  A
    # sample count is populated only when Java supplies its run-scoped opaque
    # analysis key; patient_count remains unknown when the sensitive patient
    # identity is not part of the runtime projection.
    sample_count: _NonNegativeInt | None = None
    patient_count: _NonNegativeInt | None = None
    feature_count: _NonNegativeInt | None = None
    available_dimensions: list[_StateText] = Field(default_factory=list)
    # Full semantic IDs for numeric fields explicitly capable of serving as
    # scientific outcomes. Numeric covariates such as age are not included.
    available_outcomes: list[_StateText] = Field(default_factory=list)
    group_state: GroupDataState = Field(default_factory=GroupDataState)
    project_state: ProjectDataState = Field(default_factory=ProjectDataState)
    covariate_state: CovariateDataState = Field(default_factory=CovariateDataState)
    data_quality: DataQualityState = Field(default_factory=DataQualityState)

    @field_validator("available_dimensions", "available_outcomes")
    @classmethod
    def reject_duplicate_dimensions(cls, value: list[str]) -> list[str]:
        return _reject_duplicates(value)


class GroupComparisonState(ClosedModel):
    status: AnalysisStatus = "not_started"
    mean_difference: float | None = None
    effect_size: float | None = None
    p_value: float | None = None
    confidence_interval_low: float | None = None
    confidence_interval_high: float | None = None


class ConfounderAdjustmentState(ClosedModel):
    status: AnalysisStatus = "not_started"
    adjusted_group_effect: float | None = None
    adjusted_effect_size: float | None = None
    adjusted_p_value: float | None = None
    adjusted_covariates: list[_StateText] = Field(default_factory=list)
    used_row_count: _NonNegativeInt = 0
    dropped_row_count: _NonNegativeInt = 0

    @field_validator("adjusted_covariates")
    @classmethod
    def reject_duplicate_adjusted_covariates(cls, value: list[str]) -> list[str]:
        return _reject_duplicates(value)


class CrossProjectValidationState(ClosedModel):
    status: AnalysisStatus = "not_started"
    project_count: _NonNegativeInt = 0
    positive_project_count: _NonNegativeInt = 0
    negative_project_count: _NonNegativeInt = 0
    neutral_project_count: _NonNegativeInt = 0
    effect_min: float | None = None
    effect_max: float | None = None
    heterogeneity: HeterogeneityLevel = "unknown"


class StratifiedAnalysisState(ClosedModel):
    status: AnalysisStatus = "not_started"
    stratify_fields: list[_StateText] = Field(default_factory=list)
    stratum_count: _NonNegativeInt = 0
    positive_stratum_count: _NonNegativeInt = 0
    negative_stratum_count: _NonNegativeInt = 0
    neutral_stratum_count: _NonNegativeInt = 0
    effect_min: float | None = None
    effect_max: float | None = None
    heterogeneity: HeterogeneityLevel = "unknown"

    @field_validator("stratify_fields")
    @classmethod
    def reject_duplicate_stratify_fields(cls, value: list[str]) -> list[str]:
        return _reject_duplicates(value)


class ProjectionAnalysisState(ClosedModel):
    status: AnalysisStatus = "not_started"
    result_count: _NonNegativeInt = 0


class CrossDiseaseValidationState(ClosedModel):
    status: AnalysisStatus = "not_started"
    disease_count: _NonNegativeInt = 0
    heterogeneity: HeterogeneityLevel = "unknown"


class ScientificAnalysisState(ClosedModel):
    group_comparison: GroupComparisonState = Field(default_factory=GroupComparisonState)
    confounder_adjustment: ConfounderAdjustmentState = Field(default_factory=ConfounderAdjustmentState)
    cross_project_validation: CrossProjectValidationState = Field(default_factory=CrossProjectValidationState)
    stratified_analysis: StratifiedAnalysisState = Field(default_factory=StratifiedAnalysisState)
    projection_analysis: ProjectionAnalysisState = Field(default_factory=ProjectionAnalysisState)
    cross_disease_validation: CrossDiseaseValidationState = Field(default_factory=CrossDiseaseValidationState)


class ScientificEvidenceState(ClosedModel):
    status: EvidenceStatus = "not_started"
    evidence_count: _NonNegativeInt = 0
    support_count: _NonNegativeInt = 0
    conflict_count: _NonNegativeInt = 0
    context_count: _NonNegativeInt = 0
    consistency: EvidenceConsistency = "unknown"


class ScientificProgressState(ClosedModel):
    completed_actions: list[ScientificActionName] = Field(default_factory=list)
    action_counts: dict[ScientificActionName, _NonNegativeInt] = Field(default_factory=dict)
    last_action: ScientificActionName | None = None
    remaining_objectives: list[ScientificObjective] = Field(default_factory=list)
    action_count: _NonNegativeInt = 0

    @field_validator("completed_actions")
    @classmethod
    def reject_duplicate_completed_actions(cls, value: list[ScientificActionName]) -> list[ScientificActionName]:
        return _reject_duplicates(value)

    @field_validator("remaining_objectives")
    @classmethod
    def reject_duplicate_remaining_objectives(cls, value: list[ScientificObjective]) -> list[ScientificObjective]:
        return _reject_duplicates(value)


class ScientificActionSpaceState(ClosedModel):
    available_actions: list[ScientificActionName] = Field(default_factory=list)

    @field_validator("available_actions")
    @classmethod
    def reject_duplicate_available_actions(cls, value: list[ScientificActionName]) -> list[ScientificActionName]:
        return _reject_duplicates(value)


class ScientificDecisionState(ClosedModel):
    """The only state object intended to cross the future Qwen policy boundary."""

    task: ScientificTaskState
    data_state: ScientificDataState = Field(default_factory=ScientificDataState)
    analysis_state: ScientificAnalysisState = Field(default_factory=ScientificAnalysisState)
    evidence_state: ScientificEvidenceState = Field(default_factory=ScientificEvidenceState)
    progress: ScientificProgressState = Field(default_factory=ScientificProgressState)
    action_space: ScientificActionSpaceState = Field(default_factory=ScientificActionSpaceState)


__all__ = [
    "AnalysisStatus",
    "ConfounderAdjustmentState",
    "CovariateDataState",
    "CrossDiseaseValidationState",
    "CrossProjectValidationState",
    "DataQualityState",
    "EvidenceConsistency",
    "EvidenceStatus",
    "GroupComparisonState",
    "GroupDataState",
    "HeterogeneityLevel",
    "ImbalanceLevel",
    "MissingnessLevel",
    "ProjectDataState",
    "ProjectionAnalysisState",
    "SampleSizeLevel",
    "ScientificActionSpaceState",
    "ScientificAnalysisState",
    "ScientificDataState",
    "ScientificDecisionState",
    "ScientificEvidenceState",
    "ScientificObjective",
    "ScientificProgressState",
    "ScientificTaskConstraints",
    "ScientificTaskState",
    "StratifiedAnalysisState",
]
