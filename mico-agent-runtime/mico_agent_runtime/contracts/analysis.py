from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, field_validator, model_validator

from .base import ClosedModel
from .tools import JavaTransientSnapshotId


AnalysisActionName = Literal[
    "inspect_cohort",
    "compare_groups",
    "stratified_analysis",
    "adjust_confounders",
    "cross_project_validate",
    "cross_disease_validate",
    "retrieve_evidence",
    "analyze_projection",
]
AnalysisStatus = Literal["COMPLETED", "PARTIAL", "REJECTED", "FAILED"]
AnalysisSupportStatus = Literal["supported", "speculative", "conflicted", "partial", "unsupported"]
AnalysisObservationId = Annotated[
    str,
    StringConstraints(pattern=r"^observation-[0-9a-f]{32}$"),
]
AnalysisId = Annotated[str, StringConstraints(pattern=r"^analysis-[0-9a-f]{32}$")]
AnalysisEvidenceId = Annotated[str, StringConstraints(pattern=r"^evidence-[0-9a-f]{32}$")]
AnalysisHash = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]
AnalysisFieldName = Annotated[
    str,
    StringConstraints(
        pattern=r"^[A-Za-z][A-Za-z0-9_]{0,63}(?:\.[a-z][a-z0-9_]{1,63})?$",
        max_length=128,
    ),
]
AnalysisCode = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z][a-z0-9_]{1,63}$", max_length=64),
]
AnalysisGoal = Annotated[str, StringConstraints(min_length=1, max_length=512)]
AnalysisVersion = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$",
    ),
]
AnalysisResultText = Annotated[str, StringConstraints(min_length=1, max_length=512)]


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return value.astimezone(timezone.utc)


def _reject_sensitive_text(value: str) -> str:
    lowered = value.lower()
    forbidden = (
        "sourcesampleid=",
        "internalrecordid=",
        "cohortcondition",
        "patient_data_manager",
        "authorization",
        "bearer ",
        "mysql://",
        "postgresql://",
        "ssh://",
    )
    if any(marker in lowered for marker in forbidden):
        raise ValueError("analysis text contains forbidden runtime content")
    if (
        re.search(r"(?i)https?://|file://|[A-Za-z]:\\|^/|\\\\", value)
        or re.search(r"(?i)\b(?:select|insert|update|delete|drop|alter|create)\b", value)
        or any(marker in lowered for marker in ("{", "}", "[", "]", "\n", "\r", "\t"))
        or re.search(r"(?i)\b(?:srr|err|drr)\d+\b|\bmv_[a-z0-9_-]+\b", value)
    ):
        raise ValueError("analysis text contains forbidden runtime content")
    return value


class AnalysisPlan(ClosedModel):
    """A generic scientific operation plan, never a disease-specific script."""

    analysisId: AnalysisId
    actionName: AnalysisActionName
    sourceObservationIds: list[AnalysisObservationId] = Field(default_factory=list, max_length=8)
    analysisGoal: AnalysisGoal
    dimensions: list[AnalysisFieldName] = Field(default_factory=list, max_length=8)
    confounders: list[AnalysisFieldName] = Field(default_factory=list, max_length=8)
    methodCode: AnalysisCode = "model_selected"
    outputLimit: int = Field(strict=True, ge=1, le=20, default=20)

    @field_validator("analysisGoal")
    @classmethod
    def reject_sensitive_goal(cls, value: str) -> str:
        return _reject_sensitive_text(value)

    @field_validator("sourceObservationIds", "dimensions", "confounders")
    @classmethod
    def reject_duplicates(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("analysis references must be unique")
        return value

    @model_validator(mode="after")
    def require_sources_for_projection(self) -> "AnalysisPlan":
        if self.actionName != "inspect_cohort" and not self.sourceObservationIds:
            raise ValueError("analysis action requires a source observation")
        # Cross validation may operate on one grouped tabular observation;
        # the typed operator compares the project/disease groups in its rows.
        # Multiple source observations are still accepted when the caller has
        # separate bounded reads to combine.
        if self.actionName == "stratified_analysis" and not self.dimensions:
            raise ValueError("stratified analysis requires at least one dimension")
        if self.actionName == "adjust_confounders" and not self.confounders:
            raise ValueError("confounder analysis requires confounder fields")
        return self


class AnalysisEvidence(ClosedModel):
    """Source binding for a derived analysis result."""

    evidenceId: AnalysisEvidenceId
    analysisId: AnalysisId
    observationId: AnalysisObservationId
    source: Literal["java_controlled_read", "python_bounded_analysis"]
    dataSnapshotId: JavaTransientSnapshotId
    snapshotPersistence: Literal["transient"]
    queryHash: AnalysisHash
    schemaVersion: AnalysisVersion
    rowCount: int = Field(strict=True, ge=0, le=1000000)
    generatedAt: datetime
    supportStatus: AnalysisSupportStatus = "supported"

    @field_validator("generatedAt")
    @classmethod
    def require_utc(cls, value: datetime) -> datetime:
        return _utc(value)


class AnalysisFeatureResult(ClosedModel):
    featureName: Annotated[str, StringConstraints(min_length=1, max_length=256)]
    status: Literal["supported", "insufficient_data", "failed"] = "supported"
    metrics: dict[AnalysisCode, float] = Field(default_factory=dict, max_length=16)
    supportStatus: AnalysisSupportStatus = "supported"

    @field_validator("featureName")
    @classmethod
    def reject_sensitive_feature(cls, value: str) -> str:
        checked = _reject_sensitive_text(value)
        if any(separator in checked for separator in ("|", ";", "=")):
            raise ValueError("feature name must be a single display value")
        return checked


class AnalysisGroupResult(ClosedModel):
    """One group summary from a typed two-group comparison."""

    group: AnalysisResultText
    n: int = Field(strict=True, ge=0, le=1000000)
    mean: float


class AnalysisStratumResult(ClosedModel):
    """One within-stratum two-group comparison."""

    stratum: AnalysisResultText
    feature_name: AnalysisResultText | None = None
    group_a_n: int = Field(strict=True, ge=0, le=1000000)
    group_b_n: int = Field(strict=True, ge=0, le=1000000)
    mean_difference: float
    p_value: float = Field(ge=0.0, le=1.0)
    # ``p_value`` remains the raw per-stratum value for backwards
    # compatibility.  Numeric stratification additionally records the
    # multiple-testing corrected value explicitly; categorical legacy
    # results leave these optional aliases unset.
    raw_p_value: float | None = Field(default=None, ge=0.0, le=1.0)
    adjusted_p_value: float | None = Field(default=None, ge=0.0, le=1.0)
    q_value: float | None = Field(default=None, ge=0.0, le=1.0)
    direction: Literal["positive", "negative", "neutral"]


class AnalysisValidationResult(ClosedModel):
    """One repeated validation-dimension comparison."""

    validation_value: AnalysisResultText
    group_a_n: int = Field(strict=True, ge=0, le=1000000)
    group_b_n: int = Field(strict=True, ge=0, le=1000000)
    mean_difference: float
    p_value: float = Field(ge=0.0, le=1.0)
    direction: Literal["positive", "negative", "neutral"]


class AnalysisResult(ClosedModel):
    """Safe, bounded output of one dynamic analysis operation."""

    analysisId: AnalysisId
    actionName: AnalysisActionName
    analysis_type: str | None = None
    execution_mode: Literal["typed", "generated"] = "generated"
    method_used: AnalysisResultText | None = None
    status: AnalysisStatus
    plannerMode: Literal["model", "deterministic"]
    codeVersion: Literal["sandbox-python-v1", "typed-analysis-operator-v1"]
    sourceObservationIds: list[AnalysisObservationId] = Field(min_length=1, max_length=8)
    rowsAnalyzed: int = Field(strict=True, ge=0, le=1000000)
    metrics: dict[AnalysisCode, float] = Field(default_factory=dict, max_length=32)
    group_results: list[AnalysisGroupResult] = Field(default_factory=list, max_length=64)
    stratum_results: list[AnalysisStratumResult] = Field(default_factory=list, max_length=64)
    validation_results: list[AnalysisValidationResult] = Field(default_factory=list, max_length=64)
    feature_results: list[AnalysisFeatureResult] = Field(default_factory=list, max_length=64)
    ranking_method: Annotated[str, StringConstraints(min_length=1, max_length=256)] | None = None
    adjusted_covariates: list[AnalysisFieldName] = Field(default_factory=list, max_length=16)
    used_row_count: int = Field(strict=True, ge=0, le=1000000, default=0)
    dropped_row_count: int = Field(strict=True, ge=0, le=1000000, default=0)
    # Execution success and conclusion eligibility are separate. Generated
    # analyses begin exploratory until Runtime's independent validator can
    # establish a stronger action-specific scientific claim.
    scientific_result_valid: bool = False
    scientific_conclusion_eligible: bool = False
    warnings: list[Annotated[str, StringConstraints(
        pattern=r"^[A-Z][A-Z0-9_]{2,95}$"
    )]] = Field(default_factory=list, max_length=16)
    generated_code_hash: AnalysisHash | None = None
    generated_program_hash: AnalysisHash | None = None
    topFeatures: list[AnalysisFeatureResult] = Field(default_factory=list, max_length=20)
    evidence: list[AnalysisEvidence] = Field(min_length=1, max_length=8)
    limitations: list[Literal[
        "generated_code_was_sandbox_validated",
        "typed_plan_executed_by_approved_operator",
        "input_projection_was_bounded",
        "snapshot_is_transient_and_not_replayable",
        "analysis_is_not_a_clinical_conclusion",
        "method_was_selected_at_runtime",
    ]] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def validate_evidence(self) -> "AnalysisResult":
        evidence_ids = {item.observationId for item in self.evidence}
        if not set(self.sourceObservationIds).issubset(evidence_ids):
            raise ValueError("analysis result must bind every source observation")
        return self


class AnalysisSourceMetadata(ClosedModel):
    """Public source metadata; no raw rows, identifiers or locator values."""

    observationId: AnalysisObservationId
    source: Literal["java_controlled_read", "python_bounded_analysis", "knowledge_hybrid"]
    dataSnapshotId: JavaTransientSnapshotId | None = None
    snapshotPersistence: Literal["transient"] | None = None
    queryHash: AnalysisHash
    schemaVersion: AnalysisVersion
    rowCount: int = Field(strict=True, ge=0, le=1000000)
    generatedAt: datetime
    taxonomyVersion: AnalysisVersion | None = None
    featureVersion: AnalysisVersion | None = None
    sourceBatch: AnalysisVersion | None = None

    @field_validator("generatedAt")
    @classmethod
    def require_utc(cls, value: datetime) -> datetime:
        return _utc(value)

    @model_validator(mode="after")
    def validate_snapshot(self) -> "AnalysisSourceMetadata":
        if self.dataSnapshotId is not None and self.snapshotPersistence != "transient":
            raise ValueError("snapshot metadata must remain transient")
        if self.dataSnapshotId is None and self.snapshotPersistence is not None:
            raise ValueError("snapshot persistence requires a snapshot id")
        return self


__all__ = [
    "AnalysisActionName",
    "AnalysisEvidence",
    "AnalysisFeatureResult",
    "AnalysisGroupResult",
    "AnalysisPlan",
    "AnalysisResult",
    "AnalysisResultText",
    "AnalysisSourceMetadata",
    "AnalysisStratumResult",
    "AnalysisValidationResult",
]
