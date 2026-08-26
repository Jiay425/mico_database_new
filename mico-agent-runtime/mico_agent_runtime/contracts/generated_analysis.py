from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StringConstraints, field_validator

from .base import ClosedModel


SafeAnalysisText = Annotated[str, StringConstraints(min_length=1, max_length=256)]
AnalysisCodeText = Annotated[str, StringConstraints(min_length=1, max_length=12000)]
AnalysisFeedbackCode = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Z][A-Z0-9_]{2,95}$", max_length=96),
]


class AnalysisPlannerContext(ClosedModel):
    """脱敏后的前几行数据，只用于让模型选择分析表达式。"""

    questionSummary: Annotated[str, StringConstraints(min_length=1, max_length=1200)]
    workflow: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    columns: list[Annotated[str, StringConstraints(min_length=1, max_length=64)]] = Field(
        max_length=64
    )
    previewRows: list[dict[str, object]] = Field(max_length=20)
    # Opaque Runtime feedback after a sandbox rejection.  It is never source
    # code, a data value, an exception message, or an execution environment.
    executionFeedback: list[AnalysisFeedbackCode] = Field(default_factory=list, max_length=3)

    @field_validator("executionFeedback")
    @classmethod
    def reject_duplicate_feedback(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("duplicate analysis feedback is not allowed")
        return value


class GeneratedAnalysisPlan(ClosedModel):
    """Closed model output; Python code is still untrusted until sandbox validation."""

    language: Literal["python"]
    analysisType: Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
    code: AnalysisCodeText

    @field_validator("code")
    @classmethod
    def reject_transport_content(cls, value: str) -> str:
        lowered = value.lower()
        forbidden = (
            "sourceSampleId", "internalRecordId", "cohortCondition", "authorization",
            "bearer ", "mysql", "ssh", "http://", "https://",
            "import ", "__import__", "eval(", "exec(", "open(",
        )
        if any(token.lower() in lowered for token in forbidden):
            raise ValueError("analysis code contains forbidden runtime content")
        if "result" not in value:
            raise ValueError("analysis code must assign result")
        return value


class GeneratedAnalysisPlannerResult(ClosedModel):
    plan: GeneratedAnalysisPlan
    mode: Literal["model", "deterministic"]
    fallbackCode: str | None = None


class GeneratedAnalysisFeature(ClosedModel):
    featureName: SafeAnalysisText
    metricName: Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
    metricValue: float
    group: SafeAnalysisText | None = None


class GeneratedAnalysisResult(ClosedModel):
    status: Literal["COMPLETED"]
    analysisType: Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
    plannerMode: Literal["model", "deterministic"]
    codeVersion: Literal["sandbox-python-v1"]
    rowCount: int = Field(ge=0, le=1000)
    metrics: dict[
        Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,63}$")], float
    ] = Field(default_factory=dict, max_length=32)
    topFeatures: list[GeneratedAnalysisFeature] = Field(default_factory=list, max_length=20)
    limitations: list[Literal[
        "generated_code_was_sandbox_validated",
        "preview_was_redacted_before_model_access",
        "snapshot_is_transient_and_not_replayable",
        "analysis_is_not_a_clinical_conclusion",
    ]] = Field(min_length=1)
