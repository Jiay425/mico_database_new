from __future__ import annotations

from typing import Annotated, ClassVar, Literal

from pydantic import Field, StringConstraints, field_validator

from .analysis import AnalysisGroupResult, AnalysisStratumResult, AnalysisValidationResult
from .base import ClosedModel
from .materialization import AnalysisPlan as TypedAnalysisPlan
from .materialization import SemanticFieldId


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
    # The typed path is used by Dynamic Scientific Runtime.  These fields are
    # optional so the historical intent/runtime compatibility path can keep
    # using the original code planner contract.
    actionName: Annotated[str, StringConstraints(min_length=1, max_length=64)] | None = None
    sourceObservationIds: list[
        Annotated[str, StringConstraints(pattern=r"^observation-[0-9a-f]{32}$", max_length=45)]
    ] = Field(default_factory=list, max_length=8)
    availableSemanticFields: list[SemanticFieldId] = Field(default_factory=list, max_length=64)
    # Required semantic IDs are supplied by Runtime for cross validation so
    # the model cannot silently substitute an unrelated numeric field.
    requiredSemanticFields: list[SemanticFieldId] = Field(default_factory=list, max_length=8)
    requiredGroupField: SemanticFieldId | None = None
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

    allowed_control_character_fields: ClassVar[frozenset[str]] = frozenset({"code"})
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


class GeneratedAnalysisInputBindings(ClosedModel):
    """Runtime-owned identity semantics for one generated-code input.

    Observation identifiers name Runtime objects.  They are deliberately
    separate from opaque per-row sample keys and are never row values that a
    generated program may use as a filter.
    """

    observation_ids: list[
        Annotated[str, StringConstraints(pattern=r"^observation-[0-9a-f]{32}$", max_length=45)]
    ] = Field(default_factory=list, max_length=8)
    row_source: Literal["observation_rows"] = "observation_rows"
    sample_key_column: Annotated[str, StringConstraints(
        pattern=r"^[A-Za-z][A-Za-z0-9_]{0,63}$"
    )] | None = None
    group_column: Annotated[str, StringConstraints(
        pattern=r"^[A-Za-z][A-Za-z0-9_]{0,63}$"
    )] | None = None


class GeneratedAnalysisProgram(ClosedModel):
    """A Runtime-bound, model-authored program ready for the sandbox.

    ``GeneratedAnalysisPlan`` is the legacy model response envelope.  This
    contract is deliberately created by Runtime *after* Capability Registry
    selected ``SUPPORTED_GENERATED``.  It prevents model code from choosing
    its action, input rows, output limits, timeout, or execution mode.
    """

    allowed_control_character_fields: ClassVar[frozenset[str]] = frozenset({"code"})

    action_name: Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
    analysis_goal: SafeAnalysisText
    input_observation_ids: list[
        Annotated[str, StringConstraints(pattern=r"^observation-[0-9a-f]{32}$", max_length=45)]
    ] = Field(min_length=1, max_length=8)
    required_columns: list[Annotated[str, StringConstraints(
        pattern=r"^[A-Za-z][A-Za-z0-9_]{0,63}$"
    )]] = Field(min_length=1, max_length=64)
    code: AnalysisCodeText
    expected_outputs: list[Literal[
        "metrics", "group_results", "stratum_results", "validation_results", "feature_results"
    ]] = Field(min_length=1, max_length=5)
    metrics_schema: list[Annotated[str, StringConstraints(
        pattern=r"^[a-z][a-z0-9_]{0,63}$"
    )]] = Field(default_factory=list, max_length=32)
    timeout_seconds: int = Field(strict=True, ge=1, le=10, default=5)
    max_rows: int = Field(strict=True, ge=1, le=1000, default=1000)
    max_output_bytes: int = Field(strict=True, ge=1024, le=20000, default=20000)
    max_memory_mb: int = Field(strict=True, ge=64, le=1024, default=256)
    input_bindings: GeneratedAnalysisInputBindings = Field(
        default_factory=GeneratedAnalysisInputBindings
    )

    @field_validator("input_observation_ids", "required_columns", "expected_outputs", "metrics_schema")
    @classmethod
    def reject_duplicate_values(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("generated analysis program contains duplicate values")
        return value

    @field_validator("code")
    @classmethod
    def validate_program_code(cls, value: str) -> str:
        # Keep the legacy model response and the runtime-bound program under
        # the same source-code safety policy.
        return GeneratedAnalysisPlan(language="python", analysisType="generated", code=value).code


class GeneratedAnalysisPlannerResult(ClosedModel):
    plan: GeneratedAnalysisPlan
    mode: Literal["model", "deterministic"]
    fallbackCode: str | None = None


class TypedAnalysisPlannerResult(ClosedModel):
    """Materializer output for the Dynamic Scientific Runtime path.

    The plan contains only semantic IDs and opaque observation references.
    It never contains Python source, SQL, credentials, or raw data values.
    """

    plan: TypedAnalysisPlan
    mode: Literal["model", "deterministic"]
    fallbackCode: str | None = None
    repairCodes: list[AnalysisFeedbackCode] = Field(default_factory=list, max_length=3)


class GeneratedAnalysisFeature(ClosedModel):
    featureName: SafeAnalysisText
    metricName: Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
    metricValue: float
    group: SafeAnalysisText | None = None


class GeneratedAnalysisFeatureResult(ClosedModel):
    """One feature-specific result for a multi-feature analysis."""

    featureName: SafeAnalysisText
    # A feature can be scientifically ineligible even when other features in
    # the same multi-feature operation succeed (for example a singular
    # covariate design).  Preserve that fact instead of dropping the feature
    # or failing the complete analysis.
    status: Literal["supported", "insufficient_data", "failed"] = "supported"
    metrics: dict[
        Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,63}$")], float
    ] = Field(default_factory=dict, max_length=32)
    group_results: list[AnalysisGroupResult] = Field(default_factory=list, max_length=8)
    stratum_results: list[AnalysisStratumResult] = Field(default_factory=list, max_length=64)
    adjusted_covariates: list[SemanticFieldId] = Field(default_factory=list, max_length=16)


class GeneratedAnalysisResult(ClosedModel):
    status: Literal["COMPLETED"]
    analysisType: Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
    execution_mode: Literal["typed", "generated"] = "generated"
    method_used: SafeAnalysisText | None = None
    plannerMode: Literal["model", "deterministic"]
    codeVersion: Literal["sandbox-python-v1", "typed-analysis-operator-v1"]
    rowCount: int = Field(ge=0, le=20_000)
    metrics: dict[
        Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,63}$")], float
    ] = Field(default_factory=dict, max_length=32)
    group_results: list[AnalysisGroupResult] = Field(default_factory=list, max_length=64)
    stratum_results: list[AnalysisStratumResult] = Field(default_factory=list, max_length=64)
    validation_results: list[AnalysisValidationResult] = Field(default_factory=list, max_length=64)
    feature_results: list[GeneratedAnalysisFeatureResult] = Field(
        default_factory=list, max_length=64
    )
    # Deterministic, auditable ordering used after all feature-level tests
    # have received BH-FDR q-values.  It is absent for single-feature plans.
    ranking_method: SafeAnalysisText | None = None
    adjusted_covariates: list[SemanticFieldId] = Field(default_factory=list, max_length=16)
    used_row_count: int = Field(strict=True, ge=0, le=1000000, default=0)
    dropped_row_count: int = Field(strict=True, ge=0, le=1000000, default=0)
    topFeatures: list[GeneratedAnalysisFeature] = Field(default_factory=list, max_length=20)
    # Runtime, never the model, determines whether a generated output is
    # scientifically meaningful.  Successful sandbox execution alone is not
    # a scientific result.
    scientific_result_valid: bool = False
    scientific_conclusion_eligible: bool = False
    warnings: list[AnalysisFeedbackCode] = Field(default_factory=list, max_length=16)
    program_hash: Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")] | None = None
    code_hash: Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")] | None = None
    limitations: list[Literal[
        "generated_code_was_sandbox_validated",
        "typed_plan_executed_by_approved_operator",
        "preview_was_redacted_before_model_access",
        "snapshot_is_transient_and_not_replayable",
        "analysis_is_not_a_clinical_conclusion",
    ]] = Field(min_length=1)
