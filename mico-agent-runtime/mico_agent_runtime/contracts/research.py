from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import Field, StringConstraints, TypeAdapter, field_validator, model_validator

from .analysis import (
    AnalysisActionName,
    AnalysisCode,
    AnalysisEvidence,
    AnalysisPlan,
    AnalysisResult,
    AnalysisSourceMetadata,
)
from .base import ClosedModel, Identifier
from .graph_rag import GroundedClaim, ReasoningStep
from .schema_catalog import SchemaSemanticCatalog
from .materialization import QueryPlan, RelationId, SemanticFieldId
from .unified_evidence import UnifiedEvidenceCandidate
from .tools import DynamicSqlText, JavaTransientSnapshotId, Limit20000


ResearchContractVersion = Literal["v1"]
ResearchIntent = Literal[
    "data_fact",
    "focused_comparison",
    "literature_or_relationship",
    "scientific_exploration",
]
ResearchScope = Literal[
    "mico:query:read",
    "mico:evidence:read",
    "mico:research:read",
]

# These are execution capabilities, not diseases, cohorts or analysis methods.
# The question and the model decide the concrete research content.
ScientificActionName = Literal[
    "execute_read_query",
    "inspect_cohort",
    "compare_groups",
    "stratified_analysis",
    "adjust_confounders",
    "cross_project_validate",
    "cross_disease_validate",
    "retrieve_evidence",
    "analyze_projection",
    "finish",
]
ObservationStatus = Literal["VALIDATED", "PARTIAL", "REJECTED", "FAILED"]
ObservationSource = Literal[
    "java_controlled_read",
    "python_bounded_analysis",
    "knowledge_hybrid",
]
SupportStatus = Literal["supported", "speculative", "conflicted", "partial", "unsupported"]
TraceStatus = Literal["RUNNING", "COMPLETED", "FAILED", "REJECTED"]
StopStatus = Literal["CONTINUE", "FINISH"]
StopReasonCode = Literal[
    "EVIDENCE_SUFFICIENT",
    "NO_NEW_INFORMATION",
    "QUALITY_RISK",
    "ACTION_BUDGET_EXHAUSTED",
    "UPSTREAM_REJECTED",
    "UNSUPPORTED_ACTION",
    "USER_REQUESTED_STOP",
]
ResearchLimitationCode = Literal[
    "snapshot_is_transient_and_not_replayable",
    "internal_record_is_not_a_subject_count",
    "subject_link_is_unverified",
    "scientific_evidence_is_not_clinical_diagnosis",
    "unmapped_or_ambiguous_labels_are_not_resolved",
    "dynamic_query_is_java_validated_and_bounded",
    "generated_analysis_is_bounded_and_sandboxed",
    "objective_unavailable_due_to_data",
    "objective_unavailable_due_to_environment",
]


ResearchQuestion = Annotated[str, StringConstraints(min_length=1, max_length=4096)]
QuestionSummary = Annotated[str, StringConstraints(min_length=1, max_length=1200)]
ActionRationale = Annotated[str, StringConstraints(min_length=1, max_length=256)]
PlannerFeedbackCode = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Z][A-Z0-9_]{2,95}$", max_length=96),
]
MetricCode = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z][a-z0-9_]{1,63}$", max_length=64),
]
VersionToken = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$",
    ),
]
Sha256Hash = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]
ActionId = Annotated[str, StringConstraints(pattern=r"^action-[0-9a-f]{32}$")]
ObservationId = Annotated[str, StringConstraints(pattern=r"^observation-[0-9a-f]{32}$")]
FindingId = Annotated[str, StringConstraints(pattern=r"^finding-[0-9a-f]{32}$")]
BindingId = Annotated[str, StringConstraints(pattern=r"^binding-[0-9a-f]{32}$")]
EvidenceReference = Annotated[
    str,
    StringConstraints(pattern=r"^(?:evidence|document|path)-[0-9a-f]{32}$"),
]


def _utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return value.astimezone(timezone.utc)


def _reject_untrusted_action_text(value: str) -> str:
    lowered = value.lower()
    forbidden = (
        "sourcesampleid=",
        "internalrecordid=",
        "cohortcondition",
        "authorization",
        "bearer ",
        "mysql://",
        "postgresql://",
        "patient_data_manager",
    )
    if any(marker in lowered for marker in forbidden):
        raise ValueError("action text contains forbidden runtime content")
    # AnalysisPlan consumes the same free-text goal downstream.  Keep the
    # planner contract at least as strict as that execution contract so a
    # model action cannot validate here and fail later as ANALYSIS_PLAN_INVALID.
    if (
        re.search(r"(?i)https?://|file://|[A-Za-z]:\\\\|^/|\\\\\\\\", value)
        or re.search(r"(?i)\b(?:select|insert|update|delete|drop|alter|create)\b", value)
        or any(marker in lowered for marker in ("{", "}", "[", "]", "\n", "\r", "\t"))
        or re.search(r"(?i)\b(?:srr|err|drr)\d+\b|\bmv_[a-z0-9_-]+\b", value)
    ):
        raise ValueError("action text contains forbidden runtime content")
    return value


class ResearchTask(ClosedModel):
    """Closed input for an open-ended scientific exploration run.

    The question is the variable research content. It is never a SQL or
    Python execution slot. Concrete actions are chosen from generic runtime
    capabilities and remain bounded by Java/Runtime policy.
    """

    dataContractVersion: ResearchContractVersion = "v1"
    runId: Identifier
    taskId: Identifier
    requesterId: Identifier
    traceId: Identifier
    question: ResearchQuestion
    intent: ResearchIntent = "scientific_exploration"
    requestedScopes: list[ResearchScope] = Field(min_length=1, max_length=3)
    allowedActions: list[ScientificActionName] = Field(min_length=1, max_length=10)
    maxActions: int = Field(strict=True, ge=1, le=8, default=6)
    createdAt: datetime

    @field_validator("createdAt")
    @classmethod
    def require_utc_timestamp(cls, value: datetime) -> datetime:
        return _utc_datetime(value)

    @field_validator("requestedScopes", "allowedActions")
    @classmethod
    def reject_duplicate_values(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("duplicate values are not allowed")
        return value


class ResearchPlannerContext(ClosedModel):
    """Redacted context sent to a planner; no raw values or payloads.

    The optional catalog is metadata-only and is supplied by Java; it is not
    a business result or a permission to bypass Java validation.
    """

    questionSummary: QuestionSummary
    intent: ResearchIntent
    approvedActions: list[ScientificActionName] = Field(min_length=1, max_length=10)
    remainingActionBudget: int = Field(strict=True, ge=1, le=8)
    schemaCatalog: SchemaSemanticCatalog | None = None

    @field_validator("questionSummary")
    @classmethod
    def reject_sensitive_summary(cls, value: str) -> str:
        return _reject_untrusted_action_text(value)


class ScientificObservationSummary(ClosedModel):
    """Safe metadata summary supplied to the next planning step."""

    observationId: ObservationId
    actionName: ScientificActionName
    status: ObservationStatus
    source: ObservationSource
    rowCount: int = Field(strict=True, ge=0, le=1000000)
    qualityCodes: list[MetricCode] = Field(default_factory=list, max_length=32)
    versionCodes: list[VersionToken] = Field(default_factory=list, max_length=8)
    # Safe semantic shape of a Dynamic QueryPlan. Physical tables, SQL,
    # filters and returned values never cross the planner boundary.
    queryPlanFields: list[SemanticFieldId] = Field(default_factory=list, max_length=16)
    queryPlanRelationPath: list[RelationId] = Field(default_factory=list, max_length=2)


class ScientificPlannerContext(ClosedModel):
    """Redacted state for one dynamic State → Action decision."""

    questionSummary: QuestionSummary
    intent: ResearchIntent
    approvedActions: list[ScientificActionName] = Field(min_length=1, max_length=10)
    remainingActionBudget: int = Field(strict=True, ge=1, le=8)
    observations: list[ScientificObservationSummary] = Field(default_factory=list, max_length=8)
    # Opaque, allow-listed Runtime feedback for a bounded re-plan.  This is
    # deliberately a code rather than Java error text, SQL, or any data value.
    executionFeedback: list[PlannerFeedbackCode] = Field(default_factory=list, max_length=3)
    # Closed semantic requirements derived from the requested action and the
    # current Java catalog.  These are hints/constraints for the Dynamic
    # Materializer, never physical columns or data values.  They let a
    # cross-validation read request the actual grouping dimension and numeric
    # outcome instead of accepting an unrelated numeric field such as age.
    requiredSemanticFields: list[SemanticFieldId] = Field(default_factory=list, max_length=8)
    requiredGroupField: SemanticFieldId | None = None
    # Runtime-only capability facts used to prevent a materializer from
    # reintroducing a field already proven to have zero cohort coverage.
    # This is not part of ScientificPolicyInput/Decision State.
    unavailableSemanticFields: list[SemanticFieldId] = Field(default_factory=list, max_length=8)
    schemaCatalog: SchemaSemanticCatalog | None = None
    # The Dynamic Scientific materializer may receive the policy-facing
    # six-block state as a read-only view.  It is deliberately a plain JSON
    # payload here to avoid a contracts import cycle (decision_state imports
    # ScientificActionName from this module); the graph creates it only from
    # a validated ScientificDecisionState.  Raw rows, SQL and physical catalog
    # identifiers never belong in this field.
    decisionState: dict[str, Any] | None = None

    @field_validator("questionSummary")
    @classmethod
    def reject_sensitive_question_summary(cls, value: str) -> str:
        return _reject_untrusted_action_text(value)


class ExecuteReadQueryArguments(ClosedModel):
    """Read arguments with a typed-plan path and legacy SQL compatibility.

    New Dynamic Runtime materialization must populate ``queryPlan``.  ``sql``
    remains optional only so historical tool fixtures and the non-scientific
    compatibility boundary can be read without rewriting old assets.
    """

    actionName: Literal["execute_read_query"] = "execute_read_query"
    queryPlan: QueryPlan | None = None
    sql: DynamicSqlText | None = None
    limit: Limit20000 | None = None

    @model_validator(mode="after")
    def require_read_shape(self) -> "ExecuteReadQueryArguments":
        if (self.queryPlan is None) == (self.sql is None):
            raise ValueError("read arguments require exactly one queryPlan or legacy sql")
        if self.queryPlan is not None:
            if self.limit is not None and self.limit != self.queryPlan.limit:
                raise ValueError("query plan limit and read limit must match")
            return self
        normalized = self.sql.lstrip().lower()
        if not normalized.startswith(("select", "with")):
            raise ValueError("only a SELECT or WITH draft may be proposed")
        if ";" in self.sql or "--" in self.sql or "/*" in self.sql:
            raise ValueError("query draft contains unsupported syntax")
        if not re.search(r"\blimit\s+\d+(?:\s+offset\s+\d+)?\s*$", normalized):
            raise ValueError("bounded SQL must end with a literal LIMIT")
        return self


class InspectCohortArguments(ClosedModel):
    """Metadata-first read with typed-plan and legacy SQL compatibility."""

    actionName: Literal["inspect_cohort"] = "inspect_cohort"
    queryPlan: QueryPlan | None = None
    sql: DynamicSqlText | None = None
    limit: Limit20000 | None = None

    @model_validator(mode="after")
    def require_read_shape(self) -> "InspectCohortArguments":
        if (self.queryPlan is None) == (self.sql is None):
            raise ValueError("read arguments require exactly one queryPlan or legacy sql")
        if self.queryPlan is not None:
            if self.limit is not None and self.limit != self.queryPlan.limit:
                raise ValueError("query plan limit and read limit must match")
            return self
        normalized = self.sql.lstrip().lower()
        if not normalized.startswith(("select", "with")):
            raise ValueError("only a SELECT or WITH draft may be proposed")
        if ";" in self.sql or "--" in self.sql or "/*" in self.sql:
            raise ValueError("query draft contains unsupported syntax")
        if not re.search(r"\blimit\s+\d+(?:\s+offset\s+\d+)?\s*$", normalized):
            raise ValueError("bounded SQL must end with a literal LIMIT")
        return self


class ProjectionAnalysisArguments(ClosedModel):
    """Generic analysis over one or more already validated observations."""

    actionName: Literal[
        "compare_groups",
        "stratified_analysis",
        "adjust_confounders",
        "cross_project_validate",
        "cross_disease_validate",
    ]
    observationIds: list[ObservationId] = Field(min_length=1, max_length=8)
    analysisGoal: Annotated[str, StringConstraints(min_length=1, max_length=512)]
    dimensions: list[
        Annotated[
            str,
            StringConstraints(
                pattern=r"^[A-Za-z][A-Za-z0-9_]{0,63}(?:\.[a-z][a-z0-9_]{1,63})?$",
                max_length=128,
            ),
        ]
    ] = Field(default_factory=list, max_length=8)
    confounders: list[
        Annotated[
            str,
            StringConstraints(
                pattern=r"^[A-Za-z][A-Za-z0-9_]{0,63}(?:\.[a-z][a-z0-9_]{1,63})?$",
                max_length=128,
            ),
        ]
    ] = Field(default_factory=list, max_length=8)

    @field_validator("analysisGoal")
    @classmethod
    def reject_sensitive_goal(cls, value: str) -> str:
        return _reject_untrusted_action_text(value)

    @field_validator("observationIds", "dimensions", "confounders")
    @classmethod
    def reject_duplicates(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("analysis references must be unique")
        return value

    @model_validator(mode="after")
    def require_confounders_when_requested(self) -> "ProjectionAnalysisArguments":
        if self.actionName == "adjust_confounders" and not self.confounders:
            raise ValueError("adjust_confounders requires confounder fields")
        if self.actionName == "stratified_analysis" and not self.dimensions:
            raise ValueError("stratified_analysis requires dimensions")
        # A single Java observation is sufficient when it contains the
        # requested group field and numeric outcome: the bounded operator
        # compares the groups present in that observation.  Multiple source
        # observations remain valid for callers that intentionally combine
        # independent bounded reads.
        return self


class RetrieveEvidenceArguments(ClosedModel):
    actionName: Literal["retrieve_evidence"] = "retrieve_evidence"
    topics: list[Annotated[str, StringConstraints(min_length=1, max_length=256)]] = Field(
        min_length=1, max_length=8
    )
    retrievalMode: Literal["vector", "graph", "hybrid"] = "hybrid"
    topK: int = Field(strict=True, ge=1, le=20)
    maxHops: int = Field(strict=True, ge=0, le=3)

    @field_validator("topics")
    @classmethod
    def reject_sensitive_topics(cls, value: list[str]) -> list[str]:
        return [_reject_untrusted_action_text(item) for item in value]


class AnalyzeProjectionArguments(ClosedModel):
    actionName: Literal["analyze_projection"] = "analyze_projection"
    observationId: ObservationId
    analysisGoal: Annotated[str, StringConstraints(min_length=1, max_length=512)]

    @field_validator("analysisGoal")
    @classmethod
    def reject_sensitive_goal(cls, value: str) -> str:
        return _reject_untrusted_action_text(value)


class FinishArguments(ClosedModel):
    actionName: Literal["finish"] = "finish"
    reasonCode: StopReasonCode


class ScientificActionBase(ClosedModel):
    actionId: ActionId
    rationale: ActionRationale

    @field_validator("rationale")
    @classmethod
    def reject_sensitive_rationale(cls, value: str) -> str:
        return _reject_untrusted_action_text(value)


class ExecuteReadQueryAction(ScientificActionBase):
    actionName: Literal["execute_read_query"]
    arguments: ExecuteReadQueryArguments


class InspectCohortAction(ScientificActionBase):
    actionName: Literal["inspect_cohort"]
    arguments: InspectCohortArguments


class CompareGroupsAction(ScientificActionBase):
    actionName: Literal["compare_groups"]
    arguments: ProjectionAnalysisArguments


class StratifiedAnalysisAction(ScientificActionBase):
    actionName: Literal["stratified_analysis"]
    arguments: ProjectionAnalysisArguments


class AdjustConfoundersAction(ScientificActionBase):
    actionName: Literal["adjust_confounders"]
    arguments: ProjectionAnalysisArguments


class CrossProjectValidateAction(ScientificActionBase):
    actionName: Literal["cross_project_validate"]
    arguments: ProjectionAnalysisArguments


class CrossDiseaseValidateAction(ScientificActionBase):
    actionName: Literal["cross_disease_validate"]
    arguments: ProjectionAnalysisArguments


class RetrieveEvidenceAction(ScientificActionBase):
    actionName: Literal["retrieve_evidence"]
    arguments: RetrieveEvidenceArguments


class AnalyzeProjectionAction(ScientificActionBase):
    actionName: Literal["analyze_projection"]
    arguments: AnalyzeProjectionArguments


class FinishAction(ScientificActionBase):
    actionName: Literal["finish"]
    arguments: FinishArguments


ScientificAction: TypeAlias = Annotated[
    ExecuteReadQueryAction
    | InspectCohortAction
    | CompareGroupsAction
    | StratifiedAnalysisAction
    | AdjustConfoundersAction
    | CrossProjectValidateAction
    | CrossDiseaseValidateAction
    | RetrieveEvidenceAction
    | AnalyzeProjectionAction
    | FinishAction,
    Field(discriminator="actionName"),
]
ScientificActionAdapter = TypeAdapter(ScientificAction)


def validate_scientific_action(value: object) -> ScientificAction:
    action = ScientificActionAdapter.validate_python(value)
    arguments = getattr(action, "arguments", None)
    argument_name = getattr(arguments, "actionName", None)
    if argument_name is not None and argument_name != action.actionName:
        raise ValueError("action and argument discriminators must match")
    return action


class ObservationMetric(ClosedModel):
    code: MetricCode
    count: int = Field(strict=True, ge=0)


class Observation(ClosedModel):
    """Metadata-first observation; raw Java rows are never part of this model."""

    observationId: ObservationId
    actionId: ActionId
    actionName: ScientificActionName
    status: ObservationStatus
    source: ObservationSource
    queryHash: Sha256Hash
    rowCount: int = Field(strict=True, ge=0, le=1000000)
    generatedAt: datetime
    schemaVersion: VersionToken
    dataSnapshotId: JavaTransientSnapshotId | None = None
    snapshotPersistence: Literal["transient"] | None = None
    replayable: Literal[False] = False
    featureVersion: VersionToken | None = None
    taxonomyVersion: VersionToken | None = None
    sourceBatch: VersionToken | None = None
    sampleRecordCount: int | None = Field(default=None, strict=True, ge=0, le=1000000)
    sampleKeyCount: int | None = Field(default=None, strict=True, ge=0, le=1000000)
    missingnessSummary: list[ObservationMetric] = Field(default_factory=list, max_length=64)
    exclusionSummary: list[ObservationMetric] = Field(default_factory=list, max_length=64)
    # Retain only the semantic read shape so a later Dynamic Materializer can
    # avoid proposing the same bounded projection for a fresh observation.
    queryPlanFields: list[SemanticFieldId] = Field(default_factory=list, max_length=16)
    queryPlanRelationPath: list[RelationId] = Field(default_factory=list, max_length=2)

    @field_validator("generatedAt")
    @classmethod
    def require_utc_timestamp(cls, value: datetime) -> datetime:
        return _utc_datetime(value)

    @model_validator(mode="after")
    def validate_snapshot_boundary(self) -> "Observation":
        if self.dataSnapshotId is not None and self.snapshotPersistence != "transient":
            raise ValueError("Java snapshots must be explicitly marked transient")
        if self.dataSnapshotId is None and self.snapshotPersistence is not None:
            raise ValueError("snapshotPersistence requires dataSnapshotId")
        return self


class ResearchEvidenceBinding(ClosedModel):
    bindingId: BindingId
    observationId: ObservationId
    evidenceReference: EvidenceReference
    supportStatus: SupportStatus
    source: ObservationSource
    dataSnapshotId: JavaTransientSnapshotId | None = None
    queryHash: Sha256Hash | None = None
    snapshotPersistence: Literal["transient"] | None = None

    @model_validator(mode="after")
    def validate_snapshot_boundary(self) -> "ResearchEvidenceBinding":
        if self.dataSnapshotId is not None and self.snapshotPersistence != "transient":
            raise ValueError("bound snapshots must be marked transient")
        return self


class ResearchFinding(ClosedModel):
    findingId: FindingId
    findingType: Literal[
        "descriptive",
        "association_candidate",
        "quality",
        "literature_support",
    ]
    supportStatus: SupportStatus
    statement: Annotated[str, StringConstraints(min_length=1, max_length=1200)]
    evidenceBindingIds: list[BindingId] = Field(min_length=1, max_length=10)


class StopDecision(ClosedModel):
    status: StopStatus
    reasonCode: StopReasonCode
    nextActionName: ScientificActionName | None = None

    @model_validator(mode="after")
    def validate_next_action(self) -> "StopDecision":
        if self.status == "FINISH" and self.nextActionName is not None:
            raise ValueError("finished runs cannot specify a next action")
        if self.status == "CONTINUE" and self.nextActionName is None:
            raise ValueError("continuing runs require a next action")
        return self


class ResearchTraceEvent(ClosedModel):
    actionId: ActionId
    actionName: ScientificActionName
    status: Literal["STARTED", "COMPLETED", "REJECTED", "FAILED"]
    observationId: ObservationId | None = None
    errorCode: Annotated[
        str,
        StringConstraints(pattern=r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+$", max_length=128),
    ] | None = None
    occurredAt: datetime

    @field_validator("occurredAt")
    @classmethod
    def require_utc_timestamp(cls, value: datetime) -> datetime:
        return _utc_datetime(value)


class ResearchTrace(ClosedModel):
    dataContractVersion: ResearchContractVersion = "v1"
    traceId: Identifier
    runId: Identifier
    taskId: Identifier
    status: TraceStatus
    events: list[ResearchTraceEvent] = Field(default_factory=list, max_length=32)
    stopDecision: StopDecision | None = None
    startedAt: datetime
    endedAt: datetime | None = None

    @field_validator("startedAt", "endedAt")
    @classmethod
    def require_utc_timestamp(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _utc_datetime(value)


class ResearchReasoningStep(ClosedModel):
    stepIndex: int = Field(strict=True, ge=1, le=20)
    actionName: ScientificActionName
    description: Annotated[str, StringConstraints(min_length=1, max_length=1200)]
    supportStatus: SupportStatus
    evidenceBindingIds: list[BindingId] = Field(min_length=1, max_length=10)


class ResearchExplorationReport(ClosedModel):
    """Structured report exposing evidence metadata, not raw data."""

    dataContractVersion: ResearchContractVersion = "v1"
    traceId: Identifier
    runId: Identifier
    taskId: Identifier
    status: Literal["COMPLETED", "FAILED", "REJECTED"]
    findings: list[ResearchFinding] = Field(default_factory=list, max_length=20)
    evidenceBindings: list[ResearchEvidenceBinding] = Field(default_factory=list, max_length=40)
    unifiedEvidence: list[UnifiedEvidenceCandidate] = Field(default_factory=list, max_length=20)
    analysisResults: list[AnalysisResult] = Field(default_factory=list, max_length=20)
    analysisEvidence: list[AnalysisEvidence] = Field(default_factory=list, max_length=40)
    sourceMetadata: list[AnalysisSourceMetadata] = Field(default_factory=list, max_length=40)
    groundedClaims: list[GroundedClaim] = Field(default_factory=list, max_length=20)
    groundedReasoningSteps: list[ReasoningStep] = Field(default_factory=list, max_length=20)
    generationMode: Literal["deterministic_grounded", "model_grounded"] = "deterministic_grounded"
    generationFallbackCode: Annotated[
        str,
        StringConstraints(
            min_length=3,
            max_length=128,
            pattern=r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+$",
        ),
    ] | None = None
    reasoningSteps: list[ResearchReasoningStep] = Field(default_factory=list, max_length=20)
    conclusion: Annotated[str, StringConstraints(min_length=1, max_length=1600)] | None = None
    limitations: list[ResearchLimitationCode] = Field(min_length=1, max_length=8)
    nonDiagnostic: Literal["scientific_evidence_not_clinical_diagnosis"] = (
        "scientific_evidence_not_clinical_diagnosis"
    )

    @model_validator(mode="after")
    def validate_evidence_bindings(self) -> "ResearchExplorationReport":
        binding_ids = {binding.bindingId for binding in self.evidenceBindings}
        for finding in self.findings:
            if not set(finding.evidenceBindingIds).issubset(binding_ids):
                raise ValueError("finding references an unknown evidence binding")
        for step in self.reasoningSteps:
            if not set(step.evidenceBindingIds).issubset(binding_ids):
                raise ValueError("reasoning step references an unknown evidence binding")

        candidate_ids = {candidate.candidateId for candidate in self.unifiedEvidence}
        path_to_evidence: dict[str, str] = {}
        path_statuses: dict[str, set[str]] = {}
        for candidate in self.unifiedEvidence:
            for path in candidate.reasoningPaths:
                if path.pathId in path_to_evidence and path_to_evidence[path.pathId] != candidate.candidateId:
                    raise ValueError("reasoning path is bound to multiple evidence candidates")
                path_to_evidence[path.pathId] = candidate.candidateId
                path_statuses[path.pathId] = {
                    path.status,
                    *(hop.supportStatus for hop in path.hops),
                }

        for generated in [*self.groundedClaims, *self.groundedReasoningSteps]:
            if not set(generated.evidenceIds).issubset(candidate_ids):
                raise ValueError("generated evidence references an unknown candidate")
            if not set(generated.reasoningPathIds).issubset(path_to_evidence):
                raise ValueError("generated reasoning references an unknown path")
            if not all(
                path_to_evidence[path_id] in generated.evidenceIds
                for path_id in generated.reasoningPathIds
            ):
                raise ValueError("generated reasoning path is not bound to its evidence")
            statuses = {
                status
                for path_id in generated.reasoningPathIds
                for status in path_statuses[path_id]
            }
            if generated.supportStatus == "supported" and statuses.intersection({"conflicted", "speculative"}):
                raise ValueError("generated output cannot upgrade uncertain paths")
        return self
