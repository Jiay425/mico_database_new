from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from pydantic import Field, StringConstraints, field_validator, model_validator

from .audit import AuditEvent
from .generated_analysis import GeneratedAnalysisResult
from .metrics import RuntimeMetrics
from .base import ClosedModel, Identifier
from .tools import AllowedScope, ToolStatus


ApprovedWorkflow = Literal["dynamic_read_query", "knowledge_retrieval"]
IntentRetrievalMode = Literal["vector", "graph", "hybrid"]
IntentQueryType = Literal["semantic_fact", "relation", "multi_hop", "composite"]
IntentRetrievalBranch = Literal["vector", "graph"]

IntentQuestion = Annotated[str, StringConstraints(min_length=1, max_length=4096)]
PlannerQuestionSummary = Annotated[str, StringConstraints(min_length=1, max_length=1200)]
PlannerSqlDraft = Annotated[str, StringConstraints(min_length=1, max_length=16000)]


class IntentTaskRequest(ClosedModel):
    """Closed browser-to-runtime research request.

    The question is used for intent recognition only. Tool arguments are built
    by the selected workflow from fixed contracts and verified Java results.
    """

    runId: Identifier
    taskId: Identifier
    requesterId: Identifier
    requestedScopes: list[AllowedScope] = Field(min_length=1, max_length=8)
    question: IntentQuestion
    allowedWorkflows: list[ApprovedWorkflow] = Field(min_length=1, max_length=2)
    createdAt: datetime
    traceId: Identifier

    @field_validator("createdAt")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("createdAt must include a timezone")
        return value.astimezone(timezone.utc)

    @field_validator("requestedScopes", "allowedWorkflows")
    @classmethod
    def reject_duplicates(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("duplicate values are not allowed")
        return value


class IntentPlannerContext(ClosedModel):
    """Only redacted intent context is allowed to reach a model port."""

    questionSummary: PlannerQuestionSummary
    allowedWorkflows: list[ApprovedWorkflow] = Field(min_length=1, max_length=7)


class IntentRoutePlan(ClosedModel):
    workflow: ApprovedWorkflow
    responseMode: Literal["structured_analysis_job", "structured_evidence_review"]
    safetyProfile: Literal["non_diagnostic"]
    retrievalMode: IntentRetrievalMode = "hybrid"
    queryType: IntentQueryType = "semantic_fact"
    routeConfidence: float = Field(default=0.0, ge=0.0, le=1.0)
    retrievalBranches: list[IntentRetrievalBranch] = Field(default_factory=list, max_length=2)
    classificationSignals: list[Literal[
        "semantic", "relation", "multi_hop", "composite", "deterministic", "model"
    ]] = Field(default_factory=list, max_length=6)
    # Untrusted model proposal. Java execute_read_query is the only component
    # allowed to validate or execute this text.
    sqlDraft: PlannerSqlDraft | None = None

    @model_validator(mode="after")
    def bind_sql_draft_to_dynamic_workflow(self) -> "IntentRoutePlan":
        expected_branches = {
            "vector": ["vector"],
            "graph": ["graph"],
            "hybrid": ["vector", "graph"],
        }[self.retrievalMode] if self.workflow == "knowledge_retrieval" else []
        if not self.retrievalBranches:
            object.__setattr__(self, "retrievalBranches", expected_branches)
        if self.workflow == "knowledge_retrieval":
            if self.retrievalBranches != expected_branches:
                raise ValueError("retrievalBranches must match retrievalMode")
        elif self.retrievalBranches:
            raise ValueError("retrievalBranches is only valid for knowledge workflow")
        if self.sqlDraft is not None and self.workflow != "dynamic_read_query":
            raise ValueError("sqlDraft is only valid for a dynamic read workflow")
        if self.workflow == "dynamic_read_query" and self.responseMode != "structured_analysis_job":
            raise ValueError("dynamic read workflow requires structured analysis mode")
        if self.workflow == "knowledge_retrieval" and self.responseMode != "structured_evidence_review":
            raise ValueError("knowledge workflow requires structured evidence mode")
        if self.workflow != "knowledge_retrieval" and self.retrievalMode != "hybrid":
            raise ValueError("retrievalMode is only valid for knowledge workflow")
        if self.workflow == "dynamic_read_query" and self.sqlDraft is not None:
            if not self.sqlDraft.lstrip().lower().startswith(("select", "with")):
                raise ValueError("sqlDraft must be a read query draft")
        return self


class IntentQueryReport(ClosedModel):
    """Safe metadata-only report for a model-proposed Java read query."""

    status: Literal["COMPLETED"]
    workflow: Literal["dynamic_read_query"]
    source: str
    rowCount: int = Field(ge=0)
    columns: list[Annotated[str, StringConstraints(min_length=1, max_length=256)]] = Field(
        max_length=128
    )
    dataSnapshotId: str
    snapshotPersistence: Literal["transient"]
    queryHash: str
    generatedAt: datetime
    schemaVersion: str
    limitations: list[Literal[
        "query_is_model_proposed_and_java_validated",
        "snapshot_is_transient_and_not_replayable",
        "query_result_is_not_a_clinical_conclusion",
        "bounded_result_projection_only",
        "dynamic_python_analysis_is_bounded_and_sandboxed",
    ]] = Field(min_length=1)
    nonDiagnostic: Literal["not_clinical_diagnostic_or_treatment_advice"]
    generatedAnalysis: GeneratedAnalysisResult | None = None


class IntentRunResult(ClosedModel):
    traceId: Identifier
    runId: Identifier
    taskId: Identifier
    status: ToolStatus
    workflow: ApprovedWorkflow | None = None
    plannerMode: Literal["deterministic", "model"] | None = None
    errorCode: str | None = None
    report: Any = None
    auditEvents: list[AuditEvent] = Field(default_factory=list, max_length=1000)
    metrics: RuntimeMetrics | None = None


class IntentHttpResponse(ClosedModel):
    """Stable external response; audit and runtime identity stay internal."""

    status: ToolStatus
    workflow: ApprovedWorkflow | None = None
    errorCode: str | None = None
    report: Any = None
