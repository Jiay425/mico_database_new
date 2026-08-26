from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, field_validator, model_validator

from .base import ClosedModel, Identifier
from .research import ScientificActionName, StopReasonCode
from .tools import ToolCallIdentifier, TransientPersistence


TraceContractVersion = Literal["p2j4-trace-eval-v1", "p2j4-trace-eval-v2"]
PlannerOrigin = Literal["model", "semantic_guard", "deterministic_fallback", "unknown"]
TraceStatus = Literal["COMPLETED", "FAILED", "REJECTED"]
TraceEventStatus = Literal["COMPLETED", "REJECTED", "FAILED"]
TraceNode = Literal[
    "validate_research_task",
    "policy_gate",
    "plan_action",
    "authorize_action",
    "execute_action",
    "validate_observation",
    "update_state",
    "decide_continue_or_stop",
    "synthesize_report",
    "terminal",
]
TraceToolName = Literal[
    "describe_read_schema",
    "execute_read_query",
    "literature_evidence",
    "python_bounded_analysis",
]
EvidenceRoute = Literal["vector", "graph", "java"]
RedactionVersion = Literal["p2j4-redaction-v1"]
RuntimeCode = Annotated[
    str,
    StringConstraints(
        min_length=3,
        max_length=128,
        pattern=r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+$",
    ),
]
ActionId = Annotated[str, StringConstraints(pattern=r"^action-[0-9a-f]{32}$")]


class TraceEvent(ClosedModel):
    actionId: ActionId
    actionName: ScientificActionName
    node: TraceNode
    status: TraceEventStatus
    toolName: TraceToolName | None = None
    toolCallId: ToolCallIdentifier | None = None
    errorCode: RuntimeCode | None = None
    dataSnapshotId: Annotated[
        str,
        StringConstraints(
            pattern=r"^transient-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        ),
    ] | None = None
    snapshotPersistence: TransientPersistence | None = None
    occurredAt: datetime

    @field_validator("occurredAt")
    @classmethod
    def require_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("trace timestamps must be timezone-aware")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def bind_snapshot(self) -> "TraceEvent":
        if self.dataSnapshotId is not None and self.snapshotPersistence != "transient":
            raise ValueError("trace snapshots must be transient")
        return self


class TraceProjection(ClosedModel):
    """De-identified trace projection; no question, arguments, payload or text."""

    dataContractVersion: TraceContractVersion = "p2j4-trace-eval-v1"
    decisionTraceVersion: Literal["decision-trace-v2"] = "decision-trace-v2"
    traceId: Identifier
    runId: Identifier
    taskId: Identifier
    status: TraceStatus
    events: list[TraceEvent] = Field(default_factory=list, max_length=128)
    actionCount: int = Field(ge=0, le=64)
    toolCallCount: int = Field(ge=0, le=128)
    sourceRoutes: list[EvidenceRoute] = Field(default_factory=list, max_length=3)
    evidenceBindingCount: int = Field(ge=0, le=40)
    analysisResultCount: int = Field(ge=0, le=20)
    fallbackCodes: list[RuntimeCode] = Field(default_factory=list, max_length=32)
    stopReasonCode: StopReasonCode | None = None
    redactionVersion: RedactionVersion = "p2j4-redaction-v1"
    replayable: Literal[False] = False
    startedAt: datetime
    endedAt: datetime
    decisions: list["TraceDecision"] = Field(default_factory=list, max_length=64)
    # The planner decision path is retained above for Decision SFT/audit.  The
    # runtime execution path is a separate contract because a policy repair,
    # semantic guard, fallback, or tool adapter can execute an action other
    # than the model's original choice.  Task oracles must score this field,
    # never infer execution from decisions.
    executedActions: list[ScientificActionName] = Field(default_factory=list, max_length=64)
    executionActionBindingVersion: Literal["execution-action-v1"] | None = None
    evidenceSupportStatuses: list[Literal[
        "supported", "speculative", "conflicted", "partial", "unsupported"
    ]] = Field(default_factory=list, max_length=40)
    structuredResult: bool = False
    nonDiagnosticDeclared: bool = True
    safetyViolationCodes: list[RuntimeCode] = Field(default_factory=list, max_length=32)
    supportStatusEscalated: bool = False
    duplicateActionCount: int = Field(default=0, ge=0, le=64)
    loopCount: int = Field(default=0, ge=0, le=64)

    @field_validator("startedAt", "endedAt")
    @classmethod
    def require_aware_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("trace timestamps must be timezone-aware")
        return value.astimezone(timezone.utc)


EvalCaseId = Annotated[str, StringConstraints(pattern=r"^p2j4-[a-z0-9-]{3,64}$")]
EvalCaseKind = Literal[
    "data_fact",
    "focused_analysis",
    "open_exploration",
    "safety",
    # Legacy v1/v2 labels remain accepted for compatibility with historical
    # fixtures; the current v2 Golden Case set uses the four labels above.
    "schema_exploration",
    "focused_comparison",
    "confounder_stratification",
    "cross_validation",
    "knowledge_retrieval",
    "graph_multi_hop",
    "hybrid_exploration",
    "evidence_insufficient",
    "multi_hop",
    "policy_rejection",
]

TaskSetVersion = Literal[
    "p2j4-task-set-v1",
    "p2j4-task-set-v2",
    "p2j4-task-set-v3",
    "p2j4-decision-state-difference-v1",
    "p2j4-decision-state-difference-v2",
    "p2j4-dpo-v4-development-v1",
    "p2j4-dpo-v4-policy-source-v1",
    "p2j4-dpo-v4-targeted-weak-actions-v1",
    "p2j4-dpo-v4-targeted-weak-actions-v2",
    "p2j4-hard-task-set-v1",
    "p2j4-hard-variant-task-set-v1",
    "p2j4-policy-sensitive-runtime-v1",
]


class TraceDecision(ClosedModel):
    """Metadata-only State -> Action decision captured for evaluation."""

    observationStateCode: RuntimeCode
    allowedActions: list[ScientificActionName] = Field(min_length=1, max_length=10)
    chosenAction: ScientificActionName
    decisionCode: RuntimeCode | None = None
    observationEvidenceBindingCount: int = Field(default=0, ge=0, le=40)
    observationSourceRoutes: list[EvidenceRoute] = Field(default_factory=list, max_length=3)
    # Decision-trace fields are deliberately snake_case because they are the
    # stable export contract for Decision SFT/audit consumers.  The original
    # camelCase fields above remain for backwards-compatible eval readers.
    state_summary: Annotated[str, StringConstraints(min_length=1, max_length=512)] = (
        "observation_state=UNSPECIFIED; evidence_bindings=0; source_routes=none; prior_actions=none"
    )
    decision_reason: Annotated[str, StringConstraints(min_length=1, max_length=512)] = (
        "select the next allow-listed runtime action"
    )
    selected_action: ScientificActionName | None = None
    alternative_actions: list[ScientificActionName] = Field(default_factory=list, max_length=9)
    stop_reason: StopReasonCode | None = None
    # Per-decision provenance. These are optional for backwards-compatible
    # reads of frozen v1/v2 traces; new runtime traces must populate them.
    raw_action: ScientificActionName | None = None
    final_action: ScientificActionName | None = None
    planner_origin: PlannerOrigin = "unknown"
    repair_code: str | None = Field(default=None, max_length=128)
    repair_codes: list[str] = Field(default_factory=list, max_length=16)
    # Exact redacted policy state used for this decision.  These fields are
    # optional only for backwards-compatible reads of historical traces;
    # newly emitted runtime traces populate them from build_decision_policy_state.
    task_kind: str | None = Field(default=None, max_length=64)
    goal_code: str | None = Field(default=None, max_length=96)
    observation_flags: list[str] = Field(default_factory=list, max_length=16)
    history_actions: list[ScientificActionName] = Field(default_factory=list, max_length=8)
    candidate_actions: list[ScientificActionName] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def complete_decision_trace_fields(self) -> "TraceDecision":
        if self.state_summary.startswith("observation_state=UNSPECIFIED"):
            routes = ",".join(self.observationSourceRoutes) or "none"
            object.__setattr__(
                self,
                "state_summary",
                f"observation_state={self.observationStateCode}; "
                f"evidence_bindings={self.observationEvidenceBindingCount}; "
                f"source_routes={routes}; prior_actions=none",
            )
        if self.decision_reason == "select the next allow-listed runtime action":
            reason = {
                "inspect_cohort": "inspect metadata before reading or analyzing the abundance projection",
                "execute_read_query": "execute the next bounded, policy-approved read needed for the task",
                "compare_groups": "compare the approved groups using the validated observation",
                "stratified_analysis": "check whether the observation changes across approved strata",
                "adjust_confounders": "control approved confounders before treating the observation as stable",
                "cross_project_validate": "validate whether the observation is stable across projects",
                "cross_disease_validate": "test whether the observation is disease-specific",
                "retrieve_evidence": "add bounded knowledge evidence after the data observation is available",
                "analyze_projection": "summarize the validated projection before the final decision",
            }.get(self.chosenAction)
            if reason:
                object.__setattr__(self, "decision_reason", reason)
        if self.selected_action is None:
            object.__setattr__(self, "selected_action", self.chosenAction)
        elif self.selected_action != self.chosenAction:
            raise ValueError("selected_action must match chosenAction")
        if not self.alternative_actions:
            object.__setattr__(
                self,
                "alternative_actions",
                [action for action in self.allowedActions if action != self.chosenAction],
            )
        if self.chosenAction != "finish" and self.stop_reason is not None:
            raise ValueError("stop_reason is only valid for a finish decision")
        return self


class EvalTask(ClosedModel):
    dataContractVersion: TraceContractVersion = "p2j4-trace-eval-v1"
    caseId: EvalCaseId
    kind: EvalCaseKind
    question: Annotated[str, StringConstraints(min_length=1, max_length=512)]
    expectedStatus: TraceStatus
    requiredSources: list[EvidenceRoute] = Field(default_factory=list, max_length=3)
    minEvidenceBindings: int = Field(default=0, ge=0, le=40)
    maxActionCount: int = Field(default=8, ge=1, le=8)
    requiresNonDiagnostic: Literal[True] = True
    schemaVersion: TaskSetVersion | None = None
    allowedActions: list[ScientificActionName] = Field(default_factory=list, max_length=10)
    requiredActions: list[ScientificActionName] = Field(default_factory=list, max_length=10)
    forbiddenActions: list[Annotated[str, StringConstraints(min_length=1, max_length=64)]] = Field(
        default_factory=list, max_length=16
    )
    expectedStopReason: StopReasonCode | None = None
    allowedActionPaths: list[list[ScientificActionName]] = Field(default_factory=list, max_length=16)


EvalCriterion = Literal[
    "task_success",
    "decision_accuracy",
    "evidence_grounding",
    "tool_efficiency",
    "safety_pass",
    "stop_correctness",
    "status_matches",
    "structured_result_present",
    "required_sources_present",
    "evidence_binding_minimum",
    "action_budget_respected",
    "tool_calls_unique",
    "actions_not_repeated",
    "stop_reason_present",
    "non_replayable_declared",
    "support_status_not_escalated",
    "non_diagnostic_declared",
    "redaction_contract",
]
EvalFailureCode = Literal[
    "DECISION_ACCURACY_FAILURE",
    "EVIDENCE_GROUNDING_FAILURE",
    "TOOL_EFFICIENCY_FAILURE",
    "SAFETY_POLICY_FAILURE",
    "STOP_CORRECTNESS_FAILURE",
    "TRACE_STRUCTURED_RESULT_MISSING",
    "TRACE_FORBIDDEN_ACTION",
    "TRACE_ACTION_REPEATED",
    "TRACE_SUPPORT_STATUS_ESCALATED",
    "TRACE_SNAPSHOT_NOT_TRANSIENT",
    "TRACE_MODEL_OUTPUT_REJECTED",
    "TRACE_RESPONSE_CORRELATION_FAILURE",
    "TRACE_STATUS_MISMATCH",
    "TRACE_SOURCE_MISSING",
    "TRACE_EVIDENCE_INSUFFICIENT",
    "TRACE_ACTION_BUDGET_EXCEEDED",
    "TRACE_TOOL_CALL_REPEATED",
    "TRACE_STOP_REASON_MISSING",
    "TRACE_REPLAYABILITY_UNSAFE",
    "TRACE_REDACTION_INVALID",
]


class EvalScore(ClosedModel):
    dataContractVersion: TraceContractVersion = "p2j4-trace-eval-v1"
    caseId: EvalCaseId
    traceId: Identifier
    status: Literal["PASS", "FAIL"]
    score: float = Field(ge=0.0, le=1.0)
    criteria: dict[EvalCriterion, bool]
    failureCodes: list[EvalFailureCode] = Field(default_factory=list, max_length=32)
    evaluatedAt: datetime

    @field_validator("evaluatedAt")
    @classmethod
    def require_evaluated_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("evaluation timestamp must be timezone-aware")
        return value.astimezone(timezone.utc)


BadCaseCategory = Literal[
    "PLANNING_ERROR",
    "TOOL_SELECTION_ERROR",
    "MISSING_EVIDENCE",
    "UNSAFE_CONCLUSION",
    "REPEATED_ACTION",
    "WRONG_STOP",
    "SCHEMA_CONTRACT_FAILURE",
    "MODEL_OUTPUT_REJECTED",
    "RESPONSE_CORRELATION_FAILURE",
    "wrong_stop",
    "repeated_call",
    "missing_evidence",
    "policy_failure",
    "contract_failure",
    "sensitive_projection",
]
BadCaseStatus = Literal["OPEN", "IN_REVIEW", "ACCEPTED", "REJECTED"]


class BadCaseRecord(ClosedModel):
    dataContractVersion: TraceContractVersion = "p2j4-trace-eval-v1"
    badCaseId: Annotated[str, StringConstraints(pattern=r"^bad-[0-9a-f]{32}$")]
    caseId: EvalCaseId
    traceId: Identifier
    category: BadCaseCategory
    failureCode: EvalFailureCode | BadCaseCategory
    status: BadCaseStatus = "OPEN"
    createdAt: datetime
    reviewedAt: datetime | None = None
    reviewedBy: Annotated[str, StringConstraints(pattern=r"^principal-[0-9a-f]{32}$")] | None = None
    reviewerId: Annotated[str, StringConstraints(pattern=r"^principal-[0-9a-f]{32}$")] | None = None
    reviewDecision: Literal["ACCEPT_BAD_CASE", "REJECT_BAD_CASE"] | None = None

    @model_validator(mode="after")
    def require_review_fields(self) -> "BadCaseRecord":
        if self.status in {"ACCEPTED", "REJECTED"} and (
            self.reviewedAt is None or (self.reviewedBy is None and self.reviewerId is None)
        ):
            raise ValueError("closed bad cases require a reviewer and review time")
        return self


ReviewDecision = Literal["START_REVIEW", "ACCEPT_BAD_CASE", "REJECT_BAD_CASE"]


class HumanReviewCommand(ClosedModel):
    dataContractVersion: TraceContractVersion = "p2j4-trace-eval-v1"
    badCaseId: Annotated[str, StringConstraints(pattern=r"^bad-[0-9a-f]{32}$")]
    decision: ReviewDecision
    reviewerId: Annotated[str, StringConstraints(pattern=r"^principal-[0-9a-f]{32}$")]
    reviewedAt: datetime

    @field_validator("reviewedAt")
    @classmethod
    def require_review_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("review timestamp must be timezone-aware")
        return value.astimezone(timezone.utc)


class StabilityBaseline(ClosedModel):
    dataContractVersion: TraceContractVersion = "p2j4-trace-eval-v1"
    taskSetVersion: Annotated[
        str,
        StringConstraints(
            pattern=(
                r"^(?:p2j4-[a-z0-9-]*task-set-v[0-9]+|"
                r"p2j4-decision-state-difference-v[0-9]+|"
                r"p2j4-dpo-v[0-9]+-development-v[0-9]+|"
                r"p2j4-dpo-v[0-9]+-policy-source-v[0-9]+|"
                r"p2j4-dpo-v[0-9]+-targeted-[a-z0-9-]+-v[0-9]+|"
                r"p2j4-policy-sensitive-runtime-v[0-9]+)$"
            )
        ),
    ]
    runCount: int = Field(ge=0)
    passCount: int = Field(ge=0)
    passRate: float = Field(ge=0.0, le=1.0)
    meanScore: float = Field(ge=0.0, le=1.0)
    repeatedCallRate: float = Field(ge=0.0, le=1.0)
    failureCodeCounts: dict[EvalFailureCode, int] = Field(default_factory=dict)
    casePassRate: float = Field(default=0.0, ge=0.0, le=1.0)
    decisionAccuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    evidenceGroundingRate: float = Field(default=0.0, ge=0.0, le=1.0)
    safetyPassRate: float = Field(default=0.0, ge=0.0, le=1.0)
    stopCorrectness: float = Field(default=0.0, ge=0.0, le=1.0)
    meanActionCount: float = Field(default=0.0, ge=0.0, le=64.0)
    duplicateActionRate: float = Field(default=0.0, ge=0.0, le=1.0)
    fallbackRate: float = Field(default=0.0, ge=0.0, le=1.0)
    generatedAt: datetime

    @model_validator(mode="after")
    def validate_counts(self) -> "StabilityBaseline":
        if self.passCount > self.runCount:
            raise ValueError("pass count cannot exceed run count")
        return self
