from __future__ import annotations

from typing import Any, TypedDict

from mico_agent_runtime.contracts.audit import AuditEvent
from mico_agent_runtime.contracts.analysis import AnalysisEvidence, AnalysisPlan, AnalysisResult
from mico_agent_runtime.contracts.decision_state import ScientificDecisionState
from mico_agent_runtime.contracts.research import (
    Observation,
    ResearchTask,
    ScientificAction,
)
from mico_agent_runtime.contracts.schema_catalog import SchemaSemanticCatalog
from mico_agent_runtime.contracts.trace_eval import TraceDecision


class ScientificState(TypedDict, total=False):
    request: ResearchTask | dict[str, Any]
    decisionState: ScientificDecisionState
    schemaCatalog: SchemaSemanticCatalog | None
    currentAction: ScientificAction | None
    observations: list[Observation]
    rawObservationPayloads: dict[str, object]
    analysisPlans: list[AnalysisPlan]
    analysisResults: list[AnalysisResult]
    analysisEvidence: list[AnalysisEvidence]
    # Runtime-only audit copies of Registry-approved generated programs. They
    # never enter ScientificDecisionState or policy input.
    generatedAnalysisPrograms: list[dict[str, object]]
    actionHistory: list[str]
    actionSignatures: list[str]
    # Runtime-only diagnostics for Hard Action Availability.  This map is
    # never copied into the policy-facing ScientificDecisionState.
    actionAvailabilityReasons: dict[str, str]
    # Runtime-only objective lifecycle.  Entries may be active, completed, or
    # blocked by a proven data-capability boundary; this is never copied into
    # the six-block Policy input.
    objectiveResolution: dict[str, Any]
    knowledgeAvailable: bool
    taskUnderstandingMode: str
    taskUnderstandingModel: str
    plannerFeedback: list[str]
    # True when the runtime is using the dynamic Policy/Materializer path.
    # Observation projection uses this to keep aggregate rows out of the
    # sample-level analysis input without exposing a policy hint.
    dynamicMaterialization: bool
    # Bounded provider/contract diagnostic retained for canary audit only;
    # never copied into the six-block Decision State sent to a policy model.
    plannerFailureDetail: str
    # Runtime-only Catalog-derived data requirements used to bind the next
    # read projection.  This is deliberately outside ScientificDecisionState.
    dataRequirements: dict[str, Any]
    # Runtime-only capability ledger populated by real zero-coverage
    # observations. It is consumed by QueryPlan pruning and never exposed to
    # the six-block Policy input.
    unavailableFields: list[str]
    unavailableDataCapabilities: dict[str, Any]
    readReplanCount: int
    plannerMode: str
    # Runtime-only one-shot allowance for a terminal Policy finish decision
    # after the non-terminal action budget has been consumed. This does not
    # enter ScientificDecisionState and never selects ``finish`` itself.
    terminalFinishTurnGranted: bool
    fallbackCodes: list[str]
    decisionRecords: list[TraceDecision]
    pendingObservation: Observation
    pendingPayload: object
    pendingResponse: object
    # The typed materializer's snake_case plan is retained only until the
    # validated Python observation is projected into decisionState.
    pendingTypedAnalysisPlan: object
    stopReasonCode: str
    route: str
    report: Any
    status: str
    errorCode: str
    auditEvents: list[AuditEvent]
    progressCallback: Any
