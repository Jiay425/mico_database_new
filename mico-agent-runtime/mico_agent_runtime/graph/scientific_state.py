from __future__ import annotations

from typing import Any, TypedDict

from mico_agent_runtime.contracts.audit import AuditEvent
from mico_agent_runtime.contracts.analysis import AnalysisEvidence, AnalysisPlan, AnalysisResult
from mico_agent_runtime.contracts.research import (
    Observation,
    ResearchTask,
    ScientificAction,
)
from mico_agent_runtime.contracts.schema_catalog import SchemaSemanticCatalog
from mico_agent_runtime.contracts.trace_eval import TraceDecision


class ScientificState(TypedDict, total=False):
    request: ResearchTask | dict[str, Any]
    schemaCatalog: SchemaSemanticCatalog | None
    currentAction: ScientificAction | None
    observations: list[Observation]
    rawObservationPayloads: dict[str, object]
    analysisPlans: list[AnalysisPlan]
    analysisResults: list[AnalysisResult]
    analysisEvidence: list[AnalysisEvidence]
    actionHistory: list[str]
    actionSignatures: list[str]
    plannerFeedback: list[str]
    readReplanCount: int
    plannerMode: str
    fallbackCodes: list[str]
    decisionRecords: list[TraceDecision]
    pendingObservation: Observation
    pendingPayload: object
    pendingResponse: object
    stopReasonCode: str
    route: str
    report: Any
    status: str
    errorCode: str
    auditEvents: list[AuditEvent]
    progressCallback: Any
