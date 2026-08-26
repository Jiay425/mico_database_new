from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from time import perf_counter
from typing import Any

from mico_agent_runtime.contracts.audit import AuditEvent
from mico_agent_runtime.contracts.research import ResearchExplorationReport, ResearchTask
from mico_agent_runtime.contracts.schema_catalog import SchemaSemanticCatalog
from mico_agent_runtime.graph.scientific_workflow import build_scientific_graph
from mico_agent_runtime.ports.java_agent import JavaAgentToolPort
from mico_agent_runtime.ports.java_agent import JavaPortContractError, JavaPortTransportError
from mico_agent_runtime.ports.knowledge import KnowledgeSearchPort
from mico_agent_runtime.ports.schema_catalog import SchemaCatalogPort
from mico_agent_runtime.ports.scientific_planner import ScientificPlannerPort
from mico_agent_runtime.knowledge.synthesis import GraphRagSynthesisPort
from mico_agent_runtime.contracts.trace_eval import TraceDecision


@dataclass(frozen=True)
class ScientificRuntimeResult:
    """Typed execution result that remains compatible with old dict callers."""

    runId: str
    taskId: str
    traceId: str
    status: str
    errorCode: str | None
    report: ResearchExplorationReport | None
    auditEvents: list[AuditEvent]
    plannerMode: str | None
    actionCount: int
    durationMs: int
    decisionRecords: list[TraceDecision] = field(default_factory=list)
    stopReasonCode: str | None = None
    fallbackCodes: list[str] = field(default_factory=list)
    safetyViolationCodes: list[str] = field(default_factory=list)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


class ScientificRuntime:
    """In-memory Scientific Agent Loop; persistence is injected separately."""

    def __init__(
        self,
        java_port: JavaAgentToolPort,
        planner: ScientificPlannerPort,
        *,
        knowledge_port: KnowledgeSearchPort | None = None,
        schema_catalog: SchemaSemanticCatalog | None = None,
        schema_catalog_port: SchemaCatalogPort | None = None,
        checkpointer: Any | None = None,
        synthesis_port: GraphRagSynthesisPort | None = None,
    ) -> None:
        self._graph = build_scientific_graph(
            java_port,
            planner,
            knowledge_port=knowledge_port,
            schema_catalog=schema_catalog,
            checkpointer=checkpointer,
            synthesis_port=synthesis_port,
        )
        self._schema_catalog = schema_catalog
        self._schema_catalog_port = schema_catalog_port
        self._java_port = java_port
        self._planner = planner
        self._knowledge_port = knowledge_port
        self._checkpointer = checkpointer
        self._synthesis_port = synthesis_port

    def _load_schema_catalog_if_needed(self, request: ResearchTask | dict[str, Any]) -> str | None:
        if self._schema_catalog is not None or self._schema_catalog_port is None:
            return None
        if not isinstance(request, ResearchTask):
            return None
        try:
            self._schema_catalog = self._schema_catalog_port.load(
                run_id=request.runId,
                task_id=request.taskId,
            )
        except JavaPortContractError:
            return "JAVA_SCHEMA_CATALOG_CALL_INVALID"
        except JavaPortTransportError as exc:
            return exc.code
        except Exception:
            return "JAVA_SCHEMA_CATALOG_UNAVAILABLE"
        self._graph = build_scientific_graph(
            self._java_port,
            self._planner,
            knowledge_port=self._knowledge_port,
            schema_catalog=self._schema_catalog,
            checkpointer=self._checkpointer,
            synthesis_port=self._synthesis_port,
        )
        return None

    def with_java_port(
        self,
        java_port: JavaAgentToolPort,
        *,
        planner: ScientificPlannerPort | None = None,
    ) -> "ScientificRuntime":
        """Create one isolated run view over an Eval-only bounded Java supply."""

        return ScientificRuntime(
            java_port,
            planner or self._planner,
            knowledge_port=self._knowledge_port,
            schema_catalog=self._schema_catalog,
            schema_catalog_port=self._schema_catalog_port,
            checkpointer=self._checkpointer,
            synthesis_port=self._synthesis_port,
        )

    def with_knowledge_port(
        self,
        knowledge_port: KnowledgeSearchPort | None,
    ) -> "ScientificRuntime":
        """Create one isolated Eval view over a bounded knowledge scenario."""

        return ScientificRuntime(
            self._java_port,
            self._planner,
            knowledge_port=knowledge_port,
            schema_catalog=self._schema_catalog,
            schema_catalog_port=self._schema_catalog_port,
            checkpointer=self._checkpointer,
            synthesis_port=self._synthesis_port,
        )

    def run(self, request: ResearchTask | dict[str, Any]) -> ScientificRuntimeResult:
        started = perf_counter()
        validated_request = request if isinstance(request, ResearchTask) else None
        run_id = validated_request.runId if validated_request else ""
        task_id = validated_request.taskId if validated_request else ""
        trace_id = validated_request.traceId if validated_request else ""
        schema_error = self._load_schema_catalog_if_needed(request)
        if schema_error is not None:
            return ScientificRuntimeResult(
                runId=run_id, taskId=task_id, traceId=trace_id,
                status="FAILED", errorCode=schema_error, report=None,
                auditEvents=[], plannerMode=None, actionCount=0,
                durationMs=max(0, int((perf_counter() - started) * 1000)),
            )
        state = self._graph.invoke({
            "request": request,
            "auditEvents": [],
            "observations": [],
            "rawObservationPayloads": {},
            "analysisPlans": [],
            "analysisResults": [],
            "analysisEvidence": [],
            "actionHistory": [],
            "actionSignatures": [],
            "plannerFeedback": [],
            "readReplanCount": 0,
            "fallbackCodes": [],
            "schemaCatalog": self._schema_catalog,
        })
        report = state.get("report")
        final_request = state.get("request")
        if isinstance(final_request, ResearchTask):
            run_id, task_id, trace_id = final_request.runId, final_request.taskId, final_request.traceId
        return ScientificRuntimeResult(
            runId=run_id,
            taskId=task_id,
            traceId=trace_id,
            status=state.get("status", "FAILED"),
            errorCode=state.get("errorCode"),
            report=report,
            auditEvents=state.get("auditEvents", []),
            plannerMode=state.get("plannerMode"),
            actionCount=len(state.get("actionHistory", [])),
            durationMs=max(0, int((perf_counter() - started) * 1000)),
            decisionRecords=state.get("decisionRecords", []),
            stopReasonCode=state.get("stopReasonCode"),
            fallbackCodes=state.get("fallbackCodes", []),
            safetyViolationCodes=state.get("safetyViolationCodes", []),
        )
