from __future__ import annotations

from typing import Any

from mico_agent_runtime.contracts.evidence import EvidenceRunResult, EvidenceTaskRequest
from mico_agent_runtime.graph.evidence_workflow import build_evidence_graph
from mico_agent_runtime.ports.evidence import EvidenceSearchPort


class EvidenceRuntime:
    """P4 evidence boundary; external search is dependency-injected and fail-closed by default."""

    def __init__(self, search_port: EvidenceSearchPort) -> None:
        self._search_port = search_port
        self._graph = build_evidence_graph(search_port)

    def run(self, request: EvidenceTaskRequest | dict[str, Any]) -> EvidenceRunResult:
        state = self._graph.invoke({"request": request, "auditEvents": []})
        parsed = state.get("request")
        if isinstance(parsed, EvidenceTaskRequest):
            trace_id, run_id, task_id = parsed.traceId, parsed.runId, parsed.taskId
        else:
            trace_id = run_id = task_id = "invalid"
        return EvidenceRunResult(
            traceId=trace_id,
            runId=run_id,
            taskId=task_id,
            status=state.get("status", "FAILED"),
            errorCode=state.get("errorCode"),
            report=state.get("report"),
        )

    async def close(self) -> None:
        close = getattr(self._search_port, "close", None)
        if close is not None:
            close()
