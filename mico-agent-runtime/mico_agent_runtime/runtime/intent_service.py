from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from typing import Any

from mico_agent_runtime.contracts.intent import IntentRunResult, IntentTaskRequest
from mico_agent_runtime.contracts.review import GraphReviewResumeCommand
from langgraph.types import Command
from mico_agent_runtime.graph.intent_workflow import build_intent_graph
from mico_agent_runtime.ports.java_agent import JavaAgentToolPort
from mico_agent_runtime.ports.knowledge import KnowledgeSearchPort
from mico_agent_runtime.ports.research_planner import IntentPlannerPort
from mico_agent_runtime.knowledge.synthesis import GraphRagSynthesisPort
from mico_agent_runtime.contracts.audit import AuditEvent
from mico_agent_runtime.contracts.metrics import RuntimeMetrics


class IntentRuntime:
    """The single dynamic LangGraph entry point with optional checkpointing."""

    def __init__(self, java_port: JavaAgentToolPort, planner: IntentPlannerPort,
                 checkpoint_saver: Any | None = None,
                 knowledge_port: KnowledgeSearchPort | None = None,
                 synthesis_port: GraphRagSynthesisPort | None = None,
                 interrupt_on_review: bool = False) -> None:
        self._graph = build_intent_graph(
            java_port,
            planner,
            checkpointer=checkpoint_saver,
            knowledge_port=knowledge_port,
            synthesis_port=synthesis_port,
            interrupt_on_review=interrupt_on_review,
        )
        self._checkpoint_enabled = checkpoint_saver is not None

    @staticmethod
    def _checkpoint_config(request: IntentTaskRequest) -> dict[str, Any]:
        return {
            "configurable": {
                "thread_id": request.runId,
                "runtime_task_id": request.taskId,
                "runtime_trace_id": request.traceId,
            }
        }

    def run(self, request: IntentTaskRequest | dict[str, Any],
            progress_callback: Callable[[AuditEvent], None] | None = None) -> IntentRunResult:
        started = perf_counter()
        parsed = request if isinstance(request, IntentTaskRequest) else None
        invoke_config = self._checkpoint_config(parsed) if self._checkpoint_enabled and parsed else None
        state = self._graph.invoke({
            "request": request,
            "auditEvents": [],
            "progressCallback": progress_callback,
        }, config=invoke_config)
        return self._result_from_state(state, started)

    async def run_async(
        self,
        request: IntentTaskRequest | dict[str, Any],
        progress_callback: Callable[[AuditEvent], None] | None = None,
    ) -> IntentRunResult:
        """Run the graph and async checkpoint saver on the caller's loop."""
        started = perf_counter()
        parsed = request if isinstance(request, IntentTaskRequest) else None
        invoke_config = self._checkpoint_config(parsed) if self._checkpoint_enabled and parsed else None
        state = await self._graph.ainvoke({
            "request": request,
            "auditEvents": [],
            "progressCallback": progress_callback,
        }, config=invoke_config)
        return self._result_from_state(state, started)

    @staticmethod
    def _result_from_state(state: dict[str, Any], started: float) -> IntentRunResult:
        parsed_request = state.get("request")
        if isinstance(parsed_request, IntentTaskRequest):
            trace_id, run_id, task_id = (
                parsed_request.traceId, parsed_request.runId, parsed_request.taskId
            )
        else:
            trace_id = run_id = task_id = "invalid"
        events = state.get("auditEvents", [])
        metric_events = list(events)
        sub_result = state.get("subResult")
        sub_events = getattr(sub_result, "auditEvents", None)
        if isinstance(sub_events, list):
            # The intent router keeps the subgraph audit inside process-local
            # state.  Include it only in bounded counters; never expose or
            # merge its raw event list into the outer response.
            metric_events.extend(sub_events)
        tool_call_ids = {
            event.toolCallId for event in metric_events
            if isinstance(event, AuditEvent) and event.toolCallId is not None
        }
        snapshot_ids = {
            event.dataSnapshotId for event in metric_events
            if isinstance(event, AuditEvent) and event.dataSnapshotId is not None
        }
        planner_mode = state.get("plannerMode")
        metrics = RuntimeMetrics(
            durationMs=max(0, int((perf_counter() - started) * 1000)),
            plannerMode=planner_mode if planner_mode in {"deterministic", "model"} else None,
            modelCallCount=1 if planner_mode == "model" else 0,
            javaToolCallCount=len(tool_call_ids),
            snapshotCount=len(snapshot_ids),
        )
        return IntentRunResult(
            traceId=trace_id,
            runId=run_id,
            taskId=task_id,
            status=state.get("status", "FAILED"),
            workflow=state.get("workflow"),
            plannerMode=state.get("plannerMode"),
            errorCode=state.get("errorCode"),
            report=state.get("report"),
            auditEvents=events,
            metrics=metrics,
        )

    def resume(
        self,
        request: IntentTaskRequest,
        review_decision: GraphReviewResumeCommand | dict[str, Any] | None = None,
    ) -> IntentRunResult:
        """Continue from the latest encrypted LangGraph checkpoint.

        This method is exposed only when a checkpoint saver was injected by
        Runtime persistence.  It never accepts a client-supplied checkpoint or
        raw state payload.
        """

        if not self._checkpoint_enabled:
            raise RuntimeError("RUNTIME_RECOVERY_NOT_ENABLED")
        input_value: Any = None
        if review_decision is not None:
            command = review_decision if isinstance(review_decision, GraphReviewResumeCommand) else GraphReviewResumeCommand.model_validate(review_decision)
            input_value = Command(resume=command.model_dump(mode="json"))
        state = self._graph.invoke(input_value, config=self._checkpoint_config(request))
        return self._result_from_state(state, perf_counter())

    async def resume_async(
        self,
        request: IntentTaskRequest,
        review_decision: GraphReviewResumeCommand | dict[str, Any] | None = None,
    ) -> IntentRunResult:
        if not self._checkpoint_enabled:
            raise RuntimeError("RUNTIME_RECOVERY_NOT_ENABLED")
        input_value: Any = None
        if review_decision is not None:
            command = review_decision if isinstance(review_decision, GraphReviewResumeCommand) else GraphReviewResumeCommand.model_validate(review_decision)
            input_value = Command(resume=command.model_dump(mode="json"))
        state = await self._graph.ainvoke(input_value, config=self._checkpoint_config(request))
        return self._result_from_state(state, perf_counter())
