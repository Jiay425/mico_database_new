from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mico_agent_runtime.contracts.intent import ApprovedWorkflow


WorkflowToolName = Literal["execute_read_query", "literature_evidence"]


@dataclass(frozen=True)
class WorkflowToolPlan:
    """Runtime-owned plan; no model or browser input can alter this tuple."""

    workflow: ApprovedWorkflow
    toolNames: tuple[WorkflowToolName, ...]
    maxCalls: int
    implemented: bool


_PLANS: dict[ApprovedWorkflow, WorkflowToolPlan] = {
    "dynamic_read_query": WorkflowToolPlan(
        workflow="dynamic_read_query",
        toolNames=("execute_read_query",),
        maxCalls=1,
        implemented=True,
    ),
    "knowledge_retrieval": WorkflowToolPlan(
        workflow="knowledge_retrieval",
        toolNames=("literature_evidence",),
        maxCalls=0,
        implemented=True,
    ),
}


def get_workflow_tool_plan(workflow: ApprovedWorkflow) -> WorkflowToolPlan:
    try:
        return _PLANS[workflow]
    except KeyError as exc:
        raise ValueError("workflow is not registered") from exc
