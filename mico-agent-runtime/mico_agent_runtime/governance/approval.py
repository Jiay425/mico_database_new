from __future__ import annotations

from typing import Literal

from mico_agent_runtime.contracts.approval import ApprovalOperation
from mico_agent_runtime.contracts.base import ClosedModel
ApprovalDecisionCode = Literal[
    "APPROVAL_NOT_REQUIRED",
    "APPROVAL_REQUIRED",
    "APPROVAL_WORKFLOW_NOT_IMPLEMENTED",
]


class ApprovalGateDecision(ClosedModel):
    operation: ApprovalOperation
    allowed: bool
    code: ApprovalDecisionCode
    requiresHumanApproval: bool


class ApprovalGate:
    """Fail-closed policy for operations that are not part of P2/P3 reads."""

    _NO_APPROVAL = frozenset({"read_only_research", "statistical_analysis"})

    def evaluate(self, operation: ApprovalOperation) -> ApprovalGateDecision:
        if operation in self._NO_APPROVAL:
            return ApprovalGateDecision(
                operation=operation,
                allowed=True,
                code="APPROVAL_NOT_REQUIRED",
                requiresHumanApproval=False,
            )
        return ApprovalGateDecision(
            operation=operation,
            allowed=False,
            code="APPROVAL_REQUIRED",
            requiresHumanApproval=True,
        )

    def require_decision(self, operation: ApprovalOperation) -> ApprovalGateDecision:
        """Never treats a caller-provided flag as an approval decision."""

        decision = self.evaluate(operation)
        if not decision.allowed:
            return decision.model_copy(update={"code": "APPROVAL_WORKFLOW_NOT_IMPLEMENTED"})
        return decision
