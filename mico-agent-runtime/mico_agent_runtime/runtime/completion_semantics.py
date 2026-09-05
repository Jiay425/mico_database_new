"""Explicit separation of workflow, execution, result, and conclusion status."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ScientificCompletionAssessment:
    """Derived status; never stored as a Decision State strategy hint."""

    workflow_completed: bool
    analysis_execution_completed: bool
    scientific_result_valid: bool
    scientific_conclusion_eligible: bool
    reason_codes: tuple[str, ...] = ()

    def model_dump(self) -> dict[str, Any]:
        return {
            "workflow_completed": self.workflow_completed,
            "analysis_execution_completed": self.analysis_execution_completed,
            "scientific_result_valid": self.scientific_result_valid,
            "scientific_conclusion_eligible": self.scientific_conclusion_eligible,
            "reason_codes": list(self.reason_codes),
        }


def assess_scientific_completion(
    *,
    workflow_completed: bool,
    analysis_result: Mapping[str, Any] | None,
) -> ScientificCompletionAssessment:
    """Assess an analysis result without rewriting historical trace fields.

    A completed workflow only means that the graph reached its terminal node.
    Execution completion is the operator status.  Result validity requires a
    non-empty, shape-appropriate summary.  Conclusion eligibility is stricter:
    a generated compatibility result or an explicit insufficient-data result
    is never promoted to a scientific conclusion.
    """

    reasons: list[str] = []
    if not workflow_completed:
        reasons.append("WORKFLOW_NOT_COMPLETED")
    if not isinstance(analysis_result, Mapping):
        reasons.append("ANALYSIS_RESULT_MISSING")
        return ScientificCompletionAssessment(
            workflow_completed=workflow_completed,
            analysis_execution_completed=False,
            scientific_result_valid=False,
            scientific_conclusion_eligible=False,
            reason_codes=tuple(reasons),
        )

    execution_completed = analysis_result.get("status") == "COMPLETED"
    if not execution_completed:
        reasons.append("ANALYSIS_EXECUTION_NOT_COMPLETED")
    metrics = analysis_result.get("metrics")
    if not isinstance(metrics, Mapping) or not metrics:
        reasons.append("ANALYSIS_METRICS_EMPTY")
    analysis_type = str(analysis_result.get("analysisType") or "")
    strata = analysis_result.get("stratum_results")
    features = analysis_result.get("feature_results")
    if analysis_type == "stratified_comparison" and not (
        isinstance(strata, list) and strata
    ) and not (
        isinstance(features, list)
        and any(
            isinstance(item, Mapping)
            and item.get("status", "supported") == "supported"
            and item.get("stratum_results")
            for item in features
        )
    ):
        reasons.append("STRATIFIED_RESULT_EMPTY")
    result_valid = execution_completed and not any(
        code in {"ANALYSIS_RESULT_MISSING", "ANALYSIS_METRICS_EMPTY", "STRATIFIED_RESULT_EMPTY"}
        for code in reasons
    )
    limitations = analysis_result.get("limitations")
    generated = isinstance(limitations, list) and any(
        "generated" in str(value).lower() for value in limitations
    )
    if generated:
        reasons.append("GENERATED_COMPATIBILITY_RESULT")
    if analysis_result.get("execution_mode") == "generated":
        generated = True
    if not result_valid:
        reasons.append("SCIENTIFIC_RESULT_INVALID")
    eligible = bool(
        workflow_completed
        and execution_completed
        and result_valid
        and not generated
    )
    if not eligible and "SCIENTIFIC_CONCLUSION_INELIGIBLE" not in reasons:
        reasons.append("SCIENTIFIC_CONCLUSION_INELIGIBLE")
    return ScientificCompletionAssessment(
        workflow_completed=workflow_completed,
        analysis_execution_completed=execution_completed,
        scientific_result_valid=result_valid,
        scientific_conclusion_eligible=eligible,
        reason_codes=tuple(dict.fromkeys(reasons)),
    )


__all__ = ["ScientificCompletionAssessment", "assess_scientific_completion"]
