"""Adapt a redacted Trace into closed result-oracle assertion codes.

The seven ``TRACE_ORACLE_READY`` cases do not have a replay scenario that can
be persisted.  Their durable input is therefore the metadata-only
``TraceProjection``.  This adapter turns only bounded runtime facts into
closed assertion and limitation codes; it never copies question text, SQL,
row values, identifiers, or evidence content into the observation.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from mico_agent_runtime.contracts.trace_eval import EvalTask, TraceProjection

from evals.p2j4_controlled_scenarios import ResultOracleObservation


class TraceOracleAssertionError(RuntimeError):
    """A safe, closed adapter failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class TraceOracleAssertionAdapter:
    """Translate redacted Trace metadata into value-free oracle codes."""

    _MISSINGNESS_TERMS = ("缺失", "覆盖不足", "覆盖", "不足", "missing", "coverage")
    _COUNTRY_TERMS = ("country", "国家", "地区", "地域")
    _PROJECT_TERMS = ("project", "队列", "项目")
    _MATCHED_COHORT_TERMS = (
        "可比",
        "同时具有",
        "病例和对照",
        "匹配",
        "overlap",
        "matched",
        "case-control",
    )
    _MULTIFEATURE_TERMS = (
        "多个微生物",
        "共同变化",
        "组合",
        "菌群",
        "multiple",
        "combination",
        "co-change",
    )
    _CONFLICT_TERMS = (
        "冲突",
        "推测",
        "speculative",
        "conflicted",
        "不确定",
    )
    _FUNCTION_TERMS = ("功能", "functional", "pathway", "代谢")

    @staticmethod
    def _actions(trace: TraceProjection) -> list[str]:
        decisions = [decision.chosenAction for decision in trace.decisions]
        if decisions:
            return list(decisions)
        return [
            event.actionName
            for event in trace.events
            if event.node == "execute_action"
        ]

    @staticmethod
    def _contains(text: str, terms: Iterable[str]) -> bool:
        lowered = text.lower()
        return any(term.lower() in lowered for term in terms)

    @staticmethod
    def _has_action_before(actions: Sequence[str], first: str, second: str) -> bool:
        try:
            return actions.index(first) < actions.index(second)
        except ValueError:
            return False

    @staticmethod
    def _add(codes: list[str], *values: str) -> None:
        for value in values:
            if value not in codes:
                codes.append(value)

    def observe(self, task: EvalTask, trace: TraceProjection) -> ResultOracleObservation:
        """Return only closed codes derived from a completed redacted trace."""

        if trace.taskId != task.caseId and not trace.taskId.startswith("task-"):
            raise TraceOracleAssertionError("TRACE_TASK_BINDING_INVALID")

        actions = self._actions(trace)
        if not actions:
            raise TraceOracleAssertionError("TRACE_ACTION_PATH_UNAVAILABLE")

        question = task.question
        routes = list(trace.sourceRoutes)
        assertions: list[str] = []
        limitations: list[str] = []
        has_java_snapshot = any(
            event.status == "COMPLETED"
            and event.dataSnapshotId is not None
            and event.snapshotPersistence == "transient"
            for event in trace.events
        )

        if "java" in routes and has_java_snapshot:
            self._add(assertions, "JAVA_TRANSIENT_EVIDENCE")

        has_retrieval = "retrieve_evidence" in actions
        has_analysis = "analyze_projection" in actions
        has_comparison = "compare_groups" in actions
        has_cross_project = "cross_project_validate" in actions
        has_cross_disease = "cross_disease_validate" in actions
        has_stratification = "stratified_analysis" in actions

        if has_retrieval and "vector" in routes and trace.evidenceBindingCount > 0:
            self._add(assertions, "VECTOR_EVIDENCE_BOUND")
        if (
            has_retrieval
            and "vector" in routes
            and trace.evidenceBindingCount > 0
            and has_analysis
            and self._has_action_before(actions, "analyze_projection", "retrieve_evidence")
        ):
            self._add(assertions, "VECTOR_EVIDENCE_AFTER_ANALYSIS")
        if (
            has_retrieval
            and "graph" in routes
            and trace.evidenceBindingCount > 0
        ):
            self._add(assertions, "GRAPH_PATH_BOUND")
        if trace.structuredResult and trace.evidenceBindingCount > 0 and has_retrieval:
            self._add(assertions, "EVIDENCE_BOUND")

        if has_stratification and self._contains(question, self._COUNTRY_TERMS):
            self._add(assertions, "COUNTRY_DIMENSION_ANALYZED")
        if has_stratification and (
            self._contains(question, self._PROJECT_TERMS) or has_cross_project
        ):
            self._add(assertions, "PROJECT_DIMENSION_ANALYZED")
        if has_stratification and self._contains(question, self._MISSINGNESS_TERMS):
            self._add(assertions, "MISSINGNESS_RETAINED")
            if (
                self._contains(question, self._COUNTRY_TERMS)
                and self._contains(question, ("coverage", "覆盖", "不足"))
            ):
                self._add(limitations, "COUNTRY_COVERAGE_GAP_REPORTED")
            else:
                self._add(limitations, "COVERAGE_GAP_REPORTED")

        if has_cross_project and self._contains(question, self._MATCHED_COHORT_TERMS):
            self._add(
                assertions,
                "OVERLAPPING_PROJECTS_IDENTIFIED",
                "SINGLE_ARM_PROJECTS_EXCLUDED",
            )
            if has_comparison:
                self._add(assertions, "COMPARABLE_COHORT_BOUND")

        if has_analysis and self._contains(question, self._MULTIFEATURE_TERMS):
            self._add(assertions, "MULTIFEATURE_ANALYSIS_COMPLETED")
        if (
            has_comparison
            and has_retrieval
            and self._has_action_before(actions, "compare_groups", "retrieve_evidence")
        ):
            self._add(assertions, "DATA_COMPARISON_BEFORE_RETRIEVAL")
        if has_cross_project:
            self._add(assertions, "PROJECT_VALIDATION_COMPLETED")
        if has_cross_disease:
            self._add(assertions, "DISEASE_VALIDATION_COMPLETED")

        conflict_or_speculation = (
            self._contains(question, self._CONFLICT_TERMS)
            or any(status in {"conflicted", "speculative"} for status in trace.evidenceSupportStatuses)
            or trace.stopReasonCode == "QUALITY_RISK"
        )
        if conflict_or_speculation:
            self._add(assertions, "CONFLICT_OR_SPECULATION_DETECTED")
            if trace.stopReasonCode == "QUALITY_RISK" or trace.supportStatusEscalated:
                self._add(assertions, "CLAIM_DOWNGRADED")
        if (
            self._contains(question, ("speculative", "推测"))
            and "graph" in routes
            and has_retrieval
            and trace.evidenceBindingCount > 0
            and not trace.supportStatusEscalated
        ):
            self._add(assertions, "SPECULATIVE_STATUS_RETAINED")

        if has_comparison or has_stratification or has_analysis or has_cross_project:
            self._add(limitations, "OBSERVATIONAL_COMPARISON_ONLY")
        if trace.stopReasonCode == "QUALITY_RISK":
            self._add(limitations, "QUALITY_RISK_REPORTED", "NO_CAUSAL_CONCLUSION")
        if has_cross_disease and has_analysis:
            self._add(limitations, "NO_CAUSAL_CONCLUSION")
        if (
            self._contains(question, self._FUNCTION_TERMS)
            and "graph" in routes
            and has_retrieval
        ):
            self._add(limitations, "NO_FUNCTIONAL_CAUSAL_CONCLUSION")

        return ResultOracleObservation(
            caseId=task.caseId,
            actionPath=list(dict.fromkeys(actions))[:8],
            sourceRoutes=routes,
            assertionCodes=assertions,
            limitationCodes=limitations,
            stopReasonCode=trace.stopReasonCode,
        )


def closed_observation_for_trace(
    task: EvalTask, trace: TraceProjection
) -> ResultOracleObservation:
    """Convenience wrapper used by the runner and offline audit."""

    return TraceOracleAssertionAdapter().observe(task, trace)
