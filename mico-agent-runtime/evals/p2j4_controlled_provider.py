"""Real, bounded Java scenario supply for the five P2-J4.1 controlled cases.

Queries are owned here (not by an LLM), pass through the existing Java
read-only boundary, and remain process-local.  Persistent eval artifacts see
only scenario status plus closed assertion/limitation codes.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Mapping

from mico_agent_runtime.contracts.tools import ExecuteReadQueryArguments, ExecuteReadQueryJavaToolCall, JavaToolResponse
from mico_agent_runtime.ports.java_agent import JavaAgentToolPort

from evals.p2j4_controlled_scenarios import ControlledScenarioSeed
from evals.p2j4_java_bounded_assertions import (
    ClosedJavaAssertions,
    JavaBoundedAssertionError,
    JavaBoundedResultAssertionAdapter,
)


class ControlledScenarioUnavailable(RuntimeError):
    """No genuine matching scenario was found; no synthetic fallback exists."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _call_id(run_id: str, scenario_ref: str, stage: str) -> str:
    return "call-" + sha256(f"{run_id}|{scenario_ref}|{stage}".encode()).hexdigest()[:32]


def _sql_text(value: str) -> str:
    """Quote an in-memory Java-returned value for another Java-validated read."""

    return "'" + value.replace("'", "''").replace("\\", "\\\\") + "'"


def _bounded_sql(sql: str) -> str:
    """The closed Python tool contract forbids control characters in SQL."""

    return " ".join(sql.split())


@dataclass
class ControlledScenarioSession:
    """A prepared real scenario exposed as a Java-port decorator for one Eval run."""

    base_port: JavaAgentToolPort
    seed: ControlledScenarioSeed
    scenario_sql: str
    adapter_method: str
    assertions: ClosedJavaAssertions
    _adapter: JavaBoundedResultAssertionAdapter = field(default_factory=JavaBoundedResultAssertionAdapter)

    def execute(self, call: Any) -> JavaToolResponse:
        if getattr(call, "toolName", None) == "describe_read_schema":
            return self.base_port.execute(call)
        if getattr(call, "toolName", None) != "execute_read_query":
            return self.base_port.execute(call)
        controlled = ExecuteReadQueryJavaToolCall(
            toolName="execute_read_query",
            runId=call.runId,
            toolCallId=call.toolCallId,
            arguments=ExecuteReadQueryArguments(sql=_bounded_sql(self.scenario_sql), limit=1000),
        )
        response = self.base_port.execute(controlled)
        method = getattr(self._adapter, self.adapter_method)
        if self.adapter_method == "project_distribution":
            closed, _ = method(response)
        else:
            closed = method(response)
        self.assertions = self.assertions.merged(closed)
        return response


class ControlledScenarioProvider:
    """Provision only real scenarios using bounded Java reads, never fixture data."""

    _NONZERO_AGGREGATE = """
SELECT a.microbe_name_standard AS candidate_feature,
       COUNT(DISTINCT a.sample_id) AS coverage_count
FROM microbe_abundance_standard a
WHERE a.microbe_name_standard IS NOT NULL AND a.abundance_value > 0
GROUP BY a.microbe_name_standard
ORDER BY coverage_count DESC
LIMIT 1
""".strip()

    _CANDIDATE_SCAN = """
SELECT DISTINCT a.microbe_name_standard AS candidate_feature
FROM microbe_abundance_standard a
WHERE a.microbe_name_standard IS NOT NULL AND a.abundance_value > 0
LIMIT 64
""".strip()

    _GROUPED_AGGREGATE = """
WITH candidate AS (
  SELECT a.microbe_name_standard AS candidate_feature
  FROM microbe_abundance_standard a
  WHERE a.microbe_name_standard IS NOT NULL AND a.abundance_value > 0
  GROUP BY a.microbe_name_standard
  ORDER BY COUNT(DISTINCT a.sample_id) DESC
  LIMIT 1
)
SELECT a.microbe_name_standard AS candidate_feature,
       p.disease AS disease_label,
       COUNT(DISTINCT a.sample_id) AS coverage_count
FROM microbe_abundance_standard a
JOIN patients p ON p.patient_id = a.patient_id
JOIN candidate c ON c.candidate_feature = a.microbe_name_standard
WHERE p.disease IS NOT NULL AND a.abundance_value > 0
GROUP BY a.microbe_name_standard, p.disease
ORDER BY coverage_count DESC
LIMIT 1000
""".strip()

    _TWO_PROJECT_FEATURE = """
WITH candidate AS (
  SELECT p.disease AS scenario_disease,
         a.microbe_name_standard AS candidate_feature
  FROM patients p
  JOIN meta2db_sample_metadata m ON m.patient_id = p.patient_id
  JOIN microbe_abundance_standard a ON a.patient_id = p.patient_id
  WHERE p.disease IS NOT NULL AND m.project_name IS NOT NULL
    AND a.microbe_name_standard IS NOT NULL AND a.abundance_value > 0
  GROUP BY p.disease, a.microbe_name_standard
  HAVING COUNT(DISTINCT m.project_name) >= 2
  ORDER BY COUNT(DISTINCT a.sample_id) DESC
  LIMIT 1
)
SELECT c.scenario_disease AS scenario_disease,
       m.project_name AS project_key,
       c.candidate_feature AS candidate_feature,
       COUNT(DISTINCT a.sample_id) AS coverage_count
FROM candidate c
JOIN patients p ON p.disease = c.scenario_disease
JOIN meta2db_sample_metadata m ON m.patient_id = p.patient_id
JOIN microbe_abundance_standard a ON a.patient_id = p.patient_id
  AND a.microbe_name_standard = c.candidate_feature
WHERE m.project_name IS NOT NULL AND a.abundance_value > 0
GROUP BY c.scenario_disease, m.project_name, c.candidate_feature
ORDER BY coverage_count DESC
LIMIT 1000
""".strip()

    _PROJECT_DISTRIBUTION = """
SELECT p.disease AS disease_label,
       m.project_name AS project_key,
       COUNT(DISTINCT m.sample_id) AS cohort_count
FROM patients p
JOIN meta2db_sample_metadata m ON m.patient_id = p.patient_id
WHERE p.disease IS NOT NULL AND m.project_name IS NOT NULL
GROUP BY p.disease, m.project_name
ORDER BY cohort_count DESC
LIMIT 1000
""".strip()

    _SINGLE_PROJECT_CANDIDATE = """
WITH candidate AS (
  SELECT p.disease AS disease_label,
         a.microbe_name_standard AS candidate_feature
  FROM patients p
  JOIN meta2db_sample_metadata m ON m.patient_id = p.patient_id
  JOIN microbe_abundance_standard a ON a.patient_id = p.patient_id
  WHERE p.disease IS NOT NULL AND m.project_name IS NOT NULL
    AND a.microbe_name_standard IS NOT NULL AND a.abundance_value > 0
  GROUP BY p.disease, a.microbe_name_standard
  HAVING COUNT(DISTINCT m.project_name) = 1
  ORDER BY COUNT(DISTINCT a.sample_id) DESC
  LIMIT 1
)
SELECT c.disease_label AS disease_label,
       m.project_name AS project_key,
       c.candidate_feature AS candidate_feature,
       COUNT(DISTINCT a.sample_id) AS coverage_count
FROM candidate c
JOIN patients p ON p.disease = c.disease_label
JOIN meta2db_sample_metadata m ON m.patient_id = p.patient_id
JOIN microbe_abundance_standard a ON a.patient_id = p.patient_id
  AND a.microbe_name_standard = c.candidate_feature
WHERE m.project_name IS NOT NULL AND a.abundance_value > 0
GROUP BY c.disease_label, m.project_name, c.candidate_feature
ORDER BY coverage_count DESC
LIMIT 1000
""".strip()

    def __init__(self, java_port: JavaAgentToolPort) -> None:
        self._java_port = java_port
        self._adapter = JavaBoundedResultAssertionAdapter()

    def _read(self, run_id: str, seed: ControlledScenarioSeed, stage: str, sql: str) -> JavaToolResponse:
        call = ExecuteReadQueryJavaToolCall(
            toolName="execute_read_query",
            runId=run_id,
            toolCallId=_call_id(run_id, seed.scenarioRef, stage),
            arguments=ExecuteReadQueryArguments(sql=_bounded_sql(sql), limit=1000),
        )
        return self._java_port.execute(call)

    def _candidate_features(
        self, run_id: str, seed: ControlledScenarioSeed, *, maximum: int = 12
    ) -> list[str]:
        response = self._read(run_id, seed, "candidate_scan", self._CANDIDATE_SCAN)
        try:
            rows = self._adapter._rows(response)
        except JavaBoundedAssertionError as exc:
            raise ControlledScenarioUnavailable(exc.code) from exc
        candidates = [
            row["candidate_feature"]
            for row in rows
            if isinstance(row.get("candidate_feature"), str) and row["candidate_feature"].strip()
        ]
        unique = list(dict.fromkeys(candidates))[:maximum]
        if not unique:
            raise ControlledScenarioUnavailable("CONTROLLED_SCENARIO_CANDIDATE_UNAVAILABLE")
        return unique

    @staticmethod
    def _candidate_coverage_sql(candidate: str) -> str:
        return (
            "SELECT a.microbe_name_standard AS candidate_feature, "
            "COUNT(DISTINCT a.sample_id) AS coverage_count "
            "FROM microbe_abundance_standard a "
            f"WHERE a.microbe_name_standard = {_sql_text(candidate)} AND a.abundance_value > 0 "
            "GROUP BY a.microbe_name_standard LIMIT 1"
        )

    @staticmethod
    def _candidate_grouped_sql(candidate: str) -> str:
        return (
            "SELECT a.microbe_name_standard AS candidate_feature, p.disease AS disease_label, "
            "COUNT(DISTINCT a.sample_id) AS coverage_count "
            "FROM microbe_abundance_standard a JOIN patients p ON p.patient_id = a.patient_id "
            f"WHERE a.microbe_name_standard = {_sql_text(candidate)} "
            "AND p.disease IS NOT NULL AND a.abundance_value > 0 "
            "GROUP BY a.microbe_name_standard, p.disease ORDER BY coverage_count DESC LIMIT 1000"
        )

    @staticmethod
    def _candidate_project_sql(candidate: str) -> str:
        return (
            "SELECT p.disease AS scenario_disease, m.project_name AS project_key, "
            "a.microbe_name_standard AS candidate_feature, COUNT(DISTINCT a.sample_id) AS coverage_count "
            "FROM patients p JOIN meta2db_sample_metadata m ON m.patient_id = p.patient_id "
            "JOIN microbe_abundance_standard a ON a.patient_id = p.patient_id "
            f"WHERE a.microbe_name_standard = {_sql_text(candidate)} "
            "AND p.disease IS NOT NULL AND m.project_name IS NOT NULL AND a.abundance_value > 0 "
            "GROUP BY p.disease, m.project_name, a.microbe_name_standard ORDER BY coverage_count DESC LIMIT 1000"
        )

    @staticmethod
    def _candidate_single_project_sql(candidate: str) -> str:
        return (
            "SELECT p.disease AS disease_label, m.project_name AS project_key, "
            "a.microbe_name_standard AS candidate_feature, COUNT(DISTINCT a.sample_id) AS coverage_count "
            "FROM patients p JOIN meta2db_sample_metadata m ON m.patient_id = p.patient_id "
            "JOIN microbe_abundance_standard a ON a.patient_id = p.patient_id "
            f"WHERE a.microbe_name_standard = {_sql_text(candidate)} "
            "AND p.disease IS NOT NULL AND m.project_name IS NOT NULL AND a.abundance_value > 0 "
            "GROUP BY p.disease, m.project_name, a.microbe_name_standard ORDER BY coverage_count DESC LIMIT 1000"
        )

    @staticmethod
    def _rows(response: JavaToolResponse) -> list[Mapping[str, Any]]:
        payload = response.data
        rows = payload.get("rows") if isinstance(payload, Mapping) else None
        if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
            raise ControlledScenarioUnavailable("CONTROLLED_SCENARIO_JAVA_ROWS_INVALID")
        return rows

    def provision(self, run_id: str, seed: ControlledScenarioSeed) -> ControlledScenarioSession:
        try:
            if seed.scenarioKind == "bounded_aggregate":
                for candidate in self._candidate_features(run_id, seed):
                    sql = self._candidate_coverage_sql(candidate)
                    response = self._read(run_id, seed, "seed", sql)
                    try:
                        assertions = self._adapter.bounded_aggregate(response)
                    except JavaBoundedAssertionError:
                        continue
                    return ControlledScenarioSession(self._java_port, seed, sql, "bounded_aggregate", assertions)
                raise ControlledScenarioUnavailable("CONTROLLED_SCENARIO_AGGREGATE_UNAVAILABLE")
            if seed.scenarioKind == "bounded_grouped_aggregate":
                for candidate in self._candidate_features(run_id, seed):
                    sql = self._candidate_grouped_sql(candidate)
                    response = self._read(run_id, seed, "seed", sql)
                    try:
                        assertions = self._adapter.bounded_grouped_aggregate(response)
                    except JavaBoundedAssertionError:
                        continue
                    return ControlledScenarioSession(self._java_port, seed, sql, "bounded_grouped_aggregate", assertions)
                raise ControlledScenarioUnavailable("CONTROLLED_SCENARIO_GROUPED_AGGREGATE_UNAVAILABLE")
            if seed.scenarioKind == "within_disease_project_comparison":
                for candidate in self._candidate_features(run_id, seed):
                    sql = self._candidate_project_sql(candidate)
                    response = self._read(run_id, seed, "seed", sql)
                    try:
                        assertions = self._adapter.within_disease_project_comparison(response)
                    except JavaBoundedAssertionError:
                        continue
                    return ControlledScenarioSession(self._java_port, seed, sql, "within_disease_project_comparison", assertions)
                raise ControlledScenarioUnavailable("CONTROLLED_SCENARIO_TWO_PROJECT_UNAVAILABLE")
            if seed.scenarioKind == "cohort_reconstruction":
                response = self._read(run_id, seed, "distribution", self._PROJECT_DISTRIBUTION)
                assertions, rows = self._adapter.project_distribution(response)
                self._assert_reconstructable_cohort(run_id, seed, rows)
                assertions = assertions.merged(ClosedJavaAssertions(
                    ("PROJECT_IMBALANCE_DETECTED", "ELIGIBLE_COHORT_RECONSTRUCTED", "EMPTY_RESULT_BRANCH_HANDLED"),
                    ("EXCLUDED_PROJECTS_REPORTED",),
                ))
                return ControlledScenarioSession(self._java_port, seed, self._PROJECT_DISTRIBUTION, "project_distribution", assertions)
            if seed.scenarioKind == "single_project_downgrade":
                for candidate in self._candidate_features(run_id, seed, maximum=64):
                    sql = self._candidate_single_project_sql(candidate)
                    response = self._read(run_id, seed, "seed", sql)
                    try:
                        assertions = self._adapter.single_project_candidate(response)
                    except JavaBoundedAssertionError:
                        continue
                    return ControlledScenarioSession(self._java_port, seed, sql, "single_project_candidate", assertions)
                raise ControlledScenarioUnavailable("CONTROLLED_SCENARIO_SINGLE_PROJECT_UNAVAILABLE")
        except JavaBoundedAssertionError as exc:
            raise ControlledScenarioUnavailable(exc.code) from exc
        raise ControlledScenarioUnavailable("CONTROLLED_SCENARIO_KIND_UNSUPPORTED")

    def _assert_reconstructable_cohort(
        self, run_id: str, seed: ControlledScenarioSeed, rows: list[Mapping[str, Any]]
    ) -> None:
        by_project: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            project = row.get("project_key")
            disease = row.get("disease_label")
            if isinstance(project, str) and project and isinstance(disease, str) and disease:
                by_project[project].add(disease)
        eligible = [(project, labels) for project, labels in by_project.items() if len(labels) >= 2]
        isolated = [(project, next(iter(labels))) for project, labels in by_project.items() if len(labels) == 1]
        if not eligible or not isolated:
            raise ControlledScenarioUnavailable("CONTROLLED_SCENARIO_RECONSTRUCTION_UNAVAILABLE")
        eligible_labels = set().union(*(labels for _, labels in eligible))
        excluded_project, excluded_label = next(
            ((project, label) for project, label in isolated if label in eligible_labels),
            (None, None),
        )
        alternative_label = next(
            (label for label in eligible_labels if label != excluded_label), None
        )
        if not isinstance(excluded_project, str) or not isinstance(alternative_label, str):
            raise ControlledScenarioUnavailable("CONTROLLED_SCENARIO_RECONSTRUCTION_UNAVAILABLE")
        empty_sql = (
            "SELECT COUNT(DISTINCT m.sample_id) AS cohort_count "
            "FROM patients p JOIN meta2db_sample_metadata m ON m.patient_id = p.patient_id "
            f"WHERE p.disease = {_sql_text(alternative_label)} "
            f"AND m.project_name = {_sql_text(excluded_project)} LIMIT 1"
        )
        empty_response = self._read(run_id, seed, "empty_branch", empty_sql)
        empty_rows = self._rows(empty_response)
        if len(empty_rows) != 1 or empty_rows[0].get("cohort_count") != 0:
            raise ControlledScenarioUnavailable("CONTROLLED_SCENARIO_EMPTY_BRANCH_UNAVAILABLE")


def closed_observation_for_session(session: ControlledScenarioSession, trace: Any):
    """Combine a real supplied scenario with an Agent trace without retaining values."""

    from evals.p2j4_controlled_scenarios import ResultOracleObservation

    decisions = list(getattr(trace, "decisions", []) or [])
    actions = [getattr(item, "chosenAction", None) for item in decisions]
    if not actions:
        actions = [
            getattr(event, "actionName", None)
            for event in getattr(trace, "events", []) or []
            if getattr(event, "node", None) == "execute_action"
        ]
    action_path = [action for action in actions if isinstance(action, str)]
    assertion_codes = list(session.assertions.assertion_codes)
    limitation_codes = list(session.assertions.limitation_codes)
    source_routes = list(getattr(trace, "sourceRoutes", []) or [])
    if "java" not in source_routes:
        source_routes.append("java")
    if session.seed.scenarioKind == "single_project_downgrade":
        if "cross_project_validate" in action_path:
            assertion_codes.extend(("CROSS_PROJECT_VALIDATION_COMPLETED", "CLAIM_DOWNGRADED"))
        if "vector" in source_routes:
            assertion_codes.append("VECTOR_EVIDENCE_BOUND")
    return ResultOracleObservation(
        caseId=session.seed.caseId,
        scenarioRef=session.seed.scenarioRef,
        actionPath=list(dict.fromkeys(action_path)),
        sourceRoutes=list(dict.fromkeys(source_routes)),
        assertionCodes=list(dict.fromkeys(assertion_codes)),
        limitationCodes=list(dict.fromkeys(limitation_codes)),
        stopReasonCode=getattr(trace, "stopReasonCode", None),
    )
