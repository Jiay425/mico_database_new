"""Adapt transient Java read results into closed, value-free assertion codes.

The Java tool remains the only SQL/database boundary.  This module receives
its already-bounded response in memory and returns codes only: it deliberately
does not retain a row, column value, query, identifier, or numeric result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from mico_agent_runtime.contracts.tools import JavaToolResponse


class JavaBoundedAssertionError(RuntimeError):
    """A closed adapter failure; its message is safe to persist."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class ClosedJavaAssertions:
    """Durable-safe output of a transient Java bounded-result inspection."""

    assertion_codes: tuple[str, ...]
    limitation_codes: tuple[str, ...]

    def merged(self, other: "ClosedJavaAssertions") -> "ClosedJavaAssertions":
        return ClosedJavaAssertions(
            assertion_codes=tuple(dict.fromkeys((*self.assertion_codes, *other.assertion_codes))),
            limitation_codes=tuple(dict.fromkeys((*self.limitation_codes, *other.limitation_codes))),
        )


class JavaBoundedResultAssertionAdapter:
    """Translate known bounded-result shapes into no-value assertion codes."""

    _JAVA_TRANSIENT = "JAVA_TRANSIENT_EVIDENCE"

    def _rows(self, response: JavaToolResponse) -> list[Mapping[str, Any]]:
        if response.status != "COMPLETED" or response.dataSnapshot is None:
            raise JavaBoundedAssertionError("JAVA_BOUNDED_RESULT_UNAVAILABLE")
        if response.dataSnapshot.snapshotPersistence != "transient":
            raise JavaBoundedAssertionError("JAVA_BOUNDED_RESULT_PERSISTENCE_INVALID")
        payload = response.data
        if not isinstance(payload, Mapping):
            raise JavaBoundedAssertionError("JAVA_BOUNDED_RESULT_SHAPE_INVALID")
        rows = payload.get("rows")
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise JavaBoundedAssertionError("JAVA_BOUNDED_RESULT_ROWS_INVALID")
        if len(rows) > response.dataSnapshot.rowCount:
            raise JavaBoundedAssertionError("JAVA_BOUNDED_RESULT_ROW_BOUND_INVALID")
        if not all(isinstance(row, Mapping) for row in rows):
            raise JavaBoundedAssertionError("JAVA_BOUNDED_RESULT_ROW_INVALID")
        return list(rows)

    @staticmethod
    def _has_nonempty_text(rows: Sequence[Mapping[str, Any]], name: str) -> bool:
        return any(isinstance(row.get(name), str) and row[name].strip() for row in rows)

    @staticmethod
    def _has_nonnegative_number(rows: Sequence[Mapping[str, Any]], name: str) -> bool:
        return any(
            isinstance(row.get(name), (int, float))
            and not isinstance(row.get(name), bool)
            and row[name] >= 0
            for row in rows
        )

    def bounded_aggregate(self, response: JavaToolResponse) -> ClosedJavaAssertions:
        rows = self._rows(response)
        if not rows or not self._has_nonempty_text(rows, "candidate_feature"):
            raise JavaBoundedAssertionError("JAVA_BOUNDED_CANDIDATE_UNAVAILABLE")
        if not self._has_nonnegative_number(rows, "coverage_count"):
            raise JavaBoundedAssertionError("JAVA_BOUNDED_AGGREGATE_INVALID")
        return ClosedJavaAssertions(
            (self._JAVA_TRANSIENT, "AGGREGATE_NONNEGATIVE", "CANDIDATE_SEED_BOUND"),
            ("SAMPLE_KEY_NOT_SUBJECT_COUNT",),
        )

    def bounded_grouped_aggregate(self, response: JavaToolResponse) -> ClosedJavaAssertions:
        rows = self._rows(response)
        labels = {row.get("disease_label") for row in rows if isinstance(row.get("disease_label"), str)}
        if len(labels) < 2 or not self._has_nonempty_text(rows, "candidate_feature"):
            raise JavaBoundedAssertionError("JAVA_BOUNDED_GROUPED_RESULT_UNAVAILABLE")
        if not self._has_nonnegative_number(rows, "coverage_count"):
            raise JavaBoundedAssertionError("JAVA_BOUNDED_AGGREGATE_INVALID")
        return ClosedJavaAssertions(
            (self._JAVA_TRANSIENT, "GROUPED_AGGREGATE_PRESENT", "CANDIDATE_SEED_BOUND"),
            ("DISEASE_LABEL_UNVERIFIED", "SAMPLE_KEY_NOT_SUBJECT_COUNT"),
        )

    def within_disease_project_comparison(self, response: JavaToolResponse) -> ClosedJavaAssertions:
        rows = self._rows(response)
        projects = {row.get("project_key") for row in rows if isinstance(row.get("project_key"), str)}
        if len(projects) < 2 or not self._has_nonempty_text(rows, "candidate_feature"):
            raise JavaBoundedAssertionError("JAVA_BOUNDED_TWO_PROJECT_SCENARIO_UNAVAILABLE")
        if not self._has_nonnegative_number(rows, "coverage_count"):
            raise JavaBoundedAssertionError("JAVA_BOUNDED_AGGREGATE_INVALID")
        return ClosedJavaAssertions(
            (
                self._JAVA_TRANSIENT,
                "TWO_PROJECT_COHORTS_NONEMPTY",
                "FEATURE_SEED_BOUND",
                "PROJECT_FIELD_RETAINED",
            ),
            ("OBSERVATIONAL_COMPARISON_ONLY",),
        )

    def project_distribution(self, response: JavaToolResponse) -> tuple[ClosedJavaAssertions, list[Mapping[str, Any]]]:
        """Return transient rows only to the in-memory provisioner, never to a record."""

        rows = self._rows(response)
        if not rows or not self._has_nonnegative_number(rows, "cohort_count"):
            raise JavaBoundedAssertionError("JAVA_BOUNDED_PROJECT_DISTRIBUTION_UNAVAILABLE")
        if not self._has_nonempty_text(rows, "project_key") or not self._has_nonempty_text(rows, "disease_label"):
            raise JavaBoundedAssertionError("JAVA_BOUNDED_PROJECT_DISTRIBUTION_INVALID")
        return ClosedJavaAssertions((self._JAVA_TRANSIENT,), ("OBSERVATIONAL_COMPARISON_ONLY",)), rows

    def single_project_candidate(self, response: JavaToolResponse) -> ClosedJavaAssertions:
        rows = self._rows(response)
        projects_by_disease: dict[str, set[str]] = {}
        for row in rows:
            disease = row.get("disease_label")
            project = row.get("project_key")
            if isinstance(disease, str) and disease and isinstance(project, str) and project:
                projects_by_disease.setdefault(disease, set()).add(project)
        if not any(len(projects) == 1 for projects in projects_by_disease.values()) \
                or not self._has_nonempty_text(rows, "candidate_feature"):
            raise JavaBoundedAssertionError("JAVA_BOUNDED_SINGLE_PROJECT_SCENARIO_UNAVAILABLE")
        if not self._has_nonnegative_number(rows, "coverage_count"):
            raise JavaBoundedAssertionError("JAVA_BOUNDED_AGGREGATE_INVALID")
        return ClosedJavaAssertions(
            (self._JAVA_TRANSIENT, "SINGLE_PROJECT_PATTERN_DETECTED"),
            ("NOT_A_STABLE_BIOMARKER", "NO_CAUSAL_CONCLUSION"),
        )
