from __future__ import annotations

from datetime import datetime, timezone

from evals.p2j4_controlled_provider import ControlledScenarioProvider
from evals.p2j4_controlled_scenarios import validate_controlled_scenario_set
from evals.p2j4_java_bounded_assertions import JavaBoundedResultAssertionAdapter
from evals.p2j4_result_oracles import validate_result_oracle_set
from evals.p2j4_runner import load_task_set
from mico_agent_runtime.contracts.tools import JavaDataSnapshot, JavaQualitySummary, JavaToolResponse
from tests.conftest import make_call


class _ScenarioJavaPort:
    """Returns raw-looking rows only in memory; assertions must not retain them."""

    def execute(self, call):
        sql = getattr(call.arguments, "sql", "")
        if "p.disease =" in sql and "cohort_count" in sql:
            rows = [{"cohort_count": 0}]
        elif "AS disease_label" in sql and "GROUP BY p.disease, m.project_name" in sql and "a.microbe_name_standard" in sql:
            rows = [{
                "disease_label": "disease-private", "project_key": "project-private",
                "candidate_feature": "feature-private", "coverage_count": 3,
            }]
        elif "AS scenario_disease" in sql and "GROUP BY p.disease, m.project_name" in sql:
            rows = [
                {"scenario_disease": "disease-private", "project_key": "project-a", "candidate_feature": "feature-private", "coverage_count": 3},
                {"scenario_disease": "disease-private", "project_key": "project-b", "candidate_feature": "feature-private", "coverage_count": 2},
            ]
        elif "GROUP BY a.microbe_name_standard, p.disease" in sql:
            rows = [
                {"candidate_feature": "feature-private", "disease_label": "disease-a", "coverage_count": 3},
                {"candidate_feature": "feature-private", "disease_label": "disease-b", "coverage_count": 2},
            ]
        elif "GROUP BY p.disease, m.project_name" in sql:
            rows = [
                {"disease_label": "disease-a", "project_key": "project-a", "cohort_count": 3},
                {"disease_label": "disease-a", "project_key": "project-b", "cohort_count": 2},
                {"disease_label": "disease-b", "project_key": "project-b", "cohort_count": 2},
            ]
        else:
            rows = [{"candidate_feature": "feature-private", "coverage_count": 3}]
        now = datetime(2026, 8, 24, tzinfo=timezone.utc)
        snapshot = JavaDataSnapshot(
            dataSnapshotId="transient-550e8400-e29b-41d4-a716-446655440000",
            dataSource="java_agent_read_model",
            queryHash="sha256:" + "a" * 64,
            rowCount=len(rows),
            generatedAt=now,
            snapshotPersistence="transient",
        )
        return JavaToolResponse(
            toolCallId=call.toolCallId,
            runId=call.runId,
            status="COMPLETED",
            source="java_agent_read_model",
            rowCount=len(rows),
            schemaVersion="java-read-model-v1",
            generatedAt=now,
            dataSnapshot=snapshot,
            qualitySummary=JavaQualitySummary(subjectLinkStatus="unverified"),
            data={"columns": list(rows[0]), "rows": rows},
        )


def _seeds():
    oracles = validate_result_oracle_set(load_task_set())["oracles"]
    return validate_controlled_scenario_set(oracles)["seeds"]


def test_java_bounded_adapter_emits_codes_without_raw_candidate_value() -> None:
    port = _ScenarioJavaPort()
    response = port.execute(make_call())
    assertions = JavaBoundedResultAssertionAdapter().bounded_aggregate(response)

    assert assertions.assertion_codes == (
        "JAVA_TRANSIENT_EVIDENCE", "AGGREGATE_NONNEGATIVE", "CANDIDATE_SEED_BOUND",
    )
    assert "feature-private" not in repr(assertions)


def test_every_controlled_scenario_is_supplied_from_a_bounded_java_result() -> None:
    port = _ScenarioJavaPort()
    provider = ControlledScenarioProvider(port)
    for seed in _seeds():
        session = provider.provision("run-00000000000000000000000000000001", seed)
        response = session.execute(make_call())
        serialized = repr(session.assertions)
        assert response.dataSnapshot is not None
        assert response.dataSnapshot.snapshotPersistence == "transient"
        assert "feature-private" not in serialized
        # The remaining single-project codes are emitted only after the real
        # Agent completes cross-project validation and vector retrieval.
        assert "JAVA_TRANSIENT_EVIDENCE" in session.assertions.assertion_codes
        assert set(seed.expectedLimitationCodes).issubset(session.assertions.limitation_codes)
