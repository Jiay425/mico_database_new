from __future__ import annotations

from evals.p2j4_trace_oracle_audit import audit_existing_traces


def test_seven_trace_oracles_pass_without_external_calls() -> None:
    payload = audit_existing_traces()

    assert payload["externalCalls"] is False
    assert payload["caseCount"] == 7
    assert payload["passCount"] == 7
    assert all(item["status"] == "PASS" for item in payload["results"])


def test_trace_oracle_audit_persists_only_closed_verification_fields() -> None:
    payload = audit_existing_traces()
    serialized = repr(payload)

    assert "question" not in serialized
    assert "sourceSampleId" not in serialized
    assert "internalRecordId" not in serialized
    assert "select " not in serialized.lower()
    assert "password" not in serialized.lower()
