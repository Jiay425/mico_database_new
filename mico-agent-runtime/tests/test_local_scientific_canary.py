from __future__ import annotations

from scripts.run_local_scientific_canary import (
    build_fixture_catalog,
    run_failure_canaries,
    run_happy,
)


def test_local_canary_runs_three_real_policy_http_turns(tmp_path):
    result = run_happy(tmp_path)
    trace = result["trace"]
    assert trace["policy_request_count"] == 3
    assert trace["policy_origin"] == "local_http_stub"
    assert trace["training_eligible"] is False
    assert trace["action_history"] == [
        "execute_read_query",
        "compare_groups",
        "adjust_confounders",
    ]
    assert len(trace["rounds"]) == 3
    assert trace["rounds"][0]["state_before"]["data_state"]["has_tabular_data"] is False
    assert trace["rounds"][0]["state_after"]["data_state"]["has_tabular_data"] is True
    assert trace["rounds"][1]["state_after"]["analysis_state"]["group_comparison"]["status"] == "completed"
    assert trace["rounds"][2]["state_after"]["analysis_state"]["confounder_adjustment"]["status"] == "completed"
    states = [record.state_snapshot for record in result["final_state"]["decisionRecords"]]
    assert states[0]["data_state"]["has_tabular_data"] is False
    assert states[1]["data_state"]["has_tabular_data"] is True
    assert states[1]["data_state"]["group_state"]["group_count"] == 2
    assert states[2]["analysis_state"]["group_comparison"]["status"] == "completed"
    assert result["final_state"]["decisionState"].analysis_state.confounder_adjustment.status == "completed"
    assert all(item["codeVersion"] == "typed-analysis-operator-v1" for item in result["analysis_results"])


def test_local_canary_failure_injections_are_fail_closed(tmp_path):
    failures = run_failure_canaries(tmp_path, build_fixture_catalog())
    assert set(failures) == {
        "F1_invalid_action",
        "F2_unavailable_action",
        "F3_http_500",
        "F4_materializer_mismatch",
        "F5_materializer_failure",
        "F6_insufficient_data",
    }
    assert failures["F1_invalid_action"]["outcome"]["code"] == "POLICY_DECISION_FAILED"
    assert failures["F2_unavailable_action"]["outcome"]["selected_action"] == "execute_read_query"
    assert failures["F3_http_500"]["outcome"]["code"] == "POLICY_DECISION_FAILED"
    assert failures["F4_materializer_mismatch"]["code"] == "MATERIALIZATION_ACTION_MISMATCH"
    assert failures["F5_materializer_failure"]["status"] == "fail_closed"
    assert failures["F6_insufficient_data"]["status"] == "raised"
