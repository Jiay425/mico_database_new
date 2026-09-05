from scripts.run_4d4_runtime_capability_closure import _run_checks


def test_4d4_offline_runtime_capability_checks_are_provider_free() -> None:
    checks = _run_checks()
    assert checks["task_a_capability_boundary_fixed"] is True
    assert checks["task_d_analysis_plan_contract_fixed"] is True
    assert checks["objective_boundary_closure"] is True
    assert checks["blocked_objective_semantics"] is True
    assert checks["task_a_can_finish_with_limitation"] is True
    assert checks["task_d_can_finish_or_fail_closed_with_limitation"] is True
    assert checks["decision_state_schema_changed"] is False
    assert checks["numeric_stratifier_contract_ready"] is True
    assert checks["numeric_stratifier_materializer_contract_ready"] is True
    assert checks["stratified_typed_route_fixed"] is True
    assert checks["numeric_insufficient_sample_fail_closed"] is True
    assert checks["completion_semantics_fixed"] is True
    assert checks["policy_or_model_calls"] == 0
    assert checks["a100_started"] is False
