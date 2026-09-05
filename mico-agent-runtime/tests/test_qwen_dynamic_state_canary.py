from __future__ import annotations

import json

import httpx

from scripts.run_qwen_dynamic_state_canary import (
    STATE_NAMES,
    RecordingTransport,
    build_dynamic_states,
    prepare_dynamic_canary,
    run_live_canary,
)


def test_dynamic_states_are_valid_and_keep_provenance_outside_policy_state() -> None:
    states, provenance, _catalog = build_dynamic_states()

    assert tuple(states) == STATE_NAMES
    assert all(state.task.query for state in states.values())
    assert all(set(state.model_dump(mode="json")) == {
        "task",
        "data_state",
        "analysis_state",
        "evidence_state",
        "progress",
        "action_space",
    } for state in states.values())
    assert provenance["s1"]["state_origin"] == "previous_gemini_canary_cache"
    assert all(
        provenance[name]["state_origin"] == "controlled_canary"
        for name in ("s3", "s4", "s5", "s6")
    )
    assert all(item["training_eligible"] is False for item in provenance.values())


def test_dynamic_pair_boundaries_are_explicit() -> None:
    states, _provenance, _catalog = build_dynamic_states()

    assert states["s1"].analysis_state.group_comparison.status == "not_started"
    assert states["s2"].analysis_state.group_comparison.status == "completed"
    assert states["s2"].analysis_state.confounder_adjustment.status == "not_started"
    assert states["s3"].analysis_state.confounder_adjustment.status == "completed"
    assert states["s3"].analysis_state.group_comparison.effect_size == 0.62
    assert states["s3"].analysis_state.confounder_adjustment.adjusted_effect_size == 0.18
    assert states["s4"].analysis_state.cross_project_validation.heterogeneity == "low"
    assert states["s5"].analysis_state.cross_project_validation.heterogeneity == "high"
    assert states["s4"].data_state.project_state.project_count == 4
    assert states["s5"].data_state.project_state.project_count == 4
    assert states["s6"].progress.remaining_objectives == []
    assert "finish" in states["s6"].action_space.available_actions


def test_prepare_is_audit_only_and_writes_six_state_manifest(tmp_path) -> None:
    report = prepare_dynamic_canary(tmp_path)

    assert report["status"] == "PREPARED"
    assert report["DYNAMIC_CANARY_READY"] is True
    assert report["FULL_E2E_READY"] is False
    assert report["BASE_QWEN_SMOKE"] == "PASS"
    assert report["remote_contacted"] is False
    assert report["transport_attempts"] == 0
    assert report["expected_policy_calls"] == 6
    assert all((tmp_path / f"state_{name}.json").exists() for name in STATE_NAMES)
    manifest = json.loads((tmp_path / "state_manifest.json").read_text(encoding="utf-8"))
    assert manifest["no_remote_contact"] is True
    assert manifest["training_eligible"] is False
    assert manifest["state_origin_policy"]["s4"] == "controlled_canary"
    assert manifest["state_origin_policy"]["s5"] == "controlled_canary"
    assert all(item["expected_policy_calls"] == 1 for item in manifest["states"])
    pair2 = next(item for item in manifest["pairs"] if item["pair_id"] == "pair_2")
    assert pair2["numeric_baseline_visible"] is False
    assert "no mapped numeric effect" in pair2["interpretation_note"]


def test_live_runner_reuses_production_policy_contract_and_records_six_calls(tmp_path) -> None:
    states, _provenance, _catalog = build_dynamic_states()
    seen_states: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        payload = json.loads(body["messages"][1]["content"])
        seen_states.append(payload["state"])
        state = payload["state"]
        if state["evidence_state"]["status"] == "completed":
            action = "finish"
            stop_reason = "EVIDENCE_SUFFICIENT"
        elif state["analysis_state"]["cross_project_validation"]["status"] == "completed":
            action = "retrieve_evidence"
            stop_reason = None
        elif state["analysis_state"]["confounder_adjustment"]["status"] == "completed":
            action = "cross_project_validate" if "cross_project_validate" in state["action_space"]["available_actions"] else "analyze_projection"
            stop_reason = None
        elif state["analysis_state"]["group_comparison"]["status"] == "completed":
            action = "adjust_confounders"
            stop_reason = None
        elif state["data_state"]["has_tabular_data"]:
            action = "compare_groups"
            stop_reason = None
        else:
            action = "execute_read_query"
            stop_reason = None
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps({
                "selected_action": action,
                "decision_reason": f"controlled decision for {action}",
                "alternative_actions": [],
                "stop_reason": stop_reason,
            })}}],
        })

    # Exercise the production provider directly here to avoid any network and
    # still assert the same contract shape used by the live runner.
    transport = RecordingTransport(httpx.MockTransport(handler))
    from mico_agent_runtime.ports.decision_policy import HttpDecisionSftPlannerPort
    from mico_agent_runtime.ports.gemini_resilience import GeminiRequestBudget
    provider = HttpDecisionSftPlannerPort(
        "http://qwen.local:19002",
        "qwen3-8b-decision-base",
        transport=transport,
        request_budget=GeminiRequestBudget(total_limit=6, role_limits={"policy": 6}),
        policy_origin="qwen_model",
    )
    try:
        decisions = [provider.select_action(state) for state in states.values()]
    finally:
        provider.close()

    assert len(seen_states) == 6
    assert [decision.selected_action for decision in decisions] == [
        "compare_groups",
        "adjust_confounders",
        "analyze_projection",
        "retrieve_evidence",
        "retrieve_evidence",
        "finish",
    ]
    assert all(decision.decision_reason for decision in decisions)
    assert all(
        decision.selected_action in state.action_space.available_actions
        for decision, state in zip(decisions, states.values())
    )


def test_live_runner_evaluates_later_states_after_policy_failure(tmp_path) -> None:
    states, _provenance, _catalog = build_dynamic_states()
    s3_attempts = 0
    seen_states: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal s3_attempts
        body = json.loads(request.content)
        payload = json.loads(body["messages"][1]["content"])
        state = payload["state"]
        action_count = state["progress"]["action_count"]
        state_name = {
            1: "s1",
            2: "s2",
            3: "s3",
            4: "s4/s5",
            5: "s6",
        }.get(action_count, "unknown")
        seen_states.append(state_name)
        if action_count == 3:
            s3_attempts += 1
            action = "cross_project_validation" if s3_attempts == 1 else "cross_project_validate"
            stop_reason = None
        elif action_count == 4:
            action = "retrieve_evidence"
            stop_reason = None
        elif action_count == 1:
            action = "compare_groups"
            stop_reason = None
        elif action_count == 2:
            action = "adjust_confounders"
            stop_reason = None
        else:
            action = "finish"
            stop_reason = "EVIDENCE_SUFFICIENT"
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps({
                "selected_action": action,
                "decision_reason": f"test decision for {state_name}",
                "alternative_actions": [],
                "stop_reason": stop_reason,
            })}}],
        })

    report = run_live_canary(
        tmp_path,
        states=states,
        base_url="http://qwen.local:19002",
        model="qwen3-8b-decision-base",
        max_requests=12,
        transport=httpx.MockTransport(handler),
    )

    assert report["status"] == "FAILED"
    assert report["DYNAMIC_STATE_CANARY"] == "FAIL"
    assert report["DYNAMIC_CANARY_READY"] is True
    assert report["all_states_evaluated"] is True
    assert report["aborted"] is False
    assert report["transport_attempts"] == 7
    assert report["repair_count"] == 1
    assert report["model_responses"] == 7
    assert report["contract_valid"] == {"count": 5, "total": 6}
    assert report["available_action_compliance"] == {"count": 5, "total": 6}
    assert report["objective_action_confusion_count"] == 1
    assert report["unavailable_action_count"] == 1
    assert report["redundant_action_count"] == 0
    assert report["state_sensitive_pairs"] == {
        "count": 1,
        "total": 3,
        "pairs": ["pair_1"],
    }
    assert report["S6_finish"] is True
    assert report["finish_missed"] is False
    assert [item["state_name"] for item in report["states"]] == list(STATE_NAMES)
    assert report["states"][2]["state_status"] == "FAIL"
    assert report["states"][2]["failure_scope"] == "state_policy"
    assert report["states"][3]["state_status"] == "PASS"
    assert report["states"][4]["state_status"] == "PASS"
    assert report["states"][5]["state_status"] == "PASS"


def test_live_runner_aborts_later_states_on_transport_failure(tmp_path) -> None:
    states, _provenance, _catalog = build_dynamic_states()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "service unavailable"})

    report = run_live_canary(
        tmp_path,
        states=states,
        base_url="http://qwen.local:19002",
        model="qwen3-8b-decision-base",
        max_requests=12,
        transport=httpx.MockTransport(handler),
    )

    assert report["status"] == "FAILED"
    assert report["aborted"] is True
    assert report["abort_reason"].startswith("RuntimeError: POLICY_DECISION_FAILED")
    assert report["states"][0]["failure_scope"] == "global_transport"
    assert len(report["states"]) == 1
