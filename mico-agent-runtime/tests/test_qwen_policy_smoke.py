from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx

from mico_agent_runtime.contracts.decision_state import ScientificDecisionState
from mico_agent_runtime.contracts.schema_catalog import (
    SchemaEntitySemantics,
    SchemaFieldSemantics,
    SchemaSemanticCatalog,
)
from scripts.run_qwen_policy_smoke import (
    RecordingTransport,
    audit_cached_states,
    run_policy_smoke,
)


def _catalog() -> SchemaSemanticCatalog:
    return SchemaSemanticCatalog(
        schemaVersion="schema-catalog-v1",
        source="java_schema_contract",
        generatedAt=datetime.now(timezone.utc),
        entities=[
            SchemaEntitySemantics(
                entityId="sample",
                entityName="sample",
                sourceTable="sample",
                fields=[
                    SchemaFieldSemantics(
                        fieldId="sample.disease",
                        name="disease",
                        dataType="string",
                        nullable=True,
                        semanticStatus="verified",
                        filterable=True,
                        groupable=True,
                        displayable=True,
                        scientificCapabilities=["dimension", "stratifier"],
                        description="disease group",
                    ),
                    SchemaFieldSemantics(
                        fieldId="sample.age",
                        name="age",
                        dataType="integer",
                        nullable=True,
                        filterable=True,
                        groupable=True,
                        aggregatable=True,
                        displayable=True,
                        semanticStatus="verified",
                        scientificCapabilities=["covariate", "stratifier"],
                        description="age covariate",
                    ),
                ],
            ),
            SchemaEntitySemantics(
                entityId="abundance",
                entityName="abundance",
                sourceTable="abundance",
                fields=[
                    SchemaFieldSemantics(
                        fieldId="abundance.feature",
                        name="feature",
                        dataType="string",
                        nullable=True,
                        groupable=True,
                        displayable=True,
                        semanticStatus="verified",
                        scientificCapabilities=["dimension", "stratifier"],
                        description="feature",
                    ),
                    SchemaFieldSemantics(
                        fieldId="abundance.value",
                        name="value",
                        dataType="number",
                        nullable=True,
                        aggregatable=True,
                        displayable=True,
                        semanticStatus="verified",
                        scientificCapabilities=["outcome"],
                        description="numeric abundance outcome",
                    ),
                ],
            ),
        ],
        queryRules=[
            "select_or_with_only",
            "explicit_columns_only",
            "no_cross_database_reference",
            "bounded_limit_required",
            "java_final_validation",
        ],
    )


def _state(state_name: str) -> ScientificDecisionState:
    has_data = state_name != "s0"
    comparison_done = state_name == "s2"
    actions = (
        ["inspect_cohort", "execute_read_query"]
        if not has_data
        else [
            "inspect_cohort",
            "execute_read_query",
            "compare_groups",
            "analyze_projection",
            "stratified_analysis",
            "adjust_confounders",
            "cross_disease_validate",
        ]
    )
    return ScientificDecisionState.model_validate({
        "task": {
            "query": "compare disease groups",
            "objectives": [
                "group_comparison",
                "confounder_assessment",
            ],
            "constraints": {
                "disease_groups": ["T2D", "Healthy"],
                "focus_covariates": ["sample.age"],
            },
        },
        "data_state": {
            "has_tabular_data": has_data,
            "row_count": 100 if has_data else 0,
            "sample_count": 10 if has_data else None,
            "feature_count": 2 if has_data else None,
            "available_dimensions": (
                ["sample.disease", "sample.age", "abundance.feature"]
                if has_data else []
            ),
            "available_outcomes": ["abundance.value"] if has_data else [],
            "group_state": {
                "group_field": "sample.disease" if has_data else None,
                "group_count": 2 if has_data else 0,
                "group_sizes": {"T2D": 5, "Healthy": 5} if has_data else {},
            },
            "covariate_state": {
                "available_covariates": ["sample.age"] if has_data else [],
                "imbalance": {},
            },
        },
        "analysis_state": {
            "group_comparison": {
                "status": "completed" if comparison_done else "not_started",
            },
        },
        "action_space": {"available_actions": actions},
    })


def _write_states(directory, catalog: SchemaSemanticCatalog) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for state_name in ("s0", "s1", "s2"):
        (directory / f"state_{state_name}.json").write_text(
            json.dumps(_state(state_name).model_dump(mode="json"), ensure_ascii=False),
            encoding="utf-8",
        )
    (directory / "semantic_catalog.json").write_text(
        json.dumps(catalog.model_dump(mode="json"), ensure_ascii=False),
        encoding="utf-8",
    )


def test_cached_state_audit_recomputes_hard_actions(tmp_path) -> None:
    state_dir = tmp_path / "states"
    _write_states(state_dir, _catalog())

    audit = audit_cached_states(
        state_dir,
        catalog_path=state_dir / "semantic_catalog.json",
    )

    assert audit["audit_pass"] is True
    assert audit["all_schema_valid"] is True
    assert audit["availability_confirmed"] is True
    assert audit["policy_boundary_has_no_raw_rows"] is True
    assert all(item["training_eligible"] is False for item in audit["states"])
    assert audit["states"][0]["available_actions"] == [
        "inspect_cohort", "execute_read_query",
    ]


def test_audit_only_is_safe_default_and_marks_local_readiness(tmp_path) -> None:
    state_dir = tmp_path / "states"
    _write_states(state_dir, _catalog())
    report = run_policy_smoke(
        tmp_path / "out",
        state_dir=state_dir,
        catalog_path=state_dir / "semantic_catalog.json",
    )

    assert report["status"] == "AUDIT_ONLY_READY"
    assert report["READY_TO_START_A100"] is True
    assert report["transport_attempts"] == 0
    assert not list((tmp_path / "out").glob("qwen_smoke_*_request.json"))
    readiness = json.loads((tmp_path / "out" / "a100_readiness.json").read_text())
    assert readiness["a100_started"] is False
    assert readiness["remote_qwen_contacted"] is False


def test_live_smoke_uses_production_provider_and_saves_three_requests(tmp_path) -> None:
    state_dir = tmp_path / "states"
    _write_states(state_dir, _catalog())
    selected = {
        "s0": "execute_read_query",
        "s1": "compare_groups",
        "s2": "adjust_confounders",
    }
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        state_payload = json.loads(body["messages"][1]["content"])
        state_name = "s0" if not state_payload["state"]["data_state"]["has_tabular_data"] else (
            "s2" if state_payload["state"]["analysis_state"]["group_comparison"]["status"] == "completed" else "s1"
        )
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps({
                "selected_action": selected[state_name],
                "decision_reason": f"policy decision for {state_name}",
                "alternative_actions": [],
                "stop_reason": None,
            })}}],
        })

    transport = RecordingTransport(httpx.MockTransport(handler))
    report = run_policy_smoke(
        tmp_path / "out",
        state_dir=state_dir,
        catalog_path=state_dir / "semantic_catalog.json",
        base_url="http://qwen.local:19002",
        model="qwen3-8b-decision-base",
        live=True,
        transport=transport,
    )

    assert report["status"] == "PASS"
    assert report["transport_attempts"] == 3
    assert report["model_responses"] == 3
    assert report["repair_count"] == 0
    assert report["first_model_response_valid"] is True
    assert [item["selected_action"] for item in report["states"]] == [
        "execute_read_query", "compare_groups", "adjust_confounders",
    ]
    assert all(item["policy_origin"] == "qwen_model" for item in report["states"])
    assert all(item["deterministic_policy_fallback_used"] is False for item in report["states"])
    assert len(seen) == 3
    for state_name in ("s0", "s1", "s2"):
        request = json.loads((tmp_path / "out" / f"qwen_smoke_{state_name}_request.json").read_text())
        assert request["attempts"]
        assert "rows" not in json.dumps(request, ensure_ascii=False)
        decision = json.loads((tmp_path / "out" / f"qwen_smoke_{state_name}_decision.json").read_text())
        assert decision["parsed_decision"]["selected_action"] == selected[state_name]


def test_unavailable_action_fails_closed_without_policy_fallback(tmp_path) -> None:
    state_dir = tmp_path / "states"
    _write_states(state_dir, _catalog())

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps({
                "selected_action": "cross_project_validate",
                "decision_reason": "intentionally unavailable",
                "alternative_actions": [],
                "stop_reason": None,
            })}}],
        })

    report = run_policy_smoke(
        tmp_path / "out",
        state_dir=state_dir,
        catalog_path=state_dir / "semantic_catalog.json",
        live=True,
        transport=RecordingTransport(httpx.MockTransport(handler)),
    )

    assert report["status"] == "FAILED"
    assert report["transport_attempts"] == 2
    assert report["repair_count"] == 1
    assert report["deterministic_policy_fallback_used"] is False
    assert report["states"][0]["provider_error"].startswith("RuntimeError: POLICY_ACTION_NOT_AVAILABLE")
    assert report["states"][0]["available_action_compliance"] is False
