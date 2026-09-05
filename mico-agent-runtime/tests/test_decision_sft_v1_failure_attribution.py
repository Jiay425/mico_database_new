from __future__ import annotations

import pytest

from mico_agent_runtime.contracts.materialization import AnalysisPlan
from mico_agent_runtime.runtime.analysis_capability_registry import (
    AnalysisCapabilityContext,
    match_analysis_capability,
)
from scripts.audit_decision_sft_v1_full_dynamic import (
    DEFAULT_INPUT_DIR,
    build_report,
)


def test_existing_e2e_failures_are_attributed_without_provider_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Make the wiring assertion deterministic without relying on a developer's
    # shell environment or opening either knowledge store.
    for name, value in {
        "MICO_KNOWLEDGE_RETRIEVAL_BACKEND": "database",
        "MICO_KNOWLEDGE_VECTOR_ENABLED": "true",
        "MICO_KNOWLEDGE_GRAPH_ENABLED": "true",
        "MICO_KNOWLEDGE_VECTOR_DATABASE_URL": "postgresql://test_user@127.0.0.1:55432/mico_knowledge",
        "MICO_KNOWLEDGE_NEO4J_URI": "bolt://127.0.0.1:57687",
        "MICO_KNOWLEDGE_NEO4J_USER": "neo4j",
        "MICO_KNOWLEDGE_NEO4J_PASSWORD": "test-only-password",
        "MICO_LOCAL_KNOWLEDGE_INDEX_DIR": "references/knowledge/medical/rag",
    }.items():
        monkeypatch.setenv(name, value)
    audit = build_report(DEFAULT_INPUT_DIR)

    assert audit["external_calls_made"] is False
    assert audit["model_calls_made"] == 0
    assert audit["layered_assessment"]["policy"]["assessment"] == "PASS"
    assert audit["layered_assessment"]["policy"]["request_count"] == 15
    assert audit["layered_assessment"]["tool_data"]["assessment"] == "PASS"
    assert audit["flags"] == {
        "TASK_A_ROOT_CAUSE": "DATA_CAPABILITY_MISSING",
        "TASK_D_ROOT_CAUSE": "ANALYSIS_PLAN_SCHEMA",
        "STRATIFIED_TYPED_ROUTE_FIXED": False,
        "KNOWLEDGE_BACKEND_WIRED": True,
        "LOCAL_RUNTIME_FIX_READY": True,
        "READY_FOR_E2E_RERUN": False,
    }

    a = audit["task_a"]
    assert a["capability_registry"]["capability_code"] == "VALIDATION_DIMENSION_REQUIRED"
    assert a["exact_rejection_or_error"]["trace_error_code"] == "ANALYSIS_CAPABILITY_UNSUPPORTED"
    assert a["materializer"]["first_validation"]["valid"] is False
    assert a["materializer"]["final_validation"]["valid"] is True
    assert a["materializer"]["repair_prompt_persisted"] is True

    d = audit["task_d"]
    assert d["root_cause"] == "ANALYSIS_PLAN_SCHEMA"
    assert d["materializer"]["first_validation"]["valid"] is False
    assert d["materializer"]["repair_prompt_persisted"] is False
    assert d["exact_rejection_or_error"]["trace_error_code"] == "ANALYSIS_GENERATION_FAILED"

    stratified = audit["layered_assessment"]["materialization_capability"]["stratified"]
    assert stratified["workflow_completed"] is True
    assert stratified["analysis_execution_completed"] is True
    assert stratified["scientific_result_valid"] is False
    assert stratified["scientific_conclusion_eligible"] is False
    assert stratified["capability_registry"]["capability_code"] == "GENERATED_STRATIFIED_COMPARISON"


def test_numeric_stratifier_uses_explicit_generated_compatibility_route() -> None:
    # Keep this test independent of the live E2E artifact.  It protects the
    # general registry contract that explains the stratified trace: numeric
    # strata are deliberately sent through the existing bounded generated
    # channel, while categorical strata remain eligible for typed execution.
    from tests.test_analysis_capability_registry import _catalog

    observation = "observation-" + "b" * 32
    plan = AnalysisPlan(
        analysis_type="stratified_comparison",
        source_observation_ids=[observation],
        outcome="abundance.value",
        group_field="sample.disease",
        stratify_by=["sample.age"],
        metrics=["effect_size"],
    )
    match = match_analysis_capability(
        plan,
        _catalog(),
        AnalysisCapabilityContext(
            available_observation_ids=[observation],
            available_fields=["sample.disease", "sample.age", "abundance.value"],
            observation_fields={
                observation: ["sample.disease", "sample.age", "abundance.value"]
            },
            distinct_counts={"sample.disease": 2},
        ),
    )
    assert match.mode == "SUPPORTED_GENERATED"
    assert match.capability_code == "GENERATED_STRATIFIED_COMPARISON"
    assert match.reason_code == "NON_CATEGORICAL_STRATIFIER_REQUIRES_GENERATED_EXECUTION"


def test_runner_uses_canonical_knowledge_backend_key(monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.run_decision_sft_v1_full_dynamic_e2e as runner

    sentinel = object()
    monkeypatch.setenv("MICO_KNOWLEDGE_RETRIEVAL_BACKEND", "database")
    monkeypatch.setenv("MICO_KNOWLEDGE_BACKEND", "local")
    monkeypatch.setenv("MICO_LOCAL_KNOWLEDGE_ENABLED", "false")
    monkeypatch.setattr(
        runner.DatabaseKnowledgeSearchPort,
        "from_environment",
        classmethod(lambda cls, env=None: sentinel),
    )

    port, origin = runner._build_optional_knowledge()
    assert port is sentinel
    assert origin == "database"
