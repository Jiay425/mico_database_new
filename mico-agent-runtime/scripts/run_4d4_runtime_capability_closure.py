"""Run the offline 4D-4 runtime capability closure.

This is deliberately a local, deterministic gate.  It exercises the hard
availability boundary, generic AnalysisPlan validation, numeric typed
stratification, and completion semantics.  The optional knowledge probe is
read-only and never requests an embedding.  No LLM, Java business query, or
A100 service is started.
"""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pydantic import ValidationError

from mico_agent_runtime.contracts.decision_state import (
    ScientificAnalysisState,
    ScientificDataState,
    ScientificDecisionState,
    ScientificTaskState,
)
from mico_agent_runtime.contracts.materialization import AnalysisPlan
from mico_agent_runtime.contracts.schema_catalog import (
    SchemaEntitySemantics,
    SchemaFieldSemantics,
    SchemaSemanticCatalog,
)
from mico_agent_runtime.graph.generated_analysis import (
    GeneratedAnalysisError,
    execute_typed_analysis,
)
from mico_agent_runtime.knowledge.health import probe_knowledge_backend
from mico_agent_runtime.runtime.action_availability import (
    ActionAvailabilityContext,
    evaluate_action_availability,
)
from mico_agent_runtime.runtime.analysis_capability_registry import (
    AnalysisCapabilityContext,
    match_analysis_capability,
)
from mico_agent_runtime.runtime.completion_semantics import assess_scientific_completion
from mico_agent_runtime.runtime.objective_resolution import resolve_objective_lifecycle


OBSERVATION = "observation-" + "d" * 32


def _catalog() -> SchemaSemanticCatalog:
    now = datetime(2026, 9, 3, tzinfo=timezone.utc)
    return SchemaSemanticCatalog(
        schemaVersion="schema-catalog-v1",
        source="java_schema_contract",
        generatedAt=now,
        entities=[
            SchemaEntitySemantics(
                entityName="sample",
                sourceTable="sample",
                fields=[
                    SchemaFieldSemantics(
                        fieldId="sample.disease",
                        name="disease",
                        dataType="string",
                        nullable=False,
                        semanticStatus="verified",
                        groupable=True,
                        scientificCapabilities=["dimension"],
                        description="Disease group",
                    ),
                    SchemaFieldSemantics(
                        fieldId="sample.age",
                        name="age",
                        dataType="integer",
                        nullable=True,
                        groupable=True,
                        semanticStatus="verified",
                        scientificCapabilities=["covariate", "stratifier"],
                        description="Numeric age stratifier",
                    ),
                ],
            ),
            SchemaEntitySemantics(
                entityName="metadata",
                sourceTable="metadata",
                fields=[
                    SchemaFieldSemantics(
                        fieldId="metadata.project",
                        name="project",
                        dataType="string",
                        nullable=False,
                        semanticStatus="verified",
                        groupable=True,
                        scientificCapabilities=["dimension"],
                        description="Independent project dimension",
                    ),
                ],
            ),
            SchemaEntitySemantics(
                entityName="abundance",
                sourceTable="abundance",
                fields=[
                    SchemaFieldSemantics(
                        fieldId="abundance.feature",
                        name="feature",
                        dataType="string",
                        nullable=False,
                        semanticStatus="verified",
                        groupable=True,
                        scientificCapabilities=["dimension"],
                        description="Feature dimension",
                    ),
                    SchemaFieldSemantics(
                        fieldId="abundance.value",
                        name="value",
                        dataType="number",
                        nullable=True,
                        aggregatable=True,
                        semanticStatus="verified",
                        scientificCapabilities=["outcome"],
                        description="Numeric abundance outcome",
                    ),
                ],
            ),
        ],
        queryRules=[
            "select_or_with_only",
            "explicit_columns_only",
            "bounded_limit_required",
            "java_final_validation",
        ],
    )


def _state(*, include_project: bool) -> ScientificDecisionState:
    dimensions = ["sample.disease", "sample.age", "abundance.feature"]
    if include_project:
        dimensions.append("metadata.project")
    return ScientificDecisionState(
        task=ScientificTaskState(query="4D-4 capability closure", objectives=[]),
        data_state=ScientificDataState(
            has_tabular_data=True,
            row_count=12,
            sample_count=12,
            available_dimensions=dimensions,
            available_outcomes=["abundance.value"],
            group_state={
                "group_field": "sample.disease",
                "group_count": 2,
                "group_sizes": {"A": 6, "B": 6},
            },
            project_state={
                "has_project_field": include_project,
                "project_count": 3 if include_project else 0,
            },
            covariate_state={
                "available_covariates": ["sample.age"],
                "imbalance": {"sample.age": "unknown"},
            },
        ),
        analysis_state=ScientificAnalysisState(),
    )


def _numeric_plan() -> AnalysisPlan:
    return AnalysisPlan(
        analysis_type="stratified_comparison",
        source_observation_ids=[OBSERVATION],
        outcome="abundance.value",
        feature_field="abundance.feature",
        group_field="sample.disease",
        stratify_by=["sample.age"],
        numeric_stratification={
            "stratifier": "sample.age",
            "strategy": "quantile",
            "bin_count": 2,
            "cut_points": [],
            "min_samples_per_group": 2,
            "missing_value_policy": "drop",
            "multiple_testing": "benjamini_hochberg",
        },
        metrics=["effect_size", "p_value"],
    )


def _fixture_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, age in enumerate([10, 20, 30, 40, 50, 60], start=1):
        rows.extend([
            {
                "a_analysis_sample_key": f"a{index}",
                "a_abundance_feature": "F1",
                "a_sample_disease": "A",
                "a_sample_age": age,
                "a_abundance_value": float(age),
            },
            {
                "a_analysis_sample_key": f"b{index}",
                "a_abundance_feature": "F1",
                "a_sample_disease": "B",
                "a_sample_age": age,
                "a_abundance_value": float(age + 2),
            },
        ])
    return rows


def _blocked_objective_state() -> ScientificDecisionState:
    """A real-shaped state with no project capability after the read attempt."""

    return ScientificDecisionState(
        task=ScientificTaskState(
            query="validate the result across projects",
            objectives=["cross_project_validation", "evidence_support"],
        ),
        data_state=ScientificDataState(
            has_tabular_data=True,
            row_count=100,
            available_dimensions=["sample.disease", "abundance.feature"],
            available_outcomes=["abundance.value"],
            group_state={
                "group_field": "sample.disease",
                "group_count": 2,
                "group_sizes": {"A": 50, "B": 50},
            },
            project_state={"has_project_field": False, "project_count": 0},
            covariate_state={"available_covariates": ["sample.age"]},
        ),
        analysis_state=ScientificAnalysisState(
            group_comparison={"status": "completed"},
            confounder_adjustment={"status": "completed"},
        ),
    )


def _blocked_validation_state() -> ScientificDecisionState:
    """A validation objective with no independent validation dimension."""

    state = _blocked_objective_state().model_copy(deep=True)
    state.task.objectives = ["cross_disease_validation"]
    return state


def _run_checks() -> dict[str, Any]:
    catalog = _catalog()
    expected_decision_blocks = {
        "task",
        "data_state",
        "analysis_state",
        "evidence_state",
        "progress",
        "action_space",
    }
    decision_state_schema_changed = (
        set(_state(include_project=False).model_dump(mode="json"))
        != expected_decision_blocks
    )
    capability_context = AnalysisCapabilityContext(
        available_observation_ids=[OBSERVATION],
        available_fields=[
            "sample.disease", "sample.age", "abundance.feature", "abundance.value",
        ],
        observation_fields={OBSERVATION: [
            "sample.disease", "sample.age", "abundance.feature", "abundance.value",
        ]},
        distinct_counts={"sample.disease": 2, "sample.age": 12},
    )
    no_project = evaluate_action_availability(
        _state(include_project=False),
        ActionAvailabilityContext(catalog=catalog),
    )
    task_a = (
        "cross_project_validate" not in no_project.available_actions
        and no_project.blocked_reasons.get("cross_project_validate")
        == "PROJECT_DIMENSION_NOT_IN_OBSERVATION"
    )

    same_field_rejected = False
    try:
        AnalysisPlan(
            analysis_type="cross_project_validation",
            source_observation_ids=[OBSERVATION],
            outcome="abundance.value",
            group_field="sample.disease",
            validation_field="sample.disease",
            metrics=["effect_size"],
        )
    except ValidationError:
        same_field_rejected = True
    non_capable = AnalysisPlan(
        analysis_type="cross_project_validation",
        source_observation_ids=[OBSERVATION],
        outcome="abundance.value",
        group_field="sample.disease",
        validation_field="sample.age",
        metrics=["effect_size"],
    )
    non_capable_match = match_analysis_capability(
        non_capable, catalog, capability_context
    )
    valid_cross = AnalysisPlan(
        analysis_type="cross_project_validation",
        source_observation_ids=[OBSERVATION],
        outcome="abundance.value",
        group_field="sample.disease",
        validation_field="metadata.project",
        metrics=["effect_size"],
    )
    valid_cross_match = match_analysis_capability(
        valid_cross,
        catalog,
        AnalysisCapabilityContext(
            available_observation_ids=capability_context.available_observation_ids,
            available_fields=[*capability_context.available_fields, "metadata.project"],
            observation_fields={
                OBSERVATION: [
                    *capability_context.observation_fields[OBSERVATION],
                    "metadata.project",
                ]
            },
            distinct_counts={**capability_context.distinct_counts, "metadata.project": 3},
        ),
    )
    task_d = bool(
        same_field_rejected
        and non_capable_match.capability_code == "VALIDATION_DIMENSION_REQUIRED"
        and valid_cross_match.mode == "SUPPORTED_TYPED"
    )

    boundary_state = _blocked_objective_state()
    boundary_preliminary = evaluate_action_availability(
        boundary_state,
        ActionAvailabilityContext(catalog=catalog, knowledge_available=True),
    )
    boundary_resolution = resolve_objective_lifecycle(
        boundary_state,
        catalog=catalog,
        availability_reasons=boundary_preliminary.blocked_reasons,
        observations=[
            type("AttemptedRead", (), {
                "queryPlanFields": [
                    "sample.disease", "metadata.project", "abundance.value",
                ],
                "queryPlanRelationPath": ["sample_to_metadata"],
            })(),
        ],
    )
    boundary_finish = evaluate_action_availability(
        boundary_state,
        ActionAvailabilityContext(
            catalog=catalog,
            knowledge_available=True,
            blocked_objectives=boundary_resolution.blocked_objectives,
        ),
    )
    objective_boundary = bool(
        boundary_resolution.blocked_objectives == ("cross_project_validation",)
        and boundary_resolution.active_objectives == ("evidence_support",)
        and boundary_resolution.all_requested_objectives_completed is False
        and "finish" not in boundary_finish.available_actions
    )
    # Once the remaining evidence objective is resolved, the blocked project
    # objective must no longer prevent a legal finish.
    boundary_completed = boundary_state.model_copy(deep=True)
    boundary_completed.evidence_state.status = "completed"
    completed_preliminary = evaluate_action_availability(
        boundary_completed,
        ActionAvailabilityContext(catalog=catalog, knowledge_available=True),
    )
    completed_resolution = resolve_objective_lifecycle(
        boundary_completed,
        catalog=catalog,
        availability_reasons=completed_preliminary.blocked_reasons,
        observations=[
            type("AttemptedRead", (), {
                "queryPlanFields": [
                    "sample.disease", "metadata.project", "abundance.value",
                ],
                "queryPlanRelationPath": ["sample_to_metadata"],
            })(),
        ],
    )
    completed_finish = evaluate_action_availability(
        boundary_completed,
        ActionAvailabilityContext(
            catalog=catalog,
            knowledge_available=True,
            blocked_objectives=completed_resolution.blocked_objectives,
        ),
    )
    task_a_can_finish = bool(
        completed_resolution.blocked_objectives == ("cross_project_validation",)
        and completed_resolution.all_requested_objectives_completed is False
        and "finish" in completed_finish.available_actions
    )
    task_d_state = _blocked_validation_state()
    task_d_preliminary = evaluate_action_availability(
        task_d_state,
        ActionAvailabilityContext(catalog=catalog, knowledge_available=True),
    )
    task_d_resolution = resolve_objective_lifecycle(
        task_d_state,
        catalog=catalog,
        availability_reasons=task_d_preliminary.blocked_reasons,
    )
    task_d_completed = task_d_state.model_copy(deep=True)
    task_d_completed.evidence_state.status = "completed"
    task_d_completed_resolution = resolve_objective_lifecycle(
        task_d_completed,
        catalog=catalog,
        availability_reasons=task_d_preliminary.blocked_reasons,
    )
    task_d_finish = evaluate_action_availability(
        task_d_completed,
        ActionAvailabilityContext(
            catalog=catalog,
            knowledge_available=True,
            blocked_objectives=task_d_completed_resolution.blocked_objectives,
        ),
    )
    task_d_boundary = bool(
        task_d_resolution.blocked_objectives == ("cross_disease_validation",)
        and task_d_resolution.resolutions[0].reason_code == "VALIDATION_DIMENSION_UNAVAILABLE"
        and "cross_disease_validate" not in task_d_preliminary.available_actions
        and "finish" in task_d_finish.available_actions
    )

    numeric_plan = _numeric_plan()
    numeric_match = match_analysis_capability(
        numeric_plan, catalog, capability_context
    )
    result = execute_typed_analysis(numeric_plan, _fixture_rows(), 12)
    numeric_typed = bool(
        numeric_match.mode == "SUPPORTED_TYPED"
        and numeric_match.capability_code == "TYPED_NUMERIC_STRATIFIED_COMPARISON"
        and result.execution_mode == "typed"
        and result.stratum_results
        and all(item.q_value is not None for item in result.stratum_results)
        and result.metrics.get("insufficient_strata") == 0.0
    )
    insufficient = False
    try:
        execute_typed_analysis(
            numeric_plan.model_copy(update={"feature_field": None}),
            [
                {"a_sample_disease": "A", "a_sample_age": 10, "a_abundance_value": 1.0},
                {"a_sample_disease": "B", "a_sample_age": 20, "a_abundance_value": 2.0},
            ],
            2,
        )
    except GeneratedAnalysisError as exc:
        insufficient = exc.code == "ANALYSIS_TYPED_NUMERIC_STRATIFICATION_INSUFFICIENT_SAMPLE"

    completion = assess_scientific_completion(
        workflow_completed=True,
        analysis_result=result.model_dump(mode="json"),
    )
    completion_fixed = bool(
        completion.workflow_completed
        and completion.analysis_execution_completed
        and completion.scientific_result_valid
        and completion.scientific_conclusion_eligible
    )
    return {
        "task_a_capability_boundary_fixed": task_a,
        "task_d_analysis_plan_contract_fixed": task_d,
        "objective_boundary_closure": objective_boundary,
        "blocked_objective_semantics": bool(
            boundary_resolution.blocked_objectives == ("cross_project_validation",)
            and boundary_resolution.resolutions[0].status == "blocked"
            and boundary_resolution.resolutions[0].reason_code == "PROJECT_DIMENSION_UNAVAILABLE"
        ),
        "task_a_can_finish_with_limitation": task_a_can_finish,
        "task_d_can_finish_or_fail_closed_with_limitation": task_d_boundary,
        "decision_state_schema_changed": decision_state_schema_changed,
        "objective_boundary_snapshot": {
            "blocked": boundary_resolution.model_dump(),
            "finish_before_other_objectives": "finish" in boundary_finish.available_actions,
            "finish_after_other_objectives": "finish" in completed_finish.available_actions,
        },
        "numeric_stratifier_contract_ready": numeric_match.mode == "SUPPORTED_TYPED",
        "numeric_stratifier_materializer_contract_ready": bool(
            numeric_match.mode == "SUPPORTED_TYPED"
            and "numeric_stratification" in (
                Path(REPO_ROOT / "mico_agent_runtime" / "ports" / "research_planner.py")
                .read_text(encoding="utf-8")
            )
            and "min_samples_per_group" in (
                Path(REPO_ROOT / "mico_agent_runtime" / "ports" / "research_planner.py")
                .read_text(encoding="utf-8")
            )
        ),
        "stratified_typed_route_fixed": numeric_typed and insufficient,
        "numeric_fixture_result": result.model_dump(mode="json"),
        "numeric_insufficient_sample_fail_closed": insufficient,
        "completion_semantics_fixed": completion_fixed,
        "history_preserved": True,
        "policy_or_model_calls": 0,
        "a100_started": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "4d4_runtime_capability_closure" / "closure_report.json",
    )
    parser.add_argument(
        "--skip-knowledge-connectivity",
        action="store_true",
        help="skip read-only PostgreSQL/Neo4j checks while retaining wiring/assets checks",
    )
    args = parser.parse_args()
    checks = _run_checks()
    knowledge = probe_knowledge_backend(
        probe_connectivity=not args.skip_knowledge_connectivity,
    )
    flags = {
        "TASK_A_CAPABILITY_BOUNDARY_FIXED": checks["task_a_capability_boundary_fixed"],
        "TASK_D_ANALYSIS_PLAN_CONTRACT_FIXED": checks["task_d_analysis_plan_contract_fixed"],
        "OBJECTIVE_BOUNDARY_CLOSURE": checks["objective_boundary_closure"],
        "BLOCKED_OBJECTIVE_SEMANTICS": checks["blocked_objective_semantics"],
        "TASK_A_CAN_FINISH_WITH_LIMITATION": checks["task_a_can_finish_with_limitation"],
        "TASK_D_CAN_FINISH_OR_FAIL_CLOSED_WITH_LIMITATION": checks[
            "task_d_can_finish_or_fail_closed_with_limitation"
        ],
        "DECISION_STATE_SCHEMA_CHANGED": checks["decision_state_schema_changed"],
        "NUMERIC_STRATIFIER_CONTRACT_READY": checks["numeric_stratifier_contract_ready"],
        "NUMERIC_STRATIFIER_MATERIALIZER_CONTRACT_READY": checks[
            "numeric_stratifier_materializer_contract_ready"
        ],
        "STRATIFIED_TYPED_ROUTE_FIXED": checks["stratified_typed_route_fixed"],
        "KNOWLEDGE_WIRING_PASS": knowledge["knowledge_wiring_pass"],
        "KNOWLEDGE_CONNECTIVITY_PASS": knowledge["knowledge_connectivity_pass"],
        "KNOWLEDGE_RETRIEVAL_PROBE_PASS": knowledge["knowledge_retrieval_probe_pass"],
        "COMPLETION_SEMANTICS_FIXED": checks["completion_semantics_fixed"],
        "LOCAL_RUNTIME_FIX_READY": all([
            checks["task_a_capability_boundary_fixed"],
            checks["task_d_analysis_plan_contract_fixed"],
            checks["objective_boundary_closure"],
            checks["blocked_objective_semantics"],
            checks["task_a_can_finish_with_limitation"],
            checks["task_d_can_finish_or_fail_closed_with_limitation"],
            not checks["decision_state_schema_changed"],
            checks["stratified_typed_route_fixed"],
            checks["numeric_stratifier_materializer_contract_ready"],
            knowledge["knowledge_wiring_pass"],
            checks["completion_semantics_fixed"],
        ]),
        "READY_FOR_E2E_RERUN": all([
            checks["task_a_capability_boundary_fixed"],
            checks["task_d_analysis_plan_contract_fixed"],
            checks["objective_boundary_closure"],
            checks["blocked_objective_semantics"],
            checks["task_a_can_finish_with_limitation"],
            checks["task_d_can_finish_or_fail_closed_with_limitation"],
            not checks["decision_state_schema_changed"],
            checks["stratified_typed_route_fixed"],
            checks["numeric_stratifier_materializer_contract_ready"],
            knowledge["knowledge_wiring_pass"],
            knowledge["knowledge_connectivity_pass"],
            checks["completion_semantics_fixed"],
        ]),
    }
    report = {
        "report_version": "4d4-runtime-capability-closure-v1",
        "provider_calls": {"gemini": 0, "qwen": 0, "deepseek": 0},
        "runtime_checks": checks,
        "knowledge": knowledge,
        "flags": flags,
        "historical_trace_policy": "preserved; closure is a derived report and does not rewrite prior E2E artifacts",
        "next_step": "Review flags before authorizing a short E2E rerun; no A100 was started by this command.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(flags, ensure_ascii=False, indent=2))
    print(f"wrote {args.output}")
    return 0 if flags["READY_FOR_E2E_RERUN"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
