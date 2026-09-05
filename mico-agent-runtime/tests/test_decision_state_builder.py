from __future__ import annotations

from datetime import datetime, timezone

from mico_agent_runtime.contracts.analysis import AnalysisEvidence, AnalysisPlan, AnalysisResult
from mico_agent_runtime.contracts.decision_state import ScientificDecisionState, ScientificTaskState
from mico_agent_runtime.contracts.evidence import LiteratureEvidenceItem
from mico_agent_runtime.contracts.materialization import QueryPlan
from mico_agent_runtime.contracts.research import Observation
from mico_agent_runtime.contracts.schema_catalog import (
    SchemaEntitySemantics,
    SchemaFieldSemantics,
    SchemaSemanticCatalog,
)
from mico_agent_runtime.runtime.decision_state_builder import (
    remaining_objectives,
    update_from_analysis_result,
    update_from_evidence_result,
    update_from_query_result,
    update_progress,
)


NOW = datetime(2026, 8, 30, tzinfo=timezone.utc)


def _catalog() -> SchemaSemanticCatalog:
    return SchemaSemanticCatalog(
        schemaVersion="schema-catalog-v1",
        source="java_schema_contract",
        generatedAt=NOW,
        entities=[
            SchemaEntitySemantics(
                entityId="sample",
                entityName="patient_record",
                sourceTable="patients",
                fields=[
                    SchemaFieldSemantics(
                        fieldId="sample.disease", name="disease", dataType="string",
                        nullable=True, semanticStatus="verified", filterable=True,
                        groupable=True, displayable=True,
                        scientificCapabilities=["dimension", "stratifier"],
                        description="Disease dimension",
                    ),
                    SchemaFieldSemantics(
                        fieldId="sample.age", name="age", dataType="integer",
                        nullable=True, semanticStatus="verified", filterable=True,
                        groupable=True, aggregatable=True, displayable=True,
                        scientificCapabilities=["covariate", "stratifier"],
                        description="Age covariate",
                    ),
                    SchemaFieldSemantics(
                        fieldId="sample.gender", name="gender", dataType="string",
                        nullable=True, semanticStatus="verified", filterable=True,
                        groupable=True, displayable=True, description="Gender covariate",
                        scientificCapabilities=["covariate", "dimension", "stratifier"],
                    ),
                ],
            ),
            SchemaEntitySemantics(
                entityId="metadata",
                entityName="sample_metadata",
                sourceTable="meta2db_sample_metadata",
                fields=[
                    SchemaFieldSemantics(
                        fieldId="metadata.project", name="project_name", dataType="string",
                        nullable=True, semanticStatus="verified", filterable=True,
                        groupable=True, displayable=True,
                        scientificCapabilities=["dimension"],
                        description="Project dimension",
                    )
                ],
            ),
        ],
        queryRules=["select_or_with_only", "explicit_columns_only", "bounded_limit_required"],
    )


def _observation(action: str = "execute_read_query", index: str = "1") -> Observation:
    return Observation(
        observationId="observation-" + index * 32,
        actionId="action-" + index * 32,
        actionName=action,
        status="VALIDATED",
        source="java_controlled_read",
        queryHash="sha256:" + "a" * 64,
        rowCount=4,
        generatedAt=NOW,
        schemaVersion="java-read-model-v1",
    )


def _state() -> ScientificDecisionState:
    return ScientificDecisionState(
        task=ScientificTaskState(
            query="compare disease groups",
            objectives=["group_comparison", "confounder_assessment", "evidence_support"],
        )
    )


def _payload() -> dict[str, object]:
    return {
        "columns": [
            "a_sample_disease", "a_sample_age", "a_sample_gender", "a_metadata_project",
        ],
        "rows": [
            {"a_sample_disease": "T2D", "a_sample_age": 60, "a_sample_gender": "F", "a_metadata_project": "P1"},
            {"a_sample_disease": "T2D", "a_sample_age": 62, "a_sample_gender": "M", "a_metadata_project": "P2"},
            {"a_sample_disease": "Healthy", "a_sample_age": 34, "a_sample_gender": "F", "a_metadata_project": "P1"},
            {"a_sample_disease": "Healthy", "a_sample_age": 36, "a_sample_gender": "M", "a_metadata_project": "P2"},
        ],
    }


def _read_plan() -> QueryPlan:
    return QueryPlan(
        root_entity="sample",
        select_fields=[
            "sample.disease", "sample.age", "sample.gender", "metadata.project",
        ],
        limit=20,
    )


def test_query_observation_projects_real_rows_and_semantic_catalog() -> None:
    state = update_from_query_result(
        _state(), _observation(), _payload(), query_plan=_read_plan(), catalog=_catalog(), row_count=4
    )

    assert state.data_state.has_tabular_data is True
    assert state.data_state.row_count == 4
    assert state.data_state.available_dimensions == [
        "sample.disease", "sample.age", "sample.gender", "metadata.project",
    ]
    assert state.data_state.available_outcomes == []
    assert state.data_state.group_state.group_field == "sample.disease"
    assert state.data_state.group_state.group_sizes == {"T2D": 2, "Healthy": 2}
    assert state.data_state.project_state.has_project_field is True
    assert state.data_state.project_state.project_count == 2
    assert state.data_state.covariate_state.available_covariates == ["sample.age", "sample.gender"]
    assert state.data_state.covariate_state.imbalance == {}


def test_numeric_covariate_is_not_outcome_but_catalog_outcome_is() -> None:
    catalog = _catalog().model_copy(deep=True)
    catalog.entities.append(SchemaEntitySemantics(
        entityId="abundance",
        entityName="standard_abundance",
        sourceTable="microbe_abundance_standard",
        fields=[SchemaFieldSemantics(
            fieldId="abundance.value",
            name="abundance_value",
            dataType="number",
            nullable=True,
            semanticStatus="verified",
            aggregatable=True,
            displayable=True,
            scientificCapabilities=["outcome"],
            description="Abundance outcome",
        )],
    ))
    payload = {
        "columns": ["a_sample_disease", "a_sample_age", "a_abundance_value"],
        "rows": [
            {"a_sample_disease": "T2D", "a_sample_age": 60, "a_abundance_value": 0.4},
            {"a_sample_disease": "Healthy", "a_sample_age": 35, "a_abundance_value": 0.1},
        ],
    }
    plan = QueryPlan(
        root_entity="sample",
        select_fields=["sample.disease", "sample.age", "abundance.value"],
        limit=20,
    )

    state = update_from_query_result(
        _state(), _observation(), payload, query_plan=plan, catalog=catalog, row_count=2
    )

    assert state.data_state.available_outcomes == ["abundance.value"]
    assert "sample.age" in state.data_state.available_dimensions
    assert "sample.age" in state.data_state.covariate_state.available_covariates
    assert "sample.age" not in state.data_state.available_outcomes


def test_dynamic_aggregate_is_not_sample_level_outcome() -> None:
    catalog = _catalog().model_copy(deep=True)
    catalog.entities.append(SchemaEntitySemantics(
        entityId="abundance",
        entityName="standard_abundance",
        sourceTable="microbe_abundance_standard",
        fields=[SchemaFieldSemantics(
            fieldId="abundance.value", name="abundance_value", dataType="number",
            nullable=True, semanticStatus="verified", aggregatable=True,
            displayable=True, scientificCapabilities=["outcome"],
            description="Abundance outcome",
        ), SchemaFieldSemantics(
            fieldId="abundance.feature", name="microbe_name_standard", dataType="string",
            nullable=True, semanticStatus="verified", groupable=True,
            displayable=True, scientificCapabilities=["dimension"],
            description="Feature dimension",
        )],
    ))
    plan = QueryPlan(
        root_entity="sample", relation_path=["sample_to_abundance"],
        select_fields=["sample.disease"],
        aggregations=[{"field": "abundance.value", "op": "mean"}],
        group_by=["sample.disease"], limit=10,
    )
    state = update_from_query_result(
        _state(), _observation(), {
            "columns": ["a_sample_disease", "a_abundance_value_mean"],
            "rows": [
                {"a_sample_disease": "T2D", "a_abundance_value_mean": 0.2},
                {"a_sample_disease": "Healthy", "a_abundance_value_mean": 0.1},
            ],
        }, query_plan=plan, catalog=catalog, row_count=2,
        require_sample_level_outcome=True,
    )
    assert state.data_state.available_outcomes == []
    assert state.data_state.sample_count is None
    assert state.data_state.group_state.group_count == 0


def test_dynamic_opaque_sample_key_populates_sample_and_feature_counts() -> None:
    catalog = _catalog().model_copy(deep=True)
    catalog.entities.append(SchemaEntitySemantics(
        entityId="abundance",
        entityName="standard_abundance",
        sourceTable="microbe_abundance_standard",
        fields=[SchemaFieldSemantics(
            fieldId="abundance.value", name="abundance_value", dataType="number",
            nullable=True, semanticStatus="verified", aggregatable=True,
            displayable=True, scientificCapabilities=["outcome"],
            description="Abundance outcome",
        ), SchemaFieldSemantics(
            fieldId="abundance.feature", name="microbe_name_standard", dataType="string",
            nullable=True, semanticStatus="verified", groupable=True,
            displayable=True, scientificCapabilities=["dimension"],
            description="Feature dimension",
        )],
    ))
    plan = QueryPlan(
        root_entity="sample", relation_path=["sample_to_abundance"],
        select_fields=["sample.disease", "abundance.feature", "abundance.value"],
        limit=10,
    )
    state = update_from_query_result(
        _state(), _observation(), {
            "columns": [
                "a_sample_disease", "a_abundance_feature", "a_abundance_value",
                "a_analysis_sample_key",
            ],
            "rows": [
                {"a_sample_disease": "T2D", "a_abundance_feature": "f1",
                 "a_abundance_value": 0.2, "a_analysis_sample_key": "s_a"},
                {"a_sample_disease": "T2D", "a_abundance_feature": "f2",
                 "a_abundance_value": 0.4, "a_analysis_sample_key": "s_a"},
                {"a_sample_disease": "Healthy", "a_abundance_feature": "f1",
                 "a_abundance_value": 0.1, "a_analysis_sample_key": "s_b"},
            ],
        }, query_plan=plan, catalog=catalog, row_count=3,
        require_sample_level_outcome=True,
    )
    assert state.data_state.available_outcomes == ["abundance.value"]
    assert state.data_state.sample_count == 2
    assert state.data_state.feature_count == 2


def test_query_row_count_is_authoritative_even_when_payload_length_differs() -> None:
    payload = _payload()
    payload["rows"] = payload["rows"][:2]
    state = update_from_query_result(
        _state(), _observation(), payload, query_plan=_read_plan(), catalog=_catalog(), row_count=7
    )
    assert state.data_state.row_count == 7
    assert state.data_state.group_state.group_sizes == {"T2D": 2}


def test_completed_tabular_query_with_zero_rows_still_exposes_tabular_data() -> None:
    observation = _observation()
    observation = observation.model_copy(update={"rowCount": 0})
    state = update_from_query_result(
        _state(),
        observation,
        {"columns": ["a_sample_disease"], "rows": []},
        query_plan=QueryPlan(
            root_entity="sample", select_fields=["sample.disease"], limit=20
        ),
        catalog=_catalog(),
        row_count=0,
    )
    assert state.data_state.has_tabular_data is True
    assert state.data_state.row_count == 0
    assert state.data_state.group_state.group_count == 0


def test_query_project_counts_distinguish_seven_one_and_null_values() -> None:
    rows = [
        {
            "a_sample_disease": "T2D" if index % 2 else "Healthy",
            "a_sample_age": 30 + index,
            "a_sample_gender": "F",
            "a_metadata_project": None if index == 0 else f"P{index}",
        }
        for index in range(8)
    ]
    payload = {"columns": _payload()["columns"], "rows": rows}
    state = update_from_query_result(
        _state(), _observation(), payload, query_plan=_read_plan(), catalog=_catalog(), row_count=8
    )
    assert state.data_state.project_state.has_project_field is True
    assert state.data_state.project_state.project_count == 7

    one_project_payload = {
        "columns": _payload()["columns"],
        "rows": [dict(row, a_metadata_project="P1") for row in rows],
    }
    one_project = update_from_query_result(
        _state(), _observation(), one_project_payload,
        query_plan=_read_plan(), catalog=_catalog(), row_count=8,
    )
    assert one_project.data_state.project_state.project_count == 1


def test_query_one_group_and_absent_project_are_not_promoted_to_two_groups() -> None:
    one_group_payload = {
        "columns": _payload()["columns"],
        "rows": [dict(row, a_sample_disease="T2D") for row in _payload()["rows"]],
    }
    one_group = update_from_query_result(
        _state(), _observation(), one_group_payload,
        query_plan=_read_plan(), catalog=_catalog(), row_count=4,
    )
    assert one_group.data_state.group_state.group_count == 1
    assert one_group.data_state.group_state.group_sizes == {"T2D": 4}

    absent_project_payload = {
        "columns": ["a_sample_disease", "a_sample_age", "a_sample_gender"],
        "rows": [
            {key: row[key] for key in ["a_sample_disease", "a_sample_age", "a_sample_gender"]}
            for row in _payload()["rows"]
        ],
    }
    absent_project = update_from_query_result(
        _state(), _observation(), absent_project_payload,
        query_plan=QueryPlan(
            root_entity="sample",
            select_fields=["sample.disease", "sample.age", "sample.gender"],
            limit=20,
        ),
        catalog=_catalog(), row_count=4,
    )
    assert absent_project.data_state.project_state.has_project_field is False
    assert absent_project.data_state.project_state.project_count == 0


def test_unknown_semantic_field_is_not_guessed_from_payload_name() -> None:
    state = update_from_query_result(
        _state(), _observation(),
        {"columns": ["a_sample_unknown"], "rows": [{"a_sample_unknown": "x"}]},
        query_plan=QueryPlan(root_entity="sample", select_fields=["sample.unknown"], limit=20),
        catalog=_catalog(), row_count=1,
    )
    assert state.data_state.available_dimensions == []
    assert state.data_state.available_outcomes == []


def _analysis_result(
    action: str,
    metrics: dict[str, float],
    *,
    status: str = "COMPLETED",
) -> AnalysisResult:
    observation_id = "observation-" + "1" * 32
    evidence = AnalysisEvidence(
        evidenceId="evidence-" + "1" * 32,
        analysisId="analysis-" + "1" * 32,
        observationId=observation_id,
        source="java_controlled_read",
        dataSnapshotId="transient-550e8400-e29b-41d4-a716-446655440000",
        snapshotPersistence="transient",
        queryHash="sha256:" + "a" * 64,
        schemaVersion="java-read-model-v1",
        rowCount=4,
        generatedAt=NOW,
    )
    return AnalysisResult(
        analysisId="analysis-" + "1" * 32,
        actionName=action,
        status=status,
        plannerMode="model",
        codeVersion="typed-analysis-operator-v1",
        sourceObservationIds=[observation_id],
        rowsAnalyzed=4,
        metrics=metrics,
        evidence=[evidence],
        limitations=["typed_plan_executed_by_approved_operator"],
    )


def test_analysis_mapping_keeps_unmappable_adjusted_metrics_unknown() -> None:
    state = update_from_analysis_result(
        _state(),
        _analysis_result("compare_groups", {"effect_size": 0.62, "p_value": 0.008}),
    )
    state = update_from_analysis_result(
        state,
        _analysis_result("adjust_confounders", {"effect_size": 0.18, "p_value": 0.21}),
        analysis_plan=AnalysisPlan(
            analysisId="analysis-" + "2" * 32,
            actionName="adjust_confounders",
            sourceObservationIds=["observation-" + "1" * 32],
            analysisGoal="adjust for age",
            confounders=["sample.age"],
        ),
    )
    assert state.analysis_state.group_comparison.effect_size == 0.62
    assert state.analysis_state.group_comparison.p_value == 0.008
    assert state.analysis_state.confounder_adjustment.adjusted_effect_size is None
    assert state.analysis_state.confounder_adjustment.adjusted_p_value is None
    assert state.analysis_state.confounder_adjustment.adjusted_covariates == ["sample.age"]


def test_stratified_and_cross_validation_mapping_uses_real_state_or_rows() -> None:
    base = update_from_query_result(
        _state(), _observation(), _payload(), query_plan=_read_plan(), catalog=_catalog(), row_count=4
    )
    stratified = update_from_analysis_result(
        base,
        _analysis_result("stratified_analysis", {"count": 4.0}),
        analysis_plan=AnalysisPlan(
            analysisId="analysis-" + "3" * 32,
            actionName="stratified_analysis",
            sourceObservationIds=["observation-" + "1" * 32],
            analysisGoal="stratify by age",
            dimensions=["sample.age"],
        ),
        source_payloads={"observation-" + "1" * 32: _payload()},
        catalog=_catalog(),
    )
    cross = update_from_analysis_result(
        stratified,
        _analysis_result("cross_project_validate", {"effect_size": 0.2}),
    )
    assert stratified.analysis_state.stratified_analysis.stratify_fields == ["sample.age"]
    assert stratified.analysis_state.stratified_analysis.stratum_count == 4
    assert stratified.analysis_state.stratified_analysis.heterogeneity == "unknown"
    assert cross.analysis_state.cross_project_validation.project_count == 2
    assert cross.analysis_state.cross_project_validation.heterogeneity == "unknown"


def test_projection_count_stays_zero_without_an_explicit_result_count_metric() -> None:
    state = update_from_analysis_result(
        _state(), _analysis_result("analyze_projection", {"count": 4.0})
    )
    assert state.analysis_state.projection_analysis.status == "completed"
    assert state.analysis_state.projection_analysis.result_count == 0


def test_analysis_failure_maps_to_failed_without_inventing_metrics() -> None:
    state = update_from_analysis_result(
        _state(), _analysis_result("compare_groups", {}, status="FAILED")
    )
    assert state.analysis_state.group_comparison.status == "failed"
    assert state.analysis_state.group_comparison.effect_size is None
    assert state.analysis_state.group_comparison.p_value is None


def test_cross_disease_validation_uses_real_disease_group_count() -> None:
    base = update_from_query_result(
        _state(), _observation(), _payload(), query_plan=_read_plan(), catalog=_catalog(), row_count=4
    )
    state = update_from_analysis_result(
        base, _analysis_result("cross_disease_validate", {"effect_size": 0.1})
    )
    assert state.analysis_state.cross_disease_validation.status == "completed"
    assert state.analysis_state.cross_disease_validation.disease_count == 2
    assert state.analysis_state.cross_disease_validation.heterogeneity == "unknown"


def _evidence(direction: str, suffix: str) -> LiteratureEvidenceItem:
    return LiteratureEvidenceItem(
        evidenceId="evidence-" + suffix * 32,
        source="internal_knowledge",
        externalId="doc-" + suffix,
        title="Evidence item",
        publicationYear=2025,
        direction=direction,
        summary="Bounded source summary.",
    )


def test_evidence_mapping_counts_directions_without_inventing_consistency() -> None:
    state = update_from_evidence_result(
        _state(),
        [
            _evidence("supporting", "1"), _evidence("supporting", "2"),
            _evidence("supporting", "3"), _evidence("supporting", "4"),
            _evidence("contrary", "5"), _evidence("context", "6"),
        ],
    )
    assert state.evidence_state.model_dump() == {
        "status": "completed",
        "evidence_count": 6,
        "support_count": 4,
        "conflict_count": 1,
        "context_count": 1,
        "consistency": "unknown",
    }


def test_empty_and_failed_retrieval_states_are_explicit() -> None:
    empty = update_from_evidence_result(_state(), [], observation_status="VALIDATED")
    assert empty.evidence_state.status == "insufficient"
    assert empty.evidence_state.evidence_count == 0

    failed = update_from_evidence_result(
        _state(), [_evidence("context", "4")], observation_status="FAILED"
    )
    assert failed.evidence_state.status == "failed"


def test_progress_and_remaining_objectives_are_deterministic() -> None:
    state = update_progress(_state(), "execute_read_query")
    state = update_progress(state, "compare_groups")
    assert state.progress.completed_actions == ["execute_read_query", "compare_groups"]
    assert state.progress.action_counts == {"execute_read_query": 1, "compare_groups": 1}
    assert state.progress.action_count == 2
    assert remaining_objectives(state) == [
        "group_comparison", "confounder_assessment", "evidence_support"
    ]


def test_partial_observation_counts_as_execution_but_not_completed_action() -> None:
    state = update_progress(_state(), "retrieve_evidence", completed=False)
    assert state.progress.action_counts == {"retrieve_evidence": 1}
    assert state.progress.action_count == 1
    assert state.progress.completed_actions == []


def test_progress_duplicate_action_is_counted_once_in_completed_actions() -> None:
    state = update_progress(_state(), "execute_read_query")
    state = update_progress(state, "execute_read_query")
    assert state.progress.completed_actions == ["execute_read_query"]
    assert state.progress.action_counts == {"execute_read_query": 2}
    assert state.progress.action_count == 2


def test_decision_state_has_no_policy_hint_or_answer_leakage_fields() -> None:
    serialized = _state().model_dump(mode="json")

    def keys(value: object) -> list[str]:
        if isinstance(value, dict):
            return list(value) + [
                nested for child in value.values() for nested in keys(child)
            ]
        if isinstance(value, list):
            return [nested for child in value for nested in keys(child)]
        return []

    forbidden = (
        "should_", "next_action", "recommended_action", "required_action", "_ready",
        "cross_project_required", "analysis_required", "evidence_required",
    )
    assert not any(any(marker in key for marker in forbidden) for key in keys(serialized))
