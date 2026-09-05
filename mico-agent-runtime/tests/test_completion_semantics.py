from mico_agent_runtime.runtime.completion_semantics import assess_scientific_completion


def test_completion_semantics_distinguish_workflow_from_analysis_and_conclusion() -> None:
    assessment = assess_scientific_completion(
        workflow_completed=True,
        analysis_result={
            "status": "COMPLETED",
            "analysisType": "stratified_comparison",
            "execution_mode": "generated",
            "metrics": {"stratum_count": 2.0},
            "stratum_results": [],
            "limitations": ["generated_code_was_sandbox_validated"],
        },
    )
    assert assessment.workflow_completed is True
    assert assessment.analysis_execution_completed is True
    assert assessment.scientific_result_valid is False
    assert assessment.scientific_conclusion_eligible is False
    assert "STRATIFIED_RESULT_EMPTY" in assessment.reason_codes


def test_typed_nonempty_analysis_can_be_conclusion_eligible() -> None:
    assessment = assess_scientific_completion(
        workflow_completed=True,
        analysis_result={
            "status": "COMPLETED",
            "analysisType": "stratified_comparison",
            "execution_mode": "typed",
            "metrics": {"stratum_count": 2.0},
            "stratum_results": [
                {"stratum": "bin_1", "group_a_n": 3, "group_b_n": 3},
            ],
            "limitations": ["typed_plan_executed_by_approved_operator"],
        },
    )
    assert assessment.scientific_result_valid is True
    assert assessment.scientific_conclusion_eligible is True


def test_missing_analysis_result_is_not_workflow_failure_only() -> None:
    assessment = assess_scientific_completion(
        workflow_completed=False,
        analysis_result=None,
    )
    assert assessment.workflow_completed is False
    assert assessment.analysis_execution_completed is False
    assert assessment.scientific_result_valid is False
    assert assessment.scientific_conclusion_eligible is False
    assert "WORKFLOW_NOT_COMPLETED" in assessment.reason_codes
    assert "ANALYSIS_RESULT_MISSING" in assessment.reason_codes
