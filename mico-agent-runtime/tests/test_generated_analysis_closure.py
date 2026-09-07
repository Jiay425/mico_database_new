from __future__ import annotations

import pytest

from mico_agent_runtime.contracts.generated_analysis import (
    GeneratedAnalysisInputBindings,
    GeneratedAnalysisProgram,
)
from mico_agent_runtime.graph.generated_analysis import (
    GeneratedAnalysisError,
    execute_generated_program,
    generated_code_contract,
)
from scripts.run_generated_analysis_layer_b_probe import (
    GENERATOR_SYSTEM_PROMPT,
    GeneratorProviderError,
    _contract_repair_feedback,
    _code_skeleton,
    _cases,
    _run_case,
)


OBSERVATION_ID = "observation-" + "a" * 32


def _program(**updates: object) -> GeneratedAnalysisProgram:
    payload: dict[str, object] = {
        "action_name": "analyze_projection",
        "analysis_goal": "summarize a bounded numeric projection",
        "input_observation_ids": [OBSERVATION_ID],
        # Opaque sample keys are allowed; raw identifiers are never included.
        "required_columns": ["analysis_sample_key", "value"],
        "code": (
            'values = [float(row["value"]) for row in rows if row["value"] is not None]\n'
            'result = {"metrics": {"count": float(len(values)), "mean": sum(values) / len(values)}, '
            '"used_row_count": len(values)}'
        ),
        "expected_outputs": ["metrics"],
        "timeout_seconds": 2,
    }
    payload.update(updates)
    return GeneratedAnalysisProgram.model_validate(payload)


def _rows() -> list[dict[str, object]]:
    return [
        {"analysis_sample_key": "opaque-a", "value": 1.0, "raw_patient_id": "not-bound"},
        {"analysis_sample_key": "opaque-b", "value": 3.0, "raw_patient_id": "not-bound"},
    ]


def test_generated_plan_routes_generated_fixture_program() -> None:
    result = execute_generated_program(
        _program(), _rows(), 2, analysis_type="projection", planner_mode="deterministic"
    )

    assert result.execution_mode == "generated"
    assert result.scientific_result_valid is True
    assert result.scientific_conclusion_eligible is False
    assert result.metrics == {"count": 2.0, "mean": 2.0}
    assert result.code_hash and result.program_hash


def test_generated_sandbox_blocks_network_import() -> None:
    with pytest.raises(ValueError, match="forbidden runtime content"):
        _program(code='socket = __import__("socket"); result = {"metrics": {"count": 1.0}}')


def test_generated_sandbox_blocks_file_io() -> None:
    with pytest.raises(ValueError, match="forbidden runtime content"):
        _program(code='handle = open("x", "w"); result = {"metrics": {"count": 1.0}}')


def test_generated_timeout_is_bounded() -> None:
    program = _program(
        code=(
            "total = 0\n"
            "for item in range(1000000000):\n"
            "    total = total + item\n"
            'result = {"metrics": {"count": float(total)}}'
        ),
        timeout_seconds=1,
    )
    with pytest.raises(GeneratedAnalysisError, match="ANALYSIS_GENERATED_TIMEOUT"):
        execute_generated_program(program, _rows(), 2, analysis_type="projection")


def test_empty_generated_metrics_are_not_scientifically_valid() -> None:
    program = _program(code='result = {"metrics": {}}')
    with pytest.raises(GeneratedAnalysisError, match="ANALYSIS_GENERATED_REQUIRED_METRICS_MISSING"):
        execute_generated_program(program, _rows(), 2, analysis_type="projection")


def test_generated_result_requires_declared_collection_output() -> None:
    program = _program(expected_outputs=["group_results"])
    with pytest.raises(GeneratedAnalysisError, match="ANALYSIS_GENERATED_EXPECTED_OUTPUT_MISSING"):
        execute_generated_program(program, _rows(), 2, analysis_type="projection")


def test_generated_program_receives_only_bound_columns() -> None:
    program = _program(
        code=(
            'keys = sorted(rows[0])\n'
            'result = {"metrics": {"bound_column_count": float(len(keys))}, '
            '"used_row_count": len(rows)}'
        )
    )
    result = execute_generated_program(program, _rows(), 2, analysis_type="projection")
    assert result.metrics["bound_column_count"] == 2.0


def test_generator_contract_explicitly_forbids_row_get() -> None:
    contract = generated_code_contract(_program())
    assert 'row.get("column_name")' in contract["input_api"]["forbidden_access"]


def test_generator_contract_describes_allowed_row_access() -> None:
    contract = generated_code_contract(_program())
    assert contract["input_api"]["allowed_access"] == ['row["column_name"]']
    assert "float" in contract["allowed_builtins"]


def test_generator_prompt_matches_sandbox_contract() -> None:
    contract = generated_code_contract(_program())
    assert contract["version"] in GENERATOR_SYSTEM_PROMPT
    assert "ALLOWED" in GENERATOR_SYSTEM_PROMPT
    assert "FORBIDDEN" in GENERATOR_SYSTEM_PROMPT
    assert "OUTPUT CONTRACT" in GENERATOR_SYSTEM_PROMPT
    assert "FunctionDef" not in contract["allowed_ast_nodes"]
    assert "function_definitions" in contract["forbidden_syntax"]
    assert "list.append" in contract["forbidden_method_calls"]


def test_generator_prompt_contains_required_result_schema() -> None:
    program = _program(metrics_schema=["count", "mean"])
    contract = generated_code_contract(program)
    assert contract["output_contract"]["required_top_level_keys"] == ["metrics", "used_row_count"]
    assert contract["output_contract"]["metrics_schema"] == ["count", "mean"]


def test_missing_metrics_feedback_is_contract_only() -> None:
    feedback = _contract_repair_feedback("ANALYSIS_GENERATED_REQUIRED_METRICS_MISSING")
    assert feedback["error"] == "MISSING_REQUIRED_OUTPUT: metrics"
    assert "scientific" not in " ".join(feedback.values()).lower()


def test_forbidden_method_feedback_is_contract_only() -> None:
    get_feedback = _contract_repair_feedback("ANALYSIS_CODE_FORBIDDEN_METHOD_GET")
    append_feedback = _contract_repair_feedback("ANALYSIS_CODE_FORBIDDEN_METHOD_APPEND")
    assert get_feedback["error"] == "FORBIDDEN_METHOD_CALL: row.get"
    assert 'row["column_name"]' in get_feedback["required_fix"]
    assert append_feedback["error"] == "FORBIDDEN_METHOD_CALL: list.append"
    assert "scientific" not in " ".join(append_feedback.values()).lower()


def test_provider_402_does_not_trigger_contract_repair(tmp_path) -> None:
    class ProviderRejected:
        def request(self, _payload):
            raise GeneratorProviderError("GENERATOR_HTTP_402")

    report = _run_case(_cases()[0], ProviderRejected(), tmp_path / "G1")
    assert report["transport_attempts"] == 1
    assert report["code_generator_calls"] == 0
    assert report["repair_count"] == 0
    assert report["failure"] == "GENERATOR_HTTP_402"


def test_generated_output_skeleton_matches_validator() -> None:
    program = _program(metrics_schema=["count", "mean"])
    contract = generated_code_contract(program)
    skeleton = contract["output_contract"]["skeleton"]
    assert set(skeleton) == {"metrics", "used_row_count"}
    assert set(skeleton["metrics"]) == {"count", "mean"}


def test_code_skeleton_uses_only_allowed_sandbox_language() -> None:
    skeleton = _code_skeleton(_cases()[0])
    assert 'row["a_abundance_value"]' in skeleton
    assert ".get(" not in skeleton
    assert ".append(" not in skeleton
    assert "def " not in skeleton
    assert 'result = {"metrics"' in skeleton


def test_observation_id_is_not_sample_key() -> None:
    program = _program(
        input_bindings=GeneratedAnalysisInputBindings(
            observation_ids=[OBSERVATION_ID], sample_key_column="analysis_sample_key"
        ),
        code=(
            f'values = [float(row["value"]) for row in rows if row["analysis_sample_key"] == "{OBSERVATION_ID}"]\n'
            'result = {"metrics": {"count": float(len(values))}, "used_row_count": len(values)}'
        ),
    )
    with pytest.raises(GeneratedAnalysisError, match="INVALID_IDENTITY_BINDING"):
        execute_generated_program(program, _rows(), 2, analysis_type="projection")


def test_generated_contract_declares_sample_key_column() -> None:
    program = _program(
        input_bindings=GeneratedAnalysisInputBindings(
            observation_ids=[OBSERVATION_ID], sample_key_column="analysis_sample_key"
        )
    )
    bindings = generated_code_contract(program)["input_bindings"]
    assert bindings["sample_key_column"] == "analysis_sample_key"
    assert "never derived from observation_id" in bindings["sample_key_semantics"]


def test_observation_id_not_exposed_as_filter_value() -> None:
    bindings = generated_code_contract(_program())["input_bindings"]
    assert "sample_key_filter_values" not in bindings
    assert "never a row or sample filter value" in bindings["observation_id_semantics"]


def test_zero_row_identity_mismatch_has_specific_reason() -> None:
    program = _program(
        code='result = {"metrics": {"count": 0.0}, "used_row_count": 0}'
    )
    with pytest.raises(GeneratedAnalysisError, match="ZERO_ROW_SELECTION_SUSPECTED_IDENTITY_MISMATCH"):
        execute_generated_program(program, _rows(), 2, analysis_type="projection")
