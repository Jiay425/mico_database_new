"""Run the Layer B real-model generated-analysis probe.

This is deliberately *not* an Agent/E2E runner.  It sends a code generator a
small, runtime-owned Program Contract, executes the returned code only through
the v1 sandbox, and writes local-only evidence under ``artifacts/``.  It never
sends rows, state snapshots, credentials, database details, or filesystem
paths to the model.

Usage (the live switch is intentionally explicit)::

    python scripts/run_generated_analysis_layer_b_probe.py --live
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

# Permit ``python scripts/...`` from the runtime repository without relying on
# an editable package install.  This is local import resolution only; it does
# not add model, credential, or artifact paths to the generator contract.
RUNTIME_ROOT = Path(__file__).resolve().parents[1]
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

from mico_agent_runtime.contracts.generated_analysis import GeneratedAnalysisPlan
from mico_agent_runtime.contracts.materialization import AnalysisPlan
from mico_agent_runtime.contracts.schema_catalog import (
    SchemaEntitySemantics,
    SchemaFieldSemantics,
    SchemaSemanticCatalog,
)
from mico_agent_runtime.graph.generated_analysis import (
    GeneratedAnalysisError,
    bind_generated_program,
    execute_generated_program,
    generated_code_contract,
)
from mico_agent_runtime.runtime.analysis_capability_registry import (
    AnalysisCapabilityContext,
    match_analysis_capability,
)


OBSERVATION_ID = "observation-" + "b" * 32
ARTIFACT_ROOT = Path("artifacts/generated_analysis_layer_b")
MAX_MODEL_CALLS_PER_GENERATED_CASE = 2
GENERATOR_SYSTEM_PROMPT = (
    "You are a constrained Python code generator. The user payload contains the authoritative "
    "generated-code-contract-v1 with explicit ALLOWED, FORBIDDEN, INPUT CONTRACT, and OUTPUT CONTRACT "
    "sections. Follow it exactly. In particular, use only the declared allowed row access form; never "
    "substitute a dictionary method. Start from the supplied code_skeleton and preserve its required result keys; "
    "modify only its bounded analysis logic when necessary. "
    "Return exactly one JSON object with exactly one key: code. The value is a complete replacement Python "
    "program. Do not output markdown, explanation, language, analysis type, execution mode, package advice, "
    "scientific validity, or any additional key."
)


def _catalog() -> SchemaSemanticCatalog:
    return SchemaSemanticCatalog(
        schemaVersion="schema-catalog-v1",
        source="java_schema_contract",
        generatedAt=datetime(2026, 9, 5, tzinfo=timezone.utc),
        entities=[
            SchemaEntitySemantics(
                entityName="sample",
                sourceTable="sample",
                fields=[
                    SchemaFieldSemantics(
                        fieldId="sample.disease", name="disease", dataType="string",
                        nullable=False, semanticStatus="verified", groupable=True,
                        scientificCapabilities=["dimension"], description="Group dimension",
                    ),
                    SchemaFieldSemantics(
                        fieldId="sample.age", name="age", dataType="integer",
                        nullable=True, semanticStatus="verified", groupable=True,
                        scientificCapabilities=["covariate", "stratifier"],
                        description="Numeric stratifier",
                    ),
                ],
            ),
            SchemaEntitySemantics(
                entityName="abundance",
                sourceTable="abundance",
                fields=[
                    SchemaFieldSemantics(
                        fieldId="abundance.value", name="value", dataType="number",
                        nullable=True, semanticStatus="verified", aggregatable=True,
                        scientificCapabilities=["outcome"], description="Numeric outcome",
                    ),
                ],
            ),
        ],
        queryRules=["select_or_with_only", "explicit_columns_only", "bounded_limit_required"],
    )


def _context(fields: list[str]) -> AnalysisCapabilityContext:
    return AnalysisCapabilityContext(
        available_observation_ids=[OBSERVATION_ID],
        available_fields=fields,
        observation_fields={OBSERVATION_ID: fields},
        distinct_counts={"sample.disease": 2, "sample.age": 4},
    )


@dataclass(frozen=True)
class ProbeCase:
    case_id: str
    action_name: str
    plan: AnalysisPlan
    semantic_fields: list[str]
    bound_columns: list[str]
    input_schema: dict[str, str]
    rows: list[dict[str, object]]


def _cases() -> list[ProbeCase]:
    shared_rows = [
        {"analysis_sample_key": "opaque-001", "a_abundance_value": 0.12, "a_sample_disease": "group_a", "a_sample_age": 34},
        {"analysis_sample_key": "opaque-002", "a_abundance_value": 0.22, "a_sample_disease": "group_a", "a_sample_age": 42},
        {"analysis_sample_key": "opaque-003", "a_abundance_value": 0.71, "a_sample_disease": "group_b", "a_sample_age": 51},
        {"analysis_sample_key": "opaque-004", "a_abundance_value": 0.83, "a_sample_disease": "group_b", "a_sample_age": 63},
    ]
    projection = AnalysisPlan.model_validate({
        "analysis_type": "projection", "source_observation_ids": [OBSERVATION_ID],
        "outcome": "abundance.value", "analysis_goal": "Produce a bounded exploratory numeric projection summary.",
        "metrics": ["count", "mean"],
    })
    stratified = AnalysisPlan.model_validate({
        "analysis_type": "stratified_comparison", "source_observation_ids": [OBSERVATION_ID],
        "outcome": "abundance.value", "group_field": "sample.disease", "stratify_by": ["sample.age"],
        # No NumericStratificationSpec: this is the registered generated-only
        # non-standard numeric-stratification shape.
        "analysis_goal": "Produce a bounded exploratory numeric-stratified abundance summary.",
        "metrics": ["count", "mean"],
    })
    unsupported = AnalysisPlan.model_validate({
        "analysis_type": "group_comparison", "source_observation_ids": [OBSERVATION_ID],
        "outcome": "sample.age", "group_field": "sample.disease",
        "analysis_goal": "Attempt an invalid covariate-as-outcome comparison.", "metrics": ["count"],
    })
    return [
        ProbeCase(
            "G1", "analyze_projection", projection, ["abundance.value"],
            ["analysis_sample_key", "a_abundance_value"],
            {"analysis_sample_key": "opaque_string", "a_abundance_value": "number_or_null"},
            shared_rows,
        ),
        ProbeCase(
            "G2", "stratified_analysis", stratified,
            ["abundance.value", "sample.disease", "sample.age"],
            ["analysis_sample_key", "a_abundance_value", "a_sample_disease", "a_sample_age"],
            {
                "analysis_sample_key": "opaque_string", "a_abundance_value": "number_or_null",
                "a_sample_disease": "categorical_string", "a_sample_age": "integer_or_null",
            },
            shared_rows,
        ),
        ProbeCase(
            "U1", "compare_groups", unsupported, ["sample.age", "sample.disease"],
            ["analysis_sample_key", "a_sample_age", "a_sample_disease"],
            {"analysis_sample_key": "opaque_string", "a_sample_age": "integer_or_null", "a_sample_disease": "categorical_string"},
            shared_rows,
        ),
    ]


def _endpoint(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username:
        raise ValueError("MICO_RESEARCH_PLANNER_BASE_URL is invalid")
    base = base_url.rstrip("/")
    path = parsed.path.rstrip("/")
    if (parsed.hostname or "").lower().endswith("deepseek.com") or path.endswith("/v1") or path.endswith("/openai"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


class RealCodeGenerator:
    """Narrow OpenAI-compatible adapter used only by the Layer B probe."""

    def __init__(self, *, base_url: str, model: str, token: str, timeout_seconds: float) -> None:
        self.endpoint = _endpoint(base_url)
        self.model = model
        self.token = token
        self.timeout_seconds = timeout_seconds
        self.is_deepseek = "deepseek.com" in (urlsplit(base_url).hostname or "").lower()

    def request(self, payload: dict[str, object]) -> tuple[dict[str, object], str]:
        body: dict[str, object] = {
            "model": self.model,
            "messages": [{"role": "system", "content": GENERATOR_SYSTEM_PROMPT}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            "response_format": {"type": "json_object"}, "max_tokens": 700, "temperature": 0,
        }
        if self.is_deepseek:
            body["thinking"] = {"type": "disabled"}
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        with httpx.Client(timeout=self.timeout_seconds, trust_env=False, limits=httpx.Limits(max_keepalive_connections=0)) as client:
            response = client.post(self.endpoint, headers=headers, json=body)
        if response.status_code != 200:
            # Provider/transport errors are not a code-contract failure and
            # must never consume the one permitted code-repair attempt.
            raise GeneratorProviderError(f"GENERATOR_HTTP_{response.status_code}")
        response_payload = response.json()
        content = response_payload["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise ValueError("GENERATOR_RESPONSE_CONTENT_INVALID")
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match is None:
                raise ValueError("GENERATOR_RESPONSE_NOT_JSON")
            parsed = json.loads(match.group(0))
        if not isinstance(parsed, dict) or set(parsed) != {"code"} or not isinstance(parsed.get("code"), str):
            raise ValueError("GENERATOR_RESPONSE_CONTRACT_INVALID")
        # Persist only an audit-safe response: no headers, URL, credentials, or rows.
        return {"code": parsed["code"]}, json.dumps({"content": content}, ensure_ascii=False)


class GeneratorProviderError(RuntimeError):
    """A real provider rejected the request before any code was received."""


def _template_program(case: ProbeCase):
    """Build only Runtime-owned fields needed to render the code contract."""

    placeholder = GeneratedAnalysisPlan.model_validate({
        "language": "python", "analysisType": case.plan.analysis_type,
        "code": 'result = {"metrics": {"count": 0.0, "mean": 0.0}, "used_row_count": 0}',
    })
    return bind_generated_program(
        case.plan, action_name=case.action_name,
        required_columns=case.bound_columns, generated_plan=placeholder,
    )


def _code_skeleton(case: ProbeCase) -> str:
    """Produce a safe, contract-shaped starting point without exposing rows."""

    numeric_columns = [
        name for name, kind in case.input_schema.items()
        if kind in {"number_or_null", "integer_or_null"}
    ]
    if len(numeric_columns) != 1:
        raise ValueError("Layer B probe requires exactly one declared numeric input column")
    column = numeric_columns[0]
    return (
        f'values = [float(row["{column}"]) for row in rows if row["{column}"] is not None]\n'
        "count = len(values)\n"
        "mean = sum(values) / count if count > 0 else 0.0\n"
        'result = {"metrics": {"count": float(count), "mean": mean}, "used_row_count": count}'
    )


def _generator_payload(case: ProbeCase, program: object, feedback: dict[str, str] | None = None) -> dict[str, object]:
    # ``generated_code_contract`` is derived beside the sandbox AST policy,
    # not copied from a separately maintained prompt string.
    contract = generated_code_contract(program)  # type: ignore[arg-type]
    payload: dict[str, object] = {
        "action_name": case.action_name,
        "analysis_goal": case.plan.analysis_goal,
        "bound_observation_ids": case.plan.source_observation_ids,
        "input_schema": case.input_schema,
        "generated_code_contract": contract,
        "code_skeleton": _code_skeleton(case),
    }
    if feedback is not None:
        payload["contract_repair"] = feedback
    return payload


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _record_generation_failure(report: dict[str, object], code: str) -> None:
    """Keep static, sandbox, and result-validation failure layers distinct."""

    report["failure"] = code[:120]
    if code.startswith(("ANALYSIS_CODE_CALL_", "ANALYSIS_CODE_FORBIDDEN_", "ANALYSIS_CODE_NODE_", "ANALYSIS_CODE_SYNTAX", "ANALYSIS_CODE_PRIVATE_", "ANALYSIS_CODE_AST_", "ANALYSIS_CODE_LENGTH_", "ANALYSIS_CODE_RESULT_")):
        report["static_safety_validation"] = "failed"
        return
    if code.startswith(("ANALYSIS_GENERATED_REQUIRED_", "ANALYSIS_GENERATED_EXPECTED_", "ANALYSIS_GENERATED_METRIC_", "ANALYSIS_GENERATED_NONFINITE_", "ANALYSIS_GENERATED_SAMPLE_")):
        report["sandbox_execution"] = "passed"
        report["result_validation"] = "failed"
        return
    report["sandbox_execution"] = "failed"


def _contract_repair_feedback(code: str) -> dict[str, str]:
    """Return only mechanical contract feedback; never a scientific answer."""

    if code == "ANALYSIS_CODE_FORBIDDEN_METHOD_GET":
        return {
            "error": "FORBIDDEN_METHOD_CALL: row.get",
            "required_fix": 'Use only row["column_name"] for record access; row.get(...) is forbidden.',
        }
    if code == "ANALYSIS_CODE_FORBIDDEN_METHOD_APPEND":
        return {
            "error": "FORBIDDEN_METHOD_CALL: list.append",
            "required_fix": "All methods are forbidden. Use a list comprehension instead of list.append(...).",
        }
    if code == "ANALYSIS_CODE_NODE_FUNCTIONDEF":
        return {
            "error": "FORBIDDEN_SYNTAX: function_definition",
            "required_fix": "Do not use def or return. Write top-level assignments, for/if statements, or comprehensions only.",
        }
    if code == "ANALYSIS_GENERATED_REQUIRED_METRICS_MISSING":
        return {
            "error": "MISSING_REQUIRED_OUTPUT: metrics",
            "required_fix": 'Assign result with the required key "metrics" and all schema metrics.',
        }
    return {
        "error": "GENERATED_CODE_CONTRACT_REJECTED",
        "required_fix": "Return a complete replacement program that follows generated-code-contract-v1.",
    }


def _run_case(case: ProbeCase, generator: RealCodeGenerator | None, output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    match = match_analysis_capability(case.plan, _catalog(), _context(case.semantic_fields))
    report: dict[str, object] = {
        "case_id": case.case_id, "capability": match.mode, "capability_code": match.capability_code,
        "reason_code": match.reason_code, "code_generator_calls": 0, "transport_attempts": 0, "repair_count": 0,
        "static_safety_validation": "not_run", "sandbox_execution": "not_run", "result_validation": "not_run",
    }
    template_program = _template_program(case)
    _write_json(output / "program_contract.json", _generator_payload(case, template_program))
    if match.mode == "UNSUPPORTED":
        report["fail_closed"] = True
        _write_json(output / "report.json", report)
        return report
    if match.mode != "SUPPORTED_GENERATED" or generator is None:
        report["failure"] = "GENERATOR_NOT_CONFIGURED" if generator is None else "EXPECTED_GENERATED_CAPABILITY"
        _write_json(output / "report.json", report)
        return report

    feedback: dict[str, str] | None = None
    for attempt in range(MAX_MODEL_CALLS_PER_GENERATED_CASE):
        try:
            request_payload = _generator_payload(case, template_program, feedback)
            _write_json(output / f"generator_request_{attempt + 1}.json", request_payload)
            report["transport_attempts"] = int(report["transport_attempts"]) + 1
            parsed, raw = generator.request(request_payload)
            report["code_generator_calls"] = int(report["code_generator_calls"]) + 1
            (output / f"generator_response_{attempt + 1}.json").write_text(raw, encoding="utf-8")
            generated = GeneratedAnalysisPlan.model_validate({"language": "python", "analysisType": case.plan.analysis_type, **parsed})
            program = bind_generated_program(case.plan, action_name=case.action_name, required_columns=case.bound_columns, generated_plan=generated)
            _write_json(output / f"bound_program_{attempt + 1}.json", program.model_dump(mode="json", exclude={"code"}))
            report["static_safety_validation"] = "passed"
            if attempt == 0:
                report["first_pass_ast_valid"] = True
            result = execute_generated_program(program, case.rows, len(case.rows), analysis_type=case.plan.analysis_type, planner_mode="model")
            report.update({
                "sandbox_execution": "passed", "result_validation": "passed", "scientific_result_valid": result.scientific_result_valid,
                "scientific_conclusion_eligible": result.scientific_conclusion_eligible, "execution_mode": result.execution_mode,
                "code_hash": result.code_hash, "program_hash": result.program_hash, "metrics": result.metrics,
                "first_pass_valid": attempt == 0,
                "first_pass_output_contract_valid": attempt == 0,
                "first_pass_sandbox_valid": attempt == 0,
                "first_pass_result_valid": attempt == 0,
            })
            report.pop("failure", None)
            _write_json(output / "analysis_result.json", result.model_dump(mode="json"))
            break
        except GeneratorProviderError as error:
            report["failure"] = str(error)[:120]
            # There is no model code to repair.  Stop rather than wasting the
            # bounded repair budget on a provider billing/outage response.
            break
        except (GeneratedAnalysisError, ValueError, KeyError, TypeError) as error:
            code = getattr(error, "reasonCode", None) or getattr(error, "code", None) or str(error)
            _record_generation_failure(report, str(code))
            if attempt == 0:
                report["first_pass_ast_valid"] = report["static_safety_validation"] == "passed"
                report["first_pass_output_contract_valid"] = report["result_validation"] != "failed"
                report["first_pass_sandbox_valid"] = report["sandbox_execution"] == "passed"
                report["first_pass_result_valid"] = False
            if attempt + 1 >= MAX_MODEL_CALLS_PER_GENERATED_CASE:
                break
            feedback = _contract_repair_feedback(str(code))
            report["repair_count"] = 1
    _write_json(output / "report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="permit real code-generator API calls")
    parser.add_argument(
        "--provider", choices=("research", "gemini"), default="research",
        help="real model endpoint; Gemini uses the configured canary API key",
    )
    parser.add_argument("--output", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--cases", nargs="+", choices=("G1", "G2", "U1"), default=("G1", "G2", "U1"),
        help="run only named probe cases; U1 never calls a model",
    )
    args = parser.parse_args(argv)
    if args.timeout_seconds <= 0:
        raise SystemExit("timeout must be positive")
    generator: RealCodeGenerator | None = None
    if args.live:
        if args.provider == "gemini":
            base_url = os.environ.get(
                "MICO_TASK_UNDERSTANDING_BASE_URL",
                "https://generativelanguage.googleapis.com/v1beta/openai",
            ).strip()
            model = os.environ.get("MICO_TASK_UNDERSTANDING_MODEL", "gemini-3.5-flash-lite").strip()
            token = (
                os.environ.get("MICO_GEMINI_API_KEY", "").strip()
                or os.environ.get("GEMINI_API_KEY", "").strip()
                or os.environ.get("GOOGLE_API_KEY", "").strip()
            )
        else:
            base_url = os.environ.get("MICO_RESEARCH_PLANNER_BASE_URL", "").strip()
            model = os.environ.get("MICO_RESEARCH_PLANNER_MODEL", "").strip()
            token = os.environ.get("MICO_RESEARCH_PLANNER_TOKEN", "").strip()
        if not (base_url and model and token):
            raise SystemExit("live probe requires MICO_RESEARCH_PLANNER_BASE_URL/MODEL/TOKEN")
        generator = RealCodeGenerator(base_url=base_url, model=model, token=token, timeout_seconds=args.timeout_seconds)
    run_dir = args.output / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    selected_cases = [case for case in _cases() if case.case_id in set(args.cases)]
    results = [_run_case(case, generator, run_dir / case.case_id) for case in selected_cases]
    summary = {
        "real_model_generator": bool(args.live), "provider": args.provider if args.live else None,
        "qwen_calls": 0, "a100_started": False,
        "cases": results,
        "generated_trace_complete": all((run_dir / item["case_id"] / "report.json").exists() for item in results),
        "package_expansion_required": False,
    }
    _write_json(run_dir / "summary.json", summary)
    print(json.dumps({"run_dir": str(run_dir), "summary": summary}, ensure_ascii=False))
    expected_generated = {item for item in args.cases if item in {"G1", "G2"}}
    successful = [item for item in results if item["case_id"] in expected_generated and item.get("result_validation") == "passed"]
    unsupported = next((item for item in results if item["case_id"] == "U1"), None)
    generated_ok = len(successful) == len(expected_generated)
    unsupported_ok = unsupported is None or (unsupported.get("fail_closed") and unsupported.get("code_generator_calls") == 0)
    return 0 if generated_ok and unsupported_ok else 1


if __name__ == "__main__":
    sys.exit(main())
