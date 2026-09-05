"""One-shot Gemini contract probe for ``analyze_projection``.

This is the final 4D-1 close-out probe.  It does not invoke Task Understanding,
Scientific Policy, Java, MySQL, retrieval, or the Agent Loop.  It loads a
previously captured real Decision State and observation context, then sends
one bounded Materializer call through the existing ``plan_action`` adapter.
Transport attempts, successful model responses, and contract repairs are
reported independently so network failures cannot be mistaken for a model
contract failure.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mico_agent_runtime.contracts.decision_state import ScientificDecisionState
from mico_agent_runtime.contracts.research import (
    AnalyzeProjectionArguments,
    ScientificObservationSummary,
    ScientificPlannerContext,
    validate_scientific_action,
)
from mico_agent_runtime.contracts.schema_catalog import SchemaSemanticCatalog
from mico_agent_runtime.ports.gemini_resilience import GeminiRequestBudget
from mico_agent_runtime.ports.research_planner import HttpResearchPlannerPort
from scripts.check_scientific_chain_services import run_preflight
from scripts.run_gemini_dynamic_canary import (
    GEMINI_BASE_URL,
    GEMINI_MODEL,
    RecordingTransport,
    _gemini_token,
)
from scripts.run_materializer_contract_sweep import _load_observation_id


DEFAULT_STATE = (
    REPO_ROOT / "artifacts" / "gemini_dynamic_canary_4d1g_final3" / "state_s4.json"
)
DEFAULT_CATALOG = (
    REPO_ROOT / "artifacts" / "gemini_dynamic_canary_4d1g_final3" / "semantic_catalog.json"
)
DEFAULT_CONTEXT = (
    REPO_ROOT / "artifacts" / "gemini_dynamic_canary_4d1g_final3" / "materializer_request_007.json"
)
CORE_TRACE = REPO_ROOT / "artifacts" / "gemini_dynamic_canary_4d1g_final3" / "trace.json"
STATIC_AUDIT = REPO_ROOT / "artifacts" / "materializer_contract_sweep_4d1i" / "static_audit.json"


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_state(path: Path) -> ScientificDecisionState:
    return ScientificDecisionState.model_validate(_load_json(path))


def _load_catalog(path: Path) -> SchemaSemanticCatalog:
    return SchemaSemanticCatalog.model_validate(_load_json(path))


def _projection_context(
    state: ScientificDecisionState,
    catalog: SchemaSemanticCatalog,
    observation_id: str,
) -> ScientificPlannerContext:
    return ScientificPlannerContext(
        questionSummary=state.task.query,
        intent="scientific_exploration",
        approvedActions=["analyze_projection"],
        remainingActionBudget=1,
        observations=[
            ScientificObservationSummary(
                observationId=observation_id,
                actionName="execute_read_query",
                status="VALIDATED",
                source="java_controlled_read",
                rowCount=state.data_state.row_count,
                queryPlanFields=[
                    "sample.disease",
                    "sample.age",
                    "abundance.feature",
                    "abundance.value",
                ],
            )
        ],
        schemaCatalog=catalog,
        decisionState=state.model_dump(mode="json"),
    )


def _decode_content(record: dict[str, Any]) -> Any:
    response = record.get("response")
    if not isinstance(response, dict):
        return None
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    if not isinstance(content, str):
        return content
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return content


def _validate_projection_response(payload: Any) -> tuple[bool, str | None]:
    """Validate the exact legacy dispatch envelope for this close-out probe."""

    if not isinstance(payload, dict):
        return False, "response_shape"
    if payload.get("actionName") != "analyze_projection":
        return False, "action_mismatch"
    arguments = payload.get("arguments")
    if not isinstance(arguments, dict):
        return False, "required_field"
    if "observationIds" in arguments:
        return False, "plural_observation_ids_for_projection"
    try:
        AnalyzeProjectionArguments.model_validate(arguments)
        candidate = dict(payload)
        # The Runtime owns correlation IDs; use a synthetic ID only for this
        # local contract check, exactly as the production boundary does.
        candidate["actionId"] = "action-" + "0" * 32
        validate_scientific_action(candidate)
    except (ValidationError, TypeError, ValueError):
        return False, "schema_validation"
    return True, None


def _audit_transport(transport: RecordingTransport) -> dict[str, Any]:
    attempts = list(transport.records)
    model_records = [item for item in attempts if item.get("response_status") == 200]
    decoded = [_decode_content(item) for item in model_records]
    validations = [_validate_projection_response(item) for item in decoded]
    first_valid, first_reason = validations[0] if validations else (None, "no_model_response")
    final_valid, final_reason = validations[-1] if validations else (False, "no_model_response")
    return {
        "transport_attempts": len(attempts),
        "transport_failures": [
            {
                "request_index": item.get("request_index"),
                "error": item.get("error"),
                "status": item.get("response_status"),
            }
            for item in attempts
            if item.get("response_status") != 200
        ],
        "model_responses": len(model_records),
        "model_response_request_indices": [item.get("request_index") for item in model_records],
        "contract_repairs": sum(
            len(item.get("request", {}).get("body", {}).get("messages", [])) > 2
            for item in attempts
        ),
        "first_model_response_valid": first_valid,
        "first_model_response_failure_reason": first_reason,
        "final_contract_valid": final_valid,
        "final_contract_failure_reason": final_reason,
        "first_model_response": decoded[0] if decoded else None,
        "final_model_response": decoded[-1] if decoded else None,
    }


def _save_transport(out_dir: Path, transport: RecordingTransport) -> None:
    for record in transport.records:
        index = int(record["request_index"])
        _dump(out_dir / f"projection_probe_request_{index:03d}.json", record.get("request"))
        _dump(out_dir / f"projection_probe_response_{index:03d}.json", {
            "response_status": record.get("response_status"),
            "response": record.get("response"),
            "error": record.get("error"),
        })


def run_probe(
    out_dir: Path,
    *,
    state_path: Path = DEFAULT_STATE,
    catalog_path: Path = DEFAULT_CATALOG,
    context_path: Path = DEFAULT_CONTEXT,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    preflight = asyncio.run(run_preflight())
    _dump(out_dir / "preflight.json", preflight)
    if preflight.get("status") != "pass":
        report = {
            "probe_version": "projection-materializer-probe-v1",
            "status": "blocked",
            "training_eligible": False,
            "a100_ready": False,
            "api_calls_made": 0,
            "reason": "CHAIN_PREFLIGHT_FAILED",
            "preflight": preflight,
        }
        _dump(out_dir / "projection_probe_report.json", report)
        return report

    state = _load_state(state_path)
    catalog = _load_catalog(catalog_path)
    observation_id = _load_observation_id(context_path)
    core_chain_verified = CORE_TRACE.exists()
    static_audit_payload = (
        _load_json(STATIC_AUDIT) if STATIC_AUDIT.exists() else {}
    )
    static_audit_pass = static_audit_payload.get("status") == "PASS"
    token = _gemini_token()
    model = os.environ.get("MICO_GEMINI_MODEL", GEMINI_MODEL).strip() or GEMINI_MODEL
    base_url = os.environ.get("MICO_GEMINI_BASE_URL", GEMINI_BASE_URL).strip() or GEMINI_BASE_URL
    context = _projection_context(state, catalog, observation_id)
    _dump(out_dir / "state_cached.json", state.model_dump(mode="json"))
    _dump(out_dir / "materializer_context.json", context.model_dump(mode="json"))

    # Keep this probe bounded.  It does not raise the production Gemini budget
    # and it makes any transport retry visible instead of looping indefinitely.
    budget = GeminiRequestBudget(total_limit=4, role_limits={"materializer": 4})
    transport = RecordingTransport("projection_materializer_probe")
    planner = HttpResearchPlannerPort(
        base_url,
        model,
        token,
        transport=transport,
        materializer_origin="gemini_canary_model",
        request_budget=budget,
        response_cache=None,
    )
    result_payload: Any = None
    call_error: str | None = None
    try:
        result_payload = planner.plan_action(context)
    except Exception as exc:  # pragma: no cover - provider outcome
        call_error = f"{type(exc).__name__}:{exc}"
    finally:
        planner.close()

    _save_transport(out_dir, transport)
    audit = _audit_transport(transport)
    planner_mode = getattr(result_payload, "mode", None)
    probe_pass = audit["first_model_response_valid"] is True
    a100_ready = bool(probe_pass and core_chain_verified and static_audit_pass)
    report = {
        "probe_version": "projection-materializer-probe-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if probe_pass else "fail",
        "training_eligible": False,
        "a100_ready": a100_ready,
        "model": model,
        "materializer_origin": "gemini_canary_model",
        "action": "analyze_projection",
        "canonical_contract": {
            "actionName": "analyze_projection",
            "required_argument_keys": ["actionName", "observationId", "analysisGoal"],
            "forbidden_argument_keys": ["observationIds", "dimensions", "confounders"],
        },
        "state_source": str(state_path),
        "catalog_source": str(catalog_path),
        "observation_context_source": str(context_path),
        "core_dynamic_chain_verified": core_chain_verified,
        "core_dynamic_chain_evidence": str(CORE_TRACE),
        "static_contract_audit_pass": static_audit_pass,
        "static_contract_audit_evidence": str(STATIC_AUDIT),
        "preflight": preflight,
        "call_error": call_error,
        "planner_mode": planner_mode,
        "deterministic_fallback_used": planner_mode == "deterministic",
        "budget": {
            "total_limit": budget.total_limit,
            "total_used": budget.total_used,
            "role_limits": dict(budget.role_limits),
            "role_used": dict(budget.role_used),
        },
        "audit": audit,
    }
    _dump(out_dir / "projection_probe_report.json", report)
    _dump(out_dir / "a100_readiness.json", {
        "A100_READY": a100_ready,
        "reason": (
            "projection first response passed; static audit and previously verified core chain are present"
            if a100_ready else
            "projection probe, static contract audit, or previously verified core chain is incomplete"
        ),
        "probe_report": "projection_probe_report.json",
        "training_eligible": False,
    })
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "projection_materializer_probe_4d1",
    )
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--context", type=Path, default=DEFAULT_CONTEXT)
    args = parser.parse_args(argv)
    report = run_probe(
        args.output_dir,
        state_path=args.state,
        catalog_path=args.catalog,
        context_path=args.context,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
