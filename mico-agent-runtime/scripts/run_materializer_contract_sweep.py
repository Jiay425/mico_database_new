"""4D-1I: one-pass Gemini Materializer contract sweep for all Actions.

The sweep intentionally does not invoke the Agent Loop or execute a query or
analysis.  It sends one bounded, cached-context Materializer request for each
Action whose canonical contract is model-produced.  ``inspect_cohort`` and
``finish`` are Runtime-owned and are checked deterministically.  A service
preflight runs before token lookup/provider construction, so a dead Java or
remote MySQL service consumes zero Gemini requests.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mico_agent_runtime.contracts.generated_analysis import AnalysisPlannerContext
from mico_agent_runtime.contracts.materialization import AnalysisPlan
from mico_agent_runtime.contracts.materializer_contract import (
    MATERIALIZER_CONTRACTS,
    MaterializerContract,
)
from mico_agent_runtime.contracts.research import (
    ScientificObservationSummary,
    ScientificPlannerContext,
)
from mico_agent_runtime.contracts.schema_catalog import SchemaSemanticCatalog
from mico_agent_runtime.ports.gemini_resilience import (
    GeminiRequestBudget,
    GeminiRequestBudgetExceeded,
)
from mico_agent_runtime.ports.research_planner import HttpResearchPlannerPort
from scripts.audit_materializer_contracts import build_report as build_static_report
from scripts.check_scientific_chain_services import run_preflight
from scripts.run_gemini_dynamic_canary import (
    GEMINI_BASE_URL,
    GEMINI_MODEL,
    RecordingTransport,
    _gemini_token,
    _materializer_call_audit,
    _materializer_payload_records,
)


DEFAULT_CATALOG = (
    REPO_ROOT / "artifacts" / "gemini_dynamic_canary_4d1g_final3" / "semantic_catalog.json"
)
DEFAULT_CONTEXT = (
    REPO_ROOT / "artifacts" / "gemini_dynamic_canary_4d1g_final3" / "materializer_request_007.json"
)


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _load_catalog(path: Path) -> SchemaSemanticCatalog:
    return SchemaSemanticCatalog.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _load_observation_id(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = payload.get("body", payload) if isinstance(payload, dict) else payload
    messages = body.get("messages", []) if isinstance(body, dict) else []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            context = json.loads(content)
        except json.JSONDecodeError:
            continue
        ids = context.get("sourceObservationIds", []) if isinstance(context, dict) else []
        if isinstance(ids, list) and ids and isinstance(ids[0], str):
            return ids[0]
    raise ValueError("REAL_OBSERVATION_CONTEXT_REQUIRED")


def _available_fields(catalog: SchemaSemanticCatalog) -> list[str]:
    fields: list[str] = []
    for entity in catalog.entities:
        entity_id = entity.entityId or entity.entityName
        for field in entity.fields:
            if field.sensitive or field.semanticStatus != "verified":
                continue
            fields.append(field.fieldId or f"{entity_id}.{field.name}")
    return list(dict.fromkeys(fields))[:64]


def _columns(fields: list[str]) -> list[str]:
    return ["a_" + field.replace(".", "_") for field in fields]


def _analysis_question(action: str) -> str:
    return {
        "compare_groups": "Compare the validated disease groups for the numeric abundance outcome.",
        "stratified_analysis": "Compare disease groups within the available age strata.",
        "adjust_confounders": "Assess the disease-group effect after adjusting for age.",
        "cross_project_validate": "Validate whether the disease-group effect is stable across projects.",
        "cross_disease_validate": "Test whether the abundance pattern is specific across diseases.",
        "analyze_projection": "Analyze the validated multi-feature abundance projection.",
    }[action]


def _analysis_context(
    action: str,
    catalog: SchemaSemanticCatalog,
    observation_id: str,
) -> AnalysisPlannerContext:
    fields = _available_fields(catalog)
    return AnalysisPlannerContext(
        questionSummary=_analysis_question(action),
        workflow=action,
        actionName=action,
        sourceObservationIds=[observation_id],
        availableSemanticFields=fields,
        requiredSemanticFields=[],
        requiredGroupField=None,
        columns=_columns(fields),
        previewRows=[],
        executionFeedback=[],
    )


def _action_context(
    action: str,
    catalog: SchemaSemanticCatalog,
    observation_id: str,
) -> ScientificPlannerContext:
    observation = ScientificObservationSummary(
        observationId=observation_id,
        actionName="execute_read_query",
        status="VALIDATED",
        source="java_controlled_read",
        rowCount=100,
        queryPlanFields=[
            "sample.disease",
            "sample.age",
            "abundance.feature",
            "abundance.value",
            "metadata.project",
        ],
    )
    return ScientificPlannerContext(
        questionSummary=(
            "Compare T2D and Healthy microbiome differences, assess age confounding, "
            "project stability, and literature support."
            if action != "retrieve_evidence" else
            "Retrieve literature evidence about disease-group microbiome differences and age."
        ),
        intent="scientific_exploration",
        approvedActions=[action],
        remainingActionBudget=6,
        observations=[observation] if action != "execute_read_query" else [],
        schemaCatalog=catalog,
    )


def _expected_wire_kind(contract: MaterializerContract) -> str:
    if contract.contract_kind == "analysis_plan_v2":
        return "typed_analysis_plan"
    return "scientific_action" if contract.materializer_required else "runtime_owned"


def _validate_result_shape(
    contract: MaterializerContract,
    result: Any,
    audit: dict[str, Any] | None,
    *,
    final: bool = False,
) -> tuple[bool | None, str | None]:
    """Validate one recorded provider attempt against its canonical contract.

    ``first_pass`` and ``final`` are deliberately separate observations.  A
    provider call may reject the first response and return a valid response on
    a repair attempt; consulting ``first_pass_success`` while validating the
    final attempt would incorrectly turn that successful repair into a final
    failure.  The audit record is the source of truth because a planner may
    return a deterministic compatibility fallback after exhausting retries;
    that fallback is never accepted by this Gemini-only sweep.
    """

    if not contract.materializer_required:
        return True, None
    if audit is None:
        return False, "no_provider_attempt" if final else None
    model_response_count = int(audit.get("model_response_count", 0) or 0)
    if model_response_count == 0:
        # A transport-only call has not produced a model response and must not
        # be scored as an invalid first-pass contract.
        return (False, "no_model_response") if final else (None, None)
    first_valid = audit.get("first_model_response_valid")
    final_valid = audit.get("final_model_response_valid")
    failure_reason = (
        audit.get("final_model_response_failure_reason")
        if final
        else audit.get("first_model_response_failure_reason")
    )
    if (final and final_valid is not True) or (not final and first_valid is not True):
        return False, failure_reason or "contract_failure"
    contract_kind = (
        audit.get("final_model_contract_kind")
        if final
        else audit.get("first_model_contract_kind")
    )
    if contract_kind != _expected_wire_kind(contract):
        return False, "contract_kind_mismatch"
    if contract.contract_kind == "analysis_plan_v2":
        expected = contract.discriminator.split("=", 1)[1]
        if audit.get("expected_analysis_type") != expected:
            return False, "analysis_type_mismatch"
    if final and getattr(result, "mode", None) == "deterministic":
        return False, "deterministic_materializer_fallback_not_allowed"
    return True, None


def _save_transport(out_dir: Path, action: str, transport: RecordingTransport) -> None:
    for record in transport.records:
        index = int(record["request_index"])
        _dump(out_dir / f"{action}_request_{index:03d}.json", record.get("request"))
        _dump(out_dir / f"{action}_response_{index:03d}.json", {
            "response_status": record.get("response_status"),
            "response": record.get("response"),
            "error": record.get("error"),
        })


def run_sweep(
    out_dir: Path,
    *,
    catalog_path: Path = DEFAULT_CATALOG,
    context_path: Path = DEFAULT_CONTEXT,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    preflight = asyncio.run(run_preflight())
    _dump(out_dir / "preflight.json", preflight)
    if preflight.get("status") != "pass":
        result = {
            "status": "blocked",
            "training_eligible": False,
            "api_calls_made": 0,
            "reason": "CHAIN_PREFLIGHT_FAILED",
            "preflight": preflight,
        }
        _dump(out_dir / "sweep_report.json", result)
        return result

    catalog = _load_catalog(catalog_path)
    observation_id = _load_observation_id(context_path)
    token = _gemini_token()
    model = os.environ.get("MICO_GEMINI_MODEL", GEMINI_MODEL).strip() or GEMINI_MODEL
    base_url = os.environ.get("MICO_GEMINI_BASE_URL", GEMINI_BASE_URL).strip() or GEMINI_BASE_URL
    static_report = build_static_report()
    _dump(out_dir / "static_audit_snapshot.json", static_report)

    # One first-pass request per model-backed Action, with bounded headroom for
    # contract repair.  The budget is intentionally fixed at the current
    # 15-request free-tier-safe ceiling; this sweep must expose amplification,
    # not hide it by raising the quota.
    budget = GeminiRequestBudget(total_limit=15, role_limits={"materializer": 15})
    per_action: list[dict[str, Any]] = []
    for contract in MATERIALIZER_CONTRACTS:
        item: dict[str, Any] = {
            "action": contract.action,
            "materializer_required": contract.materializer_required,
            "canonical_contract_kind": contract.contract_kind,
            "canonical_schema": contract.schema_name,
            "canonical_discriminator": contract.discriminator,
            "status": "not_applicable" if not contract.materializer_required else "pending",
            "first_pass_valid": None if not contract.materializer_required else None,
            "repair_success": False,
            "final_valid": None if not contract.materializer_required else None,
            "transport_attempts": 0,
            "model_responses": 0,
            "contract_repairs": 0,
            "first_model_response_valid": None,
            "transport_failure_reasons": [],
            "provider_http_requests": 0,
            "attempt_count": 0,
            "first_failure_reason": None,
            "final_failure_reason": None,
        }
        if not contract.materializer_required:
            per_action.append(item)
            continue

        # Preserve an explicit row for every model-backed Action even when a
        # preceding Action consumed the bounded request budget.  Omitting the
        # remaining Actions would make a partial sweep look like a complete
        # ten-Action audit and would under-count first-pass failures.
        if budget.total_used >= budget.total_limit:
            item.update({
                "status": "budget_blocked",
                "first_pass_valid": None,
                "final_valid": None,
                "first_failure_reason": "GEMINI_REQUEST_BUDGET_EXHAUSTED",
                "final_failure_reason": "GEMINI_REQUEST_BUDGET_EXHAUSTED",
                "call_error": "GeminiRequestBudgetExceeded:materializer budget exhausted",
            })
            per_action.append(item)
            continue

        transport = RecordingTransport(f"materializer_sweep_{contract.action}")
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
            if contract.contract_kind == "analysis_plan_v2":
                result_payload = planner.generate_typed_analysis(
                    _analysis_context(contract.action, catalog, observation_id)
                )
            else:
                result_payload = planner.plan_action(
                    _action_context(contract.action, catalog, observation_id)
                )
        except GeminiRequestBudgetExceeded as exc:
            call_error = f"{type(exc).__name__}:{exc}"
        except Exception as exc:  # pragma: no cover - exercised by real provider
            call_error = f"{type(exc).__name__}:{exc}"
        finally:
            planner.close()

        _save_transport(out_dir, contract.action, transport)
        records = _materializer_payload_records(transport, catalog=catalog)
        audit = _materializer_call_audit(records)
        _dump(out_dir / f"{contract.action}_audit.json", audit)
        first = audit[0] if audit else None
        final = audit[-1] if audit else None
        first_valid, first_reason = _validate_result_shape(
            contract, result_payload, first, final=False
        )
        final_valid, final_reason = _validate_result_shape(
            contract, result_payload, final, final=True
        )
        item.update({
            "status": (
                "pass"
                if final_valid is True and first_valid is True
                else "pass_after_repair"
                if final_valid is True
                else "fail"
            ),
            "first_pass_valid": first_valid,
            "repair_success": bool(
                first is not None
                and first_valid is False
                and final_valid is True
                and int(final.get("contract_repair_count", 0) or 0) > 0
            ),
            "contract_repair_success": bool(
                first is not None
                and first_valid is False
                and final_valid is True
                and int(final.get("contract_repair_count", 0) or 0) > 0
            ),
            "transport_retry_success": bool(
                first is not None
                and first_valid is True
                and final_valid is True
                and int(final.get("transport_failure_count", 0) or 0) > 0
            ),
            "final_valid": final_valid,
            "transport_attempts": int(final.get("transport_attempts", len(transport.records)))
            if final else len(transport.records),
            "model_responses": int(final.get("model_response_count", 0)) if final else 0,
            "contract_repairs": int(final.get("contract_repair_count", 0)) if final else 0,
            "first_model_response_valid": first_valid,
            "transport_failure_reasons": (
                list(final.get("transport_failure_reasons", [])) if final else []
            ),
            "provider_http_requests": len(transport.records),
            "attempt_count": int(final.get("attempt_count", 0)) if final else 0,
            "first_failure_reason": first_reason,
            "final_failure_reason": final_reason,
            "call_error": call_error,
            "result_mode": getattr(result_payload, "mode", None),
            "deterministic_fallback_used": bool(
                getattr(result_payload, "mode", None) == "deterministic"
            ),
            "result_plan": (
                getattr(getattr(result_payload, "plan", None), "model_dump", lambda **_: None)(mode="json")
                if result_payload is not None and contract.contract_kind == "analysis_plan_v2" else
                getattr(getattr(result_payload, "action", None), "model_dump", lambda **_: None)(mode="json")
                if result_payload is not None else None
            ),
            "audit": final,
        })
        per_action.append(item)

    required = [item for item in per_action if item["materializer_required"]]
    first_pass_valid = sum(item["first_pass_valid"] is True for item in required)
    final_valid = sum(bool(item["final_valid"]) for item in required)
    repaired = sum(bool(item["contract_repair_success"]) for item in required)
    model_response_actions = sum(
        item.get("first_model_response_valid") is not None for item in required
    )
    model_responses = sum(int(item.get("model_responses", 0) or 0) for item in required)
    transport_attempts = sum(int(item.get("transport_attempts", 0) or 0) for item in required)
    contract_repairs = sum(int(item.get("contract_repairs", 0) or 0) for item in required)
    attempted_actions = sum(item["provider_http_requests"] > 0 for item in required)
    budget_blocked_actions = sum(item["status"] == "budget_blocked" for item in required)
    api_calls = sum(int(item["provider_http_requests"]) for item in per_action)
    report = {
        "sweep_version": "materializer-contract-sweep-v1",
        "status": "pass" if first_pass_valid / len(required) >= 0.9 and len(required) == 8 else "fail",
        "training_eligible": False,
        "model": model,
        "materializer_origin": "gemini_canary_model",
        "preflight": preflight,
        "catalog_source": str(catalog_path),
        "observation_context_source": str(context_path),
        "actions": per_action,
        "required_action_count": 8,
        "attempted_action_count": attempted_actions,
        "budget_blocked_action_count": budget_blocked_actions,
        "first_pass_valid_count": first_pass_valid,
        "first_model_response_action_count": model_response_actions,
        "first_model_response_valid_rate": (
            first_pass_valid / model_response_actions if model_response_actions else None
        ),
        "repair_success_count": repaired,
        "contract_repair_count": contract_repairs,
        "model_response_count": model_responses,
        "transport_attempt_count": transport_attempts,
        "final_valid_count": final_valid,
        "first_pass_rate": first_pass_valid / 8,
        "final_valid_rate": final_valid / 8,
        "api_calls_made": api_calls,
        # Sol's metric is calls divided by *all* Actions that require a
        # Materializer, not only the subset that happened to start before a
        # budget stop.  Keep attempted_action_count separately for diagnosis.
        "materializer_amplification": api_calls / len(required) if required else None,
        "gemini_request_budget": {
            "total_limit": budget.total_limit,
            "total_used": budget.total_used,
            "role_limits": dict(budget.role_limits),
            "role_used": dict(budget.role_used),
        },
        "static_audit": "static_audit_snapshot.json",
    }
    _dump(out_dir / "sweep_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts" / "materializer_contract_sweep_4d1i")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--context", type=Path, default=DEFAULT_CONTEXT)
    args = parser.parse_args(argv)
    result = run_sweep(args.output_dir, catalog_path=args.catalog, context_path=args.context)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
