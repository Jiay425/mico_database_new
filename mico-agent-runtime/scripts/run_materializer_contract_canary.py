"""4D-1H: focused Gemini Materializer contract canary.

This runner deliberately does not invoke Task Understanding, Scientific Policy,
Java reads, or statistical execution.  It loads one real, previously captured
``stratified_analysis`` context from the canary and exercises only the
AnalysisPlan-v2 Materializer boundary.  A service preflight is still required
before the first Gemini request so a dead Java/MySQL chain cannot spend model
quota.

All output is test provenance and is marked ``training_eligible=false``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mico_agent_runtime.contracts.generated_analysis import AnalysisPlannerContext
from mico_agent_runtime.ports.gemini_resilience import (
    GeminiRequestBudget,
    GeminiRequestBudgetExceeded,
)
from mico_agent_runtime.ports.research_planner import HttpResearchPlannerPort
from scripts.check_scientific_chain_services import run_preflight
from scripts.run_gemini_dynamic_canary import (
    GEMINI_BASE_URL,
    GEMINI_MODEL,
    RecordingTransport,
    _gemini_token,
    _materializer_call_audit,
    _materializer_payload_records,
)


DEFAULT_CONTEXT = (
    REPO_ROOT
    / "artifacts"
    / "gemini_dynamic_canary_4d1g_final3"
    / "materializer_request_007.json"
)


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _load_context(path: Path) -> AnalysisPlannerContext:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("body"), dict):
        messages = payload["body"].get("messages", [])
        if len(messages) < 2:
            raise ValueError("context request has no user message")
        payload = json.loads(messages[1]["content"])
    return AnalysisPlannerContext.model_validate(payload)


def _save_transport_records(out_dir: Path, transport: RecordingTransport) -> None:
    for record in transport.records:
        index = int(record["request_index"])
        _dump(out_dir / f"materializer_request_{index:03d}.json", record["request"])
        _dump(
            out_dir / f"materializer_response_{index:03d}.json",
            {
                "response_status": record.get("response_status"),
                "response": record.get("response"),
                "error": record.get("error"),
            },
        )


def run_canary(
    out_dir: Path,
    *,
    context_path: Path = DEFAULT_CONTEXT,
    repetitions: int = 5,
) -> dict[str, Any]:
    if repetitions < 1 or repetitions > 10:
        raise ValueError("repetitions must be between 1 and 10")
    out_dir.mkdir(parents=True, exist_ok=True)

    # This check performs no model calls.  It is intentionally before token
    # lookup and provider construction, and therefore before any Gemini quota
    # can be consumed.
    preflight = asyncio.run(run_preflight())
    _dump(out_dir / "preflight.json", preflight)
    if preflight.get("status") != "pass":
        result = {
            "status": "blocked",
            "training_eligible": False,
            "reason": "CHAIN_PREFLIGHT_FAILED",
            "gemini_calls_made": 0,
            "preflight": preflight,
        }
        _dump(out_dir / "report.json", result)
        return result

    context = _load_context(context_path)
    if context.actionName != "stratified_analysis":
        raise ValueError("focused canary requires actionName=stratified_analysis")
    _dump(out_dir / "context.json", context.model_dump(mode="json"))

    token = _gemini_token()
    model = os.environ.get("MICO_GEMINI_MODEL", GEMINI_MODEL).strip() or GEMINI_MODEL
    base_url = os.environ.get("MICO_GEMINI_BASE_URL", GEMINI_BASE_URL).strip() or GEMINI_BASE_URL
    transport = RecordingTransport("materializer_contract_canary")
    # Five independent first-pass probes are allowed, with only a small repair
    # headroom.  No response cache is used: each repetition must measure the
    # actual provider contract rather than a cached prior answer.
    budget = GeminiRequestBudget(
        total_limit=max(repetitions + 3, 5),
        role_limits={"materializer": max(repetitions + 3, 5)},
    )
    planner = HttpResearchPlannerPort(
        base_url,
        model,
        token,
        transport=transport,
        materializer_origin="gemini_canary_model",
        request_budget=budget,
        response_cache=None,
    )
    results: list[dict[str, Any]] = []
    try:
        for index in range(1, repetitions + 1):
            try:
                materialized = planner.generate_typed_analysis(context)
                results.append({
                    "probe": index,
                    "status": "returned",
                    "mode": materialized.mode,
                    "analysis_type": materialized.plan.analysis_type,
                    "source_observation_count": len(materialized.plan.source_observation_ids),
                    "feature_field": materialized.plan.feature_field,
                    "stratify_by": materialized.plan.stratify_by,
                    "repair_codes": materialized.repairCodes,
                })
            except GeminiRequestBudgetExceeded as exc:
                results.append({
                    "probe": index,
                    "status": "budget_exhausted",
                    "error": str(exc),
                })
                break
            except Exception as exc:  # pragma: no cover - exercised by real provider
                results.append({
                    "probe": index,
                    "status": "failed",
                    "error": f"{type(exc).__name__}:{exc}",
                })
    finally:
        planner.close()

    _save_transport_records(out_dir, transport)
    records = _materializer_payload_records(transport)
    audit = _materializer_call_audit(records)
    _dump(out_dir / "materializer_audit.json", audit)

    first_pass_valid = sum(
        bool(item.get("first_pass_success"))
        and item.get("contract_kind") == "typed_analysis_plan"
        for item in audit
    )
    final_valid = sum(
        item.get("final_failure_reason") is None
        and item.get("final_contract_kind") == "typed_analysis_plan"
        for item in audit
    )
    result = {
        "status": "pass" if first_pass_valid >= min(4, repetitions) else "fail",
        "training_eligible": False,
        "task_understanding_origin": "cached_real_state",
        "policy_origin": "not_invoked",
        "materializer_origin": "gemini_canary_model",
        "action_name": context.actionName,
        "expected_analysis_type": "stratified_comparison",
        "requested_repetitions": repetitions,
        "provider_http_requests": len(transport.records),
        "provider_results": results,
        "first_pass_valid": first_pass_valid,
        "final_valid_after_repairs": final_valid,
        "first_pass_rate": first_pass_valid / repetitions,
        "final_valid_rate": final_valid / repetitions,
        "gemini_calls_made": len(transport.records),
        "gemini_request_budget": {
            "total_limit": budget.total_limit,
            "total_used": budget.total_used,
            "role_limits": dict(budget.role_limits),
            "role_used": dict(budget.role_used),
        },
        "preflight": preflight,
        "context_source": str(context_path),
    }
    _dump(out_dir / "report.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts" / "materializer_contract_canary_4d1h")
    parser.add_argument("--context", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args(argv)
    try:
        result = run_canary(
            args.output_dir,
            context_path=args.context,
            repetitions=args.repetitions,
        )
    except Exception as exc:
        result = {
            "status": "error",
            "training_eligible": False,
            "error": f"{type(exc).__name__}:{exc}",
        }
        _dump(args.output_dir / "report.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
