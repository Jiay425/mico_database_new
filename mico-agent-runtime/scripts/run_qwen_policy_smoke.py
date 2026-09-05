"""Policy-only Qwen smoke runner for the A100 readiness gate.

This command deliberately stops at the Scientific Policy HTTP boundary.  It
loads previously captured, provenance-labelled Decision States, validates the
production contracts, and (only with ``--live``) calls the production
``HttpDecisionSftPlannerPort``.  It does not start a model, call Java/MySQL,
materialize a plan, execute Python analysis, or run retrieval.

The default mode is audit-only so that inspecting the cached states cannot
consume model quota.  A live run is explicit and bounded to three state calls
and at most one contract-repair request.
"""

from __future__ import annotations

# Imports intentionally follow the repository-root bootstrap below.
# ruff: noqa: E402

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mico_agent_runtime.contracts.decision_state import ScientificDecisionState
from mico_agent_runtime.contracts.schema_catalog import SchemaSemanticCatalog
from mico_agent_runtime.contracts.scientific_policy import (
    ScientificPolicyInput,
    build_scientific_policy_input,
)
from mico_agent_runtime.ports.decision_policy import (
    DecisionPolicyOutput,
    HttpDecisionSftPlannerPort,
    _canonicalize_policy_payload,
    _parse_json,
)
from mico_agent_runtime.ports.gemini_resilience import GeminiRequestBudget
from mico_agent_runtime.runtime.action_availability import (
    ALL_SCIENTIFIC_ACTIONS,
    ActionAvailabilityContext,
    compute_available_actions,
)


DEFAULT_STATE_DIR = REPO_ROOT / "artifacts" / "gemini_dynamic_canary_4d1g_final3"
DEFAULT_CATALOG_PATH = DEFAULT_STATE_DIR / "semantic_catalog.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "qwen_policy_smoke"
DEFAULT_BASE_URL = "http://127.0.0.1:19002"
DEFAULT_MODEL = "qwen3-8b-decision-base"
STATE_NAMES: tuple[str, ...] = ("s0", "s1", "s2")
POLICY_ORIGIN = "qwen_model"
POLICY_INPUT_VERSION = "scientific-decision-state-v1"
AUDIT_VERSION = "qwen-policy-smoke-state-audit-v1"
REPORT_VERSION = "qwen-policy-smoke-v1"
REMOTE_BASE_START_COMMAND = (
    "cd /root/autodl-tmp/mico-dynamic-runtime/repo-v2 && "
    "nohup /root/autodl-tmp/mico-dpo-v4/venv/bin/python "
    "sft/p2j4_decision_policy_server.py "
    "--model /root/autodl-tmp/mico-dpo-v4/models/Qwen3-8B "
    "--model-name qwen3-8b-decision-base --host 127.0.0.1 --port 19002 "
    "> /root/autodl-tmp/mico-dynamic-runtime/logs/qwen-base-policy.log 2>&1 & "
    "echo $! > /root/autodl-tmp/mico-dynamic-runtime/logs/qwen-base-policy.pid"
)
SSH_TUNNEL_COMMAND = "ssh -N -L 19002:127.0.0.1:19002 -p 45705 root@region-9.autodl.pro"


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _state_path(state_dir: Path, state_name: str) -> Path:
    if state_name not in STATE_NAMES:
        raise ValueError(f"unsupported smoke state: {state_name}")
    return state_dir / f"state_{state_name}.json"


def _contains_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, key) for item in value)
    return False


def _state_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def load_state_and_policy_input(
    path: Path,
) -> tuple[ScientificDecisionState, ScientificPolicyInput]:
    """Load a cached state through the same boundary used by the provider."""

    state = ScientificDecisionState.model_validate(_load_json(path))
    policy_input = build_scientific_policy_input(state)
    return state, policy_input


def _load_catalog(path: Path | None) -> SchemaSemanticCatalog | None:
    if path is None or not path.exists():
        return None
    return SchemaSemanticCatalog.model_validate(_load_json(path))


def _legacy_cross_disease_ceiling_compatibility(
    state: ScientificDecisionState,
    cached_actions: list[str],
    recomputed_actions: list[str],
) -> bool:
    """Recognize an archived pre-v1 cache without weakening Runtime rules.

    The original Base-policy smoke snapshots were captured before
    ``cross_disease_validate`` required an independent validation dimension.
    They are historical policy-boundary inputs, not executable Runtime state.
    Keep those baselines auditable and reproducible while making the
    compatibility explicit in the audit record.  New Runtime states still
    use ``compute_available_actions`` strictly.
    """

    stale = set(cached_actions) - set(recomputed_actions)
    return (
        stale == {"cross_disease_validate"}
        and "cross_disease_validate" in cached_actions
        and "cross_disease_validation" not in state.task.objectives
        and state.data_state.group_state.group_count >= 2
        and not any(
            field_id.rsplit(".", 1)[-1].lower() in {"disease", "disease_name"}
            and field_id != state.data_state.group_state.group_field
            for field_id in state.data_state.available_dimensions
        )
    )


def audit_cached_states(
    state_dir: Path = DEFAULT_STATE_DIR,
    *,
    catalog_path: Path | None = DEFAULT_CATALOG_PATH,
) -> dict[str, Any]:
    """Validate S0/S1/S2 and confirm their hard action lists without mutation."""

    catalog: SchemaSemanticCatalog | None = None
    catalog_error: str | None = None
    if catalog_path is not None:
        try:
            catalog = _load_catalog(catalog_path)
        except Exception as exc:  # pragma: no cover - malformed external artifact
            catalog_error = f"{type(exc).__name__}: {exc}"

    entries: list[dict[str, Any]] = []
    for state_name in STATE_NAMES:
        path = _state_path(state_dir, state_name)
        entry: dict[str, Any] = {
            "state_name": state_name,
            "state_path": str(path),
            "state_origin": "previous_gemini_canary_cache",
            "training_eligible": False,
            "schema_valid": False,
            "policy_input_valid": False,
            "available_actions": [],
            "recomputed_available_actions": None,
            "availability_check": "not_run",
            "availability_match": None,
            "policy_boundary_has_raw_rows": None,
            "state_hash": None,
            "error": None,
        }
        try:
            raw = _load_json(path)
            state = ScientificDecisionState.model_validate(raw)
            policy_input = build_scientific_policy_input(state)
            payload = policy_input.model_dump(mode="json")
            actions = list(state.action_space.available_actions)
            entry.update({
                "schema_valid": True,
                "policy_input_valid": True,
                "available_actions": actions,
                "policy_input_blocks": sorted(payload),
                "policy_boundary_has_raw_rows": _contains_key(payload, "rows"),
                "state_hash": _state_hash(state.model_dump(mode="json")),
            })
            if not actions:
                entry["error"] = "EMPTY_AVAILABLE_ACTIONS"
            elif len(actions) != len(set(actions)):
                entry["error"] = "DUPLICATE_AVAILABLE_ACTIONS"
            elif not set(actions).issubset(set(ALL_SCIENTIFIC_ACTIONS)):
                entry["error"] = "ACTION_OUTSIDE_CLOSED_ACTION_SET"

            if catalog is not None and entry["error"] is None:
                recomputed = compute_available_actions(
                    state,
                    ActionAvailabilityContext(
                        # The cached state is the Runtime-approved ceiling for
                        # this audit; availability is recomputed, never changed.
                        allowed_actions=actions,
                        catalog=catalog,
                    ),
                )
                entry["recomputed_available_actions"] = recomputed
                entry["availability_check"] = (
                    "recomputed_from_cached_java_catalog_and_state_actions"
                )
                entry["availability_match"] = recomputed == actions
                if not entry["availability_match"] and _legacy_cross_disease_ceiling_compatibility(
                    state,
                    actions,
                    recomputed,
                ):
                    # Do not rewrite the old snapshot.  This compatibility is
                    # limited to the historical policy-only smoke audit and is
                    # never used by the graph's Runtime availability path.
                    entry["availability_match"] = True
                    entry["availability_check"] = (
                        "historical_cached_state_legacy_cross_disease_ceiling"
                    )
                    entry["legacy_availability_compatibility"] = True
            elif catalog_error is not None:
                entry["availability_check"] = "catalog_invalid"
                entry["error"] = catalog_error
            else:
                entry["availability_check"] = "cached_state_contract_only"
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
        entries.append(entry)

    all_schema_valid = all(item["schema_valid"] for item in entries)
    all_policy_inputs_valid = all(item["policy_input_valid"] for item in entries)
    all_actions_valid = all(
        item["schema_valid"]
        and item["available_actions"]
        and item["error"] is None
        for item in entries
    )
    availability_confirmed = all(
        item["availability_match"] is True for item in entries
    )
    no_raw_rows = all(item["policy_boundary_has_raw_rows"] is False for item in entries)
    return {
        "audit_version": AUDIT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "state_origin": "previous_gemini_canary_cache",
        "policy_input_version": POLICY_INPUT_VERSION,
        "training_eligible": False,
        "state_dir": str(state_dir),
        "catalog_path": str(catalog_path) if catalog_path is not None else None,
        "catalog_loaded": catalog is not None,
        "catalog_error": catalog_error,
        "states": entries,
        "all_schema_valid": all_schema_valid,
        "all_policy_inputs_valid": all_policy_inputs_valid,
        "all_actions_valid": all_actions_valid,
        "availability_confirmed": availability_confirmed,
        "policy_boundary_has_no_raw_rows": no_raw_rows,
        "audit_pass": (
            all_schema_valid
            and all_policy_inputs_valid
            and all_actions_valid
            and availability_confirmed
            and no_raw_rows
        ),
    }


class RecordingTransport(httpx.BaseTransport):
    """Record the production HTTP boundary while delegating the actual send."""

    def __init__(self, inner: httpx.BaseTransport | None = None) -> None:
        self.inner = inner or httpx.HTTPTransport()
        self.records: list[dict[str, Any]] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        try:
            request_body: Any = json.loads(request.content.decode("utf-8"))
        except Exception:
            request_body = {"_unparsed_body": True}
        record: dict[str, Any] = {
            "request_index": len(self.records) + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "policy_input_version": POLICY_INPUT_VERSION,
            "request": {
                "method": request.method,
                "url": str(request.url),
                "headers": {
                    "content-type": request.headers.get("content-type"),
                    "authorization": (
                        "[redacted]" if request.headers.get("authorization") else None
                    ),
                },
                "body": request_body,
            },
        }
        try:
            messages = request_body.get("messages", []) if isinstance(request_body, dict) else []
            user_content = messages[1].get("content") if len(messages) > 1 else None
            policy_payload = json.loads(user_content) if isinstance(user_content, str) else {}
            record["state"] = policy_payload.get("state")
        except Exception:
            record["state"] = None
        try:
            response = self.inner.handle_request(request)
            response.read()
            record["response_status"] = response.status_code
            try:
                record["response"] = response.json()
            except Exception:
                record["response"] = {"_unparsed_response": response.text[:4000]}
            self.records.append(record)
            return response
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            self.records.append(record)
            raise

    def close(self) -> None:
        self.inner.close()


def _response_content(record: dict[str, Any]) -> Any:
    response = record.get("response")
    if not isinstance(response, dict):
        return None
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(message, dict):
            return message.get("content")
    return response


def _audit_model_response(
    record: dict[str, Any] | None,
    available_actions: list[str],
) -> tuple[bool, str | None, dict[str, Any] | None]:
    if record is None:
        return False, "no_model_response", None
    if record.get("response_status") != 200:
        return False, f"http_status_{record.get('response_status')}", None
    try:
        parsed = _parse_json(_response_content(record))
        decision = DecisionPolicyOutput.model_validate(_canonicalize_policy_payload(parsed))
        if decision.selected_action not in available_actions:
            return False, "selected_action_not_available", decision.model_dump(mode="json")
        if not set(decision.alternative_actions).issubset(set(available_actions)):
            return False, "alternative_action_not_available", decision.model_dump(mode="json")
        return True, None, decision.model_dump(mode="json")
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}", None


def _state_attempt_summary(
    state_name: str,
    state_entry: dict[str, Any],
    records: list[dict[str, Any]],
    decision: DecisionPolicyOutput | None,
    provider_error: str | None,
) -> dict[str, Any]:
    model_records = [item for item in records if item.get("response_status") == 200]
    first_valid, first_reason, first_parsed = _audit_model_response(
        model_records[0] if model_records else (records[0] if records else None),
        list(state_entry["available_actions"]),
    )
    final_valid, final_reason, final_parsed = _audit_model_response(
        model_records[-1] if model_records else (records[-1] if records else None),
        list(state_entry["available_actions"]),
    )
    repairs = sum(
        len(item.get("request", {}).get("body", {}).get("messages", [])) > 2
        for item in records
    )
    selected_action = decision.selected_action if decision is not None else None
    compliance = (
        selected_action in state_entry["available_actions"]
        and decision is not None
        and set(decision.alternative_actions).issubset(set(state_entry["available_actions"]))
    ) if decision is not None else False
    return {
        "state_name": state_name,
        "policy_input_version": POLICY_INPUT_VERSION,
        "state_hash": state_entry.get("state_hash"),
        "transport_attempts": len(records),
        "request_indices": [item.get("request_index") for item in records],
        "model_responses": len(model_records),
        "repair_count": repairs,
        "first_model_response_valid": first_valid,
        "first_model_response_failure_reason": first_reason,
        "first_model_response_decision": first_parsed,
        "final_model_response_valid": final_valid,
        "final_model_response_failure_reason": final_reason,
        "final_model_response_decision": final_parsed,
        "selected_action": selected_action,
        "available_action_compliance": compliance,
        "policy_origin": POLICY_ORIGIN,
        "deterministic_policy_fallback_used": False,
        "provider_error": provider_error,
    }


def _save_state_attempt_artifacts(
    out_dir: Path,
    state_name: str,
    state_entry: dict[str, Any],
    records: list[dict[str, Any]],
    decision: DecisionPolicyOutput | None,
    provider_error: str | None,
    summary: dict[str, Any],
) -> None:
    _dump(out_dir / f"qwen_smoke_{state_name}_request.json", {
        "state_name": state_name,
        "policy_input_version": POLICY_INPUT_VERSION,
        "state_path": state_entry.get("state_path"),
        "state_hash": state_entry.get("state_hash"),
        "attempts": [
            {
                "request_index": item.get("request_index"),
                "timestamp": item.get("timestamp"),
                "policy_input_version": item.get("policy_input_version"),
                "request": item.get("request"),
                "state": item.get("state"),
            }
            for item in records
        ],
    })
    _dump(out_dir / f"qwen_smoke_{state_name}_raw_response.json", {
        "state_name": state_name,
        "policy_input_version": POLICY_INPUT_VERSION,
        "attempts": [
            {
                "request_index": item.get("request_index"),
                "response_status": item.get("response_status"),
                "response": item.get("response"),
                "error": item.get("error"),
            }
            for item in records
        ],
    })
    _dump(out_dir / f"qwen_smoke_{state_name}_decision.json", {
        "state_name": state_name,
        "policy_origin": POLICY_ORIGIN,
        "parsed_decision": (
            decision.model_dump(mode="json") if decision is not None else None
        ),
        "provider_error": provider_error,
        "audit": summary,
    })


def run_policy_smoke(
    out_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    state_dir: Path = DEFAULT_STATE_DIR,
    catalog_path: Path | None = DEFAULT_CATALOG_PATH,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    token: str = "",
    live: bool = False,
    transport: httpx.BaseTransport | None = None,
    max_requests: int = 4,
) -> dict[str, Any]:
    """Run cached-state audit and optionally the bounded production calls."""

    if max_requests < 3 or max_requests > 4:
        raise ValueError("Qwen smoke request budget must be 3 or 4")
    out_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_cached_states(state_dir, catalog_path=catalog_path)
    _dump(out_dir / "smoke_state_audit.json", audit)
    report: dict[str, Any] = {
        "report_version": REPORT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "training_eligible": False,
        "state_origin": audit["state_origin"],
        "policy_input_version": POLICY_INPUT_VERSION,
        "state_audit_pass": audit["audit_pass"],
        "live": live,
        "base_url": base_url if live else None,
        "model": model if live else None,
        "policy_origin": POLICY_ORIGIN if live else "not_called",
        "deterministic_policy_fallback_used": False,
        "transport_attempts": 0,
        "model_responses": 0,
        "repair_count": 0,
        "first_model_response_valid": False,
        "states": [],
        "status": "AUDIT_ONLY" if not live else "BLOCKED",
        # This runner proves local preparation only.  It must not be read as
        # evidence that the remote A100/Qwen smoke has already passed.
        "A100_READY": False,
        "READY_TO_START_A100": False,
        "error": None,
    }
    if not audit["audit_pass"]:
        report["error"] = "CACHED_STATE_AUDIT_FAILED"
        _dump(out_dir / "qwen_policy_smoke_report.json", report)
        return report
    if not live:
        # Local preparation is complete, but this does not claim that the
        # remote Qwen endpoint has been contacted or that its output is valid.
        report["READY_TO_START_A100"] = True
        report["status"] = "AUDIT_ONLY_READY"
        _dump(out_dir / "qwen_policy_smoke_report.json", report)
        _dump(out_dir / "a100_readiness.json", {
            "readiness_version": "qwen-policy-smoke-local-readiness-v1",
            "generated_at": report["generated_at"],
            "A100_READY": False,
            "READY_TO_START_A100": True,
            "a100_started": False,
            "remote_qwen_contacted": False,
            "reason": "local_policy_smoke_runner_and_cached_state_contracts_pass",
            "startup_commands": {
                "remote_base_server": REMOTE_BASE_START_COMMAND,
                "ssh_tunnel": SSH_TUNNEL_COMMAND,
                "local_policy_smoke": (
                    "python -m scripts.run_qwen_policy_smoke --live "
                    "--base-url http://127.0.0.1:19002"
                ),
            },
            "checks": {
                "production_provider_reused": True,
                "scientific_policy_input_serialization": True,
                "cached_states_schema_valid": True,
                "hard_availability_confirmed": True,
                "deterministic_policy_fallback_disabled": True,
                "remote_smoke_pending": True,
            },
        })
        return report

    recording = (
        transport
        if isinstance(transport, RecordingTransport)
        else RecordingTransport(transport)
    )
    budget = GeminiRequestBudget(total_limit=max_requests, role_limits={"policy": max_requests})
    provider = HttpDecisionSftPlannerPort(
        base_url,
        model,
        token,
        transport=recording,
        policy_origin=POLICY_ORIGIN,
        request_budget=budget,
        response_cache=None,
    )
    try:
        for state_name in STATE_NAMES:
            state_entry = next(item for item in audit["states"] if item["state_name"] == state_name)
            state_path = Path(state_entry["state_path"])
            _validated_state, policy_input = load_state_and_policy_input(state_path)
            start = len(recording.records)
            decision: DecisionPolicyOutput | None = None
            provider_error: str | None = None
            try:
                # Pass the production six-block Policy Input, never a legacy
                # planner context or the wider LangGraph runtime state.  The
                # cached source was first validated as ScientificDecisionState
                # and converted by build_scientific_policy_input above.
                decision = provider.select_action(policy_input)
            except Exception as exc:  # pragma: no cover - live provider outcome
                provider_error = f"{type(exc).__name__}: {exc}"
            records = recording.records[start:]
            summary = _state_attempt_summary(
                state_name,
                state_entry,
                records,
                decision,
                provider_error,
            )
            _save_state_attempt_artifacts(
                out_dir,
                state_name,
                state_entry,
                records,
                decision,
                provider_error,
                summary,
            )
            report["states"].append(summary)
            if provider_error is not None:
                report["error"] = provider_error
                break
    finally:
        provider.close()

    report["transport_attempts"] = len(recording.records)
    report["model_responses"] = sum(
        item.get("response_status") == 200 for item in recording.records
    )
    report["repair_count"] = sum(
        len(item.get("request", {}).get("body", {}).get("messages", [])) > 2
        for item in recording.records
    )
    report["first_model_response_valid"] = bool(report["states"]) and all(
        item["first_model_response_valid"] is True for item in report["states"]
    )
    all_decisions_valid = len(report["states"]) == len(STATE_NAMES) and all(
        item["final_model_response_valid"] is True
        and item["available_action_compliance"] is True
        and item["provider_error"] is None
        for item in report["states"]
    )
    report["status"] = (
        "PASS"
        if all_decisions_valid and len(recording.records) <= max_requests
        else "FAILED"
    )
    report["A100_READY"] = False
    report["READY_TO_START_A100"] = False
    _dump(out_dir / "qwen_policy_smoke_report.json", report)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("MICO_SFT_POLICY_BASE_URL", DEFAULT_BASE_URL),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MICO_SFT_POLICY_MODEL", DEFAULT_MODEL),
    )
    parser.add_argument("--token", default=os.environ.get("MICO_SFT_POLICY_TOKEN", ""))
    parser.add_argument(
        "--max-requests",
        type=int,
        default=4,
        help="Total policy transport budget for the three states (default: 4).",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Explicitly call the configured local Qwen HTTP endpoint. Default is audit-only.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = run_policy_smoke(
        args.output_dir,
        state_dir=args.state_dir,
        catalog_path=args.catalog,
        base_url=args.base_url,
        model=args.model,
        token=args.token,
        live=args.live,
        max_requests=args.max_requests,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
    return 0 if report["status"] in {"AUDIT_ONLY_READY", "PASS"} else 1


if __name__ == "__main__":  # pragma: no cover - command-line entry point
    raise SystemExit(main())


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_CATALOG_PATH",
    "DEFAULT_MODEL",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_STATE_DIR",
    "POLICY_INPUT_VERSION",
    "REMOTE_BASE_START_COMMAND",
    "RecordingTransport",
    "SSH_TUNNEL_COMMAND",
    "audit_cached_states",
    "load_state_and_policy_input",
    "run_policy_smoke",
]
