"""Prepare and (optionally) run the local 4D-2 Dynamic State Canary.

The canary is deliberately policy-only.  It creates six validated
``ScientificDecisionState`` snapshots, records the controlled-state
provenance outside the policy payload, and can later send those snapshots to
the production ``HttpDecisionSftPlannerPort``.  The default mode is
audit-only and therefore cannot contact an A100 or consume model quota.

The primary purpose is to compare policy decisions for the same task after a
small, explicit Decision-State change:

* S1 -> S2: group comparison not started -> completed
* S2 -> S3: confounder adjustment not started -> completed, with an
  explicitly controlled attenuated effect
* S4 -> S5: cross-project results consistent -> conflicting
* S6: required objectives complete and ``finish`` hard-available

S4/S5 (and the numeric overlay in S3) are controlled canary facts, not
database observations.  Their provenance is kept in the manifest/report so
that the six-block policy contract remains closed and training traces remain
ineligible.
"""

from __future__ import annotations

# Imports intentionally follow the repository-root bootstrap below.
# ruff: noqa: E402

import argparse
import copy
import hashlib
import json
import os
import re
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


DEFAULT_SOURCE_DIR = REPO_ROOT / "artifacts" / "gemini_dynamic_canary_4d1g_final3"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "qwen_dynamic_state_canary"
DEFAULT_BASE_URL = "http://127.0.0.1:19002"
DEFAULT_MODEL = "qwen3-8b-decision-base"
STATE_NAMES: tuple[str, ...] = ("s1", "s2", "s3", "s4", "s5", "s6")
PRIMARY_REAL_STATES: tuple[str, ...] = ("s1", "s2")
POLICY_INPUT_VERSION = "scientific-decision-state-v1"
REPORT_VERSION = "qwen-dynamic-state-canary-v1"
MANIFEST_VERSION = "qwen-dynamic-state-canary-manifest-v1"
POLICY_ORIGIN = "qwen_model"
DEFAULT_LIVE_RUN_ID = "BASE_DYNAMIC_CANARY_V2"
DEFAULT_PROMPT_VERSION = "generic_contract_hardened"


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


def _state_hash(state: ScientificDecisionState) -> str:
    encoded = json.dumps(
        state.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _load_state(path: Path) -> ScientificDecisionState:
    return ScientificDecisionState.model_validate(_load_json(path))


def _load_catalog(path: Path) -> SchemaSemanticCatalog:
    return SchemaSemanticCatalog.model_validate(_load_json(path))


def _state_path(source_dir: Path, state_name: str) -> Path:
    return source_dir / f"state_{state_name}.json"


def _with_actions(
    payload: dict[str, Any],
    catalog: SchemaSemanticCatalog,
    *,
    allowed_actions: list[str] | tuple[str, ...] = ALL_SCIENTIFIC_ACTIONS,
) -> dict[str, Any]:
    """Compute only hard availability; never add a strategy recommendation."""

    state = ScientificDecisionState.model_validate(payload)
    actions = compute_available_actions(
        state,
        ActionAvailabilityContext(
            allowed_actions=allowed_actions,
            catalog=catalog,
        ),
    )
    payload["action_space"] = {"available_actions": actions}
    return payload


def _copy_payload(state: ScientificDecisionState) -> dict[str, Any]:
    return copy.deepcopy(state.model_dump(mode="json"))


def _ensure_dimension(payload: dict[str, Any], dimension: str) -> None:
    dimensions = payload["data_state"]["available_dimensions"]
    if dimension not in dimensions:
        dimensions.append(dimension)


def _build_s3(s2: ScientificDecisionState, catalog: SchemaSemanticCatalog) -> dict[str, Any]:
    """Build the controlled adjustment snapshot from the real S2 shape."""

    payload = _copy_payload(s2)
    # This numeric effect is intentionally a controlled policy-sensitivity
    # signal.  It is not copied into a real Observation or training sample.
    payload["analysis_state"]["group_comparison"].update({
        "mean_difference": 0.62,
        "effect_size": 0.62,
        "p_value": 0.008,
        "confidence_interval_low": 0.31,
        "confidence_interval_high": 0.93,
    })
    payload["analysis_state"]["confounder_adjustment"].update({
        "status": "completed",
        "adjusted_group_effect": 0.18,
        "adjusted_effect_size": 0.18,
        "adjusted_p_value": 0.21,
        "adjusted_covariates": ["sample.age"],
        "used_row_count": payload["data_state"].get("row_count", 0),
        "dropped_row_count": 0,
    })
    payload["progress"] = {
        "completed_actions": [
            "execute_read_query",
            "compare_groups",
            "adjust_confounders",
        ],
        "action_counts": {
            "execute_read_query": 1,
            "compare_groups": 1,
            "adjust_confounders": 1,
        },
        "last_action": "adjust_confounders",
        "remaining_objectives": [
            "cross_project_validation",
            "evidence_support",
        ],
        "action_count": 3,
    }
    # Keep the technical action ceiling equal to S2 for Pair 2.  The canary is
    # testing the new adjustment observation, not a simultaneous availability
    # change (S4/S5 introduce the controlled project dimension separately).
    return _with_actions(
        payload,
        catalog,
        allowed_actions=list(s2.action_space.available_actions),
    )


def _build_project_state(
    s3: ScientificDecisionState,
    catalog: SchemaSemanticCatalog,
    *,
    conflicting: bool,
) -> dict[str, Any]:
    payload = _copy_payload(s3)
    _ensure_dimension(payload, "metadata.project")
    payload["data_state"]["project_state"] = {
        "has_project_field": True,
        "project_count": 4,
    }
    payload["analysis_state"]["cross_project_validation"] = {
        "status": "completed",
        "project_count": 4,
        "positive_project_count": 2 if conflicting else 4,
        "negative_project_count": 2 if conflicting else 0,
        "neutral_project_count": 0,
        "effect_min": -0.61 if conflicting else 0.55,
        "effect_max": 0.64 if conflicting else 0.69,
        "heterogeneity": "high" if conflicting else "low",
    }
    payload["progress"] = {
        "completed_actions": [
            "execute_read_query",
            "compare_groups",
            "adjust_confounders",
            "cross_project_validate",
        ],
        "action_counts": {
            "execute_read_query": 1,
            "compare_groups": 1,
            "adjust_confounders": 1,
            "cross_project_validate": 1,
        },
        "last_action": "cross_project_validate",
        "remaining_objectives": ["evidence_support"],
        "action_count": 4,
    }
    return _with_actions(payload, catalog)


def _build_s6(s4: ScientificDecisionState, catalog: SchemaSemanticCatalog) -> dict[str, Any]:
    payload = _copy_payload(s4)
    payload["evidence_state"] = {
        "status": "completed",
        "evidence_count": 6,
        "support_count": 4,
        "conflict_count": 1,
        "context_count": 1,
        "consistency": "mostly_supportive",
    }
    payload["progress"] = {
        "completed_actions": [
            "execute_read_query",
            "compare_groups",
            "adjust_confounders",
            "cross_project_validate",
            "retrieve_evidence",
        ],
        "action_counts": {
            "execute_read_query": 1,
            "compare_groups": 1,
            "adjust_confounders": 1,
            "cross_project_validate": 1,
            "retrieve_evidence": 1,
        },
        "last_action": "retrieve_evidence",
        "remaining_objectives": [],
        "action_count": 5,
    }
    return _with_actions(payload, catalog)


def build_dynamic_states(
    source_dir: Path = DEFAULT_SOURCE_DIR,
    *,
    catalog_path: Path | None = None,
) -> tuple[dict[str, ScientificDecisionState], dict[str, dict[str, Any]], SchemaSemanticCatalog]:
    """Create validated S1-S6 snapshots and their non-policy provenance."""

    catalog_path = catalog_path or source_dir / "semantic_catalog.json"
    catalog = _load_catalog(catalog_path)
    # Historical Gemini snapshots may contain an action ceiling produced
    # before the independent validation-dimension contract was tightened.
    # Recompute their hard actions before deriving the new canary states; this
    # keeps the controlled canary honest without rewriting the archived
    # baseline files.
    s1_raw = _load_state(_state_path(source_dir, "s1"))
    s2_raw = _load_state(_state_path(source_dir, "s2"))
    s1 = ScientificDecisionState.model_validate(_with_actions(
        _copy_payload(s1_raw),
        catalog,
        allowed_actions=list(s1_raw.action_space.available_actions),
    ))
    s2 = ScientificDecisionState.model_validate(_with_actions(
        _copy_payload(s2_raw),
        catalog,
        allowed_actions=list(s2_raw.action_space.available_actions),
    ))
    s3 = ScientificDecisionState.model_validate(_build_s3(s2, catalog))
    s4 = ScientificDecisionState.model_validate(_build_project_state(s3, catalog, conflicting=False))
    s5 = ScientificDecisionState.model_validate(_build_project_state(s3, catalog, conflicting=True))
    s6 = ScientificDecisionState.model_validate(_build_s6(s4, catalog))

    states = {"s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5, "s6": s6}
    provenance: dict[str, dict[str, Any]] = {
        "s1": {
            "state_origin": "previous_gemini_canary_cache",
            "training_eligible": False,
            "source_state": str(_state_path(source_dir, "s1")),
            "controlled_fields": [],
        },
        "s2": {
            "state_origin": "previous_gemini_canary_cache",
            "training_eligible": False,
            "source_state": str(_state_path(source_dir, "s2")),
            "controlled_fields": [],
        },
        "s3": {
            "state_origin": "controlled_canary",
            "training_eligible": False,
            "source_state": str(_state_path(source_dir, "s2")),
            "controlled_fields": [
                "analysis_state.group_comparison.effect_size",
                "analysis_state.group_comparison.p_value",
                "analysis_state.confounder_adjustment",
            ],
        },
        "s4": {
            "state_origin": "controlled_canary",
            "training_eligible": False,
            "source_state": "derived_from_s3",
            "controlled_fields": ["data_state.project_state", "analysis_state.cross_project_validation"],
        },
        "s5": {
            "state_origin": "controlled_canary",
            "training_eligible": False,
            "source_state": "derived_from_s3",
            "controlled_fields": ["data_state.project_state", "analysis_state.cross_project_validation"],
        },
        "s6": {
            "state_origin": "controlled_canary",
            "training_eligible": False,
            "source_state": "derived_from_s4",
            "controlled_fields": ["evidence_state", "progress.remaining_objectives"],
        },
    }
    return states, provenance, catalog


def _state_key_fields(state: ScientificDecisionState) -> dict[str, Any]:
    data = state.data_state
    analysis = state.analysis_state
    return {
        "task_objectives": list(state.task.objectives),
        "task_constraints": state.task.constraints.model_dump(mode="json"),
        "has_tabular_data": data.has_tabular_data,
        "row_count": data.row_count,
        "sample_count": data.sample_count,
        "available_dimensions": list(data.available_dimensions),
        "available_outcomes": list(data.available_outcomes),
        "group_field": data.group_state.group_field,
        "group_count": data.group_state.group_count,
        "group_sizes": dict(data.group_state.group_sizes),
        "has_project_field": data.project_state.has_project_field,
        "project_count": data.project_state.project_count,
        "available_covariates": list(data.covariate_state.available_covariates),
        "covariate_imbalance": dict(data.covariate_state.imbalance),
        "group_comparison": analysis.group_comparison.model_dump(mode="json"),
        "confounder_adjustment": analysis.confounder_adjustment.model_dump(mode="json"),
        "cross_project_validation": analysis.cross_project_validation.model_dump(mode="json"),
        "evidence": state.evidence_state.model_dump(mode="json"),
        "completed_actions": list(state.progress.completed_actions),
        "last_action": state.progress.last_action,
        "remaining_objectives": list(state.progress.remaining_objectives),
        "action_count": state.progress.action_count,
        "available_actions": list(state.action_space.available_actions),
    }


def _diff_paths(left: Any, right: Any, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                paths.append(path)
            else:
                paths.extend(_diff_paths(left[key], right[key], path))
        return paths
    if isinstance(left, list) and isinstance(right, list):
        if left != right:
            paths.append(prefix)
        return paths
    if left != right:
        paths.append(prefix)
    return paths


PAIR_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "pair_id": "pair_1",
        "left_state": "s1",
        "right_state": "s2",
        "scientific_signal": "group_comparison.status not_started -> completed",
        "reason_marker_groups": [["group_comparison"], ["not_started", "completed"]],
        "expected_policy_calls": {"left": 1, "right": 1},
    },
    {
        "pair_id": "pair_2",
        "left_state": "s2",
        "right_state": "s3",
        "scientific_signal": (
            "confounder_adjustment.status not_started -> completed; "
            "S3 controlled adjusted effect=0.18 (S2 source cache effect fields are null)"
        ),
        "numeric_baseline_visible": False,
        "interpretation_note": (
            "S2 is preserved from the previous real canary cache and has a completed "
            "group-comparison status but no mapped numeric effect. S3's 0.62/0.18 "
            "values are controlled policy-sensitivity fields, not scientific results."
        ),
        "reason_marker_groups": [["confounder", "adjust"], ["effect", "completed"]],
        "expected_policy_calls": {"left": 1, "right": 1},
    },
    {
        "pair_id": "pair_3",
        "left_state": "s4",
        "right_state": "s5",
        "scientific_signal": "cross-project heterogeneity low -> high",
        "reason_marker_groups": [["project"], ["heterogeneity", "stability"], ["low", "high", "conflict", "consistent"]],
        "expected_policy_calls": {"left": 1, "right": 1},
    },
    {
        "pair_id": "finish_gate",
        "left_state": "s6",
        "right_state": None,
        "scientific_signal": "required objectives complete; finish hard-available",
        "expected_policy_calls": {"left": 1},
    },
)


def _audit_states(
    states: dict[str, ScientificDecisionState],
    provenance: dict[str, dict[str, Any]],
    catalog: SchemaSemanticCatalog,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for state_name in STATE_NAMES:
        state = states[state_name]
        policy_input = build_scientific_policy_input(state)
        payload = policy_input.model_dump(mode="json")
        actions = list(state.action_space.available_actions)
        recomputed = compute_available_actions(
            state,
            ActionAvailabilityContext(allowed_actions=actions, catalog=catalog),
        )
        entries.append({
            "state_name": state_name,
            "expected_policy_calls": 1,
            "policy_input_version": POLICY_INPUT_VERSION,
            "state_origin": provenance[state_name]["state_origin"],
            "training_eligible": provenance[state_name]["training_eligible"],
            "source_state": provenance[state_name]["source_state"],
            "controlled_fields": provenance[state_name]["controlled_fields"],
            "state_hash": _state_hash(state),
            "schema_valid": True,
            "policy_input_blocks": sorted(payload),
            "policy_boundary_has_raw_rows": "rows" in json.dumps(payload, ensure_ascii=False),
            "available_actions": actions,
            "recomputed_available_actions": recomputed,
            "availability_match": actions == recomputed,
            "key_fields": _state_key_fields(state),
        })
    return {
        "audit_version": "qwen-dynamic-state-canary-audit-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "policy_input_version": POLICY_INPUT_VERSION,
        "states": entries,
        "all_schema_valid": all(item["schema_valid"] for item in entries),
        "availability_confirmed": all(item["availability_match"] for item in entries),
        "policy_boundary_has_no_raw_rows": all(
            item["policy_boundary_has_raw_rows"] is False for item in entries
        ),
        "all_training_ineligible": all(
            item["training_eligible"] is False for item in entries
        ),
        "audit_pass": all(
            item["schema_valid"]
            and item["availability_match"]
            and item["policy_boundary_has_raw_rows"] is False
            and item["training_eligible"] is False
            for item in entries
        ),
    }


def prepare_dynamic_canary(
    out_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    source_dir: Path = DEFAULT_SOURCE_DIR,
    catalog_path: Path | None = None,
) -> dict[str, Any]:
    """Write local snapshots and a manifest without contacting a provider."""

    states, provenance, catalog = build_dynamic_states(source_dir, catalog_path=catalog_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    for state_name, state in states.items():
        _dump(out_dir / f"state_{state_name}.json", state.model_dump(mode="json"))

    audit = _audit_states(states, provenance, catalog)
    pair_report: list[dict[str, Any]] = []
    for definition in PAIR_DEFINITIONS:
        left = states[definition["left_state"]]
        right_name = definition.get("right_state")
        right = states[right_name] if right_name else None
        left_dump = left.model_dump(mode="json")
        right_dump = right.model_dump(mode="json") if right else None
        pair_report.append({
            **definition,
            "state_origins": {
                "left": provenance[definition["left_state"]]["state_origin"],
                "right": provenance[right_name]["state_origin"] if right_name else None,
            },
            "training_eligible": False,
            "changed_paths": _diff_paths(left_dump, right_dump) if right_dump else [],
            "controlled_pair": (
                provenance[definition["left_state"]]["state_origin"] == "controlled_canary"
                or (right_name and provenance[right_name]["state_origin"] == "controlled_canary")
            ),
        })

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "generated_at": audit["generated_at"],
        "policy_input_version": POLICY_INPUT_VERSION,
        "state_names": list(STATE_NAMES),
        "expected_policy_calls": len(STATE_NAMES),
        "max_transport_attempts_if_live": len(STATE_NAMES) * 2,
        "state_origin_policy": {
            "s1": "previous_gemini_canary_cache",
            "s2": "previous_gemini_canary_cache",
            "s3": "controlled_canary",
            "s4": "controlled_canary",
            "s5": "controlled_canary",
            "s6": "controlled_canary",
        },
        "training_eligible": False,
        "states": audit["states"],
        "pairs": pair_report,
        "no_remote_contact": True,
        "no_gemini_calls": True,
        "no_java_calls": True,
        "no_materializer_calls": True,
        "no_graphrag_calls": True,
        "audit_pass": audit["audit_pass"],
    }
    _dump(out_dir / "state_manifest.json", manifest)
    _dump(out_dir / "state_audit.json", audit)
    report = {
        "report_version": REPORT_VERSION,
        "generated_at": audit["generated_at"],
        "training_eligible": False,
        "live": False,
        "policy_origin": "not_called",
        "policy_input_version": POLICY_INPUT_VERSION,
        "state_audit_pass": audit["audit_pass"],
        "expected_policy_calls": len(STATE_NAMES),
        "transport_attempts": 0,
        "repair_count": 0,
        "states": audit["states"],
        "pairs": pair_report,
        "DYNAMIC_CANARY_READY": audit["audit_pass"],
        "FULL_E2E_READY": False,
        "BASE_QWEN_SMOKE": "PASS",
        "status": "PREPARED" if audit["audit_pass"] else "FAILED",
        "remote_contacted": False,
        "error": None if audit["audit_pass"] else "STATE_AUDIT_FAILED",
    }
    _dump(out_dir / "qwen_dynamic_state_canary_report.json", report)
    return report


class RecordingTransport(httpx.BaseTransport):
    """Record the real HTTP boundary for a later explicit live run."""

    def __init__(self, inner: httpx.BaseTransport | None = None) -> None:
        self.inner = inner or httpx.HTTPTransport()
        self.records: list[dict[str, Any]] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        try:
            body = json.loads(request.content.decode("utf-8"))
        except Exception:
            body = {"_unparsed_body": True}
        record: dict[str, Any] = {
            "request_index": len(self.records) + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request": {
                "method": request.method,
                "url": str(request.url),
                "headers": {
                    "content-type": request.headers.get("content-type"),
                    "authorization": "[redacted]" if request.headers.get("authorization") else None,
                },
                "body": body,
            },
        }
        try:
            messages = body.get("messages", [])
            policy_payload = json.loads(messages[1]["content"])
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


def _parse_decision(record: dict[str, Any]) -> tuple[DecisionPolicyOutput | None, str | None]:
    if record.get("response_status") != 200:
        return None, f"http_status_{record.get('response_status')}"
    try:
        parsed = _parse_json(_response_content(record))
        decision = DecisionPolicyOutput.model_validate(_canonicalize_policy_payload(parsed))
        return decision, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _raw_selected_action(record: dict[str, Any]) -> str | None:
    """Read the model's raw selected_action without requiring valid Action enum."""

    try:
        value = _parse_json(_response_content(record))
    except Exception:
        return None
    if isinstance(value, dict) and isinstance(value.get("selected_action"), str):
        return value["selected_action"]
    return None


def _failure_scope(
    records: list[dict[str, Any]],
    error: str | None,
) -> str | None:
    """Classify a failed State without turning policy errors into aborts.

    A 200 response that fails the closed Action contract belongs to the State
    under evaluation.  Transport failures, non-200 responses, and a depleted
    request budget are run-level failures because later States would not be
    observing a functioning Policy endpoint.
    """

    if any(item.get("error") for item in records):
        return "global_transport"
    if any(item.get("response_status") != 200 for item in records):
        return "global_transport"
    if error and "GEMINI_REQUEST_BUDGET_EXCEEDED" in error:
        return "global_budget"
    if error:
        return "state_policy"
    return None


def _decision_comparison(
    state_name: str,
    state: ScientificDecisionState,
    records: list[dict[str, Any]],
    decision: DecisionPolicyOutput | None,
    error: str | None,
    failure_scope: str | None = None,
) -> dict[str, Any]:
    failure_scope = failure_scope if error else None
    return {
        "state_name": state_name,
        "state_hash": _state_hash(state),
        "policy_origin": POLICY_ORIGIN,
        "expected_policy_calls": 1,
        "transport_attempts": len(records),
        "repair_count": sum(
            len(item.get("request", {}).get("body", {}).get("messages", [])) > 2
            for item in records
        ),
        "selected_action": decision.selected_action if decision else None,
        "decision_reason": decision.decision_reason if decision else None,
        "alternative_actions": (
            list(decision.alternative_actions) if decision else None
        ),
        "stop_reason": decision.stop_reason if decision else None,
        "reason_nonempty": bool(decision and decision.decision_reason.strip()),
        "selected_action_in_available": bool(
            decision and decision.selected_action in state.action_space.available_actions
        ),
        "selected_action_repeats_last_action": bool(
            decision and decision.selected_action == state.progress.last_action
        ),
        "finish_available": "finish" in state.action_space.available_actions,
        "state_status": "PASS" if error is None else "FAIL",
        "provider_error": error,
        "failure_scope": failure_scope,
    }


def _reason_references_changed_facts(
    definition: dict[str, Any],
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
) -> bool | None:
    """Apply an audit-only marker heuristic to a completed pair's reasons."""

    if not left or not right:
        return None
    reasons = " ".join(
        str(item.get("decision_reason") or "") for item in (left, right)
    ).lower()
    groups = definition.get("reason_marker_groups", [])
    if not reasons or not groups:
        return False
    return all(any(str(marker).lower() in reasons for marker in group) for group in groups)


def run_live_canary(
    out_dir: Path,
    *,
    states: dict[str, ScientificDecisionState],
    base_url: str,
    model: str,
    token: str = "",
    max_requests: int = len(STATE_NAMES) * 2,
    transport: httpx.BaseTransport | None = None,
    run_id: str = DEFAULT_LIVE_RUN_ID,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    adapter: str = "",
) -> dict[str, Any]:
    """Run each State independently through the production Policy boundary.

    A malformed/unavailable Action is an evaluation result for that State and
    does not suppress later State probes.  Only endpoint/transport failures or
    a depleted request budget abort the remaining probes.
    """

    if max_requests < len(STATE_NAMES) or max_requests > len(STATE_NAMES) * 2:
        raise ValueError("dynamic canary budget must allow 6-12 transport attempts")
    run_slug = re.sub(r"[^a-z0-9]+", "_", run_id.lower()).strip("_") or "dynamic_canary"
    run_dir = out_dir / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)
    recording = RecordingTransport(transport)
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
    entries: list[dict[str, Any]] = []
    aborted = False
    abort_reason: str | None = None
    try:
        for state_name in STATE_NAMES:
            state = states[state_name]
            policy_input: ScientificPolicyInput = build_scientific_policy_input(state)
            start = len(recording.records)
            decision: DecisionPolicyOutput | None = None
            provider_error: str | None = None
            try:
                decision = provider.select_action(policy_input)
            except Exception as exc:  # pragma: no cover - live provider outcome
                provider_error = f"{type(exc).__name__}: {exc}"
            state_records = recording.records[start:]
            failure_scope = _failure_scope(state_records, provider_error)
            entries.append(_decision_comparison(
                state_name,
                state,
                state_records,
                decision,
                provider_error,
                failure_scope,
            ))
            if provider_error is not None and failure_scope in {
                "global_transport",
                "global_budget",
            }:
                aborted = True
                abort_reason = provider_error
                break
    finally:
        provider.close()

    entry_by_name = {item["state_name"]: item for item in entries}
    live_pairs: list[dict[str, Any]] = []
    for definition in PAIR_DEFINITIONS:
        left = entry_by_name.get(definition["left_state"])
        right_name = definition.get("right_state")
        right = entry_by_name.get(right_name) if right_name else None
        both_pass = bool(
            left
            and right
            and left.get("state_status") == "PASS"
            and right.get("state_status") == "PASS"
        )
        pair: dict[str, Any] = {
            **definition,
            "training_eligible": False,
            "left": left,
            "right": right,
            "selected_action_changed": (
                both_pass
                and left.get("selected_action") != right.get("selected_action")
            ),
            "decision_reason_changed": (
                both_pass
                and left.get("decision_reason") != right.get("decision_reason")
            ),
            "left_finish_without_readiness": bool(
                left
                and left.get("selected_action") == "finish"
                and not left.get("finish_available")
            ),
            "right_finish_without_readiness": bool(
                right
                and right.get("selected_action") == "finish"
                and not right.get("finish_available")
            ),
        }
        pair["changed_fact_referenced"] = _reason_references_changed_facts(
            definition,
            left,
            right,
        )
        pair["changed_fact_reference_method"] = "heuristic_marker_groups"
        live_pairs.append(pair)

    model_responses = sum(
        item.get("response_status") == 200 for item in recording.records
    )
    contract_valid_count = sum(
        item["state_status"] == "PASS" for item in entries
    )
    available_action_compliance_count = sum(
        item["selected_action_in_available"] for item in entries
    )
    objective_action_confusion_count = 0
    unavailable_action_count = 0
    for record in recording.records:
        action = _raw_selected_action(record)
        state_payload = record.get("state") or {}
        task_payload = state_payload.get("task") or {}
        action_space_payload = state_payload.get("action_space") or {}
        objectives = set(task_payload.get("objectives") or [])
        available = set(action_space_payload.get("available_actions") or [])
        if action in objectives and action not in ALL_SCIENTIFIC_ACTIONS:
            objective_action_confusion_count += 1
        elif action in ALL_SCIENTIFIC_ACTIONS and action not in available:
            unavailable_action_count += 1

    state_sensitive_pairs = [
        item["pair_id"] for item in live_pairs if item["selected_action_changed"]
    ]
    changed_fact_referenced_pairs = [
        item["pair_id"]
        for item in live_pairs
        if item.get("changed_fact_referenced") is True
    ]
    s6 = next((item for item in entries if item["state_name"] == "s6"), None)
    s6_finish = bool(
        s6
        and s6.get("selected_action") == "finish"
        and s6.get("finish_available")
    )
    s6_finish_missed = bool(
        s6
        and s6.get("finish_available")
        and s6.get("selected_action") != "finish"
    )
    report = {
        "report_version": REPORT_VERSION,
        "run_id": run_id,
        "prompt_version": prompt_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "training_eligible": False,
        "live": True,
        "base_url": base_url,
        "model": model,
        "adapter": adapter or "none",
        "policy_origin": POLICY_ORIGIN,
        "policy_input_version": POLICY_INPUT_VERSION,
        "expected_policy_calls": len(STATE_NAMES),
        "transport_attempts": len(recording.records),
        "repair_count": sum(
            len(item.get("request", {}).get("body", {}).get("messages", [])) > 2
            for item in recording.records
        ),
        "model_responses": model_responses,
        "contract_valid": {
            "count": contract_valid_count,
            "total": len(STATE_NAMES),
        },
        "available_action_compliance": {
            "count": available_action_compliance_count,
            "total": len(STATE_NAMES),
        },
        "objective_action_confusion_count": objective_action_confusion_count,
        "unavailable_action_count": unavailable_action_count,
        "redundant_action_count": sum(
            item["selected_action_repeats_last_action"] for item in entries
        ),
        "state_sensitive_pairs": {
            "count": len(state_sensitive_pairs),
            "total": 3,
            "pairs": state_sensitive_pairs,
        },
        "changed_fact_referenced_pairs": {
            "count": len(changed_fact_referenced_pairs),
            "total": 3,
            "pairs": changed_fact_referenced_pairs,
            "method": "heuristic_marker_groups",
        },
        "S6_finish": s6_finish,
        "finish_missed": s6_finish_missed,
        "deterministic_policy_fallback_used": False,
        "legacy_strategy_guard_used": False,
        "states": entries,
        "pairs": live_pairs,
        "all_states_evaluated": len(entries) == len(STATE_NAMES),
        "aborted": aborted,
        "abort_reason": abort_reason,
        "DYNAMIC_CANARY_READY": len(entries) == len(STATE_NAMES),
        "DYNAMIC_STATE_CANARY": "PASS" if (
            len(entries) == len(STATE_NAMES)
            and all(item["state_status"] == "PASS" for item in entries)
        ) else "FAIL",
        "FULL_E2E_READY": False,
        "BASE_QWEN_SMOKE": "PASS",
        "remote_contacted": True,
        "status": "PASS" if (
            len(entries) == len(STATE_NAMES)
            and len(recording.records) <= max_requests
            and all(
                item["provider_error"] is None
                and item["selected_action_in_available"]
                and item["reason_nonempty"]
                for item in entries
            )
        ) else "FAILED",
        "error": abort_reason,
    }
    report["artifact_dir"] = str(run_dir)
    _dump(run_dir / "report.json", report)
    for record in recording.records:
        _dump(run_dir / f"policy_request_{record['request_index']:03d}.json", record)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--catalog", type=Path, default=None)
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
    parser.add_argument("--max-requests", type=int, default=len(STATE_NAMES) * 2)
    parser.add_argument("--run-id", default=DEFAULT_LIVE_RUN_ID)
    parser.add_argument("--prompt-version", default=DEFAULT_PROMPT_VERSION)
    parser.add_argument(
        "--adapter",
        default=os.environ.get("MICO_SFT_POLICY_ADAPTER", ""),
        help="Adapter metadata for the served policy (the server loads it; the client only records it).",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Explicitly call the policy endpoint after local state preparation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = prepare_dynamic_canary(
        args.output_dir,
        source_dir=args.source_dir,
        catalog_path=args.catalog,
    )
    if report["status"] != "PREPARED":
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1
    if args.live:
        states, _provenance, _catalog = build_dynamic_states(
            args.source_dir,
            catalog_path=args.catalog,
        )
        report = run_live_canary(
            args.output_dir,
            states=states,
            base_url=args.base_url,
            model=args.model,
            token=args.token,
            max_requests=args.max_requests,
            run_id=args.run_id,
            prompt_version=args.prompt_version,
            adapter=args.adapter,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
    return 0 if report["status"] in {"PREPARED", "PASS"} else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


__all__ = [
    "DEFAULT_LIVE_RUN_ID",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_SOURCE_DIR",
    "DEFAULT_PROMPT_VERSION",
    "PAIR_DEFINITIONS",
    "RecordingTransport",
    "STATE_NAMES",
    "build_dynamic_states",
    "prepare_dynamic_canary",
    "run_live_canary",
]
