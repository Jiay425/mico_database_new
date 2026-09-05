"""Offline contract and preparation gate for Decision SFT v1 Full E2E.

This module does not contact Java, MySQL, Gemini, Qwen, DeepSeek, GraphRAG,
or an A100.  It validates the frozen, open-trajectory task set and writes the
acceptance manifest used by the later explicit ``--live`` runner.

The task set is intentionally not a policy oracle.  It contains natural
questions, scope/availability obligations, and provenance only.  It does not
contain a required Action, a fixed Action sequence, a QueryPlan, an
AnalysisPlan, raw rows, or a fabricated result.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
DEFAULT_TASK_SET_PATH = REPO_ROOT / "evals" / "decision_sft_v1_full_dynamic_e2e_task_set_v1.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "decision_sft_v1_full_dynamic_e2e_v1"
DEFAULT_FREEZE_MANIFEST = REPO_ROOT / "artifacts" / "decision_sft_v1_training" / "decision_sft_v1_freeze_manifest.json"

FULL_DYNAMIC_E2E_VERSION = "decision-sft-v1-full-dynamic-e2e-v1"
E2E_TASK_SET_SCHEMA_VERSION = "decision-sft-v1-full-dynamic-e2e-task-set-v1"
POLICY_INPUT_VERSION = "scientific-decision-state-v1"

ALL_SCIENTIFIC_ACTIONS: tuple[str, ...] = (
    "inspect_cohort",
    "execute_read_query",
    "compare_groups",
    "analyze_projection",
    "stratified_analysis",
    "adjust_confounders",
    "cross_project_validate",
    "cross_disease_validate",
    "retrieve_evidence",
    "finish",
)

VALID_SCOPES = {
    "mico:query:read",
    "mico:research:read",
    "mico:evidence:read",
}
VALID_OBJECTIVES = {
    "group_comparison",
    "projection_analysis",
    "stratified_analysis",
    "confounder_assessment",
    "cross_project_validation",
    "cross_disease_validation",
    "evidence_support",
}
VALID_SCENARIOS = {
    "real_dynamic",
    "real_availability_boundary",
    "real_evidence_optional",
    "controlled_finish_boundary",
}
VALID_OBLIGATIONS = {
    "task_understanding",
    "real_query",
    "real_analysis",
    "real_observation",
    "state_update",
    "next_policy_decision",
    "real_retrieval_optional",
    "availability_boundary",
    "insufficient_data_observation",
    "finish_boundary",
}

# These names are rejected recursively, including if a future task author
# tries to hide a policy oracle in a nested acceptance object.
FORBIDDEN_LEGACY_FIELDS = {
    "goal_code",
    "goalCode",
    "observation_flags",
    "observationFlags",
    "candidate_actions",
    "candidateActions",
    "next_action_hint",
    "nextActionHint",
    "recommended_action",
    "recommendedAction",
    "required_action",
    "requiredAction",
    "planner_hint",
    "plannerHint",
    "strategy_hint",
    "strategyHint",
    "forced_action",
    "forcedAction",
    "repair_target",
    "repairTarget",
}

# No raw records, secrets, physical SQL, or provider endpoints belong in a
# task-set source file.  These checks are intentionally conservative.
SENSITIVE_MARKERS = (
    "bearer ",
    "api_key",
    "apikey",
    "password",
    "mysql://",
    "postgresql://",
    "select ",
    "insert ",
    "update ",
    "delete ",
    "drop ",
    "patient_id",
    "patientid",
    "sample_id",
    "sampleid",
    "internalrecordid",
    "sourcesampleid",
    "raw_rows",
    '"rows"',
)


class TaskSetValidationError(ValueError):
    """Raised when the versioned E2E task set violates its preparation contract."""

    def __init__(self, errors: list[str] | str) -> None:
        if isinstance(errors, str):
            errors = [errors]
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _iter_keys(value: object, path: str = "$") -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_path = f"{path}.{key}"
            found.append((key_path, str(key)))
            found.extend(_iter_keys(child, key_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_iter_keys(child, f"{path}[{index}]"))
    return found


def _contains_sensitive_text(value: object, path: str = "$") -> list[str]:
    hits: list[str] = []
    if isinstance(value, str):
        lowered = value.lower()
        for marker in SENSITIVE_MARKERS:
            if marker in lowered:
                hits.append(f"{path}:sensitive_marker={marker}")
        # Reject local/remote physical paths and URL values, while allowing
        # natural-language questions to mention no provider details.
        if re.search(r"(?i)(?:https?://|ssh\s+-p|[A-Za-z]:\\\\|/root/)", value):
            hits.append(f"{path}:physical_endpoint_or_path")
    elif isinstance(value, dict):
        for key, child in value.items():
            hits.extend(_contains_sensitive_text(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            hits.extend(_contains_sensitive_text(child, f"{path}[{index}]"))
    return hits


def load_task_set(path: Path = DEFAULT_TASK_SET_PATH) -> dict[str, Any]:
    """Load a JSON task set without contacting any external service."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TaskSetValidationError(f"task_set_not_found:{path}") from exc
    except json.JSONDecodeError as exc:
        raise TaskSetValidationError(f"task_set_invalid_json:{exc.msg}") from exc
    if not isinstance(value, dict):
        raise TaskSetValidationError("task_set_root_must_be_object")
    return value


def _require_string(errors: list[str], value: object, path: str, *, max_length: int = 4096) -> None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{path}:non_empty_string_required")
    elif len(value) > max_length:
        errors.append(f"{path}:too_long")


def _require_string_list(
    errors: list[str],
    value: object,
    path: str,
    *,
    allowed: set[str] | None = None,
    non_empty: bool = False,
) -> None:
    if not isinstance(value, list) or (non_empty and not value):
        errors.append(f"{path}:string_list_required")
        return
    if any(not isinstance(item, str) or not item.strip() for item in value):
        errors.append(f"{path}:all_values_must_be_non_empty_strings")
    if len(value) != len(set(value)):
        errors.append(f"{path}:duplicate_values")
    if allowed is not None:
        unknown = sorted(set(value) - allowed)
        if unknown:
            errors.append(f"{path}:unknown_values={unknown}")


def validate_task_set(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate the immutable E2E task-set contract and return audit data."""

    errors: list[str] = []
    required_root = {
        "schema_version",
        "status",
        "purpose",
        "training_eligible",
        "policy_input_version",
        "scientific_actions",
        "provider_roles",
        "execution_policy",
        "tasks",
    }
    missing_root = sorted(required_root - set(payload))
    if missing_root:
        errors.append(f"root:missing={missing_root}")
    unknown_root = sorted(set(payload) - required_root)
    if unknown_root:
        errors.append(f"root:unknown_fields={unknown_root}")
    if payload.get("schema_version") != E2E_TASK_SET_SCHEMA_VERSION:
        errors.append("root:schema_version_mismatch")
    if payload.get("status") != "PREPARED_NOT_RUN":
        errors.append("root:status_must_be_PREPARED_NOT_RUN")
    if payload.get("training_eligible") is not False:
        errors.append("root:training_eligible_must_be_false")
    if payload.get("policy_input_version") != POLICY_INPUT_VERSION:
        errors.append("root:policy_input_version_mismatch")
    if tuple(payload.get("scientific_actions", ())) != ALL_SCIENTIFIC_ACTIONS:
        errors.append("root:scientific_actions_must_match_fixed_ten_actions")
    if not isinstance(payload.get("purpose"), str) or not payload["purpose"].strip():
        errors.append("root:purpose_required")

    provider_roles = payload.get("provider_roles")
    if not isinstance(provider_roles, dict):
        errors.append("provider_roles:object_required")
    else:
        expected_roles = {
            "task_understanding": "gemini_model",
            "scientific_policy": "qwen_sft_model",
            "materializer": "gemini_canary_model",
            "java_query": "real_java_tool",
            "python_analysis": "real_typed_runtime",
            "retrieval": "real_if_available",
        }
        if provider_roles != expected_roles:
            errors.append("provider_roles:must_match_frozen_roles")

    execution_policy = payload.get("execution_policy")
    if not isinstance(execution_policy, dict):
        errors.append("execution_policy:object_required")
    else:
        expected_policy = {
            "max_actions_per_task": 6,
            "open_trajectory": True,
            "fixed_action_sequence": False,
            "runtime_may_skip_unavailable_task": True,
            "deterministic_scientific_action_fallback": False,
            "deterministic_materializer_fallback": False,
        }
        for key, expected in expected_policy.items():
            if execution_policy.get(key) != expected:
                errors.append(f"execution_policy.{key}:must_equal_{expected!r}")

    legacy_hits = [f"{path}:{key}" for path, key in _iter_keys(payload) if key in FORBIDDEN_LEGACY_FIELDS]
    errors.extend(f"forbidden_legacy_field={hit}" for hit in legacy_hits)
    errors.extend(_contains_sensitive_text(payload))

    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 5:
        errors.append("tasks:exactly_five_required")
        tasks = []
    seen_ids: set[str] = set()
    seen_scenarios: set[str] = set()
    task_audit: list[dict[str, Any]] = []
    task_required = {
        "task_id",
        "label",
        "question",
        "scenario",
        "requested_scopes",
        "expected_objectives",
        "obligations",
        "max_actions",
        "real_data_required",
        "skip_if_unavailable",
        "training_eligible",
    }
    for index, task in enumerate(tasks):
        prefix = f"tasks[{index}]"
        if not isinstance(task, dict):
            errors.append(f"{prefix}:object_required")
            continue
        missing = sorted(task_required - set(task))
        if missing:
            errors.append(f"{prefix}:missing={missing}")
        unknown_task = sorted(set(task) - task_required)
        if unknown_task:
            errors.append(f"{prefix}:unknown_fields={unknown_task}")
        task_id = task.get("task_id")
        _require_string(errors, task_id, f"{prefix}.task_id", max_length=128)
        if isinstance(task_id, str):
            if task_id in seen_ids:
                errors.append(f"{prefix}.task_id:duplicate")
            seen_ids.add(task_id)
        _require_string(errors, task.get("label"), f"{prefix}.label", max_length=160)
        _require_string(errors, task.get("question"), f"{prefix}.question", max_length=4096)
        scenario = task.get("scenario")
        if scenario not in VALID_SCENARIOS:
            errors.append(f"{prefix}.scenario:unknown={scenario!r}")
        elif isinstance(scenario, str):
            seen_scenarios.add(scenario)
        _require_string_list(
            errors,
            task.get("requested_scopes"),
            f"{prefix}.requested_scopes",
            allowed=VALID_SCOPES,
            non_empty=True,
        )
        _require_string_list(
            errors,
            task.get("expected_objectives"),
            f"{prefix}.expected_objectives",
            allowed=VALID_OBJECTIVES,
        )
        _require_string_list(
            errors,
            task.get("obligations"),
            f"{prefix}.obligations",
            allowed=VALID_OBLIGATIONS,
            non_empty=True,
        )
        max_actions = task.get("max_actions")
        if type(max_actions) is not int or max_actions != 6:
            errors.append(f"{prefix}.max_actions:must_be_6")
        if task.get("training_eligible") is not False:
            errors.append(f"{prefix}.training_eligible:must_be_false")
        if type(task.get("real_data_required")) is not bool:
            errors.append(f"{prefix}.real_data_required:boolean_required")
        if type(task.get("skip_if_unavailable")) is not bool:
            errors.append(f"{prefix}.skip_if_unavailable:boolean_required")
        if scenario == "controlled_finish_boundary":
            if task.get("real_data_required") is not False or task.get("skip_if_unavailable") is not True:
                errors.append(f"{prefix}:controlled_finish_must_not_claim_real_data")
        elif task.get("real_data_required") is not True:
            errors.append(f"{prefix}:real_scenario_requires_real_data")
        task_audit.append({
            "task_id": task_id,
            "scenario": scenario,
            "max_actions": max_actions,
            "training_eligible": task.get("training_eligible"),
            "real_data_required": task.get("real_data_required"),
            "skip_if_unavailable": task.get("skip_if_unavailable"),
            "expected_objective_count": len(task.get("expected_objectives", [])) if isinstance(task.get("expected_objectives"), list) else None,
            "obligation_count": len(task.get("obligations", [])) if isinstance(task.get("obligations"), list) else None,
        })

    if seen_scenarios != VALID_SCENARIOS:
        errors.append(f"tasks:scenario_coverage_mismatch={sorted(seen_scenarios)}")

    if errors:
        raise TaskSetValidationError(errors)

    return {
        "schema_valid": True,
        "task_count": len(tasks),
        "task_ids": sorted(seen_ids),
        "scenario_coverage": sorted(seen_scenarios),
        "all_training_ineligible": all(item["training_eligible"] is False for item in task_audit),
        "all_max_actions_six": all(item["max_actions"] == 6 for item in task_audit),
        "fixed_action_set": list(ALL_SCIENTIFIC_ACTIONS),
        "policy_input_blocks": [
            "task",
            "data_state",
            "analysis_state",
            "evidence_state",
            "progress",
            "action_space",
        ],
        "task_audit": task_audit,
        "forbidden_legacy_fields": [],
        "sensitive_text_hits": [],
    }


def _freeze_manifest_audit(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "DECISION_SFT_V1_FROZEN": False, "path": str(path)}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"exists": True, "DECISION_SFT_V1_FROZEN": False, "path": str(path)}
    return {
        "exists": True,
        "DECISION_SFT_V1_FROZEN": value.get("DECISION_SFT_V1_FROZEN") is True,
        "dataset_sha256": value.get("dataset_sha256"),
        "prompt_sha256": value.get("prompt_sha256"),
        "selected_checkpoint": (value.get("selection") or {}).get("selected_checkpoint") or (value.get("adapter") or {}).get("selected_checkpoint"),
        "selected_adapter_sha256": (value.get("adapter") or {}).get("adapter_model_safetensors_sha256"),
        "path": str(path),
    }


def prepare_task_set(
    task_set_path: Path = DEFAULT_TASK_SET_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    freeze_manifest_path: Path = DEFAULT_FREEZE_MANIFEST,
) -> dict[str, Any]:
    """Validate and snapshot the E2E task set without any external calls."""

    payload = load_task_set(task_set_path)
    audit = validate_task_set(payload)
    freeze = _freeze_manifest_audit(freeze_manifest_path)
    task_bytes = task_set_path.read_bytes()
    task_sha = _sha256_bytes(task_bytes)
    canonical_sha = _sha256_bytes(_canonical_json(payload))

    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / "task_set_snapshot.json"
    snapshot_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    generated_at = _utc_now()
    gates = {
        "task_set_schema": audit["schema_valid"],
        "fixed_ten_scientific_actions": audit["fixed_action_set"] == list(ALL_SCIENTIFIC_ACTIONS),
        "open_trajectory": payload["execution_policy"]["open_trajectory"] is True,
        "no_fixed_action_sequence": payload["execution_policy"]["fixed_action_sequence"] is False,
        "max_actions_is_six": audit["all_max_actions_six"],
        "training_ineligible": audit["all_training_ineligible"] and payload["training_eligible"] is False,
        "no_legacy_policy_fields": True,
        "no_sensitive_or_raw_data": True,
        "frozen_sft_v1_manifest": freeze["DECISION_SFT_V1_FROZEN"],
        "sft_adapter_reference_present": bool(freeze.get("selected_checkpoint") and freeze.get("selected_adapter_sha256")),
        "scientific_policy_provider_is_sft_qwen": payload["provider_roles"]["scientific_policy"] == "qwen_sft_model",
        "materializer_provider_is_gemini": payload["provider_roles"]["materializer"] == "gemini_canary_model",
        "no_deterministic_scientific_action_fallback": payload["execution_policy"]["deterministic_scientific_action_fallback"] is False,
        "no_deterministic_materializer_fallback": payload["execution_policy"]["deterministic_materializer_fallback"] is False,
        # Live gates intentionally remain pending until the explicit runner is
        # invoked after the operator confirms the services and GPU session.
        "java_mysql_preflight": "PENDING_LIVE",
        "qwen_sft_http_endpoint": "PENDING_LIVE",
        "gemini_task_understanding": "PENDING_LIVE",
        "gemini_materializer": "PENDING_LIVE",
        "real_observation_and_state_update": "PENDING_LIVE",
        "trace_provenance": "PENDING_LIVE",
    }
    manifest = {
        "manifest_version": FULL_DYNAMIC_E2E_VERSION + "-preparation-manifest",
        "generated_at": generated_at,
        "status": "PREPARED_NOT_RUN",
        "task_set_path": str(task_set_path),
        "task_set_sha256": task_sha,
        "task_set_canonical_sha256": canonical_sha,
        "task_set_schema_version": E2E_TASK_SET_SCHEMA_VERSION,
        "policy_input_version": POLICY_INPUT_VERSION,
        "freeze_manifest": freeze,
        "task_audit": audit,
        "acceptance_gates": gates,
        "no_remote_contact": True,
        "model_calls": 0,
        "a100_started": False,
        "dpo_started": False,
        "training_started": False,
        "full_e2e_started": False,
        "training_eligible": False,
        "FULL_E2E_READY": False,
        # The preparation gate intentionally cannot authorize a GPU session;
        # the live service/preflight gates remain pending until the operator
        # explicitly invokes the separate live runner.
        "READY_TO_START_A100": False,
    }
    (output_dir / "preparation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "acceptance_gates.json").write_text(
        json.dumps(gates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "task_set_sha256.txt").write_text(task_sha + "\n", encoding="utf-8")
    return manifest


__all__ = [
    "ALL_SCIENTIFIC_ACTIONS",
    "DEFAULT_FREEZE_MANIFEST",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_TASK_SET_PATH",
    "E2E_TASK_SET_SCHEMA_VERSION",
    "FULL_DYNAMIC_E2E_VERSION",
    "POLICY_INPUT_VERSION",
    "TaskSetValidationError",
    "load_task_set",
    "prepare_task_set",
    "validate_task_set",
]
