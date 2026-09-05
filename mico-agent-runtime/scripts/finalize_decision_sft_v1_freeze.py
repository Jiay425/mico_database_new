"""Finalize the completed Decision SFT v1 experiment without model calls.

The training session already produced the Base/Frozen/Validation/Canary
artifacts.  This utility only derives clearer metric names, breaks down the
captured contract failures, and writes an immutable experiment manifest.  It
never loads a model, reads the training examples for relabeling, or reruns an
evaluation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


EXPECTED_DATASET_SHA = "6e3db8ab22786d4004bdd9846a1d09b5dba7ba1a3522917551bfca74dccf8ef1"
EXPECTED_PROMPT_SHA = "c5ab7dbc7ff2ee99f5720f0b99dc2d0fb7d71150a00ffd7a5f6bf5858dcb1db4"
FIXED_ACTIONS = {
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
}
STOP_REASONS = {
    "EVIDENCE_SUFFICIENT",
    "NO_NEW_INFORMATION",
    "QUALITY_RISK",
    "ACTION_BUDGET_EXHAUSTED",
    "UPSTREAM_REJECTED",
    "UNSUPPORTED_ACTION",
    "USER_REQUESTED_STOP",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_first_object(raw: Any) -> Mapping[str, Any] | None:
    if not isinstance(raw, str):
        return None
    decoder = json.JSONDecoder()
    for index, character in enumerate(raw):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def contract_failure_category(row: Mapping[str, Any]) -> str | None:
    """Classify a failed captured model response, without changing it."""

    if row.get("contract_valid"):
        return None
    payload = parse_first_object(row.get("raw_response"))
    if payload is None:
        return "json_parse"
    if set(payload) != {"selected_action", "decision_reason", "alternative_actions", "stop_reason"}:
        return "response_shape"
    selected = payload.get("selected_action")
    reason = payload.get("decision_reason")
    alternatives = payload.get("alternative_actions")
    stop_reason = payload.get("stop_reason")
    if not isinstance(selected, str) or selected not in FIXED_ACTIONS:
        return "selected_action_shape"
    if not isinstance(reason, str) or not reason.strip():
        return "decision_reason"
    if not isinstance(alternatives, list) or any(
        not isinstance(item, str) or item not in FIXED_ACTIONS for item in alternatives
    ) or len(alternatives) != len(set(alternatives)) or selected in alternatives:
        return "alternative_actions"
    if (selected == "finish" and stop_reason not in STOP_REASONS) or (
        selected != "finish" and stop_reason is not None
    ):
        return "stop_reason"
    return "other_contract"


def failure_breakdown(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    categories: dict[str, list[str]] = {}
    for row in rows:
        category = contract_failure_category(row)
        if category is not None:
            categories.setdefault(category, []).append(str(row.get("sample_id")))
    counts = {key: len(value) for key, value in sorted(categories.items())}
    for key in ("json_parse", "response_shape", "selected_action_shape", "decision_reason", "alternative_actions", "stop_reason", "other_contract"):
        counts.setdefault(key, 0)
    return {
        "sample_count": len(rows),
        "contract_invalid_count": sum(counts.values()),
        "counts": counts,
        "sample_ids": {key: value for key, value in sorted(categories.items())},
    }


def corrected_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Rename ambiguous report fields while retaining all numeric values."""

    result = copy.deepcopy(dict(metrics))
    if "objective_action_confusion_count" in result:
        result["invalid_objective_as_action_count"] = result.pop("objective_action_confusion_count")
    buckets = result.get("hard_case_bucket_metrics")
    if isinstance(buckets, dict) and "objective_action_confusion" in buckets:
        boundary = buckets.pop("objective_action_confusion")
        buckets["objective_action_boundary"] = boundary
        if isinstance(boundary, dict):
            result["objective_action_boundary_accuracy"] = {
                "count": boundary.get("count", 0),
                "correct": round(float(boundary.get("selected_action_accuracy", 0.0)) * int(boundary.get("count", 0))),
                "total": boundary.get("count", 0),
                "rate": boundary.get("selected_action_accuracy", 0.0),
            }
    return result


def corrected_report(source: Mapping[str, Any], base_breakdown: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(source))
    result["report_semantics"] = {
        "invalid_objective_as_action_count": "Number of responses whose selected_action is an objective literal rather than one of the ten Scientific Actions.",
        "objective_action_boundary_accuracy": "Golden selected_action accuracy on samples labeled objective_action_confusion in the historical hard-case taxonomy; this is not an invalid-objective count.",
        "contract_failure_breakdown": "Derived from the captured raw Base Frozen Test responses; no model inference was rerun.",
    }
    result["base_frozen_test"] = corrected_metrics(result["base_frozen_test"])
    result["final_sft_frozen_test"] = corrected_metrics(result["final_sft_frozen_test"])
    training = result.get("training")
    if isinstance(training, dict):
        for epoch in training.get("epochs", []):
            if isinstance(epoch, dict) and isinstance(epoch.get("validation_metrics"), dict):
                epoch["validation_metrics"] = corrected_metrics(epoch["validation_metrics"])
    result["base_contract_failure_breakdown"] = dict(base_breakdown)
    result["DECISION_SFT_V1_FROZEN"] = True
    result["postprocessing_only"] = True
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--training-prep-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    session = args.session_dir
    prep = args.training_prep_dir
    output = args.output_dir
    source_report_path = session / "session_report.json"
    base_results_path = session / "base_frozen_test.jsonl"
    selection_path = session / "checkpoint_selection.json"
    if not source_report_path.is_file() or not base_results_path.is_file() or not selection_path.is_file():
        raise SystemExit("completed_session_artifacts_missing")
    source_report = read_json(source_report_path)
    if source_report.get("DECISION_SFT_V1_TRAINING") != "PASS":
        raise SystemExit("session_not_pass")
    if source_report.get("frozen_test_used_for_selection") is not False:
        raise SystemExit("frozen_test_selection_flag_changed")
    dataset_sha = source_report.get("dataset_sha256")
    if dataset_sha != EXPECTED_DATASET_SHA:
        raise SystemExit(f"dataset_sha_mismatch:{dataset_sha}")
    prompt_sha = source_report.get("prompt_sha256")
    if prompt_sha != EXPECTED_PROMPT_SHA:
        raise SystemExit(f"prompt_sha_mismatch:{prompt_sha}")

    base_breakdown = failure_breakdown(base_results_path)
    corrected = corrected_report(source_report, base_breakdown)
    write_json(output / "session_report_corrected.json", corrected)
    write_json(output / "metric_naming_correction.json", {
        "schema_version": "decision-sft-v1-metric-correction-v1",
        "source_report": str(source_report_path),
        "invalid_objective_as_action_count": {
            "base": corrected["base_frozen_test"]["invalid_objective_as_action_count"],
            "final_sft": corrected["final_sft_frozen_test"]["invalid_objective_as_action_count"],
        },
        "objective_action_boundary_accuracy": {
            "base": corrected["base_frozen_test"].get("objective_action_boundary_accuracy"),
            "final_sft": corrected["final_sft_frozen_test"].get("objective_action_boundary_accuracy"),
        },
        "base_contract_failure_breakdown": base_breakdown,
        "model_rerun": False,
    })

    training_config = prep / "training_config.json"
    dataset_fingerprint = prep / "dataset_fingerprint.json"
    freeze_manifest = {
        "schema_version": "decision-sft-v1-freeze-manifest-v1",
        "DECISION_SFT_V1_FROZEN": True,
        "frozen_test_mutable": False,
        "dataset_sha256": EXPECTED_DATASET_SHA,
        "prompt_sha256": EXPECTED_PROMPT_SHA,
        "training_config_sha256": sha256(training_config),
        "dataset_fingerprint_file_sha256": sha256(dataset_fingerprint),
        "model": "Qwen3-8B",
        "adapter": {
            "origin": "remote_a100",
            "selected_checkpoint": "/root/autodl-tmp/mico-dpo-v4/outputs/decision-sft-v1-20260902-session2/checkpoint_epoch2",
            "adapter_model_safetensors_sha256": "c1935825a3099c7633585088e1afd7e015e9c1b18806755bca502ca505b07577",
            "adapter_config_json_sha256": "0d698ccca7ff1f1d28c27d5bad20e55d3466133b88b02b4f809741fd0166f07e",
            "checkpoint_epoch1": "/root/autodl-tmp/mico-dpo-v4/outputs/decision-sft-v1-20260902-session2/checkpoint_epoch1",
            "checkpoint_epoch2": "/root/autodl-tmp/mico-dpo-v4/outputs/decision-sft-v1-20260902-session2/checkpoint_epoch2",
        },
        "split_counts": {"train": 388, "validation": 65, "frozen_test": 65},
        "training": {
            "epochs": 2,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "precision": "bf16",
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "learning_rate": 0.0002,
            "scheduler": "cosine",
            "optimizer": "adamw_torch",
            "max_seq_length": 1280,
            "assistant_only_loss": True,
            "enable_thinking": False,
        },
        "selection": read_json(selection_path),
        "provenance": {
            "base_frozen_test_captured_once": True,
            "base_baseline_reused_after_transport_recovery": True,
            "final_frozen_test_captured_once": True,
            "external_canary_training_eligible": False,
            "dpo_started": False,
            "full_e2e_started": False,
            "gpu_process_stopped": True,
            "model_rerun_for_metric_rename": False,
        },
    }
    write_json(output / "decision_sft_v1_freeze_manifest.json", freeze_manifest)
    print(json.dumps({
        "DECISION_SFT_V1_FROZEN": True,
        "base_contract_failure_breakdown": base_breakdown["counts"],
        "corrected_report": str(output / "session_report_corrected.json"),
        "freeze_manifest": str(output / "decision_sft_v1_freeze_manifest.json"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
