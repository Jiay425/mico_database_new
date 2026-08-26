"""Split one combined v4 evaluation into auditable frozen evaluation sets.

This script is intentionally local and deterministic.  It does not load a
model and does not call any external API.  The remote inference result is
split by the source record IDs, then the existing deterministic reason audit
is run once per group.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from p2j4_audit_decision_reasons import audit as audit_reasons


GROUPS = {
    "test70": "sft-data/decision-v4-staging/test.jsonl",
    "ood_v2": "evals/p2j4-external-ood-v2/records.jsonl",
    "residual_repair": "evals/p2j4-state-obligation-residual-repair-v2/records.jsonl",
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"EXPECTED_OBJECT:{path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(cases)
    json_valid = sum(bool(case.get("prediction")) for case in cases)
    action_correct = sum(bool(case.get("action_correct")) for case in cases)
    action_allowed = sum(bool(case.get("action_allowed")) for case in cases)
    stop_consistent = sum(bool(case.get("stop_consistent")) for case in cases)
    policy_pass = sum(
        bool(case.get("action_correct")) and bool(case.get("stop_consistent"))
        for case in cases
    )

    def rate(value: int) -> float:
        return value / count if count else 0.0

    return {
        "caseCount": count,
        "jsonValidCount": json_valid,
        "jsonValidRate": rate(json_valid),
        "actionCorrectCount": action_correct,
        "actionAccuracy": rate(action_correct),
        "actionAllowedCount": action_allowed,
        "actionAllowedRate": rate(action_allowed),
        "stopConsistentCount": stop_consistent,
        "stopConsistencyRate": rate(stop_consistent),
        "policyPassCount": policy_pass,
        "policyPassRate": rate(policy_pass),
        "cases": cases,
    }


def make_group_prediction(
    combined: dict[str, Any],
    group: str,
    record_ids: set[str],
    input_file: Path,
) -> dict[str, Any]:
    base_cases = [
        case for case in combined.get("base", {}).get("cases", [])
        if case.get("record_id") in record_ids
    ]
    sft_cases = [
        case for case in combined.get("sft", {}).get("cases", [])
        if case.get("record_id") in record_ids
    ]
    if {case.get("record_id") for case in base_cases} != record_ids:
        raise ValueError(f"BASE_ID_MISMATCH:{group}")
    if {case.get("record_id") for case in sft_cases} != record_ids:
        raise ValueError(f"SFT_ID_MISMATCH:{group}")
    return {
        "schemaVersion": "p2j4-decision-sft-eval-v1",
        "testFile": input_file.name,
        "testCount": len(record_ids),
        "trainingStarted": True,
        "group": group,
        "base": summarize(base_cases),
        "sft": summarize(sft_cases),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()

    runtime_root = args.runtime_root.resolve()
    bundle_dir = args.bundle.resolve()
    combined_path = bundle_dir / "v4-eval.json"
    combined = read_json(combined_path)

    all_ids: set[str] = set()
    group_predictions: dict[str, dict[str, Any]] = {}
    input_paths: dict[str, Path] = {}

    for group, relative_input in GROUPS.items():
        input_path = runtime_root / relative_input
        input_paths[group] = input_path
        records = read_jsonl(input_path)
        ids = [record["id"] for record in records]
        if len(ids) != len(set(ids)):
            raise ValueError(f"DUPLICATE_INPUT_IDS:{group}")
        group_ids = set(ids)
        if all_ids.intersection(group_ids):
            raise ValueError(f"CROSS_GROUP_ID_OVERLAP:{group}")
        all_ids.update(group_ids)
        group_predictions[group] = make_group_prediction(
            combined, group, group_ids, input_path
        )

    combined_ids = {
        case.get("record_id")
        for case in combined.get("sft", {}).get("cases", [])
    }
    if combined_ids != all_ids:
        raise ValueError("COMBINED_ID_SET_MISMATCH")

    outputs: dict[str, Any] = {}
    for group, prediction in group_predictions.items():
        prediction_path = bundle_dir / f"{group}-v4-eval.json"
        prediction_path.write_text(
            json.dumps(prediction, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        reason_path = bundle_dir / f"{group}-v4-reason-audit.json"
        reason_result = audit_reasons(
            prediction_path, input_paths[group], reason_path
        )
        outputs[group] = {
            "predictionFile": prediction_path.name,
            "predictionSha256": sha256(prediction_path),
            "reasonAuditFile": reason_path.name,
            "reasonAuditSha256": sha256(reason_path),
            "testCount": prediction["testCount"],
            "base": {
                key: value
                for key, value in prediction["base"].items()
                if key != "cases"
            },
            "sft": {
                key: value
                for key, value in prediction["sft"].items()
                if key != "cases"
            },
            "sftReasonAudit": {
                "passCount": reason_result["passCount"],
                "reviewCount": reason_result["reviewCount"],
                "passRate": reason_result["passRate"],
            },
        }

    summary = {
        "schemaVersion": "p2j4-decision-sft-v4-eval-summary-v1",
        "trainingStarted": True,
        "combinedPredictionFile": combined_path.name,
        "combinedPredictionSha256": sha256(combined_path),
        "combinedRecordCount": len(all_ids),
        "groups": outputs,
        "notes": [
            "All metrics are from one remote Base/SFT v4 inference pass.",
            "Reason audit is deterministic and local; it does not call an external judge.",
            "No model inference is performed by this script.",
        ],
    }
    summary_path = bundle_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "PASS",
        "summary": str(summary_path),
        "combinedRecordCount": len(all_ids),
        "groups": {
            group: {
                "count": value["testCount"],
                "sftAction": value["sft"]["actionAccuracy"],
                "sftPolicy": value["sft"]["policyPassRate"],
                "sftReasonReview": value["sftReasonAudit"]["reviewCount"],
            }
            for group, value in outputs.items()
        },
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
