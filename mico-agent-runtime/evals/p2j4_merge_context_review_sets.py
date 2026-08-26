"""Merge immutable context review sets without reintroducing duplicate signatures."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schemaVersion") != "p2j4-decision-review-set-v2":
        raise ValueError("DECISION_REVIEW_SET_INVALID:" + str(path))
    return payload


def merge(inputs: list[Path], output: Path) -> dict[str, Any]:
    selected: dict[str, dict[str, Any]] = {}
    raw_candidate_count = 0
    source_artifacts: list[str] = []
    hard_case_ids: set[str] = set()
    hard_case_count = 0

    for path in inputs:
        payload = _read(path)
        raw_candidate_count += int(payload.get("rawCandidateCount", 0))
        source_artifacts.append(path.name)
        hard_case_count = max(hard_case_count, int(payload.get("hardCaseCount", 0)))
        for item in payload.get("items", []):
            signature_hash = item["signatureHash"]
            incoming = json.loads(json.dumps(item, ensure_ascii=False))
            hard_case_ids.update(incoming.get("hardCaseIds", []))
            existing = selected.get(signature_hash)
            if existing is None:
                selected[signature_hash] = incoming
                continue
            existing["replicationCount"] += incoming.get("replicationCount", 0)
            existing["sourceTraceCount"] += incoming.get("sourceTraceCount", 0)
            existing["sourceRuns"] = sorted(set(existing.get("sourceRuns", [])) | set(incoming.get("sourceRuns", [])))
            existing["taskFamilies"] = sorted(set(existing.get("taskFamilies", [])) | set(incoming.get("taskFamilies", [])))
            existing["hardCaseIds"] = sorted(set(existing.get("hardCaseIds", [])) | set(incoming.get("hardCaseIds", [])))
            merged_classes = Counter(existing.get("hardClassCounts", {}))
            merged_classes.update(incoming.get("hardClassCounts", {}))
            existing["hardClassCounts"] = dict(merged_classes)
            existing["reviewReasons"] = list(dict.fromkeys(
                existing.get("reviewReasons", []) + incoming.get("reviewReasons", [])
            ))

    items = sorted(
        selected.values(),
        key=lambda item: (
            0 if "hard_case_coverage" in item.get("reviewReasons", []) else 1,
            item["candidate"]["task_family"],
            item["candidate"]["selected_action"],
            item["signatureHash"],
        ),
    )
    result = {
        "schemaVersion": "p2j4-decision-review-set-v2",
        "sourceReviewSets": source_artifacts,
        "status": "OWNER_REVIEW_PENDING",
        "trainingStarted": False,
        "rawCandidateCount": raw_candidate_count,
        "reviewItemCount": len(items),
        "contextExactSignatureCount": len(items),
        "hardCaseCount": hard_case_count,
        "hardCaseCoverageCount": len(hard_case_ids),
        "hardCaseCoverageComplete": len(hard_case_ids) >= hard_case_count,
        "contextContract": {
            "signatureFields": [
                "task_kind", "goal_code", "task_family", "hard_case_class",
                "observation_flags", "history_actions", "candidate_actions",
                "state_summary", "decision_reason", "selected_action",
                "alternative_actions", "stop_reason",
            ],
            "rawQuestionIncluded": False,
            "rawSqlIncluded": False,
            "rawPayloadIncluded": False,
        },
        "reviewPolicy": {
            "oneContextExactSignatureRepresentative": True,
            "oneInformativeRepresentativePerHardCase": True,
            "mergedBySignatureHash": True,
            "rawPoolsRetainedAsAuditEvidence": True,
            "noTrainingStarted": True,
        },
        "items": items,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "status": result["status"],
        "rawCandidateCount": raw_candidate_count,
        "reviewItemCount": len(items),
        "contextExactSignatureCount": len(items),
        "hardCaseCoverageComplete": result["hardCaseCoverageComplete"],
        "output": str(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = merge(args.input, args.output)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "INVALID", "errorCode": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
