"""Audit Decision SFT v3 context, review, and family-split artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from evals.p2j4_decision_context import DecisionSftCandidate
from evals.p2j4_build_context_review_set import SIGNATURE_FIELDS, _signature


FORBIDDEN_KEYS = {
    "question",
    "sql",
    "query",
    "arguments",
    "payload",
    "rawQuestion",
    "rawSql",
    "rawPayload",
}


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("DECISION_CONTEXT_AUDIT_PAYLOAD_INVALID")
    return payload


def _forbidden_paths(value: Any, path: str = "$", result: list[str] | None = None) -> list[str]:
    result = result if result is not None else []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_KEYS:
                result.append(f"{path}.{key}")
            _forbidden_paths(child, f"{path}.{key}", result)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _forbidden_paths(child, f"{path}[{index}]", result)
    return result


def audit(candidate_path: Path, review_path: Path, split_path: Path) -> dict[str, Any]:
    candidate_payload = _read(candidate_path)
    review_payload = _read(review_path)
    split_payload = _read(split_path)

    candidates = [
        DecisionSftCandidate.model_validate(item)
        for item in candidate_payload.get("candidates", [])
    ]
    candidate_forbidden = _forbidden_paths(candidate_payload)
    review_forbidden = _forbidden_paths(review_payload)
    split_forbidden = _forbidden_paths(split_payload)

    signatures = {_signature(candidate) for candidate in candidates}
    review_items = review_payload.get("items", [])
    review_signatures = {
        tuple(
            tuple(item["candidate"][field])
            if isinstance(item["candidate"][field], list)
            else item["candidate"][field]
            for field in SIGNATURE_FIELDS
        )
        for item in review_items
    }
    review_signature_hashes = {item["signatureHash"] for item in review_items}
    split_items = [
        item
        for values in split_payload.get("splits", {}).values()
        for item in values
    ]
    split_signature_hashes = {item["signatureHash"] for item in split_items}

    checks = {
        "candidateSchemaValid": len(candidates) == candidate_payload.get("candidateCount"),
        "candidateNoForbiddenKeys": not candidate_forbidden,
        "reviewNoForbiddenKeys": not review_forbidden,
        "splitNoForbiddenKeys": not split_forbidden,
        "candidateContextSignatureCount": len(signatures) == review_payload.get(
            "contextExactSignatureCount"
        ),
        "reviewMatchesContextSignatures": review_signatures.issubset(signatures),
        "reviewHasUniqueSignatureHashes": len(review_signature_hashes) == len(review_items),
        "reviewHardCoverageComplete": review_payload.get("hardCaseCoverageComplete") is True,
        "splitCoversReviewSet": split_signature_hashes == review_signature_hashes,
        "splitFamilyDisjoint": split_payload.get("familyDisjoint") is True,
        "splitSourceTraceDisjoint": split_payload.get("sourceTraceDisjoint") is True,
        "trainingNotStarted": (
            candidate_payload.get("trainingStarted") is False
            and review_payload.get("trainingStarted") is False
            and split_payload.get("trainingStarted") is False
        ),
    }
    status = "PASS" if all(checks.values()) else "REVIEW_REQUIRED"
    return {
        "schemaVersion": "p2j4-decision-context-audit-v1",
        "status": status,
        "trainingStarted": False,
        "sourceArtifacts": {
            "candidate": candidate_path.name,
            "reviewSet": review_path.name,
            "familySplit": split_path.name,
        },
        "candidateCount": len(candidates),
        "contextExactSignatureCount": len(signatures),
        "reviewItemCount": len(review_items),
        "splitItemCount": len(split_items),
        "taskKindCounts": dict(Counter(candidate.task_kind for candidate in candidates)),
        "goalCounts": dict(Counter(candidate.goal_code for candidate in candidates)),
        "taskFamilyCounts": dict(Counter(candidate.task_family for candidate in candidates)),
        "checks": checks,
        "forbiddenKeyPaths": {
            "candidate": candidate_forbidden,
            "reviewSet": review_forbidden,
            "familySplit": split_forbidden,
        },
        "releaseGate": {
            "decisionSftFrozen": False,
            "ownerReviewRequired": True,
            "taskFamilySplitRequired": True,
            "sftDpoGrpoStarted": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit P2-J4 Decision SFT context assets")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--review-set", type=Path, required=True)
    parser.add_argument("--family-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = audit(args.candidate, args.review_set, args.family_split)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "INVALID", "errorCode": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({
        "status": result["status"],
        "candidateCount": result["candidateCount"],
        "contextExactSignatureCount": result["contextExactSignatureCount"],
        "reviewItemCount": result["reviewItemCount"],
        "splitItemCount": result["splitItemCount"],
        "trainingStarted": result["trainingStarted"],
        "output": str(args.output),
    }, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
