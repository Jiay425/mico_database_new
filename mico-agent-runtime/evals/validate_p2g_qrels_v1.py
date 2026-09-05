"""Validate the 200-case qrels artifact before it can enter model selection."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


TARGETS = {"single_fact": 50, "relation": 60, "multi_hop": 50, "document_synthesis": 40}


def validate(payload: dict) -> list[str]:
    errors: list[str] = []
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != 200:
        errors.append("caseCount must be exactly 200")
        return errors
    counts = Counter(case.get("category") for case in cases)
    if dict(counts) != TARGETS:
        errors.append(f"category counts must be {TARGETS}, got {dict(counts)}")
    case_ids = [case.get("caseId") for case in cases]
    if len(case_ids) != len(set(case_ids)):
        errors.append("caseId values must be unique")
    for case in cases:
        if not case.get("query") or not case.get("qrels"):
            errors.append(f"{case.get('caseId')}: query and qrels are required")
        for qrel in case.get("qrels", []):
            if not qrel.get("chunkId"):
                errors.append(f"{case.get('caseId')}: qrel chunkId is required")
            if qrel.get("provenanceGrade") not in {0, 1, 2, 3}:
                errors.append(f"{case.get('caseId')}: provenanceGrade must be 0..3")
            if qrel.get("humanGrade") is not None and qrel.get("humanGrade") not in {0, 1, 2, 3}:
                errors.append(f"{case.get('caseId')}: humanGrade must be 0..3 or null")
            if qrel.get("humanGrade") is not None and not qrel.get("reviewerId"):
                errors.append(f"{case.get('caseId')}: humanGrade requires reviewerId")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("qrels", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.qrels.read_text(encoding="utf-8"))
    errors = validate(payload)
    if errors:
        raise SystemExit("QRELS_INVALID: " + "; ".join(errors[:10]))
    print(json.dumps({
        "status": "VALID",
        "caseCount": len(payload["cases"]),
        "caseCountsByCategory": dict(Counter(case["category"] for case in payload["cases"])),
        "humanAdjudicationStatus": payload.get("humanAdjudicationStatus"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
