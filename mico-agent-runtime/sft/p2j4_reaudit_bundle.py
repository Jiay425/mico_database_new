"""Recompute deterministic reason audits for an existing model eval artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from p2j4_audit_decision_reasons import _audit_case


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Re-audit existing bundle predictions")
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload: dict[str, Any] = json.loads(args.eval.read_text(encoding="utf-8"))
    records = {
        record["id"]: record
        for record in (
            json.loads(line)
            for line in args.records.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    group_ids = {
        name: {
            record_id for record_id, metadata in (manifest.get("records") or {}).items()
            if name in (metadata.get("groups") or [])
        }
        for name in (manifest.get("groups") or {})
    }
    for model_name in ("base", "sft"):
        cases = payload[model_name]["cases"]
        for case in cases:
            audit = _audit_case(records[case["record_id"]], case)
            case["reason_audit"] = {
                "status": audit["status"],
                "checks": audit["checks"],
                "review_reasons": audit["review_reasons"],
            }
        reviews = sum(case["reason_audit"]["status"] != "PASS" for case in cases)
        metrics = payload[model_name]["overall"]
        metrics["reasonAuditReviewCount"] = reviews
        metrics["reasonAuditPassCount"] = len(cases) - reviews
        metrics["reasonAuditPassRate"] = (len(cases) - reviews) / len(cases) if cases else 0.0
        for group_name, group_metrics in payload[model_name]["groups"].items():
            ids = group_ids.get(group_name, set())
            group_cases = [case for case in cases if case["record_id"] in ids]
            if not group_cases:
                raise ValueError("BUNDLE_GROUP_CASES_MISSING:" + group_name)
            group_reviews = sum(case["reason_audit"]["status"] != "PASS" for case in group_cases)
            group_metrics["reasonAuditReviewCount"] = group_reviews
            group_metrics["reasonAuditPassCount"] = len(group_cases) - group_reviews
            group_metrics["reasonAuditPassRate"] = (len(group_cases) - group_reviews) / len(group_cases)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
