"""Build a single-case task repair without mutating the frozen task set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    matches = [case for case in payload.get("cases", []) if case.get("caseId") == args.case_id]
    if len(matches) != 1:
        raise ValueError("POLICY_SOURCE_REPAIR_CASE_NOT_UNIQUE")
    repaired = dict(matches[0])
    repaired["question"] = (
        repaired["question"]
        + " Use two independent bounded evidence observations before stopping."
    )
    output = {
        "schemaVersion": payload["schemaVersion"],
        "name": payload["name"] + " single-case contract repair",
        "purpose": "development_only_model_participation_required_task_repair",
        "caseCount": 1,
        "reviewStatus": "REPAIRED_CASE_CONTRACT",
        "kindDistribution": {repaired["kind"]: 1},
        "trainingStarted": False,
        "cases": [repaired],
        "repairAudit": {
            "sourceTaskSet": args.input.name,
            "caseId": args.case_id,
            "reason": "make_minEvidenceBindings_obligation_explicit",
        },
    }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"caseId": args.case_id, "status": "READY"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
