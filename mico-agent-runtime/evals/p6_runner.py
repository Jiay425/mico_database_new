from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from mico_agent_runtime.governance.guardrails import (
    evaluate_input,
    evaluate_output,
    evaluate_workflow,
)


def evaluate_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload["cases"]
    by_kind: Counter[str] = Counter()
    passed = 0
    failed_case_ids: list[str] = []
    for case in cases:
        kind = case["kind"]
        by_kind[kind] += 1
        if kind == "input":
            decision = evaluate_input(case["value"])
        elif kind == "output":
            decision = evaluate_output(case["value"])
        elif kind == "tool":
            decision = evaluate_workflow(
                case["workflow"],
                case["allowedWorkflows"],
                case["requestedScopes"],
                set(case["requiredScopes"]),
            )
        else:
            failed_case_ids.append(case["caseId"])
            continue
        if decision.verdict == case["expectedVerdict"] and decision.code == case["expectedCode"]:
            passed += 1
        else:
            failed_case_ids.append(case["caseId"])
    return {
        "schemaVersion": payload["schemaVersion"],
        "total": len(cases),
        "passed": passed,
        "failed": len(failed_case_ids),
        "byKind": dict(sorted(by_kind.items())),
        "failedCaseIds": failed_case_ids,
    }


if __name__ == "__main__":
    manifest = Path(__file__).with_name("p6-eval-cases-v1.json")
    print(json.dumps(evaluate_manifest(manifest), ensure_ascii=False, sort_keys=True))
