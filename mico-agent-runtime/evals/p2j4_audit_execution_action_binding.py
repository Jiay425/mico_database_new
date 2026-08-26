"""Validate explicit Runtime action binding in a completed real-run artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def audit(input_path: Path, output_path: Path) -> dict[str, Any]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    traces = payload.get("traces")
    if not isinstance(traces, list) or not traces:
        raise ValueError("ACTION_BINDING_TRACES_MISSING")
    cases: list[dict[str, Any]] = []
    for trace in traces:
        events = trace.get("events", [])
        executed = trace.get("executedActions", [])
        decision_path = [item.get("chosenAction") for item in trace.get("decisions", [])]
        event_actions = [
            event.get("actionName") for event in events
            if event.get("node") == "execute_action"
        ]
        binding_ok = trace.get("executionActionBindingVersion") == "execution-action-v1"
        event_count_ok = len(event_actions) == len(executed)
        event_path_ok = event_actions == executed
        cases.append({
            "traceId": trace.get("traceId"),
            "bindingPresent": binding_ok,
            "eventCountMatches": event_count_ok,
            "eventPathMatches": event_path_ok,
            "decisionExecutionAgreement": decision_path == executed,
            "decisionPath": decision_path,
            "executedPath": executed,
        })
    result = {
        "schemaVersion": "p2j4-execution-action-binding-audit-v1",
        "sourceArtifact": str(input_path),
        "caseCount": len(cases),
        "bindingPassCount": sum(item["bindingPresent"] and item["eventCountMatches"] and item["eventPathMatches"] for item in cases),
        "decisionExecutionAgreementCount": sum(item["decisionExecutionAgreement"] for item in cases),
        "status": "PASS" if all(
            item["bindingPresent"] and item["eventCountMatches"] and item["eventPathMatches"]
            for item in cases
        ) else "FAIL",
        "cases": cases,
    }
    if output_path.exists():
        raise FileExistsError("ACTION_BINDING_OUTPUT_EXISTS")
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = audit(args.input, args.output)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "FAILED", "errorCode": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({key: result[key] for key in (
        "status", "caseCount", "bindingPassCount", "decisionExecutionAgreementCount"
    )}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
