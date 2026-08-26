"""Offline migration for the P2-J4 Decision Trace v2 fields.

The migration only reads an already-redacted runner payload.  It does not
start a runtime, call a model, or access Java/database services.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mico_agent_runtime.contracts.trace_eval import TraceDecision, TraceProjection


def _state_summary(trace_decision: TraceDecision, prior_actions: list[str]) -> str:
    routes = ",".join(trace_decision.observationSourceRoutes) or "none"
    prior = ",".join(prior_actions) or "none"
    return (
        f"observation_state={trace_decision.observationStateCode}; "
        f"evidence_bindings={trace_decision.observationEvidenceBindingCount}; "
        f"source_routes={routes}; prior_actions={prior}"
    )


def _finish_reason(reason: str | None) -> str:
    return {
        "EVIDENCE_SUFFICIENT": "validated evidence is sufficient for the bounded task",
        "NO_NEW_INFORMATION": "the bounded loop has no new information to add",
        "QUALITY_RISK": "the remaining evidence quality risk requires a bounded stop",
        "ACTION_BUDGET_EXHAUSTED": "the configured action budget has been reached",
        "UPSTREAM_REJECTED": "an upstream policy or contract rejection requires a safe stop",
        "UNSUPPORTED_ACTION": "no supported action remains for the current state",
        "USER_REQUESTED_STOP": "the user requested that the exploration stop",
    }.get(reason or "", "finish the bounded runtime exploration")


def migrate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    migrated = dict(payload)
    output_traces: list[dict[str, Any]] = []
    for raw_trace in payload.get("traces", []):
        trace = TraceProjection.model_validate(raw_trace)
        prior_actions: list[str] = []
        decisions: list[TraceDecision] = []
        for decision in trace.decisions:
            stop_reason = trace.stopReasonCode if decision.chosenAction == "finish" else None
            decision_reason = decision.decision_reason
            if decision.chosenAction == "finish" and decision_reason == "select the next allow-listed runtime action":
                decision_reason = _finish_reason(stop_reason)
            decisions.append(decision.model_copy(update={
                "state_summary": _state_summary(decision, prior_actions),
                "decision_reason": decision_reason,
                "selected_action": decision.chosenAction,
                "alternative_actions": [
                    action for action in decision.allowedActions
                    if action != decision.chosenAction
                ],
                "stop_reason": stop_reason,
            }))
            prior_actions.append(decision.chosenAction)
        migrated_trace = trace.model_copy(update={
            "decisionTraceVersion": "decision-trace-v2",
            "decisions": decisions,
        })
        output_traces.append(migrated_trace.model_dump(mode="json"))
    migrated["traces"] = output_traces
    migrated["decisionTraceMigration"] = {
        "version": "decision-trace-v2",
        "offline": True,
        "modelCalls": 0,
        "trainingStarted": False,
    }
    return migrated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Migrate redacted P2-J4 traces to Decision Trace v2")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        result = migrate_payload(payload)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "INVALID", "errorCode": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({
        "status": "READY",
        "traceCount": len(result.get("traces", [])),
        "modelCalls": 0,
        "trainingStarted": False,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
