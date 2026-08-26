"""Build reviewed, metadata-only Agent Decision training candidates.

This module never starts training and never accepts raw questions, SQL,
arguments, locators, payloads or document text as output fields.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pydantic import Field, StringConstraints
from typing import Annotated, Literal

from mico_agent_runtime.contracts.base import ClosedModel, Identifier
from mico_agent_runtime.contracts.research import ScientificActionName, StopReasonCode
from mico_agent_runtime.contracts.trace_eval import (
    BadCaseRecord,
    EvalScore,
    TraceProjection,
)


class AgentDecisionCandidate(ClosedModel):
    schemaVersion: Literal["p2j4-decision-dataset-v2"] = "p2j4-decision-dataset-v2"
    sourceTraceId: Identifier
    # New auditable State -> Action fields.  These use the exact names that
    # the Decision SFT export consumes.
    state_summary: Annotated[str, StringConstraints(min_length=1, max_length=512)]
    decision_reason: Annotated[str, StringConstraints(min_length=1, max_length=512)]
    selected_action: ScientificActionName
    alternative_actions: list[ScientificActionName] = Field(default_factory=list, max_length=9)
    stop_reason: StopReasonCode | None = None
    # Legacy metadata fields remain in v2 so existing audit readers can keep
    # consuming the same closed action/state vocabulary during migration.
    stateSummaryCode: Annotated[str, StringConstraints(pattern=r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+$")]
    allowedActions: list[ScientificActionName] = Field(min_length=1, max_length=10)
    chosenAction: ScientificActionName
    decisionCode: Annotated[str, StringConstraints(pattern=r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+$")]
    evidenceScope: Literal["observation_metadata_only"] = "observation_metadata_only"
    reviewStatus: Literal["passed", "accepted"]


def _accepted_trace_ids(
    scores: list[EvalScore],
    bad_cases: list[BadCaseRecord],
) -> dict[str, str]:
    result = {
        score.traceId: "passed"
        for score in scores
        if score.status == "PASS"
    }
    for record in bad_cases:
        if record.status == "ACCEPTED":
            result[record.traceId] = "accepted"
    return result


def build_decision_dataset(
    traces: list[TraceProjection],
    scores: list[EvalScore],
    bad_cases: list[BadCaseRecord],
) -> dict[str, Any]:
    """Return only candidates sourced from PASS or human-ACCEPTED traces."""

    score_by_trace = {score.traceId: score for score in scores}
    allowed_trace_ids = _accepted_trace_ids(scores, bad_cases)
    candidates: list[AgentDecisionCandidate] = []
    inferred_stop_reason_count = 0
    for trace in traces:
        review_status = allowed_trace_ids.get(trace.traceId)
        if review_status is None:
            continue
        score = score_by_trace.get(trace.traceId)
        if score is None:
            raise ValueError("DECISION_DATASET_SCORE_MISSING")
        for decision in trace.decisions:
            stop_reason = decision.stop_reason
            if (
                decision.chosenAction == "finish"
                and stop_reason is None
                and trace.stopReasonCode is not None
            ):
                # Historical baseline traces may predate the snake_case
                # terminal field even though the projection already carries
                # the closed trace-level stop reason. Bind that reason back
                # to the terminal State -> Action sample instead of exporting
                # an ambiguous ``finish`` decision.
                stop_reason = trace.stopReasonCode
                inferred_stop_reason_count += 1
            candidates.append(AgentDecisionCandidate(
                sourceTraceId=trace.traceId,
                state_summary=decision.state_summary,
                decision_reason=decision.decision_reason,
                selected_action=decision.selected_action or decision.chosenAction,
                alternative_actions=decision.alternative_actions,
                stop_reason=stop_reason,
                stateSummaryCode=decision.observationStateCode,
                allowedActions=decision.allowedActions,
                chosenAction=decision.chosenAction,
                decisionCode=decision.decisionCode or "DECISION_CODE_UNSPECIFIED",
                reviewStatus=review_status,
            ))
    return {
        "schemaVersion": "p2j4-decision-dataset-v2",
        "candidateCount": len(candidates),
        "sourceTraceCount": len({candidate.sourceTraceId for candidate in candidates}),
        "trainingStarted": False,
        "candidateFieldRepairs": {
            "inferredStopReasonCount": inferred_stop_reason_count,
            "inferredStopReasonSource": "trace.stopReasonCode",
        },
        "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
    }


def build_from_runner_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Parse only the already-redacted runner sections."""

    traces = [TraceProjection.model_validate(item) for item in payload.get("traces", [])]
    scores = [EvalScore.model_validate(item) for item in payload.get("scores", [])]
    bad_cases = [BadCaseRecord.model_validate(item) for item in payload.get("badCases", [])]
    return build_decision_dataset(traces, scores, bad_cases)


def build_from_runner_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """Build one candidate review set from multiple immutable run payloads."""

    traces: list[TraceProjection] = []
    scores: list[EvalScore] = []
    bad_cases: list[BadCaseRecord] = []
    trace_ids: set[str] = set()
    score_ids: set[str] = set()
    for payload in payloads:
        for item in payload.get("traces", []):
            trace = TraceProjection.model_validate(item)
            if trace.traceId in trace_ids:
                raise ValueError("DECISION_DATASET_DUPLICATE_TRACE")
            trace_ids.add(trace.traceId)
            traces.append(trace)
        for item in payload.get("scores", []):
            score = EvalScore.model_validate(item)
            if score.traceId in score_ids:
                raise ValueError("DECISION_DATASET_DUPLICATE_SCORE")
            score_ids.add(score.traceId)
            scores.append(score)
        bad_cases.extend(
            BadCaseRecord.model_validate(item)
            for item in payload.get("badCases", [])
        )
    return build_decision_dataset(traces, scores, bad_cases)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build reviewed P2-J4 decision candidates")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.input]
        result = build_from_runner_payloads(payloads)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "INVALID", "errorCode": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({"status": "READY", "candidateCount": result["candidateCount"], "trainingStarted": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
