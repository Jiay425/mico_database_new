"""Backfill safe Goal/Observation context into Decision SFT v3 candidates."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from evals.p2j4_decision_context import (
    DecisionSftCandidate,
    derive_goal_code,
    derive_observation_flags,
    derive_task_family,
    normalize_task_kind,
)


_STATE_SUMMARY_RE = re.compile(
    r"observation_state=(?P<state>[^;]+);\s*"
    r"evidence_bindings=(?P<bindings>\d+);\s*"
    r"source_routes=(?P<routes>[^;]+);\s*"
    r"prior_actions=(?P<prior>.*)$"
)


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("DECISION_CONTEXT_PAYLOAD_INVALID")
    return payload


def _task_index(paths: list[Path]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for path in paths:
        for item in _read(path).get("cases", []):
            result[item["caseId"]] = {
                "kind": item.get("kind", "open_exploration"),
                "question": item.get("question", ""),
            }
    return result


def _hard_index(paths: list[Path]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in paths:
        for item in _read(path).get("cases", []):
            result[item["caseId"]] = item["hardCaseClass"]
    return result


def _repair_state_summary_history(state_summary: str, history_actions: list[str]) -> tuple[str, bool]:
    """Bind the closed preceding-action sequence into legacy summaries.

    Historical traces sometimes persisted ``prior_actions=none`` before the
    decision projection was migrated, while the immutable decision sequence
    still contains the correct preceding actions.  The repaired value is
    derived only from closed trace fields and never from user content.
    """

    match = _STATE_SUMMARY_RE.fullmatch(state_summary.strip())
    if match is None:
        return state_summary, False
    expected = ",".join(history_actions) if history_actions else "none"
    current = match.group("prior") or "none"
    if current == expected:
        return state_summary, False
    return (
        f"observation_state={match.group('state')}; "
        f"evidence_bindings={match.group('bindings')}; "
        f"source_routes={match.group('routes')}; "
        f"prior_actions={expected}",
        True,
    )


def build(
    candidate_path: Path,
    source_paths: list[Path],
    task_paths: list[Path],
    hard_spec_paths: list[Path],
) -> dict[str, Any]:
    candidate_payload = _read(candidate_path)
    source_payloads = [_read(path) for path in source_paths]
    task_by_case = _task_index(task_paths)
    hard_by_case = _hard_index(hard_spec_paths)

    source_by_trace: dict[str, dict[str, Any]] = {}
    for payload in source_payloads:
        traces = {trace["traceId"]: trace for trace in payload.get("traces", [])}
        for score in payload.get("scores", []):
            trace = traces.get(score["traceId"])
            if trace is None:
                raise ValueError("DECISION_CONTEXT_TRACE_MISSING:" + score["traceId"])
            source_by_trace[score["traceId"]] = {
                "caseId": score["caseId"],
                "trace": trace,
            }

    cursors: defaultdict[str, int] = defaultdict(int)
    candidates: list[DecisionSftCandidate] = []
    unknown_task_cases: set[str] = set()
    history_mismatch_count = 0
    state_summary_history_repair_count = 0
    goal_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    flag_counts: Counter[str] = Counter()

    for raw in candidate_payload.get("candidates", []):
        source = source_by_trace.get(raw["sourceTraceId"])
        if source is None:
            raise ValueError("DECISION_CONTEXT_SOURCE_TRACE_MISSING:" + raw["sourceTraceId"])
        case_id = source["caseId"]
        task_meta = task_by_case.get(case_id)
        if task_meta is None:
            unknown_task_cases.add(case_id)
            task_meta = {"kind": "open_exploration", "question": ""}
        hard_case_class = hard_by_case.get(case_id)
        task_kind = normalize_task_kind(task_meta["kind"])
        goal_code = derive_goal_code(task_meta["question"], task_kind, hard_case_class)
        task_family = derive_task_family(task_kind, goal_code, hard_case_class)
        if case_id.startswith("p2j4-state-diff-v2-"):
            task_family = f"{task_family}:state_difference_v2"

        trace_decisions = source["trace"].get("decisions", [])
        cursor = cursors[raw["sourceTraceId"]]
        matched_index = None
        for index in range(cursor, len(trace_decisions)):
            decision = trace_decisions[index]
            if decision.get("chosenAction") == raw["chosenAction"]:
                matched_index = index
                break
        if matched_index is None:
            history_actions: list[str] = []
            history_mismatch_count += 1
        else:
            history_actions = [
                item.get("chosenAction")
                for item in trace_decisions[:matched_index]
                if item.get("chosenAction")
            ]
            cursors[raw["sourceTraceId"]] = matched_index + 1

        flags = derive_observation_flags(
            state_summary=raw["state_summary"],
            history_actions=history_actions,
            allowed_actions=raw["allowedActions"],
            question=task_meta["question"],
            goal_code=goal_code,
            hard_case_class=hard_case_class,
        )
        state_summary, state_summary_repaired = _repair_state_summary_history(
            raw["state_summary"], history_actions
        )
        if state_summary_repaired:
            state_summary_history_repair_count += 1
        candidate = DecisionSftCandidate(
            sourceTraceId=raw["sourceTraceId"],
            task_kind=task_kind,
            goal_code=goal_code,
            task_family=task_family,
            hard_case_class=hard_case_class,
            observation_flags=flags,
            history_actions=history_actions,
            candidate_actions=raw["allowedActions"],
            state_summary=state_summary,
            decision_reason=raw["decision_reason"],
            selected_action=raw["selected_action"],
            alternative_actions=raw["alternative_actions"],
            stop_reason=raw["stop_reason"],
            reviewStatus=raw["reviewStatus"],
        )
        candidates.append(candidate)
        goal_counts[goal_code] += 1
        family_counts[task_family] += 1
        flag_counts.update(flags)

    return {
        "schemaVersion": "p2j4-decision-dataset-v3",
        "sourceCandidateArtifact": candidate_path.name,
        "candidateCount": len(candidates),
        "sourceTraceCount": len({candidate.sourceTraceId for candidate in candidates}),
        "trainingStarted": False,
        "contextContract": {
            "fields": [
                "task_kind", "goal_code", "task_family", "hard_case_class",
                "observation_flags", "history_actions", "candidate_actions",
                "state_summary", "decision_reason", "selected_action",
            ],
            "rawQuestionIncluded": False,
            "rawSqlIncluded": False,
            "rawPayloadIncluded": False,
        },
        "fieldDerivation": {
            "taskKindSource": "task-set.kind",
            "goalCodeSource": "closed taxonomy over task-set question; question not exported",
            "observationFlagsSource": "task metadata + decision state summary + action history",
            "historyActionsSource": "preceding TraceDecision.chosenAction values",
            "candidateActionsSource": "TraceDecision.allowedActions",
            "historyMismatchCount": history_mismatch_count,
            "stateSummaryHistoryRepairCount": state_summary_history_repair_count,
            "unknownTaskCaseCount": len(unknown_task_cases),
            "unknownTaskCases": sorted(unknown_task_cases),
        },
        "goalCounts": dict(goal_counts.most_common()),
        "taskFamilyCounts": dict(family_counts.most_common()),
        "observationFlagCounts": dict(flag_counts.most_common()),
        "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build P2-J4 Decision SFT v3 context candidates")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--task-set", type=Path, action="append", required=True)
    parser.add_argument("--hard-spec", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = build(args.candidates, args.source, args.task_set, args.hard_spec)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "INVALID", "errorCode": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({
        "status": "READY",
        "candidateCount": result["candidateCount"],
        "sourceTraceCount": result["sourceTraceCount"],
        "historyMismatchCount": result["fieldDerivation"]["historyMismatchCount"],
        "unknownTaskCaseCount": result["fieldDerivation"]["unknownTaskCaseCount"],
        "trainingStarted": False,
        "output": str(args.output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
