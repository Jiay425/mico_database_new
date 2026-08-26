"""Merge context-enriched Decision candidate pools with trace-level uniqueness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evals.p2j4_decision_context import DecisionSftCandidate


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schemaVersion") != "p2j4-decision-dataset-v3":
        raise ValueError("DECISION_CONTEXT_CANDIDATE_POOL_INVALID:" + str(path))
    return payload


def merge(inputs: list[Path], output: Path) -> dict[str, Any]:
    candidates: list[DecisionSftCandidate] = []
    trace_ids: set[str] = set()
    raw_candidate_count = 0
    source_artifacts: list[str] = []
    for path in inputs:
        payload = _read(path)
        source_artifacts.append(path.name)
        raw_candidate_count += int(payload.get("candidateCount", 0))
        for raw in payload.get("candidates", []):
            candidate = DecisionSftCandidate.model_validate(raw)
            candidates.append(candidate)
            trace_ids.add(candidate.sourceTraceId)

    result = {
        "schemaVersion": "p2j4-decision-dataset-v3",
        "sourceCandidateArtifacts": source_artifacts,
        "candidateCount": len(candidates),
        "sourceTraceCount": len(trace_ids),
        "rawCandidateCount": raw_candidate_count,
        "trainingStarted": False,
        "contextContract": {
            "fields": [
                "task_kind", "goal_code", "task_family", "hard_case_class",
                "observation_flags", "history_actions", "candidate_actions",
                "state_summary", "decision_reason", "selected_action",
                "alternative_actions", "stop_reason",
            ],
            "rawQuestionIncluded": False,
            "rawSqlIncluded": False,
            "rawPayloadIncluded": False,
        },
        "candidatePoolsMerged": len(inputs),
        "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "status": "READY",
        "candidateCount": len(candidates),
        "sourceTraceCount": len(trace_ids),
        "rawCandidateCount": raw_candidate_count,
        "trainingStarted": False,
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
