"""Audit real, training-eligible policy states for a future DPO v4.

This deliberately creates no DPO pairs and no training files.  It identifies
the real Trace states that have Runtime-like candidate widths and at least one
unperformed legal alternative.  Official Golden/Hard/Test/OOD evaluation cases
are never read as pair sources.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "evals" / "p2j4-decision-candidates-context-v5-20260824.json"
SPLIT = ROOT / "evals" / "p2j4-decision-family-split-v2-20260824.json"
OUT = ROOT / "evals" / "p2j4-dpo-v4-real-source-audit-20260825.json"

ACTION_ORDER = (
    "inspect_cohort", "execute_read_query", "compare_groups", "analyze_projection",
    "stratified_analysis", "adjust_confounders", "cross_project_validate",
    "cross_disease_validate", "retrieve_evidence", "finish",
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _train_trace_ids() -> set[str]:
    split = _read(SPLIT)
    return {
        item["candidate"]["sourceTraceId"]
        for item in split["splits"]["train"]
    }


def _candidate(raw: dict[str, Any]) -> dict[str, Any] | None:
    actions = raw["candidate_actions"]
    selected = raw["selected_action"]
    history = raw["history_actions"]
    if not 4 <= len(actions) <= 7 or selected not in actions or selected in history:
        return None
    alternatives = [action for action in actions if action != selected and action not in history]
    if not alternatives:
        return None
    return {
        "sourceTraceId": raw["sourceTraceId"],
        "sourceKind": "real_training_trace_v5",
        "taskKind": raw["task_kind"],
        "goalCode": raw["goal_code"],
        "taskFamily": raw["task_family"],
        "hardCaseClass": raw.get("hard_case_class"),
        "observationFlags": raw["observation_flags"],
        "historyActions": history,
        "candidateActions": actions,
        "selectedAction": selected,
        "legalUnperformedAlternatives": alternatives,
        "stateSummary": raw["state_summary"],
        "decisionReason": raw["decision_reason"],
        "stopReason": raw["stop_reason"],
        "reviewStatus": "PAIR_REVIEW_REQUIRED",
    }


def main() -> int:
    allowed_traces = _train_trace_ids()
    raw = _read(RAW)["candidates"]
    candidates = [item for item in (_candidate(value) for value in raw) if item and item["sourceTraceId"] in allowed_traces]
    by_action = Counter(item["selectedAction"] for item in candidates)
    by_width = Counter(len(item["candidateActions"]) for item in candidates)
    by_hard = Counter("hard_derived" if item["hardCaseClass"] else "ordinary_real_trace" for item in candidates)
    coverage = {action: by_action.get(action, 0) for action in ACTION_ORDER}
    target_feasible = {
        "inspect_cohort": 70, "execute_read_query": 30, "compare_groups": 80,
        "analyze_projection": 70, "stratified_analysis": 30, "adjust_confounders": 70,
        "cross_project_validate": 80, "cross_disease_validate": 50,
        "retrieve_evidence": 60, "finish": 60,
    }
    feasibility = {action: coverage[action] >= count for action, count in target_feasible.items()}
    payload = {
        "schemaVersion": "p2j4-dpo-v4-real-source-audit-v1",
        "status": "PAIR_REVIEW_REQUIRED",
        "purpose": "real Trace source audit only; not a DPO pair dataset",
        "source": {
            "rawCandidateCount": len(raw),
            "trainingEligibleTraceCount": len(allowed_traces),
            "excluded": ["Golden50", "Hard30 official eval", "Test70", "OOD30", "Runtime20"],
        },
        "candidateCount": len(candidates),
        "uniqueTraceCount": len({item["sourceTraceId"] for item in candidates}),
        "candidateWidthDistribution": dict(sorted(by_width.items())),
        "sourceComposition": dict(by_hard),
        "chosenActionCoverage": coverage,
        "proposedBalancedTarget": target_feasible,
        "proposedTargetFeasibleFromRealOnly": feasibility,
        "allActionsCovered": all(coverage.values()),
        "reviewRules": [
            "chosen action and rejected action must both be in candidateActions",
            "rejected action must be unperformed in historyActions",
            "pair reviewer must label preference as efficiency, evidence depth, stop boundary, or reason quality",
            "official evaluation cases are excluded even when their scenario type is reused",
        ],
        "candidates": candidates,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"], "candidateCount": payload["candidateCount"],
        "uniqueTraceCount": payload["uniqueTraceCount"], "chosenActionCoverage": coverage,
        "targetFeasible": feasibility,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
