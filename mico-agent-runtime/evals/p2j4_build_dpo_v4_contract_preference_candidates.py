"""Build contract-derived action preference candidates from PASS model traces.

This is still a review/staging artifact.  It derives a preference only when
the model's final action equals the next action required by the task contract;
it never treats an arbitrary allowed alternative as inferior.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TRACE_FILES = (
    ROOT / "evals" / "p2j4-dpo-v4-targeted-weak-action-runs-final-model-pass-v1.json",
    ROOT / "evals" / "p2j4-dpo-v4-targeted-weak-action-runs-v2-final-pass-v1.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-collection-runs-v3-final-pass.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-collection-runs-v4-final-pass.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-cross-only-runs-v5-final-pass.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-collection-runs-v6-final-pass.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-collection-runs-v7-final-pass.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-collection-runs-v8-final-pass.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-collection-runs-v9-gemini-canary-v2.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-collection-runs-v9-gemini-36-canary.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-collection-runs-v9-remaining-gemini36.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-collection-runs-v9-remaining-deepseek.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-collection-runs-v10-gemini-lite-canary.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-collection-runs-v10-remaining-gemini-lite.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-collection-runs-v11-project-deepseek-canary-fixed-01.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-collection-runs-v11-project-deepseek-canary-fixed-02-03.json",
    # v12 records enter separately only after per-case PASS/fallback audit.
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v12-adjust-deepseek-canary-01-repair.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v12-adjust-deepseek-canary-03.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v12-project-deepseek-canary-04.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v12-project-deepseek-canary-06.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v12-disease-deepseek-canary-07.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v12-disease-deepseek-canary-09.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-adjust-deepseek-canary-01.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-adjust-deepseek-canary-02.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-adjust-deepseek-canary-03.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-project-deepseek-canary-01.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-project-deepseek-canary-02.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-disease-deepseek-canary-02.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-disease-deepseek-canary-03.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-batch-01-adjust-04.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-batch-01-adjust-05-repair.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-batch-01-adjust-06.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-batch-01-project-04.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-batch-01-project-06.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-batch-01-disease-04.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-batch-01-disease-05.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-batch-01-disease-06.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-batch-02-adjust-08.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-batch-02-adjust-09.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-batch-02-project-07.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-batch-02-project-08.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-batch-02-disease-07.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-batch-02-disease-08.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-batch-02-disease-09.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-batch-03-adjust-10.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-batch-03-adjust-11.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-batch-03-adjust-12-repair.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-batch-03-project-10.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-batch-04-disease-10.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-batch-04-disease-11.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-batch-04-disease-12.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-batch-05-adjust-16.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-batch-05-adjust-17.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-batch-05-adjust-19.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-batch-06-disease-13.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-batch-06-disease-14.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-batch-07-disease-15.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-batch-07-disease-18.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-batch-08-repair-adjust-07.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-batch-08-repair-project-13-v2.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-batch-08-repair-project-14.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-batch-08-repair-project-15.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-batch-08-repair2-project-17.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-batch-08-repair-project-canary-03.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-batch-08-repair2-disease-17.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-batch-08-repair3-disease-16.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-batch-08-repair3-disease-canary-01.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-batch-08-repair4-adjust-18.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-batch-08-repair4-project-18.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-batch-08-repair5-project-05.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-batch-08-repair5-project-09.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-batch-08-repair5-project-11.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-batch-08-repair5-project-12.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-batch-08-repair5-adjust-15.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-batch-08-repair5-adjust-20.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-batch-08-project-19.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-batch-08-project-20.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-batch-08-disease-19.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13-batch-08-repair-disease-20.json",
) + tuple(sorted(
    (ROOT / "evals").glob("p2j4-dpo-v4-weak-action-source-runs-v14-*.json")
))
TASK_FILES = (
    ROOT / "evals" / "p2j4-dpo-v4-targeted-weak-action-task-set-v1.json",
    ROOT / "evals" / "p2j4-dpo-v4-targeted-weak-action-task-set-v2.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-collection-task-set-v3.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-collection-task-set-v4.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-cross-only-task-set-v5.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-collection-task-set-v6.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-collection-task-set-v7.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-collection-task-set-v8.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-collection-task-set-v9.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-collection-task-set-v10.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-collection-task-set-v11.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-task-set-v12.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-task-set-v13.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-task-set-v13-repair-disease-01.json",
    ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-task-set-v13-repair-disease-20.json",
    ROOT / "evals" / "p2j4-dpo-v4-weak-action-source-task-set-v14.json",
    ROOT / "evals" / "p2j4-dpo-v4-weak-action-source-task-set-v14-repair-compare-09.json",
    ROOT / "evals" / "p2j4-dpo-v4-weak-action-source-task-set-v14-repair-projection-01.json",
    ROOT / "evals" / "p2j4-dpo-v4-weak-action-source-task-set-v14-repair-projection-05-07.json",
)


def _planner_source_metadata(payload: dict[str, Any], path: Path) -> dict[str, str | None]:
    """Persist provider provenance without inferring it from a filename alone."""
    services = payload.get("servicesObserved")
    services = services if isinstance(services, dict) else {}
    model = services.get("plannerModel")
    provider = services.get("plannerProvider")
    # Only the new v12/v13 DeepSeek collection series has an explicit
    # operational record: Gemini was unavailable/rate-limited and the user
    # directed DeepSeek Flash.  Historical sources remain review-required.
    switch_reason = None
    if (
        isinstance(model, str)
        and model == "deepseek-v4-flash"
        and ("state-matrix-runs-v12" in path.name or "state-matrix-runs-v13" in path.name)
    ):
        switch_reason = "GEMINI_UNAVAILABLE_OR_RATE_LIMITED;USER_DIRECTED_DEEPSEEK_FLASH"
    return {
        "planner_provider": provider if isinstance(provider, str) else None,
        "planner_model": model if isinstance(model, str) else None,
        "planner_switch_reason": switch_reason,
    }


def _hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _tasks() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in TASK_FILES:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for task in payload.get("cases", []):
            result[task["caseId"]] = task
    return result


def _history(summary: str) -> list[str]:
    match = re.search(r"prior_actions=(.*?)(?:; stop_reason=|$)", summary or "")
    if not match or match.group(1) == "none":
        return []
    return [item for item in match.group(1).split(",") if item]


def _next_required(task: dict[str, Any], history: list[str]) -> str | None:
    required = list(task.get("requiredActions", []))
    for action in required:
        if action not in history:
            return action
    return "finish" if "finish" in task.get("allowedActions", []) else None


def _dimension(chosen: str) -> tuple[str, list[str]]:
    if chosen == "finish":
        return "stop_boundary", ["EVIDENCE_OBLIGATION_SATISFIED"]
    if chosen in {"compare_groups", "stratified_analysis", "adjust_confounders", "cross_project_validate", "cross_disease_validate"}:
        return "evidence_requirement", ["REQUIRED_EVIDENCE_STEP"]
    if chosen in {"inspect_cohort", "execute_read_query"}:
        return "efficiency", ["NEXT_BOUNDED_READ_STEP"]
    return "exploration_depth", ["NEXT_REQUIRED_ANALYSIS_STEP"]


def build(paths: list[Path] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    task_map = _tasks()
    records: list[dict[str, Any]] = []
    scanned = 0
    model_decisions = 0
    incomplete_model_decisions = 0
    complete_model_decision_rows = 0
    unique_model_states: set[str] = set()
    for path in paths or list(TRACE_FILES):
        payload = json.loads(path.read_text(encoding="utf-8"))
        planner_metadata = _planner_source_metadata(payload, path)
        score_to_case = {score.get("traceId"): score.get("caseId") for score in payload.get("scores", []) if score.get("status") == "PASS"}
        for trace in payload.get("traces", []):
            case_id = score_to_case.get(trace.get("traceId"))
            task = task_map.get(case_id or "")
            if not task:
                continue
            for decision in trace.get("decisions", []):
                scanned += 1
                if decision.get("planner_origin") != "model":
                    continue
                model_decisions += 1
                # These fields must come from the structured decision source.
                # Never reconstruct them from state_summary.
                if (
                    decision.get("task_kind") is None
                    or decision.get("goal_code") is None
                    or decision.get("observation_flags") is None
                    or decision.get("history_actions") is None
                    or decision.get("candidate_actions") is None
                ):
                    incomplete_model_decisions += 1
                    continue
                chosen = decision.get("final_action") or decision.get("selected_action")
                if not chosen or decision.get("raw_action") != chosen:
                    continue
                state = {
                    "observationStateCode": decision.get("observationStateCode"),
                    "observationEvidenceBindingCount": decision.get("observationEvidenceBindingCount"),
                    "observationSourceRoutes": decision.get("observationSourceRoutes", []),
                    "stateSummary": decision.get("state_summary", ""),
                    "taskKind": decision.get("task_kind"),
                    "goalCode": decision.get("goal_code"),
                    "observationFlags": list(decision.get("observation_flags", [])),
                    "historyActions": list(decision.get("history_actions", [])),
                    "candidateActions": list(decision.get("candidate_actions", [])),
                }
                complete_model_decision_rows += 1
                unique_model_states.add(_hash(state))
                allowed = list(dict.fromkeys(decision.get("candidate_actions", [])))
                if not allowed or allowed != list(dict.fromkeys(decision.get("allowedActions", []))):
                    continue
                history = list(decision.get("history_actions", []))
                expected = _next_required(task, history)
                if expected != chosen:
                    continue
                alternatives = [item for item in decision.get("alternative_actions", []) if item in allowed and item != chosen]
                dimension, codes = _dimension(chosen)
                signature = _hash(state)
                for rejected in alternatives:
                    records.append({
                        "source_kind": "real_training_trace",
                        "source_trace_id": trace.get("traceId"),
                        "source_file": path.name,
                        **planner_metadata,
                        "case_id": case_id,
                        "state_signature": signature,
                        "task_family": task.get("family") or task.get("kind") or decision.get("task_kind"),
                        "task_kind": decision.get("task_kind"),
                        "goal_code": decision.get("goal_code"),
                        "observation_flags": decision.get("observation_flags"),
                        "history_actions": decision.get("history_actions"),
                        "candidate_actions": allowed,
                        "state_summary": decision.get("state_summary", ""),
                        "planner_origin": "model",
                        "raw_action": decision.get("raw_action"),
                        "final_action": chosen,
                        "chosen_action": chosen,
                        "rejected_action": rejected,
                        "chosen_origin": "model_contract_validated",
                        "rejected_origin": "legal_alternative_not_required_by_task_contract",
                        "preference_type": "action",
                        "preference_dimension": dimension,
                        "preference_codes": codes,
                        "preference_rationale": f"The task contract requires {chosen} as the next bounded step; {rejected} is legal but not the next required step in this state.",
                        "preference_rationale_source": "oracle_required_action",
                        "review_status": "CONTRACT_DERIVED_REVIEW_REQUIRED",
                    })
    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in records:
        unique.setdefault((record["state_signature"], record["chosen_action"], record["rejected_action"]), record)
    rows = list(unique.values())
    audit = {
        "schemaVersion": "p2j4-dpo-v4-contract-preference-candidates-audit-v1",
        "status": "REVIEW_REQUIRED" if rows else "BLOCKED_INPUT_CONTRACT_INCOMPLETE",
        "trainingStarted": False,
        "pairConstructionStarted": False,
        "scannedDecisionCount": scanned,
        "modelDecisionCount": model_decisions,
        "incompleteModelDecisionCount": incomplete_model_decisions,
        "completeModelDecisionRowCount": complete_model_decision_rows,
        "uniqueModelStateCount": len(unique_model_states),
        "rawCandidateCount": len(records),
        "uniqueCandidateCount": len(rows),
        "chosenActionCounts": dict(Counter(row["chosen_action"] for row in rows)),
        "rejectedActionCounts": dict(Counter(row["rejected_action"] for row in rows)),
        "preferenceDimensionCounts": dict(Counter(row["preference_dimension"] for row in rows)),
        "preferenceRationaleSourceCounts": dict(Counter(row["preference_rationale_source"] for row in rows)),
        "rule": "model_origin AND raw_action==final_action AND final_action==next_required_task_action",
        "notFinalDpoPairs": True,
        "nextAction": "audit_contract_derived_candidates_for_leakage_and_global_balance",
    }
    return rows, audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    rows, audit = build()
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")
    args.audit_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ("status", "uniqueCandidateCount", "chosenActionCounts")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
