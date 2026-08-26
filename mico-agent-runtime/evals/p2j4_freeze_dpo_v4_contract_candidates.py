"""Construct and audit a bounded DPO v4 staging freeze from audited sources."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = ROOT / "evals" / "p2j4-dpo-v4-contract-preference-candidates-20260826.jsonl"
LEGACY_FILE = ROOT / "evals" / "p2j4-dpo-v4-real-source-audit-20260825.json"
EVAL_AUDITS = (ROOT / "evals" / "p2j4-v4-eval-bundle" / "test70-v4-reason-audit.json", ROOT / "evals" / "p2j4-v4-eval-bundle" / "ood_v2-v4-reason-audit.json")
ACTIONS = {"inspect_cohort", "execute_read_query", "compare_groups", "analyze_projection", "stratified_analysis", "adjust_confounders", "cross_project_validate", "cross_disease_validate", "retrieve_evidence", "finish"}
MODEL_QUOTA = {"retrieve_evidence": 21, "adjust_confounders": 27, "cross_project_validate": 48, "execute_read_query": 57, "stratified_analysis": 60, "analyze_projection": 60, "finish": 60, "compare_groups": 60, "cross_disease_validate": 57}
LEGACY_QUOTA = {"inspect_cohort": 60, "retrieve_evidence": 10, "cross_project_validate": 15, "adjust_confounders": 20, "analyze_projection": 10, "compare_groups": 10, "cross_disease_validate": 10, "stratified_analysis": 10, "finish": 5}


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:20]


def _dim(action: str) -> tuple[str, list[str]]:
    if action == "finish": return "stop_boundary", ["EVIDENCE_OBLIGATION_SATISFIED"]
    if action in {"compare_groups", "stratified_analysis", "adjust_confounders", "cross_project_validate", "cross_disease_validate"}: return "evidence_requirement", ["REQUIRED_EVIDENCE_STEP"]
    if action in {"inspect_cohort", "execute_read_query"}: return "efficiency", ["NEXT_BOUNDED_READ_STEP"]
    return "exploration_depth", ["NEXT_REQUIRED_ANALYSIS_STEP"]


def _prompt(row: dict[str, Any]) -> list[dict[str, str]]:
    state = {"task_kind": row["task_kind"], "observation_flags": row.get("observation_flags", []), "history_actions": row.get("history_actions", []), "candidate_actions": row["candidate_actions"], "state_summary": row["state_summary"]}
    return [{"role": "system", "content": "Choose one next action from the supplied de-identified policy state."}, {"role": "user", "content": json.dumps({"policy_state": state}, ensure_ascii=False, sort_keys=True)}]


def _normalized_prompt_state(row: dict[str, Any]) -> str:
    state = json.loads(_prompt(row)[1]["content"])["policy_state"]
    return _hash({k: state[k] for k in ("candidate_actions", "history_actions", "observation_flags", "state_summary", "task_kind") if k in state})


def _eval_state_signatures() -> set[str]:
    result: set[str] = set()
    for path in EVAL_AUDITS:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for case in payload.get("cases", []):
            state = case.get("state", {})
            result.add(_hash({k: state[k] for k in ("candidate_actions", "history_actions", "observation_flags", "state_summary", "task_kind") if k in state}))
    return result


def _pair(row: dict[str, Any], source_kind: str, rationale_source: str) -> dict[str, Any]:
    dim, codes = _dim(row["chosen_action"])
    rationale = row.get("preference_rationale") or f"The recorded contract selects {row['chosen_action']} for this state; {row['rejected_action']} is a legal alternative outside the required next step."
    return {
        "id": "dpo-v4-" + _hash([row["source_kind"], row["state_signature"], row["chosen_action"], row["rejected_action"]]),
        "prompt": _prompt(row),
        "chosen": json.dumps({"selected_action": row["chosen_action"], "decision_reason": rationale, "alternative_actions": [row["rejected_action"]], "stop_reason": "EVIDENCE_SUFFICIENT" if row["chosen_action"] == "finish" else None}, ensure_ascii=False),
        "rejected": json.dumps({"selected_action": row["rejected_action"], "decision_reason": "Legal alternative retained as the rejected comparison action.", "alternative_actions": [row["chosen_action"]], "stop_reason": None}, ensure_ascii=False),
        "metadata": {
            "source_kind": source_kind, "source_trace_id": row.get("source_trace_id"), "source_file": row.get("source_file"), "state_signature": row["state_signature"], "task_family": row.get("task_family"), "planner_origin": row.get("planner_origin"), "provenance_class": "model_origin" if row.get("planner_origin") == "model" else "provenance_unknown", "preference_type": "action", "preference_dimension": dim, "preference_codes": codes, "preference_rationale": rationale, "preference_rationale_source": rationale_source, "raw_action": row.get("raw_action"), "final_action": row.get("final_action"), "review_status": "LOCAL_CONTRACT_AUDITED",
        },
    }


def _model_rows() -> list[dict[str, Any]]:
    return [json.loads(line) for line in MODEL_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]


def _legacy_rows() -> list[dict[str, Any]]:
    payload = json.loads(LEGACY_FILE.read_text(encoding="utf-8")); rows = []
    for item in payload.get("candidates", []):
        actions = item.get("candidateActions", []); chosen = item.get("selectedAction")
        if not chosen or not actions or not (4 <= len(actions) <= 7): continue
        for rejected in item.get("legalUnperformedAlternatives", []):
            if rejected in actions and rejected != chosen:
                state = {"taskKind": item.get("taskKind"), "goalCode": item.get("goalCode"), "observationFlags": item.get("observationFlags", []), "historyActions": item.get("historyActions", []), "candidateActions": actions, "stateSummary": item.get("stateSummary", "")}
                rows.append({"source_kind": "legacy_real_state", "source_trace_id": item.get("sourceTraceId"), "source_file": LEGACY_FILE.name, "state_signature": _hash(state), "task_family": item.get("taskFamily"), "task_kind": item.get("taskKind"), "observation_flags": item.get("observationFlags", []), "history_actions": item.get("historyActions", []), "candidate_actions": actions, "state_summary": item.get("stateSummary", ""), "planner_origin": "provenance_unknown", "chosen_action": chosen, "rejected_action": rejected})
    return rows


def _select(rows: list[dict[str, Any]], quota: dict[str, int], source_kind: str, rationale_source: str, eval_signatures: set[str]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []; used: set[tuple[str, str, str]] = set(); state_counts: Counter[str] = Counter()
    for action, count in quota.items():
        pool = sorted((r for r in rows if r["chosen_action"] == action and 4 <= len(r["candidate_actions"]) <= 7), key=lambda r: (r["state_signature"], r["rejected_action"]))
        action_selected = 0
        for row in pool:
            key = (row["state_signature"], row["chosen_action"], row["rejected_action"])
            if key in used or state_counts[row["state_signature"]] >= 4 or _normalized_prompt_state(row) in eval_signatures: continue
            used.add(key); selected.append(_pair(row, source_kind, rationale_source))
            state_counts[row["state_signature"]] += 1
            action_selected += 1
            if action_selected >= count: break
        if action_selected < count: raise ValueError(f"DPO_V4_SOURCE_QUOTA_UNAVAILABLE_MAX4_PER_STATE:{source_kind}:{action}:{action_selected}<{count}")
    return selected


def _split_group(item: dict[str, Any]) -> str:
    meta = item["metadata"]; return str(meta.get("source_trace_id") or meta.get("task_family") or meta["state_signature"])


def build() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eval_signatures = _eval_state_signatures()
    model = _select(_model_rows(), MODEL_QUOTA, "real_training_trace", "oracle_required_action", eval_signatures)
    legacy = _select(_legacy_rows(), LEGACY_QUOTA, "hard_development_trace", "legacy_legal_alternative_annotation", eval_signatures)
    rows = model + legacy
    seen = set(); errors = []
    state_counts = Counter(row["metadata"]["state_signature"] for row in rows)
    for row in rows:
        sig = row["metadata"]["state_signature"]
        seen.add(sig)
        if row["metadata"]["preference_type"] != "action": errors.append("PREFERENCE_TYPE_INVALID")
    groups = sorted({_split_group(row) for row in rows}); val_groups = {g for i, g in enumerate(groups) if i % 5 == 0}
    for row in rows: row["split"] = "val" if _split_group(row) in val_groups else "train"
    counts = Counter(json.loads(row["chosen"])["selected_action"] for row in rows); sources = Counter(row["metadata"]["source_kind"] for row in rows); widths = Counter(len(json.loads(row["prompt"][1]["content"])["policy_state"]["candidate_actions"]) for row in rows)
    total = len(rows); wide = sum(n for w, n in widths.items() if 4 <= w <= 7); vals = list(counts.values())
    for condition, error in ((total == 600, "PAIR_COUNT_NOT_600"), (set(counts) == ACTIONS, "CHOSEN_ACTION_COVERAGE_INCOMPLETE"), (bool(vals) and max(vals) <= 3 * min(vals), "CHOSEN_ACTION_IMBALANCE_OVER_3X"), (wide / total >= .90 if total else False, "WIDTH4TO7_BELOW_90_PERCENT"), (sources["real_training_trace"] / total >= .70 if total else False, "MODEL_ORIGIN_BELOW_70_PERCENT")):
        if not condition: errors.append(error)
    errors.extend(["MAX_PAIRS_PER_STATE_EXCEEDED"] if state_counts and max(state_counts.values()) > 4 else [])
    audit = {"schemaVersion": "p2j4-dpo-v4-freeze-audit-v1", "status": "PASS" if not errors else "FAIL", "trainingStarted": False, "recordCount": total, "uniqueStateCount": len(state_counts), "maxPairsPerState": max(state_counts.values()) if state_counts else 0, "trainCount": sum(row["split"] == "train" for row in rows), "valCount": sum(row["split"] == "val" for row in rows), "chosenActionCounts": dict(counts), "sourceCounts": dict(sources), "candidateWidthCounts": dict(sorted(widths.items())), "wideRate": wide / total if total else 0, "modelOriginRate": sources["real_training_trace"] / total if total else 0, "preferenceTypeCounts": {"action": total}, "rationaleSourceCounts": dict(Counter(str(row["metadata"].get("preference_rationale_source")) for row in rows)), "taskGroupCounts": len(groups), "taskGroupLeakage": 0, "evaluationLeakage": 0, "errors": sorted(set(errors)), "a100Allowed": not errors}
    return rows, audit


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,required=True); parser.add_argument("--audit-output",type=Path,required=True); args=parser.parse_args(); rows,audit=build(); args.output.write_text("\n".join(json.dumps(r,ensure_ascii=False) for r in rows)+"\n",encoding="utf-8"); args.audit_output.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps({k:audit[k] for k in ("status","recordCount","chosenActionCounts","errors")},ensure_ascii=False)); return 0 if audit["status"]=="PASS" else 1
if __name__ == "__main__": raise SystemExit(main())
