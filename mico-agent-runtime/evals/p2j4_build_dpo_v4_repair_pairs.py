"""Extract explicit model-raw -> Runtime-contract repair preferences.

These rows are hard/development evidence, not pure model-origin pairs.  They
are retained only when the trace records both actions and the repair code in
the structured decision itself.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:20]


def _prompt(row: dict[str, Any]) -> list[dict[str, str]]:
    state = {key: row[key] for key in (
        "task_kind", "goal_code", "observation_flags", "history_actions", "candidate_actions", "state_summary"
    )}
    return [
        {"role": "system", "content": "Choose exactly one next action from the supplied de-identified policy state."},
        {"role": "user", "content": json.dumps({"policy_state": state}, ensure_ascii=False, sort_keys=True)},
    ]


def _rationale(repair_codes: list[str], chosen: str, rejected: str) -> tuple[str, list[str], str]:
    codes = set(repair_codes)
    if any("STRATIFIED_ORDER" in code for code in codes):
        return (
            "evidence_requirement",
            ["COMPARISON_BEFORE_STRATIFICATION", "REQUIRED_EVIDENCE_ORDER"],
            f"The Runtime contract requires {chosen} before stratification; {rejected} would skip the validated comparison boundary.",
        )
    if any("PREMATURE_FINISH" in code for code in codes):
        return (
            "evidence_requirement",
            ["EVIDENCE_RETRIEVAL_BEFORE_STOP", "STOP_BOUNDARY_NOT_READY"],
            f"The state still requires the bounded evidence step {chosen}; stopping with {rejected} would leave the evidence obligation unresolved.",
        )
    if any("REPEATED_PROJECTION" in code for code in codes):
        return (
            "stop_boundary",
            ["REPEATED_ANALYSIS_IS_NOT_NEW_EVIDENCE", "STOP_AFTER_BOUNDED_ANALYSIS"],
            f"The projection has already been analyzed, so {chosen} is the safe stop boundary instead of repeating {rejected}.",
        )
    return (
        "evidence_requirement",
        ["RUNTIME_POLICY_CONTRACT_REPAIR"],
        f"The recorded Runtime contract repairs the raw model choice {rejected} into {chosen} for this state.",
    )


def _trace_paths() -> list[Path]:
    paths = set(Path(p) for p in glob.glob(str(ROOT / "evals" / "p2j4-dpo-v4-model-origin-state-matrix-runs-v13*.json")))
    paths.update(Path(p) for p in glob.glob(str(ROOT / "evals" / "p2j4-dpo-v4-weak-action-source-runs-v14-*.json")))
    return sorted(paths)


def build(max_per_state: int = 3) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    for path in _trace_paths():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            excluded["SOURCE_UNREADABLE"] += 1
            continue
        for trace in payload.get("traces", []):
            case_id = trace.get("caseId")
            for decision in trace.get("decisions", []):
                repair_codes = list(decision.get("repair_codes") or [])
                raw = decision.get("raw_action")
                final = decision.get("final_action")
                if decision.get("planner_origin") != "model" or not raw or not final or raw == final:
                    continue
                if not repair_codes:
                    excluded["REPAIR_CODE_MISSING"] += 1
                    continue
                fields = (
                    "task_kind", "goal_code", "observation_flags", "history_actions",
                    "candidate_actions", "state_summary",
                )
                if any(decision.get(field) is None for field in fields):
                    excluded["STRUCTURED_FIELDS_INCOMPLETE"] += 1
                    continue
                candidates = list(decision.get("candidate_actions") or [])
                history = set(decision.get("history_actions") or [])
                if not (4 <= len(candidates) <= 7) or raw not in candidates or final not in candidates:
                    excluded["PAIR_ACTION_NOT_CANDIDATE"] += 1
                    continue
                # The raw model action may deliberately be a repeated action:
                # that is the bad decision the Runtime contract repaired.  Do
                # not erase this evidence; only a repaired final action that
                # was already completed is unusable as the chosen side.
                if final in history:
                    excluded["REPAIRED_ACTION_ALREADY_OBSERVED"] += 1
                    continue
                state = {key: decision[key] for key in (
                    "task_kind", "goal_code", "observation_flags", "history_actions", "candidate_actions", "state_summary"
                )}
                signature = _hash(state)
                dimension, preference_codes, rationale = _rationale(repair_codes, final, raw)
                rows.append({
                    "id": "dpo-v4-repair-" + _hash([signature, final, raw]),
                    "prompt": _prompt({**state}),
                    "chosen": json.dumps({"selected_action": final, "decision_reason": rationale, "alternative_actions": [raw], "stop_reason": "EVIDENCE_SUFFICIENT" if final == "finish" else None}, ensure_ascii=False, sort_keys=True),
                    "rejected": json.dumps({"selected_action": raw, "decision_reason": "Recorded raw model action before the Runtime contract repair.", "alternative_actions": [final], "stop_reason": None}, ensure_ascii=False, sort_keys=True),
                    **state,
                    "task_family": f"{state['task_kind']}:{state['goal_code']}:runtime_repair_derived",
                    "state_signature": signature,
                    "source_trace_id": trace.get("traceId"),
                    "source_file": path.name,
                    "case_id": case_id,
                    "source_kind": "hard_development_trace",
                    "scenario_class": "runtime_repair_derived",
                    "planner_origin": "model",
                    "provenance_class": "runtime_repair_derived",
                    "raw_action": raw,
                    "final_action": final,
                    "chosen_action": final,
                    "rejected_action": raw,
                    "repair_codes": repair_codes,
                    "preference_type": "action",
                    "preference_dimension": dimension,
                    "preference_codes": preference_codes,
                    "preference_rationale": rationale,
                    "preference_rationale_source": "runtime_policy_contract_repair_v1",
                    "review_status": "LOCAL_REPAIR_REVIEWED_PENDING_FINAL_FREEZE",
                })
    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        unique.setdefault((row["state_signature"], row["chosen_action"], row["rejected_action"]), row)
    rows = list(unique.values())
    # Keep the repair pool diverse even when one state exposes several legal
    # raw alternatives.  This is a separate pool, so it can be audited before
    # being merged with model-origin rows.
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for row in sorted(rows, key=lambda item: (item["state_signature"], item["chosen_action"], item["rejected_action"])):
        if counts[row["state_signature"]] >= max_per_state:
            continue
        counts[row["state_signature"]] += 1
        selected.append(row)
    audit = {
        "schemaVersion": "p2j4-dpo-v4-repair-pairs-audit-v1",
        "status": "LOCAL_REPAIR_REVIEWED_PENDING_FINAL_FREEZE" if selected else "BLOCKED",
        "trainingStarted": False,
        "pairConstructionStarted": True,
        "traceFileCount": len(_trace_paths()),
        "pairCount": len(selected),
        "uniqueStateCount": len(counts),
        "maxPairsPerState": max(counts.values()) if counts else 0,
        "chosenActionCounts": dict(Counter(row["chosen_action"] for row in selected)),
        "rejectedActionCounts": dict(Counter(row["rejected_action"] for row in selected)),
        "repairCodeCounts": dict(Counter(code for row in selected for code in row["repair_codes"])),
        "excludedCounts": dict(excluded),
        "notPureModelOrigin": True,
        "nextAction": "merge_with_hard_case_and_model_origin_pools",
    }
    return selected, audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    rows, audit = build()
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    args.audit_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: audit[key] for key in ("status", "pairCount", "uniqueStateCount", "maxPairsPerState", "chosenActionCounts", "excludedCounts")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
