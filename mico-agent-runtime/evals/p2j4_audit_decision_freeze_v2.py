"""Independent structural, balance, leakage and preference audit for Freeze v2."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from evals.p2j4_build_decision_freeze_v2 import (
    ALL_ACTIONS, DPO_OUTPUT, FORBIDDEN, OOD30, SFT_OUTPUT, STOP, TEST70,
    _jsonl, _policy_state, _signature, _target,
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _audit_sft(records: list[dict[str, Any]], external: set[str]) -> dict[str, Any]:
    errors: list[str] = []
    signatures: set[str] = set()
    families: set[str] = set()
    source_traces: set[str] = set()
    action_counts: Counter[str] = Counter()
    provenance: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    for record in records:
        try:
            state, target = _policy_state(record), _target(record)
            selected = target["selected_action"]
            signature = record["metadata"]["state_signature"]
            if signature != _signature(state):
                errors.append("STATE_SIGNATURE_MISMATCH")
            if signature in signatures:
                errors.append("DUPLICATE_STATE_SIGNATURE")
            signatures.add(signature)
            if signature in external:
                errors.append("EXTERNAL_EXACT_STATE_LEAK")
            if selected not in state["candidate_actions"]:
                errors.append("SELECTED_ACTION_NOT_CANDIDATE")
            if selected == "finish":
                if target.get("stop_reason") not in STOP:
                    errors.append("FINISH_STOP_REASON_INVALID")
            elif target.get("stop_reason") is not None:
                errors.append("NON_TERMINAL_STOP_REASON_PRESENT")
            alternatives = target.get("alternative_actions")
            if alternatives != [action for action in state["candidate_actions"] if action != selected]:
                errors.append("ALTERNATIVE_ACTIONS_NOT_COMPLEMENT")
            if len(target.get("decision_reason", "").split()) < 7:
                errors.append("DECISION_REASON_TOO_GENERIC")
            metadata = record["metadata"]
            family = metadata["task_family"]
            trace = metadata["source_trace_id"]
            if family in families:
                # Families may have many samples within a partition; this set
                # is kept only for the cross-split check below.
                pass
            families.add(family)
            source_traces.add(trace)
            family_counts[family] += 1
            provenance[metadata["provenance"]] += 1
            action_counts[selected] += 1
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            errors.append("SFT_RECORD_SHAPE_INVALID")
    text = json.dumps(records, ensure_ascii=False).lower()
    return {
        "count": len(records),
        "uniqueStateCount": len(signatures),
        "families": families,
        "sourceTraces": source_traces,
        "actionCounts": dict(action_counts),
        "provenanceCounts": dict(provenance),
        "maxFamilyCount": max(family_counts.values(), default=0),
        "forbiddenTextAbsent": not any(item.lower() in text for item in FORBIDDEN),
        "errors": sorted(set(errors)),
    }


def _audit_dpo(records: list[dict[str, Any]], sft_signatures: set[str]) -> dict[str, Any]:
    errors: list[str] = []
    signatures: set[str] = set()
    families: Counter[str] = Counter()
    for record in records:
        try:
            prompt = record["prompt"]
            state = json.loads(prompt[1]["content"])["policy_state"]
            chosen = json.loads(record["chosen"])
            rejected = json.loads(record["rejected"])
            signature = record["metadata"]["state_signature"]
            if signature != _signature(state) or signature not in sft_signatures:
                errors.append("DPO_SFT_STATE_LINK_INVALID")
            if signature in signatures:
                errors.append("DPO_DUPLICATE_STATE")
            signatures.add(signature)
            for target in (chosen, rejected):
                if target["selected_action"] not in state["candidate_actions"]:
                    errors.append("DPO_ACTION_NOT_CANDIDATE")
            if chosen["selected_action"] == rejected["selected_action"]:
                errors.append("DPO_CHOSEN_REJECTED_SAME_ACTION")
            if record["metadata"]["provenance"] != "controlled_state_difference_v2":
                errors.append("DPO_PROVENANCE_NOT_CONTROLLED")
            families[record["metadata"]["task_family"].split(":")[1]] += 1
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, IndexError):
            errors.append("DPO_RECORD_SHAPE_INVALID")
    return {
        "count": len(records),
        "uniqueStateCount": len(signatures),
        "familyCounts": dict(families),
        "errors": sorted(set(errors)),
    }


def audit() -> dict[str, Any]:
    sft = {name: _jsonl(SFT_OUTPUT / f"{name}.jsonl") for name in ("train", "validation")}
    dpo = {name: _jsonl(DPO_OUTPUT / f"{name}.jsonl") for name in ("train", "validation")}
    external = {_signature(_policy_state(record)) for record in [*_jsonl(TEST70), *_jsonl(OOD30)]}
    sft_audit = {name: _audit_sft(records, external) for name, records in sft.items()}
    all_sft = [*sft["train"], *sft["validation"]]
    all_signatures = {record["metadata"]["state_signature"] for record in all_sft}
    dpo_audit = {name: _audit_dpo(records, all_signatures) for name, records in dpo.items()}
    all_actions = Counter(
        _target(record)["selected_action"] for record in all_sft
    )
    train_families, val_families = sft_audit["train"]["families"], sft_audit["validation"]["families"]
    train_traces, val_traces = sft_audit["train"]["sourceTraces"], sft_audit["validation"]["sourceTraces"]
    issues = [
        *sft_audit["train"]["errors"], *sft_audit["validation"]["errors"],
        *dpo_audit["train"]["errors"], *dpo_audit["validation"]["errors"],
    ]
    if train_families & val_families:
        issues.append("TASK_FAMILY_SPLIT_LEAK")
    if train_traces & val_traces:
        issues.append("SOURCE_TRACE_SPLIT_LEAK")
    if not 600 <= len(all_sft) <= 900:
        issues.append("SFT_FREEZE_COUNT_INVALID")
    if sum(len(items) for items in dpo.values()) != 300:
        issues.append("DPO_FREEZE_COUNT_INVALID")
    if set(all_actions) != ALL_ACTIONS or any(all_actions[action] < 30 for action in ALL_ACTIONS):
        issues.append("ACTION_COVERAGE_INVALID")
    if any(audit["maxFamilyCount"] > 225 for audit in sft_audit.values()):
        issues.append("TASK_FAMILY_DOMINANCE")
    return {
        "schemaVersion": "p2j4-decision-freeze-v2-audit",
        "status": "PASS" if not issues else "FAIL",
        "trainingStarted": False,
        "sft": {name: {key: value for key, value in item.items() if key not in {"families", "sourceTraces"}} for name, item in sft_audit.items()},
        "dpo": dpo_audit,
        "overallActionCounts": dict(all_actions),
        "taskFamilySplitDisjoint": not bool(train_families & val_families),
        "sourceTraceSplitDisjoint": not bool(train_traces & val_traces),
        "externalExactStateOverlap": len(all_signatures & external),
        "hashes": {
            "sftTrain": _hash(SFT_OUTPUT / "train.jsonl"),
            "sftValidation": _hash(SFT_OUTPUT / "validation.jsonl"),
            "dpoTrain": _hash(DPO_OUTPUT / "train.jsonl"),
            "dpoValidation": _hash(DPO_OUTPUT / "validation.jsonl"),
        },
        "issues": sorted(set(issues)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "issues": result["issues"]}, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
