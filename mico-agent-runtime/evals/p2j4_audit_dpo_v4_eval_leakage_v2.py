"""Check exact normalized DPO states against frozen Test70/OOD30 states."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EVAL_FILES = {
    "test70": ROOT / "evals" / "p2j4-v4-eval-bundle" / "test70-v4-reason-audit.json",
    "ood30": ROOT / "evals" / "p2j4-v4-eval-bundle" / "ood_v2-v4-reason-audit.json",
}
STATE_FIELDS = (
    "task_kind", "goal_code", "observation_flags", "history_actions",
    "candidate_actions", "state_summary",
)


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _state(value: dict[str, Any]) -> dict[str, Any]:
    return {field: value.get(field) for field in STATE_FIELDS}


def _signature(value: dict[str, Any]) -> str:
    raw = json.dumps(_state(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def audit(path: Path) -> dict[str, Any]:
    rows = _read(path)
    eval_states: dict[str, list[dict[str, str]]] = {}
    for name, eval_path in EVAL_FILES.items():
        payload = json.loads(eval_path.read_text(encoding="utf-8"))
        for case in payload.get("cases", []):
            state = case.get("state", {})
            eval_states.setdefault(_signature(state), []).append({
                "source": name,
                "record_id": case.get("record_id"),
            })
    overlaps: list[dict[str, Any]] = []
    dpo_states: set[str] = set()
    for row in rows:
        try:
            state = json.loads(row["prompt"][1]["content"])["policy_state"]
        except (IndexError, KeyError, TypeError, json.JSONDecodeError):
            overlaps.append({"id": row.get("id"), "error": "PROMPT_STATE_INVALID"})
            continue
        signature = _signature(state)
        dpo_states.add(signature)
        if signature in eval_states:
            overlaps.append({
                "id": row.get("id"),
                "state_signature": row.get("state_signature"),
                "evaluation_matches": eval_states[signature],
            })
    result = {
        "schemaVersion": "p2j4-dpo-v4-eval-leakage-audit-v2",
        "status": "PASS" if not overlaps else "FAIL",
        "dpoRecordCount": len(rows),
        "dpoUniqueNormalizedStateCount": len(dpo_states),
        "evaluationStateCount": len(eval_states),
        "evaluationFiles": {name: str(path) for name, path in EVAL_FILES.items()},
        "normalizedOverlapCount": len(overlaps),
        "test70Ood30Leakage": len(overlaps),
        "overlaps": overlaps[:50],
        "errors": [] if not overlaps else ["NORMALIZED_STATE_OVERLAP_OR_INVALID_PROMPT"],
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.input)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
