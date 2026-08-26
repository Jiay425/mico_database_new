"""Stage de-identified Qwen Decision-DPO pairs without a model call.

Chosen completions are frozen gold Decision targets. Rejected completions come
from recorded Qwen3-8B Base outputs only when the Base action was allow-listed
but wrong. Held-out Test70/OOD records are excluded. A small set of audited
Runtime Base-vs-SFT decision pairs supplements the offline policy failures.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_EVAL = ROOT / "sft-runs" / "qwen3-8b-decision-sft-v1" / "bundle-eval-v1-audited.json"
BUNDLE_RECORDS = ROOT / "sft-data" / "eval-bundle-v1" / "records.jsonl"
BUNDLE_MANIFEST = ROOT / "sft-data" / "eval-bundle-v1" / "manifest.json"
RUNTIME_PAIRS = ROOT / "evals" / "p2j4-policy-sensitive-runtime-v1" / "dpo-preference-candidates-v1.json"
OUTPUT_DIR = ROOT / "sft-data" / "decision-dpo-v1-staging"

STOP_REASONS = {
    "EVIDENCE_SUFFICIENT", "NO_NEW_INFORMATION", "QUALITY_RISK",
    "ACTION_BUDGET_EXHAUSTED", "UPSTREAM_REJECTED", "UNSUPPORTED_ACTION",
    "USER_REQUESTED_STOP",
}
REQUIRED_TARGET_KEYS = {
    "selected_action", "decision_reason", "alternative_actions", "stop_reason",
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path.name}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _completion(value: dict[str, Any]) -> str:
    if set(value) != REQUIRED_TARGET_KEYS:
        raise ValueError("DPO_TARGET_KEYS_INVALID")
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _policy_state(record: dict[str, Any]) -> dict[str, Any]:
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError("DPO_SOURCE_MESSAGES_INVALID")
    value = json.loads(messages[1]["content"])
    state = value.get("policy_state")
    if not isinstance(state, dict):
        raise ValueError("DPO_POLICY_STATE_INVALID")
    return state


def _canonical_rejected(prediction: dict[str, Any], state: dict[str, Any]) -> dict[str, Any] | None:
    action = prediction.get("selected_action")
    candidates = state.get("candidate_actions")
    reason = prediction.get("decision_reason")
    if not isinstance(action, str) or not isinstance(candidates, list) or action not in candidates:
        return None
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 512:
        return None
    raw_stop = prediction.get("stop_reason")
    if action == "finish":
        if raw_stop not in STOP_REASONS:
            return None
        stop_reason: str | None = raw_stop
    else:
        stop_reason = None
    return {
        "selected_action": action,
        "decision_reason": reason.strip(),
        "alternative_actions": [item for item in candidates if item != action],
        "stop_reason": stop_reason,
    }


def _signature(state: dict[str, Any]) -> str:
    safe = {
        key: state.get(key)
        for key in (
            "task_kind", "goal_code", "observation_flags", "history_actions",
            "candidate_actions", "state_summary",
        )
    }
    return hashlib.sha256(json.dumps(safe, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]


def _split(signature: str) -> str:
    # State-level split prevents near-identical policy states leaking from DPO
    # train to validation. Test70/OOD remain wholly external and untouched.
    return "validation" if int(signature[:8], 16) % 10 == 0 else "train"


def _record(
    record_id: str,
    messages: list[dict[str, str]],
    chosen: dict[str, Any],
    rejected: dict[str, Any],
    *,
    source: str,
    source_id: str,
) -> dict[str, Any]:
    state = json.loads(messages[1]["content"])["policy_state"]
    signature = _signature(state)
    return {
        "id": record_id,
        "prompt": messages[:2],
        "chosen": _completion(chosen),
        "rejected": _completion(rejected),
        "metadata": {
            "source": source,
            "sourceId": source_id,
            "stateSignature": signature,
            "taskKind": state.get("task_kind"),
            "goalCode": state.get("goal_code"),
        },
    }


def build() -> dict[str, Any]:
    bundle = _read_json(BUNDLE_EVAL)
    records = {item["id"]: item for item in _read_jsonl(BUNDLE_RECORDS)}
    manifest = _read_json(BUNDLE_MANIFEST)["records"]
    base = {item["record_id"]: item for item in bundle["base"]["cases"]}
    sft = {item["record_id"]: item for item in bundle["sft"]["cases"]}
    staged: list[dict[str, Any]] = []
    used_signatures: set[str] = set()

    for source_id, record in records.items():
        meta = manifest[source_id]
        groups = meta.get("groups", [])
        if any("heldout" in group or "ood" in group for group in groups):
            continue
        base_case, sft_case = base[source_id], sft[source_id]
        if not (
            base_case.get("json_valid")
            and base_case.get("action_allowed")
            and not base_case.get("action_correct")
            and sft_case.get("policy_pass")
            and sft_case.get("reason_audit", {}).get("status") == "PASS"
        ):
            continue
        state = _policy_state(record)
        rejected = _canonical_rejected(base_case.get("prediction") or {}, state)
        chosen = json.loads(record["messages"][2]["content"])
        if rejected is None or not isinstance(chosen, dict) or set(chosen) != REQUIRED_TARGET_KEYS:
            continue
        if rejected == chosen:
            continue
        signature = _signature(state)
        if signature in used_signatures:
            continue
        used_signatures.add(signature)
        staged.append(_record(
            f"dpo-v1-bundle-{len(staged)+1:04d}", record["messages"], chosen, rejected,
            source="qwen_base_vs_sft_bundle_v1", source_id=source_id,
        ))

    runtime = _read_json(RUNTIME_PAIRS)
    for item in runtime.get("records", []):
        state = item["state"]
        messages = [
            {"role": "system", "content": (
                "You are Mico's Scientific Agent decision policy. Choose the next action from the "
                "candidate actions in the policy state. Use only the supplied structured state and "
                "action history. Return one JSON object and no Markdown. The JSON keys must be exactly: "
                "selected_action, decision_reason, alternative_actions, stop_reason. Use stop_reason "
                "only when selected_action is finish."
            )},
            {"role": "user", "content": json.dumps({"policy_state": state}, ensure_ascii=False)},
        ]
        signature = _signature(state)
        if signature in used_signatures:
            continue
        used_signatures.add(signature)
        staged.append(_record(
            f"dpo-v1-runtime-{len(staged)+1:04d}", messages,
            item["chosen"], item["rejected"],
            source="qwen_runtime_decision_pair_v1", source_id=item["recordId"],
        ))

    if len(staged) < 100:
        raise ValueError("DPO_PAIR_COUNT_BELOW_100")
    partitions = {"train": [], "validation": []}
    for item in staged:
        partitions[_split(item["metadata"]["stateSignature"])].append(item)
    if not partitions["validation"]:
        raise ValueError("DPO_VALIDATION_EMPTY")
    if len(partitions["train"]) < 80:
        raise ValueError("DPO_TRAIN_COUNT_BELOW_80")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, items in partitions.items():
        (OUTPUT_DIR / f"{name}.jsonl").write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items), encoding="utf-8"
        )
    manifest_out = {
        "schemaVersion": "p2j4-decision-dpo-v1-staging",
        "trainingStarted": False,
        "pairCount": len(staged),
        "splitCounts": {name: len(items) for name, items in partitions.items()},
        "sourceCounts": dict(Counter(item["metadata"]["source"] for item in staged)),
        "taskKindCounts": dict(Counter(item["metadata"]["taskKind"] for item in staged)),
        "heldoutPolicy": "Test70 and all OOD groups excluded from DPO staging",
        "pairContract": "gold chosen completion versus recorded Qwen Base wrong allow-listed completion",
        "runtimeExecutionBoundary": "runtime pairs are Decision-level legacy audits; no unbound execution assertion is used",
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest_out


def main() -> int:
    try:
        result = build()
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "FAILED", "errorCode": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({"status": "PASS", **result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
