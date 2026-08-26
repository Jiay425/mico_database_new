"""Build de-identified Base/SFT evaluation bundles from the reviewed pool.

The bundle keeps group membership in a manifest only.  Model messages contain
the same structured policy state as Decision SFT and never include provenance,
raw questions, SQL, task-family labels, or review metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "You are Mico's Scientific Agent decision policy. "
    "Choose the next action from the candidate actions in the policy state. "
    "Use only the supplied structured state and action history. "
    "Return one JSON object and no Markdown. The JSON keys must be exactly: "
    "selected_action, decision_reason, alternative_actions, stop_reason. "
    "Use stop_reason only when selected_action is finish."
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"EXPECTED_OBJECT:{path}")
    return value


def _validate(candidate: dict[str, Any]) -> None:
    required = {
        "task_kind", "goal_code", "observation_flags", "history_actions",
        "candidate_actions", "state_summary", "decision_reason",
        "selected_action", "alternative_actions", "stop_reason",
    }
    missing = sorted(required - set(candidate))
    if missing:
        raise ValueError("MISSING_CANDIDATE_FIELDS:" + ",".join(missing))
    actions = candidate["candidate_actions"]
    selected = candidate["selected_action"]
    alternatives = candidate["alternative_actions"]
    if selected not in actions or selected in alternatives:
        raise ValueError("INVALID_SELECTED_ACTION")
    if any(action not in actions for action in alternatives):
        raise ValueError("INVALID_ALTERNATIVE_ACTION")
    if selected == "finish" and not candidate["stop_reason"]:
        raise ValueError("FINISH_STOP_REASON_MISSING")
    if selected != "finish" and candidate["stop_reason"] is not None:
        raise ValueError("NON_TERMINAL_STOP_REASON_PRESENT")
    if not all(isinstance(candidate[field], str) and candidate[field].strip()
               for field in ("state_summary", "decision_reason")):
        raise ValueError("EMPTY_STATE_OR_REASON")


def _record(record_id: str, candidate: dict[str, Any]) -> dict[str, Any]:
    _validate(candidate)
    state = {
        "task_kind": candidate["task_kind"],
        "goal_code": candidate["goal_code"],
        "observation_flags": candidate["observation_flags"],
        "history_actions": candidate["history_actions"],
        "candidate_actions": candidate["candidate_actions"],
        "state_summary": candidate["state_summary"],
    }
    target = {
        "selected_action": candidate["selected_action"],
        "decision_reason": candidate["decision_reason"],
        "alternative_actions": candidate["alternative_actions"],
        "stop_reason": candidate["stop_reason"],
    }
    return {
        "id": record_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({"policy_state": state}, ensure_ascii=False, sort_keys=True)},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False, separators=(",", ":"))},
        ],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_item_map(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for item in items:
        signature = item.get("signatureHash")
        if not isinstance(signature, str) or not signature:
            raise ValueError("REVIEW_SIGNATURE_MISSING")
        if signature in result and result[signature] != item:
            raise ValueError("DUPLICATE_SIGNATURE_CONFLICT:" + signature)
        result[signature] = item
    return result


def _group_items(
    name: str,
    items: dict[str, dict[str, Any]],
    train_pairs: set[tuple[str, str]],
    test_signatures: set[str],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for signature, item in items.items():
        candidate = item["candidate"]
        sources = set(item.get("sourceRuns") or [])
        pair = (candidate["task_kind"], candidate["goal_code"])
        include = False
        if name == "original50_decisions":
            include = "golden_v2" in sources
        elif name == "original50_open_exploration":
            include = "golden_v2" in sources and candidate["task_kind"] == "open_exploration"
        elif name == "hard30_decisions":
            include = "hard_v2" in sources
        elif name == "hard_variant75_decisions":
            include = "hard_variant_v2" in sources
        elif name == "open_exploration_all":
            include = candidate["task_kind"] == "open_exploration"
        elif name == "ood_train_unseen_goal_pair":
            include = pair not in train_pairs
        elif name == "ood_test70_heldout":
            include = signature in test_signatures
        else:
            raise ValueError("UNKNOWN_GROUP:" + name)
        if include:
            result[signature] = item
    return result


def build(review_path: Path, split_path: Path, output_dir: Path) -> dict[str, Any]:
    review = _read(review_path)
    split = _read(split_path)
    if review.get("trainingStarted") is not False:
        raise ValueError("REVIEW_ALREADY_TRAINING")
    items = _source_item_map(review.get("items") or [])
    train_items = split["splits"]["train"]
    train_pairs = {
        (item["candidate"]["task_kind"], item["candidate"]["goal_code"])
        for item in train_items
    }
    test_signatures = {item["signatureHash"] for item in split["splits"]["test"]}
    group_names = [
        "original50_decisions",
        "original50_open_exploration",
        "hard30_decisions",
        "hard_variant75_decisions",
        "open_exploration_all",
        "ood_train_unseen_goal_pair",
        "ood_test70_heldout",
    ]
    groups = {
        name: _group_items(name, items, train_pairs, test_signatures)
        for name in group_names
    }
    union: dict[str, dict[str, Any]] = {}
    memberships: dict[str, list[str]] = {}
    for name in group_names:
        for signature, item in groups[name].items():
            union[signature] = item
            memberships.setdefault(signature, []).append(name)

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    record_meta: dict[str, Any] = {}
    for index, signature in enumerate(sorted(union), 1):
        record_id = f"bundle-{index:04d}"
        candidate = union[signature]["candidate"]
        records.append(_record(record_id, candidate))
        record_meta[record_id] = {
            "signatureHash": signature,
            "groups": sorted(memberships[signature]),
            "sourceRuns": sorted(union[signature].get("sourceRuns") or []),
            "taskKind": candidate["task_kind"],
            "goalCode": candidate["goal_code"],
        }
    records_path = output_dir / "records.jsonl"
    records_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    group_manifest: dict[str, Any] = {}
    for name in group_names:
        ids = [record_id for record_id, meta in record_meta.items() if name in meta["groups"]]
        group_manifest[name] = {
            "count": len(ids),
            "recordIds": ids,
            "sourceRuns": sorted({run for record_id in ids for run in record_meta[record_id]["sourceRuns"]}),
            "trainUnseenGoalPair": name == "ood_train_unseen_goal_pair",
            "testHeldout": name == "ood_test70_heldout",
        }
    manifest = {
        "schemaVersion": "p2j4-decision-eval-bundle-v1",
        "trainingStarted": True,
        "sourceReview": review_path.name,
        "sourceSplit": split_path.name,
        "recordCount": len(records),
        "recordsPath": records_path.name,
        "recordsSha256": _sha256(records_path),
        "groups": group_manifest,
        "records": record_meta,
        "inputContract": {
            "rawQuestionIncluded": False,
            "rawSqlIncluded": False,
            "sourceTraceIdIncluded": False,
            "taskFamilyIncluded": False,
        },
        "oodBoundary": {
            "ood_train_unseen_goal_pair": "goal pair absent from the 590 training records; some cases were validation/test held-out, so this is train-unseen, not a new biological cohort",
            "ood_test70_heldout": "the frozen 70-case test split; exact signature, family, and source-trace disjoint from train",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Base/SFT evaluation bundles")
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = build(args.review, args.split, args.output_dir)
    except (OSError, ValueError, json.JSONDecodeError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "INVALID", "errorCode": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({
        "status": "READY",
        "recordCount": result["recordCount"],
        "groups": {name: value["count"] for name, value in result["groups"].items()},
        "outputDir": str(args.output_dir),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
