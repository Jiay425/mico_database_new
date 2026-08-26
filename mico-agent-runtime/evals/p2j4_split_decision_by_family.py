"""Split the Decision SFT review set by whole task families.

This intentionally does not perform a row-level random split.  Every item in
one ``task_family`` remains in one partition, so the held-out test set really
contains unseen policy families rather than paraphrases of training rows.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("DECISION_FAMILY_SPLIT_PAYLOAD_INVALID")
    return payload


def _partition_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "itemCount": len(items),
        "sourceTraceCount": len({
            item["candidate"]["sourceTraceId"] for item in items
        }),
        "taskFamilyCount": len({
            item["candidate"]["task_family"] for item in items
        }),
        "taskFamilyCounts": dict(Counter(
            item["candidate"]["task_family"] for item in items
        )),
        "hardClassCounts": dict(Counter(
            item["candidate"]["hard_case_class"]
            for item in items
            if item["candidate"]["hard_case_class"] is not None
        )),
        "hardCaseCount": len({
            case_id
            for item in items
            for case_id in item.get("hardCaseIds", [])
        }),
    }


def build(
    review_path: Path,
    validation_families: list[str],
    test_families: list[str],
) -> dict[str, Any]:
    payload = _read(review_path)
    items = payload.get("items", [])
    if not isinstance(items, list) or not items:
        raise ValueError("DECISION_FAMILY_SPLIT_ITEMS_MISSING")

    validation = set(validation_families)
    test = set(test_families)
    if not validation or not test:
        raise ValueError("DECISION_FAMILY_SPLIT_HOLDOUT_FAMILIES_REQUIRED")
    if validation & test:
        raise ValueError("DECISION_FAMILY_SPLIT_HOLDOUT_OVERLAP")

    by_family: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        family = item["candidate"]["task_family"]
        by_family[family].append(item)
    known_families = set(by_family)
    missing = sorted((validation | test) - known_families)
    if missing:
        raise ValueError("DECISION_FAMILY_SPLIT_UNKNOWN_FAMILY:" + ",".join(missing))

    split_items: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for family in sorted(known_families):
        split = "test" if family in test else "validation" if family in validation else "train"
        split_items[split].extend(by_family[family])
    for split in split_items:
        split_items[split].sort(key=lambda item: item["signatureHash"])

    family_sets = {
        split: set(item["candidate"]["task_family"] for item in values)
        for split, values in split_items.items()
    }
    if family_sets["train"] & family_sets["validation"]:
        raise ValueError("DECISION_FAMILY_SPLIT_TRAIN_VALIDATION_FAMILY_OVERLAP")
    if family_sets["train"] & family_sets["test"]:
        raise ValueError("DECISION_FAMILY_SPLIT_TRAIN_TEST_FAMILY_OVERLAP")
    if family_sets["validation"] & family_sets["test"]:
        raise ValueError("DECISION_FAMILY_SPLIT_VALIDATION_TEST_FAMILY_OVERLAP")

    trace_sets = {
        split: set(item["candidate"]["sourceTraceId"] for item in values)
        for split, values in split_items.items()
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if trace_sets[left] & trace_sets[right]:
            raise ValueError(f"DECISION_FAMILY_SPLIT_TRACE_OVERLAP:{left}:{right}")

    return {
        "schemaVersion": "p2j4-decision-family-split-v1",
        "sourceReviewSet": review_path.name,
        "status": "SPLIT_REVIEW_PENDING",
        "trainingStarted": False,
        "sourceReviewItemCount": len(items),
        "splitPolicy": {
            "unit": "task_family",
            "randomRowSplit": False,
            "sourceTraceDisjoint": True,
            "unseenFamilyValidation": True,
            "unseenFamilyTest": True,
            "validationFamilies": sorted(validation),
            "testFamilies": sorted(test),
        },
        "splitCounts": {
            split: _partition_summary(values)
            for split, values in split_items.items()
        },
        "familyDisjoint": True,
        "sourceTraceDisjoint": True,
        "splits": split_items,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Split P2-J4 Decision review set by task family")
    parser.add_argument("--review-set", type=Path, required=True)
    parser.add_argument("--validation-family", action="append", required=True)
    parser.add_argument("--test-family", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = build(
            args.review_set,
            args.validation_family,
            args.test_family,
        )
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "INVALID", "errorCode": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({
        "status": result["status"],
        "splitCounts": result["splitCounts"],
        "familyDisjoint": result["familyDisjoint"],
        "sourceTraceDisjoint": result["sourceTraceDisjoint"],
        "output": str(args.output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
