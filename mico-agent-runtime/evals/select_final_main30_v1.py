"""Freeze the small, stratified 30-query final evaluation set."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TARGETS = {
    "direct_evidence": 6,
    "entity_relation": 6,
    "mechanism": 6,
    "multi_hop": 5,
    "cross_paper_synthesis": 4,
    "conflicting_negative": 3,
}


def run(source: Path, output: Path) -> dict[str, Any]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in payload.get("items") or []:
        if isinstance(item, dict):
            grouped[str(item.get("category") or "")].append(item)
    selected: list[dict[str, Any]] = []
    for category, count in TARGETS.items():
        candidates = sorted(
            grouped[category],
            key=lambda item: hashlib.sha256(
                f"p2g-final-main30-v1:{category}:{item.get('queryId')}".encode("utf-8")
            ).hexdigest(),
        )
        if len(candidates) < count:
            raise RuntimeError(f"INSUFFICIENT_QUERY_CATEGORY {category}")
        selected.extend(candidates[:count])
    selected.sort(key=lambda item: str(item.get("queryId") or ""))
    result = {
        "reportVersion": "p2g-final-query-set-main30-v1",
        "querySetVersion": "query-set-v1-main30",
        "status": "FROZEN_MAIN_EVALUATION_SUBSET",
        "selectionPolicy": "Deterministic stratified subset from independent frozen query-set-v1; the remaining questions are held out.",
        "sourceQuerySet": str(source),
        "counts": {"queryCount": len(selected), "byCategory": dict(sorted(Counter(item["category"] for item in selected).items()))},
        "items": selected,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = run(args.source, args.output)
    print(json.dumps({"status": value["status"], "counts": value["counts"], "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
