"""Select Final30 from frozen query-set-v1 excluding the observed Dev30."""

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


def run(source: Path, dev: Path, output: Path, extra_exclude: Path | None = None) -> dict[str, Any]:
    source_payload = json.loads(source.read_text(encoding="utf-8"))
    dev_payload = json.loads(dev.read_text(encoding="utf-8"))
    dev_ids = {str(item.get("queryId")) for item in dev_payload.get("items") or []}
    if extra_exclude is not None:
        extra_payload = json.loads(extra_exclude.read_text(encoding="utf-8"))
        dev_ids.update(str(item.get("queryId")) for item in extra_payload.get("items") or [])
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in source_payload.get("items") or []:
        if isinstance(item, dict) and str(item.get("queryId")) not in dev_ids:
            grouped[str(item.get("category") or "")].append(item)
    selected: list[dict[str, Any]] = []
    for category, count in TARGETS.items():
        candidates = sorted(
            grouped[category],
            key=lambda item: hashlib.sha256(
                f"p2g-gated-final30-v1:{category}:{item.get('queryId')}".encode("utf-8")
            ).hexdigest(),
        )
        if len(candidates) < count:
            raise RuntimeError(f"INSUFFICIENT_HOLDOUT_CATEGORY {category}")
        selected.extend(candidates[:count])
    selected.sort(key=lambda item: str(item.get("queryId") or ""))
    result = {
        "reportVersion": "p2g-gated-final30-query-set-v1",
        "querySetVersion": "query-set-v1-final30-heldout",
        "status": "FROZEN_FINAL_HOLDOUT",
        "selectionPolicy": "Deterministic stratified selection from frozen query-set-v1 after excluding Dev30.",
        "sourceQuerySet": str(source),
        "excludedDevQuerySet": str(dev),
        "extraExcludedQuerySet": str(extra_exclude) if extra_exclude is not None else None,
        "excludedDevQueryCount": len(dev_ids),
        "counts": {"queryCount": len(selected), "byCategory": dict(sorted(Counter(item["category"] for item in selected).items()))},
        "items": selected,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--extra-exclude", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.source, args.dev, args.output, args.extra_exclude)
    print(json.dumps({"status": report["status"], "counts": report["counts"], "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
