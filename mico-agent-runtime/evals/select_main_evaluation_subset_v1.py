"""Create the fixed 50-query main evaluation subset from query-set-v1 pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TARGETS = {
    "direct_evidence": 10,
    "entity_relation": 10,
    "mechanism": 10,
    "multi_hop": 8,
    "cross_paper_synthesis": 7,
    "conflicting_negative": 5,
}


def _stable_order(items: list[dict[str, Any]], category: str) -> list[dict[str, Any]]:
    return sorted(
        items,
        key=lambda item: hashlib.sha256(
            f"p2g-main-eval-50-v1:{category}:{item.get('queryId')}".encode("utf-8")
        ).hexdigest(),
    )


def run(pool_path: Path, output_path: Path) -> dict[str, Any]:
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in pool.get("cases") or []:
        grouped[str(case.get("category") or "")].append(case)
    selected: list[dict[str, Any]] = []
    for category, wanted in TARGETS.items():
        candidates = _stable_order(grouped[category], category)
        if len(candidates) < wanted:
            raise RuntimeError(f"INSUFFICIENT_CATEGORY_QUERIES {category}: {len(candidates)} < {wanted}")
        selected.extend(candidates[:wanted])
    selected.sort(key=lambda item: str(item.get("queryId")))
    result = {
        "poolVersion": "p2g-independent-query-pool-v1-main50",
        "status": "FROZEN_MAIN_EVALUATION_SUBSET",
        "selectionPolicy": "deterministic stratified sample from frozen query-set-v1; remaining 50 queries are held out.",
        "sourcePool": str(pool_path),
        "queryCount": len(selected),
        "categoryCounts": dict(sorted(Counter(str(item.get("category")) for item in selected).items())),
        "branches": pool.get("branches"),
        "cases": selected,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.pool, args.output)
    print(json.dumps({
        "status": result["status"],
        "queryCount": result["queryCount"],
        "categoryCounts": result["categoryCounts"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
