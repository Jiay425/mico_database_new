"""Audit which judged-relevant chunks RRF adds to or removes from Dense Top20."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ranked_ids(case: dict[str, Any], branch: str) -> list[str]:
    return [str(row["chunkId"]) for row in case["branches"][branch]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", required=True)
    parser.add_argument("--judged", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hybrid-branch", default="hybrid_rrf")
    args = parser.parse_args()

    pool = _load(Path(args.pool))
    judged = _load(Path(args.judged))
    labels_by_query = {
        str(row["queryId"]): {
            str(label["chunkId"]): bool(label.get("relevant"))
            for label in row.get("labels", [])
        }
        for row in judged["queries"]
    }

    summaries: list[dict[str, Any]] = []
    aggregate = Counter()
    per_category: dict[str, Counter[str]] = defaultdict(Counter)
    for case in pool["cases"]:
        query_id = str(case["queryId"])
        category = str(case["category"])
        labels = labels_by_query[query_id]
        dense = _ranked_ids(case, "dense")
        hybrid = _ranked_ids(case, args.hybrid_branch)
        dense_set, hybrid_set = set(dense), set(hybrid)
        relevant = {chunk_id for chunk_id, value in labels.items() if value}
        added = sorted((hybrid_set - dense_set) & relevant)
        displaced = sorted((dense_set - hybrid_set) & relevant)
        shared = (dense_set & hybrid_set) & relevant
        row = {
            "queryId": query_id,
            "category": category,
            "denseRelevantCount": len(dense_set & relevant),
            "hybridRelevantCount": len(hybrid_set & relevant),
            "newRelevantChunkIds": added,
            "displacedRelevantChunkIds": displaced,
            "sharedRelevantCount": len(shared),
        }
        summaries.append(row)
        stats = per_category[category]
        aggregate["newRelevantChunks"] += len(added)
        aggregate["displacedRelevantChunks"] += len(displaced)
        aggregate["denseRelevantTop20"] += len(dense_set & relevant)
        aggregate["hybridRelevantTop20"] += len(hybrid_set & relevant)
        aggregate["queriesWithNewRelevant"] += int(bool(added))
        aggregate["queriesWithDisplacedRelevant"] += int(bool(displaced))
        stats["newRelevantChunks"] += len(added)
        stats["displacedRelevantChunks"] += len(displaced)
        stats["denseRelevantTop20"] += len(dense_set & relevant)
        stats["hybridRelevantTop20"] += len(hybrid_set & relevant)

    output = {
        "reportVersion": "final-hybrid-v2-rrf-delta-audit-v1",
        "evaluationBoundary": "Dev30 only; diagnostic, not an independent final result.",
        "hybridBranch": args.hybrid_branch,
        "aggregate": dict(aggregate),
        "perCategory": {key: dict(value) for key, value in sorted(per_category.items())},
        "queries": summaries,
    }
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "AUDIT_COMPLETE", "aggregate": output["aggregate"], "output": args.output}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
