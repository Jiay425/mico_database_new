from __future__ import annotations

"""Matched audit of r3 self-loop rejections against the r4 extractor.

Matching is intentionally conservative: same evidence chunk, evidence start,
and evidence end, plus a non-self semantic r4 edge.  An unmatched record is
not called a false negative; it is a candidate that requires sentence-level
review because no deterministic replacement was emitted.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run(r3_rejected_path: Path, r4_graph_path: Path, output_path: Path) -> dict[str, Any]:
    old = [
        row for row in _jsonl(r3_rejected_path)
        if row.get("recordType") == "edge"
        and "SELF_LOOP_RELATION" in (row.get("rejectionIssueCodes") or [])
    ]
    new = [
        row for row in _jsonl(r4_graph_path)
        if row.get("recordType") == "edge"
        and row.get("relationClass") != "structural"
        and row.get("source") != row.get("target")
    ]
    by_span: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = {}
    for row in new:
        by_span.setdefault(
            (row.get("evidenceChunkId"), row.get("evidenceStart"), row.get("evidenceEnd")),
            [],
        ).append(row)
    details: list[dict[str, Any]] = []
    recovered = 0
    for row in old:
        replacements = by_span.get(
            (row.get("evidenceChunkId"), row.get("evidenceStart"), row.get("evidenceEnd")),
            [],
        )
        if replacements:
            recovered += 1
        details.append({
            "oldEdgeId": row.get("edgeId"),
            "evidenceChunkId": row.get("evidenceChunkId"),
            "oldRelation": row.get("relation"),
            "oldAssertionStatus": row.get("assertionStatus"),
            "oldEvidenceText": row.get("evidenceText"),
            "replacementCount": len(replacements),
            "replacements": [
                {
                    "edgeId": item.get("edgeId"),
                    "source": item.get("source"),
                    "relation": item.get("relation"),
                    "relationClass": item.get("relationClass"),
                    "target": item.get("target"),
                    "assertionStatus": item.get("assertionStatus"),
                    "confidence": item.get("confidence"),
                }
                for item in replacements
            ],
            "status": "recovered_distinct_relation" if replacements else "no_deterministic_replacement",
        })
    result = {
        "reportVersion": "p2g-graph-selfloop-recovery-audit-v1",
        "status": "matched_diagnostic",
        "matchingRule": "same evidenceChunkId + evidenceStart + evidenceEnd + non-self semantic r4 edge",
        "sources": {"r3Rejected": str(r3_rejected_path), "r4Graph": str(r4_graph_path)},
        "summary": {
            "r3SelfLoopCount": len(old),
            "r4DistinctSemanticEdgeCount": len(new),
            "recoveredWithDeterministicReplacement": recovered,
            "noDeterministicReplacement": len(old) - recovered,
            "recoveryRate": round(recovered / len(old), 8) if old else 0.0,
            "replacementRelationCounts": dict(sorted(Counter(
                replacement["relation"]
                for item in details
                for replacement in item["replacements"]
            ).items())),
            "replacementRelationClassCounts": dict(sorted(Counter(
                replacement["relationClass"]
                for item in details
                for replacement in item["replacements"]
            ).items())),
        },
        "records": details,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r3-rejected", type=Path, required=True)
    parser.add_argument("--r4-graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.r3_rejected, args.r4_graph, args.output)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
