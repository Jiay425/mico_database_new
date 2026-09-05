from __future__ import annotations

"""Build a deterministic, provider-free GraphRAG relation audit set.

The output is an annotation contract, not an automatic truth set.  A judge
must inspect the evidence span and fill the six boolean quality labels.  The
script can optionally rebuild a graph from the same chunk asset to recover the
rejected-record sidecar; this makes rejected false-negative auditing possible
without changing the canonical staging artifact.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from mico_agent_runtime.knowledge.graph_pipeline import build_versioned_graph


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _stable_order(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{key}:{row.get('edgeId') or row.get('nodeId') or row.get('recordId')}".encode("utf-8")
        ).hexdigest(),
    )


def _sample(rows: list[dict[str, Any]], count: int, key: str) -> list[dict[str, Any]]:
    return _stable_order(rows, key)[: max(0, count)]


def _counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(field) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _label(record: dict[str, Any], nodes: dict[str, dict[str, Any]], endpoint: str) -> str | None:
    node = nodes.get(str(record.get(endpoint)))
    if node is None:
        return None
    return str(node.get("canonicalLabel") or node.get("label") or "") or None


def _audit_item(
    edge: dict[str, Any],
    nodes: dict[str, dict[str, Any]],
    chunks: dict[str, dict[str, Any]],
    bucket: str,
    ordinal: int,
) -> dict[str, Any]:
    chunk_id = str(edge.get("evidenceChunkId") or "")
    chunk = chunks.get(chunk_id) or {}
    return {
        "auditId": f"graph-audit-{ordinal:04d}",
        "bucket": bucket,
        "graphVersion": edge.get("graphVersion"),
        "edgeId": edge.get("edgeId"),
        "recordType": edge.get("recordType", "edge"),
        "sourceEntityId": edge.get("source"),
        "sourceEntity": _label(edge, nodes, "source"),
        "relation": edge.get("relation"),
        "relationClass": edge.get("relationClass"),
        "targetEntityId": edge.get("target"),
        "targetEntity": _label(edge, nodes, "target"),
        "assertionStatus": edge.get("assertionStatus"),
        "confidence": edge.get("confidence"),
        "evidenceChunkId": chunk_id or None,
        "evidenceText": edge.get("evidenceText") or chunk.get("text"),
        "evidenceStart": edge.get("evidenceStart"),
        "evidenceEnd": edge.get("evidenceEnd"),
        "pmcid": chunk.get("pmcid") or nodes.get(str(edge.get("source")), {}).get("pmcid"),
        "section": chunk.get("section"),
        "sourceUrl": chunk.get("sourceUrl"),
        "extractionMethod": edge.get("extractionMethod"),
        "qualityStatus": edge.get("qualityStatus"),
        "rejectionIssueCodes": edge.get("rejectionIssueCodes", []),
        "judge": {
            "entitySourceCorrect": None,
            "entityTargetCorrect": None,
            "relationTypeCorrect": None,
            "directionCorrect": None,
            "negationSpeculationHandled": None,
            "evidenceSupportsRelation": None,
            "decision": "pending_llm_judge",
            "confidence": None,
            "evidenceSpan": None,
            "reason": None,
        },
    }


def run(
    graph_path: Path,
    queue_path: Path,
    chunks_path: Path,
    output_path: Path,
    rejected_path: Path | None,
    rebuild_rejected: bool,
    accepted_count: int,
    review_count: int,
    rejected_count: int,
    rejected_output_path: Path | None = None,
) -> dict[str, Any]:
    graph_rows = list(_jsonl(graph_path))
    nodes = {str(row.get("nodeId")): row for row in graph_rows if row.get("recordType") == "node"}
    semantic = [
        row for row in graph_rows
        if row.get("recordType") == "edge" and row.get("relationClass") != "structural"
    ]
    accepted = [row for row in semantic if row.get("qualityStatus") == "accepted"]
    review = [row for row in semantic if row.get("qualityStatus") == "review_required"]
    chunks = {str(row.get("chunkId")): row for row in _jsonl(chunks_path) if row.get("chunkId")}

    rejected_rows: list[dict[str, Any]] = []
    rejected_source = None
    rejected_available = False
    if rejected_path and rejected_path.exists():
        rejected_available = True
        rejected_rows = [
            row for row in _jsonl(rejected_path)
            if row.get("recordType") == "edge"
            and row.get("relationClass") != "structural"
        ]
        rejected_source = str(rejected_path)
    elif rebuild_rejected:
        rejected_available = True
        rebuilt = build_versioned_graph(
            list(_jsonl(chunks_path)),
            str((graph_rows[0] if graph_rows else {}).get("graphVersion") or "fulltext-provenance-graphrag-v4-v2m-r3"),
            input_asset=chunks_path.name,
        )
        rejected_rows = [
            row for row in rebuilt.rejectedRecords
            if row.get("recordType") == "edge" and row.get("relationClass") != "structural"
        ]
        rejected_source = "reconstructed_in_memory_from_chunk_asset"

    if rejected_output_path is not None and rejected_rows:
        rejected_output_path.parent.mkdir(parents=True, exist_ok=True)
        rejected_output_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rejected_rows),
            encoding="utf-8",
        )

    selected: list[dict[str, Any]] = []
    for bucket, rows, count in (
        ("accepted", accepted, accepted_count),
        ("review_required", review, review_count),
        ("rejected", rejected_rows, rejected_count),
    ):
        for index, row in enumerate(_sample(rows, count, bucket), 1):
            selected.append(_audit_item(row, nodes, chunks, bucket, len(selected) + 1))

    queue_meta: dict[str, Any] = {}
    if queue_path.exists():
        queue = json.loads(queue_path.read_text(encoding="utf-8"))
        queue_meta = {
            "status": queue.get("status"),
            "pendingCount": queue.get("pendingCount"),
            "queueHash": queue.get("queueHash"),
        }
    result = {
        "reportVersion": "p2g-graph-quality-audit-v1",
        "status": "pending_llm_judge",
        "annotationContract": {
            "scale": "boolean per field; all fields required before aggregation",
            "requiredChecks": [
                "entity source correct",
                "entity target correct",
                "relation type correct",
                "direction correct",
                "negation/speculation handled correctly",
                "evidence span supports the relation",
            ],
            "decisionRule": "accept only when all entity/relation/direction/evidence checks are true; keep conflicted/speculative records explicitly labelled",
        },
        "sources": {
            "graph": str(graph_path),
            "reviewQueue": str(queue_path),
            "chunks": str(chunks_path),
            "rejected": rejected_source,
            "rejectedUnavailable": not rejected_available,
        },
        "queue": queue_meta,
        "counts": {
            "graphSemanticAccepted": len(accepted),
            "graphSemanticReviewRequired": len(review),
            "graphSemanticRejectedAvailable": len(rejected_rows),
            "selectedAccepted": sum(item["bucket"] == "accepted" for item in selected),
            "selectedReviewRequired": sum(item["bucket"] == "review_required" for item in selected),
            "selectedRejected": sum(item["bucket"] == "rejected" for item in selected),
        },
        "distributions": {
            "acceptedRelationClass": _counts(accepted, "relationClass"),
            "reviewRelationClass": _counts(review, "relationClass"),
            "reviewAssertionStatus": _counts(review, "assertionStatus"),
            "rejectedIssueCode": {
                code: sum(code in (row.get("rejectionIssueCodes") or []) for row in rejected_rows)
                for code in sorted({
                    code
                    for row in rejected_rows
                    for code in (row.get("rejectionIssueCodes") or [])
                })
            },
        },
        "items": selected,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rejected", type=Path)
    parser.add_argument("--no-rebuild-rejected", action="store_true")
    parser.add_argument("--accepted-count", type=int, default=100)
    parser.add_argument("--review-count", type=int, default=100)
    parser.add_argument("--rejected-count", type=int, default=100)
    parser.add_argument("--rejected-output", type=Path)
    args = parser.parse_args()
    result = run(
        args.graph,
        args.queue,
        args.chunks,
        args.output,
        args.rejected,
        not args.no_rebuild_rejected,
        args.accepted_count,
        args.review_count,
        args.rejected_count,
        args.rejected_output,
    )
    print(json.dumps(result["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
