from __future__ import annotations

import argparse
import json
from pathlib import Path

from mico_agent_runtime.contracts.review import GraphReviewDecision
from mico_agent_runtime.knowledge.graph_pipeline import approve_graph_manifest
from mico_agent_runtime.knowledge.graph_review import (
    apply_review_decisions,
    build_publication_approval,
)
from mico_agent_runtime.contracts.review import GraphReviewQueue


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply complete typed graph review decisions; never auto-approves or publishes."
    )
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--updated-queue", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    args = parser.parse_args()

    queue = GraphReviewQueue.model_validate_json(args.queue.read_text(encoding="utf-8"))
    payload = json.loads(args.decisions.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise SystemExit("GRAPH_REVIEW_DECISIONS_MUST_BE_ARRAY")
    decisions = [GraphReviewDecision.model_validate(item) for item in payload]
    updated = apply_review_decisions(queue, decisions)
    args.updated_queue.write_text(updated.model_dump_json(indent=2), encoding="utf-8")
    if updated.status != "CLOSED" or updated.pendingCount or updated.rejectedCount:
        raise SystemExit("GRAPH_REVIEW_NOT_READY_FOR_PUBLICATION")
    approval = build_publication_approval(updated, approved_by=args.reviewer)
    approve_graph_manifest(args.manifest, updated, approval)
    args.approval.write_text(approval.model_dump_json(indent=2), encoding="utf-8")
    print(json.dumps({"graphVersion": updated.graphVersion, "status": "approved"}, sort_keys=True))


if __name__ == "__main__":
    main()
