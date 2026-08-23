from __future__ import annotations

import argparse
import json
from pathlib import Path

from mico_agent_runtime.knowledge.graph_pipeline import GraphBuildResult, load_graph_manifest
from mico_agent_runtime.knowledge.graph_review import build_graph_review_queue


def main() -> None:
    runtime_root = Path(__file__).resolve().parents[1]
    default_dir = runtime_root.parent / "mico_database_new" / "references" / "knowledge" / "medical" / "rag"
    parser = argparse.ArgumentParser(
        description="Export a typed graph review queue; this never approves or publishes a graph."
    )
    parser.add_argument("--graph", type=Path, default=default_dir / "medical_knowledge_graph_v4.jsonl")
    parser.add_argument("--manifest", type=Path, default=default_dir / "medical_knowledge_graph_v4_manifest.json")
    parser.add_argument("--output", type=Path, default=default_dir / "medical_knowledge_graph_v4_review_queue.json")
    args = parser.parse_args()
    records = [
        json.loads(line)
        for line in args.graph.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = GraphBuildResult(records=records, manifest=load_graph_manifest(args.manifest))
    queue = build_graph_review_queue(result)
    args.output.write_text(queue.model_dump_json(indent=2), encoding="utf-8")
    print(json.dumps({"graphVersion": queue.graphVersion, "status": queue.status, "pendingCount": queue.pendingCount}, sort_keys=True))


if __name__ == "__main__":
    main()
