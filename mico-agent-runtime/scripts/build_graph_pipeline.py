from __future__ import annotations

import json
from pathlib import Path

from mico_agent_runtime.knowledge.graph_pipeline import (
    GRAPH_PIPELINE_VERSION,
    build_versioned_graph,
    save_graph_build,
)


def _jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    runtime_root = Path(__file__).resolve().parents[1]
    workspace_root = runtime_root.parent
    knowledge_dir = workspace_root / "mico_database_new" / "references" / "knowledge" / "medical" / "rag"
    chunks_path = knowledge_dir / "medical_chunks.jsonl"
    output_path = knowledge_dir / "medical_knowledge_graph_v4.jsonl"
    manifest_path = knowledge_dir / "medical_knowledge_graph_v4_manifest.json"
    rows = list(_jsonl(chunks_path))
    result = build_versioned_graph(rows, GRAPH_PIPELINE_VERSION)
    save_graph_build(result, output_path, manifest_path)
    print(json.dumps(result.manifest.model_dump(mode="json"), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

