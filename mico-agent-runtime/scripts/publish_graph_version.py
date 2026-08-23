from __future__ import annotations

import json
from pathlib import Path

from mico_agent_runtime.knowledge.graph_pipeline import publish_graph_manifest


def main() -> None:
    runtime_root = Path(__file__).resolve().parents[1]
    knowledge_dir = runtime_root.parent / "mico_database_new" / "references" / "knowledge" / "medical" / "rag"
    manifest_path = knowledge_dir / "medical_knowledge_graph_v4_manifest.json"
    registry_path = knowledge_dir / "medical_knowledge_graph_registry.json"
    registry = publish_graph_manifest(manifest_path, registry_path)
    print(json.dumps(registry.model_dump(mode="json"), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

