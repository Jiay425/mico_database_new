from __future__ import annotations

import json
import os
from pathlib import Path

from mico_agent_runtime.knowledge.database_config import KnowledgeStoreConfiguration
from mico_agent_runtime.knowledge.ingest import ingest_all


def main() -> None:
    configuration = KnowledgeStoreConfiguration.from_environment(os.environ)
    result = ingest_all(configuration)
    output = configuration.indexDirectory / "medical_knowledge_store_manifest.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
