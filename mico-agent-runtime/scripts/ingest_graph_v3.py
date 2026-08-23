from __future__ import annotations

import json

from mico_agent_runtime.knowledge.database_config import KnowledgeStoreConfiguration
from mico_agent_runtime.knowledge.ingest import ingest_graph_v3_store


def main() -> None:
    configuration = KnowledgeStoreConfiguration.from_environment()
    result = ingest_graph_v3_store(configuration)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

