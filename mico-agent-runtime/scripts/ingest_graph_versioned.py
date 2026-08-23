from __future__ import annotations

import argparse
import json

from mico_agent_runtime.knowledge.database_config import KnowledgeStoreConfiguration
from mico_agent_runtime.knowledge.ingest import ingest_graph_versioned_store


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest a validated graph build into Neo4j; publication requires manifest approval."
    )
    parser.add_argument("--publish", action="store_true", help="switch the graph build to published")
    args = parser.parse_args()
    configuration = KnowledgeStoreConfiguration.from_environment()
    result = ingest_graph_versioned_store(configuration, publish=args.publish)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
