"""Run the quota-safe, read-only knowledge backend preflight.

No Gemini embedding or retrieval call is made.  The command checks canonical
wiring, pgvector schema/table visibility, Neo4j connectivity, and local index
assets, then writes a redacted JSON report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mico_agent_runtime.knowledge.health import probe_knowledge_backend  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "knowledge_backend_preflight.json",
    )
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="skip network/database connectivity and inspect only wiring/assets",
    )
    args = parser.parse_args()
    report = probe_knowledge_backend(probe_connectivity=not args.config_only)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["knowledge_wiring_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
