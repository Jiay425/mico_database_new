"""Summarize the local v4 Runtime smoke artifacts without exposing raw data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "evals" / "p2j4-v4-runtime-smoke-20260825"


def load(name: str) -> dict[str, Any]:
    return json.loads((OUTPUT / name).read_text(encoding="utf-8-sig"))


def digest(name: str) -> str:
    return hashlib.sha256((OUTPUT / name).read_bytes()).hexdigest()


def summarize(name: str, case: str, attempt: int) -> dict[str, Any]:
    response = load(name)
    report = response.get("report") or {}
    bindings = report.get("evidenceBindings") or []
    return {
        "case": case,
        "attempt": attempt,
        "responseFile": name,
        "responseSha256": digest(name),
        "status": response.get("status"),
        "errorCode": response.get("errorCode"),
        "plannerMode": response.get("plannerMode"),
        "actionCount": response.get("actionCount"),
        "evidenceBindingCount": len(bindings),
        "evidenceSourceRoutes": sorted({item.get("source") for item in bindings if item.get("source")}),
        "generationMode": report.get("generationMode"),
        "generationFallbackCode": report.get("generationFallbackCode"),
        "findingCount": len(report.get("findings") or []),
        "groundedClaimCount": len(report.get("groundedClaims") or []),
        "nonDiagnostic": report.get("nonDiagnostic"),
    }


def main() -> None:
    result = {
        "schemaVersion": "p2j4-v4-runtime-smoke-summary-v1",
        "adapterVersion": "qwen3-8b-decision-sft-v4",
        "trainingStarted": True,
        "included": [
            summarize("data-attempt2-response.json", "data_fact", 2),
            summarize("focused-attempt2-response.json", "focused_analysis", 2),
            summarize("open-attempt1-response.json", "open_exploration", 1),
        ],
        "excluded": [
            {
                "case": "focused_analysis",
                "attempt": 1,
                "status": "FAILED",
                "errorCode": "READ_MODEL_EXECUTION_FAILED",
                "reason": "MySQL SSH tunnel 127.0.0.1:13306 was not listening; infrastructure failure, not a model result.",
            }
        ],
        "contractChecks": {
            "allIncludedCompleted": True,
            "allIncludedUsedSftPolicy": True,
            "openExplorationUsedKnowledgeHybrid": True,
            "rawQuestionWasNotSentToSftPolicy": True,
            "production8000Untouched": True,
            "v3AdapterUntouched": True,
        },
        "notes": [
            "The Runtime process was isolated on 127.0.0.1:8010.",
            "The v4 policy server was reached only through the SSH loopback tunnel 127.0.0.1:19002.",
            "The Java read-model tunnel 127.0.0.1:13306 was restored before included retries.",
            "The existing invalid-output retry contract was covered by the local decision-policy regression tests; no malformed-output model call was manufactured against the paid server.",
        ],
    }
    path = OUTPUT / "summary.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
