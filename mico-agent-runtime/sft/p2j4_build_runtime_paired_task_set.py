"""Freeze the Base-vs-SFT Runtime paired evaluation task set.

This builder is deliberately deterministic.  It copies only reviewed task
contracts, records the source digest, and refuses duplicate selected IDs.  It
does not run a model, open a socket, or read any secret configuration.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "evals" / "p2j4-hard-variant-task-set-v1.json"
OUTPUT = ROOT / "evals" / "p2j4-runtime-paired-v1" / "task-set.json"

SELECTED_CASE_IDS = [
    # Tool choice: metadata-only and bounded-read guards.
    "p2j4-hard-variant-tool-metadata-01",
    "p2j4-hard-variant-tool-metadata-05",
    "p2j4-hard-variant-tool-bounded-01",
    "p2j4-hard-variant-tool-bounded-05",
    # Focused analysis: five cross-project cases and five without the
    # cross-project obligation, so action efficiency is not conflated with
    # one fixed long path.
    "p2j4-hard-variant-confounder-01",
    "p2j4-hard-variant-confounder-02",
    "p2j4-hard-variant-confounder-03",
    "p2j4-hard-variant-confounder-04",
    "p2j4-hard-variant-confounder-05",
    "p2j4-hard-variant-confounder-06",
    "p2j4-hard-variant-confounder-07",
    "p2j4-hard-variant-confounder-08",
    "p2j4-hard-variant-confounder-10",
    "p2j4-hard-variant-confounder-12",
    # Open exploration: premature-stop and evidence-conflict boundaries.
    "p2j4-hard-variant-premature-stop-01",
    "p2j4-hard-variant-premature-stop-03",
    "p2j4-hard-variant-premature-stop-05",
    "p2j4-hard-variant-premature-stop-07",
    "p2j4-hard-variant-premature-stop-09",
    "p2j4-hard-evidence-conflict-variant-02",
    "p2j4-hard-evidence-conflict-variant-04",
    "p2j4-hard-evidence-conflict-variant-06",
    "p2j4-hard-evidence-conflict-variant-08",
    "p2j4-hard-evidence-conflict-variant-10",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict[str, object]:
    source_payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    source_cases = source_payload.get("cases")
    if not isinstance(source_cases, list):
        raise ValueError("SOURCE_TASK_SET_CASES_INVALID")
    by_id = {case.get("caseId"): case for case in source_cases if isinstance(case, dict)}
    if len(by_id) != len(source_cases):
        raise ValueError("SOURCE_TASK_SET_CASE_IDS_INVALID")
    if len(SELECTED_CASE_IDS) != 24 or len(set(SELECTED_CASE_IDS)) != 24:
        raise ValueError("PAIRED_SELECTION_COUNT_OR_DUPLICATE_INVALID")
    missing = [case_id for case_id in SELECTED_CASE_IDS if case_id not in by_id]
    if missing:
        raise ValueError("PAIRED_SELECTION_CASE_MISSING:" + ",".join(missing))
    cases = [by_id[case_id] for case_id in SELECTED_CASE_IDS]
    distribution = dict(Counter(case["kind"] for case in cases))
    if distribution != {"data_fact": 4, "focused_analysis": 10, "open_exploration": 10}:
        raise ValueError("PAIRED_SELECTION_DISTRIBUTION_INVALID")
    if any(case.get("schemaVersion") != source_payload.get("schemaVersion") for case in cases):
        raise ValueError("PAIRED_SELECTION_SCHEMA_MISMATCH")
    return {
        "schemaVersion": source_payload["schemaVersion"],
        "evaluationVersion": "p2j4-base-sft-runtime-paired-v1",
        "caseCount": len(cases),
        "kindDistribution": distribution,
        "sourceTaskSet": "p2j4-hard-variant-task-set-v1",
        "sourceTaskSetSha256": _sha256(SOURCE),
        "selectedCaseIds": SELECTED_CASE_IDS,
        "pairedModes": ["base", "sft_policy"],
        "sameEnvironmentContract": {
            "runtimeEndpoint": "isolated local scientific runtime",
            "javaToolPort": "same Java internal tool endpoint",
            "readModel": "same MySQL-backed Java read model",
            "knowledgeBackend": "same pgvector + Neo4j hybrid backend",
            "plannerBase": "same Gemini planner configuration",
            "sftPolicy": "Qwen3-8B Decision SFT v4 only for sft_policy mode",
            "controlledScenarioProvider": False,
        },
        "selectionRationale": {
            "tool_choice": "metadata-only versus bounded read; tests source/tool discipline",
            "confounder_analysis": "age/sex/project/country/batch traps with and without cross-project validation",
            "open_exploration": "premature-stop and evidence-conflict boundaries",
        },
        "excludedFromThisPairedRun": [
            "p2j4-data-fact-001",
            "p2j4-focused-analysis-008",
            "p2j4-open-exploration-001",
            "ood-focused-12",
            "all prior Test70 and OOD-v2 offline-only records",
        ],
        "cases": cases,
    }


def main() -> int:
    payload = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT),
        "evaluationVersion": payload["evaluationVersion"],
        "caseCount": payload["caseCount"],
        "kindDistribution": payload["kindDistribution"],
        "sourceTaskSetSha256": payload["sourceTaskSetSha256"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
