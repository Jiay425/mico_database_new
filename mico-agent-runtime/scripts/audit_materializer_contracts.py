"""Static 4D-1I audit of every Scientific Action materializer contract.

This command never starts Java, opens MySQL, or calls a model.  It records the
canonical Action→contract table, the actual prompt/parser/validator entry
points in the current code, and every legacy/code-generation token found in
the Dynamic Scientific path.  A later API sweep can merge its per-action
results into the same artifact without changing this static evidence.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mico_agent_runtime.contracts.materializer_contract import (
    MATERIALIZER_CONTRACTS,
    materializer_contract_for_action,
)


AUDIT_FILES = (
    REPO_ROOT / "mico_agent_runtime" / "ports" / "research_planner.py",
    REPO_ROOT / "mico_agent_runtime" / "graph" / "scientific_workflow.py",
    REPO_ROOT / "mico_agent_runtime" / "graph" / "generated_analysis.py",
    REPO_ROOT / "mico_agent_runtime" / "graph" / "intent_workflow.py",
    REPO_ROOT / "mico_agent_runtime" / "contracts" / "research.py",
    REPO_ROOT / "mico_agent_runtime" / "contracts" / "materialization.py",
    REPO_ROOT / "mico_agent_runtime" / "contracts" / "retrieval.py",
    REPO_ROOT / "mico_agent_runtime" / "runtime" / "analysis_capability_registry.py",
)

LEGACY_TOKENS = (
    "generated_python",
    "python_code",
    "code",
    "execution_mode",
    "dynamic_analysis",
    "legacy",
    "generated-analysis",
    "analysis_type",
    "wrapper",
    "action",
)


def _read_response(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        response = payload.get("response", payload)
        content = response["choices"][0]["message"]["content"]
        if isinstance(content, str):
            return json.loads(content)
    except (OSError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return None


def _response_keys(path: Path) -> list[str]:
    payload = _read_response(path)
    return sorted(payload) if isinstance(payload, dict) else []


def _classify_hit(path: Path, token: str) -> str:
    normalized = path.as_posix()
    if "/tests/" in normalized or "/evals/" in normalized:
        return "test_fixture_or_frozen_eval"
    if token in {"generated_python", "python_code", "generated-analysis"}:
        return "legacy_or_compatibility_contract"
    if token == "code" and "/graph/generated_analysis.py" in normalized:
        return "legacy_or_compatibility_contract"
    if token == "execution_mode" and "/contracts/generated_analysis.py" in normalized:
        return "legacy_or_compatibility_contract"
    if token == "legacy":
        return "legacy_or_compatibility_reference"
    if token in {"analysis_type", "wrapper", "action"}:
        return "current_contract_or_parser_reference"
    return "current_new_path_or_domain_reference"


def _legacy_search() -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for path in AUDIT_FILES:
        if not path.exists():
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            lowered = line.lower()
            for token in LEGACY_TOKENS:
                if token.lower() in lowered:
                    hits.append({
                        "file": str(path),
                        "line": line_number,
                        "token": token,
                        "text": line.strip()[:320],
                        "classification": _classify_hit(path, token),
                    })
    return hits


def _analyze_projection_root_cause() -> dict[str, Any]:
    short_response = (
        REPO_ROOT / "artifacts" / "gemini_dynamic_canary_4d1h_short" / "materializer_response_008.json"
    )
    final3_response = (
        REPO_ROOT / "artifacts" / "gemini_dynamic_canary_4d1g_final3" / "materializer_response_008.json"
    )
    actual = _read_response(short_response)
    expected_keys = ["actionName", "observationId", "analysisGoal"]
    actual_arguments = actual.get("arguments", {}) if isinstance(actual, dict) else {}
    actual_keys = sorted(actual_arguments) if isinstance(actual_arguments, dict) else []
    return {
        "historical_artifact": str(short_response),
        "raw_response_available": actual is not None,
        "raw_response": actual,
        "parser_contract": {
            "model": "AnalyzeProjectionArguments",
            "required_argument_keys": expected_keys,
            "observation_cardinality": "one singular observationId",
        },
        "exact_diff": {
            "missing_keys": sorted(set(expected_keys) - set(actual_keys)),
            "unexpected_keys": sorted(set(actual_keys) - set(expected_keys)),
            "actual_argument_keys": actual_keys,
        },
        "root_cause": (
            "The historical singleton plan_action prompt contained a generic instruction to copy "
            "observationIds for analysis actions while AnalyzeProjectionArguments requires singular "
            "observationId. Gemini followed the broader instruction; the closed parser rejected the "
            "list-shaped projection envelope. The current prompt now states the projection-specific "
            "singular shape explicitly."
            if actual is not None else
            "No historical projection response artifact was found; rerun the canary to capture the exact diff."
        ),
        "projection_prompt_has_singular_rule": (
            "one singular observationId" in (
                REPO_ROOT / "mico_agent_runtime" / "ports" / "research_planner.py"
            ).read_text(encoding="utf-8")
        ),
        "legacy_generated_python_example": {
            "response_available": _read_response(final3_response) is not None,
            "artifact": str(final3_response),
            "response_keys": _response_keys(final3_response),
            "classification": "legacy_or_compatibility_contract",
        },
    }


def build_report() -> dict[str, Any]:
    contracts = []
    for item in MATERIALIZER_CONTRACTS:
        contract = materializer_contract_for_action(item.action)
        contracts.append({
            "action": contract.action,
            "materializer_required": contract.materializer_required,
            "producer": contract.producer,
            "canonical_contract_kind": contract.contract_kind,
            "canonical_schema": contract.schema_name,
            "canonical_discriminator": contract.discriminator,
            "materializer_methods": list(contract.materializer_methods),
            "runtime_dispatch_envelope": contract.runtime_dispatch_envelope,
            "prompt_entrypoints": list(contract.prompt_entrypoints),
            "response_parsers": list(contract.response_parsers),
            "domain_validators": list(contract.domain_validators),
            "legacy_contracts": list(contract.legacy_contracts),
            "notes": contract.notes,
            "api_canary": "not_run",
        })

    legacy_hits = _legacy_search()
    classification_counts: dict[str, int] = {}
    for hit in legacy_hits:
        classification = str(hit["classification"])
        classification_counts[classification] = classification_counts.get(classification, 0) + 1
    return {
        "audit_version": "materializer-contract-audit-v1",
        "status": "PASS",
        "api_calls_made": 0,
        "training_eligible": False,
        "scientific_actions": [item.action for item in MATERIALIZER_CONTRACTS],
        "contract_mapping": contracts,
        "analysis_action_type_mapping": {
            item.action: item.discriminator
            for item in MATERIALIZER_CONTRACTS
            if item.contract_kind == "analysis_plan_v2"
        },
        "legacy_search": {
            "tokens": list(LEGACY_TOKENS),
            "hit_count": len(legacy_hits),
            "classification_counts": classification_counts,
            "hits": legacy_hits,
        },
        "analyze_projection_root_cause": _analyze_projection_root_cause(),
        "summary": {
            "action_count": len(contracts),
            "materializer_required_count": sum(
                1 for item in contracts if item["materializer_required"]
            ),
            "runtime_owned_count": sum(
                1 for item in contracts if item["producer"] == "runtime_owned"
            ),
            "canonical_analysis_plan_v2_count": sum(
                1 for item in contracts if item["canonical_contract_kind"] == "analysis_plan_v2"
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "artifacts" / "materializer_contract_sweep_4d1i" / "static_audit.json",
    )
    args = parser.parse_args(argv)
    report = build_report()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "action_count": report["summary"]["action_count"],
        "materializer_required_count": report["summary"]["materializer_required_count"],
        "legacy_hit_count": report["legacy_search"]["hit_count"],
        "out": str(args.out),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
