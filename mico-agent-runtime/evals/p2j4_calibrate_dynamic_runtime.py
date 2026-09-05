"""Create a secret-free calibration manifest for Dynamic Runtime E2E.

This is a local data/contract gate.  It never calls a model, opens MySQL, or
prints environment values.  Service presence is checked only as a boolean
when ``--require-services`` is supplied by the operator immediately before a
canary run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from p2j4_freeze_dynamic_e2e_contract import (
    canonical_json_bytes,
    validate_materializer_config,
    validate_task_set,
)


REQUIRED_FILES = (
    Path("docs/agent-runtime/p2j4-dynamic-materialization-calibration-v1.md"),
    Path("mico_agent_runtime/contracts/materialization.py"),
    Path("mico_agent_runtime/contracts/generated_analysis.py"),
    Path("mico_agent_runtime/graph/generated_analysis.py"),
    Path("mico_agent_runtime/ports/research_planner.py"),
    Path("mico_agent_runtime/graph/scientific_workflow.py"),
    Path("evals/p2j4_dynamic_e2e_task_set_v1.json"),
    Path("evals/p2j4_dynamic_e2e_materializer_config_v1.json"),
    Path("evals/p2j4_freeze_dynamic_e2e_contract.py"),
)
JAVA_FILES = (
    Path("../mico_database_new/src/main/java/com/database/mico_database/agent/contract/QueryPlan.java"),
    Path("../mico_database_new/src/main/java/com/database/mico_database/agent/contract/QueryPlanCompiler.java"),
    Path("../mico_database_new/src/main/java/com/database/mico_database/agent/readmodel/SchemaSemanticCatalogReadModel.java"),
)
SAFE_ENV_KEYS = (
    "MICO_RESEARCH_PLANNER_BASE_URL",
    "MICO_RESEARCH_PLANNER_MODEL",
    "MICO_RESEARCH_PLANNER_TOKEN",
    "MICO_SFT_POLICY_BASE_URL",
    "MICO_SFT_POLICY_MODEL",
    "MICO_SFT_POLICY_TOKEN",
)
REQUIRED_SERVICE_CONFIG_KEYS = (
    "MICO_RESEARCH_PLANNER_BASE_URL",
    "MICO_RESEARCH_PLANNER_MODEL",
    "MICO_RESEARCH_PLANNER_TOKEN",
    "MICO_SFT_POLICY_BASE_URL",
    "MICO_SFT_POLICY_MODEL",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--require-services", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    checks: dict[str, bool] = {}
    failures: list[str] = []
    files: dict[str, str] = {}

    for relative in REQUIRED_FILES + JAVA_FILES:
        absolute = (root / relative).resolve()
        exists = absolute.is_file()
        key = str(relative).replace("\\", "/")
        checks[f"file:{key}"] = exists
        if exists:
            files[key] = sha256_file(absolute)
        else:
            failures.append(f"MISSING_FILE:{key}")

    for module_name in (
        "langgraph",
        "pydantic",
        "mico_agent_runtime.contracts.materialization",
    ):
        try:
            importlib.import_module(module_name)
            checks[f"import:{module_name}"] = True
        except Exception:
            checks[f"import:{module_name}"] = False
            failures.append(f"IMPORT_FAILED:{module_name}")

    task_set_path = root / "evals/p2j4_dynamic_e2e_task_set_v1.json"
    materializer_config_path = root / "evals/p2j4_dynamic_e2e_materializer_config_v1.json"
    planner_source_path = root / "mico_agent_runtime/ports/research_planner.py"
    task_set_sha256 = None
    materializer_config_sha256 = None
    materializer_prompt_hash = None
    if task_set_path.is_file():
        try:
            task_set = json.loads(task_set_path.read_text(encoding="utf-8"))
            task_failures = validate_task_set(task_set)
            checks["dynamicTaskSetContract"] = not task_failures
            failures.extend(f"DYNAMIC_TASK_SET:{item}" for item in task_failures)
            if not task_failures:
                task_set_sha256 = "sha256:" + hashlib.sha256(canonical_json_bytes(task_set)).hexdigest()
        except Exception:
            checks["dynamicTaskSetContract"] = False
            failures.append("DYNAMIC_TASK_SET:INVALID_JSON")
    if materializer_config_path.is_file() and planner_source_path.is_file():
        try:
            materializer_config = json.loads(materializer_config_path.read_text(encoding="utf-8"))
            config_failures, materializer_prompt_hash = validate_materializer_config(
                materializer_config,
                planner_source_path,
            )
            checks["materializerConfigContract"] = not config_failures
            failures.extend(f"MATERIALIZER_CONFIG:{item}" for item in config_failures)
            if not config_failures:
                materializer_config_sha256 = "sha256:" + hashlib.sha256(
                    canonical_json_bytes(materializer_config)
                ).hexdigest()
        except Exception:
            checks["materializerConfigContract"] = False
            failures.append("MATERIALIZER_CONFIG:INVALID_JSON_OR_PROMPT_BINDING")

    env_presence = {key: bool(os.environ.get(key, "").strip()) for key in SAFE_ENV_KEYS}
    if args.require_services:
        for key in REQUIRED_SERVICE_CONFIG_KEYS:
            if not env_presence[key]:
                failures.append(f"SERVICE_CONFIG_MISSING:{key}")

    manifest = {
        "manifestVersion": "dynamic-runtime-calibration-v1",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "status": "CALIBRATION_PASS" if not failures else "CALIBRATION_FAILED",
        "checks": checks,
        "files": files,
        "serviceConfigPresence": env_presence,
        "dynamicTaskSetSha256": task_set_sha256,
        "materializerConfigSha256": materializer_config_sha256,
        "materializerSystemPromptHash": materializer_prompt_hash,
        "dynamicTaskSetContractChecked": checks.get("dynamicTaskSetContract", False),
        "materializerConfigContractChecked": checks.get("materializerConfigContract", False),
        "serviceValuesIncluded": False,
        "failures": failures,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
