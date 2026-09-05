"""Validate and manifest the frozen Dynamic E2E task/config contract.

This is a local, no-API gate.  It reads only the immutable task set, the
immutable Materializer configuration, and the planner source needed to derive
the static system-prompt fingerprint.  It never starts a service, opens a
database, contacts a model, or writes raw task text to the manifest.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ACTION_NAMES = {
    "inspect_cohort",
    "execute_read_query",
    "compare_groups",
    "analyze_projection",
    "stratified_analysis",
    "adjust_confounders",
    "cross_project_validate",
    "cross_disease_validate",
    "retrieve_evidence",
    "finish",
}
ANALYSIS_ACTIONS = {
    "compare_groups",
    "analyze_projection",
    "stratified_analysis",
    "adjust_confounders",
    "cross_project_validate",
    "cross_disease_validate",
}
ANALYSIS_TYPES = {
    "group_comparison",
    "stratified_comparison",
    "confounder_adjustment",
    "projection",
    "cross_project_validation",
    "cross_disease_validation",
}
PLAN_KINDS = {"query_plan", "analysis_plan", "retrieval_plan", "finish_boundary"}
MATERIALIZER_ORIGINS = {"model", "runtime_owned"}
SCOPES = {"mico:query:read", "mico:evidence:read", "mico:research:read"}
TASK_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*-[0-9]{3}$")
FORBIDDEN_QUESTION_PATTERNS = (
    re.compile(r"(?i)api[_ -]?key|access[_ -]?token|password|bearer"),
    re.compile(r"(?i)jdbc:|mysql://|postgres(?:ql)?://|https?://"),
    re.compile(r"(?i)\bselect\s+\*|\bselect\b.*\bfrom\b|\bjoin\b.*\bon\b|\bwhere\b.*[=<>]|\bwith\b.*\bselect\b"),
    re.compile(r"[A-Za-z]:\\|/etc/|/home/|\\Users\\"),
    re.compile(r"(?i)sample[_ -]?id|patient[_ -]?id|database[_ -]?url"),
)


def _error(path: str, message: str) -> str:
    return f"{path}:{message}"


def _require(condition: bool, failures: list[str], path: str, message: str) -> None:
    if not condition:
        failures.append(_error(path, message))


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def static_materializer_prompt_hash(source_path: Path) -> str:
    """Hash only the static prompt literals, excluding runtime catalog/state."""

    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    planner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "HttpResearchPlannerPort"
    )
    prompt_parts: list[str] = []
    for function_name in ("plan_action", "generate_typed_analysis"):
        function = next(
            node
            for node in planner.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        )
        assignment = next(
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "system" for target in node.targets)
        )
        literals = [
            node.value
            for node in ast.walk(assignment.value)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ]
        prompt_parts.append(function_name + "\n" + "\n".join(literals))
    return sha256_bytes("\n---\n".join(prompt_parts).encode("utf-8"))


def validate_task_set(payload: Any) -> list[str]:
    failures: list[str] = []
    _require(isinstance(payload, dict), failures, "$", "must be an object")
    if not isinstance(payload, dict):
        return failures

    _require(payload.get("schemaVersion") == "p2j4-dynamic-e2e-task-set-v1", failures, "$.schemaVersion", "must be v1")
    _require(payload.get("status") == "FROZEN", failures, "$.status", "must be FROZEN")
    _require(payload.get("purpose") == "Dynamic Materialization contract and paired inference evaluation", failures, "$.purpose", "unexpected purpose")
    _require(payload.get("notForTraining") is True, failures, "$.notForTraining", "must be true")
    tasks = payload.get("tasks")
    _require(isinstance(tasks, list), failures, "$.tasks", "must be a list")
    if not isinstance(tasks, list):
        return failures

    task_count = payload.get("taskCount")
    canary_count = payload.get("canaryCount")
    paired_count = payload.get("pairedCount")
    _require(task_count == len(tasks), failures, "$.taskCount", "does not match tasks")
    canaries = [task for task in tasks if isinstance(task, dict) and str(task.get("kind", "")).startswith("canary_")]
    paired = [task for task in tasks if isinstance(task, dict) and not str(task.get("kind", "")).startswith("canary_")]
    _require(canary_count == len(canaries), failures, "$.canaryCount", "does not match canary tasks")
    _require(paired_count == len(paired), failures, "$.pairedCount", "does not match paired tasks")
    _require(len(canaries) == 3, failures, "$.canaryCount", "Dynamic E2E requires exactly 3 canaries")
    _require(len(paired) == 9, failures, "$.pairedCount", "Dynamic E2E freeze requires exactly 9 paired tasks")
    _require(
        {task.get("oracle", {}).get("requiredAction") for task in canaries if isinstance(task.get("oracle"), dict)}
        == {"execute_read_query", "compare_groups", "retrieve_evidence"},
        failures,
        "$.tasks[canary].oracle.requiredAction",
        "canaries must cover Dynamic QueryPlan, Dynamic AnalysisPlan and Retrieval/Loop integration",
    )

    task_ids: list[str] = []
    plan_kinds: Counter[str] = Counter()
    required_actions: Counter[str] = Counter()
    materializer_count = 0
    for index, task in enumerate(tasks):
        path = f"$.tasks[{index}]"
        if not isinstance(task, dict):
            failures.append(_error(path, "must be an object"))
            continue
        task_id = task.get("taskId")
        task_ids.append(str(task_id))
        _require(isinstance(task_id, str) and TASK_ID_RE.fullmatch(task_id) is not None, failures, f"{path}.taskId", "must be a stable kebab-case id ending in three digits")
        _require(isinstance(task.get("kind"), str) and task["kind"], failures, f"{path}.kind", "must be non-empty")
        _require(isinstance(task.get("goalCode"), str) and re.fullmatch(r"[A-Z][A-Z0-9_]+", task["goalCode"] or "") is not None, failures, f"{path}.goalCode", "must be an uppercase goal code")
        question = task.get("question")
        _require(isinstance(question, str) and 10 <= len(question) <= 512, failures, f"{path}.question", "must be a bounded generic question")
        if isinstance(question, str):
            for pattern in FORBIDDEN_QUESTION_PATTERNS:
                _require(pattern.search(question) is None, failures, f"{path}.question", "contains a forbidden raw identifier or query fragment")

        scopes = task.get("requestedScopes")
        _require(isinstance(scopes, list) and 1 <= len(scopes) <= len(SCOPES), failures, f"{path}.requestedScopes", "must be a bounded list")
        if isinstance(scopes, list):
            _require(len(scopes) == len(set(scopes)), failures, f"{path}.requestedScopes", "contains duplicates")
            _require(set(scopes).issubset(SCOPES), failures, f"{path}.requestedScopes", "contains an unknown scope")

        allowed = task.get("allowedActions")
        _require(isinstance(allowed, list) and 1 <= len(allowed) <= len(ACTION_NAMES), failures, f"{path}.allowedActions", "must be a bounded list")
        if isinstance(allowed, list):
            _require(len(allowed) == len(set(allowed)), failures, f"{path}.allowedActions", "contains duplicates")
            _require(set(allowed).issubset(ACTION_NAMES), failures, f"{path}.allowedActions", "contains an action outside the closed Action union")

        max_actions = task.get("maxActions")
        _require(isinstance(max_actions, int) and not isinstance(max_actions, bool) and 1 <= max_actions <= 8, failures, f"{path}.maxActions", "must be between 1 and 8")

        oracle = task.get("oracle")
        _require(isinstance(oracle, dict), failures, f"{path}.oracle", "must be an object")
        if not isinstance(oracle, dict):
            continue
        allowed_first = oracle.get("allowedFirstActions")
        _require(isinstance(allowed_first, list) and allowed_first, failures, f"{path}.oracle.allowedFirstActions", "must be non-empty")
        if isinstance(allowed_first, list):
            _require(set(allowed_first).issubset(set(allowed or [])), failures, f"{path}.oracle.allowedFirstActions", "must be allowed actions")
        required = oracle.get("requiredAction")
        _require(required in set(allowed or []), failures, f"{path}.oracle.requiredAction", "must be an allowed action")
        materializer_required = oracle.get("materializerRequired")
        _require(_is_bool(materializer_required), failures, f"{path}.oracle.materializerRequired", "must be boolean")
        origin = oracle.get("materializerOrigin")
        _require(origin in MATERIALIZER_ORIGINS, failures, f"{path}.oracle.materializerOrigin", "must be model or runtime_owned")
        if materializer_required is True:
            materializer_count += 1
            _require(origin == "model", failures, f"{path}.oracle.materializerOrigin", "required materialization must be model-origin")
        if origin == "runtime_owned":
            _require(materializer_required is False, failures, f"{path}.oracle.materializerRequired", "runtime-owned plans must not call Materializer")

        plan_kind = oracle.get("planKind")
        plan_kinds[str(plan_kind)] += 1
        _require(plan_kind in PLAN_KINDS, failures, f"{path}.oracle.planKind", "is not a closed plan kind")
        raw_final = oracle.get("rawFinalMayDiffer", False)
        _require(_is_bool(raw_final), failures, f"{path}.oracle.rawFinalMayDiffer", "must be boolean")
        if plan_kind == "query_plan":
            _require(required in {"inspect_cohort", "execute_read_query"}, failures, f"{path}.oracle.requiredAction", "query plan must bind to a read action")
            query = oracle.get("queryPlan")
            _require(isinstance(query, dict), failures, f"{path}.oracle.queryPlan", "is required")
            if isinstance(query, dict):
                _require(query.get("schemaVersion") == "query-plan-v1", failures, f"{path}.oracle.queryPlan.schemaVersion", "must be query-plan-v1")
                _require(isinstance(query.get("relationPathMax"), int) and 0 <= query["relationPathMax"] <= 2, failures, f"{path}.oracle.queryPlan.relationPathMax", "must be 0..2")
                _require(query.get("requiresSemanticFieldIds") is True, failures, f"{path}.oracle.queryPlan.requiresSemanticFieldIds", "must be true")
                _require(query.get("requiresPhysicalIdentifiers") is False, failures, f"{path}.oracle.queryPlan.requiresPhysicalIdentifiers", "must be false")
                _require(isinstance(query.get("limitMax"), int) and 1 <= query["limitMax"] <= 1000, failures, f"{path}.oracle.queryPlan.limitMax", "must be 1..1000")
        elif plan_kind == "analysis_plan":
            _require(required in ANALYSIS_ACTIONS, failures, f"{path}.oracle.requiredAction", "analysis plan must bind to an analysis action")
            analysis = oracle.get("analysisPlan")
            _require(isinstance(analysis, dict), failures, f"{path}.oracle.analysisPlan", "is required")
            if isinstance(analysis, dict):
                _require(analysis.get("schemaVersion") == "analysis-plan-v1", failures, f"{path}.oracle.analysisPlan.schemaVersion", "must be analysis-plan-v1")
                _require(analysis.get("analysisType") in ANALYSIS_TYPES, failures, f"{path}.oracle.analysisPlan.analysisType", "is not closed")
                _require(analysis.get("requiresClosedAnalysisType") is True, failures, f"{path}.oracle.analysisPlan.requiresClosedAnalysisType", "must be true")
                _require(analysis.get("requiresSourceObservationIds") is True, failures, f"{path}.oracle.analysisPlan.requiresSourceObservationIds", "must be true")
                _require(analysis.get("requiresSemanticFieldIds") is True, failures, f"{path}.oracle.analysisPlan.requiresSemanticFieldIds", "must be true")
                _require(analysis.get("requiresRawPython") is False, failures, f"{path}.oracle.analysisPlan.requiresRawPython", "must be false")
        elif plan_kind == "retrieval_plan":
            _require(required == "retrieve_evidence", failures, f"{path}.oracle.requiredAction", "retrieval plan must bind to retrieve_evidence")
            retrieval = oracle.get("retrievalPlan")
            _require(isinstance(retrieval, dict), failures, f"{path}.oracle.retrievalPlan", "is required")
            if isinstance(retrieval, dict):
                _require(retrieval.get("schemaVersion") == "retrieval-plan-v1", failures, f"{path}.oracle.retrievalPlan.schemaVersion", "must be retrieval-plan-v1")
                _require(retrieval.get("constructedBy") == "runtime", failures, f"{path}.oracle.retrievalPlan.constructedBy", "must be runtime")
                _require(set(retrieval.get("retrievalMode", [])) == {"vector", "graph", "hybrid"}, failures, f"{path}.oracle.retrievalPlan.retrievalMode", "must enumerate closed modes")
                _require(retrieval.get("topKMax") == 20, failures, f"{path}.oracle.retrievalPlan.topKMax", "must be 20")
                _require(retrieval.get("maxHopsMax") == 3, failures, f"{path}.oracle.retrievalPlan.maxHopsMax", "must be 3")
        elif plan_kind == "finish_boundary":
            _require(required == "finish", failures, f"{path}.oracle.requiredAction", "finish boundary must bind to finish")
            _require(materializer_required is False and origin == "runtime_owned", failures, f"{path}.oracle", "finish is Runtime-owned")
            reasons = oracle.get("finishReasonCodes")
            _require(isinstance(reasons, list) and reasons and all(isinstance(item, str) for item in reasons), failures, f"{path}.oracle.finishReasonCodes", "must be a non-empty code list")

    _require(len(task_ids) == len(set(task_ids)), failures, "$.tasks.taskId", "contains duplicate task IDs")
    _require(set(task_ids) == {task.get("taskId") for task in tasks if isinstance(task, dict)}, failures, "$.tasks.taskId", "contains a non-string task ID")
    _require("query_plan" in plan_kinds and "analysis_plan" in plan_kinds and "retrieval_plan" in plan_kinds and "finish_boundary" in plan_kinds, failures, "$.tasks.oracle.planKind", "must cover all four plan kinds")
    _require(materializer_count == 9, failures, "$.tasks.oracle.materializerRequired", "freeze expects 9 Materializer-backed tasks")
    return failures


def validate_materializer_config(payload: Any, planner_source: Path) -> tuple[list[str], str]:
    failures: list[str] = []
    _require(isinstance(payload, dict), failures, "$", "must be an object")
    if not isinstance(payload, dict):
        return failures, ""
    _require(payload.get("schemaVersion") == "p2j4-dynamic-e2e-materializer-config-v1", failures, "$.schemaVersion", "must be v1")
    _require(payload.get("status") == "FROZEN", failures, "$.status", "must be FROZEN")
    _require(payload.get("materializerProvider") == "DeepSeek", failures, "$.materializerProvider", "must be DeepSeek")
    _require(payload.get("materializerModel") == "deepseek-v4-flash", failures, "$.materializerModel", "must be deepseek-v4-flash")
    _require(payload.get("temperature") == 0, failures, "$.temperature", "must be 0")
    _require(payload.get("maxRetries") == 3, failures, "$.maxRetries", "must be 3")
    _require(payload.get("cacheScope") == "paired_experiment_only", failures, "$.cacheScope", "must be paired_experiment_only")
    _require(payload.get("reuseAcrossDifferentActions") is False, failures, "$.reuseAcrossDifferentActions", "must be false")
    _require(payload.get("rawPlansPersisted") is False, failures, "$.rawPlansPersisted", "must be false")
    _require(payload.get("secretValuesIncluded") is False, failures, "$.secretValuesIncluded", "must be false")
    expected_prompt_hash = static_materializer_prompt_hash(planner_source)
    _require(payload.get("systemPromptHash") == expected_prompt_hash, failures, "$.systemPromptHash", "does not match current planner prompt literals")
    _require(payload.get("systemPromptFunctions") == ["plan_action", "generate_typed_analysis"], failures, "$.systemPromptFunctions", "must cover both dynamic Materializer prompts")
    return failures, expected_prompt_hash


def build_manifest(task_payload: dict[str, Any], config_payload: dict[str, Any], task_path: Path, config_path: Path, prompt_hash: str) -> dict[str, Any]:
    tasks = task_payload["tasks"]
    oracle_summary = {
        "planKinds": dict(sorted(Counter(task["oracle"]["planKind"] for task in tasks).items())),
        "requiredActions": dict(sorted(Counter(task["oracle"]["requiredAction"] for task in tasks).items())),
        "materializerRequiredCount": sum(1 for task in tasks if task["oracle"]["materializerRequired"]),
        "runtimeOwnedCount": sum(1 for task in tasks if task["oracle"]["materializerOrigin"] == "runtime_owned"),
    }
    return {
        "manifestVersion": "p2j4-dynamic-e2e-contract-manifest-v1",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "taskSetStatus": task_payload["status"],
        "taskSetSchemaVersion": task_payload["schemaVersion"],
        "taskSetPath": task_path.as_posix(),
        "taskSetSha256": sha256_bytes(canonical_json_bytes(task_payload)),
        "taskCount": len(tasks),
        "canaryCount": task_payload["canaryCount"],
        "pairedCount": task_payload["pairedCount"],
        "oracleSummary": oracle_summary,
        "materializerConfigPath": config_path.as_posix(),
        "materializerConfigSha256": sha256_bytes(canonical_json_bytes(config_payload)),
        "materializerSystemPromptHash": prompt_hash,
        "notForTraining": True,
        "secretValuesIncluded": False,
        "rawTaskTextIncluded": False,
        "realRunStarted": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-set", type=Path, required=True)
    parser.add_argument("--materializer-config", type=Path, required=True)
    parser.add_argument("--planner-source", type=Path, default=Path("mico_agent_runtime/ports/research_planner.py"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    task_payload = json.loads(args.task_set.read_text(encoding="utf-8"))
    config_payload = json.loads(args.materializer_config.read_text(encoding="utf-8"))
    task_failures = validate_task_set(task_payload)
    config_failures, prompt_hash = validate_materializer_config(config_payload, args.planner_source)
    failures = [*task_failures, *config_failures]
    if failures:
        for failure in failures:
            print(failure)
        return 2
    manifest = build_manifest(task_payload, config_payload, args.task_set, args.materializer_config, prompt_hash)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "taskSetSha256": manifest["taskSetSha256"], "materializerSystemPromptHash": prompt_hash}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
