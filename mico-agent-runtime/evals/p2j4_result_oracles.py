"""Closed, value-free result-oracle registry for the P2-J4.1 Bad Cases.

The normal Trace intentionally excludes SQL, payloads and numerical results.
These oracles therefore describe only scenario provisioning and verifier
contracts.  A future replay verifier may consume the live bounded result in
memory, but this registry never records a query, business value, identifier or
credential.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal, Sequence

from pydantic import Field, StringConstraints, ValidationError, field_validator

from mico_agent_runtime.contracts.base import ClosedModel
from mico_agent_runtime.contracts.research import ScientificActionName, StopReasonCode


ORACLE_SET = Path(__file__).with_name("p2j4-result-oracles-v1.json")
ORACLE_SCHEMA_VERSION = "p2j4-result-oracles-v1"
_SENSITIVE_MARKERS = (
    "sourceSampleId",
    "internalRecordId",
    "mysql://",
    "postgresql://",
    "jdbc:",
    "bearer ",
    "api_key",
    "api-key",
    "token=",
    "password=",
    "select ",
    " from ",
)

OracleCode = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Z][A-Z0-9_]{2,95}$", max_length=96),
]
ScenarioRef = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z][a-z0-9_]{2,95}$", max_length=96),
]
OracleSource = Literal["java", "vector", "graph"]
ScenarioProvisioning = Literal[
    "TRACE_ORACLE_READY",
    "CONTROLLED_SCENARIO_REQUIRED",
]


class BadCaseResultOracle(ClosedModel):
    caseId: Annotated[str, StringConstraints(pattern=r"^p2j4-[a-z0-9-]+$", max_length=128)]
    badCaseClass: OracleCode
    scenarioProvisioning: ScenarioProvisioning
    scenarioRef: ScenarioRef
    resultVerifierCode: OracleCode
    requiredActionSubsequence: list[ScientificActionName] = Field(min_length=1, max_length=8)
    requiredSources: list[OracleSource] = Field(default_factory=list, max_length=3)
    assertionCodes: list[OracleCode] = Field(min_length=1, max_length=12)
    limitationCodes: list[OracleCode] = Field(default_factory=list, max_length=8)
    expectedStopReason: StopReasonCode
    requiresRealReplay: bool = True

    @field_validator(
        "requiredActionSubsequence", "requiredSources", "assertionCodes", "limitationCodes"
    )
    @classmethod
    def reject_duplicates(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("duplicate oracle values are not allowed")
        return value


def _read_payload(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("RESULT_ORACLE_SET_UNREADABLE") from exc
    if not isinstance(payload, dict):
        raise ValueError("RESULT_ORACLE_SET_SHAPE_INVALID")
    if payload.get("schemaVersion") != ORACLE_SCHEMA_VERSION:
        raise ValueError("RESULT_ORACLE_SET_VERSION_INVALID")
    return payload


def _is_ordered_subsequence(required: Sequence[str], actual: Sequence[str]) -> bool:
    position = 0
    for action in actual:
        if position < len(required) and action == required[position]:
            position += 1
    return position == len(required)


def validate_result_oracle_set(
    tasks: Sequence[object], path: Path = ORACLE_SET
) -> dict:
    """Validate the registry against closed Eval Actions without external work."""
    payload = _read_payload(path)
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 12:
        raise ValueError("RESULT_ORACLE_SET_COUNT_INVALID")
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True).lower()
    if any(marker.lower() in serialized for marker in _SENSITIVE_MARKERS):
        raise ValueError("RESULT_ORACLE_SET_SENSITIVE_VALUE")
    try:
        oracles = [BadCaseResultOracle.model_validate(record) for record in records]
    except ValidationError as exc:
        raise ValueError("RESULT_ORACLE_SET_RECORD_INVALID") from exc
    case_ids = [oracle.caseId for oracle in oracles]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("RESULT_ORACLE_SET_CASE_ID_DUPLICATE")
    task_by_case_id = {getattr(task, "caseId", None): task for task in tasks}
    if set(case_ids) - set(task_by_case_id):
        raise ValueError("RESULT_ORACLE_TASK_NOT_FOUND")
    for oracle in oracles:
        task = task_by_case_id[oracle.caseId]
        allowed_actions = set(getattr(task, "allowedActions", []))
        if not set(oracle.requiredActionSubsequence).issubset(allowed_actions):
            raise ValueError("RESULT_ORACLE_ACTION_NOT_ALLOWED")
        allowed_paths = getattr(task, "allowedActionPaths", [])
        if allowed_paths and not any(
            _is_ordered_subsequence(oracle.requiredActionSubsequence, path)
            for path in allowed_paths
        ):
            raise ValueError("RESULT_ORACLE_PATH_NOT_ALLOWED")
        if oracle.expectedStopReason != getattr(task, "expectedStopReason", None):
            raise ValueError("RESULT_ORACLE_STOP_REASON_MISMATCH")
    return {
        "schemaVersion": ORACLE_SCHEMA_VERSION,
        "oracleCount": len(oracles),
        "scenarioRequiredCount": sum(
            oracle.scenarioProvisioning == "CONTROLLED_SCENARIO_REQUIRED"
            for oracle in oracles
        ),
        "oracles": oracles,
    }
