"""Closed, no-value controlled-scenario contracts for P2-J4.1 result oracles.

This module is deliberately an offline contract layer.  It never selects a
sample, builds a query, or opens a Java/database/model connection.  A future
provider converts a bounded Java result into ``ResultOracleObservation`` in
memory; the only durable representation is closed evidence and limitation
codes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal, Sequence

from pydantic import Field, StringConstraints, ValidationError, field_validator

from mico_agent_runtime.contracts.base import ClosedModel
from mico_agent_runtime.contracts.research import ScientificActionName, StopReasonCode

from evals.p2j4_result_oracles import (
    BadCaseResultOracle,
    OracleCode,
    OracleSource,
    ScenarioRef,
)


SCENARIO_SET = Path(__file__).with_name("p2j4-controlled-scenarios-v1.json")
SCENARIO_SCHEMA_VERSION = "p2j4-controlled-scenarios-v1"
_SENSITIVE_MARKERS = (
    "sourcesampleid",
    "internalrecordid",
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

ScenarioKind = Literal[
    "bounded_aggregate",
    "bounded_grouped_aggregate",
    "within_disease_project_comparison",
    "cohort_reconstruction",
    "single_project_downgrade",
]
ScenarioState = Literal["CONTRACT_READY"]
VerificationStatus = Literal[
    "PASS", "FAIL", "SCENARIO_REQUIRED", "SCENARIO_UNAVAILABLE", "SCENARIO_SUPPLY_FAILED"
]


class ControlledScenarioSeed(ClosedModel):
    """Requirements for one future in-memory scenario provider.

    ``CONTRACT_READY`` means the contract is testable offline; it never means
    a real database scenario has been provisioned or passed.
    """

    caseId: Annotated[str, StringConstraints(pattern=r"^p2j4-[a-z0-9-]+$", max_length=128)]
    scenarioRef: ScenarioRef
    scenarioKind: ScenarioKind
    providerCode: OracleCode
    scenarioState: ScenarioState
    resultVerifierCode: OracleCode
    expectedAssertionCodes: list[OracleCode] = Field(min_length=1, max_length=12)
    expectedLimitationCodes: list[OracleCode] = Field(default_factory=list, max_length=8)

    @field_validator("expectedAssertionCodes", "expectedLimitationCodes")
    @classmethod
    def reject_duplicates(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("duplicate scenario code is not allowed")
        return value


class ResultOracleObservation(ClosedModel):
    """No-value, in-memory result summary produced after a bounded replay."""

    caseId: Annotated[str, StringConstraints(pattern=r"^p2j4-[a-z0-9-]+$", max_length=128)]
    actionPath: list[ScientificActionName] = Field(default_factory=list, max_length=8)
    sourceRoutes: list[OracleSource] = Field(default_factory=list, max_length=3)
    assertionCodes: list[OracleCode] = Field(default_factory=list, max_length=12)
    limitationCodes: list[OracleCode] = Field(default_factory=list, max_length=8)
    stopReasonCode: StopReasonCode | None = None
    scenarioRef: ScenarioRef | None = None

    @field_validator("actionPath", "sourceRoutes", "assertionCodes", "limitationCodes")
    @classmethod
    def reject_duplicates(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("duplicate observation value is not allowed")
        return value


class ResultOracleVerification(ClosedModel):
    """Closed result of purely in-memory verification, safe to persist."""

    caseId: Annotated[str, StringConstraints(pattern=r"^p2j4-[a-z0-9-]+$", max_length=128)]
    resultVerifierCode: OracleCode
    status: VerificationStatus
    failureCodes: list[OracleCode] = Field(default_factory=list, max_length=8)

    @field_validator("failureCodes")
    @classmethod
    def reject_duplicate_failures(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("duplicate verifier failure is not allowed")
        return value


def _read_payload(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("CONTROLLED_SCENARIO_SET_UNREADABLE") from exc
    if not isinstance(payload, dict):
        raise ValueError("CONTROLLED_SCENARIO_SET_SHAPE_INVALID")
    if payload.get("schemaVersion") != SCENARIO_SCHEMA_VERSION:
        raise ValueError("CONTROLLED_SCENARIO_SET_VERSION_INVALID")
    return payload


def validate_controlled_scenario_set(
    oracles: Sequence[BadCaseResultOracle], path: Path = SCENARIO_SET
) -> dict:
    """Check that five scenario contracts exactly mirror scenario-required oracles."""

    payload = _read_payload(path)
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 5:
        raise ValueError("CONTROLLED_SCENARIO_SET_COUNT_INVALID")
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True).lower()
    if any(marker in serialized for marker in _SENSITIVE_MARKERS):
        raise ValueError("CONTROLLED_SCENARIO_SET_SENSITIVE_VALUE")
    try:
        seeds = [ControlledScenarioSeed.model_validate(record) for record in records]
    except ValidationError as exc:
        raise ValueError("CONTROLLED_SCENARIO_SET_RECORD_INVALID") from exc

    expected = {
        oracle.caseId: oracle
        for oracle in oracles
        if oracle.scenarioProvisioning == "CONTROLLED_SCENARIO_REQUIRED"
    }
    by_case_id = {seed.caseId: seed for seed in seeds}
    if len(by_case_id) != len(seeds):
        raise ValueError("CONTROLLED_SCENARIO_SET_CASE_ID_DUPLICATE")
    if set(by_case_id) != set(expected):
        raise ValueError("CONTROLLED_SCENARIO_SET_ORACLE_MISMATCH")
    for case_id, oracle in expected.items():
        seed = by_case_id[case_id]
        if (
            seed.scenarioRef != oracle.scenarioRef
            or seed.resultVerifierCode != oracle.resultVerifierCode
            or set(seed.expectedAssertionCodes) != set(oracle.assertionCodes)
            or set(seed.expectedLimitationCodes) != set(oracle.limitationCodes)
        ):
            raise ValueError("CONTROLLED_SCENARIO_SET_CONTRACT_MISMATCH")
    return {
        "schemaVersion": SCENARIO_SCHEMA_VERSION,
        "scenarioCount": len(seeds),
        "seeds": seeds,
    }


def _is_ordered_subsequence(
    required: Sequence[ScientificActionName], actual: Sequence[ScientificActionName]
) -> bool:
    position = 0
    for action in actual:
        if position < len(required) and action == required[position]:
            position += 1
    return position == len(required)


def verify_result_oracle(
    oracle: BadCaseResultOracle, observation: ResultOracleObservation
) -> ResultOracleVerification:
    """Verify closed result codes without retaining or inspecting raw values."""

    if observation.caseId != oracle.caseId:
        raise ValueError("RESULT_ORACLE_CASE_MISMATCH")
    if (
        oracle.scenarioProvisioning == "CONTROLLED_SCENARIO_REQUIRED"
        and observation.scenarioRef is None
    ):
        return ResultOracleVerification(
            caseId=oracle.caseId,
            resultVerifierCode=oracle.resultVerifierCode,
            status="SCENARIO_REQUIRED",
            failureCodes=["CONTROLLED_SCENARIO_NOT_PROVISIONED"],
        )

    failures: list[str] = []
    if observation.scenarioRef is not None and observation.scenarioRef != oracle.scenarioRef:
        failures.append("SCENARIO_REF_MISMATCH")
    if not _is_ordered_subsequence(oracle.requiredActionSubsequence, observation.actionPath):
        failures.append("REQUIRED_ACTION_SUBSEQUENCE_MISSING")
    if not set(oracle.requiredSources).issubset(observation.sourceRoutes):
        failures.append("REQUIRED_SOURCE_MISSING")
    if not set(oracle.assertionCodes).issubset(observation.assertionCodes):
        failures.append("RESULT_ASSERTION_MISSING")
    if not set(oracle.limitationCodes).issubset(observation.limitationCodes):
        failures.append("RESULT_LIMITATION_MISSING")
    if observation.stopReasonCode != oracle.expectedStopReason:
        failures.append("RESULT_STOP_REASON_MISMATCH")
    return ResultOracleVerification(
        caseId=oracle.caseId,
        resultVerifierCode=oracle.resultVerifierCode,
        status="FAIL" if failures else "PASS",
        failureCodes=failures,
    )
