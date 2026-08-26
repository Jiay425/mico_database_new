from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import Field

from mico_agent_runtime.contracts.base import ClosedModel, NonEmptyText


GuardrailVerdict = Literal["ALLOW", "REJECT"]
GuardrailLayer = Literal["input", "tool", "output"]
GuardrailCode = Literal[
    "POLICY_ALLOWED",
    "RESEARCH_SCOPE_NOT_ALLOWED",
    "WORKFLOW_NOT_ALLOWED",
    "WORKFLOW_SCOPE_MISSING",
    "OUTPUT_SAFETY_BLOCKED",
]


class GuardrailDecision(ClosedModel):
    """Closed decision record; no source text or payload is retained."""

    verdict: GuardrailVerdict
    layer: GuardrailLayer
    code: GuardrailCode
    message: NonEmptyText
    matchedRule: str | None = Field(default=None, max_length=80)


# These are policy markers, not data-value filters. Domain values are never
# passed through this list; Java validates any dynamic SQL proposal separately.
_DENY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("clinical_request", re.compile(
        r"诊断|治疗|处方|用药|药物|开药|手术|治愈|临床(?:风险|判断|建议|诊断|治疗|决策)|医生建议|是否患病|判断.*患病|病情|medical advice|diagnos(?:e|is|tic)|prescri(?:be|ption)|treatment"
    , re.IGNORECASE)),
    ("causal_claim", re.compile(
        r"因果|导致|证明.*原因|病因|cause(?:s|d)?|causal|mechanism", re.IGNORECASE)),
    ("mutation_sql", re.compile(
        r"\b(?:insert|update|delete|drop|alter|truncate|grant|revoke)\b", re.IGNORECASE)),
    ("free_sql_request", re.compile(
        r"自由\s*sql|free\s*sql|任意\s*(?:sql|where|表|字段)|任意.{0,16}(?:表|字段)|raw\s*sql", re.IGNORECASE)),
    ("credential_or_prompt_attack", re.compile(
        r"(?:忽略|绕过|ignore).{0,24}(?:规则|指令|政策|rules|policy|instruction)|(?:token|password|密钥|密码|秘密|api[-_]?key|authorization|header).{0,24}(?:显示|返回|泄露|print|show|reveal)|(?:显示|返回|泄露|print|show|reveal).{0,24}(?:token|password|密钥|密码|秘密|api[-_]?key|authorization|header)",
        re.IGNORECASE)),
)

_PUBLIC_FORBIDDEN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)sourceSampleId\s*=|internalRecordId\s*=|cohortCondition"),
    re.compile(r"(?i)\b(?:patientId|patient_id|subjectId|subject_id)\b"),
    re.compile(r"(?i)\b(?:SRR|ERR|DRR)\d+\b|\bMV_[A-Za-z0-9_-]+\b"),
    re.compile(r"(?i)\b(?:authorization|bearer|password|secret|api[-_]?key)\b"),
    re.compile(r"(?i)https?://|file://|[A-Za-z]:\\|/etc/|/var/"),
)


def _is_negated_boundary(question: str, match_start: int) -> bool:
    """Return whether a safety term is being explicitly *disclaimed*.

    Scientific requests often state a boundary such as "不作因果解释" or
    "不输出诊断结论". Those are not attempts to obtain a causal or clinical
    conclusion and must not be rejected merely because the guarded word is
    present. The check is deliberately local: a later positive request (for
    example, "不要诊断，但判断是否患病") still matches independently and is
    rejected by the normal deny rules.
    """

    prefix = question[max(0, match_start - 24):match_start]
    if re.search(r"(?:non[-\s]?|without\s+)$", prefix, re.IGNORECASE):
        return True
    return bool(re.search(
        r"(?:不|不要|不能|勿|避免|禁止|不得|仅|只).{0,16}"
        r"(?:输出|作出?|做|进行|解释|推断|给出|宣布)?$",
        prefix,
        re.IGNORECASE,
    ))


def evaluate_input(question: str) -> GuardrailDecision:
    """Apply the input guardrail without echoing the rejected question."""

    for rule, pattern in _DENY_PATTERNS:
        matched = next((
            match for match in pattern.finditer(question)
            if not (
                rule in {"clinical_request", "causal_claim"}
                and _is_negated_boundary(question, match.start())
            )
        ), None)
        if matched:
            return GuardrailDecision(
                verdict="REJECT",
                layer="input",
                code="RESEARCH_SCOPE_NOT_ALLOWED",
                message="The request is outside the non-diagnostic research scope",
                matchedRule=rule,
            )
    return GuardrailDecision(
        verdict="ALLOW",
        layer="input",
        code="POLICY_ALLOWED",
        message="The request is within the research-data scope",
    )


def evaluate_workflow(
    workflow: str,
    allowed_workflows: list[str],
    requested_scopes: list[str],
    required_scopes: set[str],
) -> GuardrailDecision:
    """Re-check planner output as a tool/workflow authorization boundary."""

    if workflow not in allowed_workflows:
        return GuardrailDecision(
            verdict="REJECT",
            layer="tool",
            code="WORKFLOW_NOT_ALLOWED",
            message="The selected workflow is not approved for this run",
        )
    if not required_scopes.issubset(set(requested_scopes)):
        return GuardrailDecision(
            verdict="REJECT",
            layer="tool",
            code="WORKFLOW_SCOPE_MISSING",
            message="The selected workflow requires an unavailable scope",
        )
    return GuardrailDecision(
        verdict="ALLOW",
        layer="tool",
        code="POLICY_ALLOWED",
        message="The selected workflow is authorized",
    )


def evaluate_output(value: object) -> GuardrailDecision:
    """Reject public projections that contain known sensitive-value shapes."""

    try:
        serialized = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return GuardrailDecision(
            verdict="REJECT",
            layer="output",
            code="OUTPUT_SAFETY_BLOCKED",
            message="The result cannot be safely serialized",
        )
    for pattern in _PUBLIC_FORBIDDEN_PATTERNS:
        if pattern.search(serialized):
            return GuardrailDecision(
                verdict="REJECT",
                layer="output",
                code="OUTPUT_SAFETY_BLOCKED",
                message="The result contains a prohibited sensitive projection",
            )
    return GuardrailDecision(
        verdict="ALLOW",
        layer="output",
        code="POLICY_ALLOWED",
        message="The result satisfies the public projection boundary",
    )
