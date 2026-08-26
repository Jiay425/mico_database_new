"""Optional Decision-SFT policy adapter for the Scientific Runtime.

The adapter is deliberately narrower than the existing research planner:
the SFT model selects an allow-listed capability from a de-identified policy
state.  Runtime code then reifies that capability into a closed
``ScientificAction`` using only approved observations and the Java semantic
catalog.  The model never supplies SQL, identifiers, permissions, or raw
research values.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Protocol
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

from mico_agent_runtime.contracts.research import (
    AdjustConfoundersAction,
    AnalyzeProjectionAction,
    AnalyzeProjectionArguments,
    FinishAction,
    FinishArguments,
    ProjectionAnalysisArguments,
    ResearchIntent,
    RetrieveEvidenceAction,
    RetrieveEvidenceArguments,
    ScientificAction,
    ScientificActionName,
    ScientificPlannerContext,
    StopReasonCode,
)
from mico_agent_runtime.ports.scientific_planner import (
    ScientificPlannerPort,
    ScientificPlannerResult,
    _deterministic_action,
    _evidence_action,
    _finish_action,
    _final_projection_action,
    _projection_action,
)


SFT_POLICY_FALLBACK_CODE = "SFT_POLICY_SAFE_FALLBACK"
SFT_POLICY_DISABLED_CODE = "SFT_POLICY_DISABLED"
SFT_POLICY_MAX_RETRIES = 3


class DecisionPolicyOutput(BaseModel):
    """Closed output contract emitted by the Decision-SFT model."""

    model_config = ConfigDict(extra="forbid")

    selected_action: ScientificActionName
    decision_reason: str = Field(min_length=1, max_length=512)
    alternative_actions: list[ScientificActionName] = Field(default_factory=list, max_length=9)
    stop_reason: StopReasonCode | None = None

    @model_validator(mode="after")
    def validate_stop_shape(self) -> "DecisionPolicyOutput":
        if self.selected_action == "finish" and self.stop_reason is None:
            raise ValueError("finish decisions require stop_reason")
        if self.selected_action != "finish" and self.stop_reason is not None:
            raise ValueError("stop_reason is only valid for finish decisions")
        if len(self.alternative_actions) != len(set(self.alternative_actions)):
            raise ValueError("alternative_actions must be unique")
        if self.selected_action in self.alternative_actions:
            raise ValueError("selected_action cannot be an alternative action")
        return self


class DecisionPolicyFallback(Protocol):
    def plan_action(self, context: ScientificPlannerContext) -> ScientificPlannerResult:
        ...


def _parse_json(content: object) -> object:
    if isinstance(content, Mapping):
        return content
    if not isinstance(content, str):
        raise ValueError("decision policy content is not text")
    text = re.sub(r"(?is)<think>.*?</think>", "", content).strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) < 3:
            raise ValueError("decision policy JSON fence is empty")
        text = "\n".join(lines[1:-1]).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            raise
        parsed, end = json.JSONDecoder().raw_decode(text[start:])
        if text[start + end:].strip():
            raise ValueError("trailing non-json decision policy content")
        return parsed


def _feedback(error: Exception) -> str:
    errors_method = getattr(error, "errors", None)
    if callable(errors_method):
        try:
            entries = errors_method(include_url=False)
        except TypeError:
            entries = errors_method()
        details = []
        for entry in entries[:8]:
            location = ".".join(str(item) for item in entry.get("loc", ())) or "root"
            details.append(f"{location}: {entry.get('msg', 'invalid value')}")
        if details:
            return "; ".join(details)[:1000]
    return f"{type(error).__name__}: {(str(error).strip() or 'closed contract rejected')[:700]}"


def _task_kind(intent: ResearchIntent) -> str:
    return {
        "data_fact": "data_fact",
        "focused_comparison": "focused_analysis",
        "literature_or_relationship": "evidence_review",
        "scientific_exploration": "open_exploration",
    }[intent]


def _goal_code(question_summary: str) -> str:
    """Map the redacted question to a stable, non-biological goal code."""

    question = question_summary.lower()

    def has(*terms: str) -> bool:
        return any(term in question for term in terms)

    if has("冲突", "conflict", "相反", "不一致", "矛盾", "否定"):
        return "evidence_conflict_resolution"
    if has("跨 project", "跨项目", "cross-project", "cross project"):
        return "cross_project_stability"
    if has("混杂", "confounder", "控制年龄", "控制性别", "batch effect", "不平衡"):
        return "confounder_control"
    if has("跨疾病", "cross-disease", "disease-specific", "特异性"):
        return "disease_specificity"
    if has("证据", "文献", "evidence", "literature", "机制", "图谱"):
        return "evidence_grounding"
    if has("样本", "记录数", "多少", "count", "覆盖"):
        return "cohort_fact"
    if has("差异", "比较", "compare", "difference"):
        return "group_comparison"
    return "scientific_exploration"


def _observation_flags(context: ScientificPlannerContext, goal_code: str) -> list[str]:
    observations = context.observations
    history_actions = [item.actionName for item in observations]
    validated_actions = {
        item.actionName for item in observations if item.status in {"VALIDATED", "PARTIAL"}
    }
    approved_actions = set(context.approvedActions)
    flags: list[str] = []
    if not observations:
        flags.extend(("NO_OBSERVATION", "METADATA_FIRST"))
    else:
        if any(item.status in {"VALIDATED", "PARTIAL"} for item in observations):
            flags.append("OBSERVATION_VALIDATED")
        if any(item.status not in {"VALIDATED", "PARTIAL"} for item in observations):
            flags.append("OBSERVATION_REQUIRES_REVIEW")
        if any(item.source == "java_controlled_read" for item in observations):
            flags.extend(("JAVA_OBSERVED", "ANALYSIS_AVAILABLE"))
        if any(item.source == "knowledge_hybrid" for item in observations):
            flags.append("EVIDENCE_RETRIEVED")
        if any(item.qualityCodes for item in observations):
            flags.append("EVIDENCE_INCOMPLETE")
    if goal_code == "evidence_conflict_resolution":
        flags.append("EVIDENCE_CONFLICT")
    if any(item.actionName == "cross_project_validate" for item in observations):
        flags.append("PROJECT_VALIDATED")
    if any(item.actionName == "adjust_confounders" for item in observations):
        flags.append("CONFOUNDER_ADJUSTED")
    if "analyze_projection" in validated_actions:
        flags.append("ANALYSIS_COMPLETE")
    if any(
        item.source == "knowledge_hybrid"
        and item.status in {"VALIDATED", "PARTIAL"}
        and not item.qualityCodes
        for item in observations
    ):
        flags.append("EVIDENCE_GROUNDED")

    # Keep the Runtime state aligned with the reviewed Decision dataset.  A
    # generic VALIDATED_OBSERVATION flag is not enough to distinguish “the
    # metadata read succeeded” from “the requested projection is complete”.
    # These are closed, de-identified obligations derived only from the goal,
    # action history, and the already approved action catalog.
    if observations and goal_code in {"cohort_fact", "species_coverage", "project_coverage"}:
        if "execute_read_query" in approved_actions and "execute_read_query" not in history_actions:
            flags.append("BOUNDED_READ_REQUIRED")
    if "compare_groups" in history_actions and "analyze_projection" not in history_actions:
        if "analyze_projection" in approved_actions:
            flags.append("ANALYSIS_REQUIRED")
    if goal_code == "cross_project_stability":
        if "compare_groups" in history_actions and "cross_project_validate" not in history_actions:
            if "cross_project_validate" in approved_actions:
                flags.append("CROSS_PROJECT_REQUIRED")
        elif "cross_project_validate" in history_actions and "retrieve_evidence" not in history_actions:
            if "retrieve_evidence" in approved_actions:
                flags.append("EVIDENCE_REQUIRED")
    if goal_code == "confounder_control" and context.intent == "scientific_exploration":
        # In an open exploration, confounder adjustment is not the terminal
        # scientific check: the adjusted pattern still needs a stability
        # check across projects, followed by bounded evidence before report.
        if "adjust_confounders" in history_actions and "cross_project_validate" not in history_actions:
            if "cross_project_validate" in approved_actions:
                flags.append("CROSS_PROJECT_REQUIRED")
        elif "cross_project_validate" in history_actions and "retrieve_evidence" not in history_actions:
            if "retrieve_evidence" in approved_actions:
                flags.append("EVIDENCE_REQUIRED")
    if goal_code == "disease_specificity":
        if "analyze_projection" in history_actions and "cross_disease_validate" not in history_actions:
            if "cross_disease_validate" in approved_actions:
                flags.append("DISEASE_VALIDATION_REQUIRED")
        elif "cross_disease_validate" in history_actions and "retrieve_evidence" not in history_actions:
            if "retrieve_evidence" in approved_actions:
                flags.append("EVIDENCE_REQUIRED")
    if goal_code == "evidence_grounding":
        if "analyze_projection" in history_actions and "retrieve_evidence" not in history_actions:
            if "retrieve_evidence" in approved_actions:
                flags.append("EVIDENCE_REQUIRED")
    return list(dict.fromkeys(flags))


def build_decision_policy_state(context: ScientificPlannerContext) -> dict[str, Any]:
    """Build the only state that may cross the SFT policy HTTP boundary."""

    goal_code = _goal_code(context.questionSummary)
    flags = _observation_flags(context, goal_code)
    history_actions = [item.actionName for item in context.observations]
    phase = "NO_OBSERVATION" if not context.observations else (
        "EVIDENCE_RETRIEVED" if "EVIDENCE_RETRIEVED" in flags else "OBSERVATION_VALIDATED"
    )
    state_summary = (
        f"phase={phase}; goal={goal_code}; observation_state={flags[0]}; "
        f"evidence_bindings={len(context.observations)}; "
        f"prior_actions={','.join(history_actions) if history_actions else 'none'}; "
        f"flags={','.join(flags)}"
    )
    return {
        "task_kind": _task_kind(context.intent),
        "goal_code": goal_code,
        "observation_flags": flags,
        "history_actions": history_actions,
        "candidate_actions": list(context.approvedActions),
        "state_summary": state_summary[:512],
    }


def _verified_groupable_fields(context: ScientificPlannerContext) -> list[str]:
    if context.schemaCatalog is None:
        return []
    fields = [
        field.name
        for entity in context.schemaCatalog.entities
        for field in entity.fields
        if field.groupable and not field.sensitive and field.semanticStatus == "verified"
    ]
    return list(dict.fromkeys(fields))[:4]


def _reify_action(context: ScientificPlannerContext, decision: DecisionPolicyOutput) -> ScientificAction:
    """Turn a closed capability choice into a Runtime-owned action object."""

    if decision.selected_action not in context.approvedActions:
        raise ValueError("decision action is not approved")

    if decision.selected_action == "finish":
        base = _finish_action(context)
        return base.model_copy(update={
            "arguments": FinishArguments(
                actionName="finish",
                reasonCode=decision.stop_reason,
            )
        })
    if decision.selected_action == "retrieve_evidence":
        return _evidence_action(context)
    if decision.selected_action == "analyze_projection":
        ids = [
            item.observationId
            for item in context.observations
            if item.source in {"java_controlled_read", "python_bounded_analysis"}
        ]
        if not ids:
            raise ValueError("analyze_projection requires a tabular observation")
        return _final_projection_action(context, ids[-1])
    if decision.selected_action in {
        "compare_groups",
        "stratified_analysis",
        "adjust_confounders",
        "cross_project_validate",
        "cross_disease_validate",
    }:
        ids = [
            item.observationId
            for item in context.observations
            if item.source in {"java_controlled_read", "python_bounded_analysis"}
        ]
        minimum = 2 if decision.selected_action in {
            "cross_project_validate", "cross_disease_validate"
        } else 1
        if len(ids) < minimum:
            raise ValueError("selected analysis action lacks approved observations")
        fields = _verified_groupable_fields(context)
        if decision.selected_action == "stratified_analysis" and not fields:
            raise ValueError("stratified_analysis lacks verified dimensions")
        if decision.selected_action == "adjust_confounders" and not fields:
            raise ValueError("adjust_confounders lacks verified confounders")
        if decision.selected_action == "adjust_confounders":
            action_id = "action-" + sha256(
                (context.questionSummary + "|adjust_confounders|" + "|".join(ids[-1:])).encode("utf-8")
            ).hexdigest()[:32]
            return AdjustConfoundersAction(
                actionId=action_id,
                actionName="adjust_confounders",
                rationale="Control verified confounder dimensions before treating the observation as stable",
                arguments=ProjectionAnalysisArguments(
                    actionName="adjust_confounders",
                    observationIds=ids[-1:],
                    analysisGoal=context.questionSummary,
                    confounders=fields,
                ),
            )
        action = _projection_action(
            context,
            action_name=decision.selected_action,
            observation_ids=ids[-minimum:] if minimum == 2 else ids[-1:],
            dimensions=fields if decision.selected_action == "stratified_analysis" else None,
        )
        if action is not None:
            if decision.selected_action == "stratified_analysis":
                return action.model_copy(update={
                    "arguments": action.arguments.model_copy(update={"dimensions": fields})
                })
            return action
        raise ValueError("selected analysis action could not be reified")

    # Read actions remain entirely deterministic and catalog-owned.  This is
    # also the final safeguard against an SFT model injecting SQL.
    deterministic = _deterministic_action(context)
    if deterministic.actionName != decision.selected_action:
        raise ValueError("selected read action could not be safely reified")
    return deterministic


@dataclass
class HttpDecisionSftPlannerPort:
    """Call an OpenAI-compatible SFT policy endpoint with safe fallback."""

    base_url: str
    model: str
    token: str = ""
    fallback: DecisionPolicyFallback | None = None
    transport: httpx.BaseTransport | None = None

    def __post_init__(self) -> None:
        parsed = urlsplit(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username:
            raise ValueError("SFT policy URL is invalid")
        if parsed.query or parsed.fragment or not self.model.strip():
            raise ValueError("SFT policy configuration is invalid")
        base = self.base_url.rstrip("/")
        path = parsed.path.rstrip("/")
        if path.endswith("/v1") or path.endswith("/openai"):
            self._endpoint = base + "/chat/completions"
        else:
            self._endpoint = base + "/v1/chat/completions"
        # The SFT endpoint is normally a loopback/SSH-tunnel service.  Do not
        # route it through a workstation HTTP proxy (which can turn a healthy
        # tunnel response into a local 502).
        self._client = httpx.Client(
            transport=self.transport,
            timeout=30.0,
            trust_env=False,
        )

    def _fallback(self, context: ScientificPlannerContext) -> ScientificPlannerResult:
        if self.fallback is None:
            raise RuntimeError("SFT policy has no safe fallback planner")
        result = self.fallback.plan_action(context)
        codes = [result.fallbackCode] if result.fallbackCode else []
        codes.append(SFT_POLICY_FALLBACK_CODE)
        return ScientificPlannerResult(
            action=result.action,
            mode=result.mode,
            fallbackCode="+".join(codes),
            fallbackReasonCode=result.fallbackReasonCode,
        )

    def plan_action(self, context: ScientificPlannerContext) -> ScientificPlannerResult:
        policy_state = build_decision_policy_state(context)
        system = (
            "Return exactly one JSON object with keys selected_action, decision_reason, "
            "alternative_actions, stop_reason. selected_action must be one of the supplied "
            "candidate_actions. stop_reason is required only for finish and must otherwise be null. "
            "Use only the structured policy state; do not invent SQL, IDs, sample values, URLs, "
            "permissions, disease facts, or evidence. Return JSON only, no Markdown."
        )
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(
                    {"policy_state": policy_state}, ensure_ascii=False, separators=(",", ":")
                )},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": 256,
            "temperature": 0,
        }
        headers = {"Content-Type": "application/json"}
        if self.token.strip():
            headers["Authorization"] = f"Bearer {self.token}"

        for attempt in range(SFT_POLICY_MAX_RETRIES + 1):
            try:
                response = self._client.post(self._endpoint, headers=headers, json=body)
                if response.status_code != 200:
                    raise ValueError("SFT policy response rejected")
                payload = response.json()
                content: object
                if isinstance(payload, dict) and "choices" in payload:
                    content = payload["choices"][0]["message"]["content"]
                else:
                    content = payload
                decision = DecisionPolicyOutput.model_validate(_parse_json(content))
                action = _reify_action(context, decision)
                return ScientificPlannerResult(action=action, mode="sft_policy")
            except Exception as error:
                if attempt < SFT_POLICY_MAX_RETRIES:
                    body["messages"].append({
                        "role": "user",
                        "content": (
                            "The previous DecisionPolicy output was rejected by the closed contract. "
                            f"Schema-only feedback: {_feedback(error)}. Retry exactly once with JSON only; "
                            "do not add keys or invent values."
                        ),
                    })
        return self._fallback(context)

    def close(self) -> None:
        self._client.close()
