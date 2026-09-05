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
import math
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

from mico_agent_runtime.contracts.scientific_policy import (
    ScientificPolicyInput,
    build_scientific_policy_input,
)
from mico_agent_runtime.contracts.decision_state import ScientificDecisionState
from mico_agent_runtime.contracts.research import (
    ResearchIntent,
    ScientificActionName,
    ScientificPlannerContext,
    StopReasonCode,
)
from mico_agent_runtime.ports.gemini_resilience import (
    GeminiRequestBudget,
    GeminiRequestBudgetExceeded,
    GeminiResponseCache,
    parse_retry_delay_seconds,
    sleep_before_retry,
)


SFT_POLICY_MAX_RETRIES = 3

# Kept as a named contract so offline Decision-SFT preparation can import the
# exact system message used by the serving adapter.  The value is intentionally
# unchanged from the previously inline scientific-policy prompt.
SCIENTIFIC_POLICY_PROMPT_VERSION = "generic_contract_hardened"
SCIENTIFIC_POLICY_SYSTEM_PROMPT = (
    "Return exactly one JSON object with keys selected_action, decision_reason, "
    "alternative_actions, stop_reason. The values in state.task.objectives describe "
    "user goals; objective names are not Scientific Action names and must never be used "
    "as selected_action. Choose exactly one Scientific Action. selected_action MUST be "
    "copied exactly, character-for-character, from state.action_space.available_actions; "
    "do not choose an unavailable action even when its related objective remains incomplete. "
    "stop_reason is required only for finish and must otherwise be null. "
    "Use only the six-block scientific decision state; do not invent SQL, IDs, sample values, "
    "URLs, permissions, disease facts, or evidence. If the state contains a validated Java "
    "read with row_count=0 or fewer than two observed groups, and execute_read_query is "
    "available, prefer a new execute_read_query so the materializer can request a different "
    "catalog-bounded projection; do not repeat inspect_cohort solely to spend a turn. If a "
    "validated inspect_cohort observation is already present, repeat it only when the state "
    "shows a genuinely new technical inspection is possible. These are policy guidelines; "
    "if a completed execute_read_query is present but group_comparison is still not_started, "
    "and the current group_count is below two, choose execute_read_query over inspect_cohort "
    "when it is available so the materializer can request a materially different projection. "
    "The currently registered group-analysis operators require exactly two observed groups. "
    "If group_count is not exactly two and execute_read_query is available, choose a fresh "
    "execute_read_query instead of an analysis action that the hard capability boundary has "
    "not made executable; do not treat an arbitrary multi-group discovery result as a two-group "
    "comparison. "
    "When group_count is exactly two, a numeric outcome is available, and compare_groups is "
    "available while group_comparison is not completed, prefer compare_groups before issuing "
    "another read solely to discover optional project or feature dimensions; the Runtime will "
    "keep those as separate facts for later actions. "
    "Runtime still validates availability and never substitutes an action. Return JSON only, "
    "no Markdown."
)


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


class _DecisionResponseRejected(ValueError):
    def __init__(self, response: httpx.Response) -> None:
        super().__init__("SFT policy response rejected")
        self.response = response
        self.retry_delay_seconds = parse_retry_delay_seconds(response)


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


def _canonicalize_policy_payload(value: object) -> object:
    """Apply one narrow provider typo repair before the closed contract.

    Gemini occasionally emits a null ``decision_res`` bookkeeping key along
    with the four documented policy fields.  It carries no decision data and
    is not admitted into ``DecisionPolicyOutput``; dropping only this exact
    null alias keeps the wire contract closed while allowing a harmless
    provider formatting variation to be retried without changing policy
    semantics.
    """

    if isinstance(value, Mapping) and value.get("decision_res") is None:
        return {key: item for key, item in value.items() if key != "decision_res"}
    return value


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
        if any(
            code != "numeric_outcome_available"
            for item in observations
            for code in item.qualityCodes
        ):
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


@dataclass
class HttpDecisionSftPlannerPort:
    """Call the SFT/DPO endpoint for a closed next-action choice only.

    A separate research planner must materialize the selected action into
    question-specific SQL or a bounded analysis plan.  This policy endpoint
    is intentionally never allowed to fall back to catalog-templated reads.
    """

    base_url: str
    model: str
    token: str = ""
    transport: httpx.BaseTransport | None = None
    # Explicit provenance keeps local HTTP canaries distinct from remote Qwen.
    policy_origin: str = "qwen_model"
    request_budget: GeminiRequestBudget | None = None
    response_cache: GeminiResponseCache | None = None
    sleep_fn: Callable[[float], None] = time.sleep
    max_retry_delay_seconds: float | None = None
    # Remote A100 inference can legitimately take longer than the old fixed
    # 30-second workstation timeout (especially on the first request after
    # adapter load).  Keep this explicit and configurable instead of turning
    # a slow but healthy policy response into a false contract failure.
    timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        parsed = urlsplit(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username:
            raise ValueError("SFT policy URL is invalid")
        if parsed.query or parsed.fragment or not self.model.strip():
            raise ValueError("SFT policy configuration is invalid")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("SFT policy timeout must be a positive finite number")
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
            timeout=self.timeout_seconds,
            trust_env=False,
            limits=httpx.Limits(max_keepalive_connections=0, max_connections=10),
        )
        self.request_count = 0
        self.cache_hit_count = 0

    def _post_json(self, body: dict[str, Any]) -> httpx.Response:
        """POST one policy request with optional Gemini run controls."""

        headers = {"Content-Type": "application/json"}
        if self.token.strip():
            headers["Authorization"] = f"Bearer {self.token}"
        key = self.response_cache.key(self._endpoint, body) if self.response_cache else None
        request = self._client.build_request("POST", self._endpoint, headers=headers, json=body)
        if key is not None:
            cached = self.response_cache.get(key, request)
            if cached is not None:
                self.cache_hit_count += 1
                return cached
        if self.request_budget is not None:
            self.request_budget.reserve("policy")
        self.request_count += 1
        response = self._client.send(request)
        if key is not None and response.status_code == 200:
            self.response_cache.put(key, response)
        return response

    def _select_legacy_action(self, context: ScientificPlannerContext) -> DecisionPolicyOutput:
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
                response = self._post_json(body)
                if response.status_code != 200:
                    raise _DecisionResponseRejected(response)
                payload = response.json()
                content: object
                if isinstance(payload, dict) and "choices" in payload:
                    content = payload["choices"][0]["message"]["content"]
                else:
                    content = payload
                decision = DecisionPolicyOutput.model_validate(
                    _canonicalize_policy_payload(_parse_json(content))
                )
                if decision.selected_action not in context.approvedActions:
                    raise ValueError("decision action is not approved")
                if not set(decision.alternative_actions).issubset(set(context.approvedActions)):
                    raise ValueError("decision alternatives are not approved")
                return decision
            except Exception as error:
                if isinstance(error, GeminiRequestBudgetExceeded):
                    raise
                if attempt < SFT_POLICY_MAX_RETRIES:
                    sleep_before_retry(
                        error,
                        sleep=self.sleep_fn,
                        max_delay_seconds=self.max_retry_delay_seconds,
                    )
                    body["messages"].append({
                        "role": "user",
                        "content": (
                            "The previous DecisionPolicy output was rejected by the closed contract. "
                            f"Schema-only feedback: {_feedback(error)}. Retry exactly once with JSON only; "
                            "do not add keys or invent values."
                        ),
                    })
        raise RuntimeError("DYNAMIC_ACTION_SELECTION_FAILED")

    def _select_scientific_action(
        self,
        policy_input: ScientificPolicyInput,
    ) -> DecisionPolicyOutput:
        """Select from the hard-available actions using only Decision State."""

        system = SCIENTIFIC_POLICY_SYSTEM_PROMPT
        state_payload = {
            "decision_type": "scientific_action",
            "state": policy_input.model_dump(mode="json"),
        }
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(
                    state_payload, ensure_ascii=False, separators=(",", ":")
                )},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": 256,
            "temperature": 0,
        }
        headers = {"Content-Type": "application/json"}
        if self.token.strip():
            headers["Authorization"] = f"Bearer {self.token}"

        available = set(policy_input.action_space.available_actions)
        last_error: Exception | None = None
        unavailable = False
        # A new-state policy gets one structured repair opportunity.  The
        # repair describes only technical illegality and never ranks actions.
        for attempt in range(2):
            try:
                response = self._post_json(body)
                if response.status_code != 200:
                    raise _DecisionResponseRejected(response)
                payload = response.json()
                content: object
                if isinstance(payload, dict) and "choices" in payload:
                    content = payload["choices"][0]["message"]["content"]
                else:
                    content = payload
                decision = DecisionPolicyOutput.model_validate(
                    _canonicalize_policy_payload(_parse_json(content))
                )
                if decision.selected_action not in available:
                    unavailable = True
                    raise ValueError("POLICY_ACTION_NOT_AVAILABLE")
                if not set(decision.alternative_actions).issubset(available):
                    unavailable = True
                    raise ValueError("POLICY_ACTION_NOT_AVAILABLE")
                return decision
            except Exception as error:
                if isinstance(error, GeminiRequestBudgetExceeded):
                    raise
                last_error = error
                if attempt == 0:
                    sleep_before_retry(
                        error,
                        sleep=self.sleep_fn,
                        max_delay_seconds=self.max_retry_delay_seconds,
                    )
                    feedback = _feedback(error)
                    body["messages"] = [
                        *body["messages"][:2],
                        {
                            "role": "user",
                            "content": (
                                "The selected action is not currently executable or the output "
                                "failed the closed contract. Objective names are not action names. "
                                "Choose exactly one action copied character-for-character from "
                                f"the current available_actions list {sorted(available)!r}; do not "
                                "choose an unavailable action. Return the same JSON keys. Technical "
                                f"validation feedback only: {feedback}"
                            ),
                        },
                    ]
        if unavailable:
            raise RuntimeError("POLICY_ACTION_NOT_AVAILABLE") from last_error
        # The new Decision-State policy is not allowed to choose a scientific
        # fallback when the provider/contract fails.  Returning a deterministic
        # action here would silently transfer policy ownership back to Runtime.
        raise RuntimeError("POLICY_DECISION_FAILED") from last_error

    def select_action(
        self,
        context: ScientificPlannerContext | ScientificDecisionState | ScientificPolicyInput,
    ) -> DecisionPolicyOutput:
        """Select using new Decision State; retain legacy calls temporarily."""

        if isinstance(context, (ScientificDecisionState, ScientificPolicyInput)):
            return self._select_scientific_action(build_scientific_policy_input(context))
        return self._select_legacy_action(context)

    def plan_action(self, context: ScientificPlannerContext):
        """Reject direct use: HybridIntentPlannerPort owns materialization."""

        del context
        raise RuntimeError("DYNAMIC_ACTION_MATERIALIZATION_REQUIRED")

    def close(self) -> None:
        self._client.close()
