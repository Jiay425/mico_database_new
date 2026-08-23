from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from typing import Protocol
from urllib.parse import urlsplit

import httpx

from mico_agent_runtime.contracts.graph_rag import (
    EvidenceSynthesisContext,
    EvidenceSynthesisResult,
    GroundedClaim,
    ModelReasoningStep,
    ModelGroundedClaim,
    ReasoningStep,
)


GRAPH_RAG_SYNTHESIS_FALLBACK = "GRAPHRAG_SYNTHESIS_DETERMINISTIC_FALLBACK"
GRAPH_RAG_SYNTHESIS_MODEL_MODE = "GRAPHRAG_SYNTHESIS_MODEL"
_DANGEROUS_OUTPUT = re.compile(
    r"(?i)(?:sourceSampleId|internalRecordId|cohortCondition|patient_id|subject_id|"
    r"https?://|file://|\b(?:SRR|ERR|DRR)\d+\b|\bMV_[A-Za-z0-9_-]+\b|\b(?:select|insert|update|delete|drop)\b)"
)


class GraphRagSynthesisError(ValueError):
    """Safe, provider-independent synthesis error."""


class GraphRagSynthesisPort(Protocol):
    def synthesize(self, context: EvidenceSynthesisContext) -> EvidenceSynthesisResult:
        ...


def _claim_id(statement: str, evidence_ids: list[str], path_ids: list[str]) -> str:
    raw = statement + "|" + "|".join(sorted(evidence_ids)) + "|" + "|".join(sorted(path_ids))
    return "claim-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _safe_statement(value: str) -> bool:
    return bool(value.strip()) and not _DANGEROUS_OUTPUT.search(value)


def _status_can_be_reported(status: str, path_statuses: set[str]) -> bool:
    """A model may not upgrade a speculative/conflicted path to supported."""
    if "conflicted" in path_statuses and status == "supported":
        return False
    if "speculative" in path_statuses and status == "supported":
        return False
    return True


class DeterministicGraphRagSynthesisPort:
    """Structured, source-bound fallback when no generation model is configured."""

    def synthesize(self, context: EvidenceSynthesisContext) -> EvidenceSynthesisResult:
        claims: list[GroundedClaim] = []
        reasoning_steps: list[ReasoningStep] = []
        for evidence in context.evidence:
            for path in evidence.reasoningPaths:
                if not path.hops:
                    continue
                path_text = " → ".join(
                    f"{hop.fromEntity} [{hop.relation}] {hop.toEntity}"
                    for hop in path.hops
                )
                statement = "来源图路径（非医学结论）：" + path_text
                claims.append(GroundedClaim(
                    claimId=_claim_id(statement, [evidence.evidenceId], [path.pathId]),
                    statement=statement,
                    supportStatus=path.status,
                    evidenceIds=[evidence.evidenceId],
                    reasoningPathIds=[path.pathId],
                ))
                for index, hop in enumerate(path.hops, start=1):
                    reasoning_steps.append(ReasoningStep(
                        stepIndex=min(20, len(reasoning_steps) + 1),
                        description=(
                            f"{hop.fromEntity} [{hop.relation}] {hop.toEntity}; "
                            "the step is retained only as a source-bound graph assertion."
                        ),
                        supportStatus=hop.supportStatus,
                        evidenceIds=[evidence.evidenceId],
                        reasoningPathIds=[path.pathId],
                    ))
                if len(claims) >= 20:
                    return EvidenceSynthesisResult(
                        claims=claims,
                        reasoningSteps=reasoning_steps[:20],
                        conclusion=(
                            "The result is a source-bound literature graph summary. "
                            "Supported, speculative, and conflicted steps are kept distinct; "
                            "it is not a medical conclusion."
                        ),
                        mode="deterministic",
                        fallbackCode=GRAPH_RAG_SYNTHESIS_FALLBACK,
                    )
        return EvidenceSynthesisResult(
            claims=claims,
            reasoningSteps=reasoning_steps[:20],
            conclusion=(
                "The result is a source-bound literature graph summary and does not establish "
                "a medical, causal, or treatment conclusion."
            ),
            mode="deterministic",
            fallbackCode=GRAPH_RAG_SYNTHESIS_FALLBACK,
        )


class HttpGraphRagSynthesisPort:
    """OpenAI-compatible structured generator with evidence-only grounding."""

    def __init__(
        self,
        base_url: str,
        model: str,
        token: str,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username:
            raise GraphRagSynthesisError("GRAPHRAG_SYNTHESIS_CONFIGURATION_INVALID")
        if parsed.query or parsed.fragment or not model.strip() or not token.strip():
            raise GraphRagSynthesisError("GRAPHRAG_SYNTHESIS_CONFIGURATION_INVALID")
        base = base_url.rstrip("/")
        if (parsed.hostname or "").lower().endswith("deepseek.com") or parsed.path.rstrip("/").endswith("/v1"):
            self._endpoint = base + "/chat/completions"
        else:
            self._endpoint = base + "/v1/chat/completions"
        self._model = model.strip()
        self._token = token
        self._is_deepseek = (parsed.hostname or "").lower().endswith("deepseek.com")
        self._client = httpx.Client(transport=transport, timeout=45.0)
        self._fallback = DeterministicGraphRagSynthesisPort()

    @staticmethod
    def _parse_json(value: object) -> object:
        if not isinstance(value, str):
            raise ValueError("invalid model content")
        text = value.strip()
        text = re.sub(r"(?is)<think>.*?</think>", "", text).strip()
        if text.startswith("```") and text.endswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1]).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Some OpenAI-compatible providers prepend a short sentence even
            # when JSON mode is requested.  Parse only the first JSON object;
            # no surrounding text is ever returned to callers.
            start = text.find("{")
            if start < 0:
                raise
            parsed, end = json.JSONDecoder().raw_decode(text[start:])
            if text[start + end:].strip():
                raise ValueError("trailing non-json model content")
            return parsed

    def synthesize(self, context: EvidenceSynthesisContext) -> EvidenceSynthesisResult:
        system = (
            "Return only closed JSON with exactly these useful fields: "
            "reasoningSteps:[{stepIndex,description,supportStatus,evidenceIds,reasoningPathIds}], "
            "conclusion:string, claims:[{statement,supportStatus,evidenceIds,reasoningPathIds}]. "
            "Use the exact field names stepIndex, description, supportStatus, evidenceIds, and "
            "reasoningPathIds; never use step, stepId, source, citation, or free-form nested objects. "
            "Each reasoningSteps item must include at least one evidenceId and may include path IDs only "
            "from the supplied graph paths. Example: {\"reasoningSteps\":[{\"stepIndex\":1,"
            "\"description\":\"A source-bound step.\",\"supportStatus\":\"supported\","
            "\"evidenceIds\":[\"evidence-...\"],\"reasoningPathIds\":[\"path-...\"]}],"
            "\"conclusion\":\"A cautious source-bound summary.\",\"claims\":[]}. "
            "Use only evidenceIds and reasoningPathIds present in the supplied context. "
            "Every reasoning step must cite at least one supplied evidenceId and may not expose hidden chain-of-thought. "
            "When the supplied context contains a graph path, return at least one grounded claim; "
            "do not use an empty claims array merely to avoid answering. "
            "A speculative or conflicted path must not be upgraded to supported. "
            "Every statement must be a cautious literature evidence statement, never a diagnosis, "
            "causal claim, treatment recommendation, or unsupported inference. Do not emit sample IDs, "
            "record IDs, locators, URLs, SQL, payloads, credentials, or hidden chain-of-thought. "
            "The graph paths are the auditable reasoning structure; cite them instead of inventing steps."
        )
        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": context.model_dump_json()},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": 2400,
            "temperature": 0,
        }
        if self._is_deepseek:
            body["thinking"] = {"type": "disabled"}
        try:
            return self._synthesize_model_response(context, body)
        except Exception:
            retry_body = dict(body)
            retry_messages = list(body["messages"])
            retry_messages[0] = {
                "role": "system",
                "content": (
                    system
                    + " STRICT RETRY: output exactly one object; every reasoning step and claim "
                    "must contain all five named fields and cite known IDs. Use an empty claims "
                    "array if no claim can be safely grounded."
                ),
            }
            retry_body["messages"] = retry_messages
            try:
                return self._synthesize_model_response(context, retry_body)
            except Exception:
                fallback = self._fallback.synthesize(context)
                return fallback.model_copy(update={"fallbackCode": GRAPH_RAG_SYNTHESIS_FALLBACK})

    def _synthesize_model_response(
        self,
        context: EvidenceSynthesisContext,
        body: dict[str, object],
    ) -> EvidenceSynthesisResult:
        response = self._client.post(
            self._endpoint,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
            },
            json=body,
        )
        # DeepSeek-compatible deployments may support optional JSON/thinking
        # hints only on selected models. The evidence validation remains
        # mandatory after either retry.
        if response.status_code in {400, 422}:
            fallback_body = dict(body)
            fallback_body.pop("response_format", None)
            response = self._client.post(
                self._endpoint,
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Content-Type": "application/json",
                },
                json=fallback_body,
            )
            if response.status_code in {400, 422} and "thinking" in fallback_body:
                fallback_body.pop("thinking", None)
                response = self._client.post(
                    self._endpoint,
                    headers={
                        "Authorization": f"Bearer {self._token}",
                        "Content-Type": "application/json",
                    },
                    json=fallback_body,
                )
        if response.status_code != 200:
            raise ValueError("model response rejected")
        payload = response.json()
        parsed = self._parse_json(payload["choices"][0]["message"]["content"])
        raw_claims = parsed.get("claims") if isinstance(parsed, dict) else None
        raw_steps = parsed.get("reasoningSteps") if isinstance(parsed, dict) else None
        conclusion = parsed.get("conclusion") if isinstance(parsed, dict) else None
        if (
            not isinstance(raw_claims, list)
            or not isinstance(raw_steps, list)
            or not isinstance(conclusion, str)
            or not _safe_statement(conclusion)
        ):
            raise ValueError("structured reasoning output missing")
        if not raw_claims and any(item.reasoningPaths for item in context.evidence):
            raise ValueError("grounded claim missing")
        validated = [ModelGroundedClaim.model_validate(item) for item in raw_claims[:20]]
        validated_steps = [ModelReasoningStep.model_validate(item) for item in raw_steps[:20]]
        allowed_evidence = {item.evidenceId for item in context.evidence}
        allowed_paths = {
            path.pathId
            for item in context.evidence
            for path in item.reasoningPaths
        }
        path_statuses = {
            path.pathId: {path.status, *(hop.supportStatus for hop in path.hops)}
            for item in context.evidence
            for path in item.reasoningPaths
        }
        path_evidence = {
            path.pathId: evidence.evidenceId
            for evidence in context.evidence
            for path in evidence.reasoningPaths
        }

        def paths_bind_to_evidence(evidence_ids: list[str], path_ids: list[str]) -> bool:
            return all(path_evidence.get(path_id) in evidence_ids for path_id in path_ids)

        reasoning_steps: list[ReasoningStep] = []
        for item in validated_steps:
            if (
                not _safe_statement(item.description)
                or not set(item.evidenceIds).issubset(allowed_evidence)
                or not set(item.reasoningPathIds).issubset(allowed_paths)
                or not paths_bind_to_evidence(item.evidenceIds, item.reasoningPathIds)
                or not _status_can_be_reported(
                    item.supportStatus,
                    {status for path_id in item.reasoningPathIds for status in path_statuses.get(path_id, set())},
                )
            ):
                raise ValueError("model reasoning step is not grounded")
            reasoning_steps.append(ReasoningStep(
                stepIndex=item.stepIndex,
                description=item.description,
                supportStatus=item.supportStatus,
                evidenceIds=item.evidenceIds,
                reasoningPathIds=item.reasoningPathIds,
            ))
        claims: list[GroundedClaim] = []
        for item in validated:
            if (
                not _safe_statement(item.statement)
                or not set(item.evidenceIds).issubset(allowed_evidence)
                or not set(item.reasoningPathIds).issubset(allowed_paths)
                or not paths_bind_to_evidence(item.evidenceIds, item.reasoningPathIds)
                or not _status_can_be_reported(
                    item.supportStatus,
                    {status for path_id in item.reasoningPathIds for status in path_statuses.get(path_id, set())},
                )
            ):
                raise ValueError("model claim is not grounded")
            claims.append(GroundedClaim(
                claimId=_claim_id(item.statement, item.evidenceIds, item.reasoningPathIds),
                statement=item.statement,
                supportStatus=item.supportStatus,
                evidenceIds=item.evidenceIds,
                reasoningPathIds=item.reasoningPathIds,
            ))
        return EvidenceSynthesisResult(
            claims=claims,
            reasoningSteps=reasoning_steps,
            conclusion=conclusion,
            mode="model",
            fallbackCode=None,
        )

    def close(self) -> None:
        self._client.close()


def build_graph_rag_synthesis_port(
    env: Mapping[str, str] | None = None,
    transport: httpx.BaseTransport | None = None,
) -> GraphRagSynthesisPort:
    source = os.environ if env is None else env
    base_url = source.get("MICO_GRAPH_RAG_GENERATOR_BASE_URL", "").strip()
    model = source.get("MICO_GRAPH_RAG_GENERATOR_MODEL", "").strip()
    token = source.get("MICO_GRAPH_RAG_GENERATOR_TOKEN", "").strip()
    if not base_url or not model or not token:
        return DeterministicGraphRagSynthesisPort()
    try:
        return HttpGraphRagSynthesisPort(base_url, model, token, transport=transport)
    except GraphRagSynthesisError:
        return DeterministicGraphRagSynthesisPort()
