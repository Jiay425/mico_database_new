"""Optional model reranker boundary for source-bound retrieval candidates."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Mapping, Protocol


class RerankerPort(Protocol):
    """A reranker may score existing candidates but cannot create evidence."""

    def rerank(self, query: str, candidates: list[dict[str, str]]) -> dict[str, float]:
        ...


@dataclass(frozen=True)
class HttpCrossEncoderReranker:
    """OpenAI-compatible/REST cross-encoder adapter.

    The adapter is opt-in. Invalid, missing or partial provider output is
    rejected and treated as an empty score map, so an unavailable model never
    changes the source-bound retrieval contract or causes fabricated evidence.
    """

    endpoint: str
    model: str
    timeout_seconds: float = 8.0
    api_key: str = ""

    @classmethod
    def from_environment(cls, env: Mapping[str, str] | None = None) -> "HttpCrossEncoderReranker | None":
        source = os.environ if env is None else env
        if source.get("MICO_RERANKER_ENABLED", "").strip().lower() != "true":
            return None
        endpoint = source.get("MICO_RERANKER_ENDPOINT", "").strip()
        model = source.get("MICO_RERANKER_MODEL", "").strip()
        if not endpoint or not model or not endpoint.startswith(("http://", "https://")):
            return None
        try:
            timeout = max(0.5, min(30.0, float(source.get("MICO_RERANKER_TIMEOUT_SECONDS", "8"))))
        except ValueError:
            timeout = 8.0
        return cls(endpoint, model, timeout, source.get("MICO_RERANKER_API_KEY", "").strip())

    def rerank(self, query: str, candidates: list[dict[str, str]]) -> dict[str, float]:
        if not candidates:
            return {}
        try:
            import httpx

            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            response = httpx.post(
                self.endpoint,
                headers=headers,
                json={"model": self.model, "query": query, "documents": candidates},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload: Any = response.json()
        except Exception:
            return {}
        raw = payload.get("results") if isinstance(payload, dict) else payload
        if not isinstance(raw, list):
            return {}
        scores: dict[str, float] = {}
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            key = str(entry.get("chunkId") or entry.get("id") or "")
            value = entry.get("score")
            try:
                score = float(value)
            except (TypeError, ValueError):
                continue
            if key and math.isfinite(score):
                scores[key] = max(0.0, min(1.0, score))
        return scores
