from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlsplit

import httpx

from mico_agent_runtime.contracts.evidence import EvidenceQuery, LiteratureEvidenceItem


class EvidenceSearchError(RuntimeError):
    """Safe boundary error for an external evidence adapter."""


class EvidenceSearchPort(Protocol):
    def search(self, query: EvidenceQuery) -> list[LiteratureEvidenceItem]:
        ...


class UnconfiguredEvidenceSearchPort:
    """Fail-closed P4 default; it never fabricates literature results."""

    def search(self, _query: EvidenceQuery) -> list[LiteratureEvidenceItem]:
        return []


class EvidenceSearchConfigurationError(ValueError):
    """The external evidence adapter is not safely configured."""


@dataclass(frozen=True)
class EvidenceSearchConfiguration:
    pubmedBaseUrl: str | None = None
    crossrefBaseUrl: str | None = None

    @classmethod
    def from_environment(cls, env: dict[str, str] | None = None) -> "EvidenceSearchConfiguration":
        source = os.environ if env is None else env
        if source.get("MICO_EVIDENCE_SEARCH_ENABLED", "").strip().lower() != "true":
            raise EvidenceSearchConfigurationError("EVIDENCE_SEARCH_DISABLED")
        pubmed = source.get("MICO_EVIDENCE_PUBMED_BASE_URL", "").strip() or None
        crossref = source.get("MICO_EVIDENCE_CROSSREF_BASE_URL", "").strip() or None
        if pubmed is None and crossref is None:
            raise EvidenceSearchConfigurationError("EVIDENCE_SEARCH_URL_MISSING")
        for value in (pubmed, crossref):
            if value is not None:
                parsed = urlsplit(value)
                if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.query or parsed.fragment:
                    raise EvidenceSearchConfigurationError("EVIDENCE_SEARCH_URL_INVALID")
        return cls(pubmedBaseUrl=pubmed, crossrefBaseUrl=crossref)


class HttpEvidenceSearchPort:
    """Metadata-only PubMed/CrossRef adapter; all output stays in the evidence layer."""

    def __init__(self, configuration: EvidenceSearchConfiguration,
                 transport: httpx.BaseTransport | None = None) -> None:
        if configuration.pubmedBaseUrl is None and configuration.crossrefBaseUrl is None:
            raise EvidenceSearchConfigurationError("EVIDENCE_SEARCH_URL_MISSING")
        self._configuration = configuration
        self._client = httpx.Client(transport=transport, timeout=5.0)

    def search(self, query: EvidenceQuery) -> list[LiteratureEvidenceItem]:
        results: list[LiteratureEvidenceItem] = []
        if self._configuration.pubmedBaseUrl:
            results.extend(self._search_pubmed(query))
        if self._configuration.crossrefBaseUrl:
            results.extend(self._search_crossref(query))
        return results[: query.limit]

    def _search_pubmed(self, query: EvidenceQuery) -> list[LiteratureEvidenceItem]:
        try:
            base = self._configuration.pubmedBaseUrl.rstrip("/")
            search = self._client.get(
                f"{base}/esearch.fcgi",
                params={
                    "db": "pubmed",
                    "term": f"{query.topic} {query.taxonName or ''}".strip(),
                    "retmode": "json",
                    "retmax": str(query.limit),
                },
            )
            search.raise_for_status()
            ids = search.json().get("esearchresult", {}).get("idlist", [])
            if not isinstance(ids, list):
                raise EvidenceSearchError("EVIDENCE_SOURCE_FAILED")
            if not ids:
                return []
            summary = self._client.get(
                f"{base}/esummary.fcgi",
                params={"db": "pubmed", "id": ",".join(str(item) for item in ids), "retmode": "json"},
            )
            summary.raise_for_status()
            payload = summary.json().get("result", {})
            results: list[LiteratureEvidenceItem] = []
            for item_id in ids:
                item = payload.get(str(item_id), {})
                title = item.get("title")
                if not isinstance(title, str) or not title.strip():
                    continue
                results.append(LiteratureEvidenceItem(
                    evidenceId=_opaque_evidence_id("pubmed", str(item_id)),
                    taxonName=query.taxonName,
                    source="pubmed",
                    externalId=f"PMID:{item_id}",
                    title=title[:512],
                    journal=str(item.get("fulljournalname", ""))[:256] or None,
                    publicationYear=_year(item.get("pubdate")),
                    direction=query.direction,
                    summary="PubMed metadata match; full-text and study-design review is still required.",
                ))
            return results
        except EvidenceSearchError:
            raise
        except Exception as exc:
            raise EvidenceSearchError("EVIDENCE_SOURCE_FAILED") from exc

    def _search_crossref(self, query: EvidenceQuery) -> list[LiteratureEvidenceItem]:
        try:
            response = self._client.get(
                self._configuration.crossrefBaseUrl.rstrip("/") + "/works",
                params={
                    "query.bibliographic": f"{query.topic} {query.taxonName or ''}".strip(),
                    "rows": str(query.limit),
                    "select": "DOI,title,container-title,published",
                },
            )
            response.raise_for_status()
            items = response.json().get("message", {}).get("items", [])
            if not isinstance(items, list):
                raise EvidenceSearchError("EVIDENCE_SOURCE_FAILED")
            results: list[LiteratureEvidenceItem] = []
            for item in items:
                doi = item.get("DOI")
                titles = item.get("title")
                if not isinstance(doi, str) or not doi or not isinstance(titles, list) or not titles:
                    continue
                containers = item.get("container-title") or []
                results.append(LiteratureEvidenceItem(
                    evidenceId=_opaque_evidence_id("crossref", doi),
                    taxonName=query.taxonName,
                    source="crossref",
                    externalId=f"DOI:{doi}",
                    title=str(titles[0])[:512],
                    journal=str(containers[0])[:256] if containers else None,
                    publicationYear=_year(item.get("published")),
                    direction=query.direction,
                    summary="CrossRef metadata match; bibliographic evidence is not a causal or clinical conclusion.",
                ))
            return results
        except EvidenceSearchError:
            raise
        except Exception as exc:
            raise EvidenceSearchError("EVIDENCE_SOURCE_FAILED") from exc

    def close(self) -> None:
        self._client.close()


def _opaque_evidence_id(source: str, value: str) -> str:
    from hashlib import sha256

    return "evidence-" + sha256(f"{source}|{value}".encode("utf-8")).hexdigest()[:32]


def _year(value: object) -> int:
    if isinstance(value, dict):
        parts = value.get("date-parts")
        if isinstance(parts, list) and parts and isinstance(parts[0], list) and parts[0]:
            candidate = parts[0][0]
            if isinstance(candidate, int) and 1900 <= candidate <= 2100:
                return candidate
    text = str(value or "")
    for token in text.split():
        if len(token) == 4 and token.isdigit():
            year = int(token)
            if 1900 <= year <= 2100:
                return year
    return 2000
