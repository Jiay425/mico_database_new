"""Value-free controlled knowledge evidence for Hard Eval conflict cases."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from mico_agent_runtime.contracts.evidence import EvidenceQuery, LiteratureEvidenceItem
from mico_agent_runtime.contracts.graph_rag import GraphEvidencePath, GraphPathStep


HARD_CONFLICT_CASE_PREFIX = "p2j4-hard-evidence-conflict-"


def is_hard_conflict_case(case_id: str) -> bool:
    return case_id.startswith(HARD_CONFLICT_CASE_PREFIX)


def _evidence_id(query: EvidenceQuery) -> str:
    import hashlib

    return "evidence-" + hashlib.sha256((HARD_CONFLICT_CASE_PREFIX + query.topic).encode()).hexdigest()[:32]


def _path(query: EvidenceQuery) -> GraphEvidencePath:
    import hashlib

    path_id = "path-" + hashlib.sha256(("conflicted|" + query.topic).encode()).hexdigest()[:32]
    return GraphEvidencePath(
        pathId=path_id,
        status="conflicted",
        hops=[GraphPathStep(
            fromEntity="disease association",
            relation="ASSOCIATED_WITH",
            toEntity="microbial feature",
            evidenceChunkId="hard-conflict-chunk",
            supportStatus="conflicted",
        )],
        sourceDocumentIds=["hard-conflict-document"],
        confidence=0.25,
    )


def _supported_path(query: EvidenceQuery) -> GraphEvidencePath:
    import hashlib

    path_id = "path-" + hashlib.sha256(("supported|" + query.topic).encode()).hexdigest()[:32]
    return GraphEvidencePath(
        pathId=path_id,
        status="supported",
        hops=[GraphPathStep(
            fromEntity="disease association",
            relation="ASSOCIATED_WITH",
            toEntity="microbial feature",
            evidenceChunkId="hard-supported-chunk",
            supportStatus="supported",
        )],
        sourceDocumentIds=["hard-supported-document"],
        confidence=0.9,
    )


class HardConflictKnowledgePort:
    """Bounded vector+graph fixture; it never connects to a database."""

    def search(self, query: EvidenceQuery) -> list[LiteratureEvidenceItem]:
        return self.search_parallel(query, branches=("vector", "graph"))

    def search_parallel(
        self,
        query: EvidenceQuery,
        branches: tuple[Literal["vector", "graph"], ...],
        plan: object | None = None,
    ) -> list[LiteratureEvidenceItem]:
        evidence_id = _evidence_id(query)
        common = {
            "evidenceId": evidence_id,
            "source": "internal_knowledge",
            "externalId": "hard-conflict-document#hard-conflict-chunk",
            "title": "Controlled conflicting evidence fixture",
            "journal": "Controlled Eval Source",
            "publicationYear": 2025,
            "direction": "context",
            "summary": "A bounded fixture containing one graph assertion and one opposing literature route.",
            "evidenceTier": "fulltext",
            "sourceChunkId": "hard-conflict-chunk",
            "retrievalScore": 0.8,
            "sourceExcerpt": "Controlled source-bound conflict; not a medical conclusion.",
        }
        result: list[LiteratureEvidenceItem] = []
        if "vector" in branches:
            result.append(LiteratureEvidenceItem(
                **common,
                retrievalRoute="vector",
                retrievalModel="gemini-embedding-2",
                vectorScore=0.8,
                graphScore=0.0,
                rerankScore=0.8,
                retrievalSources=["vector"],
            ))
        if "graph" in branches:
            result.append(LiteratureEvidenceItem(
                **common,
                retrievalRoute="graph",
                retrievalModel=None,
                vectorScore=0.0,
                graphScore=0.7,
                rerankScore=0.7,
                graphPaths=[_path(query)],
                reasoningPaths=[],
                retrievalSources=["graph"],
            ))
        return result

    def close(self) -> None:
        return None


class HardSupportedKnowledgePort(HardConflictKnowledgePort):
    """Bounded supported-evidence fixture for variant stability cases."""

    def search_parallel(
        self,
        query: EvidenceQuery,
        branches: tuple[Literal["vector", "graph"], ...],
        plan: object | None = None,
    ) -> list[LiteratureEvidenceItem]:
        evidence_id = _evidence_id(query)
        common = {
            "evidenceId": evidence_id,
            "source": "internal_knowledge",
            "externalId": "hard-supported-document#hard-supported-chunk",
            "title": "Controlled supported evidence fixture",
            "journal": "Controlled Eval Source",
            "publicationYear": 2025,
            "direction": "context",
            "summary": "A bounded fixture containing source-bound supported evidence.",
            "evidenceTier": "fulltext",
            "sourceChunkId": "hard-supported-chunk",
            "retrievalScore": 0.9,
            "sourceExcerpt": "Controlled source-bound supported evidence; not a medical conclusion.",
        }
        result: list[LiteratureEvidenceItem] = []
        if "vector" in branches:
            result.append(LiteratureEvidenceItem(
                **common,
                retrievalRoute="vector",
                retrievalModel="gemini-embedding-2",
                vectorScore=0.9,
                graphScore=0.0,
                rerankScore=0.9,
                retrievalSources=["vector"],
            ))
        if "graph" in branches:
            result.append(LiteratureEvidenceItem(
                **common,
                retrievalRoute="graph",
                retrievalModel=None,
                vectorScore=0.0,
                graphScore=0.9,
                rerankScore=0.9,
                graphPaths=[_supported_path(query)],
                reasoningPaths=[],
                retrievalSources=["graph"],
            ))
        return result
