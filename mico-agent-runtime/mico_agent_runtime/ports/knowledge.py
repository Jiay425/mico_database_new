"""Ports for versioned local full-text knowledge retrieval."""

from typing import Literal, Protocol

from mico_agent_runtime.contracts.evidence import EvidenceQuery, LiteratureEvidenceItem
from mico_agent_runtime.contracts.retrieval import RetrievalPlan
from mico_agent_runtime.knowledge.local_retriever import (
    LocalKnowledgeIndexConfiguration,
    LocalKnowledgeIndexConfigurationError,
    LocalKnowledgeSearchPort,
)
from mico_agent_runtime.knowledge.database_retriever import DatabaseKnowledgeSearchPort


class KnowledgeSearchPort(Protocol):
    def search(self, query: EvidenceQuery) -> list[LiteratureEvidenceItem]:
        ...


class PlannedKnowledgeSearchPort(KnowledgeSearchPort, Protocol):
    """Production retrieval port with the closed P2-G2 execution plan."""

    def search_parallel(
        self,
        query: EvidenceQuery,
        branches: tuple[Literal["vector", "graph"], ...],
        plan: RetrievalPlan | None = None,
    ) -> list[LiteratureEvidenceItem]:
        ...


__all__ = [
    "KnowledgeSearchPort",
    "PlannedKnowledgeSearchPort",
    "LocalKnowledgeIndexConfiguration",
    "LocalKnowledgeIndexConfigurationError",
    "LocalKnowledgeSearchPort",
    "DatabaseKnowledgeSearchPort",
]
