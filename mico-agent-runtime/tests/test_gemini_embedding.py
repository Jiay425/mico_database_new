from __future__ import annotations

import json

import pytest

from mico_agent_runtime.contracts.evidence import EvidenceQuery
from mico_agent_runtime.knowledge.embeddings import (
    EmbeddingConfigurationError,
    EmbeddingRequestError,
    GeminiEmbeddingConfiguration,
    GeminiEmbeddingPort,
    extract_embedding,
    prepare_document,
    prepare_query,
)
from mico_agent_runtime.knowledge.local_retriever import (
    LocalKnowledgeIndexConfiguration,
    LocalKnowledgeSearchPort,
)


class FakeModels:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def embed_content(self, *, model, contents):
        self.calls.append((model, contents))
        return self.response


class FakeClient:
    def __init__(self, response):
        self.models = FakeModels(response)


def test_gemini_task_prefixes_and_normalized_response() -> None:
    fake = FakeClient({"embeddings": [{"values": [3.0, 4.0]}]})
    port = GeminiEmbeddingPort(
        GeminiEmbeddingConfiguration(modelName="gemini-embedding-2", apiKey="test-only"),
        client=fake,
    )
    assert prepare_query("gut microbiome") == "task: search result | query: gut microbiome"
    assert prepare_document("Paper", "evidence") == "title: Paper | text: evidence"
    assert port.embed_query("gut microbiome") == [0.6, 0.8]
    assert fake.models.calls == [
        ("gemini-embedding-2", "task: search result | query: gut microbiome"),
    ]


def test_gemini_configuration_requires_explicit_enable_and_key() -> None:
    with pytest.raises(EmbeddingConfigurationError):
        GeminiEmbeddingConfiguration.from_environment({})
    with pytest.raises(EmbeddingConfigurationError):
        GeminiEmbeddingConfiguration.from_environment({
            "MICO_GEMINI_EMBEDDING_ENABLED": "true",
        })
    with pytest.raises(EmbeddingConfigurationError):
        GeminiEmbeddingConfiguration.from_environment({
            "MICO_GEMINI_EMBEDDING_ENABLED": "true",
            "GEMINI_API_KEY": "test-only",
            "MICO_GEMINI_EMBEDDING_MODEL": "other-model",
        })


def test_invalid_provider_response_is_safe() -> None:
    with pytest.raises(EmbeddingRequestError) as error:
        extract_embedding({"embeddings": [{"values": [0.0, 0.0]}]})
    assert str(error.value) == "GEMINI_EMBEDDING_RESPONSE_INVALID"


def test_gemini_backend_uses_dense_index_and_preserves_graph_provenance(tmp_path) -> None:
    from tests.test_local_knowledge_retrieval import write_index

    write_index(tmp_path)
    dense_meta = {
        "indexVersion": "fulltext-gemini-paper-embedding-v1",
        "model": "gemini-embedding-2",
        "granularity": "paper",
        "dimension": 2,
    }
    (tmp_path / "medical_gemini_paper_embedding_meta.json").write_text(
        json.dumps(dense_meta), encoding="utf-8"
    )
    dense_rows = [
        {"indexVersion": "fulltext-gemini-paper-embedding-v1", "evidenceTier": "fulltext",
         "paperId": "PMC1", "embedding": [1.0, 0.0]},
        {"indexVersion": "fulltext-gemini-paper-embedding-v1", "evidenceTier": "fulltext",
         "paperId": "PMC2", "embedding": [0.0, 1.0]},
    ]
    with (tmp_path / "medical_gemini_paper_embedding_index.jsonl").open("w", encoding="utf-8") as handle:
        for row in dense_rows:
            handle.write(json.dumps(row) + "\n")

    class FakeEmbedding:
        modelName = "gemini-embedding-2"

        def embed_query(self, _query: str) -> list[float]:
            return [1.0, 0.0]

        def embed_document(self, _title: str | None, _text: str) -> list[float]:
            return [1.0, 0.0]

    port = LocalKnowledgeSearchPort(
        LocalKnowledgeIndexConfiguration(tmp_path, retrievalBackend="gemini"),
        embedding_port=FakeEmbedding(),
    )
    results = port.search(EvidenceQuery(
        topic="semantic literature search",
        direction="context",
        retrievalMode="hybrid",
        limit=2,
    ))
    assert results[0].externalId == "PMCID:PMC1#PMC1-A001"
    assert results[0].retrievalModel == "gemini-embedding-2"
    assert results[0].retrievalRoute == "hybrid"
    assert results[0].sourceChunkId == "PMC1-A001"
