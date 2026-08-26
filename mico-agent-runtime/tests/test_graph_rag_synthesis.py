from __future__ import annotations

import json

import httpx

from mico_agent_runtime.contracts.graph_rag import (
    EvidenceSynthesisContext,
    GraphEvidencePath,
    GraphPathStep,
    ReasoningPath,
    SynthesisEvidence,
)
from mico_agent_runtime.knowledge.synthesis import (
    GRAPH_RAG_SYNTHESIS_FALLBACK,
    DeterministicGraphRagSynthesisPort,
    HttpGraphRagSynthesisPort,
)


def context() -> EvidenceSynthesisContext:
    path = GraphEvidencePath(
        pathId="path-11111111111111111111111111111111",
        status="supported",
        hops=[GraphPathStep(
            fromEntity="type_2_diabetes",
            relation="MENTIONED_BY",
            toEntity="PMC1-A001",
            evidenceChunkId="PMC1-A001",
        )],
        sourceDocumentIds=["PMCID:PMC1"],
        confidence=1.0,
    )
    return EvidenceSynthesisContext(
        questionSummary="diabetes microbiome evidence",
        evidence=[SynthesisEvidence(
            evidenceId="evidence-11111111111111111111111111111111",
            title="Full-text paper",
            summary="A bounded source summary.",
            graphPaths=[path],
        )],
    )


def transport_for(payload: dict) -> httpx.MockTransport:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps(payload)}}],
        })

    return httpx.MockTransport(handler)


def test_deepseek_compatible_base_url_uses_chat_completions_endpoint() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps({
                "claims": [{
                    "statement": "The supplied source path is retained as literature evidence.",
                    "supportStatus": "supported",
                    "evidenceIds": ["evidence-11111111111111111111111111111111"],
                    "reasoningPathIds": ["path-11111111111111111111111111111111"],
                }],
                "reasoningSteps": [{
                    "stepIndex": 1,
                    "description": "The supplied source path is source-bound.",
                    "supportStatus": "supported",
                    "evidenceIds": ["evidence-11111111111111111111111111111111"],
                    "reasoningPathIds": ["path-11111111111111111111111111111111"],
                }],
                "conclusion": "No source-bound claim was generated.",
            })}}],
        })

    port = HttpGraphRagSynthesisPort(
        "https://api.deepseek.com",
        "deepseek-v4-flash",
        "test-token",
        transport=httpx.MockTransport(handler),
    )
    result = port.synthesize(context())
    assert result.mode == "model"
    assert seen == ["https://api.deepseek.com/chat/completions"]


def test_gemini_openai_compatible_base_url_uses_chat_completions_endpoint() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps({
                "claims": [{
                    "statement": "The supplied source path is retained as literature evidence.",
                    "supportStatus": "supported",
                    "evidenceIds": ["evidence-11111111111111111111111111111111"],
                    "reasoningPathIds": ["path-11111111111111111111111111111111"],
                }],
                "reasoningSteps": [{
                    "stepIndex": 1,
                    "description": "The supplied source path is source-bound.",
                    "supportStatus": "supported",
                    "evidenceIds": ["evidence-11111111111111111111111111111111"],
                    "reasoningPathIds": ["path-11111111111111111111111111111111"],
                }],
                "conclusion": "No source-bound claim was generated.",
            })}}],
        })

    port = HttpGraphRagSynthesisPort(
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "gemini-3.5-flash",
        "test-token",
        transport=httpx.MockTransport(handler),
    )
    result = port.synthesize(context())

    assert result.mode == "model"
    assert seen == [
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    ]


def test_deterministic_synthesis_is_path_bound() -> None:
    result = DeterministicGraphRagSynthesisPort().synthesize(context())
    assert result.mode == "deterministic"
    assert result.fallbackCode == GRAPH_RAG_SYNTHESIS_FALLBACK
    assert result.claims[0].evidenceIds == ["evidence-11111111111111111111111111111111"]
    assert result.claims[0].reasoningPathIds == ["path-11111111111111111111111111111111"]
    assert result.reasoningSteps[0].reasoningPathIds == ["path-11111111111111111111111111111111"]
    assert result.conclusion is not None


def test_deterministic_synthesis_preserves_conflict_from_path_hops() -> None:
    conflicted_path = GraphEvidencePath(
        pathId="path-22222222222222222222222222222222",
        status="supported",
        hops=[GraphPathStep(
            fromEntity="type_2_diabetes",
            relation="ASSOCIATED_WITH",
            toEntity="microbe concept",
            evidenceChunkId="PMC1-A001",
            supportStatus="conflicted",
        )],
        sourceDocumentIds=["PMCID:PMC1"],
        confidence=0.4,
    )
    conflicted_context = context().model_copy(update={
        "evidence": [context().evidence[0].model_copy(update={
            "graphPaths": [conflicted_path],
            "reasoningPaths": [ReasoningPath.from_graph_path(conflicted_path)],
        })]
    })
    result = DeterministicGraphRagSynthesisPort().synthesize(conflicted_context)
    assert result.claims[0].supportStatus == "conflicted"
    assert result.reasoningSteps[0].supportStatus == "conflicted"


def test_model_synthesis_accepts_only_known_evidence_and_paths() -> None:
    port = HttpGraphRagSynthesisPort(
        "https://generator.example",
        "model",
        "test-token",
        transport=transport_for({
            "claims": [{
                "statement": "The source path connects the reviewed term to a full-text chunk.",
                "supportStatus": "supported",
                "evidenceIds": ["evidence-11111111111111111111111111111111"],
                "reasoningPathIds": ["path-11111111111111111111111111111111"],
            }],
            "reasoningSteps": [{
                "stepIndex": 1,
                "description": "The supplied graph hop is source-bound.",
                "supportStatus": "supported",
                "evidenceIds": ["evidence-11111111111111111111111111111111"],
                "reasoningPathIds": ["path-11111111111111111111111111111111"],
            }],
            "conclusion": "This is a source-bound literature evidence summary, not a medical conclusion.",
        }),
    )
    result = port.synthesize(context())
    assert result.mode == "model"
    assert result.fallbackCode is None
    assert result.claims[0].claimId.startswith("claim-")
    assert result.reasoningSteps[0].evidenceIds == ["evidence-11111111111111111111111111111111"]
    assert result.conclusion is not None


def test_model_synthesis_downgrades_untrusted_claim_without_leaking_it() -> None:
    port = HttpGraphRagSynthesisPort(
        "https://generator.example",
        "model",
        "test-token",
        transport=transport_for({
            "claims": [{
                "statement": "sourceSampleId=must-not-cross-boundary",
                "supportStatus": "supported",
                "evidenceIds": ["evidence-11111111111111111111111111111111"],
                "reasoningPathIds": ["path-11111111111111111111111111111111"],
            }],
            "reasoningSteps": [],
            "conclusion": "This output is rejected because it contains a forbidden value.",
        }),
    )
    result = port.synthesize(context())
    assert result.mode == "deterministic"
    assert result.fallbackCode == GRAPH_RAG_SYNTHESIS_FALLBACK
    assert "sourceSampleId" not in result.model_dump_json()


def test_model_synthesis_rejects_path_bound_to_different_evidence() -> None:
    base = context()
    context_with_two = base.model_copy(update={
        "evidence": base.evidence + [SynthesisEvidence(
            evidenceId="evidence-33333333333333333333333333333333",
            title="Second source",
            summary="Another bounded source.",
        )]
    })
    port = HttpGraphRagSynthesisPort(
        "https://generator.example",
        "model",
        "test-token",
        transport=transport_for({
            "claims": [{
                "statement": "This claim uses a path from the wrong evidence item.",
                "supportStatus": "supported",
                "evidenceIds": ["evidence-33333333333333333333333333333333"],
                "reasoningPathIds": ["path-11111111111111111111111111111111"],
            }],
            "reasoningSteps": [],
            "conclusion": "This output must be rejected.",
        }),
    )
    result = port.synthesize(context_with_two)
    assert result.mode == "deterministic"
    assert result.fallbackCode == GRAPH_RAG_SYNTHESIS_FALLBACK
