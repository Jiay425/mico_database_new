# P2-G Retrieval Quality Plan v2

## Integrity rule

`p2g-graded-qrels-v1.json` is a source-bound, expert-assisted provisional
benchmark. It is useful for recall diagnostics, but it is not independent human
relevance judgment. Its provisional grades must never select production
weights. `tune_p2g_rrf_v1.py` therefore refuses qrels whose
`humanAdjudicationStatus` is not `approved`.

## Two-label protocol

Each query/candidate pair carries two independent labels:

1. `humanGrade` (0–3): a reviewer judges whether the chunk answers the query;
2. `provenanceGrade` (0–3): the pipeline records whether an accepted graph edge
   or direct source passage supports it.

The two grades must not be collapsed. A chunk can be source-bound but irrelevant
to the question, or textually relevant but not support the claimed relation.
Disagreements are retained as review cases rather than averaged away.

## Release gate

A candidate retrieval/reranker version can replace the active version only when
the approved qrels show no regression in all of:

- Recall@50;
- nDCG@10;
- MRR@10;
- Evidence Precision@5;
- graph path correctness and source binding;
- P50/P95 latency budget.

The comparison must include dense-only, sparse-only, graph-only, RRF, and the
reranker candidate on the same qrels fingerprint. If no candidate wins the
primary metric without violating a guardrail, retain the current version and
record the failed A/B rather than tuning to one favorable metric.

## Implemented in the current runtime

- `DatabaseKnowledgeSearchPort` now collects Dense, PostgreSQL FTS (BM25-like
  `ts_rank_cd`) and Neo4j candidates independently, with a Top-50 internal
  pool and a public Top-10 boundary.
- `merge_and_rerank` uses query-type-aware three-way RRF (`k=60`), then a
  deterministic listwise feature over the RRF Top-50. An optional HTTP
  cross-encoder can be enabled with `MICO_RERANKER_*`; provider failures fail
  closed and cannot add evidence.
- Top-10 selection limits same-document near-duplicates, keeps a graph path for
  multi-hop queries when one is available, prefers multiple documents for
  synthesis, and defers speculative/conflicted paths behind supported chunks.
- Existing stores can be migrated with the nullable chunk `embedding` column;
  `MICO_KNOWLEDGE_CHUNK_EMBEDDINGS_ENABLED=true` enables explicit, provider-costed
  chunk backfill during ingestion. Until then Dense falls back to document
  vectors and remains clearly labeled.
- Query embeddings are distinct from stored document embeddings: a new text
  query needs one same-dimension vector to execute cosine search. The database
  and local ports support `MICO_KNOWLEDGE_QUERY_EMBED_CACHE_PATH`, a hash-keyed
  cross-process cache that stores no raw query text. The graded evaluator enables
  this cache by default, and quota/429 responses fail fast once; Hybrid keeps
  the surviving Sparse/Graph branches available instead of retrying the provider.

## Required implementation order

1. Chunk-level dense and PostgreSQL full-text candidate pools, each Top-50;
2. Graph direct-edge and bounded multi-hop Top-50;
3. RRF over branch ranks, with weights selected only from approved qrels;
4. Cross-encoder/listwise reranker over the RRF Top-50;
5. constrained Top-10 selection and versioned rollout.

## Chunk-v2 implementation status (2026-08-27)

The existing `medical_chunks.jsonl` remains the immutable chunk-v1 baseline
(3,617 chunks / 65 papers).  It is already a chunked corpus; the historical
dense asset is the part that is paper-level (65 vectors, with the full text
truncated to 12,000 characters per paper).  No v1 file is overwritten by the
following assets.

Three structure-aware variants were generated from the same 65-paper source:

| variant | target / min / max tokens | chunks | below min | above max |
| --- | ---: | ---: | ---: | ---: |
| v2-small | 256 / 100 / 340 | 2,305 | 97 | 0 |
| v2-medium | 512 / 150 / 650 | 1,308 | 40 | 0 |
| v2-large | 768 / 220 / 960 | 953 | 59 | 0 |

The lower-than-v1 count is intentional: v2 removes boilerplate sections and
merges complete paragraphs instead of producing overlapping character windows.
Short tails are retained when they can be merged without crossing the max
bound; captions/tables are retained even when shorter than the normal minimum.
The low-minimum rows are therefore explicit short tails or metadata-like
records, not silent truncation.  All three assets pass the independent
validator: 65 papers, unique IDs, valid offsets, complete embedding headers,
and zero max-token violations.

Each row contains `chunkVersion`, `variant`, `chunkId`, PMCID/PMID, title/topic,
section/subsection, `chunkType`, paragraph and sentence ranges, character
offsets, `offsetBase`, source URL, source `text`, and an `embeddingText` that
prepends title/topic/section/subsection context.  Offsets are explicitly
relative to normalized subsection text; they are not byte offsets into the
original PDF/XML.

The old v4 graph was built from chunk-v1 and is kept intact.  Separate local
Graph assets, `fulltext-provenance-graphrag-v4-v2m`, `-r2` and current `-r3`
variant, were built from v2-medium and point at
`medical_chunks_v2_medium.jsonl`; they are `review_pending` and have not been
published to Neo4j.  Their build is source-bound but not a claim of improved
retrieval quality.  The current `-r3` graph contains 1,308 chunk nodes, 1,219
accepted semantic relations, 97 conflict-review relations, and 379 rejected
self-loop records; these quality counts must be reviewed before publication.

Chunk-level Gemini construction is implemented separately in
`build_medical_chunk_embedding_index.py`.  It is provider-free by default:
only an explicit `--execute` can make calls, output names are versioned, and
`--resume` reuses a vector only when the input fingerprint is unchanged.  A
dry-run over v2-medium currently reports 1,297 pending provider calls and made
zero calls.  Therefore no new Gemini quota was consumed by chunk-v2 generation.
