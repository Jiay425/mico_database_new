# P2G partial-index diagnostic report v1

## Boundary

This report is a development diagnostic only.  The medium corpus contains
1,308 chunks, while 1,001 have a `gemini-embedding-2` vector.  The remaining
307 chunks are excluded from the controlled candidate universe; they are not
treated as irrelevant judgments.

```text
corpus: chunk-v2 / medium
embedded: 1001 / 1308
coverage: 76.529052%
status: partial-index development set
```

## Coverage audit

The missing vectors are not random.  They are concentrated in the source
order: 13 papers have zero embedded chunks and one paper is partially covered.
The other 51 papers are fully covered.  Topic coverage ranges from 59.5% to
100%, so the partial Dense result is not suitable for cross-topic comparison.

The machine-readable audit is
[`p2g-embedding-coverage-v1.json`](../../evals/p2g-embedding-coverage-v1.json).

## Controlled corpus

`dense-partial-corpus-v1` contains exactly the 1,001 chunk IDs that have a
valid 3,072-dimensional vector.  Sparse and graph diagnostics use the same
IDs.  The graph projection retains only edges whose `evidenceChunkId` belongs
to that set:

```text
chunks: 1001
embeddings: 1001
graph nodes: 2005
graph edges: 8471
```

Manifest and assets:
[`p2g-controlled-corpus-v1.json`](../../evals/partial-corpus-v1/p2g-controlled-corpus-v1.json)

## Provider-free smoke diagnostic

The 18 deterministic v2-medium smoke cases were evaluated without issuing new
Gemini requests.  Dense was marked unavailable because the query-vector cache
contains only one entry and none of the 18 case queries had a cache hit.
Sparse and graph use deterministic local proxies over the controlled assets;
their numbers are diagnostic, not Neo4j/pgvector production metrics.

| Branch | Recall@50 | nDCG@10 | MRR@10 | Hit@10 |
|---|---:|---:|---:|---:|
| Sparse | 0.6205 | 0.4304 | 0.7395 | 0.8333 |
| Graph proxy | 0.6008 | 0.4260 | 0.6483 | 0.9444 |
| Standard RRF (available branches) | 0.6982 | 0.5135 | 0.7565 | 1.0000 |
| Query-weighted RRF (available branches) | 0.5679 | 0.4733 | 0.7426 | 1.0000 |

The full per-case ranking expansion is
[`p2g-partial-diagnostic-v1.json`](../../evals/p2g-partial-diagnostic-v1.json).
Because Dense was unavailable, this does **not** establish a final Dense or
three-way Hybrid result.  It does show that adding a weak branch through
hand-tuned weights can hurt ranking, while the clean standard-RRF baseline is
stable on this controlled smoke set.

## Fusion implementation

`RetrievalPlan` now accepts a `fusion_version`.  `rrf-standard-v1` uses equal
branch weights and skips the deterministic listwise stage, giving a clean RRF
ablation.  The existing `rrf-v2-listwise-v1` remains the production default.
Both paths keep branch provenance, graph paths, and evidence constraints.

## Next gate

The remaining 307 vectors must be resumed before any full-corpus model
selection.  After completion, run the same cases and compare the partial
diagnostic against the full 1,308-chunk result, then pool candidates for the
reviewed qrels set.
