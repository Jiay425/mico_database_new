# P2G quality audit and fusion debug v1

This document records work performed while the final 307 Gemini vectors are
blocked.  It is not a substitute for the final qrels evaluation.

## RRF displacement diagnosis

The frozen 18-case smoke list was replayed with the controlled 1,001-chunk
corpus.  Per-gold-chunk rank movement is in
[`p2g-rrf-displacement-v1.json`](../../evals/p2g-rrf-displacement-v1.json).

The result is unambiguous for this diagnostic boundary:

```text
cases: 18
gold chunks present in branches: 48
weighted RRF moved gold downward: 48 movements (36 by >=3 ranks)
mean nDCG@10 delta (weighted - standard): -0.04020103
cases with a first-hit@10 loss: 0
```

The absence of a case-level hit loss does not make weighted RRF safe: several
core chunks leave the top ten while another gold chunk remains, which lowers
nDCG and evidence precision.  Relation cases are the clearest failure mode
(mean nDCG delta `-0.0817115`).  Therefore `rrf-standard-v1` stays the fixed
fusion baseline; query-type weights are not promoted or tuned on these cases.
The retrieval-plan default is now also `rrf-standard-v1`; weighted/listwise
stages are opt-in for controlled experiments until qrels calibration.

## Graph extraction quality fix

The v2-medium r3 staging graph had 379 rejected semantic records, all rejected
as `SELF_LOOP_RELATION`.  Inspection showed that the extractor treated
adjacent aliases such as `lipopolysaccharide (LPS)` as two mentions of the same
node and paired them into a self-loop.

The extractor now:

1. compacts overlapping/adjacent aliases for the same node;
2. forbids self-loop semantic pairs and searches the next deterministic
   distinct mention pair;
3. recognizes high-frequency `TLR4`/`toll-like receptor 4` aliases.

The isolated rebuild is
`fulltext-provenance-graphrag-v4-v2m-r4-selfloopfix`.  Its extractor is now
frozen: `extractorStatus=frozen`; only the 93 review-required relations and
the unresolved self-loop audit may change its publication decision.  No new
alias or chunking rule is being added from this smoke data.

| Graph build | Semantic edges | Review required | Rejected records | Self-loops |
|---|---:|---:|---:|---:|
| v2m-r3 | 1,219 | 97 | 379 | 379 rejected |
| v2m-r4-selfloopfix | 1,292 | 93 | 0 | 0 |

All r4 semantic edges retain an evidence chunk and evidence text.  This is a
quality/recall repair, not a claim that r4 has higher retrieval nDCG: the
smoke gold itself is derived from the old graph and must not be used to claim
that result.

The relation audit contract samples 100 accepted, all 93 review-required, and
up to 100 rejected records (the r4 rejected bucket is correctly empty):
[`p2g-graph-quality-audit-v2-selfloopfix.json`](../../evals/p2g-graph-quality-audit-v2-selfloopfix.json).

The previous r3 rejected sidecar is retained for false-negative auditing:
[`p2g-graph-rejected-v4-v2m-r3.jsonl`](../../evals/p2g-graph-rejected-v4-v2m-r3.jsonl).

A conservative matched audit on the same evidence span finds 116/379 (`30.60686%`)
old self-loops with a distinct r4 replacement; 263 have no deterministic
replacement and remain sentence-level review candidates.  This is not a
false-negative rate yet—the replacement audit still needs a judge to decide
whether the new pair is scientifically correct:
[`p2g-graph-selfloop-recovery-audit-v1.json`](../../evals/p2g-graph-selfloop-recovery-audit-v1.json).

The provider-free sentence triage now adds a second, non-mutating audit layer:
it detects conservative acronym/symbol and taxon-like surface candidates that
the controlled extractor did not normalize.  Among the 263 unresolved spans,
the current heuristic breakdown is:

```text
correct_drop:                  166
entity_normalization_error:     97 candidates
missed_distinct_relation:        0
relation_trigger_error:          0
ambiguous:                       0
```

The 97 `entity_normalization_error` records are *candidates for review*, not
accepted graph relations.  The detector found 101 unresolved spans with at
least two distinct surface candidates and 148 with at least one unrecognized
candidate; repeated aliases and ordinary hyphenated modifiers are filtered.
The full records retain character spans, detector type, relation-trigger hits,
and a pending judge decision in
[`p2g-unresolved-selfloop-triage-v2-surface-audit.json`](../../evals/p2g-unresolved-selfloop-triage-v2-surface-audit.json).

## Independent query freeze

The provider-free query freeze contains 100 generation packets with the target
distribution:

```text
direct evidence: 20
entity-relation: 20
mechanism: 20
multi-hop: 15
cross-paper synthesis: 15
conflicting/negative: 10
```

It reads only chunk-v2 title/text/section metadata.  It does not read GraphRAG
edges or old qrels.  Questions remain `null` and `pending_llm_generation`
until a judge/generator can read the packets; this prevents fabricated “gold”
labels during the embedding quota outage:
[`p2g-independent-query-freeze-v1.json`](../../evals/p2g-independent-query-freeze-v1.json).

Provider-free materialization now supplies 100 conservative heuristic question
drafts.  The audit rewrote 62 weak or duplicate-template assignments, yielding
100/100 automatic passes, 100 unique questions, and zero duplicate questions.
This is now frozen as `query-set-v1`, the fixed independent evaluation question
set.  It is not LLM-judged qrels yet: `qrels-v1` is explicitly marked
`not_built` and the file contains no relevance labels:
[`p2g-independent-query-set-v1-frozen.json`](../../evals/p2g-independent-query-set-v1-frozen.json).

## Next release gate

Do not publish r4 or train a reranker from the smoke set.  The release order is:

1. finish the 307 chunk embeddings and verify 1,308/1,308 coverage;
2. keep `query-set-v1` immutable and run it through pooled
   LLM relevance judging with evidence spans;
3. replay Dense, Sparse, Graph, standard RRF, weighted-RRF, and reranker
   variants on fixed train/validation/test qrels;
4. approve a graph version only after the relation audit closes and the
   manifest records graph/chunk/embedding/fusion versions.
