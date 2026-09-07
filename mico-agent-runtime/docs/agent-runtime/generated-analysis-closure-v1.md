# Generated Analysis Closure v1

## Ownership boundary

`ScientificDecisionState -> Qwen Policy` selects only one fixed Scientific
Action. The Materializer emits an `AnalysisPlan` with semantic field IDs.
`AnalysisCapabilityRegistry` then decides the execution channel before either
executor runs:

| Capability decision | Executor | Fallback rule |
| --- | --- | --- |
| `SUPPORTED_TYPED` | approved typed operator | never switches to generated code after failure |
| `SUPPORTED_GENERATED` | Runtime-bound `GeneratedAnalysisProgram` sandbox | generated program is selected before execution |
| `UNSUPPORTED` | none | fail closed |

The Materializer cannot author `execution_mode`, SQL, filesystem paths,
network access, package installation, or database connections.

## Action capability matrix

| Scientific Action | Typed | Generated | Notes |
| --- | --- | --- | --- |
| `compare_groups` | Standard two-group methods | Non-standard approved method/metric shapes | Generated output is exploratory in v1. |
| `analyze_projection` | No | Yes | Projection has no registered typed semantics. |
| `stratified_analysis` | Categorical or explicit numeric specification | Non-standard method; legacy numeric plan without a typed numeric specification | One sample-level projection only. |
| `adjust_confounders` | Standard linear adjustment | Approved custom method family | Covariates must remain catalog-verified. |
| `cross_project_validate` | Standard validation | Non-standard validation method | Requires independent project dimension. |
| `cross_disease_validate` | No current typed operator | Yes, after validation-dimension checks | Requires an independent disease dimension. |
| `inspect_cohort`, `execute_read_query`, `retrieve_evidence`, `finish` | N/A | N/A | They are not Python analysis actions. |

## Program and sandbox contract

Runtime binds `GeneratedAnalysisProgram` to the approved action, analysis
goal, source observation IDs, selected safe columns, row/output caps, and
timeout plus memory cap. The model contributes only Python source. The sandbox is a separate
isolated Python process with an empty/minimal environment and safe builtins.
It disallows imports, filesystem I/O, network, subprocesses, shell access,
database access, `eval`, `exec`, `open`, and private names.

The v1 language subset is intentionally stricter than the permitted
scientific-package allowlist: it exposes no third-party package imports until
they can be isolated with the same process/resource guarantees. This is a
security restriction, not a typed-to-generated fallback.

### `generated-code-contract-v1`

The runtime exports a versioned generator-facing contract directly from the
same AST allowlist used by the sandbox. It contains the input record API and
allowed columns, permitted builtins and AST nodes, forbidden method calls and
syntax, the exact result/output schema, a safe result skeleton, and resource
limits. In particular, record access is `row["column"]`; all methods,
including `row.get(...)` and `list.append(...)`, are forbidden. Function
definitions and `return` are also outside the v1 language.

Contract repairs contain only a stable mechanical error and the corresponding
allowed replacement form. Provider failures (for example HTTP 402) are not
repairs and never trigger another model request.

### Input identity boundary

`input_bindings` distinguishes the Runtime observation object from data rows:
an `observation_id` is provenance only and must never be used as a sample-key
or row-filter value. A sample key may be read only through the explicitly
bound opaque `sample_key_column`; it is never derived from an observation ID.
The sandbox rejects a literal comparison of that bound sample-key column to an
observation ID as `INVALID_IDENTITY_BINDING`. A non-empty input yielding zero
used rows is reported as
`ZERO_ROW_SELECTION_SUSPECTED_IDENTITY_MISMATCH`, rather than being hidden as
an ordinary empty analysis result.

Linux production additionally applies process address-space and CPU limits.
Windows development runs retain the same isolated environment, timeout, row,
and output bounds; they do not inherit credentials or user environment.

## Result and trace validation

A program must return the shared bounded result contract. Runtime validates
declared outputs, non-empty metrics, finite numeric values, sample/row bounds,
and metric-schema requirements. Generated results can be
`scientific_result_valid=true` but remain
`scientific_conclusion_eligible=false` by default, so they are explicitly
exploratory until an action-specific conclusion validator is introduced.

Runtime trace data retains the capability decision plus program and code
hashes. The Runtime-only audit copy contains the approved generated program;
it is never put in `ScientificDecisionState` or exposed to Qwen.

## Verification layers

Layer A is fully local: fixed programs exercise success, strict output checks,
network/file rejection, timeout, bounded column projection, typed routing, and
no typed-failure fallback. Layer B (real model-authored code) is intentionally
not invoked by this closure and requires a separate approval after Layer A is
accepted.
