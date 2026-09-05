# P2-J4 Dynamic Materialization Calibration Contract v1

Status: `LOCAL_DYNAMIC_RUNTIME_READY__REAL_SERVICE_CANARY_REQUIRED`

This document is the mandatory calibration contract for every Dynamic Runtime
E2E run. It is intentionally separate from the frozen Decision SFT/DPO assets.
It contains no secret, raw user question, SQL text, Python source, filter value,
or data row.

The Decision SFT v5 and DPO v4 Controlled r2 model assets are already trained
and frozen for this experiment. Dynamic Materialization is an inference-time
runtime experiment and must not retrain DPO. The `trainingStarted=false` field
in the DPO freeze manifest describes the freeze/reference-logprob stage only;
the completed training evidence is the DPO run metrics, adapter, generation
smoke, Test70 and OOD30 outputs.

## 1. Fixed responsibility boundary

```text
Decision Policy (Qwen SFT/DPO)
    redacted policy state -> one high-level Action

Action Materializer (Research Planner)
    transient research goal + state + final Action + catalog
    -> QueryPlan / AnalysisPlan

Runtime Harness
    guardrail + bounded retry + provenance + failure classification

Java / Python / Retrieval executors
    QueryPlan -> prepared SQL -> MySQL
    AnalysisPlan -> approved operator or sandboxed Python
    Runtime-constructed RetrievalPlan -> bounded vector / graph retrieval
```

The Decision Policy never produces SQL, query join keys, raw Python, database
locations, identifiers, or credentials. Training assets remain strictly
`State -> Action` and are not modified by this contract.

## 2. QueryPlan v1

`execute_read_query` materialization must produce a typed QueryPlan rather than
raw SQL. Java is the only SQL compiler and executor.

```json
{
  "root_entity": "sample",
  "relation_path": ["sample_to_abundance", "sample_to_metadata"],
  "select_fields": ["metadata.project"],
  "aggregations": [{"field": "abundance.value", "op": "mean"}],
  "filters": [{"field": "sample.disease", "operator": "eq", "value": "<transient>"}],
  "group_by": ["metadata.project"],
  "limit": 100
}
```

The value placeholder above is illustrative only and must never be persisted in
Trace, DPO/SFT data, reports, or calibration output.

### 2.1 Java compiler rules

- `root_entity`, field IDs, aggregation operations, filter operators, and
  relation IDs are closed catalog values. Field IDs are stable semantic names
  such as `sample.disease` or `abundance.value`; Java maps them to physical
  table and column names. The Materializer never emits physical identifiers.
- `relation_path` is an ordered list of catalog relation IDs. Java validates
  every hop from `root_entity` in the declared order.
- A relation selection contains only its catalog relation ID. The model never
  supplies `ON`, join type, left/right keys, table names, or SQL fragments.
- The catalog owns each relation's endpoints, keys, join type, cardinality, and
  whether it is enabled for research reads.
- The relation path is connected from `root_entity`, has at most two hops, has
  no duplicate relation/entity, and cannot form a cycle.
- Java rejects unknown, sensitive, non-displayable, non-filterable,
  non-groupable, or non-aggregatable fields as appropriate.
- Filter values use `PreparedStatement` bindings. Identifiers are compiler
  selected from the catalog, never bound or interpolated from model output.
- v1 rejects arbitrary SQL, arbitrary `ON`, subqueries, CTEs, window functions,
  free-form functions, cross-catalog references, and implicit Cartesian joins.
- Java enforces `1 <= limit <= 1000`, read-only execution, timeout, transient
  snapshots, and bounded response rows.
- After a successful Dynamic read, the next planner context may contain only
  the read's semantic `queryPlanFields` and ordered `queryPlanRelationPath`.
  This lets a later read request a fresh projection without exposing SQL,
  filter values, or rows. An identical compiled plan remains rejected as an
  action repeat.
- For aggregate plans, the compiler checks catalog cardinality metadata and
  rejects paths that can introduce an unreviewed many-to-many amplification.
- The historical raw-SQL compatibility field remains only for legacy intent
  fixtures; Dynamic Scientific Runtime rejects that representation with
  `DYNAMIC_QUERY_PLAN_REQUIRED`.
- `inspect_cohort` is Runtime-owned metadata-first behavior. It does not
  require a DeepSeek Materializer call; it may use the Java schema/catalog
  inspection boundary before the first model-materialized data query.
- The Materializer request receives only this semantic projection of the
  catalog: entity IDs, field IDs, field capabilities, relation IDs,
  relationship status, cardinality, and query rules. Physical table names,
  physical column names, join keys, and catalog descriptions are not included
  in the model message.

## 3. AnalysisPlan v1

Analysis actions materialize typed analysis parameters first:

```text
analysis_type: group_comparison | stratified_comparison |
               confounder_adjustment | projection |
               cross_project_validation | cross_disease_validation
outcome field
group field
covariates
stratification fields
requested metrics
source observation IDs
```

The Python runtime first dispatches to an approved bounded operator for the
typed plan. The v1 operator covers count, mean, median, effect size, and the
bounded two-sided normal-approximation `p_value`/95% confidence-interval width
metrics without a free-form code call. A sandboxed generated Python
implementation is permitted only for an explicitly unsupported metric/plan shape and remains subject to AST, import,
network, filesystem, timeout, row-bound, and structured-result constraints.
The cross-project and cross-disease types require an outcome and the matching
semantic group field (`*.project` or `*.disease`; the Catalog's `disease.name`
is also valid for cross-disease). A single validated Java tabular observation
is sufficient when its rows contain the requested group field and numeric
outcome; the bounded operator compares the groups present in that observation.
Multiple source observations remain supported when a task intentionally
combines independent bounded reads. The Runtime must never manufacture a
second read merely to satisfy an observation-count rule.

Retries may return opaque rejection codes to the Materializer. After the fixed
retry budget is exhausted, the result is `ANALYSIS_GENERATION_FAILED`; no
row-count or deterministic-analysis fallback may be recorded as a successful
dynamic analysis.

The Runtime may perform a lossless schema normalization for a bounded model
response (for example, a scalar list field or the legacy aggregation spelling
`function` to the closed `op` enum). The normalized plan is still validated by
the same Pydantic and Java catalog gates and the decision receives
`DYNAMIC_MATERIALIZER_SCHEMA_NORMALIZED` provenance. Unknown fields, unknown
operations, invalid grouping, non-numeric outcomes, and unsupported semantics
remain failures.

For cross validation, Runtime derives a closed required semantic-field set
from the current catalog and redacted goal. The read must include the matching
project/disease grouping field in `group_by` and a verified aggregatable
numeric outcome (not merely `count`). This makes the Java result group-complete
even when the underlying table's default row order is dominated by one group.
When the model has already selected those semantic fields but omits this
group-complete shape, the targeted normalization emits
`DYNAMIC_CROSS_QUERY_GROUP_NORMALIZED`; this is a cross-validation contract
repair, not a generic SQL template or a hidden filter/value.
The typed operator also requires at least two observed group values; a single
group is a bounded data insufficiency and fails closed rather than being
reported as a successful cross-boundary result.

The successful result identifies the execution path as either
`typed-analysis-operator-v1` or `sandbox-python-v1`; the latter is never a
Runtime-owned deterministic template.

The typed-operator-to-sandbox handoff is closed by the shared
`TYPED_ANALYSIS_SANDBOX_FALLBACK_CODES` set. It includes an unsupported typed
shape and a missing grouping shape; both are execution-shape failures that
may be repaired by the sandboxed Python Materializer. Missing numeric
outcomes and insufficient cross-group coverage are not in that set and remain
fail-closed.

`inspect_cohort` is a Runtime-owned metadata probe and is not counted as the
tabular observation required by a Dynamic AnalysisPlan. A policy proposal for
analysis immediately after inspection is therefore repaired to a bounded
`execute_read_query` when that action is approved.

At execution time, if a model-supplied analysis action references the
metadata-only inspection observation, Dynamic Runtime filters that reference
to the explicitly available Java `execute_read_query` observation (or binds to
the newest such observation when the model supplied no tabular reference).
The repair is recorded as `DYNAMIC_ANALYSIS_NON_TABULAR_OBSERVATION_FILTERED`
or `DYNAMIC_ANALYSIS_INSPECTION_REBOUND`; Runtime never concatenates metadata
probe rows with the analysis projection.

Cross-project and cross-disease validation is intentionally stricter than a
single-group comparison: it needs at least two observed project/disease group
values and a numeric outcome available in the returned projection. Those
groups may be present in one validated, group-complete Java observation; two
Java observations are not required unless the task explicitly asks to combine
independent reads or snapshots. A generic task that does not identify a
numeric outcome may therefore fail closed with
`ANALYSIS_TYPED_OPERATOR_NO_NUMERIC_OUTCOME`; this is not converted into an
age/dimension surrogate or a row-count PASS. Likewise, a valid read with only
one observed group is `ANALYSIS_CROSS_VALIDATION_GROUP_COVERAGE_REQUIRED`, not
a materializer or execution failure.

### 3.1 Returned-column capability boundary

Java returns compiler-owned aliases such as `a_sample_age` or
`a_abundance_value_mean`, not physical SQL names. Before a Dynamic
AnalysisPlan is materialized, the Runtime maps those aliases back to verified,
non-sensitive Catalog field IDs. Only fields actually present in the returned
projection are offered to the AnalysisPlan Materializer, and only the
corresponding scalar preview columns are shown to it.

The Runtime also records an opaque capability code for a Java read:

```text
numeric_outcome_available
numeric_outcome_missing
```

If an analysis policy proposal follows a Java read whose bounded result has no
numeric Catalog field, the Harness preserves the raw policy decision and
repairs the next high-level step to `execute_read_query`. The Materializer is
then given the opaque `ANALYSIS_NUMERIC_OUTCOME_REQUIRED` feedback and must
produce a fresh Catalog-bounded QueryPlan. No fixed field, SQL fragment, or
data value is inserted by Runtime, and a row-count-only fallback is never
accepted as a successful Dynamic analysis.

## 4. Guardrail semantics

Guardrails may correct a high-level Action when it violates a closed runtime
obligation. They do not create SQL or Python templates.

```text
raw_action       = Qwen policy proposal
final_action     = action after an explicit guard repair, if any
materialization  = Research Planner creates parameters for final_action
```

Every repair is retained as a policy-quality signal, not hidden as a success:

```text
raw_action
final_action
repair_codes
action_planner_origin
materializer_origin
```

For the metadata-first `inspect_cohort` step, `materializer_origin` is
`runtime_owned`: Qwen still supplies the high-level policy decision, but the
typed inspection plan is built from the Java semantic Catalog and no DeepSeek
Materializer call is made. This is not a query-template fallback.

An E2E task can complete after a justified guard repair. Policy quality and
task completion must be reported separately.

## 5. Required redacted provenance

Every new Dynamic Runtime E2E trace must include only the following material
provenance fields in addition to the existing redacted trace fields:

```text
task_kind
goal_code
raw_action
final_action
repair_codes
action_planner_origin
materializer_origin
query_plan_hash
query_plan_validation_status
analysis_plan_hash
analysis_execution_status
catalog_hash
```

Never persist original questions, raw SQL, Python source, filter values,
preview rows, identifiers, database URLs, tokens, or credentials.

## 6. Calibration before every Dynamic E2E run

The following calibration must pass before any real case is started:

1. Verify the exact source revision, QueryPlan/AnalysisPlan schema versions,
   catalog hash, Qwen base/SFT/DPO adapter hashes, and Materializer model ID.
2. Verify Java and MySQL health locally. The database remains local and is not
   exposed to the A100 host.
3. Verify Qwen policy service health through its SSH tunnel and that its served
   adapter name/hash matches the intended SFT or DPO arm.
4. Validate the fixed Dynamic E2E task set and confirm the output path is new.
   Historical Runtime20 artifacts must never be overwritten.
5. Run no-API contract tests: QueryPlan compiler acceptance/rejection,
   prepared-parameter binding, relation-path checks, AnalysisPlan dispatch,
   trace redaction, and provenance audit.
6. Freeze one calibration manifest containing only versions, hashes, boolean
   service health, retry limits, temperature, and task-set hash.

The local helper is `evals/p2j4_calibrate_dynamic_runtime.py`. It writes only
the secret-free manifest and never prints environment values or contacts a
model/database.

The fixed local inputs for that helper are:

```text
evals/p2j4_dynamic_e2e_task_set_v1.json
evals/p2j4_dynamic_e2e_materializer_config_v1.json
evals/p2j4_freeze_dynamic_e2e_contract.py
```

The task set is immutable and contains exactly 3 canaries plus 9 paired E2E
tasks. It is an execution-contract set, not a training set and not a source
for performance claims. Its manifest records the canonical task-set hash,
oracle plan-kind counts, and the fact that no real run has started. The
Materializer configuration is also immutable: DeepSeek `deepseek-v4-flash`,
temperature `0`, three retries, paired-experiment-only cache scope, and no
reuse across different final Actions. The prompt hash is derived from the
static prompt literals in `HttpResearchPlannerPort`; the runtime catalog and
database snapshot are separate hashes and must be supplied at real-run time.

Every subsequent calibration must revalidate these files and fail closed if
the task set, Materializer configuration, or current prompt hash changes. A
passing local manifest therefore proves contract consistency only; it does
not prove that Java, MySQL, the Qwen policy service, or DeepSeek has completed
a real canary.

Any failed calibration is `CALIBRATION_FAILED`; do not start the canary.

### Retry semantics

Each failure carries only an opaque, allow-listed feedback code into the next
planning attempt. The Runtime Analysis Harness has at most three typed-plan
attempts and, when the shared fallback set permits it, at most three sandbox
code attempts. Java semantic-read replanning also has at most three attempts;
an infrastructure/transport failure is not blindly retried. Separately, each
HTTP planner method has `MAX_MODEL_RETRIES=3`, meaning the initial request plus
up to three provider retries, with validation feedback appended to the next
request. Therefore “three retries” is not a global three-HTTP-request cap:
nested Harness and provider budgets can result in more requests, while every
individual loop remains bounded and a successful attempt stops that loop
immediately. This distinction must be reported in pressure-eval results.

The local code path also has a separate, non-benchmark Materializer smoke
check. It may use a fixed test policy to verify DeepSeek materialization and
local Java/MySQL/Python execution while the remote Qwen service is offline.
Its result must not be merged into the frozen 3-canary or 9-task paired
metrics.

### 6.1 Local Materializer pressure evaluation

The separate local pressure runner is:

```text
evals/p2j4_dynamic_materialization_eval_v1.py
```

It tests a fixed high-level Action supplied by the harness; it is not a Qwen
policy evaluation and it does not create DPO data. The default 200-case shape
contains 80 `execute_read_query` cases and 120 AnalysisPlan cases covering
single-entity reads, catalog relation paths, aggregation/grouping,
comparison, stratification, confounder adjustment, projection, and the two
cross-validation actions. The worker cap is bounded at five parallel model
requests. Existing `results.jsonl` entries are append-only and are skipped by
case ID, so an interrupted run cannot silently overwrite a completed case.

Default execution uses a transient redacted in-memory analysis fixture. This
measures DeepSeek materialization, semantic validation, typed-operator
dispatch, and the explicitly allowed sandbox retry without repeatedly
scanning MySQL. `--real-execution` is a separate mode: it prepares bounded
Java snapshots and executes QueryPlans through Java/MySQL. Its output must be
reported as a different execution mode. `--retrieval-smoke` separately checks
Runtime-built vector, graph, and hybrid RetrievalPlans against the local
knowledge index.

The pressure runner records only hashes, closed plan summaries, status codes,
repair codes, provider-request counts, and timing. It never persists model
response text, SQL, Python source, filter values, preview rows, credentials,
or database URLs. A deterministic plan can be useful for Runtime compatibility
but is not a Dynamic Materializer success; only `materializerMode=model` may
count toward Materialization Success.

The 2026-08-27 calibration history is interpreted as follows:

```text
14-case calibration (pre-fix): 11/14 reported PASS, but two AnalysisPlan
  sandbox cases were affected by the code transport/AST boundary below;
  this is diagnostic history, not a frozen benchmark result.
105-case r7 partial run: 96 PASS / 9 FAIL before the final corrections;
  it was intentionally stopped before 200 and has no final summary.json.
  It is not a completed 200-case score.
Targeted post-fix analysis-005: model typed plan accepted; typed operator
  failed closed as unsupported shape; sandbox fallback passed.
Post-fix parallel replay of the other six affected analysis cases: 2/6
  recovered through the sandbox fallback and 4/6 still ended with the opaque
  `ANALYSIS_CODE_REJECTED` after the bounded retry budget. These are genuine
  Materializer-quality failures, not permission to add a deterministic code
  template or to count the case as a PASS.
Retrieval smoke: vector 5/5, graph 5/5, hybrid 5/5 results returned.
```

After the failed-case-only recovery loop, the post-retry 200-case pressure
result is:

```text
overall:                         200/200 PASS
QueryPlan:                       80/80 PASS
  model materializer fallback:  0
  contract pass:                80/80
  semantic pass:                80/80
AnalysisPlan:                    120/120 PASS
  typed operator:               86
  sandbox Python fallback:      34
  contract pass:                120/120
  semantic pass:                120/120
deterministic materializer:      0
query/analysis fallback errors:  0
```

The 200 cases were not blindly restarted. The original run completed 170/200;
its 30 failures were submitted to a separate append-only recovery file. The
first recovery raised the merged result to 178/200, the next recovery to
190/200, and the final 10 failures passed after the bounded-code prompt and
feedback refinement. The final view is in
`tmp-runtime-smoke/dynamic-materialization-eval-20260827-r7/post-retry-summary.json`;
the original run and every recovery attempt remain separately auditable.

This 200-case result uses `redacted_in_memory_fixture` execution mode. It
proves DeepSeek materialization, plan contract validation, typed AnalysisPlan
dispatch, sandbox recovery, and failure feedback behavior. It does not claim
80 QueryPlans were executed against Java/MySQL; that is a separate real-
execution evaluation and must be reported separately. The local Retrieval
vector/graph/hybrid smoke is independently recorded above.

### 6.2 Saved QueryPlan real-execution replay

The first real Java/MySQL boundary check replays only the already saved,
redacted QueryPlan summaries:

```text
evals/p2j4_replay_saved_query_plans_real_execution.py
```

It makes no DeepSeek request and never reconstructs a redacted filter value.
From the 80 pressure cases it selected 11 unique saved plan hashes. The
result was:

```text
catalog validation: 11/11
Java/MySQL execution: 11/11 COMPLETED
execution failures: 0
raw SQL/rows/secrets persisted: no
```

The replay covers zero-hop and one-/two-hop Catalog paths, bounded limits,
grouped aggregates, and metadata/sample roots. The saved pressure plans have
no filter fields, so filter execution is not claimed by this result and must
be covered by a separately frozen real-execution case when such a plan is
available. This replay validates Catalog validation, Java compilation/tool
dispatch, MySQL execution, and the redacted response boundary; it does not
yet constitute the full Qwen Loop or the final Observation-construction E2E.
The result is recorded at
`tmp-runtime-smoke/dynamic-materialization-eval-20260827-r7/real-query-execution-replay-20260827.json`.

### 6.3 Real Java Observation to AnalysisPlan dataflow

Six saved AnalysisPlan representatives were rebound to real transient Java
Observations, covering `group_comparison`, `stratified_comparison`,
`confounder_adjustment`, `projection`, `cross_project_validation`, and
`cross_disease_validation`. No DeepSeek request was made during this replay.

```text
Java Observation binding: 6/6
source fields present: 6/6
AnalysisPlan contract: 6/6
typed operator: 4/6
restricted sandbox fixture: 2/6
structured analysis execution: 6/6
result Observation construction/binding: 6/6
failures: 0
```

The two sandbox executions are explicitly labelled integration fixtures: the
saved plans were valid but requested shapes not covered by the typed operator.
They verify that real Java rows can cross the bounded sandbox boundary; they
are not represented as new DeepSeek output. Cross-project and cross-disease
reads use group-complete aggregates and returned 85 and 314 observed groups,
respectively. The merged result is recorded at
`tmp-runtime-smoke/dynamic-materialization-eval-20260827-r7/real-analysis-dataflow-final-20260827.json`;
the two-case recovery is separate at
`tmp-runtime-smoke/dynamic-materialization-eval-20260827-r7/real-analysis-dataflow-retry-20260827.json`.

### 6.4 PreparedStatement filter probe

One targeted equality-filter QueryPlan was sent through the Catalog and local
Java/MySQL compiler. The transient filter value is held only in process:

```text
Catalog validation: passed
Java/MySQL execution: COMPLETED
result classification: OBSERVATION_NONEMPTY
filter value persisted: no
raw SQL persisted: no
```

This validates the filter-binding path without adding the probe to the frozen
200-case pressure score. Its redacted result is at
`tmp-runtime-smoke/dynamic-materialization-eval-20260827-r7/real-query-filter-probe-20260827.json`.

Two local boundary defects found during this calibration were corrected and
covered by no-API tests: generated Python transport normalization now preserves
line structure and indentation, and only the explicitly bounded generated
analysis `code` field may carry structural newlines/tabs through the closed
model. The sandbox AST allowlist adds only side-effect-free `is`/`is not`
comparisons; it still rejects imports, attributes, file/network access,
arbitrary calls, and unbounded execution. Typed-operator rejection and
sandbox recovery are stored as separate fields so a recovered case is not
reported as an execution failure.

## 7. Controlled Materializer variable

For a paired SFT-vs-DPO evaluation, both arms must use the same:

```text
Materializer model/version
Materializer system prompt hash
temperature (target 0)
retry policy
catalog hash
database snapshot/version
task-set hash
```

An optional in-memory, single-run cache may reuse a successful materialization
only for the exact `(goal fingerprint, state signature, final action,
catalog hash, materializer configuration hash)` key. It must not be persisted
with raw plan content and may not substitute a plan for a different Action.

## 8. Run sequence and acceptance criteria

```text
calibration PASS
-> 3 fixed Dynamic E2E canaries
-> audit each result
-> 6-10 fixed Dynamic E2E cases for SFT v5
-> audit and freeze
-> the same fixed cases for DPO v4
-> comparative audit
```

The three canaries cover the three real integration boundaries: a Dynamic
QueryPlan through DeepSeek, the Java compiler and MySQL; a Dynamic AnalysisPlan
through the typed operator or restricted sandbox; and Retrieval/Loop
integration through the Runtime-built RetrievalPlan and the next Observation.
`inspect_cohort` remains Runtime-owned and is covered by local regression, but
does not consume one of the three real Canary slots. Failed cases remain Bad
Case records; only the failed case may be re-run after classifying the fault as
configuration, compiler/contract, Materializer, guardrail, or executor failure.

Report at least:

```text
Raw Policy Accuracy
Guardrail Intervention Rate
Materialization Success Rate
QueryPlan Validation/Execution Success Rate
AnalysisPlan Execution Success Rate
E2E Task Completion Rate
Action Binding Rate
Query Template Fallback Count (must be 0)
Analysis Template Fallback Count (must be 0)
```

`Raw Policy Accuracy` must support multiple legal actions. Each E2E oracle
therefore declares `allowed_actions` and may declare an ordered
`preferred_actions` list; it must not force a single Gold Action when multiple
actions satisfy the same state obligation. Runtime-owned `inspect_cohort` and
an explicit high-level guard repair are reported separately and are not counted
as a query or analysis template fallback. An accepted high-level guard repair
does not make the E2E case a failure: preserve `raw_action` and `final_action`
separately and score raw policy quality, guard intervention and E2E completion
independently. Do not require `raw_action == final_action` for every case.

## 9. Frozen Dynamic E2E contract

The current frozen contract manifest is generated locally at:

```text
tmp-runtime-smoke/dynamic-e2e-contract-manifest-v2.json
```

It is a no-API artifact. The current task set has 12 tasks, 3 canaries and 9
paired tasks. The oracle distribution is 2 query plans, 7 analysis plans, 2
Runtime-built retrieval plans and 1 finish boundary. Nine tasks require the
model Materializer; three are Runtime-owned (retrieval and finish boundaries,
as declared by the oracle). The Runtime-owned `inspect_cohort` path is covered
by local regression and is not part of this frozen real-run task set. This distinction is intentional
and must remain separate in later reports.

The task set is frozen before any service result is observed. Canary results
are integration evidence only. The 9 paired tasks are the fixed small E2E
comparison set for SFT v5 and DPO v4 after the user supplies the remote policy
service; they are not a replacement for Test70, OOD30 or Runtime20.

## 10. Asset status

### 10.1 Current project freeze

The current implementation status is frozen as follows:

```text
Decision Policy post-training       DONE
Dynamic Materializer + Harness      DONE
Java/Python/Retrieval verification  DONE
Full Qwen Dynamic Loop              OPTIONAL FINAL SMOKE
```

The completed evidence is sufficient to describe the system as a dynamic
materialization and execution Harness: the model proposes bounded semantic
plans, while Catalog validation, typed execution, sandbox policy, retry, and
provenance convert the proposal into an auditable executable result. A future
Qwen full-loop smoke, if run, is an integration compatibility check only. It
must use three to five frozen cases, must not be reported as a new benchmark,
and must not trigger DPO retraining or a new large-scale Case collection.

The following remain immutable completed or historical assets:

```text
Decision SFT Freeze v2 (865)
Qwen3-8B Decision SFT v5
DPO v4 Controlled Preference Freeze r2 (408), trained adapter and reports
Test70 / OOD30
Runtime20 policy-only artifacts
```

The Dynamic Runtime E2E result is a new experiment and must use a new versioned
task set and output directory. It answers an execution-materialization
question; it does not overwrite or reinterpret policy-only results.
