# Decision SFT v1 Full Dynamic E2E v1

This is the post-training canary contract for replacing the Scientific Policy
provider with the selected Decision SFT v1 adapter.  It is deliberately a
separate artifact from the frozen SFT dataset and from the older `p2j4`
materialization task set.

## Provider roles

```text
Gemini Task Understanding
        ↓
Qwen3-8B + Decision SFT v1 adapter (Scientific Policy)
        ↓
Gemini Materializer (QueryPlan / AnalysisPlan)
        ↓
real Java Tool → real MySQL / real typed Python analysis
        ↓
Observation Builder → ScientificDecisionState → next Qwen decision
```

The policy sees only the six-block `ScientificPolicyInput`.  The canary does
not add a strategy hint, fixed action sequence, QueryPlan, AnalysisPlan, raw
rows, or identity value.  Runtime hard availability and plan validation remain
deterministic infrastructure; Runtime never chooses a replacement scientific
Action.

## Task set

The frozen source is
`evals/decision_sft_v1_full_dynamic_e2e_task_set_v1.json`.  It contains five
coverage tasks:

* a real T2D/Healthy comparison with age and evidence objectives;
* a real comparison with a requested stratified follow-up;
* an evidence/GraphRAG task, skipped explicitly when a real knowledge port is
  not configured;
* a real availability/insufficient-data boundary for a missing project
  dimension; and
* a controlled finish boundary.  The last item is never presented as a real
  database trace and is always `training_eligible=false`.

Every task has a six-action budget.  A model may choose any legal action and
the trajectory remains open; the task file describes obligations and skip
semantics, not an oracle for the next action.

## Offline preparation (safe default)

```text
python scripts/prepare_decision_sft_v1_full_dynamic_e2e.py
```

or:

```text
python scripts/run_decision_sft_v1_full_dynamic_e2e.py
```

Both commands only validate and snapshot the task set, verify the frozen SFT
manifest, and write:

* `artifacts/decision_sft_v1_full_dynamic_e2e_v1/task_set_snapshot.json`;
* `preparation_manifest.json`;
* `acceptance_gates.json`; and
* `task_set_sha256.txt`.

They make zero model calls and do not open a Java/MySQL/A100 connection.

## Explicit live run

Only after an operator has started the services and confirmed the Qwen endpoint:

```text
python scripts/run_decision_sft_v1_full_dynamic_e2e.py --live
```

The live runner performs Java/MySQL preflight first.  A failed preflight stops
before any model token is read.  It then uses the configured Qwen SFT policy
endpoint, Gemini Task Understanding/Materializer, the real Java port, and the
existing typed runtime.  Deterministic Scientific Action fallback and
deterministic Materializer fallback are disabled.  The runner never starts or
stops an A100 and never runs training or DPO.

Each real task writes provider requests/responses, policy-safe `state_s*.json`
snapshots, Java-call summaries, materializer/analysis audits, and `trace.json`.
Controlled or unavailable tasks are written as explicit skip records.  Raw
rows and credentials are not copied into policy snapshots.

## Live acceptance gates

The final live report may be promoted only when all applicable gates pass:

* Java Tool and MySQL preflight passed before model calls;
* Qwen SFT HTTP request/response obeyed `ScientificPolicyInput` and the closed
  `DecisionPolicyOutput` contract;
* every selected Action was supplied by Qwen and was hard-available;
* Gemini materialized the exact selected Action family with no deterministic
  fallback;
* at least one real QueryPlan reached Java/MySQL and one real AnalysisPlan
  reached a typed operator;
* the resulting Observation changed the six-block State used by the next
  policy request;
* all rounds stayed within six Actions and recorded policy/materializer
  provenance; and
* no DPO, legacy strategy guard, or scientific-action fallback was used.

The preparation report intentionally keeps these service/model gates as
`PENDING_LIVE`; therefore `FULL_E2E_READY=false` and no A100 is authorized by
the offline step.
