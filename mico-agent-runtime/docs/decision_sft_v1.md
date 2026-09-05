# Decision-State-native SFT v1

This is an offline data-engineering artifact. The builder does not start an
LLM, call Gemini/Qwen/DeepSeek, connect to an A100, or launch training.

## Frozen model contract

Candidate generation uses the same serving-shaped input as the Scientific
Policy boundary:

```json
{
  "decision_type": "scientific_action",
  "state": {
    "task": {},
    "data_state": {},
    "analysis_state": {},
    "evidence_state": {},
    "progress": {},
    "action_space": {}
  }
}
```

The only model output fields are:

```json
{
  "selected_action": "compare_groups",
  "decision_reason": "grounded in current State facts",
  "alternative_actions": [],
  "stop_reason": null
}
```

The three non-negotiable rules are encoded in the module and validator:

1. **Objective is not Action.** `cross_project_validation` is an objective;
   `cross_project_validate` is the Action.
2. **The selected action must be executable in the current state.** The
   builder recomputes `available_actions` from the real semantic Catalog and
   hard-availability context. A label is accepted only when it is in that
   recomputed list; alternatives obey the same rule.
3. **Training state must match serving state.** Provenance and review metadata
   never enter model JSONL. No legacy flags, raw rows, SQL, plans, or identity
   fields are admitted.

## Builder and output files

Run from `mico-agent-runtime`:

```text
python scripts/build_decision_sft_v1.py
```

The default output is `artifacts/decision_sft_v1/`:

- `all_candidates.jsonl`: the candidate pool, including intentionally invalid
  records retained for audit;
- `approved.jsonl`: records that pass the closed schema, availability,
  rationale, finish, privacy, and dedup checks;
- `train.jsonl`, `validation.jsonl`, `test.jsonl`: model-shaped records only;
- `audit_report.json`: counts, distributions, rejection reasons, examples and
  readiness checks;
- `manifest.json`: frozen input/output contract and split policy;
- `schema.json`: compact schema summary;
- `validation_issues.json`: rejected/duplicate candidate IDs and issue codes.

The current deterministic pool is 543 candidates: 480 unique contract-valid
records, 20 exact duplicates, and 43 intentionally invalid records. The
resulting pre-review split is 360/60/60 (train/validation/frozen test). No
eligible `real_trace` was available in this build; all prior canary traces are
marked `training_eligible=false` and are not silently converted into training
data.

The 480 records are **automatic contract-approved** candidates, not yet the
final dataset. Run the offline review command after the builder:

```text
python scripts/review_decision_sft_v1.py
```

The review preserves those candidates as `candidate_v1.jsonl`, reads every
State/target directly with the repository-agent rubric, adds controlled
availability/redundancy/finish boundary cases, performs a high-risk second
pass, then revalidates, deduplicates, and re-splits the accepted records. It
does not call Gemini/Qwen/DeepSeek, start a service, or launch training. Final
artifacts are written to `artifacts/decision_sft_v1_reviewed/` by default.

The review gate is deliberately four-stage:

1. deterministic contract validation;
2. repository-agent full semantic review;
3. high-risk bucket second-pass adjudication;
4. deterministic post-review validation.

Only a successful final review sets
`DECISION_SFT_V1_DATA_READY=true` and `training_eligible=true`; no training is
started by either command.

## Coverage and leakage controls

The candidate pool covers every fixed Scientific Action (at least 15 examples
each), availability boundaries, objective/action confusion, missing project or
covariate dimensions, insufficient data, redundancy, premature/missed finish,
heterogeneity conflict, missing evidence, and minimal-change paired states.
Paired states and trajectory members are assigned as a group, never split across
train/validation/test. Exact normalized State+target duplicates are removed;
coarse near-duplicate signatures are reported separately and paired groups are
excluded from that intentional-difference audit.

`metadata` records source type, trajectory/pair group, hard-case class,
availability profile, and review provenance for offline auditing only. It is
never included in `train.jsonl`, `validation.jsonl`, or `test.jsonl`.

## Training preparation gate (still offline)

After the repository-agent review is frozen, run:

```text
python scripts/prepare_decision_sft_v1_training.py
```

The command does not call a model, start an A100, or train. It validates the
518-record freeze, writes SHA-256 fingerprints for all reviewed files, records
the historical SFT hyperparameters (and explicitly excludes the DPO learning
rate), audits Qwen tokenizer lengths, and performs a CPU-only
JSONL → Qwen chat-template → assistant-only labels → collator dry run.

Preparation artifacts are written below
`artifacts/decision_sft_v1_reviewed/training_prep/`. The resulting trainer
configuration uses the serving `ScientificPolicyInput` and the exact
`generic_contract_hardened` system prompt, `enable_thinking=false`, BF16,
rank-16/alpha-32/dropout-0.05 LoRA, the historical `2e-4` SFT learning rate,
and two initial epochs. The trainer entry point
`sft/p2j4_train_decision_sft_v1.py` accepts only train and validation; the
frozen test is read by the separate evaluator contract. `split_manifest.json`
records the immutable train/validation/test boundary and file fingerprints;
`base_sft_evaluation_schema.json` freezes the later Base-vs-SFT metric and
external-canary comparison protocol without running either model.

After a model run is locked, score its frozen-test prediction JSONL with
`python scripts/evaluate_decision_sft_v1.py --predictions <predictions.jsonl>`.
Without `--predictions` the command only prints the evaluator contract.

The prior Base/SFT Dynamic Canary reports remain external evaluation only;
their `training_eligible=false` provenance is checked and no canary row is
copied into v1. No training command is run by this preparation step.
