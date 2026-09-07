# Mico Agent Runtime

The LangGraph runtime behind Mico's evidence-aware scientific analysis
workflow. It turns a research question into an auditable sequence of bounded
data reads, statistical analyses, literature retrieval, and structured
observations.

## What it does

- Runs an observation-driven `State → Action → Observation` agent loop.
- Keeps scientific policy, plan materialization, capability validation, and
  execution as separate responsibilities.
- Uses a fixed, closed set of scientific actions and validates every action
  against the current runtime capability.
- Materializes catalog-bound `QueryPlan` and `AnalysisPlan` contracts instead
  of accepting raw SQL or unrestricted code.
- Supports deterministic typed statistics and capability-approved generated
  analysis in an isolated, bounded sandbox.
- Retrieves source-bound evidence through dense, sparse, and graph retrieval
  with provenance-preserving fusion.
- Emits structured traces for decisions, plans, capability checks, execution,
  observations, limitations, and completion semantics.

## Safety model

Business data remains behind the Java tool boundary. The runtime does not
store credentials in source control, does not issue unconstrained database
queries, and does not use a model response as direct executable code or SQL.

See the repository [README](../README.md) for the full architecture.
