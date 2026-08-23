# Mico Agent Runtime

This directory contains the independent Python + LangGraph runtime. The
current checked-in workflows are deterministic by default and do not connect
to the business MySQL database or read Java configuration files. An
OpenAI-compatible planner is optional and fail-closes to deterministic mode;
no token is stored in the repository.

## Boundary

The only future business-data port is:

```text
POST {MICO_JAVA_AGENT_TOOL_BASE_URL}/internal/agent/tools/execute
```

The base URL and Java service token are read only from environment variables.
The port refuses to construct when either value is missing. P2-A tests inject a
fake port or an `httpx.MockTransport`; they do not call a real Java service.

The runtime HTTP entry is an app factory for closed internal endpoints,
including:

```text
POST /internal/runtime/intent-runs
POST /internal/runtime/evidence-runs
GET  /internal/runtime/intent-runs/{runId}/progress
GET  /internal/runtime/intent-runs/{runId}/events
```

All endpoints are disabled unless `MICO_RUNTIME_INTERNAL_TOKEN` is configured.
The browser-facing entry, when explicitly enabled in Java, is the Java BFF;
the browser never calls this Runtime directly. `execute_read_query` accepts
only an untrusted model SQL draft and Java performs the final bounded,
read-only policy validation and execution.

The progress and event endpoints expose only a process-local, redacted view
of completed or currently executing intent runs. They are not durable
checkpoint recovery, Redis-backed queue state, or a replayable audit store;
the Java BFF is the only browser-facing path.

## Main research path

For a free-form科研问题, the only active business path is:

```text
自然语言问题
  -> LangGraph IntentRuntime
  -> Planner route_intent（模型只输出闭合 routePlan）
  -> execute_read_query（仅当路由为动态只读工作流）
  -> Java DynamicReadQueryPolicy + PreparedStatement
  -> bounded Java result / transient receipt
  -> Python model-generated, AST-validated analysis
  -> Java BFF safe report
```

文献/机制类问题可以走同一个 LangGraph 入口的
`knowledge_retrieval` 分支：本地 65 篇全文的 TF-IDF 向量检索与来源图谱检索
并行打分，结果保留全文证据等级和 chunk provenance，不调用业务 MySQL。

The planner may provide a bounded `SELECT`/`WITH ... SELECT` draft for the
dynamic read workflow. The draft is untrusted text: Java is the only place
that decides whether it can run and the only place that reaches
`patient_data_manager`. Python does not add tables, rewrite permissions or
execute a fallback query. Without a complete planner configuration, the
Runtime explicitly reports deterministic mode; a dynamic query that has no
verified SQL draft stops with a fixed contract error rather than inventing a
query.

The independent MySQL Runtime Store is opt-in only. When
`MICO_AGENT_RUNTIME_MYSQL_ENABLED=true` and its database URL, state key and
key ID all pass fail-closed validation, `RuntimePersistenceCoordinator` writes
only encrypted run state plus fixed step/tool/snapshot metadata. It never
opens the business database and `recoverable=False` remains in force.

## Local full-text knowledge retrieval

The Runtime can use the explicitly configured 65-paper full-text index for
evidence retrieval. The JSONL backend is an explicit local mode and uses
`MICO_LOCAL_KNOWLEDGE_ENABLED=true` plus
`MICO_KNOWLEDGE_RETRIEVAL_BACKEND=local`.
The real database backend requires
`MICO_KNOWLEDGE_VECTOR_ENABLED=true`,
`MICO_KNOWLEDGE_VECTOR_DATABASE_URL` for the independent `mico_knowledge`
PostgreSQL database, `MICO_KNOWLEDGE_GRAPH_ENABLED=true`,
`MICO_KNOWLEDGE_NEO4J_URI`, `MICO_KNOWLEDGE_NEO4J_USER` and
`MICO_KNOWLEDGE_NEO4J_PASSWORD`. Run
`python scripts/ingest_knowledge_stores.py` to load the immutable corpus into
pgvector and Neo4j. When both database enable flags are true and no backend is
explicitly specified, Runtime selects `database` automatically; it does not
silently fall back to JSONL. For Gemini dense retrieval, additionally set
`MICO_LOCAL_KNOWLEDGE_RETRIEVAL_BACKEND=gemini` and inject
`MICO_GEMINI_EMBEDDING_ENABLED=true`, `MICO_GEMINI_EMBEDDING_MODEL=gemini-embedding-2`,
and `MICO_GEMINI_API_KEY` through the deployment environment. The key is never
stored in the repository or emitted in logs.
The local port uses either the full-text TF-IDF baseline or the explicit
Gemini paper-level index (`fulltext-gemini-paper-embedding-v1`) plus
provenance-graph matching. Database mode performs the equivalent retrieval in
PostgreSQL+pgvector and Neo4j; JSONL remains the versioned ingestion source.
Gemini paper ranking is mapped back to the existing full-text chunks, and every
result remains labeled `evidenceTier=fulltext`. A structured intent plan selects
`vector`, `graph`, or `vector + graph`; the hybrid path runs both branches
independently and applies one bounded reranker. Graph hops expose `supported`,
`speculative`, or `conflicted` status, and the generator returns only
evidence-bound `reasoningSteps` plus a bounded `conclusion`.
It does not read Java configuration or connect to business MySQL. If the selected index
is enabled but invalid, the Runtime fails closed instead of falling back to an
unrelated evidence source.

P2-G2 adds a closed `RetrievalPlan` shared by the vector and graph ports, one
`merge_and_rerank` implementation with an auditable score breakdown, and
`ReasoningPath` objects that bind every hop to its evidence chunk.
`MICO_KNOWLEDGE_GRAPH_VERSION` selects the explicit graph version and defaults
to the existing v3 until a staging build passes the review queue and is
explicitly published. A `--publish` ingestion flag cannot bypass the manifest
approval marker. Structured generation
can only cite returned evidence and paths.

For optional evidence-grounded structured generation, configure
`MICO_GRAPH_RAG_GENERATOR_BASE_URL`, `MICO_GRAPH_RAG_GENERATOR_MODEL` and
`MICO_GRAPH_RAG_GENERATOR_TOKEN`. Without these values the Runtime uses the
explicit `deterministic_grounded` path; it never treats a fixed fallback as
  model planning. Model claims must reference returned evidence and graph path
  IDs or they are rejected and downgraded.
  For DeepSeek-compatible deployments, use `https://api.deepseek.com` as the
  base URL; the Runtime uses `/chat/completions` there. Other OpenAI-compatible
  base URLs use `/v1/chat/completions`. Optional provider JSON/thinking hints may
  be removed on a 400/422 response, but evidence/path validation remains mandatory.

## Development

Install only into the project-specific environment when approved:

```text
python -m pip install -e ".[test]"
pytest
```

The repository records the dependency contract in `pyproject.toml`, including
`uvicorn` for an explicitly managed local run. This repository does not create
an environment or start a server as part of its tests.
