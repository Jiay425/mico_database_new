# P2 Controlled Dynamic Read Query Contract v1

> **历史 P2 动态查询契约快照**：本文仍准确描述动态 SQL 的 Java 最终校验、
> Python 不直连业务 MySQL 和安全边界，但其“Catalog 只有
> `execute_read_query`”及旧固定工具并存的表述是阶段性快照。当前 active
> Catalog 还提供 metadata-only `describe_read_schema`；当前综合状态见
> `mico-agent-runtime/docs/agent-runtime/p2j4-1-architecture-calibration-v1.md`。

## Purpose

LangGraph is the orchestration layer. A configured model may recognize the
research intent and propose a SQL draft for a read task. The draft is
untrusted input. Python never opens a MySQL connection and never executes SQL.
It sends a closed `execute_read_query` ToolCall to Java; Java is the only
component that validates and executes the query.

The effective flow is:

```text
user question
  -> LangGraph intent recognition
  -> closed execute_read_query(sqlDraft, limit)
  -> Java DynamicReadQueryPolicy
  -> Java JDBC read-only connection
  -> bounded result + transient receipt
```

`dynamic_read_query` is the explicit generic research-query workflow. The
legacy workflow name `differential_analysis` remains accepted for compatibility.
“动态 SQL” means that the query text is not hard-coded per user question. It
does not mean that the model can bypass Java policy or send a write statement.

The planner receives a versioned names-only read-schema guide v1 so it can
construct explicit projections and approved internal-record joins. The guide
contains table/column names and semantic warnings only; it contains no sample
values, locator, credentials, URLs, database connection details or payload.
It is advisory context, not an execution allowlist or a substitute for Java
validation.

## Tool contract

| Field | Rule |
|---|---|
| `toolName` | `execute_read_query` only |
| `requiredScopes` | `mico:query:read` |
| `readOnly` | `true` |
| `riskLevel` | `HIGH` |
| `requiresSnapshot` | `true`; the receipt is transient |
| `arguments.sql` | non-empty model proposal, max 16,000 characters |
| `arguments.limit` | optional 1–1,000 execution bound |

The transport envelope remains closed and contains only `toolName`, `runId`,
`toolCallId`, and `arguments`. `requesterId`, scope elevation, database URL,
table selection outside the policy surface, and credentials cannot be sent by
Python.

## Java execution policy

Before JDBC preparation Java requires:

- exactly one statement, with no semicolon or SQL comments in SQL syntax;
  semicolons, `@`, `?`, or comment-looking text inside quoted data literals are
  treated as data and are not rejected merely because of their characters;
- `SELECT` or `WITH ... SELECT` only;
- a literal bounded `LIMIT` between 1 and 1,000; optional `OFFSET`/comma offset is
  also bounded to prevent unbounded page scans;
- no mutation, DDL, grants, file export, stored procedure, lock,
  sleep/benchmark, session command, cross-database reference, or
  information-schema access. `UNION`, wildcard projections, arbitrary
  `JOIN`/`WHERE` clauses, grouping and ordering are valid read-query shapes;
- no static table-name allowlist is applied. The query is executed only against
  the Java connection's configured business catalog, with unqualified table
  references and the eventual Runtime account's database permissions as the
  physical data boundary. This keeps the SQL shape dynamic as the business
  read model evolves without allowing cross-database selection;
- a Java read-only connection, statement max rows, and a fixed query timeout.

Java returns a bounded generic projection plus `source`, `queryHash`, row count,
generation time, and `snapshotPersistence=transient`. SQL text is not copied
into the snapshot, audit record, or external report. When LangGraph is asked to
compute a generic two-group difference, Python accepts rows only when the
model-proposed query explicitly aliases `taxon_name`, `group_key`, and
`abundance_value`; it computes bounded medians and Mann–Whitney/BH summaries
in memory, then discards the raw rows. Missing aliases or sensitive values do
not get guessed or projected.

## Security interpretation

A MySQL read-only account is the primary database permission boundary, while
Java adds statement-level safeguards for destructive operations, cross-database
access, resource bounds and output size. The eventual Java Agent query account
must have only `SELECT` on the approved business catalog; it must not have
`INSERT`, `UPDATE`, `DELETE`, DDL, `GRANT OPTION`, or access to other catalogs.
Java remains the only component allowed to hold that database connection.

There is deliberately no fixed Java SQL template for this tool. The model may
choose the table, columns, joins, filters, grouping, ordering and aggregation
needed for the approved research intent. The SQL is still not an unrestricted
write interface: the Java policy accepts only one bounded read statement and
the database account must be read-only. Python/LangGraph sends the model draft
to Java; it never executes or interprets SQL itself.

The existing fixed tools (`resolve_sample`, `get_sample`, `build_cohort`, and
others) remain available for typed, reproducible operations. The dynamic tool
is intended for LangGraph research branches that need a query shape selected
from the user intent. It does not permit Python direct MySQL access or browser
direct database access.

## Current boundary

The deterministic planner does not invent SQL. Without a configured model (or
an injected planner that supplies a valid `sqlDraft`) a `dynamic_read_query`
route returns `QUERY_PLAN_REQUIRED`. A model-proposed query still must pass Java
validation. Differential statistics, effect-size calculation, literature
retrieval, persistent snapshots, and clinical interpretation remain separate
future capabilities.
