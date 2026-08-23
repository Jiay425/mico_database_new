# Mico P1-C Internal Agent Tool HTTP API v1

## 1. Boundary

`POST /internal/agent/tools/execute` is an internal machine-to-machine endpoint for a future Python + LangGraph Runtime. It is not a browser, public, legacy API, MCP, or Python database interface. Java remains the only business fact source:

```text
Python Runtime -> Bearer service token + closed JSON
               -> Java Internal Agent Tool API
               -> AgentReadOnlyToolExecutor
               -> AgentReadModelService -> fixed parameterized MyBatis SQL
```

The endpoint is default-closed by `mico.agent.internal.enabled=false`. Enabling it requires an explicit non-sensitive runtime configuration. The service token is read only from `MICO_AGENT_INTERNAL_TOKEN`; there is no default token and no token is stored in source, configuration, logs, test reports, or responses.

The endpoint does not start a listener by itself in tests. A future deployment must keep it on an internal network boundary and apply normal network policy in addition to the application token check.

## 2. Authentication and fail-closed identity

Clients must send:

```http
Authorization: Bearer <service-token>
Content-Type: application/json
```

Disabled API returns `503`. Missing, blank, or invalid token returns `401`. Token comparison uses constant-time byte comparison. Authentication failures happen before JSON adaptation and before Executor/Read Model invocation.

The JSON request cannot submit or override `requesterId` or `requestedScopes`. Java binds:

```text
requesterId = internal-agent-runtime
effectiveScopes = {
  mico:study:read,
  mico:sample:read,
  mico:abundance:read
}
```

These are service scopes, not proof of a human user's identity. P1-C does not implement trusted end-user identity propagation; `internal-agent-runtime` is only the internal service identity.

## 3. Closed request model

The only accepted top-level fields are:

```json
{
  "toolName": "resolve_sample",
  "runId": "run_...",
  "toolCallId": "call_...",
  "arguments": {}
}
```

Top-level `requesterId`, `requestedScopes`, `patientId`, `subjectId`, `sampleKey`, `sql`, `rawSql`, `query`, `where`, `table`, `database`, and `schema` are not transport fields. Unknown top-level fields produce safe `400` transport errors. Unknown argument names are retained by the adapter and passed to the existing validator so they produce a structured `REJECTED` response rather than being silently discarded.

The adapter enforces JSON types before constructing `AgentToolRequest`:

- declared string arguments must be JSON strings;
- `limit` must be a JSON integer;
- `recordProfileLocator` must be an object with exactly `internalRecordId` and `sourceSampleId`;
- locator IDs must be JSON integers, not strings or floating-point values;
- locator extra fields are rejected;
- no client JSON value is treated as SQL or a database selector.

The adapter has no Mapper, JDBC, SQL, table, or tool business logic. It only constructs the existing typed contract and applies the Java-bound identity/scopes.

## 4. Tool availability

The API can complete only these five fixed read-only tools:

| Tool | Java method | Scope bound by Java |
|---|---|---|
| `resolve_study` | `resolveStudy` | `mico:study:read` |
| `resolve_sample` | `resolveSampleCandidates` | `mico:sample:read` |
| `get_sample` | `getSampleByRecordProfileLocator` | `mico:sample:read` |
| `list_sample_profiles` | `listSampleProfiles` | `mico:abundance:read` |
| `get_abundance_summary` | `getAbundanceSummary` | `mico:abundance:read` |

`resolve_disease` and fixed `build_cohort` are now available through this API with Java-bound ontology/cohort scopes and their closed P2-C2 argument shapes. `get_data_snapshot` and `get_quality_summary` remain unavailable/`NOT_IMPLEMENTED`. A client cannot add scopes, arbitrary comparisons, SQL, or free conditions; all requests remain subject to the fixed effective scope set and validator.

## 5. Response and errors

Authenticated, structurally valid requests return HTTP `200` with the existing `AgentToolResponse` envelope. `COMPLETED`, `REJECTED`, `FAILED`, and `NOT_IMPLEMENTED` retain their existing meanings. Completed responses preserve the real Read Model `source`, row count, transient snapshot, and quality warnings.

Transport-level errors use a safe envelope:

```json
{
  "code": "INVALID_REQUEST",
  "message": "The request JSON does not match the closed internal Agent Tool contract"
}
```

Transport errors never include token text, SQL, table names, database addresses, credentials, key paths, exception stacks, or internal class names. Suggested statuses are `503` for disabled, `401` for authentication failure, and `400` for malformed JSON or closed-shape/type failure. An authenticated tool rejection or failure remains HTTP `200` with the specific `AgentToolResponse.status`.

## 6. Snapshot and quality boundary

The five completed tools carry the P1-B2 transient snapshot converted from the real `ReadReceipt`. `dataSnapshotId` is a short-lived `transient-*` evidence label, not a persisted or replayable database snapshot. Python cannot use it for cross-process replay. `get_data_snapshot` remains unavailable.

The response retains `subjectLinkStatus=unverified`, payload-level missing counts, and conservative warnings. `internalRecordId` is not a Subject ID and no independent real-patient count is established. Standard-abundance-only Meta2DB enrichment fields remain `null` when the Read Model does not provide them.

## 7. Operational restrictions

The implementation does not write the database, create schema, alter legacy controllers or APIs, expose MCP, or connect Python directly to MySQL. Requests and Authorization headers are not logged by this transport. Remote integration tests use MockMvc without starting a long-lived HTTP server and use only the existing Java SSH tunnel for SELECT-only Read Model verification.

## 8. Spring registration and security verification

`AgentInternalToolController` keeps its single `@RestController` registration and is discovered once by component scanning. `AgentInternalToolConfiguration` provides only the validator, read-only Executor, properties, and JSON adapter beans; it does not declare a second Controller `@Bean`. Bean overriding is not enabled.

P1-C must be verified with both standalone MockMvc tests and a real `@SpringBootTest(webEnvironment = MOCK)` context using the actual `SecurityFilterChain`. The full-context test asserts exactly one `AgentInternalToolController` bean and sends the disabled-endpoint request through the real filter chain. With `mico.agent.internal.enabled=false`, the request reaches the Controller and returns JSON `503 INTERNAL_AGENT_DISABLED`, rather than being redirected to the legacy login page or rejected by the legacy security rules. The full-context test does not start a real HTTP port and does not invoke a tool or query the database.
