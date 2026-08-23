# P5 Progress and Approval Boundary v1

## Progress

The Runtime exposes two internal, same-token-protected projections for the
intent workflow:

```text
GET /internal/runtime/intent-runs/{runId}/progress
GET /internal/runtime/intent-runs/{runId}/events
```

The Java BFF is the only browser-facing proxy:

```text
GET /agent/research/{runId}/progress
```

The persistence coordinator has a typed `checkpoint` primitive and a real
LangGraph `BaseCheckpointSaver` adapter. The adapter serializes the latest
LangGraph checkpoint inside the AES-256-GCM envelope stored in
`agent_run.encrypted_state_payload`, bound to run/task/trace and `v1`; it
never projects channel values into ordinary columns or progress events. A
new Runtime process can reconstruct the latest checkpoint and continue a
cooperatively interrupted graph, as proven by an offline two-process
simulation. The underlying `MySqlRuntimeStore.recoverable=False` flag remains
intentional: the store alone is not a graph lifecycle. The Runtime app marks
resume available only when the encrypted saver and an internal graph resume
handler are both injected.

The projection contains only opaque run/task identifiers, approved workflow,
fixed node names, tool status, fixed error codes, transient Java snapshot IDs,
snapshot persistence and UTC timestamps. It never contains trace IDs, the
request question, SQL, locator values, Java payload, tokens or database
information. The live progress projection remains process-local and finite.
Intent graph nodes publish safe audit projections while synchronous execution
runs in a worker thread; injected legacy one-argument Runtime adapters remain
compatible. The current Mico dashboard renders the last validated node and a
bounded list of fixed node/status pairs through the Java BFF; it never renders
the question, SQL, locator or Java payload.

When the explicit Runtime MySQL switch and encryption configuration are valid,
the progress endpoint falls back to a safe projection rebuilt from persisted
`agent_run` and `agent_step` metadata when the live registry has no entry. This
is only a safe progress view: it is not Redis-backed and does not expose
checkpoint state. Detailed node state remains encrypted and is only consumed
by the internal resume path. Redis-backed queueing, multi-worker lease
coordination, replay/time-travel and a remote production interruption drill
remain future deployment work.

## Approval

`ApprovalGate` allows only `read_only_research` and `statistical_analysis`
without human approval. `sensitive_batch_read`, `bulk_export`,
`external_publish` and `business_write` return fixed
`APPROVAL_REQUIRED`/`APPROVAL_WORKFLOW_NOT_IMPLEMENTED` decisions. No caller
boolean, free-text reason or token can turn those decisions into approval.

The Runtime has a closed, opt-in ticket boundary:

```text
POST /internal/runtime/approvals
GET  /internal/runtime/approvals/{approvalId}
POST /internal/runtime/approvals/{approvalId}/decision
```

Ticket creation accepts only `runId`, a fixed operation enum, `requestedBy`
and `dataContractVersion=v1`; it never accepts raw arguments, locator values or
free justification text. Decision requests contain only `APPROVED` or
`REJECTED`. The decision endpoint requires a separate deployment-injected
decision token and principal binding; it does not trust a caller-supplied
approver identity. The API is disabled unless
`MICO_RUNTIME_APPROVAL_ENABLED=true` and an injected `ApprovalCoordinator`
exists.

The ticket lifecycle is durable when backed by Runtime MySQL and is extended
by migration `0002_approval_ticket_lifecycle` (the migration is a separate
approved deployment asset). The Java BFF exposes request/status proxies bound
to the authenticated principal and run ID. A
decision proxy is also available only when Java receives an explicit
reviewer-principal allowlist; the browser cannot submit reviewer identity and
the Runtime still requires its separate decision token/principal binding.
Without that allowlist the BFF returns `APPROVAL_REVIEWER_NOT_ALLOWED` before
making a Runtime request. The deployed Store still advertises
`recoverable=false` as a store-level capability; the Runtime coordinator can
release a run only when its encrypted LangGraph saver and internal resume
handler are present. The coordinator binding is fail-closed.

The Java reviewer allowlist is injected only as opaque `principal-<32hex>`
values through `MICO_AGENT_APPROVAL_REVIEWER_PRINCIPALS`; it is not accepted
from the browser and is not copied into Runtime approval payloads.

## Run control

The Runtime exposes token-protected internal `cancel` and `resume` boundaries
for explicit fail-closed behavior. Only queued/waiting states can be
cancelled; an active synchronous graph returns
`RUN_CANCEL_NOT_SUPPORTED_FOR_ACTIVE_RUN` instead of being marked cancelled
while it may continue executing. Resume returns
`RUNTIME_RECOVERY_NOT_ENABLED` unless the Runtime has an encrypted LangGraph
saver, and returns `RUNTIME_RESUME_NOT_IMPLEMENTED` unless an internal graph
resume handler is bound. It never trusts client-supplied checkpoint state. The
current app binds the handler only after explicit Runtime persistence
configuration succeeds.

An `APPROVED` ticket is bound to the injected run-control coordinator. The
approval coordinator first loads the ticket's typed `runId`, then requires a
successful `RUN_RESUMED` response before writing the approval as `APPROVED`.
Without a configured recovery/resume adapter, approval remains `PENDING` and
returns `APPROVAL_RESUME_NOT_AVAILABLE`; this prevents an approval record from
being mistaken for released execution. Rejection remains terminal without
resume. A remote production crash/restart drill and the pending approval
migration are still required before claiming operational approval recovery.

The current Mico page includes a bounded approval-status projection for the
ordinary research card. A completed read-only run shows the fixed
`APPROVAL_NOT_REQUIRED` state; safe approval-related error codes are rendered
as fixed status text when returned by the Java BFF. The page has no approval
decision control unless a future page state carries a validated approval
reference; it never renders reviewer identity, raw arguments, SQL, locator
values, tokens or Runtime credentials. Post-approval graph release remains
conditional on the lifecycle migration, Runtime application credentials and a
deployment-level recovery drill.
