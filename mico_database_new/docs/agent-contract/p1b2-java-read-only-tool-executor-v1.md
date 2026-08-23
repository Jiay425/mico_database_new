# Mico P1-B2 Java Read-Only Tool Executor v1

## 1. Boundary and scope

`AgentReadOnlyToolExecutor` is an internal Java boundary. It receives the existing P1-A typed `AgentToolRequest`, validates it with `AgentToolRequestValidator`, and dispatches only the five fixed read-only operations already supplied by `AgentReadModelService`. Java remains the only business fact source.

This phase does not add HTTP, MCP, Python, FastAPI, LangGraph, Controller, page, or public API integration. It does not accept JDBC connections, MyBatis Mappers, SQL text, table names, dynamic columns, dynamic ordering, or free-form conditions. The existing P1-A `AgentToolContractExecutor` remains contract-only and is not changed into a database executor.

## 2. Fixed dispatch map

| Tool | Fixed Read Model method | Input boundary | Default/max result | Evidence boundary |
|---|---|---|---:|---|
| `resolve_study` | `resolveStudy(studyKey)` | Exact `studyKey` string | 1 study result | No single abundance version/batch is inferred |
| `resolve_sample` | `resolveSampleCandidates(sampleAccession, sourceSampleId, studyKey, limit)` | Candidate search; at least accession or sourceSampleId | default 20 / max 20 | Candidate search is not Subject identification |
| `get_sample` | `getSampleByRecordProfileLocator(recordProfileLocator)` | Typed `(internalRecordId, sourceSampleId)` only | 1 detail result | Single feature/source values are copied only when unambiguous |
| `list_sample_profiles` | `listSampleProfiles(locator, taxonomyVersion, featureVersion, sourceBatch)` | Typed locator plus all three exact version/batch fields | Read Model bound | Input versions are copied exactly |
| `get_abundance_summary` | `getAbundanceSummary(locator, taxonomyVersion, featureVersion, sourceBatch, limit)` | Typed locator plus all three exact version/batch fields | default 100 / max 100 | Input versions are copied exactly; features remain bounded Top-N rows |

The Executor uses an enum switch. It does not select the first candidate, convert `sampleAccession` into a locator, or interpret `patientId`/`subjectId` as `internalRecordId`. `recordProfileLocator` must already be a validated `RecordProfileLocator` object.

## 3. Response and transient evidence

Every `COMPLETED` response is built from the `ReadModelResult`'s real `ReadReceipt`:

| Response/snapshot field | Source or rule |
|---|---|
| response `source` | `ReadReceipt.source` |
| response/snapshot `rowCount` | `ReadReceipt.rowCount` |
| response/snapshot `generatedAt` | `ReadReceipt.generatedAt` |
| snapshot `dataSource` | `ReadReceipt.source` |
| snapshot `queryHash` | `ReadReceipt.queryHash` |
| snapshot `dataSnapshotId` | Newly generated short-lived `transient-*` call evidence ID; no persistence or lookup store exists |
| snapshot `snapshotPersistence` | Fixed value `transient` |
| snapshot `importBatch` | `null`; no reliable physical import-batch source is exposed by P1-B1 |
| snapshot `diseaseMappingVersion` | `null`; disease mapping is not implemented |
| snapshot `taxonomyVersion` | Exact input for the two versioned profile tools; always `null` for `get_sample` because that call has no verified single physical taxonomy version |
| snapshot `featureVersion` / `sourceBatch` | Exact inputs for profile tools, or an unambiguous single value from `get_sample`; otherwise `null` |
| snapshot `cohortCondition` | A bounded structured condition summary, never SQL; `get_sample` uses `internalRecordId=<id>;sourceSampleId=<value>` |

The transient ID is an evidence label for the current process call. It is not a durable `dataSnapshotId`, cannot be replayed after process termination, and is not accepted as a persistent snapshot lookup. `get_data_snapshot` therefore remains `NOT_IMPLEMENTED`.

Study and candidate responses cannot prove a single abundance version or batch, so those snapshot fields remain `null`. Standard-abundance-only candidates may have missing Meta2DB enrichment fields; the Executor preserves `null` and never fabricates accession, profile, Study, or sample-group values.

For `get_sample`, the locator summary occupies `cohortCondition`; it must not be shifted into a version field. If the detail model aggregates multiple feature versions or source batches, the corresponding snapshot field is `null` and the quality warnings state that the detail cannot alone serve as a strictly reproducible single-version abundance snapshot.

## 4. Conservative QualitySummary

Each completed result carries:

- `subjectLinkStatus=unverified`;
- missing counts calculated only from fields present in the returned typed payload;
- `duplicateCount=null` and `orphanCount=null` unless a future payload supplies defensible counts—zero is not asserted;
- warnings that `internalRecordId` is not a Subject ID, the result does not establish independent real-patient count, the snapshot is transient/non-replayable, and disease mapping is not implemented;
- an additional missing-enrichment warning when standard-only or otherwise incomplete Meta2DB fields are present.

These are payload-level quality boundaries, not whole-database quality conclusions. `storedRowCount` remains the stored abundance-row count under the exact Read Model condition, not a unique-species count.

## 5. Error policy

Validation always occurs before Read Model invocation. Invalid tools, scopes, locators, limits, dangerous argument names, unknown fields, and missing versions return `REJECTED` and the Read Model is not called.

Semantic `IllegalArgumentException` from the Read Model, such as an unsupported taxonomy version, returns `REJECTED` with `READ_MODEL_ARGUMENT_REJECTED`, `retryable=false`, and a generic message. Mapper, connection, and other runtime failures return `FAILED` with `READ_MODEL_EXECUTION_FAILED`, `retryable=true`. Neither path exposes exception text, SQL, table names, database addresses, usernames, passwords, key paths, or connection details.

At the P2-C2 boundary, `resolve_disease` and fixed `build_cohort` are implemented as aggregate/read-only Java tools. `get_data_snapshot` and `get_quality_summary` remain no-data `NOT_IMPLEMENTED` responses. This distinction means a tool may be catalogued before its safe implementation exists; it does not claim that all real tools are absent.

## 6. Verification boundary

`AgentReadOnlyToolExecutorTest` uses a typed in-memory Read Model stub and never starts Spring Boot or connects to a database. `AgentReadOnlyToolExecutorRemoteIntegrationTest` is opt-in through `mico.remote.integration=true`, creates only the existing Java SSH-tunnel-backed MyBatis read session, and executes SELECT-only checks for `SRR1518476`, `2015_Castro-NallarE`, and the standard-only sample fixture. It does not print credentials, key paths, connection strings, full abundance payloads, or expose a transport endpoint.
