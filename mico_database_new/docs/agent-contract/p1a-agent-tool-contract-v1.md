# Mico P1-A Agent Tool Contract v1

## Scope and authority

P1-A establishes a Java-owned, strongly typed, read-only and auditable contract. Java remains the only business fact source. A future Python + LangGraph Runtime may call only the Java-controlled tool boundary (Java API or MCP adapter); it must not connect to business MySQL. The original typed tools remain fixed. P2 adds a separate `execute_read_query` tool for model-proposed dynamic SQL: the SQL shape is not hard-coded per question, but the untrusted statement is still validated by Java for read-only semantics, catalog boundary, limit and timeout before JDBC execution.

This phase contains models, a catalog, deterministic policy validation, a response boundary and offline tests. The original `AgentToolContractExecutor` remains a contract-only offline executor: it does not connect to MySQL, create a snapshot, change the existing `AgentApiController`, or alter existing page/API behavior. Valid calls through that contract-only executor return `NOT_IMPLEMENTED`; they never return simulated rows or a fabricated live snapshot. P1-B2 adds a separate internal `AgentReadOnlyToolExecutor` that may dispatch only to the already verified P1-B1 Read Model.

The presence of a `requestedScopes` field is a policy input, not proof that user authentication or authorization has already been integrated. A production adapter must bind the caller identity and effective scopes to the platform's authenticated context before execution.

## Core semantic boundaries

| Contract entity | Meaning | P1-A identity rule |
|---|---|---|
| Study | Research/project context | `studyKey` identifies a project/study namespace. `2015_Castro-NallarE` is a project name, not a sample name. |
| Subject | Real-world participant, only when an independently verified subject link exists | Default `subjectLinkStatus=unverified`. `patient_id` is never promoted to `subjectId`. |
| Sample | A biological/metagenomic sample or accession in a source namespace | `SRR1518476` is a sample accession/sample name, not a Study. |
| TaxonomicProfile | A versioned species/feature abundance profile associated with a Sample | Requires `recordProfileLocator` plus `taxonomyVersion`, `featureVersion`, and `sourceBatch`. |
| DiseaseAssertion | A raw disease label plus an explicit normalization/mapping status | Unreviewed labels may safely return `rawLabel` with `mappingStatus=unmapped`. |
| InternalRecord | Internal business row keyed by `patient_id` | `patient_id` is `internalRecordId`; it is not a Subject ID or a Sample primary key. |

The P0.1 live evidence reports 24,264 internal business records, 23,072 distinct standard-abundance sample keys, and 13,897 Meta2DB samples matched to the standard abundance namespace. These are evidence-scoped facts, not permissions to expose raw tables. The contract describes those numbers as internal record/sample-key counts, never as unique human patient counts.

The locator distinction is mandatory:

```text
sampleAccession = 候选检索标识
recordProfileLocator = 精确业务查询定位器
internalRecordId != Subject ID
```

`sampleAccession` (for example `SRR1518476`) can find candidate samples, but it is not by itself an unqualified abundance-profile identity. `RecordProfileLocator` is the typed pair `(internalRecordId, sourceSampleId)`. It addresses one source sample/profile in one current Mico internal business record. It is not a Subject ID, not a global Sample primary key, and not a concatenated `sampleKey` string. `internalRecordId` must be a positive integer; `sourceSampleId` must be non-blank and no longer than 4,096 characters.

The minimum future snapshot metadata is:

`dataSnapshotId`, `dataSource`, `importBatch`, `diseaseMappingVersion`, `taxonomyVersion`, `featureVersion`, `sourceBatch`, `cohortCondition`, `queryHash`, `rowCount`, `generatedAt`.

## Tool Catalog

The catalog contains the fixed typed tools plus the controlled dynamic read tool. `readOnly=true` means no business-table mutation. `requiresSnapshot=true` means a real adapter must bind its result to evidence metadata or return a policy/error response. In P1-A no tool creates or reads a live snapshot; P1-B2's separate executor attaches only a non-persistent transient receipt converted from the P1-B1 Read Model.

| Tool | Structured input | Required scope | Risk | Read-only | Snapshot | Max result limit |
|---|---|---|---|---|---|---:|
| `resolve_study` | `studyKey` | `mico:study:read` | LOW | yes | yes | 100 |
| `resolve_sample` | `sampleAccession` or `sourceSampleId`; optional `studyKey`, `limit` | `mico:sample:read` | MEDIUM | yes | yes | 20 |
| `get_sample` | `recordProfileLocator` | `mico:sample:read` | MEDIUM | yes | yes | 1 |
| `list_sample_profiles` | `recordProfileLocator`, `taxonomyVersion`, `featureVersion`, `sourceBatch` | `mico:abundance:read` | MEDIUM | yes | yes | 100 |
| `get_abundance_summary` | `recordProfileLocator`, `taxonomyVersion`, `featureVersion`, `sourceBatch`; optional `limit` | `mico:abundance:read` | MEDIUM | yes | yes | 100 |
| `resolve_disease` | closed `rawLabel` | `mico:ontology:read` | LOW | yes | yes | 100 |
| `build_cohort` | closed `comparisonId=t2d_vs_healthy_v1` | `mico:cohort:read` | HIGH | yes | yes | 5,000 |
| `submit_differential_analysis` | closed `comparisonId=t2d_vs_healthy_v1` | `mico:analysis:submit` | HIGH | yes | yes | 1 |
| `get_differential_summary` | opaque `analysisDatasetHandle` | `mico:analysis:read` | HIGH | yes | yes | 20 |
| `read_analysis_dataset_page` | opaque analysis handle plus bounded page | `mico:analysis:read` | HIGH | yes | yes | 1,000 |
| `execute_read_query` | model-proposed `sql` plus optional limit | `mico:query:read` | HIGH | yes | yes | 1,000 |
| `get_data_snapshot` | `dataSnapshotId` | `mico:snapshot:read` | LOW | yes | no | 1 |
| `get_quality_summary` | `dataSnapshotId` | `mico:quality:read` | LOW | yes | yes | 1 |

Each catalog entry exposes `toolName`, `description`, `requiredScopes`, `riskLevel`, `readOnly`, `requiresSnapshot`, `maxResultLimit` and a closed `argumentSchema`. `build_cohort` is currently limited to the reviewed fixed comparison. It has no `sql`, `rawSql`, `query`, `where`, table name, arbitrary column or arbitrary predicate input. Only the separate `execute_read_query` tool accepts a model-generated `sql` field.

## TaskPacket

`TaskPacket` is a model-only hand-off object. It contains `runId`, `sessionId`, `question`, non-sensitive `pageContext`, `requesterId`, `requestedScopes`, `riskPolicy`, catalog-checked `allowedTools`, fixed `dataContractVersion=v1` and `createdAt`. It does not contain database credentials, connection strings or raw user-sensitive payloads. `allowedTools` rejects names outside the catalog. It is not wired into the existing controller in this phase.

## Request policy and errors

`AgentToolRequestValidator` rejects an unknown tool, malformed/blank `runId`, invalid version, missing required scope, missing required arguments, unknown arguments, non-positive or over-limit `limit`, dangerous argument names (`rawSql`, `query`, `where`, `table`, `database`, `schema`, and related selectors), `patientId`/`subjectId`, unconstrained cohort conditions, and abundance requests without an exact typed `recordProfileLocator` plus version/source-batch conditions. The sole exception is the dedicated `execute_read_query.sql` property, which is still passed through `DynamicReadQueryPolicy` and cannot be used by any fixed tool. A request that carries `patient_id` must use an explicitly named internal-record concept; it cannot be interpreted as a Subject.

For exact sample/profile tools, a string `sampleKey` is not accepted. The Java contract requires a `RecordProfileLocator` object, and the JSON Schema requires its closed nested shape: `internalRecordId` and `sourceSampleId`, with no additional fields. A candidate accession alone can call `resolve_sample`, but it cannot call `get_sample`, `list_sample_profiles`, or `get_abundance_summary`.

Domain values are not SQL. Values such as `2015_Castro-NallarE|SRR1518476_Study_of_microbial_diversity_from_samples_derived_from_throat_swabs_of_schizophrenia_patients_hum_removed_nt_mhl15` and `T2D;fatty_liver` are valid sample/project or disease text and must pass value validation. The validator does not reject ordinary occurrences of `from`, `patients`, `diseases`, underscores, or semicolons. Its remaining value-level check only catches an input that clearly starts with a SQL statement or stacks a second SQL statement after a semicolon; it also applies bounded length and control-character checks.

This narrow value check is not the final SQL injection guarantee. For fixed tools, the guarantee comes from the fixed closed Tool Schema, fixed Java implementations and parameterized Mapper SQL. For `execute_read_query`, the guarantee comes from Java's read-only dynamic query policy plus the restricted Runtime database account. Real domain values still must not be treated as SQL syntax; allowing raw labels or composite sample keys does not grant them query execution authority.

The dynamic query extension is documented separately in `p2-controlled-dynamic-query-v1.md`. LangGraph and a model may propose a query shape, but Java remains the final execution authority. Python never receives a DataSource and never executes SQL itself.

Every rejection is a structured `AgentToolError` with:

`code`, `message`, `field`, `retryable`, `policyReason`.

## Response and implementation boundary

`AgentToolResponse<T>` exposes `toolCallId`, `runId`, `status`, `source`, `rowCount`, `schemaVersion`, `generatedAt`, `dataSnapshot`, `qualitySummary`, `data` and `error`. Statuses are `COMPLETED`, `REJECTED`, `NOT_IMPLEMENTED` and `FAILED`.

`COMPLETED` requires snapshot metadata, so a successful data adapter cannot silently detach rows from their evidence/version boundary. `NOT_IMPLEMENTED` has no `data`, `rowCount`, `dataSnapshot` or `qualitySummary`; its message is generic because a tool may be registered before its safe Java implementation exists. In the current P2-C2 implementation, `resolve_disease` and the fixed `build_cohort` are backed by Java Read Model SELECTs; other registered tools may still be `NOT_IMPLEMENTED`.

The JSON request schema is in `p1a-agent-tool-schema-v1.json`. Its top-level and per-tool argument objects are closed with `additionalProperties=false`, and the only legal properties are the fixed catalog fields. This is a contract boundary, not a database schema.

## Offline verification

`AgentToolContractTest` runs without a Spring context, remote database, SSH tunnel, application server or real query. It verifies catalog completeness, scope/risk/read-only metadata, fixed disease/cohort request shapes, legal and illegal request shapes, SQL rejection, subject-link safeguards, unimplemented response semantics, snapshot-required completion, and the machine-readable schema's dangerous-field exclusion. P2-C2 adds offline typed executor coverage and an opt-in remote aggregate test.
