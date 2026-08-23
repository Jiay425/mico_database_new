# P1-B1 Java Read Model v1

## 1. Scope and authority

P1-B1 adds a Java-side, typed, read-only adapter for the five facts required by the next Agent Runtime stage. Java remains the only business fact source. The adapter is an internal Java service/Mapper boundary; it does not add an HTTP, MCP, FastAPI, Python, or LangGraph interface.

The adapter uses the existing Java SSH-tunnel mechanism when a remote integration test is explicitly enabled. Python must not connect to MySQL. Future Python + LangGraph Runtime calls must go through a controlled Java API/MCP layer built on the existing typed contract; P1-B1 does not expose that layer yet.

The domain terms remain strict:

- `SRR1518476` is a sample accession and candidate lookup value.
- `2015_Castro-NallarE` is a Study/project name.
- `patient_id` is exposed only as `internalRecordId`; it is not a Subject ID.
- `recordProfileLocator = (internalRecordId, sourceSampleId)` identifies the current Mico business record's source sample/profile. It is not a global Sample primary key and does not prove an independent real-person relationship.
- `subjectLinkStatus` is returned as `unverified` by this adapter. `host_subject_id` is not used to upgrade it.

The adapter is read-only: all five statements are fixed SQL with `#{}` parameter binding. There is no free SQL, dynamic table/column/order/WHERE fragment, or `SELECT *` path.

## 2. Typed capabilities

| Capability | Input | Typed output | Maximum | Fixed semantic boundary |
|---|---|---|---:|---|
| `resolveStudy(studyKey)` | Exact non-empty Study key | `StudyReadModel` | 1 study row | Matches `meta2db_sample_metadata.project_name` exactly and returns the count of Meta2DB sample records in that project. The project name is never extracted from a sample string. |
| `resolveSampleCandidates(sampleAccession, sourceSampleId, studyKey, limit)` | At least one of accession/source sample ID; optional exact study; `1..20` limit | `List<SampleCandidateReadModel>` | 20 | Candidate search covers standard-abundance samples. Every provided condition is an intersection filter; all bounded matches are retained and the adapter never auto-selects the first candidate. |
| `getSampleByRecordProfileLocator(locator)` | Complete `RecordProfileLocator` | `SampleDetailReadModel` | 1 detail row | Exact query is anchored in `microbe_abundance_standard` and uses both `patient_id = internalRecordId` and binary-exact `sample_id = sourceSampleId`; Meta2DB is optional enrichment. |
| `listSampleProfiles(locator, taxonomyVersion, featureVersion, sourceBatch)` | Complete locator and all three exact version/batch values | `List<TaxonomicProfileReadModel>` | Current fixed profile groups for the exact condition | Reads only the exact record/profile and exact abundance version/batch; `storedRowCount` is `COUNT(*)` of stored abundance rows. |
| `getAbundanceSummary(locator, taxonomyVersion, featureVersion, sourceBatch, limit)` | Complete locator, all exact version/batch values, `1..100` limit | `List<AbundanceFeatureReadModel>` | 100 | Returns fixed-order Top-N stored rows for the exact profile/version/batch. It does not claim a unique species count. |

All result DTOs use explicit fields. They do not return `Map<String,Object>`, raw database rows, patient display names, independent patient counts, or Subject IDs. The core identifiers are deliberately named `internalRecordId`, `sampleAccession`, `sourceSampleId`, `profileSample`, and `studyKey`.

## 3. Physical source mapping

The Mapper reads explicit columns from:

- `meta2db_sample_metadata` for `sample_id`, `project_name`, `profile_sample`, `health_disease_status`, and `raw_metadata.run_acc`;
- `patients` for the raw disease and limited demographic context needed by the typed sample detail;
- `microbe_abundance_standard` for stored abundance features, `feature_version`, `source_batch`, and stored-row counts.

The Meta2DB-to-standard join is explicitly constrained by both `patient_id` and binary comparison of `sample_id`, because the two physical sample columns have different character-set metadata. No implicit cross-character-set join is used.

Candidate discovery uses two fixed branches. The Meta2DB branch is the indexed metadata-enrichment path for accession and study filters. A second branch is enabled only when `sourceSampleId` is supplied and aggregates `microbe_abundance_standard` by `(patient_id, sample_id)` before left-joining `patients`; it adds only standard-abundance samples with no matching Meta2DB row. This avoids expanding one candidate per stored feature row and avoids using an accession-only scan of the full standard-abundance table. The standard-only branch has no reliable accession, profile, study, or sample-group value, so those Meta2DB enrichment fields remain `null` rather than being inferred.

`sampleAccession` and `studyKey` are optional Meta2DB enrichment fields, not fields guaranteed across the full standard-abundance namespace. A standard-only sample can be discovered by exact `sourceSampleId`, but candidate discovery is not Subject or real-patient identification: `internalRecordId` remains an internal business record key and `subjectLinkStatus` remains `unverified`.

`getSampleByRecordProfileLocator` starts at `microbe_abundance_standard`, so it covers every `(patient_id, sample_id)` pair with stored standard-abundance rows, including standard-only samples absent from Meta2DB. It then left-joins `patients` and `meta2db_sample_metadata`. Consequently, Meta2DB-only fields—`sampleAccession`, `profileSample`, `studyKey`, and `sampleGroup`—remain `null` when no Meta2DB row exists; the adapter does not infer or fill them. The detail model exposes the distinct physical `featureVersion` and `sourceBatch` values for the locator; if more than one exists, the typed string contains a comma-separated ordered summary rather than a fabricated single version.

The current physical standard abundance evidence has `feature_version` and `source_batch`; it does not provide a confirmed physical `taxonomy_version` column. Therefore P1-B1 accepts the fixed adapter vocabulary `species-v1` as `taxonomyVersion` and binds it into the typed result. This is a contract-level vocabulary for the verified imported species profile, not a claim that a database column exists. A future taxonomy-version migration must change the contract and SQL deliberately.

The known live-domain examples are consistent with the P0/P0.1 audit: `SRR1518476` can be used to find the Study `2015_Castro-NallarE`. The P1-B1 remote check reads the physical version columns for this fixture as `featureVersion=meta2db-species-v1` and `sourceBatch=meta2db-species-20260726:2015_Castro-NallarE`; this live result takes precedence for the adapter over an earlier historical field-label example and should be reconciled in the next evidence refresh. Remote integration tests verify these facts without printing raw rows.

## 4. SQL safety and bounds

Every statement is a named MyBatis statement with a closed method signature. Values are passed with `#{...}` only. The XML contains no `${...}`, `SELECT *`, dynamic identifiers, dynamic sort expressions, or caller-provided SQL fragments. Sorting is fixed in XML. `LIMIT` is a bound integer validated by the service (`20` for candidate search and `100` for abundance summary).

The candidate statement treats each input as an optional predicate: absent accession/source/study values do not filter, while every supplied value must match. Thus multiple supplied conditions are an AND intersection, never an OR union. The exact profile statements always bind both locator components. A sample accession alone is never accepted by an exact profile method; it only searches candidates. P1-B2 must preserve these boundaries and use parameterized/fixed statements for any additional tool implementation.

`storedRowCount` is always a stored abundance-row count under the statement's exact locator/version condition. It is not a unique-species count.

## 5. Read receipt versus persistent DataSnapshot

Each read result carries a transient `ReadReceipt` with:

```text
source
generatedAt
queryHash
rowCount
schemaVersion
snapshotPersistence = transient
```

The receipt is an audit hint for this in-process read and is not a `dataSnapshotId`. P1-B1 does not create persistent or replayable analysis snapshots, does not write an audit table, and does not claim reproducibility after process termination. A durable `DataSnapshot` with import batch, mapping version, taxonomy version, feature version, cohort condition, and replay storage belongs to the later Agent Runtime/audit-storage stage.

## 6. Test boundary

`AgentReadModelServiceTest` is offline and uses a typed fake Mapper. It verifies locator validation, bounds, candidate non-selection, stored-row semantics, transient receipt fields, and the absence of `${...}`/raw-row/free-SQL Mapper paths.

`AgentReadModelRemoteIntegrationTest` is tagged `remote-integration` and is disabled unless `-Dmico.remote.integration=true` is supplied. It creates a MyBatis session directly, without starting Spring Boot, and uses only the existing Java SSH tunnel and SELECT statements. It verifies the known accession/Study relation, the all-provided-condition intersection and mismatch case, construction of a complete locator, exact sample lookup, exact profile version/batch, bounded abundance results, a test-only standard-only fixture whose Meta2DB fields are null, and the fixed Subject evidence ledger. The fixture lookup and evidence aggregate are not exposed through production Mapper or Service methods. The test does not print credentials, key paths, complete connection strings, or abundance payloads.

P1-B1's original five read-model methods remain unchanged. P2-C2 adds separate fixed `resolve_disease` and `build_cohort` read-model methods; `get_data_snapshot`, `get_quality_summary`, a public Controller, MCP, and Python direct database integration remain out of scope.
