package com.database.mico_database.agent.contract;

import java.time.Instant;

/** Evidence metadata for one tool call; P1-B2 only creates transient, non-replayable evidence. */
public class DataSnapshot {

    public static final String SNAPSHOT_PERSISTENCE_TRANSIENT = "transient";

    private String dataSnapshotId;
    private String dataSource;
    private String importBatch;
    private String diseaseMappingVersion;
    private String taxonomyVersion;
    private String featureVersion;
    private String sourceBatch;
    private String cohortCondition;
    private String queryHash;
    private Long rowCount;
    private Instant generatedAt;
    private String snapshotPersistence = SNAPSHOT_PERSISTENCE_TRANSIENT;

    public String getDataSnapshotId() {
        return dataSnapshotId;
    }

    public void setDataSnapshotId(String dataSnapshotId) {
        this.dataSnapshotId = dataSnapshotId;
    }

    public String getDataSource() {
        return dataSource;
    }

    public void setDataSource(String dataSource) {
        this.dataSource = dataSource;
    }

    public String getImportBatch() {
        return importBatch;
    }

    public void setImportBatch(String importBatch) {
        this.importBatch = importBatch;
    }

    public String getDiseaseMappingVersion() {
        return diseaseMappingVersion;
    }

    public void setDiseaseMappingVersion(String diseaseMappingVersion) {
        this.diseaseMappingVersion = diseaseMappingVersion;
    }

    public String getTaxonomyVersion() {
        return taxonomyVersion;
    }

    public void setTaxonomyVersion(String taxonomyVersion) {
        this.taxonomyVersion = taxonomyVersion;
    }

    public String getFeatureVersion() {
        return featureVersion;
    }

    public void setFeatureVersion(String featureVersion) {
        this.featureVersion = featureVersion;
    }

    public String getSourceBatch() {
        return sourceBatch;
    }

    public void setSourceBatch(String sourceBatch) {
        this.sourceBatch = sourceBatch;
    }

    public String getCohortCondition() {
        return cohortCondition;
    }

    public void setCohortCondition(String cohortCondition) {
        this.cohortCondition = cohortCondition;
    }

    public String getQueryHash() {
        return queryHash;
    }

    public void setQueryHash(String queryHash) {
        this.queryHash = queryHash;
    }

    public Long getRowCount() {
        return rowCount;
    }

    public void setRowCount(Long rowCount) {
        this.rowCount = rowCount;
    }

    public Instant getGeneratedAt() {
        return generatedAt;
    }

    public void setGeneratedAt(Instant generatedAt) {
        this.generatedAt = generatedAt;
    }

    public String getSnapshotPersistence() {
        return snapshotPersistence;
    }

    public void setSnapshotPersistence(String snapshotPersistence) {
        if (!SNAPSHOT_PERSISTENCE_TRANSIENT.equals(snapshotPersistence)) {
            throw new IllegalArgumentException("Only transient snapshot evidence is supported in P1-B2");
        }
        this.snapshotPersistence = snapshotPersistence;
    }
}
