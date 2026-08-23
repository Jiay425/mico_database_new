package com.database.mico_database.agent.readmodel;

import java.time.Instant;

/**
 * A non-persistent receipt for a P1-B1 read. It is not a replayable DataSnapshot.
 */
public final class ReadReceipt {

    public static final String SCHEMA_VERSION = "p1b1-java-read-model-v1";
    public static final String TRANSIENT_PERSISTENCE = "transient";

    private final String source;
    private final Instant generatedAt;
    private final String queryHash;
    private final Long rowCount;
    private final String schemaVersion;
    private final String snapshotPersistence;

    public ReadReceipt(String source, Instant generatedAt, String queryHash, Long rowCount) {
        this(source, generatedAt, queryHash, rowCount, SCHEMA_VERSION, TRANSIENT_PERSISTENCE);
    }

    public ReadReceipt(String source, Instant generatedAt, String queryHash, Long rowCount,
                       String schemaVersion, String snapshotPersistence) {
        this.source = source;
        this.generatedAt = generatedAt;
        this.queryHash = queryHash;
        this.rowCount = rowCount;
        this.schemaVersion = schemaVersion;
        this.snapshotPersistence = snapshotPersistence;
    }

    public String getSource() {
        return source;
    }

    public Instant getGeneratedAt() {
        return generatedAt;
    }

    public String getQueryHash() {
        return queryHash;
    }

    public Long getRowCount() {
        return rowCount;
    }

    public String getSchemaVersion() {
        return schemaVersion;
    }

    public String getSnapshotPersistence() {
        return snapshotPersistence;
    }
}
