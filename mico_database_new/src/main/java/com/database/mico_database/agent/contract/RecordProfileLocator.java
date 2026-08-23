package com.database.mico_database.agent.contract;

/**
 * Exact locator for one Mico business record's source sample/profile.
 * It is not a Subject identifier, a global Sample primary key, or a free-form sampleKey string.
 */
public class RecordProfileLocator {

    public static final int MAX_SOURCE_SAMPLE_ID_LENGTH = 4096;

    private Long internalRecordId;
    private String sourceSampleId;

    public RecordProfileLocator() {
    }

    public RecordProfileLocator(long internalRecordId, String sourceSampleId) {
        this.internalRecordId = internalRecordId;
        this.sourceSampleId = sourceSampleId;
    }

    public Long getInternalRecordId() {
        return internalRecordId;
    }

    public void setInternalRecordId(Long internalRecordId) {
        this.internalRecordId = internalRecordId;
    }

    public String getSourceSampleId() {
        return sourceSampleId;
    }

    public void setSourceSampleId(String sourceSampleId) {
        this.sourceSampleId = sourceSampleId;
    }
}
