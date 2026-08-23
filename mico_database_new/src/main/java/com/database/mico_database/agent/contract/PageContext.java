package com.database.mico_database.agent.contract;

/** Non-sensitive page context that may help an agent address a user-visible record. */
public class PageContext {

    private String page;
    private String sampleAccession;
    private String studyKey;
    private String internalRecordId;

    public PageContext() {
    }

    public String getPage() {
        return page;
    }

    public void setPage(String page) {
        this.page = page;
    }

    public String getSampleAccession() {
        return sampleAccession;
    }

    public void setSampleAccession(String sampleAccession) {
        this.sampleAccession = sampleAccession;
    }

    public String getStudyKey() {
        return studyKey;
    }

    public void setStudyKey(String studyKey) {
        this.studyKey = studyKey;
    }

    public String getInternalRecordId() {
        return internalRecordId;
    }

    public void setInternalRecordId(String internalRecordId) {
        this.internalRecordId = internalRecordId;
    }
}
