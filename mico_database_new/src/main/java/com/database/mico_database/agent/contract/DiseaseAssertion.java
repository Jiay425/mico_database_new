package com.database.mico_database.agent.contract;

import java.time.Instant;

/** Typed disease assertion payload; mapping may remain unmapped until reviewed. */
public class DiseaseAssertion {

    private String rawLabel;
    private String canonicalDiseaseId;
    private String canonicalNameZh;
    private String mappingStatus;
    private Double mappingConfidence;
    private String mappingVersion;
    private String sourceEvidence;
    private String reviewedBy;
    private Instant reviewedAt;

    public String getRawLabel() {
        return rawLabel;
    }

    public void setRawLabel(String rawLabel) {
        this.rawLabel = rawLabel;
    }

    public String getCanonicalDiseaseId() {
        return canonicalDiseaseId;
    }

    public void setCanonicalDiseaseId(String canonicalDiseaseId) {
        this.canonicalDiseaseId = canonicalDiseaseId;
    }

    public String getCanonicalNameZh() {
        return canonicalNameZh;
    }

    public void setCanonicalNameZh(String canonicalNameZh) {
        this.canonicalNameZh = canonicalNameZh;
    }

    public String getMappingStatus() {
        return mappingStatus;
    }

    public void setMappingStatus(String mappingStatus) {
        this.mappingStatus = mappingStatus;
    }

    public Double getMappingConfidence() {
        return mappingConfidence;
    }

    public void setMappingConfidence(Double mappingConfidence) {
        this.mappingConfidence = mappingConfidence;
    }

    public String getMappingVersion() {
        return mappingVersion;
    }

    public void setMappingVersion(String mappingVersion) {
        this.mappingVersion = mappingVersion;
    }

    public String getSourceEvidence() {
        return sourceEvidence;
    }

    public void setSourceEvidence(String sourceEvidence) {
        this.sourceEvidence = sourceEvidence;
    }

    public String getReviewedBy() {
        return reviewedBy;
    }

    public void setReviewedBy(String reviewedBy) {
        this.reviewedBy = reviewedBy;
    }

    public Instant getReviewedAt() {
        return reviewedAt;
    }

    public void setReviewedAt(Instant reviewedAt) {
        this.reviewedAt = reviewedAt;
    }
}
