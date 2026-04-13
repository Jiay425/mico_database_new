package com.database.mico_database.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

import java.util.Collections;
import java.util.List;

@Data
public class PredictionRequest {

    @JsonProperty("raw_data")
    private String rawData;

    @JsonProperty("sample_id")
    private String sampleId;

    private String source;

    private List<FeatureItem> features;

    public boolean hasPayload() {
        return (rawData != null && !rawData.trim().isEmpty()) || !safeFeatures().isEmpty();
    }

    public List<FeatureItem> safeFeatures() {
        return features == null ? Collections.emptyList() : features;
    }

    @Data
    public static class FeatureItem {
        private String name;
        private Double value;
        private String meta;
    }
}
