package com.database.mico_ai.client;

import com.database.mico_ai.config.UpstreamProperties;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;

import java.util.LinkedHashMap;
import java.util.Map;

@Component
public class BusinessApiClient {

    private final RestClient restClient;
    private final ObjectMapper objectMapper;

    public BusinessApiClient(RestClient.Builder builder, UpstreamProperties properties, ObjectMapper objectMapper) {
        this.restClient = builder
                .baseUrl(trimBaseUrl(properties.getBusinessBaseUrl()))
                .defaultHeader(HttpHeaders.ACCEPT, MediaType.APPLICATION_JSON_VALUE)
                .build();
        this.objectMapper = objectMapper;
    }

    public Map<String, Object> getDashboardSummary() {
        return getJson("/dashboard/summary");
    }

    public Map<String, Object> searchPatients(String queryType, String queryValue, int page, int size) {
        return getJson(restClient.get()
                .uri(uriBuilder -> uriBuilder
                        .path("/patients/search")
                        .queryParam("queryType", queryType)
                        .queryParam("queryValue", queryValue)
                        .queryParam("page", page)
                        .queryParam("size", size)
                        .build()));
    }

    public Map<String, Object> getPatient(String patientId) {
        return getJson("/patients/{patientId}", patientId);
    }

    public Map<String, Object> listPatientSamples(String patientId) {
        return getJson("/patients/{patientId}/samples", patientId);
    }

    public Map<String, Object> getSampleTopFeatures(String patientId, String sampleId, int limit) {
        return getJson(restClient.get()
                .uri(uriBuilder -> uriBuilder
                        .path("/patients/{patientId}/samples/{sampleId}/top-features")
                        .queryParam("limit", limit)
                        .build(patientId, sampleId)));
    }

    public Map<String, Object> getPredictionPayload(String patientId, String sampleId) {
        return getJson("/patients/{patientId}/samples/{sampleId}/prediction-payload", patientId, sampleId);
    }

    public Map<String, Object> getHealthyReference(String patientId, String sampleId, int limit) {
        return getJson(restClient.get()
                .uri(uriBuilder -> uriBuilder
                        .path("/patients/{patientId}/samples/{sampleId}/healthy-reference")
                        .queryParam("limit", limit)
                        .build(patientId, sampleId)));
    }

    public Map<String, Object> runPrediction(Map<String, Object> payload) {
        try {
            String raw = restClient.post()
                    .uri("/predict/run")
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(payload)
                    .retrieve()
                    .body(String.class);
            return parseJsonBody(raw, "/predict/run");
        } catch (Exception ex) {
            return upstreamError("/predict/run", ex.getMessage());
        }
    }

    private String trimBaseUrl(String value) {
        if (value == null || value.isBlank()) {
            return "http://localhost:5000/api";
        }
        return value.endsWith("/") ? value.substring(0, value.length() - 1) : value;
    }

    private Map<String, Object> getJson(String path, Object... uriVariables) {
        try {
            String raw = restClient.get()
                    .uri(path, uriVariables)
                    .retrieve()
                    .body(String.class);
            return parseJsonBody(raw, path);
        } catch (Exception ex) {
            return upstreamError(path, ex.getMessage());
        }
    }

    private Map<String, Object> getJson(RestClient.RequestHeadersSpec<?> spec) {
        try {
            String raw = spec.retrieve().body(String.class);
            return parseJsonBody(raw, "dynamic-get");
        } catch (Exception ex) {
            return upstreamError("dynamic-get", ex.getMessage());
        }
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> parseJsonBody(String raw, String path) {
        if (raw == null || raw.isBlank()) {
            return upstreamError(path, "Empty response body");
        }

        String body = raw.trim();
        if (body.startsWith("<")) {
            return upstreamError(path, "Upstream returned HTML instead of JSON");
        }

        try {
            return objectMapper.readValue(body, Map.class);
        } catch (Exception ex) {
            return upstreamError(path, ex.getMessage());
        }
    }

    private Map<String, Object> upstreamError(String path, String detail) {
        Map<String, Object> response = new LinkedHashMap<>();
        response.put("success", false);
        response.put("error", "Upstream API request failed");
        response.put("path", path);
        response.put("detail", detail);
        return response;
    }
}
