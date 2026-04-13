package com.database.mico_database.service;

import com.database.mico_database.model.PredictionRequest;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.LinkedHashMap;
import java.util.Map;

@Service
public class ModelGatewayService {

    private final ObjectMapper objectMapper;
    private final RestTemplate restTemplate;

    @Value("${app.model.enabled:false}")
    private boolean modelEnabled;

    @Value("${app.model.service-url:http://localhost:5001}")
    private String modelServiceUrl;

    public ModelGatewayService(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper;

        SimpleClientHttpRequestFactory factory = new SimpleClientHttpRequestFactory();
        factory.setConnectTimeout(4000);
        factory.setReadTimeout(15000);
        this.restTemplate = new RestTemplate(factory);
    }

    public Map<String, Object> predict(PredictionRequest request) {
        return callModelService("/predict", request, "Prediction service is not ready.");
    }

    public Map<String, Object> abundanceCurves(PredictionRequest request) {
        return callModelService("/abundance_curves", request, "Abundance curve service is not ready.");
    }

    public Map<String, Object> pcoaData(PredictionRequest request) {
        return callModelService("/pcoa_data", request, "PCoA service is not ready.");
    }

    private Map<String, Object> callModelService(String path, PredictionRequest request, String notReadyMessage) {
        if (!modelEnabled) {
            return buildNotReadyResponse(notReadyMessage, "app.model.enabled is false");
        }

        try {
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            HttpEntity<String> entity = new HttpEntity<>(objectMapper.writeValueAsString(request), headers);
            String responseJson = restTemplate.postForObject(normalizeUrl(path), entity, String.class);
            if (responseJson == null || responseJson.trim().isEmpty()) {
                return buildNotReadyResponse(notReadyMessage, "Model service returned an empty response.");
            }
            return objectMapper.readValue(responseJson, new TypeReference<Map<String, Object>>() {});
        } catch (Exception ex) {
            return buildNotReadyResponse(notReadyMessage, ex.getMessage());
        }
    }

    private String normalizeUrl(String path) {
        String base = modelServiceUrl == null ? "http://localhost:5001" : modelServiceUrl.trim();
        if (base.endsWith("/")) {
            base = base.substring(0, base.length() - 1);
        }
        return base + path;
    }

    private Map<String, Object> buildNotReadyResponse(String message, String detail) {
        Map<String, Object> response = new LinkedHashMap<>();
        response.put("success", false);
        response.put("model_ready", false);
        response.put("error", message);
        response.put("detail", detail);
        response.put("service_url", modelServiceUrl);
        response.put("suggestion", "Train the model artifacts and start the inference service first.");
        return response;
    }
}
