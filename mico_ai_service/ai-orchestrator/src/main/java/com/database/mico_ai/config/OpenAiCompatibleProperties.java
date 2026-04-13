package com.database.mico_ai.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "app.ai.openai")
public class OpenAiCompatibleProperties {

    private String apiKey;
    private String baseUrl;
    private String modelName = "gpt-4o";
    private Double temperature = 0.2D;
    private boolean logRequests;
    private boolean logResponses;

    public String getApiKey() {
        return apiKey;
    }

    public void setApiKey(String apiKey) {
        this.apiKey = apiKey;
    }

    public String getBaseUrl() {
        return baseUrl;
    }

    public void setBaseUrl(String baseUrl) {
        this.baseUrl = baseUrl;
    }

    public String getModelName() {
        return modelName;
    }

    public void setModelName(String modelName) {
        this.modelName = modelName;
    }

    public Double getTemperature() {
        return temperature;
    }

    public void setTemperature(Double temperature) {
        this.temperature = temperature;
    }

    public boolean isLogRequests() {
        return logRequests;
    }

    public void setLogRequests(boolean logRequests) {
        this.logRequests = logRequests;
    }

    public boolean isLogResponses() {
        return logResponses;
    }

    public void setLogResponses(boolean logResponses) {
        this.logResponses = logResponses;
    }

    public boolean isConfigured() {
        return apiKey != null && !apiKey.isBlank()
                && baseUrl != null && !baseUrl.isBlank()
                && modelName != null && !modelName.isBlank();
    }
}
