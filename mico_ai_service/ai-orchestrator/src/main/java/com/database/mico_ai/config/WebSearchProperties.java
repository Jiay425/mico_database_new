package com.database.mico_ai.config;

import java.util.List;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "app.web-search")
public class WebSearchProperties {

    private boolean enabled;
    private String provider = "tavily";
    private String apiKey;
    private String baseUrl = "https://api.tavily.com";
    private String searchPath = "/search";
    private int maxResults = 3;
    private String searchDepth = "basic";
    private String topic = "general";
    private boolean includeAnswer;
    private List<String> medicalAllowedDomains = List.of(
            "pubmed.ncbi.nlm.nih.gov",
            "pmc.ncbi.nlm.nih.gov",
            "nih.gov",
            "who.int",
            "cdc.gov",
            "mayoclinic.org",
            "clevelandclinic.org",
            "hopkinsmedicine.org"
    );

    public boolean isEnabled() {
        return enabled;
    }

    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
    }

    public String getProvider() {
        return provider;
    }

    public void setProvider(String provider) {
        this.provider = provider;
    }

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

    public String getSearchPath() {
        return searchPath;
    }

    public void setSearchPath(String searchPath) {
        this.searchPath = searchPath;
    }

    public int getMaxResults() {
        return maxResults;
    }

    public void setMaxResults(int maxResults) {
        this.maxResults = maxResults;
    }

    public String getSearchDepth() {
        return searchDepth;
    }

    public void setSearchDepth(String searchDepth) {
        this.searchDepth = searchDepth;
    }

    public String getTopic() {
        return topic;
    }

    public void setTopic(String topic) {
        this.topic = topic;
    }

    public boolean isIncludeAnswer() {
        return includeAnswer;
    }

    public void setIncludeAnswer(boolean includeAnswer) {
        this.includeAnswer = includeAnswer;
    }

    public List<String> getMedicalAllowedDomains() {
        return medicalAllowedDomains;
    }

    public void setMedicalAllowedDomains(List<String> medicalAllowedDomains) {
        this.medicalAllowedDomains = medicalAllowedDomains;
    }

    public boolean isConfigured() {
        return enabled
                && apiKey != null && !apiKey.isBlank()
                && baseUrl != null && !baseUrl.isBlank();
    }
}