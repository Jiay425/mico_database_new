package com.database.mico_ai.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "app.upstream")
public class UpstreamProperties {

    private String businessBaseUrl = "http://localhost:5000/api";

    public String getBusinessBaseUrl() {
        return businessBaseUrl;
    }

    public void setBusinessBaseUrl(String businessBaseUrl) {
        this.businessBaseUrl = businessBaseUrl;
    }
}
