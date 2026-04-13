package com.database.mico_ai.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "app.ai")
public class AiProviderProperties {

    private boolean providerEnabled;
    private boolean summaryEnabled = true;

    public boolean isProviderEnabled() {
        return providerEnabled;
    }

    public void setProviderEnabled(boolean providerEnabled) {
        this.providerEnabled = providerEnabled;
    }

    public boolean isSummaryEnabled() {
        return summaryEnabled;
    }

    public void setSummaryEnabled(boolean summaryEnabled) {
        this.summaryEnabled = summaryEnabled;
    }
}