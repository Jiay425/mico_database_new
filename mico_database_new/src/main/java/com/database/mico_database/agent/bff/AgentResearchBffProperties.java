package com.database.mico_database.agent.bff;

import java.net.URI;

/** Default-closed configuration for the one browser research entry point. */
public final class AgentResearchBffProperties {

    private final boolean enabled;
    private final String runtimeBaseUrl;
    private final String runtimeToken;

    public AgentResearchBffProperties(boolean enabled, String runtimeBaseUrl, String runtimeToken) {
        this.enabled = enabled;
        this.runtimeBaseUrl = normalize(runtimeBaseUrl);
        this.runtimeToken = normalize(runtimeToken);
    }

    public boolean isEnabled() {
        return enabled;
    }

    public String getRuntimeEndpoint() {
        if (runtimeBaseUrl == null) {
            return null;
        }
        return (runtimeBaseUrl.endsWith("/") ? runtimeBaseUrl : runtimeBaseUrl + "/")
                + "internal/runtime/intent-runs";
    }

    public String getRuntimeToken() {
        return runtimeToken;
    }

    public boolean isUsable() {
        if (!enabled || runtimeBaseUrl == null || runtimeToken == null) {
            return false;
        }
        try {
            URI uri = URI.create(runtimeBaseUrl);
            return ("http".equalsIgnoreCase(uri.getScheme())
                    || "https".equalsIgnoreCase(uri.getScheme()))
                    && uri.getHost() != null
                    && uri.getUserInfo() == null
                    && uri.getQuery() == null
                    && uri.getFragment() == null;
        } catch (IllegalArgumentException exception) {
            return false;
        }
    }

    private String normalize(String value) {
        return value == null || value.trim().isEmpty() ? null : value.trim();
    }
}
