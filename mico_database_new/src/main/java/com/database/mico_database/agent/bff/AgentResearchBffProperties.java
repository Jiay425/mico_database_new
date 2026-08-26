package com.database.mico_database.agent.bff;

import java.net.URI;

/** Default-closed configuration for the one browser research entry point. */
public final class AgentResearchBffProperties {

    public static final String INTENT_ENDPOINT_PATH = "/internal/runtime/intent-runs";
    public static final String SCIENTIFIC_ENDPOINT_PATH = "/internal/runtime/scientific-runs";

    private final boolean enabled;
    private final String runtimeBaseUrl;
    private final String runtimeToken;
    private final String runtimeEndpointPath;

    public AgentResearchBffProperties(boolean enabled, String runtimeBaseUrl, String runtimeToken) {
        this(enabled, runtimeBaseUrl, runtimeToken, INTENT_ENDPOINT_PATH);
    }

    public AgentResearchBffProperties(boolean enabled, String runtimeBaseUrl, String runtimeToken,
                                      String runtimeEndpointPath) {
        this.enabled = enabled;
        this.runtimeBaseUrl = normalize(runtimeBaseUrl);
        this.runtimeToken = normalize(runtimeToken);
        this.runtimeEndpointPath = normalizeEndpointPath(runtimeEndpointPath);
    }

    public boolean isEnabled() {
        return enabled;
    }

    public String getRuntimeEndpoint() {
        if (runtimeBaseUrl == null) {
            return null;
        }
        return (runtimeBaseUrl.endsWith("/") ? runtimeBaseUrl : runtimeBaseUrl + "/")
                + runtimeEndpointPath.substring(1);
    }

    public boolean isScientificEndpoint() {
        return SCIENTIFIC_ENDPOINT_PATH.equals(runtimeEndpointPath);
    }

    public String getRuntimeToken() {
        return runtimeToken;
    }

    public boolean isUsable() {
        if (!enabled || runtimeBaseUrl == null || runtimeToken == null || runtimeEndpointPath == null) {
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

    private String normalizeEndpointPath(String value) {
        String normalized = normalize(value);
        if (normalized == null) {
            return INTENT_ENDPOINT_PATH;
        }
        if (!normalized.startsWith("/")
                || !(INTENT_ENDPOINT_PATH.equals(normalized) || SCIENTIFIC_ENDPOINT_PATH.equals(normalized))) {
            return null;
        }
        return normalized;
    }
}
