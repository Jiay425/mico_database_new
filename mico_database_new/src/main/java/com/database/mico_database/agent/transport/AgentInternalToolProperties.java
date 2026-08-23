package com.database.mico_database.agent.transport;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;

/** Default-closed settings for the internal M2M endpoint. */
public final class AgentInternalToolProperties {

    private final boolean enabled;
    private final String serviceToken;

    public AgentInternalToolProperties(boolean enabled, String serviceToken) {
        this.enabled = enabled;
        this.serviceToken = normalize(serviceToken);
    }

    public boolean isEnabled() {
        return enabled;
    }

    public boolean matchesBearerToken(String authorizationHeader) {
        String presented = extractBearerToken(authorizationHeader);
        byte[] expectedBytes = serviceToken == null
                ? new byte[0] : serviceToken.getBytes(StandardCharsets.UTF_8);
        byte[] presentedBytes = presented == null
                ? new byte[0] : presented.getBytes(StandardCharsets.UTF_8);
        return serviceToken != null
                && MessageDigest.isEqual(expectedBytes, presentedBytes);
    }

    private String extractBearerToken(String authorizationHeader) {
        if (authorizationHeader == null || !authorizationHeader.startsWith("Bearer ")) {
            return null;
        }
        String token = authorizationHeader.substring("Bearer ".length());
        return token.trim().isEmpty() ? null : token;
    }

    private String normalize(String token) {
        return token == null || token.trim().isEmpty() ? null : token;
    }
}
