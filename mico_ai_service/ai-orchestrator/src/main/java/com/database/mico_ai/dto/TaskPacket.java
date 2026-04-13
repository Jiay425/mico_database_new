package com.database.mico_ai.dto;

import java.util.List;
import java.util.Map;

public record TaskPacket(
        String sessionId,
        String message,
        AiContext context,
        String intent,
        String riskLevel,
        List<String> availableTools,
        Map<String, Object> executionPolicy,
        Map<String, Object> evidencePolicy,
        long createdAt
) {
}