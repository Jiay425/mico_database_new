package com.database.mico_ai.dto;

import java.util.List;
import java.util.Map;

public record StagedExecutionResult(
        boolean enabled,
        String flowId,
        String qualityGate,
        String finalSummary,
        List<Map<String, Object>> stages,
        Map<String, Object> metadata
) {
}