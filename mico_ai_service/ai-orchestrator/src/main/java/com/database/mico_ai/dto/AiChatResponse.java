package com.database.mico_ai.dto;

import java.util.List;
import java.util.Map;

public record AiChatResponse(
        boolean success,
        String intent,
        String summary,
        List<AiCard> cards,
        List<String> actions,
        Map<String, Object> metadata
) {
}
