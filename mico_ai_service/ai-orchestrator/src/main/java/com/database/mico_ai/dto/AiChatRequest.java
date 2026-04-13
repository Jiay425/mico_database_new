package com.database.mico_ai.dto;

import jakarta.validation.constraints.NotBlank;

public record AiChatRequest(
        @NotBlank(message = "sessionId is required") String sessionId,
        @NotBlank(message = "message is required") String message,
        AiContext context
) {
}
