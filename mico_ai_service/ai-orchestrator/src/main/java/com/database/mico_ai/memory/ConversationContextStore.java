package com.database.mico_ai.memory;

import com.database.mico_ai.dto.AiContext;
import org.springframework.stereotype.Component;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@Component
public class ConversationContextStore {

    private final Map<String, AiContext> sessionContext = new ConcurrentHashMap<>();

    public AiContext get(String sessionId) {
        return sessionContext.get(sessionId);
    }

    public void put(String sessionId, AiContext context) {
        if (sessionId != null && context != null) {
            sessionContext.put(sessionId, context);
        }
    }
}
