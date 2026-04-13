package com.database.mico_ai.service;

import com.database.mico_ai.agent.CoordinatorAiAgent;
import com.database.mico_ai.agent.DataAnalysisAiAgent;
import com.database.mico_ai.agent.KnowledgeReviewAiAgent;
import com.database.mico_ai.agent.PredictionExplanationAiAgent;
import com.database.mico_ai.agent.SummaryAiAgent;
import com.database.mico_ai.agent.TriageAiAgent;
import com.database.mico_ai.config.AiProviderProperties;
import com.database.mico_ai.dto.AiChatRequest;
import com.database.mico_ai.dto.AiChatResponse;
import com.database.mico_ai.dto.AiContext;
import com.database.mico_ai.dto.AiIntent;
import com.fasterxml.jackson.databind.ObjectMapper;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.DisabledChatModel;
import org.springframework.stereotype.Service;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

@Service
public class LangChainAgentService {

    private final TriageAiAgent triageAiAgent;
    private final SummaryAiAgent summaryAiAgent;
    private final DataAnalysisAiAgent dataAnalysisAiAgent;
    private final KnowledgeReviewAiAgent knowledgeReviewAiAgent;
    private final PredictionExplanationAiAgent predictionExplanationAiAgent;
    private final CoordinatorAiAgent coordinatorAiAgent;
    private final ChatModel chatModel;
    private final AiProviderProperties aiProviderProperties;
    private final ObjectMapper objectMapper;

    public LangChainAgentService(TriageAiAgent triageAiAgent,
                                 SummaryAiAgent summaryAiAgent,
                                 DataAnalysisAiAgent dataAnalysisAiAgent,
                                 KnowledgeReviewAiAgent knowledgeReviewAiAgent,
                                 PredictionExplanationAiAgent predictionExplanationAiAgent,
                                 CoordinatorAiAgent coordinatorAiAgent,
                                 ChatModel chatModel,
                                 AiProviderProperties aiProviderProperties,
                                 ObjectMapper objectMapper) {
        this.triageAiAgent = triageAiAgent;
        this.summaryAiAgent = summaryAiAgent;
        this.dataAnalysisAiAgent = dataAnalysisAiAgent;
        this.knowledgeReviewAiAgent = knowledgeReviewAiAgent;
        this.predictionExplanationAiAgent = predictionExplanationAiAgent;
        this.coordinatorAiAgent = coordinatorAiAgent;
        this.chatModel = chatModel;
        this.aiProviderProperties = aiProviderProperties;
        this.objectMapper = objectMapper;
    }

    public AiIntent resolveIntent(String message, AiContext context, AiIntent fallback) {
        if (!llmAvailable()) {
            return fallback;
        }

        try {
            String result = triageAiAgent.route(
                    message,
                    context == null ? "" : nullSafe(context.page()),
                    context == null ? "" : nullSafe(context.patientId()),
                    context == null ? "" : nullSafe(context.sampleId())
            );
            return parseIntent(result, fallback);
        } catch (Exception ex) {
            return fallback;
        }
    }

    public String runDataAgent(String intent, String question, Map<String, Object> payload) {
        return runSpecialist(() -> dataAnalysisAiAgent.analyze(intent, question, toJson(payload)));
    }

    public String runKnowledgeAgent(String intent, String question, Map<String, Object> payload) {
        return runSpecialist(() -> knowledgeReviewAiAgent.review(intent, question, toJson(payload)));
    }

    public String runPredictionAgent(String intent, String question, Map<String, Object> payload) {
        return runSpecialist(() -> predictionExplanationAiAgent.explain(intent, question, toJson(payload)));
    }

    public String runCoordinatorAgent(String intent, String question, Map<String, Object> payload) {
        return runSpecialist(() -> coordinatorAiAgent.coordinate(intent, question, toJson(payload)));
    }

    public AiChatResponse enhanceResponse(AiChatRequest request, AiContext context, AiChatResponse baseResponse) {
        Map<String, Object> metadata = new LinkedHashMap<>();
        if (baseResponse.metadata() != null) {
            metadata.putAll(baseResponse.metadata());
        }
        metadata.put("llmEnabled", aiProviderProperties.isProviderEnabled());
        metadata.put("llmAvailable", llmAvailable());

        if (!llmAvailable() || !aiProviderProperties.isSummaryEnabled()) {
            metadata.put("executionMode", "rule-based");
            return new AiChatResponse(
                    baseResponse.success(),
                    baseResponse.intent(),
                    baseResponse.summary(),
                    baseResponse.cards(),
                    baseResponse.actions(),
                    metadata
            );
        }

        try {
            Map<String, Object> payload = new LinkedHashMap<>();
            payload.put("context", context);
            payload.put("cards", baseResponse.cards());
            payload.put("actions", baseResponse.actions());
            payload.put("metadata", baseResponse.metadata());

            String summary = summaryAiAgent.summarize(
                    baseResponse.intent(),
                    request.message(),
                    objectMapper.writeValueAsString(payload)
            );

            metadata.put("executionMode", "langchain4j");
            return new AiChatResponse(
                    baseResponse.success(),
                    baseResponse.intent(),
                    summary == null || summary.isBlank() ? baseResponse.summary() : summary.trim(),
                    baseResponse.cards(),
                    baseResponse.actions(),
                    metadata
            );
        } catch (Exception ex) {
            metadata.put("executionMode", "rule-based");
            metadata.put("llmError", ex.getMessage());
            return new AiChatResponse(
                    baseResponse.success(),
                    baseResponse.intent(),
                    baseResponse.summary(),
                    baseResponse.cards(),
                    baseResponse.actions(),
                    metadata
            );
        }
    }

    public boolean llmAvailable() {
        return aiProviderProperties.isProviderEnabled() && !(chatModel instanceof DisabledChatModel);
    }

    private String runSpecialist(AgentCall call) {
        if (!llmAvailable()) {
            return "";
        }
        try {
            String result = call.run();
            return result == null ? "" : result.trim();
        } catch (Exception ex) {
            return "";
        }
    }

    private String toJson(Map<String, Object> payload) {
        try {
            return objectMapper.writeValueAsString(payload == null ? Map.of() : payload);
        } catch (Exception ex) {
            return "{}";
        }
    }

    private AiIntent parseIntent(String raw, AiIntent fallback) {
        if (raw == null) {
            return fallback;
        }
        String normalized = raw.trim().toUpperCase();
        for (AiIntent intent : AiIntent.values()) {
            if (Objects.equals(intent.name(), normalized)) {
                return intent;
            }
        }
        return fallback;
    }

    private String nullSafe(String value) {
        return value == null ? "" : value;
    }

    @FunctionalInterface
    private interface AgentCall {
        String run() throws Exception;
    }
}