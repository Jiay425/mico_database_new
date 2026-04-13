package com.database.mico_ai.service;

import com.database.mico_ai.config.FlowEngineProperties;
import com.database.mico_ai.dto.AiChatResponse;
import com.database.mico_ai.dto.AiContext;
import com.database.mico_ai.dto.AiIntent;
import com.database.mico_ai.dto.StagedExecutionResult;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

@Service
public class StagedExecutionStateMachineService {

    private final FlowEngineProperties flowEngineProperties;
    private final FlowDefinitionRegistryService flowDefinitionRegistryService;
    private final LangChainAgentService langChainAgentService;
    private final ObjectMapper objectMapper;

    public StagedExecutionStateMachineService(FlowEngineProperties flowEngineProperties,
                                              FlowDefinitionRegistryService flowDefinitionRegistryService,
                                              LangChainAgentService langChainAgentService,
                                              ObjectMapper objectMapper) {
        this.flowEngineProperties = flowEngineProperties;
        this.flowDefinitionRegistryService = flowDefinitionRegistryService;
        this.langChainAgentService = langChainAgentService;
        this.objectMapper = objectMapper;
    }

    public StagedExecutionResult execute(String message,
                                         AiIntent intent,
                                         AiContext context,
                                         AiChatResponse baseResponse) {
        if (!flowEngineProperties.isEnabled()) {
            return disabledResult("staged-flow-disabled");
        }

        FlowEngineProperties.FlowDefinition flow = flowDefinitionRegistryService.flowOf(intent).orElse(null);
        if (flow == null) {
            return disabledResult("flow-not-found");
        }

        List<Map<String, Object>> stageOutputs = new ArrayList<>();
        String rollingInput = message == null ? "" : message;
        String finalSummary = "";
        String qualityGate = "PASS";

        for (FlowEngineProperties.FlowStep step : flow.getSteps()) {
            String role = normalizeRole(step.getRole());
            String prompt = renderPrompt(step.getPromptTemplate(), rollingInput, message, intent, context, baseResponse, stageOutputs);
            String output = executeRole(role, intent, rollingInput, prompt, context, baseResponse, stageOutputs);
            if (output == null || output.isBlank()) {
                output = fallbackOutput(role, baseResponse);
            }

            Map<String, Object> stage = new LinkedHashMap<>();
            stage.put("key", nullSafe(step.getKey()));
            stage.put("role", role);
            stage.put("sequence", step.getSequence());
            stage.put("required", step.isRequired());
            stage.put("completionSignal", nullSafe(step.getCompletionSignal()));
            stage.put("output", output);
            stage.put("completed", matchesCompletion(output, step.getCompletionSignal()));
            stageOutputs.add(stage);

            rollingInput = output;
            if ("quality_supervisor".equals(role)) {
                qualityGate = evaluateQualityGate(output);
            }
            if ("response_assistant".equals(role) || "coordinator".equals(role)) {
                finalSummary = output;
            }
        }

        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("stagedFlowEnabled", true);
        metadata.put("stagedFlowId", nullSafe(flow.getId()));
        metadata.put("stagedFlowIntent", intent == null ? "UNKNOWN" : intent.name());
        metadata.put("stagedFlowDescription", nullSafe(flow.getDescription()));
        metadata.put("stagedFlowStageCount", stageOutputs.size());
        metadata.put("stagedFlowQualityGate", qualityGate);
        metadata.put("stagedFlowOverrideSummary", flowEngineProperties.isOverrideSummary());
        return new StagedExecutionResult(true, nullSafe(flow.getId()), qualityGate, finalSummary, stageOutputs, metadata);
    }

    public AiChatResponse apply(AiChatResponse response, StagedExecutionResult result) {
        if (response == null || result == null || !result.enabled()) {
            return response;
        }

        Map<String, Object> metadata = new LinkedHashMap<>();
        if (response.metadata() != null) {
            metadata.putAll(response.metadata());
        }
        metadata.putAll(result.metadata());
        metadata.put("stagedFlowStages", result.stages());

        String summary = response.summary();
        if (flowEngineProperties.isOverrideSummary() && result.finalSummary() != null && !result.finalSummary().isBlank()) {
            summary = result.finalSummary();
        }

        List<String> actions = new ArrayList<>();
        if (response.actions() != null) {
            actions.addAll(response.actions());
        }
        if (!actions.contains("Review staged execution")) {
            actions.add("Review staged execution");
        }

        return new AiChatResponse(
                response.success(),
                response.intent(),
                summary,
                response.cards(),
                actions,
                metadata
        );
    }

    private StagedExecutionResult disabledResult(String reason) {
        return new StagedExecutionResult(false, "", "", "", List.of(), Map.of(
                "stagedFlowEnabled", false,
                "stagedFlowReason", reason
        ));
    }

    private String executeRole(String role,
                               AiIntent intent,
                               String question,
                               String prompt,
                               AiContext context,
                               AiChatResponse baseResponse,
                               List<Map<String, Object>> stageOutputs) {
        Map<String, Object> payload = buildPayload(intent, context, baseResponse, stageOutputs, prompt);
        String intentName = intent == null ? "UNKNOWN" : intent.name();
        String mergedQuestion = mergeQuestion(question, prompt);

        return switch (role) {
            case "analyzer", "data_analysis" -> langChainAgentService.runDataAgent(intentName, mergedQuestion, payload);
            case "executor", "precision_executor" ->
                    intent == AiIntent.PREDICTION_ASSISTANT
                            ? langChainAgentService.runPredictionAgent(intentName, mergedQuestion, payload)
                            : langChainAgentService.runDataAgent(intentName, mergedQuestion, payload);
            case "quality_supervisor", "knowledge_reviewer" -> langChainAgentService.runKnowledgeAgent(intentName, mergedQuestion, payload);
            case "response_assistant", "coordinator" -> langChainAgentService.runCoordinatorAgent(intentName, mergedQuestion, payload);
            default -> langChainAgentService.runDataAgent(intentName, mergedQuestion, payload);
        };
    }

    private Map<String, Object> buildPayload(AiIntent intent,
                                             AiContext context,
                                             AiChatResponse baseResponse,
                                             List<Map<String, Object>> stageOutputs,
                                             String prompt) {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("intent", intent == null ? "UNKNOWN" : intent.name());
        payload.put("context", context);
        payload.put("baseSummary", baseResponse == null ? "" : nullSafe(baseResponse.summary()));
        payload.put("baseCards", baseResponse == null ? List.of() : baseResponse.cards());
        payload.put("baseMetadata", baseResponse == null ? Map.of() : baseResponse.metadata());
        payload.put("stageOutputs", stageOutputs);
        payload.put("stagePrompt", prompt);
        return payload;
    }

    private String mergeQuestion(String question, String prompt) {
        String q = question == null ? "" : question.trim();
        if (prompt == null || prompt.isBlank()) {
            return q;
        }
        if (q.isBlank()) {
            return prompt;
        }
        return q + "\n\n" + prompt;
    }

    private String renderPrompt(String template,
                                String rollingInput,
                                String originalMessage,
                                AiIntent intent,
                                AiContext context,
                                AiChatResponse baseResponse,
                                List<Map<String, Object>> stageOutputs) {
        if (template == null || template.isBlank()) {
            return "";
        }

        Map<String, String> values = new LinkedHashMap<>();
        values.put("message", nullSafe(originalMessage));
        values.put("rolling_input", nullSafe(rollingInput));
        values.put("intent", intent == null ? "UNKNOWN" : intent.name());
        values.put("page", context == null ? "" : nullSafe(context.page()));
        values.put("patient_id", context == null ? "" : nullSafe(context.patientId()));
        values.put("sample_id", context == null ? "" : nullSafe(context.sampleId()));
        values.put("base_summary", baseResponse == null ? "" : nullSafe(baseResponse.summary()));
        values.put("stage_outputs_json", toJson(stageOutputs));
        values.put("base_metadata_json", baseResponse == null ? "{}" : toJson(baseResponse.metadata()));

        String rendered = template;
        for (Map.Entry<String, String> entry : values.entrySet()) {
            rendered = rendered.replace("{{" + entry.getKey() + "}}", entry.getValue());
        }
        return rendered;
    }

    private boolean matchesCompletion(String output, String completionSignal) {
        if (completionSignal == null || completionSignal.isBlank()) {
            return false;
        }
        String content = output == null ? "" : output;
        return content.toUpperCase(Locale.ROOT).contains(completionSignal.toUpperCase(Locale.ROOT));
    }

    private String evaluateQualityGate(String output) {
        String content = output == null ? "" : output.toUpperCase(Locale.ROOT);
        if (content.contains("FAIL")) {
            return "FAIL";
        }
        if (content.contains("OPTIMIZE") || content.contains("WARN")) {
            return "OPTIMIZE";
        }
        return "PASS";
    }

    private String fallbackOutput(String role, AiChatResponse baseResponse) {
        return switch (role) {
            case "analyzer", "data_analysis" -> "Analysis fallback: no additional stage insight available.";
            case "executor", "precision_executor" -> "Execution fallback: using baseline workflow output.";
            case "quality_supervisor", "knowledge_reviewer" -> "Quality fallback: PASS by baseline safeguards.";
            case "response_assistant", "coordinator" -> baseResponse == null ? "" : nullSafe(baseResponse.summary());
            default -> baseResponse == null ? "" : nullSafe(baseResponse.summary());
        };
    }

    private String normalizeRole(String role) {
        if (role == null || role.isBlank()) {
            return "analyzer";
        }
        return role.trim().toLowerCase(Locale.ROOT);
    }

    private String nullSafe(String value) {
        return value == null ? "" : value;
    }

    private String toJson(Object value) {
        try {
            return objectMapper.writeValueAsString(value == null ? Map.of() : value);
        } catch (Exception ex) {
            return "{}";
        }
    }
}