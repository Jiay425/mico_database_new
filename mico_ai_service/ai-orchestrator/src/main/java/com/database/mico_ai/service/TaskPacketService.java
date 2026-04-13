package com.database.mico_ai.service;

import com.database.mico_ai.dto.AiContext;
import com.database.mico_ai.dto.AiIntent;
import com.database.mico_ai.dto.TaskPacket;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

@Service
public class TaskPacketService {

    public TaskPacket build(String sessionId, String message, AiContext context, AiIntent intent) {
        String resolvedIntent = intent == null ? AiIntent.UNKNOWN.name() : intent.name();
        String riskLevel = resolveRiskLevel(message, intent, context);
        List<String> tools = resolveAvailableTools(intent, context);
        Map<String, Object> executionPolicy = buildExecutionPolicy(message, context, intent, tools, riskLevel);
        Map<String, Object> evidencePolicy = buildEvidencePolicy(intent, message);

        return new TaskPacket(
                sessionId,
                message,
                context,
                resolvedIntent,
                riskLevel,
                tools,
                executionPolicy,
                evidencePolicy,
                System.currentTimeMillis()
        );
    }

    private String resolveRiskLevel(String message, AiIntent intent, AiContext context) {
        String lower = normalize(message);
        boolean hasPatientContext = context != null
                && context.patientId() != null
                && !context.patientId().isBlank();
        boolean hasSampleContext = context != null
                && context.sampleId() != null
                && !context.sampleId().isBlank();

        if (containsAny(lower, "诊断", "确诊", "治疗", "用药", "处方", "干预", "手术")) {
            return "high";
        }
        if (intent == AiIntent.PREDICTION_ASSISTANT || hasPatientContext || hasSampleContext) {
            return "medium";
        }
        return "low";
    }

    private List<String> resolveAvailableTools(AiIntent intent, AiContext context) {
        List<String> tools = new ArrayList<>();
        tools.add("search_knowledge");
        tools.add("search_web_evidence");
        tools.add("mcp_resources");
        tools.add("mcp_prompts");

        if (intent == AiIntent.DASHBOARD_QA) {
            tools.add(0, "get_dashboard_summary");
            return tools;
        }

        boolean hasPatientContext = context != null
                && context.patientId() != null
                && !context.patientId().isBlank();
        boolean hasSampleContext = hasPatientContext
                && context.sampleId() != null
                && !context.sampleId().isBlank();

        if (hasPatientContext) {
            tools.add(0, "get_patient_summary");
        }
        if (hasSampleContext) {
            tools.add(1, "get_sample_top_features");
            tools.add(2, "get_healthy_reference");
            tools.add(3, "run_prediction");
        }
        return tools;
    }

    private Map<String, Object> buildExecutionPolicy(String message,
                                                     AiContext context,
                                                     AiIntent intent,
                                                     List<String> tools,
                                                     String riskLevel) {
        Map<String, Object> policy = new LinkedHashMap<>();
        boolean allowExternalEvidence = shouldAllowExternalEvidence(message);
        boolean canRunPrediction = tools.contains("run_prediction") && intent != AiIntent.DASHBOARD_QA;
        boolean hasSingleSample = context != null
                && context.sampleId() != null
                && !context.sampleId().isBlank();

        policy.put("allowInternalKnowledge", true);
        policy.put("allowExternalEvidence", allowExternalEvidence);
        policy.put("allowPrediction", canRunPrediction);
        policy.put("allowMcpTools", true);
        policy.put("allowMcpResources", true);
        policy.put("allowMcpPrompts", true);
        policy.put("mustUseInternalDataFirst", true);
        policy.put("mustKeepDiagnosisBoundary", true);
        policy.put("mustDiscloseEvidenceLayers", true);
        policy.put("requireSingleSample", hasSingleSample);
        policy.put("riskLevel", riskLevel);
        return policy;
    }

    private Map<String, Object> buildEvidencePolicy(AiIntent intent, String message) {
        Map<String, Object> policy = new LinkedHashMap<>();
        policy.put("requireDataObservation", true);
        policy.put("requireModelBoundary", intent == AiIntent.PATIENT_ANALYSIS || intent == AiIntent.PREDICTION_ASSISTANT);
        policy.put("requireInternalKnowledgePriority", true);
        policy.put("requireExternalEvidenceAsSupplement", true);
        policy.put("requireSourceDistinction", true);
        policy.put("requireLayeredOutput", true);
        policy.put("explicitlyRequestedExternalEvidence", shouldAllowExternalEvidence(message));
        return policy;
    }

    private boolean shouldAllowExternalEvidence(String message) {
        String lower = normalize(message);
        return containsAny(lower,
                "为什么", "机制", "文献", "资料", "研究", "证据", "指南", "最新", "官网", "联网", "搜索")
                || containsAny(lower, "latest", "evidence", "guideline", "literature", "search");
    }

    private boolean containsAny(String text, String... keywords) {
        for (String keyword : keywords) {
            if (text.contains(keyword.toLowerCase(Locale.ROOT))) {
                return true;
            }
        }
        return false;
    }

    private String normalize(String text) {
        return text == null ? "" : text.toLowerCase(Locale.ROOT);
    }
}