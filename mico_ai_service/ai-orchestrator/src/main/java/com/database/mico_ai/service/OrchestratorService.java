package com.database.mico_ai.service;

import com.database.mico_ai.dto.AiChatRequest;
import com.database.mico_ai.dto.AiChatResponse;
import com.database.mico_ai.dto.AiContext;
import com.database.mico_ai.dto.AiCard;
import com.database.mico_ai.dto.AiIntent;
import com.database.mico_ai.dto.StagedExecutionResult;
import com.database.mico_ai.dto.TaskPacket;
import com.database.mico_ai.memory.ConversationContextStore;
import com.database.mico_ai.workflow.DashboardWorkflow;
import com.database.mico_ai.workflow.PatientWorkflow;
import com.database.mico_ai.workflow.PredictionWorkflow;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

@Service
public class OrchestratorService {

    private final IntentRouterService intentRouterService;
    private final LangChainAgentService langChainAgentService;
    private final ConversationContextStore conversationContextStore;
    private final DashboardWorkflow dashboardWorkflow;
    private final PatientWorkflow patientWorkflow;
    private final PredictionWorkflow predictionWorkflow;
    private final KnowledgeAugmentationService knowledgeAugmentationService;
    private final HybridResponseService hybridResponseService;
    private final MultiAgentOrchestratorService multiAgentOrchestratorService;
    private final TaskPacketService taskPacketService;
    private final TraceRecorderService traceRecorderService;
    private final StagedExecutionStateMachineService stagedExecutionStateMachineService;
    private final FlowDefinitionRegistryService flowDefinitionRegistryService;

    public OrchestratorService(IntentRouterService intentRouterService,
                               LangChainAgentService langChainAgentService,
                               ConversationContextStore conversationContextStore,
                               DashboardWorkflow dashboardWorkflow,
                               PatientWorkflow patientWorkflow,
                               PredictionWorkflow predictionWorkflow,
                               KnowledgeAugmentationService knowledgeAugmentationService,
                               HybridResponseService hybridResponseService,
                               MultiAgentOrchestratorService multiAgentOrchestratorService,
                               TaskPacketService taskPacketService,
                               TraceRecorderService traceRecorderService,
                               StagedExecutionStateMachineService stagedExecutionStateMachineService,
                               FlowDefinitionRegistryService flowDefinitionRegistryService) {
        this.intentRouterService = intentRouterService;
        this.langChainAgentService = langChainAgentService;
        this.conversationContextStore = conversationContextStore;
        this.dashboardWorkflow = dashboardWorkflow;
        this.patientWorkflow = patientWorkflow;
        this.predictionWorkflow = predictionWorkflow;
        this.knowledgeAugmentationService = knowledgeAugmentationService;
        this.hybridResponseService = hybridResponseService;
        this.multiAgentOrchestratorService = multiAgentOrchestratorService;
        this.taskPacketService = taskPacketService;
        this.traceRecorderService = traceRecorderService;
        this.stagedExecutionStateMachineService = stagedExecutionStateMachineService;
        this.flowDefinitionRegistryService = flowDefinitionRegistryService;
    }

    public AiChatResponse chat(AiChatRequest request) {
        AiContext mergedContext = mergeContext(request.sessionId(), request.context());
        AiIntent fallbackIntent = intentRouterService.resolveIntent(request.message(), mergedContext);
        AiIntent resolvedIntent = langChainAgentService.resolveIntent(request.message(), mergedContext, fallbackIntent);
        AiIntent effectiveIntent = effectiveIntentForQuestion(resolvedIntent, request.message(), mergedContext);
        TaskPacket taskPacket = taskPacketService.build(request.sessionId(), request.message(), mergedContext, effectiveIntent);
        Map<String, Object> trace = traceRecorderService.start(taskPacket);

        traceRecorderService.recordStep(trace, "context_merged", Map.of(
                "page", safe(mergedContext == null ? null : mergedContext.page()),
                "patientId", safe(mergedContext == null ? null : mergedContext.patientId()),
                "sampleId", safe(mergedContext == null ? null : mergedContext.sampleId())
        ));
        traceRecorderService.recordStep(trace, "intent_resolved", Map.of(
                "fallbackIntent", fallbackIntent.name(),
                "resolvedIntent", resolvedIntent.name(),
                "effectiveIntent", effectiveIntent.name(),
                "combinedToolEnforced", effectiveIntent != resolvedIntent
        ));

        conversationContextStore.put(request.sessionId(), mergedContext);
        traceRecorderService.recordStep(trace, "task_packet_created", Map.of(
                "riskLevel", taskPacket.riskLevel(),
                "availableTools", taskPacket.availableTools(),
                "executionPolicy", taskPacket.executionPolicy(),
                "evidencePolicy", taskPacket.evidencePolicy()
        ));

        AiChatResponse baseResponse;
        switch (effectiveIntent) {
            case DASHBOARD_QA:
                baseResponse = dashboardWorkflow.run();
                traceRecorderService.recordStep(trace, "workflow", Map.of("name", "DashboardWorkflow"));
                baseResponse = knowledgeAugmentationService.augmentDashboardResponse(request.message(), baseResponse);
                traceRecorderService.recordStep(trace, "knowledge_augmentation", evidenceSnapshot(baseResponse, "dashboard"));
                break;
            case PREDICTION_ASSISTANT:
                baseResponse = predictionWorkflow.run(mergedContext);
                traceRecorderService.recordStep(trace, "workflow", Map.of("name", "PredictionWorkflow"));
                baseResponse = knowledgeAugmentationService.augmentPredictionResponse(request.message(), baseResponse);
                traceRecorderService.recordStep(trace, "knowledge_augmentation", evidenceSnapshot(baseResponse, "prediction"));
                break;
            case PATIENT_ANALYSIS:
            case UNKNOWN:
            default:
                baseResponse = patientWorkflow.run(request.message(), mergedContext);
                traceRecorderService.recordStep(trace, "workflow", Map.of("name", "PatientWorkflow"));
                baseResponse = knowledgeAugmentationService.augmentPatientResponse(request.message(), baseResponse);
                traceRecorderService.recordStep(trace, "knowledge_augmentation", evidenceSnapshot(baseResponse, "patient"));
                break;
        }

        baseResponse = attachRuntimeMetadata(baseResponse, taskPacket, trace);
        baseResponse = hybridResponseService.compose(request.message(), effectiveIntent, baseResponse);
        traceRecorderService.recordStep(trace, "hybrid_orchestration", Map.of(
                "mode", "hybrid-single-orchestrator",
                "analysisLayers", safeList(baseResponse.metadata(), "analysisLayers"),
                "evidenceChannels", safeList(baseResponse.metadata(), "evidenceChannels")
        ));
        baseResponse = multiAgentOrchestratorService.compose(request.message(), effectiveIntent, baseResponse);
        traceRecorderService.recordStep(trace, "multi_agent_orchestration", Map.of(
                "mode", "multi-agent-orchestrator",
                "activeAgents", safeList(baseResponse.metadata(), "activeAgents"),
                "agentInsightKeys", safeMapKeys(baseResponse.metadata(), "agentInsights")
        ));
        StagedExecutionResult stagedExecutionResult = stagedExecutionStateMachineService.execute(
                request.message(),
                effectiveIntent,
                mergedContext,
                baseResponse
        );
        baseResponse = stagedExecutionStateMachineService.apply(baseResponse, stagedExecutionResult);
        traceRecorderService.recordStep(trace, "staged_execution", Map.of(
                "enabled", stagedExecutionResult.enabled(),
                "flowId", safe(stagedExecutionResult.flowId()),
                "qualityGate", safe(stagedExecutionResult.qualityGate()),
                "stageCount", stagedExecutionResult.stages() == null ? 0 : stagedExecutionResult.stages().size()
        ));
        baseResponse = polishForDisplay(baseResponse, effectiveIntent);
        traceRecorderService.recordStep(trace, "display_polish", Map.of(
                "summaryLength", baseResponse.summary() == null ? 0 : baseResponse.summary().length(),
                "cardCount", baseResponse.cards() == null ? 0 : baseResponse.cards().size(),
                "actionCount", baseResponse.actions() == null ? 0 : baseResponse.actions().size()
        ));

        Map<String, Object> finalTrace = traceRecorderService.finish(trace, baseResponse);
        baseResponse = attachTrace(baseResponse, finalTrace);
        return baseResponse;
    }

    private Map<String, Object> evidenceSnapshot(AiChatResponse response, String target) {
        Map<String, Object> metadata = response.metadata() == null ? Map.of() : response.metadata();
        return Map.of(
                "target", target,
                "knowledgeHitCount", metadata.getOrDefault("knowledgeHitCount", 0),
                "webEvidenceHitCount", metadata.getOrDefault("webEvidenceHitCount", 0),
                "webSearchEnabled", metadata.getOrDefault("webSearchEnabled", false),
                "knowledgeReady", metadata.getOrDefault("knowledgeReady", false)
        );
    }

    private AiChatResponse attachRuntimeMetadata(AiChatResponse response, TaskPacket taskPacket, Map<String, Object> trace) {
        Map<String, Object> metadata = new LinkedHashMap<>();
        if (response.metadata() != null) {
            metadata.putAll(response.metadata());
        }
        metadata.put("taskPacket", taskPacket);
        metadata.put("executionPolicy", taskPacket.executionPolicy());
        metadata.put("evidencePolicy", taskPacket.evidencePolicy());
        metadata.put("runtimeTrace", trace);
        metadata.put("harnessRuntime", true);
        metadata.putAll(flowDefinitionRegistryService.status());
        return new AiChatResponse(
                response.success(),
                response.intent(),
                response.summary(),
                response.cards(),
                response.actions(),
                metadata
        );
    }

    private AiChatResponse attachTrace(AiChatResponse response, Map<String, Object> finalTrace) {
        Map<String, Object> metadata = new LinkedHashMap<>();
        if (response.metadata() != null) {
            metadata.putAll(response.metadata());
        }
        metadata.put("runtimeTrace", finalTrace);
        metadata.put("review", finalTrace.getOrDefault("review", Map.of()));
        return new AiChatResponse(
                response.success(),
                response.intent(),
                response.summary(),
                response.cards(),
                response.actions(),
                metadata
        );
    }

    private AiContext mergeContext(String sessionId, AiContext incoming) {
        AiContext stored = conversationContextStore.get(sessionId);
        if (stored == null) {
            return incoming;
        }
        if (incoming == null) {
            return stored;
        }
        return new AiContext(
                incoming.page() != null ? incoming.page() : stored.page(),
                incoming.patientId() != null ? incoming.patientId() : stored.patientId(),
                incoming.sampleId() != null ? incoming.sampleId() : stored.sampleId()
        );
    }

    private String safe(String value) {
        return value == null ? "" : value;
    }

    @SuppressWarnings("unchecked")
    private List<Object> safeList(Map<String, Object> metadata, String key) {
        if (metadata == null) {
            return List.of();
        }
        Object value = metadata.get(key);
        return value instanceof List<?> list ? (List<Object>) list : List.of();
    }

    @SuppressWarnings("unchecked")
    private List<String> safeMapKeys(Map<String, Object> metadata, String key) {
        if (metadata == null) {
            return List.of();
        }
        Object value = metadata.get(key);
        if (!(value instanceof Map<?, ?> map)) {
            return List.of();
        }
        return map.keySet().stream().map(String::valueOf).toList();
    }

    private AiChatResponse polishForDisplay(AiChatResponse response, AiIntent intent) {
        if (response == null) {
            return response;
        }
        String summary = buildPracticalSummary(intent, response.metadata(), response.summary());
        summary = enrichPatientSummary(intent, summary, response.metadata());
        List<AiCard> cards = compactCards(response.cards());
        List<String> actions = practicalActions(intent);
        return new AiChatResponse(
                response.success(),
                response.intent(),
                summary,
                cards,
                actions,
                response.metadata()
        );
    }

    private String buildPracticalSummary(AiIntent intent, Map<String, Object> metadata, String original) {
        String normalizedOriginal = normalizeText(original);
        Map<String, Object> safeMetadata = metadata == null ? Map.of() : metadata;
        String summary;
        switch (intent) {
            case DASHBOARD_QA -> {
                String total = read(safeMetadata, "totalPatients");
                String disease = read(safeMetadata, "topDisease");
                String region = read(safeMetadata, "topRegion");
                summary = "首页关键点：总样本" + fallback(total, "--") + "，高频疾病为" + fallback(disease, "--") + "，主要来源地区为" + fallback(region, "--") + "。";
            }
            case PREDICTION_ASSISTANT -> {
                String top1 = read(safeMetadata, "predictionTop1Label");
                String top1p = read(safeMetadata, "predictionTop1Probability");
                String top2 = read(safeMetadata, "predictionTop2Label");
                summary = "预测结论：Top1 为" + fallback(top1, "--") + "（" + fallback(top1p, "--") + "%），Top2 为" + fallback(top2, "--") + "，建议结合健康组偏离一起解读。";
            }
            case PATIENT_ANALYSIS, UNKNOWN -> {
                String sample = read(safeMetadata, "sampleId");
                String deviation = read(safeMetadata, "healthyDeviationLabel");
                String top1 = read(safeMetadata, "predictionTop1Label");
                String consistency = read(safeMetadata, "predictionConsistency");
                summary = "患者要点：样本" + fallback(sample, "--") + "相对健康组为" + fallback(deviation, "待补充") + "，模型主方向为" + fallback(top1, "--") + "（" + fallback(consistency, "一致性待确认") + "）。";
            }
            default -> summary = normalizedOriginal;
        }
        if (summary == null || summary.isBlank()) {
            summary = normalizedOriginal;
        }
        if (summary == null || summary.isBlank()) {
            return "已完成分析，请继续追问“与健康组差异在哪里”或“哪些特征最支持当前结论”。";
        }
        return summary;
    }

    private List<AiCard> compactCards(List<AiCard> cards) {
        if (cards == null || cards.isEmpty()) {
            return List.of();
        }
        Set<String> hidden = Set.of("multi_agent", "agent_insights", "hybrid_orchestration");
        Set<String> priority = Set.of("medical_evidence", "knowledge_support", "web_evidence");
        LinkedHashMap<String, AiCard> kept = new LinkedHashMap<>();

        // Keep evidence cards first so they are not squeezed out by generic cards.
        for (AiCard card : cards) {
            if (card == null) {
                continue;
            }
            String type = normalizeKey(card.type());
            if (!priority.contains(type)) {
                continue;
            }
            String title = normalizeText(card.title());
            String content = normalizeText(card.content());
            if (title.isBlank() || content.isBlank()) {
                continue;
            }
            String key = type + "|" + title;
            kept.putIfAbsent(key, new AiCard(type, title, content));
            if (kept.size() >= 4) {
                return new ArrayList<>(kept.values());
            }
        }

        for (AiCard card : cards) {
            if (card == null) {
                continue;
            }
            String type = normalizeKey(card.type());
            if (hidden.contains(type)) {
                continue;
            }
            String title = normalizeText(card.title());
            String content = normalizeText(card.content());
            if (title.isBlank() || content.isBlank()) {
                continue;
            }
            if (content.length() > 520) {
                content = content.substring(0, 520) + "...";
            }
            String key = type + "|" + title;
            kept.putIfAbsent(key, new AiCard(type, title, content));
            if (kept.size() >= 4) {
                break;
            }
        }
        return new ArrayList<>(kept.values());
    }

    private List<String> practicalActions(AiIntent intent) {
        if (intent == AiIntent.DASHBOARD_QA) {
            return List.of("当前最值得关注的疾病类别是什么", "地区分布是否有异常集中", "下一步应先看哪个人群层");
        }
        if (intent == AiIntent.PREDICTION_ASSISTANT) {
            return List.of("为什么不是 Top2", "结果的不确定性来自哪里", "哪些特征最影响当前预测");
        }
        return List.of("当前样本和健康组相比有什么区别", "相对健康基线偏离在哪里", "这些偏离和预测结果一致吗");
    }

    private String normalizeKey(String value) {
        return value == null ? "" : value.trim().toLowerCase();
    }

    private String normalizeText(String value) {
        if (value == null) {
            return "";
        }
        return value.replaceAll("\\s+", " ").replace("Fo.", "").trim();
    }

    private String read(Map<String, Object> metadata, String key) {
        Object value = metadata.get(key);
        if (value == null) {
            return "";
        }
        if (value instanceof List<?> list) {
            StringBuilder builder = new StringBuilder();
            for (Object item : list) {
                if (item == null) {
                    continue;
                }
                String text = String.valueOf(item).trim();
                if (text.isBlank()) {
                    continue;
                }
                if (builder.length() > 0) {
                    builder.append("、");
                }
                builder.append(text);
            }
            return builder.toString();
        }
        return String.valueOf(value).trim();
    }

    private String fallback(String value, String fallback) {
        return value == null || value.isBlank() ? fallback : value;
    }

    private String enrichPatientSummary(AiIntent intent, String summary, Map<String, Object> metadata) {
        if (intent != AiIntent.PATIENT_ANALYSIS && intent != AiIntent.UNKNOWN) {
            return summary;
        }
        Map<String, Object> safeMetadata = metadata == null ? Map.of() : metadata;
        String deviationTop = read(safeMetadata, "deviationTopTaxa");
        String spatialLevel = read(safeMetadata, "spatialDeviationLevel");
        if ((deviationTop == null || deviationTop.isBlank()) && (spatialLevel == null || spatialLevel.isBlank())) {
            return summary;
        }
        StringBuilder builder = new StringBuilder(summary == null ? "" : summary.trim());
        if (builder.length() > 0 && !builder.toString().endsWith("。")) {
            builder.append("。");
        }
        if (deviationTop != null && !deviationTop.isBlank()) {
            builder.append(" 联合差异菌重点：").append(deviationTop).append("。");
        }
        if (spatialLevel != null && !spatialLevel.isBlank()) {
            builder.append(" 与健康样本空间偏离等级：").append(spatialLevel).append("。");
        }
        return builder.toString();
    }

    private AiIntent effectiveIntentForQuestion(AiIntent resolvedIntent, String message, AiContext context) {
        if (resolvedIntent == AiIntent.DASHBOARD_QA) {
            return resolvedIntent;
        }
        if (!hasPatientContext(context)) {
            return resolvedIntent;
        }
        if (requiresCombinedTools(message)) {
            return AiIntent.PATIENT_ANALYSIS;
        }
        return resolvedIntent;
    }

    private boolean hasPatientContext(AiContext context) {
        return context != null
                && context.patientId() != null
                && !context.patientId().isBlank()
                && context.sampleId() != null
                && !context.sampleId().isBlank();
    }

    private boolean requiresCombinedTools(String message) {
        String text = message == null ? "" : message.toLowerCase(Locale.ROOT);
        if (text.isBlank()) {
            return false;
        }
        String[] triggers = new String[]{
                "为什么", "为何", "差异", "区别", "偏离", "偏差", "一致", "是否一致", "和健康",
                "健康组", "why", "difference", "deviation", "consistent", "consistency"
        };
        for (String trigger : triggers) {
            if (text.contains(trigger)) {
                return true;
            }
        }
        return false;
    }
}
