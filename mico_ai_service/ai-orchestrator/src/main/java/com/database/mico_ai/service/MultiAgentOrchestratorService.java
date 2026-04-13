package com.database.mico_ai.service;

import com.database.mico_ai.dto.AiCard;
import com.database.mico_ai.dto.AiChatResponse;
import com.database.mico_ai.dto.AiIntent;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Service
public class MultiAgentOrchestratorService {

    private final LangChainAgentService langChainAgentService;

    public MultiAgentOrchestratorService(LangChainAgentService langChainAgentService) {
        this.langChainAgentService = langChainAgentService;
    }

    public AiChatResponse compose(String question, AiIntent intent, AiChatResponse baseResponse) {
        Map<String, Object> metadata = new LinkedHashMap<>();
        if (baseResponse.metadata() != null) {
            metadata.putAll(baseResponse.metadata());
        }

        Map<String, Object> specialistPayload = new LinkedHashMap<>();
        specialistPayload.put("intent", intent.name());
        specialistPayload.put("summary", baseResponse.summary());
        specialistPayload.put("cards", baseResponse.cards());
        specialistPayload.put("actions", baseResponse.actions());
        specialistPayload.put("metadata", metadata);
        specialistPayload.put("analysisLayers", metadata.get("analysisLayers"));
        specialistPayload.put("evidenceChannels", metadata.get("evidenceChannels"));

        String dataInsight = defaultDataInsight(intent, metadata);
        String knowledgeInsight = defaultKnowledgeInsight(metadata);
        String predictionInsight = defaultPredictionInsight(intent, metadata);

        String dataAgentOutput = langChainAgentService.runDataAgent(intent.name(), question, specialistPayload);
        if (!dataAgentOutput.isBlank()) {
            dataInsight = dataAgentOutput;
        }

        String knowledgeAgentOutput = langChainAgentService.runKnowledgeAgent(intent.name(), question, specialistPayload);
        if (!knowledgeAgentOutput.isBlank()) {
            knowledgeInsight = knowledgeAgentOutput;
        }

        String predictionAgentOutput = langChainAgentService.runPredictionAgent(intent.name(), question, specialistPayload);
        if (!predictionAgentOutput.isBlank()) {
            predictionInsight = predictionAgentOutput;
        }

        Map<String, Object> agentInsights = new LinkedHashMap<>();
        agentInsights.put("dataAnalysisAgent", dataInsight);
        agentInsights.put("knowledgeRetrievalAgent", knowledgeInsight);
        agentInsights.put("predictionExplanationAgent", predictionInsight);

        Map<String, Object> coordinatorPayload = new LinkedHashMap<>();
        coordinatorPayload.put("intent", intent.name());
        coordinatorPayload.put("question", question);
        coordinatorPayload.put("specialists", agentInsights);
        coordinatorPayload.put("metadata", metadata);

        String coordinatorSummary = defaultCoordinatorSummary(intent, dataInsight, knowledgeInsight, predictionInsight);
        String coordinatorOutput = langChainAgentService.runCoordinatorAgent(intent.name(), question, coordinatorPayload);
        if (!coordinatorOutput.isBlank()) {
            coordinatorSummary = coordinatorOutput;
        }
        agentInsights.put("coordinatorAgent", coordinatorSummary);

        metadata.put("multiAgentMode", true);
        metadata.put("orchestrationMode", "multi-agent-orchestrator");
        metadata.put("activeAgents", activeAgents(intent));
        metadata.put("agentInsights", agentInsights);

        List<AiCard> cards = new ArrayList<>(baseResponse.cards() == null ? List.of() : baseResponse.cards());
        cards.add(0, new AiCard("multi_agent", title(intent), "当前由数据分析、知识检索、预测解释、总控四类 agent 协同生成结果。"));
        cards.add(1, new AiCard("agent_insights", "多智能体结论", buildAgentCardContent(dataInsight, knowledgeInsight, predictionInsight)));

        List<String> actions = new ArrayList<>(baseResponse.actions() == null ? List.of() : baseResponse.actions());
        addAction(actions, "查看数据分析 agent 结论");
        addAction(actions, "查看知识检索 agent 结论");
        addAction(actions, "查看预测解释 agent 结论");

        return new AiChatResponse(
                baseResponse.success(),
                baseResponse.intent(),
                coordinatorSummary,
                cards,
                actions,
                metadata
        );
    }

    private String defaultDataInsight(AiIntent intent, Map<String, Object> metadata) {
        return switch (intent) {
            case DASHBOARD_QA -> joinNonBlank(
                    "首页重点围绕整体样本规模、主要疾病和主要地区展开。",
                    value(metadata, "topDisease"),
                    value(metadata, "topRegion")
            );
            case PREDICTION_ASSISTANT -> joinNonBlank(
                    "当前预测页的核心是样本级模型输出。",
                    value(metadata, "sampleId"),
                    value(metadata, "predictionTop1Label")
            );
            case PATIENT_ANALYSIS, UNKNOWN -> joinNonBlank(
                    "当前患者分析首先基于样本结构与健康对照。",
                    value(metadata, "sampleId"),
                    value(metadata, "healthyDeviationLabel")
            );
        };
    }

    private String defaultKnowledgeInsight(Map<String, Object> metadata) {
        String internal = metadata.containsKey("knowledgeHits") ? "已命中内部知识库规则。" : "当前未命中内部知识库规则。";
        String external = metadata.containsKey("webEvidenceHits") ? "已补充外部网页证据。" : "当前未补充外部网页证据。";
        return internal + external;
    }

    private String defaultPredictionInsight(AiIntent intent, Map<String, Object> metadata) {
        if (intent == AiIntent.DASHBOARD_QA) {
            return "首页场景以统计解读为主，预测解释不是首要层。";
        }
        return joinNonBlank(
                "模型当前主方向：" + value(metadata, "predictionTop1Label"),
                "次方向：" + value(metadata, "predictionTop2Label"),
                "一致性：" + value(metadata, "predictionConsistency")
        );
    }

    private String defaultCoordinatorSummary(AiIntent intent, String dataInsight, String knowledgeInsight, String predictionInsight) {
        return switch (intent) {
            case DASHBOARD_QA -> "多智能体已从首页统计、知识规则和证据层完成协同分析，可继续追问主要疾病、地区来源和群体结构。";
            case PREDICTION_ASSISTANT -> "多智能体已联合样本预测、模型边界与知识证据完成解释，可继续追问 Top1/Top2 差异和证据支持。";
            case PATIENT_ANALYSIS, UNKNOWN -> "多智能体已联合样本结构、健康对照、模型判断和知识证据完成患者分析，可继续追问最关键的偏离与下一步动作。";
        };
    }

    private List<String> activeAgents(AiIntent intent) {
        return switch (intent) {
            case DASHBOARD_QA -> List.of("dataAnalysisAgent", "knowledgeRetrievalAgent", "coordinatorAgent");
            case PREDICTION_ASSISTANT -> List.of("dataAnalysisAgent", "predictionExplanationAgent", "knowledgeRetrievalAgent", "coordinatorAgent");
            case PATIENT_ANALYSIS, UNKNOWN -> List.of("dataAnalysisAgent", "predictionExplanationAgent", "knowledgeRetrievalAgent", "coordinatorAgent");
        };
    }

    private String buildAgentCardContent(String dataInsight, String knowledgeInsight, String predictionInsight) {
        return "数据分析：" + safe(dataInsight)
                + " 知识检索：" + safe(knowledgeInsight)
                + " 预测解释：" + safe(predictionInsight);
    }

    private String title(AiIntent intent) {
        return switch (intent) {
            case DASHBOARD_QA -> "多智能体首页协同";
            case PREDICTION_ASSISTANT -> "多智能体预测协同";
            case PATIENT_ANALYSIS, UNKNOWN -> "多智能体患者协同";
        };
    }

    private String joinNonBlank(String... parts) {
        List<String> values = new ArrayList<>();
        for (String part : parts) {
            if (part != null && !part.isBlank()) {
                values.add(part.trim());
            }
        }
        return String.join(" ", values);
    }

    private String value(Map<String, Object> metadata, String key) {
        Object value = metadata.get(key);
        return value == null ? "" : String.valueOf(value).trim();
    }

    private String safe(String value) {
        return value == null || value.isBlank() ? "暂无" : value;
    }

    private void addAction(List<String> actions, String action) {
        if (action == null || action.isBlank() || actions.contains(action)) {
            return;
        }
        actions.add(action);
    }
}