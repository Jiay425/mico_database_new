package com.database.mico_ai.service;

import com.database.mico_ai.dto.AiCard;
import com.database.mico_ai.dto.AiChatResponse;
import com.database.mico_ai.dto.AiIntent;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;

@Service
public class HybridResponseService {

    public AiChatResponse compose(String question, AiIntent intent, AiChatResponse baseResponse) {
        Map<String, Object> metadata = new LinkedHashMap<>();
        if (baseResponse.metadata() != null) {
            metadata.putAll(baseResponse.metadata());
        }

        List<Map<String, Object>> analysisLayers = switch (intent) {
            case DASHBOARD_QA -> buildDashboardLayers(metadata);
            case PREDICTION_ASSISTANT -> buildPredictionLayers(metadata);
            case PATIENT_ANALYSIS, UNKNOWN -> buildPatientLayers(metadata);
        };

        List<String> evidenceChannels = collectEvidenceChannels(metadata);
        metadata.put("hybridMode", true);
        metadata.put("orchestrationMode", "hybrid-single-orchestrator");
        metadata.put("analysisLayers", analysisLayers);
        metadata.put("evidenceChannels", evidenceChannels);
        metadata.put("userQuestion", question);

        List<AiCard> cards = new ArrayList<>(baseResponse.cards() == null ? List.of() : baseResponse.cards());
        cards.add(0, buildHybridCard(intent, analysisLayers, evidenceChannels));

        List<String> actions = new ArrayList<>(baseResponse.actions() == null ? List.of() : baseResponse.actions());
        addAction(actions, "查看分层结论");
        addAction(actions, "区分数据观察与证据支持");

        String summary = baseResponse.summary();
        if (summary == null || summary.isBlank()) {
            summary = "已按混合式单总控模式组织数据观察、模型判断、内部知识和外部证据。";
        } else if (!summary.contains("混合式") && !summary.contains("分层")) {
            summary = summary + " 当前回答已按混合式单总控模式组织为分层结论。";
        }

        return new AiChatResponse(
                baseResponse.success(),
                baseResponse.intent(),
                summary,
                cards,
                actions,
                metadata
        );
    }

    private List<Map<String, Object>> buildPatientLayers(Map<String, Object> metadata) {
        List<Map<String, Object>> layers = new ArrayList<>();
        layers.add(layer("data_observation", "数据观察", joinParts(
                value(metadata, "sampleId"),
                value(metadata, "currentDominanceLabel"),
                listValue(metadata, "currentTopTaxa")
        )));
        layers.add(layer("healthy_reference", "健康对照", joinParts(
                value(metadata, "healthyReferenceScope"),
                value(metadata, "healthyDeviationLabel"),
                listValue(metadata, "elevatedTaxa"),
                listValue(metadata, "depletedTaxa")
        )));
        layers.add(layer("model_judgement", "模型判断", joinParts(
                value(metadata, "predictionTop1Label"),
                value(metadata, "predictionTop2Label"),
                value(metadata, "predictionConsistency")
        )));
        layers.add(layer("internal_knowledge", "内部知识", joinKnowledgeSummary(metadata)));
        layers.add(layer("external_evidence", "外部证据", joinWebEvidenceSummary(metadata)));
        return layers;
    }

    private List<Map<String, Object>> buildPredictionLayers(Map<String, Object> metadata) {
        List<Map<String, Object>> layers = new ArrayList<>();
        layers.add(layer("prediction_result", "预测结果", joinParts(
                value(metadata, "predictionTop1Label"),
                value(metadata, "predictionTop1Probability"),
                value(metadata, "predictionTop2Label"),
                value(metadata, "predictionTop2Probability")
        )));
        layers.add(layer("model_boundary", "模型边界", joinKnowledgeSummary(metadata)));
        layers.add(layer("external_evidence", "外部证据", joinWebEvidenceSummary(metadata)));
        return layers;
    }

    private List<Map<String, Object>> buildDashboardLayers(Map<String, Object> metadata) {
        List<Map<String, Object>> layers = new ArrayList<>();
        layers.add(layer("dashboard_observation", "首页观察", joinParts(
                value(metadata, "totalPatients"),
                value(metadata, "topDisease"),
                value(metadata, "topRegion"),
                listValue(metadata, "topDiseases")
        )));
        layers.add(layer("internal_knowledge", "内部知识", joinKnowledgeSummary(metadata)));
        layers.add(layer("external_evidence", "外部证据", joinWebEvidenceSummary(metadata)));
        return layers;
    }

    private AiCard buildHybridCard(AiIntent intent, List<Map<String, Object>> layers, List<String> evidenceChannels) {
        String title = switch (intent) {
            case DASHBOARD_QA -> "混合式首页分析";
            case PREDICTION_ASSISTANT -> "混合式预测分析";
            case PATIENT_ANALYSIS, UNKNOWN -> "混合式患者分析";
        };
        String content = "当前总控已联合：" + String.join("、", evidenceChannels.isEmpty() ? List.of("业务数据") : evidenceChannels)
                + "。分层数量：" + layers.size() + "。";
        return new AiCard("hybrid_orchestration", title, content);
    }

    private List<String> collectEvidenceChannels(Map<String, Object> metadata) {
        LinkedHashSet<String> channels = new LinkedHashSet<>();
        channels.add("业务数据");
        if (metadata.containsKey("predictionTop1Label")) {
            channels.add("模型判断");
        }
        if (metadata.containsKey("knowledgeHits")) {
            channels.add("内部知识库");
        }
        if (metadata.containsKey("webEvidenceHits")) {
            channels.add("外部证据");
        }
        return new ArrayList<>(channels);
    }

    private Map<String, Object> layer(String key, String title, String summary) {
        Map<String, Object> layer = new LinkedHashMap<>();
        layer.put("key", key);
        layer.put("title", title);
        layer.put("summary", summary == null || summary.isBlank() ? "暂无" : summary);
        return layer;
    }

    private String joinKnowledgeSummary(Map<String, Object> metadata) {
        Object value = metadata.get("knowledgeHits");
        if (!(value instanceof List<?> list) || list.isEmpty()) {
            return "暂无内部知识命中";
        }
        List<String> parts = new ArrayList<>();
        for (Object item : list) {
            if (!(item instanceof Map<?, ?> map)) {
                continue;
            }
            Object title = map.get("title");
            if (title != null) {
                parts.add(String.valueOf(title));
            }
            if (parts.size() >= 2) {
                break;
            }
        }
        return parts.isEmpty() ? "暂无内部知识命中" : String.join("、", parts);
    }

    private String joinWebEvidenceSummary(Map<String, Object> metadata) {
        Object value = metadata.get("webEvidenceHits");
        if (!(value instanceof List<?> list) || list.isEmpty()) {
            return "暂无外部证据命中";
        }
        List<String> parts = new ArrayList<>();
        for (Object item : list) {
            if (!(item instanceof Map<?, ?> map)) {
                continue;
            }
            Object title = map.get("title");
            Object domain = map.get("domain");
            if (title != null && domain != null) {
                parts.add(String.valueOf(title) + " @ " + String.valueOf(domain));
            } else if (title != null) {
                parts.add(String.valueOf(title));
            }
            if (parts.size() >= 2) {
                break;
            }
        }
        return parts.isEmpty() ? "暂无外部证据命中" : String.join("、", parts);
    }

    @SuppressWarnings("unchecked")
    private String listValue(Map<String, Object> metadata, String key) {
        Object value = metadata.get(key);
        if (!(value instanceof List<?> list) || list.isEmpty()) {
            return "";
        }
        List<String> parts = new ArrayList<>();
        for (Object item : list) {
            if (item != null) {
                String text = String.valueOf(item).trim();
                if (!text.isBlank()) {
                    parts.add(text);
                }
            }
        }
        return String.join("、", parts);
    }

    private String value(Map<String, Object> metadata, String key) {
        Object value = metadata.get(key);
        return value == null ? "" : String.valueOf(value).trim();
    }

    private String joinParts(String... parts) {
        List<String> values = new ArrayList<>();
        for (String part : parts) {
            if (part != null && !part.isBlank()) {
                values.add(part);
            }
        }
        return String.join("；", values);
    }

    private void addAction(List<String> actions, String action) {
        if (action == null || action.isBlank() || actions.contains(action)) {
            return;
        }
        actions.add(action);
    }
}