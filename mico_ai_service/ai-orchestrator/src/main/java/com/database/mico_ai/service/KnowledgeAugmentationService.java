package com.database.mico_ai.service;

import com.database.mico_ai.dto.AiCard;
import com.database.mico_ai.dto.AiChatResponse;
import com.database.mico_ai.dto.KnowledgeHit;
import com.database.mico_ai.dto.MedicalEvidenceHit;
import com.database.mico_ai.dto.WebEvidenceHit;
import com.database.mico_ai.tool.KnowledgeTool;
import com.database.mico_ai.tool.MedicalEvidenceTool;
import com.database.mico_ai.tool.WebEvidenceTool;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.stream.Collectors;

@Service
public class KnowledgeAugmentationService {

    private final KnowledgeTool knowledgeTool;
    private final WebEvidenceTool webEvidenceTool;
    private final MedicalEvidenceTool medicalEvidenceTool;

    public KnowledgeAugmentationService(KnowledgeTool knowledgeTool,
                                        WebEvidenceTool webEvidenceTool,
                                        MedicalEvidenceTool medicalEvidenceTool) {
        this.knowledgeTool = knowledgeTool;
        this.webEvidenceTool = webEvidenceTool;
        this.medicalEvidenceTool = medicalEvidenceTool;
    }

    public AiChatResponse augmentPatientResponse(String question, AiChatResponse baseResponse) {
        Map<String, Object> metadata = baseMetadata(baseResponse);
        String query = buildPatientKnowledgeQuery(question, metadata);
        return augment(baseResponse, metadata, query, List.of("business", "model", "medical"), "recommendation", true);
    }

    public AiChatResponse augmentDashboardResponse(String question, AiChatResponse baseResponse) {
        Map<String, Object> metadata = baseMetadata(baseResponse);
        String query = buildDashboardKnowledgeQuery(question, metadata);
        return augment(baseResponse, metadata, query, List.of("business", "medical", "model"), null, false);
    }

    public AiChatResponse augmentPredictionResponse(String question, AiChatResponse baseResponse) {
        Map<String, Object> metadata = baseMetadata(baseResponse);
        String query = buildPredictionKnowledgeQuery(question, metadata);
        return augment(baseResponse, metadata, query, List.of("model", "business", "medical"), null, true);
    }

    private AiChatResponse augment(AiChatResponse baseResponse,
                                   Map<String, Object> metadata,
                                   String query,
                                   List<String> libraries,
                                   String preferredInsertBeforeType,
                                   boolean includeMedicalEvidence) {
        metadata.put("knowledgeQuery", query);

        List<KnowledgeHit> hits = knowledgeTool.searchKnowledge(query, libraries, 3);
        List<WebEvidenceHit> webHits = webEvidenceTool.searchWebEvidence(query, 2);
        List<MedicalEvidenceHit> medicalHits = includeMedicalEvidence
                ? medicalEvidenceTool.searchMedicalEvidence(query, 3)
                : List.of();

        if (hits.isEmpty() && webHits.isEmpty() && medicalHits.isEmpty()) {
            return new AiChatResponse(
                    baseResponse.success(),
                    baseResponse.intent(),
                    appendEvidenceSummary(baseResponse.summary(), false, false, false),
                    baseResponse.cards(),
                    baseResponse.actions(),
                    metadata
            );
        }

        if (!hits.isEmpty()) {
            metadata.put("knowledgeHitCount", hits.size());
            metadata.put("knowledgeHits", hits.stream().map(this::toMap).collect(Collectors.toList()));
        }
        if (!webHits.isEmpty()) {
            metadata.put("webEvidenceHitCount", webHits.size());
            metadata.put("webEvidenceHits", webHits.stream().map(this::toMap).collect(Collectors.toList()));
        }
        if (!medicalHits.isEmpty()) {
            metadata.put("medicalEvidenceHitCount", medicalHits.size());
            metadata.put("medicalEvidenceHits", medicalHits.stream().map(this::toMap).collect(Collectors.toList()));
        }

        List<AiCard> cards = new ArrayList<>(baseResponse.cards() == null ? List.of() : baseResponse.cards());
        int insertIndex = preferredInsertBeforeType == null ? -1 : indexOfCard(cards, preferredInsertBeforeType);

        if (!hits.isEmpty()) {
            AiCard knowledgeCard = new AiCard("knowledge_support", "知识支持", buildKnowledgeCardContent(hits));
            if (insertIndex >= 0) {
                cards.add(insertIndex, knowledgeCard);
                insertIndex++;
            } else {
                cards.add(knowledgeCard);
            }
        }
        if (!medicalHits.isEmpty()) {
            AiCard medicalCard = new AiCard("medical_evidence", "医学证据", buildMedicalEvidenceCardContent(medicalHits));
            if (insertIndex >= 0) {
                cards.add(insertIndex, medicalCard);
                insertIndex++;
            } else {
                cards.add(medicalCard);
            }
        }
        if (!webHits.isEmpty()) {
            AiCard webCard = new AiCard("web_evidence", "外部证据", buildWebEvidenceCardContent(webHits));
            if (insertIndex >= 0) {
                cards.add(insertIndex, webCard);
            } else {
                cards.add(webCard);
            }
        }

        List<String> actions = new ArrayList<>(baseResponse.actions() == null ? List.of() : baseResponse.actions());
        if (!hits.isEmpty()) {
            addAction(actions, "查看知识依据");
            addAction(actions, "解释业务规则与模型边界");
        }
        if (!medicalHits.isEmpty()) {
            addAction(actions, "查看医学证据");
            addAction(actions, "区分医学证据与模型结论");
        }
        if (!webHits.isEmpty()) {
            addAction(actions, "查看外部证据");
            addAction(actions, "区分内部结论与外部资料");
        }

        return new AiChatResponse(
                baseResponse.success(),
                baseResponse.intent(),
                appendEvidenceSummary(baseResponse.summary(), !hits.isEmpty(), !webHits.isEmpty(), !medicalHits.isEmpty()),
                cards,
                actions,
                metadata
        );
    }

    private Map<String, Object> baseMetadata(AiChatResponse baseResponse) {
        Map<String, Object> metadata = new LinkedHashMap<>();
        if (baseResponse.metadata() != null) {
            metadata.putAll(baseResponse.metadata());
        }
        metadata.putAll(knowledgeTool.knowledgeStatus());
        metadata.putAll(webEvidenceTool.webSearchStatus());
        metadata.putAll(medicalEvidenceTool.medicalEvidenceStatus());
        return metadata;
    }

    private String buildPatientKnowledgeQuery(String question, Map<String, Object> metadata) {
        LinkedHashSet<String> parts = new LinkedHashSet<>();
        addPart(parts, question);
        addPart(parts, safeText(metadata.get("healthyDeviationLabel")));
        addPart(parts, safeText(metadata.get("predictionTop1Label")));
        addPart(parts, safeText(metadata.get("predictionTop2Label")));
        addPart(parts, safeText(metadata.get("predictionConsistency")));
        addAll(parts, metadata.get("elevatedTaxa"));
        addAll(parts, metadata.get("depletedTaxa"));
        addPart(parts, "健康组对照");
        addPart(parts, "预测边界");
        addPart(parts, "模型解释");
        addPart(parts, "微生物组 疾病 证据");
        addPart(parts, "页面分析模板");
        return String.join(" ", parts);
    }

    private String buildDashboardKnowledgeQuery(String question, Map<String, Object> metadata) {
        LinkedHashSet<String> parts = new LinkedHashSet<>();
        addPart(parts, question);
        addPart(parts, safeText(metadata.get("topDisease")));
        addPart(parts, safeText(metadata.get("topRegion")));
        addPart(parts, safeText(metadata.get("totalPatients")));
        addAll(parts, metadata.get("topDiseases"));
        addAll(parts, metadata.get("topRegions"));
        addPart(parts, "首页分析模板");
        addPart(parts, "疾病分布");
        addPart(parts, "地区来源");
        addPart(parts, "群体结构");
        return String.join(" ", parts);
    }

    private String buildPredictionKnowledgeQuery(String question, Map<String, Object> metadata) {
        LinkedHashSet<String> parts = new LinkedHashSet<>();
        addPart(parts, question);
        addPart(parts, safeText(metadata.get("predictionTop1Label")));
        addPart(parts, safeText(metadata.get("predictionTop2Label")));
        addPart(parts, safeText(metadata.get("predictionTop1Probability")));
        addPart(parts, safeText(metadata.get("predictionTop2Probability")));
        addPart(parts, "预测解释模板");
        addPart(parts, "预测边界");
        addPart(parts, "模型标签定义");
        addPart(parts, "微生物组 疾病 研究证据");
        addPart(parts, "特征空间");
        return String.join(" ", parts);
    }

    private String appendEvidenceSummary(String summary,
                                         boolean hasKnowledgeHits,
                                         boolean hasWebHits,
                                         boolean hasMedicalHits) {
        String base = summary == null ? "" : summary;
        if (base.isBlank()) {
            if (hasKnowledgeHits && hasMedicalHits && hasWebHits) {
                return "已补充内部知识、医学证据和外部网页证据，可继续追问业务规则、模型边界和医学来源。";
            }
            if (hasKnowledgeHits && hasMedicalHits) {
                return "已补充内部知识与医学证据，可继续追问健康对照、模型边界和文献来源。";
            }
            if (hasKnowledgeHits) {
                return "已补充内部知识库证据，可继续追问业务规则、医学解释和模型边界。";
            }
            if (hasMedicalHits) {
                return "已补充外部医学证据，可继续追问来源类型、研究支持和证据边界。";
            }
            if (hasWebHits) {
                return "已补充外部网页证据，可继续追问来源和证据边界。";
            }
            return "当前未命中知识库或外部证据，可继续追问更具体的问题。";
        }

        String enriched = base;
        if (hasKnowledgeHits && !base.contains("知识库") && !base.contains("知识依据")) {
            enriched = enriched + " 已补充内部知识库证据。";
        }
        if (hasMedicalHits && !base.contains("医学证据") && !base.contains("文献")) {
            enriched = enriched + " 已补充权威医学证据作为解释补充层。";
        }
        if (hasWebHits && !base.contains("外部证据") && !base.contains("网页证据")) {
            enriched = enriched + " 已补充带来源链接的外部网页证据。";
        }
        return enriched;
    }

    private String buildKnowledgeCardContent(List<KnowledgeHit> hits) {
        return hits.stream()
                .map(hit -> String.format(Locale.ROOT, "%s：%s。%s", hit.libraryLabel(), hit.title(), hit.snippet()))
                .collect(Collectors.joining(" "));
    }

    private String buildMedicalEvidenceCardContent(List<MedicalEvidenceHit> hits) {
        return hits.stream()
                .map(hit -> String.format(Locale.ROOT,
                        "%s：%s。来源：%s（%s）。证据类型：%s。链接：%s",
                        safeText(hit.title()),
                        safeText(hit.snippet()),
                        safeText(hit.source()),
                        safeText(hit.domain()),
                        safeText(hit.evidenceType()),
                        safeText(hit.url())
                ))
                .collect(Collectors.joining(" "));
    }

    private String buildWebEvidenceCardContent(List<WebEvidenceHit> hits) {
        return hits.stream()
                .map(hit -> {
                    String domain = hit.domain() == null || hit.domain().isBlank() ? "外部来源" : hit.domain();
                    return String.format(Locale.ROOT, "%s：%s。来源：%s。链接：%s", hit.title(), hit.snippet(), domain, hit.url());
                })
                .collect(Collectors.joining(" "));
    }

    private Map<String, Object> toMap(KnowledgeHit hit) {
        Map<String, Object> item = new LinkedHashMap<>();
        item.put("library", hit.library());
        item.put("libraryLabel", hit.libraryLabel());
        item.put("title", hit.title());
        item.put("sourcePath", hit.sourcePath());
        item.put("snippet", hit.snippet());
        item.put("score", hit.score());
        return item;
    }

    private Map<String, Object> toMap(WebEvidenceHit hit) {
        Map<String, Object> item = new LinkedHashMap<>();
        item.put("title", hit.title());
        item.put("url", hit.url());
        item.put("domain", hit.domain());
        item.put("snippet", hit.snippet());
        item.put("publishedDate", hit.publishedDate());
        item.put("score", hit.score());
        return item;
    }

    private Map<String, Object> toMap(MedicalEvidenceHit hit) {
        Map<String, Object> item = new LinkedHashMap<>();
        item.put("title", hit.title());
        item.put("url", hit.url());
        item.put("domain", hit.domain());
        item.put("source", hit.source());
        item.put("snippet", hit.snippet());
        item.put("publishedDate", hit.publishedDate());
        item.put("evidenceType", hit.evidenceType());
        item.put("score", hit.score());
        item.put("pmcid", hit.pmcid());
        item.put("pmid", hit.pmid());
        item.put("section", hit.section());
        item.put("topic", hit.topic());
        item.put("citation", hit.citation());
        return item;
    }

    private void addPart(Collection<String> parts, String value) {
        if (value != null && !value.isBlank()) {
            parts.add(value.trim());
        }
    }

    private void addAll(Collection<String> parts, Object value) {
        if (!(value instanceof Collection<?> collection)) {
            return;
        }
        for (Object item : collection) {
            String text = safeText(item);
            if (!text.isBlank()) {
                parts.add(text);
            }
        }
    }

    private void addAction(List<String> actions, String action) {
        if (action == null || action.isBlank() || actions.contains(action)) {
            return;
        }
        actions.add(action);
    }

    private int indexOfCard(List<AiCard> cards, String type) {
        for (int i = 0; i < cards.size(); i++) {
            if (type.equals(cards.get(i).type())) {
                return i;
            }
        }
        return -1;
    }

    private String safeText(Object value) {
        return value == null ? "" : String.valueOf(value).trim();
    }
}
