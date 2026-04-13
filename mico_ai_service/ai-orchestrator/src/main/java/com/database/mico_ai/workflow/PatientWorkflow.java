package com.database.mico_ai.workflow;

import com.database.mico_ai.dto.AiCard;
import com.database.mico_ai.dto.AiChatResponse;
import com.database.mico_ai.dto.AiContext;
import com.database.mico_ai.dto.AiIntent;
import com.database.mico_ai.tool.PatientTool;
import com.database.mico_ai.tool.PredictionTool;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

@Service
public class PatientWorkflow {

    private static final Pattern PATIENT_ID_PATTERN = Pattern.compile("(?<!\\d)\\d{1,6}(?!\\d)");

    private final PatientTool patientTool;
    private final PredictionTool predictionTool;

    public PatientWorkflow(PatientTool patientTool, PredictionTool predictionTool) {
        this.patientTool = patientTool;
        this.predictionTool = predictionTool;
    }

    public AiChatResponse run(String question, AiContext context) {
        String patientId = resolvePatientId(question, context);
        if (isBlank(patientId)) {
            return new AiChatResponse(false, AiIntent.PATIENT_ANALYSIS.name(), "请先在患者详情页中打开具体患者，或提供明确的 patientId。", List.of(), List.of("打开患者详情页", "提供 patientId", "重新提问当前患者问题"), Map.of());
        }

        Map<String, Object> patientResponse = patientTool.getPatient(patientId);
        if (!isSuccess(patientResponse)) {
            return upstreamFailure(AiIntent.PATIENT_ANALYSIS, "患者详情接口当前不可用。", patientResponse);
        }

        Map<String, Object> patient = asMap(patientResponse.get("data"));
        if (patient.isEmpty()) {
            return new AiChatResponse(false, AiIntent.PATIENT_ANALYSIS.name(), "未找到对应患者信息。", List.of(new AiCard("empty", "患者不存在", "请确认 patientId 是否正确。")), List.of("重新搜索患者", "检查 patientId"), Map.of("patientId", patientId));
        }

        Map<String, Object> samplesResponse = patientTool.listSamples(patientId);
        if (!isSuccess(samplesResponse)) {
            return upstreamFailure(AiIntent.PATIENT_ANALYSIS, "患者样本接口当前不可用。", samplesResponse);
        }

        List<Map<String, Object>> samples = asListOfMaps(asMap(samplesResponse.get("data")).get("items"));
        if (samples.isEmpty()) {
            return new AiChatResponse(false, AiIntent.PATIENT_ANALYSIS.name(), "该患者当前没有可用于分析的标准样本。", List.of(new AiCard("empty", "无标准样本", "请先确认标准丰度表是否已导入该患者样本。")), List.of("检查标准丰度表", "切换其他患者"), Map.of("patientId", patientId));
        }

        Map<String, Object> currentSample = resolveCurrentSample(samples, context == null ? null : context.sampleId());
        String currentSampleId = safeText(currentSample.get("sampleId"));
        List<Map<String, Object>> currentFeatures = loadTopFeatures(patientId, currentSampleId, 8);
        SampleStructure currentStructure = analyzeStructure(currentSample, currentFeatures);

        HealthInsight healthInsight = null;
        Map<String, Object> healthyReferenceData = Map.of();
        Map<String, Object> healthyReferenceResponse = patientTool.getHealthyReference(patientId, currentSampleId, 8);
        if (isSuccess(healthyReferenceResponse)) {
            healthyReferenceData = asMap(healthyReferenceResponse.get("data"));
            healthInsight = analyzeHealthyReferenceCombined(healthyReferenceData);
        }

        PredictionInsight predictionInsight = runPredictionInsight(patient, patientId, currentSampleId);

        ComparisonInsight comparisonInsight = null;
        Map<String, Object> comparisonSample = resolveComparisonSample(samples, currentSampleId);
        if (comparisonSample != null) {
            String comparisonSampleId = safeText(comparisonSample.get("sampleId"));
            List<Map<String, Object>> comparisonFeatures = loadTopFeatures(patientId, comparisonSampleId, 8);
            SampleStructure comparisonStructure = analyzeStructure(comparisonSample, comparisonFeatures);
            comparisonInsight = compareSamples(samples, currentStructure, comparisonStructure);
        }

        List<AiCard> cards = new ArrayList<>();
        cards.add(buildPatientCard(patient, samples.size()));
        cards.add(buildCurrentSampleCard(currentStructure, samples));
        if (healthInsight != null) {
            cards.add(buildHealthyCardCombined(healthInsight));
        }
        if (predictionInsight != null) {
            cards.add(buildPredictionLinkCard(predictionInsight));
        }
        if (comparisonInsight != null) {
            cards.add(buildComparisonCard(comparisonInsight));
        }
        cards.add(buildStructureCard(currentStructure));
        cards.add(buildRecommendationCard(currentStructure, healthInsight, predictionInsight, comparisonInsight));

        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("patientId", patientId);
        metadata.put("sampleId", currentStructure.sampleId());
        metadata.put("sampleCount", samples.size());
        metadata.put("currentFeatureCount", currentStructure.featureCount());
        metadata.put("currentAbundanceTotal", currentStructure.abundanceTotal());
        metadata.put("currentTop1Ratio", currentStructure.top1Ratio());
        metadata.put("currentTop3Ratio", currentStructure.top3Ratio());
        metadata.put("currentDominanceLabel", currentStructure.dominanceLabel());
        metadata.put("currentRichnessRank", rankByFeatureCount(samples, currentStructure.sampleId()));
        metadata.put("currentTopTaxa", currentStructure.displayNames());
        if (healthInsight != null) {
            metadata.put("healthyReferenceScope", healthInsight.referenceScope());
            metadata.put("healthyPatientCount", healthInsight.healthyPatientCount());
            metadata.put("healthySampleCount", healthInsight.healthySampleCount());
            metadata.put("healthyDeviationLabel", healthInsight.deviationLabel());
            metadata.put("elevatedTaxa", healthInsight.elevatedTaxa());
            metadata.put("depletedTaxa", healthInsight.depletedTaxa());
            metadata.put("deviationTopTaxa", healthInsight.deviationTopTaxa());
            metadata.put("spatialDeviationLevel", healthInsight.spatialDeviationLevel());
            metadata.put("spatialDistanceZScore", healthInsight.spatialDistanceZScore());
            metadata.put("spatialDistancePercentile", healthInsight.spatialDistancePercentile());
            metadata.put("healthySignatureTaxa", healthInsight.healthySignatureTaxa());
            metadata.put("healthyReferenceRecommendation", healthInsight.recommendation());
        }
        if (predictionInsight != null) {
            metadata.put("predictionTop1Label", predictionInsight.top1Label());
            metadata.put("predictionTop1Probability", predictionInsight.top1Probability());
            metadata.put("predictionTop2Label", predictionInsight.top2Label());
            metadata.put("predictionTop2Probability", predictionInsight.top2Probability());
            metadata.put("predictionConsistency", predictionInsight.consistencyLabel());
            metadata.put("predictionRecommendation", predictionInsight.recommendation());
        }
        if (comparisonInsight != null) {
            metadata.put("comparisonSampleId", comparisonInsight.comparisonSampleId());
            metadata.put("comparisonFeatureCount", comparisonInsight.comparisonFeatureCount());
            metadata.put("comparisonAbundanceTotal", comparisonInsight.comparisonAbundanceTotal());
            metadata.put("comparisonTop1Ratio", comparisonInsight.comparisonTop1Ratio());
            metadata.put("comparisonTop3Ratio", comparisonInsight.comparisonTop3Ratio());
            metadata.put("comparisonDominanceLabel", comparisonInsight.comparisonDominanceLabel());
            metadata.put("topFeatureOverlap", comparisonInsight.overlapCount());
            metadata.put("currentUniqueTaxa", comparisonInsight.currentUniqueTaxa());
            metadata.put("comparisonUniqueTaxa", comparisonInsight.comparisonUniqueTaxa());
            metadata.put("comparisonRecommendation", comparisonInsight.recommendation());
        }

        String summary;
        if (healthInsight != null && predictionInsight != null) {
            summary = "已把当前样本相对健康组的偏离，与模型预测方向一起联动解读，可继续追问哪些偏离最支撑当前预测结论。";
        } else if (healthInsight != null) {
            summary = "已完成当前样本相对健康组的偏离分析，并补充了样本自身结构与同患者样本差异，可继续追问哪些菌最偏离健康基线。";
        } else {
            summary = "已完成当前患者样本的结构分析，可继续追问优势菌结构、样本是否适合进入预测，或切换样本查看差异。";
        }

        List<String> actions;
        if (predictionInsight != null) {
            actions = List.of("解释预测结果与健康组偏离的关系", "查看 Top2 疾病方向", "比较其他样本");
        } else if (healthInsight != null) {
            actions = List.of("查看与健康组的主要差异", "判断是否值得进入预测", "比较其他样本");
        } else {
            actions = List.of("查看当前样本高丰度菌", "判断是否进入预测", "切换其他样本");
        }

        return new AiChatResponse(true, AiIntent.PATIENT_ANALYSIS.name(), summary, cards, actions, metadata);
    }

    private PredictionInsight runPredictionInsight(Map<String, Object> patient, String patientId, String sampleId) {
        Map<String, Object> payloadResponse = patientTool.getPredictionPayload(patientId, sampleId);
        if (!isSuccess(payloadResponse)) {
            return null;
        }

        Map<String, Object> payload = asMap(payloadResponse.get("data"));
        if (payload.isEmpty()) {
            return null;
        }

        Map<String, Object> prediction = predictionTool.runPrediction(payload);
        if (!Boolean.TRUE.equals(prediction.get("success")) && !prediction.containsKey("predictions")) {
            return null;
        }

        List<Map<String, Object>> predictions = asListOfMaps(prediction.get("predictions"));
        if (predictions.isEmpty()) {
            return null;
        }

        Map<String, Object> top1 = predictions.get(0);
        Map<String, Object> top2 = predictions.size() > 1 ? predictions.get(1) : Collections.emptyMap();

        String patientDisease = normalizeDiseaseLabel(preferNonBlank(safeText(patient.get("disease")), safeText(patient.get("group"))));
        String top1Label = normalizeDiseaseLabel(preferNonBlank(safeText(top1.get("label")), safeText(top1.get("group"))));
        String top2Label = normalizeDiseaseLabel(preferNonBlank(safeText(top2.get("label")), safeText(top2.get("group"))));
        double top1Probability = safeDouble(top1.get("probability"));
        double top2Probability = safeDouble(top2.get("probability"));

        String consistencyLabel;
        String interpretation;
        if (!isBlank(patientDisease) && patientDisease.equalsIgnoreCase(top1Label)) {
            consistencyLabel = "与当前疾病标签一致";
            interpretation = "当前预测 Top1 与患者疾病标签一致，模型方向和当前业务标签匹配。";
        } else if (!"健康".equals(patientDisease) && "健康".equals(top1Label)) {
            consistencyLabel = "与当前疾病标签不一致";
            interpretation = "当前样本虽然已进入预测，但模型更偏向健康方向，这说明结果需要结合健康组偏离一起谨慎判断。";
        } else if (!isBlank(patientDisease) && !"健康".equals(patientDisease) && !"健康".equals(top1Label)) {
            consistencyLabel = "均指向非健康方向";
            interpretation = "模型与患者标签都指向非健康状态，但具体疾病方向并不完全一致，适合结合偏离菌进一步解释。";
        } else {
            consistencyLabel = "方向需要结合上下文判断";
            interpretation = "模型结果可以作为辅助参考，但还需要结合健康组偏离和样本结构一起看。";
        }

        String recommendation = top1Probability >= 70D
                ? "当前 Top1 概率较高，可以优先围绕 Top1 疾病方向解释与健康组的偏离。"
                : "当前 Top1 与 Top2 仍有竞争，建议同时查看健康组偏离和 Top2 方向，不要过早下结论。";

        return new PredictionInsight(top1Label, top1Probability, top2Label, top2Probability, consistencyLabel, interpretation, recommendation);
    }    private AiCard buildPatientCard(Map<String, Object> patient, int sampleCount) {
        String disease = preferNonBlank(safeText(patient.get("disease")), safeText(patient.get("group")), "未标注");
        String age = blankToDash(patient.get("age"));
        String gender = normalizeGender(safeText(patient.get("gender")));
        String country = blankToDash(patient.get("country"));
        String patientName = preferNonBlank(safeText(patient.get("patientName")), safeText(patient.get("patientId")), "未命名患者");
        String content = String.format(Locale.ROOT, "%s，疾病标签 %s，年龄 %s，性别 %s，地区 %s，共有 %d 份标准样本。", patientName, disease, age, gender, country, sampleCount);
        return new AiCard("patient", "患者概况", content);
    }

    private AiCard buildCurrentSampleCard(SampleStructure structure, List<Map<String, Object>> samples) {
        String content = String.format(Locale.ROOT, "当前样本 %s，非零特征 %d，丰度总量 %s，在该患者 %d 份样本中按特征丰富度排第 %d，头部结构%s。", structure.sampleId(), structure.featureCount(), formatDecimal(structure.abundanceTotal()), samples.size(), rankByFeatureCount(samples, structure.sampleId()), structure.dominanceLabel());
        return new AiCard("sample", "当前样本", content);
    }

    private AiCard buildHealthyCard(HealthInsight insight) {
        String elevated = insight.elevatedTaxa().isEmpty() ? "未见明显升高特征" : String.join("、", insight.elevatedTaxa());
        String depleted = insight.depletedTaxa().isEmpty() ? "未见明显降低特征" : String.join("、", insight.depletedTaxa());
        String signature = insight.healthySignatureTaxa().isEmpty() ? "健康组头部特征与当前样本重叠较多" : String.join("、", insight.healthySignatureTaxa());
        String content = String.format(Locale.ROOT, "参考范围：%s，健康患者 %d 人、健康样本 %d 份。当前样本%s；相对健康组偏高：%s；偏低：%s；健康组代表特征：%s。", insight.referenceScopeLabel(), insight.healthyPatientCount(), insight.healthySampleCount(), insight.deviationLabel(), elevated, depleted, signature);
        return new AiCard("healthy_reference", "健康组对照", content);
    }

    private AiCard buildHealthyCardCombined(HealthInsight insight) {
        String elevated = insight.elevatedTaxa().isEmpty() ? "未见明显升高特征" : String.join("、", insight.elevatedTaxa());
        String depleted = insight.depletedTaxa().isEmpty() ? "未见明显降低特征" : String.join("、", insight.depletedTaxa());
        String deviationTop = insight.deviationTopTaxa().isEmpty() ? "暂无" : String.join("、", insight.deviationTopTaxa());
        String signature = insight.healthySignatureTaxa().isEmpty() ? "健康组头部特征与当前样本重叠较多" : String.join("、", insight.healthySignatureTaxa());
        String spatial = String.format(
                Locale.ROOT,
                "空间偏离=%s (Z=%.2f, 百分位=%.1f%%)",
                insight.spatialDeviationLevel(),
                insight.spatialDistanceZScore(),
                insight.spatialDistancePercentile()
        );
        String content = String.format(
                Locale.ROOT,
                "参考范围：%s，健康患者 %d 人、健康样本 %d 份。偏离结论：%s。联合评分Top差异菌：%s；偏高：%s；偏低：%s；%s；健康组代表特征：%s。",
                insight.referenceScopeLabel(),
                insight.healthyPatientCount(),
                insight.healthySampleCount(),
                insight.deviationLabel(),
                deviationTop,
                elevated,
                depleted,
                spatial,
                signature
        );
        return new AiCard("healthy_reference", "健康组对照", content);
    }

    private AiCard buildPredictionLinkCard(PredictionInsight insight) {
        String top2 = isBlank(insight.top2Label()) ? "暂无" : insight.top2Label() + "（" + formatProbability(insight.top2Probability()) + "）";
        String content = String.format(Locale.ROOT, "模型 Top1：%s（%s），Top2：%s。当前判断：%s。%s", insight.top1Label(), formatProbability(insight.top1Probability()), top2, insight.consistencyLabel(), insight.interpretation());
        return new AiCard("prediction_link", "偏离与预测联动", content);
    }

    private AiCard buildComparisonCard(ComparisonInsight insight) {
        String featureDirection = insight.featureDelta() > 0
                ? String.format(Locale.ROOT, "当前样本比对照多 %d 个非零特征", insight.featureDelta())
                : insight.featureDelta() < 0
                ? String.format(Locale.ROOT, "当前样本比对照少 %d 个非零特征", Math.abs(insight.featureDelta()))
                : "当前样本与对照样本的非零特征数接近";

        String content = String.format(Locale.ROOT, "对照样本 %s。%s，头部菌重叠 %d/%d；当前样本 Top3 占比 %s，对照样本 Top3 占比 %s。%s", insight.comparisonSampleId(), featureDirection, insight.overlapCount(), insight.windowSize(), formatPercent(insight.currentTop3Ratio()), formatPercent(insight.comparisonTop3Ratio()), insight.recommendation());
        return new AiCard("comparison", "同患者样本比较", content);
    }

    private AiCard buildStructureCard(SampleStructure structure) {
        String headline = structure.displayNames().isEmpty() ? "暂无可展示的高丰度特征。" : String.join("、", structure.displayNames());
        String content = String.format(Locale.ROOT, "头部特征：%s。Top1 占比 %s，Top3 占比 %s，属于%s。", headline, formatPercent(structure.top1Ratio()), formatPercent(structure.top3Ratio()), structure.dominanceLabel());
        return new AiCard("structure", "样本结构", content);
    }

    private AiCard buildRecommendationCard(SampleStructure currentStructure,
                                           HealthInsight healthInsight,
                                           PredictionInsight predictionInsight,
                                           ComparisonInsight comparisonInsight) {
        String content;
        if (predictionInsight != null) {
            content = predictionInsight.recommendation();
        } else if (healthInsight != null) {
            content = healthInsight.recommendation();
        } else if (comparisonInsight != null) {
            content = comparisonInsight.recommendation();
        } else {
            content = currentStructure.top1Ratio() >= 0.45
                    ? "当前样本头部菌较集中，适合先判断优势菌是否稳定，再决定是否进入预测。"
                    : "当前样本结构相对分散，信息更均衡，适合继续进入预测或与其他样本联动查看。";
        }
        return new AiCard("recommendation", "建议动作", content);
    }

    private HealthInsight analyzeHealthyReferenceCombined(Map<String, Object> data) {
        if (data.isEmpty()) {
            return null;
        }

        List<Map<String, Object>> featureComparisons = asListOfMaps(data.get("featureComparisons"));
        featureComparisons.sort(Comparator
                .comparingDouble((Map<String, Object> item) -> safeDouble(item.get("combinedScore"))).reversed()
                .thenComparing((Map<String, Object> item) -> Math.abs(safeDouble(item.get("log2FoldChange"))), Comparator.reverseOrder())
                .thenComparing((Map<String, Object> item) -> safeDouble(item.get("currentAbundance")), Comparator.reverseOrder()));

        List<String> deviationTopTaxa = featureComparisons.stream()
                .map(item -> shortenTaxonName(safeText(item.get("microbeName"))))
                .filter(name -> name != null && !name.isBlank())
                .distinct()
                .limit(5)
                .collect(Collectors.toList());

        List<String> elevated = featureComparisons.stream()
                .filter(item -> "higher".equalsIgnoreCase(safeText(item.get("direction"))))
                .map(item -> shortenTaxonName(safeText(item.get("microbeName"))))
                .distinct()
                .limit(3)
                .collect(Collectors.toList());

        List<String> depleted = featureComparisons.stream()
                .filter(item -> "lower".equalsIgnoreCase(safeText(item.get("direction"))))
                .map(item -> shortenTaxonName(safeText(item.get("microbeName"))))
                .distinct()
                .limit(3)
                .collect(Collectors.toList());

        List<String> healthySignature = asListOfMaps(data.get("healthySignatureFeatures")).stream()
                .map(item -> shortenTaxonName(safeText(item.get("microbe_name"))))
                .filter(name -> name != null && !name.isBlank())
                .distinct()
                .limit(3)
                .collect(Collectors.toList());

        Map<String, Object> spatial = asMap(data.get("spatialDeviation"));
        String spatialLevel = normalizeSpatialLevel(safeText(spatial.get("deviationLevel")));
        double spatialZ = safeDouble(spatial.get("distanceZScore"));
        double spatialPercentile = safeDouble(spatial.get("distancePercentile"));

        String deviationLabel;
        if ("高偏离".equals(spatialLevel) || (elevated.size() >= 2 && depleted.size() >= 1)) {
            deviationLabel = "对健康基线偏离较明显";
        } else if ("中等偏离".equals(spatialLevel) || !elevated.isEmpty() || !depleted.isEmpty()) {
            deviationLabel = "存在一定偏离";
        } else {
            deviationLabel = "整体接近健康基线";
        }

        String recommendation;
        if ("对健康基线偏离较明显".equals(deviationLabel)) {
            recommendation = "当前样本相对健康组偏离较明显，建议优先关注联合评分靠前的差异菌，并结合预测方向一起判断。";
        } else if ("存在一定偏离".equals(deviationLabel)) {
            recommendation = "当前样本与健康组已出现可解释偏离，建议继续核对差异菌与预测Top1/Top2的一致性。";
        } else {
            recommendation = "当前样本整体接近健康组，建议结合随访样本和临床指标再做综合判断。";
        }

        String referenceScope = safeText(data.get("referenceScope"));
        String referenceScopeLabel = "same_body_site_healthy".equals(referenceScope) ? "同 body site 健康组" : "全体健康组";

        return new HealthInsight(
                referenceScope,
                referenceScopeLabel,
                safeLong(data.get("healthyPatientCount")),
                safeLong(data.get("healthySampleCount")),
                deviationLabel,
                elevated,
                depleted,
                deviationTopTaxa,
                spatialLevel,
                spatialZ,
                spatialPercentile,
                healthySignature,
                recommendation
        );
    }

    private String normalizeSpatialLevel(String value) {
        if (value == null || value.isBlank()) {
            return "未知";
        }
        String lowered = value.trim().toLowerCase(Locale.ROOT);
        if ("high".equals(lowered)) {
            return "高偏离";
        }
        if ("moderate".equals(lowered)) {
            return "中等偏离";
        }
        if ("mild".equals(lowered)) {
            return "轻度偏离";
        }
        if ("close_to_healthy".equals(lowered)) {
            return "接近健康";
        }
        return value;
    }

    private HealthInsight analyzeHealthyReference(Map<String, Object> data) {
        if (data.isEmpty()) {
            return null;
        }

        List<Map<String, Object>> featureComparisons = asListOfMaps(data.get("featureComparisons"));
        List<String> elevated = featureComparisons.stream()
                .filter(item -> "higher".equalsIgnoreCase(safeText(item.get("direction"))))
                .sorted(Comparator.comparingDouble((Map<String, Object> item) -> ratioRank(item.get("ratioToHealthy"))).reversed())
                .map(item -> shortenTaxonName(safeText(item.get("microbeName"))))
                .distinct()
                .limit(3)
                .collect(Collectors.toList());

        List<String> depleted = featureComparisons.stream()
                .filter(item -> "lower".equalsIgnoreCase(safeText(item.get("direction"))))
                .sorted(Comparator.comparingDouble((Map<String, Object> item) -> ratioRank(item.get("ratioToHealthy"))))
                .map(item -> shortenTaxonName(safeText(item.get("microbeName"))))
                .distinct()
                .limit(3)
                .collect(Collectors.toList());

        List<String> healthySignature = asListOfMaps(data.get("healthySignatureFeatures")).stream()
                .map(item -> shortenTaxonName(safeText(item.get("microbe_name"))))
                .filter(name -> name != null && !name.isBlank())
                .distinct()
                .limit(3)
                .collect(Collectors.toList());

        String deviationLabel;
        if (elevated.size() >= 2 && depleted.size() >= 1) {
            deviationLabel = "对健康基线偏离较明显";
        } else if (!elevated.isEmpty() || !depleted.isEmpty()) {
            deviationLabel = "存在一定偏离";
        } else {
            deviationLabel = "整体接近健康基线";
        }

        String recommendation;
        if ("对健康基线偏离较明显".equals(deviationLabel)) {
            recommendation = "当前样本相对健康组偏离较明显，优先值得进入预测，并重点关注偏高菌与偏低菌的组合变化。";
        } else if ("存在一定偏离".equals(deviationLabel)) {
            recommendation = "当前样本与健康组已有可见差异，适合继续结合预测结果判断这些偏离是否与疾病标签一致。";
        } else {
            recommendation = "当前样本整体更接近健康基线，若业务上仍有疑问，建议再结合其他样本或临床指标一起判断。";
        }

        String referenceScope = safeText(data.get("referenceScope"));
        String referenceScopeLabel = "same_body_site_healthy".equals(referenceScope) ? "同 body site 健康组" : "全体健康组";

        return new HealthInsight(
                referenceScope,
                referenceScopeLabel,
                safeLong(data.get("healthyPatientCount")),
                safeLong(data.get("healthySampleCount")),
                deviationLabel,
                elevated,
                depleted,
                featureComparisons.stream()
                        .map(item -> shortenTaxonName(safeText(item.get("microbeName"))))
                        .distinct()
                        .limit(5)
                        .collect(Collectors.toList()),
                "未知",
                0D,
                50D,
                healthySignature,
                recommendation
        );
    }

    private ComparisonInsight compareSamples(List<Map<String, Object>> allSamples, SampleStructure current, SampleStructure comparison) {
        LinkedHashSet<String> currentKeys = new LinkedHashSet<>(current.featureKeys());
        LinkedHashSet<String> comparisonKeys = new LinkedHashSet<>(comparison.featureKeys());

        int overlapCount = 0;
        for (String key : currentKeys) {
            if (comparisonKeys.contains(key)) {
                overlapCount++;
            }
        }

        List<String> currentUniqueTaxa = current.featureEntries().stream()
                .filter(entry -> !comparisonKeys.contains(entry.key()))
                .map(FeatureEntry::displayName)
                .distinct()
                .limit(3)
                .collect(Collectors.toList());

        List<String> comparisonUniqueTaxa = comparison.featureEntries().stream()
                .filter(entry -> !currentKeys.contains(entry.key()))
                .map(FeatureEntry::displayName)
                .distinct()
                .limit(3)
                .collect(Collectors.toList());

        long featureDelta = current.featureCount() - comparison.featureCount();
        String recommendation = buildComparisonRecommendation(allSamples, current, comparison, featureDelta, overlapCount);

        return new ComparisonInsight(comparison.sampleId(), comparison.featureCount(), comparison.abundanceTotal(), featureDelta, overlapCount, Math.min(current.featureEntries().size(), comparison.featureEntries().size()), current.top1Ratio(), current.top3Ratio(), comparison.top1Ratio(), comparison.top3Ratio(), comparison.dominanceLabel(), currentUniqueTaxa, comparisonUniqueTaxa, recommendation);
    }

    private String buildComparisonRecommendation(List<Map<String, Object>> allSamples, SampleStructure current, SampleStructure comparison, long featureDelta, int overlapCount) {
        boolean currentMoreBalanced = current.top3Ratio() + 0.10 < comparison.top3Ratio();
        boolean currentMoreConcentrated = current.top1Ratio() > comparison.top1Ratio() + 0.10;
        int currentRank = rankByFeatureCount(allSamples, current.sampleId());
        int comparisonRank = rankByFeatureCount(allSamples, comparison.sampleId());

        if (featureDelta >= 80 && currentMoreBalanced) {
            return "当前样本信息量更高且结构更均衡，可作为同患者内部分析的优先样本。";
        }
        if (featureDelta <= -80 && comparisonRank < currentRank) {
            return "对照样本的非零特征更丰富，若要看更完整的群落结构，可优先切换到对照样本。";
        }
        if (currentMoreConcentrated && overlapCount <= 2) {
            return "当前样本头部优势菌更集中且与对照重叠较少，适合重点关注是否存在阶段性优势菌变化。";
        }
        if (overlapCount >= 4) {
            return "两份样本的头部结构较为稳定，可把重点继续放回与健康组的偏离和预测结果。";
        }
        return "两份样本结构存在可见差异，可结合样本日期或采样背景进一步判断是否有阶段性变化。";
    }

    private SampleStructure analyzeStructure(Map<String, Object> sample, List<Map<String, Object>> features) {
        List<FeatureEntry> entries = new ArrayList<>();
        for (Map<String, Object> feature : features) {
            String rawName = preferNonBlank(safeText(feature.get("microbeName")), safeText(feature.get("microbe_name")), safeText(feature.get("name")), "未命名特征");
            double value = safeDouble(preferNonNull(feature.get("abundanceValue"), feature.get("abundance_value"), feature.get("value")));
            entries.add(new FeatureEntry(normalizeFeatureKey(rawName), shortenTaxonName(rawName), value));
        }

        double abundanceTotal = safeDouble(sample.get("abundanceTotal"));
        if (abundanceTotal <= 0D) {
            abundanceTotal = entries.stream().mapToDouble(FeatureEntry::value).sum();
        }

        double top1 = entries.isEmpty() ? 0D : entries.get(0).value();
        double top3 = entries.stream().limit(3).mapToDouble(FeatureEntry::value).sum();
        double top1Ratio = abundanceTotal > 0D ? top1 / abundanceTotal : 0D;
        double top3Ratio = abundanceTotal > 0D ? top3 / abundanceTotal : 0D;

        String dominanceLabel;
        if (top1Ratio >= 0.45 || top3Ratio >= 0.75) {
            dominanceLabel = "头部较集中";
        } else if (top1Ratio <= 0.20 && top3Ratio <= 0.50) {
            dominanceLabel = "结构较分散";
        } else {
            dominanceLabel = "集中度中等";
        }

        List<String> displayNames = entries.stream().map(FeatureEntry::displayName).filter(name -> name != null && !name.isBlank()).distinct().limit(3).collect(Collectors.toList());
        return new SampleStructure(safeText(sample.get("sampleId")), safeLong(sample.get("featureCount")), abundanceTotal, top1Ratio, top3Ratio, dominanceLabel, displayNames, entries);
    }

    private List<Map<String, Object>> loadTopFeatures(String patientId, String sampleId, int limit) {
        Map<String, Object> response = patientTool.getTopFeatures(patientId, sampleId, limit);
        if (!isSuccess(response)) {
            return List.of();
        }
        return asListOfMaps(asMap(response.get("data")).get("items"));
    }

    private Map<String, Object> resolveCurrentSample(List<Map<String, Object>> samples, String sampleId) {
        if (!isBlank(sampleId)) {
            for (Map<String, Object> sample : samples) {
                if (sampleId.equals(safeText(sample.get("sampleId")))) {
                    return sample;
                }
            }
        }
        return samples.stream().sorted(sampleComparator()).findFirst().orElse(samples.get(0));
    }

    private Map<String, Object> resolveComparisonSample(List<Map<String, Object>> samples, String currentSampleId) {
        return samples.stream().filter(sample -> !Objects.equals(currentSampleId, safeText(sample.get("sampleId")))).sorted(sampleComparator()).findFirst().orElse(null);
    }

    private Comparator<Map<String, Object>> sampleComparator() {
        return Comparator.comparingLong((Map<String, Object> sample) -> safeLong(sample.get("featureCount"))).reversed()
                .thenComparing((Map<String, Object> sample) -> safeDouble(sample.get("abundanceTotal")), Comparator.reverseOrder())
                .thenComparing(sample -> safeText(sample.get("sampleId")));
    }

    private int rankByFeatureCount(List<Map<String, Object>> samples, String sampleId) {
        List<Map<String, Object>> sorted = new ArrayList<>(samples);
        sorted.sort(sampleComparator());
        for (int index = 0; index < sorted.size(); index++) {
            if (Objects.equals(sampleId, safeText(sorted.get(index).get("sampleId")))) {
                return index + 1;
            }
        }
        return 1;
    }

    private String resolvePatientId(String question, AiContext context) {
        if (context != null && !isBlank(context.patientId())) {
            return context.patientId().trim();
        }
        if (question == null) {
            return null;
        }
        Matcher matcher = PATIENT_ID_PATTERN.matcher(question);
        return matcher.find() ? matcher.group() : null;
    }

    private String normalizeDiseaseLabel(String value) {
        if (isBlank(value)) {
            return "";
        }
        String normalized = value.trim();
        if ("healthy".equalsIgnoreCase(normalized)) {
            return "健康";
        }
        return normalized;
    }

    private AiChatResponse upstreamFailure(AiIntent intent, String summary, Map<String, Object> upstream) {
        String detail = String.valueOf(upstream.getOrDefault("detail", upstream.getOrDefault("error", "Unknown upstream error")));
        return new AiChatResponse(false, intent.name(), summary, List.of(new AiCard("error", "接口异常", detail)), List.of("确认主项目 5000 已启动", "稍后重试", "检查患者或样本是否存在"), Map.of("upstream", upstream));
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> asMap(Object value) {
        return value instanceof Map<?, ?> ? (Map<String, Object>) value : Collections.emptyMap();
    }

    @SuppressWarnings("unchecked")
    private List<Map<String, Object>> asListOfMaps(Object value) {
        if (!(value instanceof List<?>)) {
            return List.of();
        }
        List<Map<String, Object>> result = new ArrayList<>();
        for (Object item : (List<?>) value) {
            if (item instanceof Map<?, ?>) {
                result.add((Map<String, Object>) item);
            }
        }
        return result;
    }    private boolean isSuccess(Map<String, Object> response) {
        return Boolean.TRUE.equals(response.get("success"));
    }

    private String normalizeGender(String gender) {
        if (gender == null) {
            return "-";
        }
        String value = gender.trim().toLowerCase(Locale.ROOT);
        if ("male".equals(value) || "m".equals(value) || "男".equals(value)) {
            return "男";
        }
        if ("female".equals(value) || "f".equals(value) || "女".equals(value)) {
            return "女";
        }
        return gender;
    }

    private String shortenTaxonName(String rawName) {
        if (rawName == null || rawName.isBlank()) {
            return "未命名特征";
        }
        String[] parts = rawName.split("[|;.]");
        for (int index = parts.length - 1; index >= 0; index--) {
            String cleaned = stripTaxonPrefix(parts[index]);
            if (!cleaned.isBlank() && !"unclassified".equalsIgnoreCase(cleaned) && !"unknown".equalsIgnoreCase(cleaned)) {
                return cleaned.length() > 26 ? cleaned.substring(0, 23) + "..." : cleaned;
            }
        }
        String fallback = stripTaxonPrefix(rawName);
        return fallback.length() > 26 ? fallback.substring(0, 23) + "..." : fallback;
    }

    private String stripTaxonPrefix(String value) {
        String cleaned = value == null ? "" : value.trim();
        cleaned = cleaned.replaceFirst("^[a-z]__", "");
        cleaned = cleaned.replace('_', ' ').replace('-', ' ').trim();
        return cleaned;
    }

    private String normalizeFeatureKey(String value) {
        return value == null ? "" : value.trim().toLowerCase(Locale.ROOT);
    }

    private Object preferNonNull(Object... values) {
        for (Object value : values) {
            if (value != null) {
                return value;
            }
        }
        return null;
    }

    private String preferNonBlank(String... values) {
        for (String value : values) {
            if (value != null && !value.isBlank()) {
                return value;
            }
        }
        return "";
    }

    private String safeText(Object value) {
        return value == null ? "" : String.valueOf(value).trim();
    }

    private String blankToDash(Object value) {
        String text = safeText(value);
        return text.isBlank() ? "-" : text;
    }

    private boolean isBlank(String value) {
        return value == null || value.isBlank();
    }

    private long safeLong(Object value) {
        if (value instanceof Number number) {
            return number.longValue();
        }
        String text = safeText(value);
        if (text.isBlank()) {
            return 0L;
        }
        try {
            return Long.parseLong(text);
        } catch (NumberFormatException ex) {
            return 0L;
        }
    }

    private double safeDouble(Object value) {
        if (value instanceof Number number) {
            return number.doubleValue();
        }
        String text = safeText(value).replace(",", "");
        if (text.isBlank()) {
            return 0D;
        }
        try {
            return Double.parseDouble(text);
        } catch (NumberFormatException ex) {
            return 0D;
        }
    }

    private double ratioRank(Object value) {
        if (value instanceof Number number) {
            double ratio = number.doubleValue();
            return Double.isFinite(ratio) ? ratio : 9999D;
        }
        return safeDouble(value);
    }

    private String formatPercent(double value) {
        return String.format(Locale.ROOT, "%.1f%%", value * 100D);
    }

    private String formatProbability(double value) {
        return String.format(Locale.ROOT, "%.2f%%", value);
    }

    private String formatDecimal(double value) {
        return String.format(Locale.ROOT, "%.4f", value);
    }

    private record FeatureEntry(String key, String displayName, double value) {
    }

    private record SampleStructure(String sampleId,
                                   long featureCount,
                                   double abundanceTotal,
                                   double top1Ratio,
                                   double top3Ratio,
                                   String dominanceLabel,
                                   List<String> displayNames,
                                   List<FeatureEntry> featureEntries) {
        List<String> featureKeys() {
            return featureEntries.stream().map(FeatureEntry::key).collect(Collectors.toList());
        }
    }

    private record HealthInsight(String referenceScope,
                                 String referenceScopeLabel,
                                 long healthyPatientCount,
                                 long healthySampleCount,
                                 String deviationLabel,
                                 List<String> elevatedTaxa,
                                 List<String> depletedTaxa,
                                 List<String> deviationTopTaxa,
                                 String spatialDeviationLevel,
                                 double spatialDistanceZScore,
                                 double spatialDistancePercentile,
                                 List<String> healthySignatureTaxa,
                                 String recommendation) {
    }

    private record PredictionInsight(String top1Label,
                                     double top1Probability,
                                     String top2Label,
                                     double top2Probability,
                                     String consistencyLabel,
                                     String interpretation,
                                     String recommendation) {
    }

    private record ComparisonInsight(String comparisonSampleId,
                                     long comparisonFeatureCount,
                                     double comparisonAbundanceTotal,
                                     long featureDelta,
                                     int overlapCount,
                                     int windowSize,
                                     double currentTop1Ratio,
                                     double currentTop3Ratio,
                                     double comparisonTop1Ratio,
                                     double comparisonTop3Ratio,
                                     String comparisonDominanceLabel,
                                     List<String> currentUniqueTaxa,
                                     List<String> comparisonUniqueTaxa,
                                     String recommendation) {
    }
}
