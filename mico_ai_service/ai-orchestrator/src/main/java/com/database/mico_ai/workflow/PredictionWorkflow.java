package com.database.mico_ai.workflow;

import com.database.mico_ai.dto.AiCard;
import com.database.mico_ai.dto.AiChatResponse;
import com.database.mico_ai.dto.AiContext;
import com.database.mico_ai.dto.AiIntent;
import com.database.mico_ai.tool.PatientTool;
import com.database.mico_ai.tool.PredictionTool;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Service
public class PredictionWorkflow {

    private final PatientTool patientTool;
    private final PredictionTool predictionTool;

    public PredictionWorkflow(PatientTool patientTool, PredictionTool predictionTool) {
        this.patientTool = patientTool;
        this.predictionTool = predictionTool;
    }

    @SuppressWarnings("unchecked")
    public AiChatResponse run(AiContext context) {
        String patientId = context == null ? null : context.patientId();
        String sampleId = context == null ? null : context.sampleId();

        if (patientId == null || patientId.isBlank() || sampleId == null || sampleId.isBlank()) {
            return new AiChatResponse(
                    false,
                    AiIntent.PREDICTION_ASSISTANT.name(),
                    "请先提供 patientId 和 sampleId，再执行预测。",
                    List.of(),
                    List.of("从患者详情页进入", "选择当前样本"),
                    Map.of()
            );
        }

        Map<String, Object> payloadResponse = patientTool.getPredictionPayload(patientId, sampleId);
        if (!Boolean.TRUE.equals(payloadResponse.get("success"))) {
            String detail = String.valueOf(payloadResponse.getOrDefault("detail", payloadResponse.getOrDefault("error", "Unknown upstream error")));
            return new AiChatResponse(
                    false,
                    AiIntent.PREDICTION_ASSISTANT.name(),
                    "预测载荷接口当前不可用。",
                    List.of(new AiCard("error", "接口异常", detail)),
                    List.of("确认主项目 5000 已启动", "确认样本是否存在", "稍后重试"),
                    Map.of("upstream", payloadResponse)
            );
        }

        Map<String, Object> payload = (Map<String, Object>) payloadResponse.getOrDefault("data", Map.of());
        if (payload.isEmpty()) {
            return new AiChatResponse(
                    false,
                    AiIntent.PREDICTION_ASSISTANT.name(),
                    "未能生成预测所需样本载荷。",
                    List.of(),
                    List.of("检查标准丰度表", "确认样本是否存在"),
                    Map.of("patientId", patientId, "sampleId", sampleId)
            );
        }

        Map<String, Object> prediction = predictionTool.runPrediction(payload);
        if (!Boolean.TRUE.equals(prediction.get("success")) && !prediction.containsKey("predictions")) {
            String detail = String.valueOf(prediction.getOrDefault("detail", prediction.getOrDefault("error", "Unknown prediction error")));
            return new AiChatResponse(
                    false,
                    AiIntent.PREDICTION_ASSISTANT.name(),
                    "预测接口当前不可用。",
                    List.of(new AiCard("error", "预测异常", detail)),
                    List.of("确认模型服务已启动", "确认样本是否有效", "稍后重试"),
                    Map.of("upstream", prediction, "patientId", patientId, "sampleId", sampleId)
            );
        }

        List<Map<String, Object>> predictions = (List<Map<String, Object>>) prediction.getOrDefault("predictions", List.of());
        Map<String, Object> top1 = predictions.isEmpty() ? Map.of() : predictions.get(0);
        Map<String, Object> top2 = predictions.size() > 1 ? predictions.get(1) : Map.of();

        String top1Label = preferNonBlank(stringValue(top1.get("label")), stringValue(top1.get("group")), "未返回标签");
        String top2Label = preferNonBlank(stringValue(top2.get("label")), stringValue(top2.get("group")), "暂无");
        double top1Probability = toDouble(top1.get("probability"));
        double top2Probability = toDouble(top2.get("probability"));

        AiCard resultCard = new AiCard(
                "prediction",
                "预测结论",
                String.format("Top1：%s（%.2f%%）；Top2：%s（%.2f%%）", top1Label, top1Probability, top2Label, top2Probability)
        );
        AiCard payloadCard = new AiCard("payload", "样本载荷", String.format("patientId=%s, sampleId=%s", patientId, sampleId));

        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("patientId", patientId);
        metadata.put("sampleId", sampleId);
        metadata.put("predictionTop1Label", top1Label);
        metadata.put("predictionTop1Probability", top1Probability);
        metadata.put("predictionTop2Label", top2Label);
        metadata.put("predictionTop2Probability", top2Probability);
        metadata.put("prediction", prediction);

        return new AiChatResponse(
                true,
                AiIntent.PREDICTION_ASSISTANT.name(),
                "已完成当前样本预测，可继续追问模型方向、概率差异和解释边界。",
                List.of(resultCard, payloadCard),
                List.of("查看概率分布", "解释模型边界", "切换样本重试"),
                metadata
        );
    }

    private String stringValue(Object value) {
        return value == null ? "" : String.valueOf(value).trim();
    }

    private String preferNonBlank(String... values) {
        for (String value : values) {
            if (value != null && !value.isBlank()) {
                return value;
            }
        }
        return "";
    }

    private double toDouble(Object value) {
        if (value instanceof Number number) {
            return number.doubleValue();
        }
        try {
            return Double.parseDouble(stringValue(value));
        } catch (NumberFormatException ex) {
            return 0D;
        }
    }
}