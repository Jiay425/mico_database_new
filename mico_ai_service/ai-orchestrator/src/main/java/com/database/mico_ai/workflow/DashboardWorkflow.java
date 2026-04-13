package com.database.mico_ai.workflow;

import com.database.mico_ai.dto.AiCard;
import com.database.mico_ai.dto.AiChatResponse;
import com.database.mico_ai.dto.AiIntent;
import com.database.mico_ai.tool.DashboardTool;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Service
public class DashboardWorkflow {

    private final DashboardTool dashboardTool;

    public DashboardWorkflow(DashboardTool dashboardTool) {
        this.dashboardTool = dashboardTool;
    }

    @SuppressWarnings("unchecked")
    public AiChatResponse run() {
        Map<String, Object> result = dashboardTool.getDashboardSummary();
        if (!Boolean.TRUE.equals(result.get("success"))) {
            String detail = String.valueOf(result.getOrDefault("detail", result.getOrDefault("error", "Unknown upstream error")));
            return new AiChatResponse(
                    false,
                    AiIntent.DASHBOARD_QA.name(),
                    "首页总览接口当前不可用。",
                    List.of(new AiCard("error", "接口异常", detail)),
                    List.of("确认主项目 5000 已启动", "确认当前会话已登录", "稍后重试"),
                    Map.of("upstream", result)
            );
        }

        Map<String, Object> data = (Map<String, Object>) result.getOrDefault("data", Map.of());
        Map<String, Object> overview = (Map<String, Object>) data.getOrDefault("overview_stats", Map.of());
        List<Map<String, Object>> diseases = (List<Map<String, Object>>) data.getOrDefault("disease_data", List.of());
        List<Map<String, Object>> origins = (List<Map<String, Object>>) data.getOrDefault("patient_origin_data", List.of());
        List<Map<String, Object>> ageGender = (List<Map<String, Object>>) data.getOrDefault("age_gender_data", List.of());

        String totalPatients = String.valueOf(overview.getOrDefault("total_patients", "-"));
        String topDisease = diseases.isEmpty() ? "暂无" : String.valueOf(diseases.get(0).getOrDefault("disease", "暂无"));
        String topRegion = origins.isEmpty() ? "暂无" : String.valueOf(origins.get(0).getOrDefault("name", "暂无"));

        List<AiCard> cards = new ArrayList<>();
        cards.add(new AiCard("overview", "患者总量", totalPatients + " 条"));
        cards.add(new AiCard("disease", "主要疾病", topDisease));
        cards.add(new AiCard("region", "主要地区", topRegion));

        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("totalPatients", totalPatients);
        metadata.put("topDisease", topDisease);
        metadata.put("topRegion", topRegion);
        metadata.put("diseaseCount", diseases.size());
        metadata.put("regionCount", origins.size());
        metadata.put("ageGenderGroupCount", ageGender.size());
        metadata.put("topDiseases", summarizeValues(diseases, "disease", 3));
        metadata.put("topRegions", summarizeValues(origins, "name", 3));
        metadata.put("source", "business-api");

        return new AiChatResponse(
                true,
                AiIntent.DASHBOARD_QA.name(),
                "已读取首页真实数据，可继续追问疾病分布、地区来源和群体结构重点。",
                cards,
                List.of("查看疾病分布", "查看地区来源", "查看年龄性别结构"),
                metadata
        );
    }

    private List<String> summarizeValues(List<Map<String, Object>> items, String key, int limit) {
        List<String> values = new ArrayList<>();
        for (Map<String, Object> item : items) {
            if (item == null) {
                continue;
            }
            Object value = item.get(key);
            if (value == null) {
                continue;
            }
            String text = String.valueOf(value).trim();
            if (!text.isBlank()) {
                values.add(text);
            }
            if (values.size() >= limit) {
                break;
            }
        }
        return values;
    }
}