package com.database.mico_ai.service;

import com.database.mico_ai.dto.AiContext;
import com.database.mico_ai.dto.AiIntent;
import org.springframework.stereotype.Service;

@Service
public class IntentRouterService {

    public AiIntent resolveIntent(String message, AiContext context) {
        String text = message == null ? "" : message.toLowerCase();
        String page = context == null || context.page() == null ? "" : context.page().toLowerCase();

        if (containsAny(text, "预测", "pred", "risk", "概率") || page.contains("prediction")) {
            return AiIntent.PREDICTION_ASSISTANT;
        }
        if (containsAny(text, "首页", "总览", "趋势", "dashboard", "疾病分布") || page.contains("index")) {
            return AiIntent.DASHBOARD_QA;
        }
        if (containsAny(text, "患者", "样本", "菌", "丰度", "patient", "sample") || page.contains("patient")) {
            return AiIntent.PATIENT_ANALYSIS;
        }
        return AiIntent.PATIENT_ANALYSIS;
    }

    private boolean containsAny(String text, String... terms) {
        for (String term : terms) {
            if (text.contains(term.toLowerCase())) {
                return true;
            }
        }
        return false;
    }
}
