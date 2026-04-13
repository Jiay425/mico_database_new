package com.database.mico_database.model;

import lombok.Data;
import java.util.List;

@Data
public class Patient {
    private String patientId;
    private String patientName;
    private String group;
    private Integer age;
    private String gender;
    private String country;
    private Double bmi;
    private String bodySite;
    private String sequencingPlatform;
    private String disease;
    private List<MicrobeAbundance> microbialAbundanceData;
    private List<CytokineData> cytokineData;

// ========== 【请复制粘贴以下代码】开始 ==========

    // 1. AI 预测风险值 (存 0.1234 这种小数)
    private Double riskScore;

    // 2. 风险等级 (HIGH / LOW)
    private String riskLevel;

    // 3. AI 建议 (例如 "建议复查")
    private String aiSuggestion;

    // Getter 和 Setter 方法 (如果您没装 Lombok 插件，必须手动加这些)
    public Double getRiskScore() {
        return riskScore;
    }

    public void setRiskScore(Double riskScore) {
        this.riskScore = riskScore;
    }

    public String getRiskLevel() {
        return riskLevel;
    }

    public void setRiskLevel(String riskLevel) {
        this.riskLevel = riskLevel;
    }

    public String getAiSuggestion() {
        return aiSuggestion;
    }

    public void setAiSuggestion(String aiSuggestion) {
        this.aiSuggestion = aiSuggestion;
    }
    // ========== 【请复制粘贴以上代码】结束 ==========
}