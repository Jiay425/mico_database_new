package com.database.mico_database.model;

import java.util.Date;
import java.util.Objects;

// 【关键修改】我们彻底移除了 @Data 注解
public class CytokineData {
    private Integer cytokineId;
    private String patientId;
    private String sampleId;
    private Date sampleDate;
    private String cytokineName;
    private Double abundanceValue;
    private String unit;
    private String assayMethod;

    // --- 以下为手动实现的、保证一定存在的方法 ---
    public CytokineData() {
    }

    public Integer getCytokineId() { return cytokineId; }
    public void setCytokineId(Integer cytokineId) { this.cytokineId = cytokineId; }

    public String getPatientId() { return patientId; }
    public void setPatientId(String patientId) { this.patientId = patientId; }

    public String getSampleId() { return sampleId; }
    public void setSampleId(String sampleId) { this.sampleId = sampleId; }

    public Date getSampleDate() { return sampleDate; }
    public void setSampleDate(Date sampleDate) { this.sampleDate = sampleDate; }

    public String getCytokineName() { return cytokineName; }
    public void setCytokineName(String cytokineName) { this.cytokineName = cytokineName; }

    public Double getAbundanceValue() { return abundanceValue; }
    public void setAbundanceValue(Double abundanceValue) { this.abundanceValue = abundanceValue; }

    public String getUnit() { return unit; }
    public void setUnit(String unit) { this.unit = unit; }

    public String getAssayMethod() { return assayMethod; }
    public void setAssayMethod(String assayMethod) { this.assayMethod = assayMethod; }

    // 以下是 @Data 应该生成的其他方法
    @Override
    public String toString() {
        return "CytokineData{" + "cytokineId=" + cytokineId + ", patientId='" + patientId + '\'' + ", sampleId='" + sampleId + '\'' + ", sampleDate=" + sampleDate + ", cytokineName='" + cytokineName + '\'' + ", abundanceValue=" + abundanceValue + ", unit='" + unit + '\'' + ", assayMethod='" + assayMethod + '\'' + '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        CytokineData that = (CytokineData) o;
        return Objects.equals(cytokineId, that.cytokineId) && Objects.equals(patientId, that.patientId) && Objects.equals(sampleId, that.sampleId) && Objects.equals(sampleDate, that.sampleDate) && Objects.equals(cytokineName, that.cytokineName) && Objects.equals(abundanceValue, that.abundanceValue) && Objects.equals(unit, that.unit) && Objects.equals(assayMethod, that.assayMethod);
    }

    @Override
    public int hashCode() {
        return Objects.hash(cytokineId, patientId, sampleId, sampleDate, cytokineName, abundanceValue, unit, assayMethod);
    }
}