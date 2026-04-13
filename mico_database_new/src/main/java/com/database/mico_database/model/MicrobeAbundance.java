package com.database.mico_database.model;

import java.util.Date;
import java.util.Objects;

// 【关键修改】我们彻底移除了 @Data 注解
public class MicrobeAbundance {
    private Integer abundanceId;
    private String patientId;
    private String sampleId;
    private Date sampleDate;
    private String microbeName;
    private Double abundanceValue;
    private String unit;
    private String method;

    // --- 以下为手动实现的、保证一定存在的方法 ---
    public MicrobeAbundance() {
    }

    public Integer getAbundanceId() { return abundanceId; }
    public void setAbundanceId(Integer abundanceId) { this.abundanceId = abundanceId; }

    public String getPatientId() { return patientId; }
    public void setPatientId(String patientId) { this.patientId = patientId; }

    public String getSampleId() { return sampleId; }
    public void setSampleId(String sampleId) { this.sampleId = sampleId; }

    public Date getSampleDate() { return sampleDate; }
    public void setSampleDate(Date sampleDate) { this.sampleDate = sampleDate; }

    public String getMicrobeName() { return microbeName; }
    public void setMicrobeName(String microbeName) { this.microbeName = microbeName; }

    public Double getAbundanceValue() { return abundanceValue; }
    public void setAbundanceValue(Double abundanceValue) { this.abundanceValue = abundanceValue; }

    public String getUnit() { return unit; }
    public void setUnit(String unit) { this.unit = unit; }

    public String getMethod() { return method; }
    public void setMethod(String method) { this.method = method; }

    // 以下是 @Data 应该生成的其他方法
    @Override
    public String toString() {
        return "MicrobeAbundance{" + "abundanceId=" + abundanceId + ", patientId='" + patientId + '\'' + ", sampleId='" + sampleId + '\'' + ", sampleDate=" + sampleDate + ", microbeName='" + microbeName + '\'' + ", abundanceValue=" + abundanceValue + ", unit='" + unit + '\'' + ", method='" + method + '\'' + '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        MicrobeAbundance that = (MicrobeAbundance) o;
        return Objects.equals(abundanceId, that.abundanceId) && Objects.equals(patientId, that.patientId) && Objects.equals(sampleId, that.sampleId) && Objects.equals(sampleDate, that.sampleDate) && Objects.equals(microbeName, that.microbeName) && Objects.equals(abundanceValue, that.abundanceValue) && Objects.equals(unit, that.unit) && Objects.equals(method, that.method);
    }

    @Override
    public int hashCode() {
        return Objects.hash(abundanceId, patientId, sampleId, sampleDate, microbeName, abundanceValue, unit, method);
    }
}