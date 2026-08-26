package com.database.mico_database.agent.contract;

import java.time.Instant;

/** Uniform response envelope for all catalogued tools. */
public class AgentToolResponse<T> {

    private String toolCallId;
    private String runId;
    private AgentToolStatus status;
    private String source;
    private Long rowCount;
    private String schemaVersion;
    private Instant generatedAt;
    private DataSnapshot dataSnapshot;
    private QualitySummary qualitySummary;
    private T data;
    private AgentToolError error;

    public static <T> AgentToolResponse<T> notImplemented(String toolCallId, String runId) {
        AgentToolResponse<T> response = base(toolCallId, runId, AgentToolStatus.NOT_IMPLEMENTED);
        response.error = AgentToolError.of(
                "NOT_IMPLEMENTED",
                "The registered tool has no implementation in the current Java read-only stage",
                null,
                false,
                "The tool is catalogued, but no safe implementation is available in this stage");
        return response;
    }

    public static <T> AgentToolResponse<T> rejected(String toolCallId, String runId, AgentToolError error) {
        AgentToolResponse<T> response = base(toolCallId, runId, AgentToolStatus.REJECTED);
        response.error = error;
        return response;
    }

    public static <T> AgentToolResponse<T> failed(String toolCallId, String runId, AgentToolError error) {
        AgentToolResponse<T> response = base(toolCallId, runId, AgentToolStatus.FAILED);
        response.error = error;
        return response;
    }

    public static <T> AgentToolResponse<T> completed(String toolCallId, String runId, String source,
                                                     Long rowCount, DataSnapshot dataSnapshot,
                                                     QualitySummary qualitySummary, T data) {
        if (dataSnapshot == null) {
            throw new IllegalArgumentException("A completed response requires dataSnapshot metadata");
        }
        AgentToolResponse<T> response = base(toolCallId, runId, AgentToolStatus.COMPLETED);
        response.source = source;
        response.rowCount = rowCount != null ? rowCount : dataSnapshot.getRowCount();
        response.dataSnapshot = dataSnapshot;
        response.qualitySummary = qualitySummary;
        response.data = data;
        return response;
    }

    /** Completed metadata response that is not a data snapshot. */
    public static <T> AgentToolResponse<T> metadataCompleted(String toolCallId, String runId,
                                                              String source, Long rowCount, T data) {
        AgentToolResponse<T> response = base(toolCallId, runId, AgentToolStatus.COMPLETED);
        response.source = source;
        response.rowCount = rowCount;
        response.data = data;
        return response;
    }

    private static <T> AgentToolResponse<T> base(String toolCallId, String runId, AgentToolStatus status) {
        AgentToolResponse<T> response = new AgentToolResponse<>();
        response.toolCallId = toolCallId;
        response.runId = runId;
        response.status = status;
        response.source = AgentContractConstants.CONTRACT_SOURCE;
        response.schemaVersion = AgentContractConstants.SCHEMA_VERSION;
        response.generatedAt = Instant.now();
        return response;
    }

    public String getToolCallId() {
        return toolCallId;
    }

    public void setToolCallId(String toolCallId) {
        this.toolCallId = toolCallId;
    }

    public String getRunId() {
        return runId;
    }

    public void setRunId(String runId) {
        this.runId = runId;
    }

    public AgentToolStatus getStatus() {
        return status;
    }

    public void setStatus(AgentToolStatus status) {
        this.status = status;
    }

    public String getSource() {
        return source;
    }

    public void setSource(String source) {
        this.source = source;
    }

    public Long getRowCount() {
        return rowCount;
    }

    public void setRowCount(Long rowCount) {
        this.rowCount = rowCount;
    }

    public String getSchemaVersion() {
        return schemaVersion;
    }

    public void setSchemaVersion(String schemaVersion) {
        this.schemaVersion = schemaVersion;
    }

    public Instant getGeneratedAt() {
        return generatedAt;
    }

    public void setGeneratedAt(Instant generatedAt) {
        this.generatedAt = generatedAt;
    }

    public DataSnapshot getDataSnapshot() {
        return dataSnapshot;
    }

    public void setDataSnapshot(DataSnapshot dataSnapshot) {
        this.dataSnapshot = dataSnapshot;
    }

    public QualitySummary getQualitySummary() {
        return qualitySummary;
    }

    public void setQualitySummary(QualitySummary qualitySummary) {
        this.qualitySummary = qualitySummary;
    }

    public T getData() {
        return data;
    }

    public void setData(T data) {
        this.data = data;
    }

    public AgentToolError getError() {
        return error;
    }

    public void setError(AgentToolError error) {
        this.error = error;
    }
}
