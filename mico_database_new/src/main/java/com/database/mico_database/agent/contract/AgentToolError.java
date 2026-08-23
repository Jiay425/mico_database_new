package com.database.mico_database.agent.contract;

/** Machine-readable rejection or failure reason. */
public class AgentToolError {

    private String code;
    private String message;
    private String field;
    private boolean retryable;
    private String policyReason;

    public AgentToolError() {
    }

    public AgentToolError(String code, String message, String field,
                          boolean retryable, String policyReason) {
        this.code = code;
        this.message = message;
        this.field = field;
        this.retryable = retryable;
        this.policyReason = policyReason;
    }

    public static AgentToolError of(String code, String message, String field,
                                    boolean retryable, String policyReason) {
        return new AgentToolError(code, message, field, retryable, policyReason);
    }

    public String getCode() {
        return code;
    }

    public void setCode(String code) {
        this.code = code;
    }

    public String getMessage() {
        return message;
    }

    public void setMessage(String message) {
        this.message = message;
    }

    public String getField() {
        return field;
    }

    public void setField(String field) {
        this.field = field;
    }

    public boolean isRetryable() {
        return retryable;
    }

    public void setRetryable(boolean retryable) {
        this.retryable = retryable;
    }

    public String getPolicyReason() {
        return policyReason;
    }

    public void setPolicyReason(String policyReason) {
        this.policyReason = policyReason;
    }
}
