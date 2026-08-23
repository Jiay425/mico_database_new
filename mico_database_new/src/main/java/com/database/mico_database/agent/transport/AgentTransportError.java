package com.database.mico_database.agent.transport;

/** Safe HTTP error envelope; it intentionally contains no exception detail. */
public final class AgentTransportError {

    private final String code;
    private final String message;

    public AgentTransportError(String code, String message) {
        this.code = code;
        this.message = message;
    }

    public String getCode() {
        return code;
    }

    public String getMessage() {
        return message;
    }
}
