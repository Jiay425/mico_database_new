package com.database.mico_database.agent.contract;

/**
 * Legacy offline contract boundary retained for compatibility tests. The real
 * dynamic read path is owned by AgentReadOnlyToolExecutor; this class never
 * opens a database connection and returns NOT_IMPLEMENTED for valid calls.
 */
public final class AgentToolContractExecutor {

    private final AgentToolRequestValidator validator;

    public AgentToolContractExecutor() {
        this(new AgentToolRequestValidator());
    }

    public AgentToolContractExecutor(AgentToolRequestValidator validator) {
        this.validator = validator;
    }

    public <T> AgentToolResponse<T> execute(AgentToolRequest request) {
        AgentToolValidationResult validation = validator.validate(request);
        if (!validation.isValid()) {
            AgentToolError error = validation.getErrors().isEmpty()
                    ? AgentToolError.of("INVALID_REQUEST", "Tool request was rejected", null, false,
                    "The request did not satisfy the typed contract")
                    : validation.getErrors().get(0);
            String toolCallId = request == null ? null : request.getToolCallId();
            String runId = request == null ? null : request.getRunId();
            return AgentToolResponse.rejected(toolCallId, runId, error);
        }
        return AgentToolResponse.notImplemented(request.getToolCallId(), request.getRunId());
    }
}
