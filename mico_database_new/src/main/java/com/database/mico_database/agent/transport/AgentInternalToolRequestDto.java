package com.database.mico_database.agent.transport;

import com.fasterxml.jackson.databind.JsonNode;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/** Parsed transport DTO. It is not the internal AgentToolRequest contract. */
public final class AgentInternalToolRequestDto {

    private final String toolName;
    private final String runId;
    private final String toolCallId;
    private final Map<String, JsonNode> arguments;

    public AgentInternalToolRequestDto(String toolName, String runId, String toolCallId,
                                       Map<String, JsonNode> arguments) {
        this.toolName = toolName;
        this.runId = runId;
        this.toolCallId = toolCallId;
        this.arguments = Collections.unmodifiableMap(new LinkedHashMap<>(arguments));
    }

    public String getToolName() {
        return toolName;
    }

    public String getRunId() {
        return runId;
    }

    public String getToolCallId() {
        return toolCallId;
    }

    public Map<String, JsonNode> getArguments() {
        return arguments;
    }
}
