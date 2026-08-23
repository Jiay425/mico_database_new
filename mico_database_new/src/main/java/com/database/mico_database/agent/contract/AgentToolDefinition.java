package com.database.mico_database.agent.contract;

import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.Set;

public final class AgentToolDefinition {

    private final String toolName;
    private final String description;
    private final Set<String> requiredScopes;
    private final AgentRiskLevel riskLevel;
    private final boolean readOnly;
    private final boolean requiresSnapshot;
    private final int maxResultLimit;
    private final AgentArgumentSchema argumentSchema;

    public AgentToolDefinition(AgentToolName toolName,
                               String description,
                               Set<String> requiredScopes,
                               AgentRiskLevel riskLevel,
                               boolean readOnly,
                               boolean requiresSnapshot,
                               int maxResultLimit,
                               AgentArgumentSchema argumentSchema) {
        if (toolName == null || description == null || requiredScopes == null || argumentSchema == null) {
            throw new IllegalArgumentException("Tool definition fields must not be null");
        }
        if (maxResultLimit <= 0) {
            throw new IllegalArgumentException("maxResultLimit must be positive");
        }
        this.toolName = toolName.getWireName();
        this.description = description;
        this.requiredScopes = Collections.unmodifiableSet(new LinkedHashSet<>(requiredScopes));
        this.riskLevel = riskLevel;
        this.readOnly = readOnly;
        this.requiresSnapshot = requiresSnapshot;
        this.maxResultLimit = maxResultLimit;
        this.argumentSchema = argumentSchema;
    }

    public String getToolName() {
        return toolName;
    }

    public String getDescription() {
        return description;
    }

    public Set<String> getRequiredScopes() {
        return requiredScopes;
    }

    public AgentRiskLevel getRiskLevel() {
        return riskLevel;
    }

    public boolean isReadOnly() {
        return readOnly;
    }

    public boolean isRequiresSnapshot() {
        return requiresSnapshot;
    }

    public int getMaxResultLimit() {
        return maxResultLimit;
    }

    public AgentArgumentSchema getArgumentSchema() {
        return argumentSchema;
    }
}
