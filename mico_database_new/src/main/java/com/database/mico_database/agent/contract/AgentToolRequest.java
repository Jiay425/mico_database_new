package com.database.mico_database.agent.contract;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;

/** Strict envelope for one catalogued tool call. Arguments are checked against the catalog. */
public class AgentToolRequest {

    private String toolName;
    private String runId;
    private String toolCallId;
    private String requesterId;
    private String dataContractVersion = AgentContractConstants.DATA_CONTRACT_VERSION;
    private Set<String> requestedScopes = Collections.emptySet();
    private Map<String, Object> arguments = Collections.emptyMap();

    public String getToolName() {
        return toolName;
    }

    public void setToolName(String toolName) {
        this.toolName = toolName;
    }

    public String getRunId() {
        return runId;
    }

    public void setRunId(String runId) {
        this.runId = runId;
    }

    public String getToolCallId() {
        return toolCallId;
    }

    public void setToolCallId(String toolCallId) {
        this.toolCallId = toolCallId;
    }

    public String getRequesterId() {
        return requesterId;
    }

    public void setRequesterId(String requesterId) {
        this.requesterId = requesterId;
    }

    public String getDataContractVersion() {
        return dataContractVersion;
    }

    public void setDataContractVersion(String dataContractVersion) {
        this.dataContractVersion = dataContractVersion;
    }

    public Set<String> getRequestedScopes() {
        return requestedScopes;
    }

    public void setRequestedScopes(Set<String> requestedScopes) {
        if (requestedScopes == null) {
            this.requestedScopes = Collections.emptySet();
        } else {
            this.requestedScopes = Collections.unmodifiableSet(new LinkedHashSet<>(requestedScopes));
        }
    }

    public Map<String, Object> getArguments() {
        return arguments;
    }

    public void setArguments(Map<String, Object> arguments) {
        if (arguments == null) {
            this.arguments = Collections.emptyMap();
        } else {
            this.arguments = Collections.unmodifiableMap(new LinkedHashMap<>(arguments));
        }
    }
}
