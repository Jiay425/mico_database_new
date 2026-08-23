package com.database.mico_database.agent.contract;

import java.time.Instant;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.Set;

/**
 * Strongly typed hand-off context for a future Agent Runtime. This is a model only;
 * it is intentionally not wired into a controller or authentication provider.
 */
public class TaskPacket {

    private String runId;
    private String sessionId;
    private String question;
    private PageContext pageContext;
    private String requesterId;
    private Set<String> requestedScopes = Collections.emptySet();
    private String riskPolicy;
    private Set<String> allowedTools = Collections.emptySet();
    private String dataContractVersion = AgentContractConstants.DATA_CONTRACT_VERSION;
    private Instant createdAt;

    public String getRunId() {
        return runId;
    }

    public void setRunId(String runId) {
        this.runId = runId;
    }

    public String getSessionId() {
        return sessionId;
    }

    public void setSessionId(String sessionId) {
        this.sessionId = sessionId;
    }

    public String getQuestion() {
        return question;
    }

    public void setQuestion(String question) {
        this.question = question;
    }

    public PageContext getPageContext() {
        return pageContext;
    }

    public void setPageContext(PageContext pageContext) {
        this.pageContext = pageContext;
    }

    public String getRequesterId() {
        return requesterId;
    }

    public void setRequesterId(String requesterId) {
        this.requesterId = requesterId;
    }

    public Set<String> getRequestedScopes() {
        return requestedScopes;
    }

    public void setRequestedScopes(Set<String> requestedScopes) {
        this.requestedScopes = immutableSet(requestedScopes);
    }

    public String getRiskPolicy() {
        return riskPolicy;
    }

    public void setRiskPolicy(String riskPolicy) {
        this.riskPolicy = riskPolicy;
    }

    public Set<String> getAllowedTools() {
        return allowedTools;
    }

    public void setAllowedTools(Set<String> allowedTools) {
        Set<String> copy = immutableSet(allowedTools);
        for (String toolName : copy) {
            if (!AgentToolCatalog.isRegistered(toolName)) {
                throw new IllegalArgumentException("allowedTools must come from the Agent Tool Catalog");
            }
        }
        this.allowedTools = copy;
    }

    public String getDataContractVersion() {
        return dataContractVersion;
    }

    public void setDataContractVersion(String dataContractVersion) {
        if (!AgentContractConstants.DATA_CONTRACT_VERSION.equals(dataContractVersion)) {
            throw new IllegalArgumentException("Only data contract version v1 is supported");
        }
        this.dataContractVersion = dataContractVersion;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(Instant createdAt) {
        this.createdAt = createdAt;
    }

    private static Set<String> immutableSet(Set<String> values) {
        if (values == null) {
            return Collections.emptySet();
        }
        return Collections.unmodifiableSet(new LinkedHashSet<>(values));
    }
}
