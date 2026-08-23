package com.database.mico_database.agent.bff;

import com.fasterxml.jackson.databind.JsonNode;
import lombok.Data;
import lombok.NoArgsConstructor;

/** Closed top-level response projection; report contents are validated by the client. */
@Data
@NoArgsConstructor
public class AgentResearchBffResponse {
    private String runId;
    private String status;
    private String workflow;
    private String errorCode;
    private JsonNode report;
}
