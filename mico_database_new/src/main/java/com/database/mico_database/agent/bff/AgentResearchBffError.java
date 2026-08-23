package com.database.mico_database.agent.bff;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class AgentResearchBffError {
    private String code;
    private String message;
}
