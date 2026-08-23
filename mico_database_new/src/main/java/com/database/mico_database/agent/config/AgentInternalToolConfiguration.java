package com.database.mico_database.agent.config;

import com.database.mico_database.agent.contract.AgentToolRequestValidator;
import com.database.mico_database.agent.executor.AgentReadOnlyToolExecutor;
import com.database.mico_database.agent.readmodel.service.DynamicReadQueryService;
import com.database.mico_database.agent.transport.AgentInternalToolProperties;
import com.database.mico_database.agent.transport.AgentInternalToolRequestAdapter;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/** Spring wiring only; no database query occurs during bean construction. */
@Configuration
public class AgentInternalToolConfiguration {

    @Bean
    public AgentToolRequestValidator agentToolRequestValidator() {
        return new AgentToolRequestValidator();
    }

    @Bean
    public AgentReadOnlyToolExecutor agentReadOnlyToolExecutor(
            AgentToolRequestValidator validator, DynamicReadQueryService dynamicReadQueryService) {
        return new AgentReadOnlyToolExecutor(validator, dynamicReadQueryService);
    }

    @Bean
    public AgentInternalToolProperties agentInternalToolProperties(
            @Value("${mico.agent.internal.enabled:false}") boolean enabled) {
        return new AgentInternalToolProperties(enabled,
                System.getenv("MICO_AGENT_INTERNAL_TOKEN"));
    }

    @Bean
    public AgentInternalToolRequestAdapter agentInternalToolRequestAdapter(ObjectMapper objectMapper) {
        return new AgentInternalToolRequestAdapter(objectMapper);
    }
}
