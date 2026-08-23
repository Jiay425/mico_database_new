package com.database.mico_database.agent.config;

import com.database.mico_database.agent.bff.AgentResearchBffProperties;
import com.database.mico_database.agent.bff.AgentResearchRuntimeClient;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.web.client.RestTemplate;

/** Wiring for the single default-closed browser-to-Runtime research entry. */
@Configuration
public class AgentResearchBffConfiguration {

    @Bean
    public AgentResearchBffProperties agentResearchBffProperties(
            @Value("${mico.agent.bff.enabled:false}") boolean enabled) {
        return new AgentResearchBffProperties(enabled,
                System.getenv("MICO_AGENT_RUNTIME_BASE_URL"),
                System.getenv("MICO_RUNTIME_INTERNAL_TOKEN"));
    }

    @Bean
    public RestTemplate agentResearchRuntimeRestTemplate() {
        SimpleClientHttpRequestFactory factory = new SimpleClientHttpRequestFactory();
        factory.setConnectTimeout(2000);
        factory.setReadTimeout(120000);
        return new RestTemplate(factory);
    }

    @Bean
    public AgentResearchRuntimeClient agentResearchRuntimeClient(
            RestTemplate agentResearchRuntimeRestTemplate,
            ObjectMapper objectMapper,
            AgentResearchBffProperties properties) {
        return new AgentResearchRuntimeClient(agentResearchRuntimeRestTemplate, objectMapper, properties);
    }
}
