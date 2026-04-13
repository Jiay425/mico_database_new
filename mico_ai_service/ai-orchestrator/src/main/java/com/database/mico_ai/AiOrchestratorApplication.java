package com.database.mico_ai;

import com.database.mico_ai.config.AiProviderProperties;
import com.database.mico_ai.config.FlowEngineProperties;
import com.database.mico_ai.config.KnowledgeBaseProperties;
import com.database.mico_ai.config.McpClientRegistryProperties;
import com.database.mico_ai.config.OpenAiCompatibleProperties;
import com.database.mico_ai.config.UpstreamProperties;
import com.database.mico_ai.config.WebSearchProperties;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.EnableConfigurationProperties;

@SpringBootApplication
@EnableConfigurationProperties({
        UpstreamProperties.class,
        AiProviderProperties.class,
        OpenAiCompatibleProperties.class,
        KnowledgeBaseProperties.class,
        WebSearchProperties.class,
        FlowEngineProperties.class,
        McpClientRegistryProperties.class
})
public class AiOrchestratorApplication {

    public static void main(String[] args) {
        SpringApplication.run(AiOrchestratorApplication.class, args);
    }
}
