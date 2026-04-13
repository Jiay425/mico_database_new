package com.database.mico_ai.tool;

import com.database.mico_ai.client.BusinessApiClient;
import dev.langchain4j.agent.tool.P;
import dev.langchain4j.agent.tool.Tool;
import org.springframework.stereotype.Component;

import java.util.Map;

@Component
public class PredictionTool {

    private final BusinessApiClient businessApiClient;

    public PredictionTool(BusinessApiClient businessApiClient) {
        this.businessApiClient = businessApiClient;
    }

    @Tool("Run disease prediction using a prepared prediction payload")
    public Map<String, Object> runPrediction(@P("prediction payload") Map<String, Object> payload) {
        return businessApiClient.runPrediction(payload);
    }
}
