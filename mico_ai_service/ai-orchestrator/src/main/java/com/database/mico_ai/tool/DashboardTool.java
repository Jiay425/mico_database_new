package com.database.mico_ai.tool;

import com.database.mico_ai.client.BusinessApiClient;
import dev.langchain4j.agent.tool.P;
import dev.langchain4j.agent.tool.Tool;
import org.springframework.stereotype.Component;

import java.util.Map;

@Component
public class DashboardTool {

    private final BusinessApiClient businessApiClient;

    public DashboardTool(BusinessApiClient businessApiClient) {
        this.businessApiClient = businessApiClient;
    }

    @Tool("Get dashboard summary data from the main business system")
    public Map<String, Object> getDashboardSummary() {
        return businessApiClient.getDashboardSummary();
    }
}
