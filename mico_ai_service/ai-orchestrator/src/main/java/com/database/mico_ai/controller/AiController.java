package com.database.mico_ai.controller;

import com.database.mico_ai.dto.AiChatRequest;
import com.database.mico_ai.dto.AiChatResponse;
import com.database.mico_ai.service.LangChainAgentService;
import com.database.mico_ai.service.OrchestratorService;
import com.database.mico_ai.service.FlowDefinitionRegistryService;
import com.database.mico_ai.service.McpClientRegistryService;
import com.database.mico_ai.service.TraceRecorderService;
import com.database.mico_ai.tool.KnowledgeTool;
import com.database.mico_ai.tool.MedicalEvidenceTool;
import com.database.mico_ai.tool.WebEvidenceTool;
import jakarta.validation.Valid;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.LinkedHashMap;
import java.util.Map;

@RestController
@RequestMapping("/api/ai")
public class AiController {

    private final OrchestratorService orchestratorService;
    private final LangChainAgentService langChainAgentService;
    private final KnowledgeTool knowledgeTool;
    private final WebEvidenceTool webEvidenceTool;
    private final MedicalEvidenceTool medicalEvidenceTool;
    private final TraceRecorderService traceRecorderService;
    private final FlowDefinitionRegistryService flowDefinitionRegistryService;
    private final McpClientRegistryService mcpClientRegistryService;

    public AiController(OrchestratorService orchestratorService,
                        LangChainAgentService langChainAgentService,
                        KnowledgeTool knowledgeTool,
                        WebEvidenceTool webEvidenceTool,
                        MedicalEvidenceTool medicalEvidenceTool,
                        TraceRecorderService traceRecorderService,
                        FlowDefinitionRegistryService flowDefinitionRegistryService,
                        McpClientRegistryService mcpClientRegistryService) {
        this.orchestratorService = orchestratorService;
        this.langChainAgentService = langChainAgentService;
        this.knowledgeTool = knowledgeTool;
        this.webEvidenceTool = webEvidenceTool;
        this.medicalEvidenceTool = medicalEvidenceTool;
        this.traceRecorderService = traceRecorderService;
        this.flowDefinitionRegistryService = flowDefinitionRegistryService;
        this.mcpClientRegistryService = mcpClientRegistryService;
    }

    @GetMapping("/health")
    public Map<String, Object> health() {
        Map<String, Object> health = new LinkedHashMap<>();
        health.put("success", true);
        health.put("service", "ai-orchestrator");
        health.put("mode", langChainAgentService.llmAvailable() ? "langchain4j" : "rule-based-fallback");
        health.put("llmReady", langChainAgentService.llmAvailable());
        health.put("mcpEndpoint", "/mcp");
        health.put("mcpReady", true);
        health.put("taskPacketReady", true);
        health.put("executionPolicyReady", true);
        health.put("evidencePolicyReady", true);
        health.put("traceReady", true);
        health.put("harnessRuntimeReady", true);
        health.putAll(traceRecorderService.status());
        health.putAll(knowledgeTool.knowledgeStatus());
        health.putAll(webEvidenceTool.webSearchStatus());
        health.putAll(medicalEvidenceTool.medicalEvidenceStatus());
        health.putAll(flowDefinitionRegistryService.status());
        health.putAll(mcpClientRegistryService.status());
        return health;
    }

    @GetMapping("/traces/{sessionId}")
    public Map<String, Object> traces(@PathVariable String sessionId) {
        return traceRecorderService.list(sessionId);
    }

    @GetMapping("/traces/{sessionId}/latest")
    public Map<String, Object> latestTrace(@PathVariable String sessionId) {
        return traceRecorderService.latest(sessionId);
    }

    @PostMapping("/chat")
    public AiChatResponse chat(@Valid @RequestBody AiChatRequest request) {
        return orchestratorService.chat(request);
    }
}
