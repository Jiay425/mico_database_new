package com.database.mico_ai.mcp;

import com.database.mico_ai.service.FlowDefinitionRegistryService;
import com.database.mico_ai.service.McpClientRegistryService;
import com.database.mico_ai.service.TraceRecorderService;
import com.database.mico_ai.tool.KnowledgeTool;
import com.database.mico_ai.tool.MedicalEvidenceTool;
import com.database.mico_ai.tool.WebEvidenceTool;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Service
public class McpResourceService {

    private final TraceRecorderService traceRecorderService;
    private final KnowledgeTool knowledgeTool;
    private final WebEvidenceTool webEvidenceTool;
    private final MedicalEvidenceTool medicalEvidenceTool;
    private final FlowDefinitionRegistryService flowDefinitionRegistryService;
    private final McpClientRegistryService mcpClientRegistryService;
    private final ObjectMapper objectMapper;

    public McpResourceService(TraceRecorderService traceRecorderService,
                              KnowledgeTool knowledgeTool,
                              WebEvidenceTool webEvidenceTool,
                              MedicalEvidenceTool medicalEvidenceTool,
                              FlowDefinitionRegistryService flowDefinitionRegistryService,
                              McpClientRegistryService mcpClientRegistryService,
                              ObjectMapper objectMapper) {
        this.traceRecorderService = traceRecorderService;
        this.knowledgeTool = knowledgeTool;
        this.webEvidenceTool = webEvidenceTool;
        this.medicalEvidenceTool = medicalEvidenceTool;
        this.flowDefinitionRegistryService = flowDefinitionRegistryService;
        this.mcpClientRegistryService = mcpClientRegistryService;
        this.objectMapper = objectMapper;
    }

    public List<Map<String, Object>> listResources() {
        List<Map<String, Object>> resources = new ArrayList<>();
        resources.add(resource("business://field-dictionary", "Field Dictionary", "Project field definitions and business constraints", "text/markdown"));
        resources.add(resource("business://group-mapping", "Group Mapping", "Mapping between Group labels and disease names", "text/markdown"));
        resources.add(resource("business://page-analysis-template", "Page Analysis Templates", "Standard analysis templates for dashboard, patient, and prediction pages", "text/markdown"));
        resources.add(resource("model://label-definition", "Model Label Definition", "Prediction label and disease direction definitions", "text/markdown"));
        resources.add(resource("model://prediction-boundaries", "Prediction Boundaries", "Model output interpretation boundaries and caveats", "text/markdown"));
        resources.add(resource("medical://healthy-baseline-rules", "Healthy Baseline Rules", "Healthy reference interpretation rules", "text/markdown"));
        resources.add(resource("medical://evidence-sources", "Medical Evidence Sources", "Whitelisted external medical evidence sources and evidence types", "application/json"));
        resources.add(resource("harness://architecture-agent", "Harness Architecture", "System-of-record architecture for the agent platform", "text/markdown"));
        resources.add(resource("harness://evals-spec", "Harness Evals Specification", "Definition of what good answers look like and how they are evaluated", "text/markdown"));
        resources.add(resource("harness://agent-runbook", "Harness Runbook", "Operational runbook for failures, conflicts, and review loop", "text/markdown"));
        resources.add(resource("harness://runtime-status", "Harness Runtime Status", "Current runtime status for trace, knowledge, and external evidence", "application/json"));
        resources.add(resource("harness://evidence-policy", "Evidence Policy", "Normalized evidence-layer rules used by the harness runtime", "application/json"));
        resources.add(resource("harness://flow-definitions", "Harness Flow Definitions", "Configured staged flow definitions for each intent", "application/json"));
        resources.add(resource("harness://mcp-client-registry", "Harness MCP Registry", "Configured MCP connectors and runtime status", "application/json"));
        return resources;
    }

    public Map<String, Object> readResource(String uri, String sessionId) {
        Map<String, Object> result = switch (uri) {
            case "business://field-dictionary" -> fileResource(uri, "text/markdown", Paths.get("E:/DeskTop/java_code/mico_database_new/references/knowledge/business/field-dictionary.md"));
            case "business://group-mapping" -> fileResource(uri, "text/markdown", Paths.get("E:/DeskTop/java_code/mico_database_new/references/knowledge/business/group-mapping.md"));
            case "business://page-analysis-template" -> fileResource(uri, "text/markdown", Paths.get("E:/DeskTop/java_code/mico_database_new/references/knowledge/business/page-analysis-templates.md"));
            case "model://label-definition" -> fileResource(uri, "text/markdown", Paths.get("E:/DeskTop/java_code/mico_database_new/references/knowledge/model/label-definition.md"));
            case "model://prediction-boundaries" -> fileResource(uri, "text/markdown", Paths.get("E:/DeskTop/java_code/mico_database_new/references/knowledge/model/prediction-boundaries.md"));
            case "medical://healthy-baseline-rules" -> fileResource(uri, "text/markdown", Paths.get("E:/DeskTop/java_code/mico_database_new/references/knowledge/medical/healthy-baseline-rules.md"));
            case "medical://evidence-sources" -> inlineJsonResource(uri, Map.of("sources", medicalEvidenceTool.listMedicalSources()));
            case "harness://architecture-agent" -> fileResource(uri, "text/markdown", Paths.get("E:/DeskTop/java_code/mico_database_new/ARCHITECTURE-AGENT.md"));
            case "harness://evals-spec" -> fileResource(uri, "text/markdown", Paths.get("E:/DeskTop/java_code/mico_database_new/EVALS.md"));
            case "harness://agent-runbook" -> fileResource(uri, "text/markdown", Paths.get("E:/DeskTop/java_code/mico_database_new/AGENT-RUNBOOK.md"));
            case "harness://runtime-status" -> inlineJsonResource(uri, runtimeStatus());
            case "harness://evidence-policy" -> inlineJsonResource(uri, evidencePolicy());
            case "harness://flow-definitions" -> inlineJsonResource(uri, flowDefinitionRegistryService.status());
            case "harness://mcp-client-registry" -> inlineJsonResource(uri, Map.of(
                    "status", mcpClientRegistryService.status(),
                    "clients", mcpClientRegistryService.listClients()
            ));
            default -> throw new IllegalArgumentException("Resource not found: " + uri);
        };
        if (sessionId != null && !sessionId.isBlank()) {
            traceRecorderService.recordMcpResourceRead(sessionId, uri, Map.of("mimeType", mimeTypeOf(result)));
        }
        return result;
    }

    private Map<String, Object> runtimeStatus() {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.putAll(traceRecorderService.status());
        payload.putAll(knowledgeTool.knowledgeStatus());
        payload.putAll(webEvidenceTool.webSearchStatus());
        payload.putAll(medicalEvidenceTool.medicalEvidenceStatus());
        payload.putAll(flowDefinitionRegistryService.status());
        payload.putAll(mcpClientRegistryService.status());
        payload.put("harnessConnectorReady", true);
        return payload;
    }

    private Map<String, Object> evidencePolicy() {
        return Map.of(
                "priorityOrder", List.of("internal_data", "model_judgement", "internal_knowledge", "external_evidence"),
                "mustDiscloseEvidenceLayers", true,
                "mustKeepDiagnosisBoundary", true,
                "externalEvidenceIsSupplement", true,
                "reviewLoopEnabled", true,
                "medicalEvidenceUsesWhitelistedSources", true
        );
    }

    private Map<String, Object> fileResource(String uri, String mimeType, Path path) {
        try {
            String text = Files.readString(path, StandardCharsets.UTF_8);
            return Map.of("contents", List.of(Map.of(
                    "uri", uri,
                    "mimeType", mimeType,
                    "text", text
            )));
        } catch (IOException ex) {
            throw new IllegalStateException("Failed to read resource: " + uri, ex);
        }
    }

    private Map<String, Object> inlineJsonResource(String uri, Map<String, Object> body) {
        String text;
        try {
            text = objectMapper.writeValueAsString(body);
        } catch (Exception ex) {
            text = "{}";
        }
        return Map.of("contents", List.of(Map.of(
                "uri", uri,
                "mimeType", "application/json",
                "text", text
        )));
    }

    @SuppressWarnings("unchecked")
    private String mimeTypeOf(Map<String, Object> result) {
        Object contents = result.get("contents");
        if (!(contents instanceof List<?> list) || list.isEmpty()) {
            return "unknown";
        }
        Object first = list.get(0);
        if (!(first instanceof Map<?, ?> map)) {
            return "unknown";
        }
        Object mimeType = ((Map<String, Object>) map).get("mimeType");
        return mimeType == null ? "unknown" : String.valueOf(mimeType);
    }

    private Map<String, Object> resource(String uri, String name, String description, String mimeType) {
        Map<String, Object> item = new LinkedHashMap<>();
        item.put("uri", uri);
        item.put("name", name);
        item.put("title", name);
        item.put("description", description);
        item.put("mimeType", mimeType);
        return item;
    }
}
