package com.database.mico_ai.mcp;

import com.database.mico_ai.service.TraceRecorderService;
import com.database.mico_ai.service.FlowDefinitionRegistryService;
import com.database.mico_ai.service.McpClientRegistryService;
import com.database.mico_ai.tool.DashboardTool;
import com.database.mico_ai.tool.KnowledgeTool;
import com.database.mico_ai.tool.MedicalEvidenceTool;
import com.database.mico_ai.tool.PatientTool;
import com.database.mico_ai.tool.PredictionTool;
import com.database.mico_ai.tool.WebEvidenceTool;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Service
public class McpToolService {

    private final PatientTool patientTool;
    private final PredictionTool predictionTool;
    private final DashboardTool dashboardTool;
    private final KnowledgeTool knowledgeTool;
    private final WebEvidenceTool webEvidenceTool;
    private final MedicalEvidenceTool medicalEvidenceTool;
    private final TraceRecorderService traceRecorderService;
    private final McpClientRegistryService mcpClientRegistryService;
    private final FlowDefinitionRegistryService flowDefinitionRegistryService;

    public McpToolService(PatientTool patientTool,
                          PredictionTool predictionTool,
                          DashboardTool dashboardTool,
                          KnowledgeTool knowledgeTool,
                          WebEvidenceTool webEvidenceTool,
                          MedicalEvidenceTool medicalEvidenceTool,
                          TraceRecorderService traceRecorderService,
                          McpClientRegistryService mcpClientRegistryService,
                          FlowDefinitionRegistryService flowDefinitionRegistryService) {
        this.patientTool = patientTool;
        this.predictionTool = predictionTool;
        this.dashboardTool = dashboardTool;
        this.knowledgeTool = knowledgeTool;
        this.webEvidenceTool = webEvidenceTool;
        this.medicalEvidenceTool = medicalEvidenceTool;
        this.traceRecorderService = traceRecorderService;
        this.mcpClientRegistryService = mcpClientRegistryService;
        this.flowDefinitionRegistryService = flowDefinitionRegistryService;
    }

    public List<Map<String, Object>> listTools() {
        List<Map<String, Object>> tools = new ArrayList<>();
        tools.add(tool("get_patient_summary", "Get patient summary", "Get patient detail and sample overview by patientId", schemaWithOptionalSession(
                stringProperty("patientId", "Patient id", true)
        )));
        tools.add(tool("get_sample_top_features", "Get sample top features", "Get top abundance features for a specific sample", schemaWithOptionalSession(
                stringProperty("patientId", "Patient id", true),
                stringProperty("sampleId", "Sample id", true),
                integerProperty("limit", "Returned feature count", false)
        )));
        tools.add(tool("get_healthy_reference", "Get healthy reference", "Get healthy baseline comparison for a sample", schemaWithOptionalSession(
                stringProperty("patientId", "Patient id", true),
                stringProperty("sampleId", "Sample id", true),
                integerProperty("limit", "Returned feature count", false)
        )));
        tools.add(tool("run_prediction", "Run prediction", "Run disease prediction for a specific sample", schemaWithOptionalSession(
                stringProperty("patientId", "Patient id", true),
                stringProperty("sampleId", "Sample id", true)
        )));
        tools.add(tool("get_dashboard_summary", "Get dashboard summary", "Get homepage dashboard summary", schemaWithOptionalSession()));
        tools.add(tool("search_knowledge", "Search internal knowledge", "Search business, medical, and model knowledge bases", schemaWithOptionalSession(
                stringProperty("query", "Search query", true),
                arrayProperty("libraries", "Preferred knowledge libraries", false),
                integerProperty("maxResults", "Returned result count", false)
        )));
        tools.add(tool("search_web_evidence", "Search web evidence", "Search external web evidence with source links", schemaWithOptionalSession(
                stringProperty("query", "Search query", true),
                integerProperty("maxResults", "Returned result count", false)
        )));
        tools.add(tool("search_medical_evidence", "Search medical evidence", "Search trusted external medical evidence from whitelisted sources", schemaWithOptionalSession(
                stringProperty("query", "Medical evidence query", true),
                integerProperty("maxResults", "Returned result count", false)
        )));
        tools.add(tool("list_medical_sources", "List medical sources", "List whitelisted medical evidence sources", schemaWithOptionalSession()));
        tools.add(tool("list_registered_mcp_clients", "List registered MCP clients", "List MCP connectors from harness registry", schemaWithOptionalSession()));
        tools.add(tool("get_registered_mcp_client", "Get registered MCP client", "Get one MCP connector detail by mcpId", schemaWithOptionalSession(
                stringProperty("mcpId", "MCP connector id", true)
        )));
        tools.add(tool("get_runtime_trace", "Get runtime trace", "Get stored harness runtime traces for a session", schema(
                stringProperty("sessionId", "Conversation session id", true),
                booleanProperty("latestOnly", "Whether to return only the latest trace", false)
        )));
        tools.add(tool("get_harness_status", "Get harness status", "Get trace, knowledge, and web evidence runtime status", schemaWithOptionalSession()));
        return tools;
    }

    public Map<String, Object> callTool(String name, Map<String, Object> arguments, String sessionId) {
        Map<String, Object> args = arguments == null ? Map.of() : arguments;
        Map<String, Object> result = switch (name) {
            case "get_patient_summary" -> callPatientSummary(args);
            case "get_sample_top_features" -> patientTool.getTopFeatures(
                    required(args, "patientId"),
                    required(args, "sampleId"),
                    intValue(args.get("limit"), 8)
            );
            case "get_healthy_reference" -> patientTool.getHealthyReference(
                    required(args, "patientId"),
                    required(args, "sampleId"),
                    intValue(args.get("limit"), 8)
            );
            case "run_prediction" -> callPrediction(args);
            case "get_dashboard_summary" -> dashboardTool.getDashboardSummary();
            case "search_knowledge" -> Map.of(
                    "success", true,
                    "results", knowledgeTool.searchKnowledge(
                            required(args, "query"),
                            stringList(args.get("libraries")),
                            optionalInt(args.get("maxResults"))
                    )
            );
            case "search_web_evidence" -> Map.of(
                    "success", true,
                    "results", webEvidenceTool.searchWebEvidence(
                            required(args, "query"),
                            optionalInt(args.get("maxResults"))
                    )
            );
            case "search_medical_evidence" -> Map.of(
                    "success", true,
                    "results", medicalEvidenceTool.searchMedicalEvidence(
                            required(args, "query"),
                            optionalInt(args.get("maxResults"))
                    )
            );
            case "list_medical_sources" -> Map.of(
                    "success", true,
                    "results", medicalEvidenceTool.listMedicalSources()
            );
            case "list_registered_mcp_clients" -> Map.of(
                    "success", true,
                    "clients", mcpClientRegistryService.listClients()
            );
            case "get_registered_mcp_client" -> mcpClientRegistryService.getClient(required(args, "mcpId"))
                    .<Map<String, Object>>map(client -> Map.of(
                            "success", true,
                            "client", client
                    ))
                    .orElseGet(() -> Map.of(
                            "success", false,
                            "error", "MCP client not found"
                    ));
            case "get_runtime_trace" -> booleanValue(args.get("latestOnly"))
                    ? traceRecorderService.latest(required(args, "sessionId"))
                    : traceRecorderService.list(required(args, "sessionId"));
            case "get_harness_status" -> harnessStatus();
            default -> throw new IllegalArgumentException("Unknown tool: " + name);
        };

        if (sessionId != null && !sessionId.isBlank()) {
            traceRecorderService.recordMcpToolCall(sessionId, name, traceDetail(args, result));
        }
        return result;
    }

    private Map<String, Object> harnessStatus() {
        Map<String, Object> status = new LinkedHashMap<>();
        status.putAll(traceRecorderService.status());
        status.putAll(knowledgeTool.knowledgeStatus());
        status.putAll(webEvidenceTool.webSearchStatus());
        status.putAll(medicalEvidenceTool.medicalEvidenceStatus());
        status.putAll(flowDefinitionRegistryService.status());
        status.putAll(mcpClientRegistryService.status());
        status.put("harnessConnectorReady", true);
        return status;
    }

    private Map<String, Object> callPatientSummary(Map<String, Object> args) {
        String patientId = required(args, "patientId");
        Map<String, Object> patient = patientTool.getPatient(patientId);
        Map<String, Object> samples = patientTool.listSamples(patientId);
        Map<String, Object> response = new LinkedHashMap<>();
        response.put("success", Boolean.TRUE.equals(patient.get("success")) && Boolean.TRUE.equals(samples.get("success")));
        response.put("patient", patient.get("data"));
        response.put("samples", samples.get("data"));
        return response;
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> callPrediction(Map<String, Object> args) {
        String patientId = required(args, "patientId");
        String sampleId = required(args, "sampleId");
        Map<String, Object> payloadResponse = patientTool.getPredictionPayload(patientId, sampleId);
        if (!Boolean.TRUE.equals(payloadResponse.get("success"))) {
            return payloadResponse;
        }
        Object data = payloadResponse.get("data");
        if (!(data instanceof Map<?, ?> payload)) {
            return Map.of("success", false, "error", "Prediction payload is empty");
        }
        return predictionTool.runPrediction((Map<String, Object>) payload);
    }

    private Map<String, Object> traceDetail(Map<String, Object> args, Map<String, Object> result) {
        Map<String, Object> detail = new LinkedHashMap<>();
        copyIfPresent(detail, args, "patientId");
        copyIfPresent(detail, args, "sampleId");
        copyIfPresent(detail, args, "query");
        copyIfPresent(detail, args, "mcpId");
        if (args.containsKey("libraries")) {
            detail.put("libraries", args.get("libraries"));
        }
        detail.put("success", result.getOrDefault("success", true));
        return detail;
    }

    private void copyIfPresent(Map<String, Object> target, Map<String, Object> source, String key) {
        Object value = source.get(key);
        if (value != null && !String.valueOf(value).isBlank()) {
            target.put(key, value);
        }
    }

    private Map<String, Object> tool(String name, String title, String description, Map<String, Object> inputSchema) {
        Map<String, Object> tool = new LinkedHashMap<>();
        tool.put("name", name);
        tool.put("title", title);
        tool.put("description", description);
        tool.put("inputSchema", inputSchema);
        return tool;
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> schemaWithOptionalSession(Map<String, Object>... properties) {
        List<Map<String, Object>> all = new ArrayList<>();
        for (Map<String, Object> property : properties) {
            if (property != null && !property.isEmpty()) {
                all.add(property);
            }
        }
        all.add(stringProperty("sessionId", "Optional trace correlation session id", false));
        return schema(all.toArray(new Map[0]));
    }

    private Map<String, Object> schema(Map<String, Object>... properties) {
        Map<String, Object> schema = new LinkedHashMap<>();
        Map<String, Object> props = new LinkedHashMap<>();
        List<String> required = new ArrayList<>();
        for (Map<String, Object> property : properties) {
            if (property == null || property.isEmpty()) {
                continue;
            }
            String name = String.valueOf(property.get("name"));
            props.put(name, property.get("schema"));
            if (Boolean.TRUE.equals(property.get("required"))) {
                required.add(name);
            }
        }
        schema.put("type", "object");
        schema.put("properties", props);
        if (!required.isEmpty()) {
            schema.put("required", required);
        }
        return schema;
    }

    private Map<String, Object> stringProperty(String name, String description, boolean required) {
        return property(name, required, Map.of("type", "string", "description", description));
    }

    private Map<String, Object> integerProperty(String name, String description, boolean required) {
        return property(name, required, Map.of("type", "integer", "description", description));
    }

    private Map<String, Object> booleanProperty(String name, String description, boolean required) {
        return property(name, required, Map.of("type", "boolean", "description", description));
    }

    private Map<String, Object> arrayProperty(String name, String description, boolean required) {
        return property(name, required, Map.of(
                "type", "array",
                "description", description,
                "items", Map.of("type", "string")
        ));
    }

    private Map<String, Object> property(String name, boolean required, Map<String, Object> schema) {
        Map<String, Object> property = new LinkedHashMap<>();
        property.put("name", name);
        property.put("required", required);
        property.put("schema", schema);
        return property;
    }

    private String required(Map<String, Object> args, String key) {
        Object value = args.get(key);
        String text = value == null ? "" : String.valueOf(value).trim();
        if (text.isBlank()) {
            throw new IllegalArgumentException("Missing required argument: " + key);
        }
        return text;
    }

    private Integer optionalInt(Object value) {
        if (value == null) {
            return null;
        }
        return intValue(value, 0);
    }

    private int intValue(Object value, int defaultValue) {
        if (value instanceof Number number) {
            return number.intValue();
        }
        if (value == null) {
            return defaultValue;
        }
        try {
            return Integer.parseInt(String.valueOf(value).trim());
        } catch (NumberFormatException ex) {
            return defaultValue;
        }
    }

    private boolean booleanValue(Object value) {
        if (value instanceof Boolean flag) {
            return flag;
        }
        return value != null && Boolean.parseBoolean(String.valueOf(value).trim());
    }

    private List<String> stringList(Object value) {
        if (!(value instanceof List<?> list)) {
            return List.of();
        }
        List<String> result = new ArrayList<>();
        for (Object item : list) {
            if (item == null) {
                continue;
            }
            String text = String.valueOf(item).trim();
            if (!text.isBlank()) {
                result.add(text);
            }
        }
        return result;
    }
}
