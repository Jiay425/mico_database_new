package com.database.mico_database.agent.bff;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.client.HttpStatusCodeException;
import org.springframework.web.client.ResourceAccessException;
import org.springframework.web.client.RestClientException;
import org.springframework.web.client.RestTemplate;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

/** Java-to-Python client for the closed intent or Scientific Agent endpoint. */
public final class AgentResearchRuntimeClient {

    private static final Set<String> ROOT_FIELDS = new HashSet<>(Arrays.asList(
            "status", "workflow", "errorCode", "report"));
    private static final Set<String> SCIENTIFIC_ROOT_FIELDS = new HashSet<>(Arrays.asList(
            "runId", "taskId", "traceId", "status", "errorCode", "plannerMode",
            "actionCount", "report"));
    private static final Set<String> FORBIDDEN_FIELDS = new HashSet<>(Arrays.asList(
            "sourcesampleid", "internalrecordid", "cohortcondition", "patientid",
            "subjectid", "samplekey", "sql", "rawsql", "query", "where",
            "authorization", "token", "password", "headers"));
    private static final Set<String> FORBIDDEN_TEXT = new HashSet<>(Arrays.asList(
            "patient_data_manager", "bearer ", "sourcesampleid=", "internalrecordid=",
            "cohortcondition"));
    private static final Set<String> WORKFLOWS = new HashSet<>(Arrays.asList(
            "dynamic_read_query", "knowledge_retrieval"));
    private static final Set<String> STATUSES = new HashSet<>(Arrays.asList(
            "COMPLETED", "REJECTED", "FAILED", "NOT_IMPLEMENTED"));

    private final RestTemplate restTemplate;
    private final ObjectMapper objectMapper;
    private final AgentResearchBffProperties properties;

    public AgentResearchRuntimeClient(RestTemplate restTemplate, ObjectMapper objectMapper,
                                      AgentResearchBffProperties properties) {
        this.restTemplate = restTemplate;
        this.objectMapper = objectMapper;
        this.properties = properties;
    }

    public AgentResearchBffResponse execute(String requesterId, String runId,
                                            String taskId, String traceId, String question) {
        if (!properties.isEnabled()) {
            throw new AgentResearchBffException(503, "AGENT_BFF_DISABLED",
                    "The research Agent entry is disabled");
        }
        if (!properties.isUsable()) {
            throw new AgentResearchBffException(503, "AGENT_RUNTIME_NOT_CONFIGURED",
                    "The research Agent runtime is not configured");
        }
        ObjectNode payload = objectMapper.createObjectNode();
        payload.put("runId", runId);
        payload.put("taskId", taskId);
        payload.put("requesterId", requesterId);
        payload.put("question", question);
        payload.putArray("requestedScopes")
                .add("mico:query:read")
                .add("mico:evidence:read");
        if (properties.isScientificEndpoint()) {
            payload.withArray("requestedScopes").add("mico:research:read");
            payload.put("intent", "scientific_exploration");
            payload.putArray("allowedActions")
                    .add("execute_read_query")
                    .add("inspect_cohort")
                    .add("compare_groups")
                    .add("stratified_analysis")
                    .add("adjust_confounders")
                    .add("cross_project_validate")
                    .add("cross_disease_validate")
                    .add("retrieve_evidence")
                    .add("analyze_projection")
                    .add("finish");
            payload.put("maxActions", 6);
        } else {
            payload.putArray("allowedWorkflows")
                    .add("dynamic_read_query")
                    .add("knowledge_retrieval");
        }
        payload.put("createdAt", java.time.Instant.now().toString());
        payload.put("traceId", traceId);

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        headers.setBearerAuth(properties.getRuntimeToken());
        try {
            ResponseEntity<String> response = restTemplate.exchange(
                    properties.getRuntimeEndpoint(), HttpMethod.POST,
                    new HttpEntity<>(payload, headers), String.class);
            if (response.getStatusCodeValue() == 401 || response.getStatusCodeValue() == 403) {
                throw new AgentResearchBffException(502, "AGENT_RUNTIME_UNAUTHORIZED",
                        "The research Agent runtime rejected the internal request");
            }
            if (!response.getStatusCode().is2xxSuccessful()) {
                throw new AgentResearchBffException(502, "AGENT_RUNTIME_UNAVAILABLE",
                        "The research Agent runtime is unavailable");
            }
            return parseSafeResponse(response.getBody());
        } catch (AgentResearchBffException exception) {
            throw exception;
        } catch (ResourceAccessException exception) {
            throw new AgentResearchBffException(504, "AGENT_RUNTIME_TIMEOUT",
                    "The research Agent runtime did not respond in time");
        } catch (HttpStatusCodeException exception) {
            throw new AgentResearchBffException(502,
                    exception.getStatusCode().value() == 401 || exception.getStatusCode().value() == 403
                            ? "AGENT_RUNTIME_UNAUTHORIZED" : "AGENT_RUNTIME_UNAVAILABLE",
                    "The research Agent runtime is unavailable");
        } catch (RestClientException exception) {
            throw new AgentResearchBffException(502, "AGENT_RUNTIME_UNAVAILABLE",
                    "The research Agent runtime is unavailable");
        }
    }

    private AgentResearchBffResponse parseSafeResponse(String body) {
        try {
            JsonNode root = objectMapper.readTree(body == null ? "" : body);
            rejectUnknown(root, properties.isScientificEndpoint() ? SCIENTIFIC_ROOT_FIELDS : ROOT_FIELDS);
            String status = text(root, "status");
            if (!STATUSES.contains(status)) {
                throw new IllegalArgumentException("status is invalid");
            }
            String workflow = text(root, "workflow");
            if (!properties.isScientificEndpoint() && workflow != null && !WORKFLOWS.contains(workflow)) {
                throw new IllegalArgumentException("workflow is invalid");
            }
            rejectUnsafeContent(root.get("report"));
            AgentResearchBffResponse result = objectMapper.treeToValue(root, AgentResearchBffResponse.class);
            if (result.getReport() != null) {
                JsonNode report = result.getReport();
                if (report.has("cohortCondition") || report.has("sourceSampleId")
                        || report.has("internalRecordId")) {
                    throw new IllegalArgumentException("report contains forbidden locator fields");
                }
            }
            return result;
        } catch (Exception exception) {
            throw new AgentResearchBffException(502, "AGENT_RUNTIME_RESPONSE_INVALID",
                    "The research Agent returned an invalid safe response");
        }
    }

    private void rejectUnsafeContent(JsonNode node) {
        if (node == null || node.isNull()) {
            return;
        }
        if (node.isObject()) {
            Iterator<String> fields = node.fieldNames();
            while (fields.hasNext()) {
                String field = fields.next();
                if (FORBIDDEN_FIELDS.contains(field.toLowerCase())) {
                    throw new IllegalArgumentException("unsafe report field");
                }
                rejectUnsafeContent(node.get(field));
            }
        } else if (node.isArray()) {
            for (JsonNode child : node) {
                rejectUnsafeContent(child);
            }
        } else if (node.isTextual()) {
            String value = node.asText().toLowerCase();
            for (String marker : FORBIDDEN_TEXT) {
                if (value.contains(marker)) {
                    throw new IllegalArgumentException("unsafe report content");
                }
            }
        }
    }

    private String text(JsonNode root, String field) {
        JsonNode node = root == null ? null : root.get(field);
        return node == null || node.isNull() ? null : node.isTextual() ? node.asText() : null;
    }

    private void rejectUnknown(JsonNode node, Set<String> fields) {
        if (node == null || !node.isObject()) {
            throw new IllegalArgumentException("response is not an object");
        }
        Iterator<String> iterator = node.fieldNames();
        while (iterator.hasNext()) {
            if (!fields.contains(iterator.next())) {
                throw new IllegalArgumentException("unknown response field");
            }
        }
    }

    public static final class AgentResearchBffException extends RuntimeException {
        private final int status;
        private final String code;
        private final String safeMessage;

        public AgentResearchBffException(int status, String code, String safeMessage) {
            super(code);
            this.status = status;
            this.code = code;
            this.safeMessage = safeMessage;
        }

        public int getStatus() { return status; }
        public String getCode() { return code; }
        public String getSafeMessage() { return safeMessage; }
    }
}
