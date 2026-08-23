package com.database.mico_database.agent.bff;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.security.authentication.AnonymousAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;
import java.util.UUID;

/** The only browser-facing Agent entry: natural-language research and evidence retrieval. */
@RestController
@RequestMapping(path = "/agent")
public final class AgentResearchBffController {

    public static final String RESEARCH_PATH = "/agent/research";
    private static final Set<String> REQUEST_FIELDS = new HashSet<>(Arrays.asList("question"));

    private final AgentResearchRuntimeClient runtimeClient;
    private final AgentResearchBffProperties properties;
    private final ObjectMapper objectMapper;

    public AgentResearchBffController(AgentResearchRuntimeClient runtimeClient,
                                      AgentResearchBffProperties properties,
                                      ObjectMapper objectMapper) {
        this.runtimeClient = runtimeClient;
        this.properties = properties;
        this.objectMapper = objectMapper;
    }

    @PostMapping(path = "/research", consumes = MediaType.APPLICATION_JSON_VALUE,
            produces = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity<?> run(@RequestBody(required = false) String rawJson) {
        if (!properties.isEnabled()) {
            return error(HttpStatus.SERVICE_UNAVAILABLE, "AGENT_BFF_DISABLED",
                    "The research Agent entry is disabled");
        }
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        if (authentication == null || !authentication.isAuthenticated()
                || authentication instanceof AnonymousAuthenticationToken) {
            return error(HttpStatus.UNAUTHORIZED, "UNAUTHORIZED", "Login is required");
        }
        try {
            JsonNode root = objectMapper.readTree(rawJson == null ? "" : rawJson);
            rejectUnknown(root);
            String question = requiredQuestion(root == null ? null : root.get("question"));
            String requesterId = principalId(authentication.getName());
            String runId = opaqueId("run-");
            AgentResearchBffResponse response = runtimeClient.execute(
                    requesterId, runId, opaqueId("task-"), opaqueId("trace-"), question);
            response.setRunId(runId);
            if ("COMPLETED".equals(response.getStatus())
                    || "NOT_IMPLEMENTED".equals(response.getStatus())) {
                return ResponseEntity.ok(response);
            }
            return error(HttpStatus.BAD_GATEWAY,
                    response.getErrorCode() == null ? "AGENT_RUNTIME_FAILED" : response.getErrorCode(),
                    "The research Agent could not complete the request");
        } catch (AgentResearchRuntimeClient.AgentResearchBffException exception) {
            return error(HttpStatus.valueOf(exception.getStatus()), exception.getCode(), exception.getSafeMessage());
        } catch (IllegalArgumentException exception) {
            return error(HttpStatus.BAD_REQUEST, "INVALID_REQUEST",
                    "The research request does not match the closed contract");
        } catch (Exception exception) {
            return error(HttpStatus.BAD_GATEWAY, "AGENT_BFF_FAILED",
                    "The research Agent request could not be completed");
        }
    }

    private String requiredQuestion(JsonNode node) {
        if (node == null || !node.isTextual() || node.asText().trim().isEmpty()
                || node.asText().length() > 4096) {
            throw new IllegalArgumentException("question is invalid");
        }
        return node.asText();
    }

    private void rejectUnknown(JsonNode node) {
        if (node == null || !node.isObject()) {
            throw new IllegalArgumentException("request is not an object");
        }
        Iterator<String> fields = node.fieldNames();
        while (fields.hasNext()) {
            if (!REQUEST_FIELDS.contains(fields.next())) {
                throw new IllegalArgumentException("unknown request field");
            }
        }
    }

    private ResponseEntity<AgentResearchBffError> error(HttpStatus status, String code, String message) {
        return ResponseEntity.status(status).body(new AgentResearchBffError(code, message));
    }

    private String principalId(String name) {
        try {
            byte[] digest = MessageDigest.getInstance("SHA-256")
                    .digest((name == null ? "unknown" : name).getBytes(StandardCharsets.UTF_8));
            StringBuilder result = new StringBuilder("principal-");
            for (int index = 0; index < 16; index++) {
                result.append(String.format("%02x", digest[index]));
            }
            return result.toString();
        } catch (Exception exception) {
            return "principal-00000000000000000000000000000000";
        }
    }

    private String opaqueId(String prefix) {
        return prefix + UUID.randomUUID().toString().replace("-", "");
    }
}
