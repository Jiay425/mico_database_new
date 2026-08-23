package com.database.mico_database.agent.transport;

import com.database.mico_database.agent.contract.AgentToolResponse;
import com.database.mico_database.agent.contract.AgentToolRequest;
import com.database.mico_database.agent.executor.AgentToolExecutorPort;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/** Internal, default-closed HTTP adapter for the Java Agent Tool boundary. */
@RestController
@RequestMapping(path = "/internal/agent/tools")
public final class AgentInternalToolController {

    public static final String EXECUTE_PATH = "/internal/agent/tools/execute";
    private static final String AUTHORIZATION_HEADER = "Authorization";

    private final AgentToolExecutorPort executor;
    private final AgentInternalToolRequestAdapter adapter;
    private final AgentInternalToolProperties properties;

    public AgentInternalToolController(AgentToolExecutorPort executor,
                                       AgentInternalToolRequestAdapter adapter,
                                       AgentInternalToolProperties properties) {
        if (executor == null || adapter == null || properties == null) {
            throw new IllegalArgumentException("internal Agent Tool transport dependencies are required");
        }
        this.executor = executor;
        this.adapter = adapter;
        this.properties = properties;
    }

    @PostMapping(path = "/execute", consumes = MediaType.APPLICATION_JSON_VALUE,
            produces = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity<?> execute(
            @RequestHeader(value = AUTHORIZATION_HEADER, required = false) String authorization,
            @RequestBody(required = false) String rawJson) {
        if (!properties.isEnabled()) {
            return error(HttpStatus.SERVICE_UNAVAILABLE, "INTERNAL_AGENT_DISABLED",
                    "The internal Agent Tool API is disabled");
        }
        if (!properties.matchesBearerToken(authorization)) {
            return error(HttpStatus.UNAUTHORIZED, "UNAUTHORIZED",
                    "Internal service authentication is required");
        }
        try {
            AgentToolRequest request = adapter.adapt(rawJson);
            AgentToolResponse<?> response = executor.execute(request);
            return ResponseEntity.ok(response);
        } catch (AgentInternalToolFormatException exception) {
            return error(HttpStatus.BAD_REQUEST, "INVALID_REQUEST",
                    "The request JSON does not match the closed internal Agent Tool contract");
        }
    }

    private ResponseEntity<AgentTransportError> error(HttpStatus status, String code, String message) {
        return ResponseEntity.status(status).body(new AgentTransportError(code, message));
    }
}
