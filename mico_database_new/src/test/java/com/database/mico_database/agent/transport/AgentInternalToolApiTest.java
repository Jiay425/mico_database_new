package com.database.mico_database.agent.transport;

import com.database.mico_database.agent.contract.AgentToolError;
import com.database.mico_database.agent.contract.AgentToolRequest;
import com.database.mico_database.agent.contract.AgentToolRequestValidator;
import com.database.mico_database.agent.contract.AgentToolResponse;
import com.database.mico_database.agent.contract.DataSnapshot;
import com.database.mico_database.agent.contract.QualitySummary;
import com.database.mico_database.agent.executor.AgentToolExecutorPort;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.http.MediaType;
import org.springframework.http.converter.json.MappingJackson2HttpMessageConverter;
import org.springframework.http.converter.StringHttpMessageConverter;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;

import java.time.Instant;

import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.not;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.content;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

class AgentInternalToolApiTest {

    private static final String TOKEN = "test-service-token";
    private ObjectMapper objectMapper;
    private RecordingExecutor executor;
    private MockMvc mockMvc;

    @BeforeEach
    void setUp() {
        objectMapper = new ObjectMapper().findAndRegisterModules();
        executor = new RecordingExecutor();
        mockMvc = mockMvc(true, TOKEN);
    }

    @Test
    void disabledApiFailsClosedWithoutCallingExecutor() throws Exception {
        mockMvc(false, TOKEN).perform(post(AgentInternalToolController.EXECUTE_PATH)
                        .contentType(MediaType.APPLICATION_JSON)
                        .header("Authorization", "Bearer " + TOKEN)
                        .content(dynamicQueryJson()))
                .andExpect(status().isServiceUnavailable())
                .andExpect(jsonPath("$.code").value("INTERNAL_AGENT_DISABLED"));
        assertEquals(0, executor.calls);
    }

    @Test
    void validDynamicReadBindsJavaIdentityAndReturnsTransientEvidence() throws Exception {
        mockMvc.perform(post(AgentInternalToolController.EXECUTE_PATH)
                        .contentType(MediaType.APPLICATION_JSON)
                        .header("Authorization", "Bearer " + TOKEN)
                        .content(dynamicQueryJson()))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.status").value("COMPLETED"))
                .andExpect(jsonPath("$.dataSnapshot.snapshotPersistence").value("transient"))
                .andExpect(jsonPath("$.qualitySummary.subjectLinkStatus").value("unverified"));
        assertEquals("internal-agent-runtime", executor.lastRequest.getRequesterId());
        assertEquals("execute_read_query", executor.lastRequest.getToolName());
        assertEquals("SELECT 1 AS value LIMIT 10", executor.lastRequest.getArguments().get("sql"));
    }

    @Test
    void browserCannotForgeIdentityOrUnknownTopLevelFields() throws Exception {
        for (String field : new String[]{"requesterId", "requestedScopes", "unknownTopLevel"}) {
            String body = "{\"toolName\":\"execute_read_query\",\"runId\":\"run_1\","
                    + "\"toolCallId\":\"call_1\",\"arguments\":{\"sql\":\"SELECT 1 LIMIT 1\"},"
                    + "\"" + field + "\":\"rejected\"}";
            mockMvc.perform(post(AgentInternalToolController.EXECUTE_PATH)
                            .contentType(MediaType.APPLICATION_JSON)
                            .header("Authorization", "Bearer " + TOKEN)
                            .content(body))
                    .andExpect(status().isBadRequest())
                    .andExpect(content().string(not(containsString("rejected"))));
        }
        assertEquals(0, executor.calls);
    }

    @Test
    void unknownArgumentsAndWriteSqlAreRejectedByJavaContract() throws Exception {
        String unknown = "{\"toolName\":\"execute_read_query\",\"runId\":\"run_1\","
                + "\"toolCallId\":\"call_1\",\"arguments\":{\"sql\":\"SELECT 1 LIMIT 1\","
                + "\"table\":\"patients\"}}";
        mockMvc.perform(post(AgentInternalToolController.EXECUTE_PATH)
                        .contentType(MediaType.APPLICATION_JSON)
                        .header("Authorization", "Bearer " + TOKEN)
                        .content(unknown))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.status").value("REJECTED"));

        String write = "{\"toolName\":\"execute_read_query\",\"runId\":\"run_1\","
                + "\"toolCallId\":\"call_1\",\"arguments\":{\"sql\":\"DELETE FROM patients LIMIT 1\"}}";
        mockMvc.perform(post(AgentInternalToolController.EXECUTE_PATH)
                        .contentType(MediaType.APPLICATION_JSON)
                        .header("Authorization", "Bearer " + TOKEN)
                        .content(write))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.status").value("REJECTED"));
    }

    private MockMvc mockMvc(boolean enabled, String token) {
        AgentInternalToolController controller = new AgentInternalToolController(
                executor,
                new AgentInternalToolRequestAdapter(objectMapper),
                new AgentInternalToolProperties(enabled, token));
        return MockMvcBuilders.standaloneSetup(controller)
                .setMessageConverters(new StringHttpMessageConverter(),
                        new MappingJackson2HttpMessageConverter(objectMapper))
                .build();
    }

    private String dynamicQueryJson() {
        return "{\"toolName\":\"execute_read_query\",\"runId\":\"run_1\","
                + "\"toolCallId\":\"call_1\",\"arguments\":{\"sql\":\"SELECT 1 AS value LIMIT 10\","
                + "\"limit\":10}}";
    }

    private static final class RecordingExecutor implements AgentToolExecutorPort {
        private final AgentToolRequestValidator validator = new AgentToolRequestValidator();
        private int calls;
        private AgentToolRequest lastRequest;

        @Override
        public AgentToolResponse<?> execute(AgentToolRequest request) {
            calls++;
            lastRequest = request;
            if (!validator.validate(request).isValid()) {
                return AgentToolResponse.rejected(request.getToolCallId(), request.getRunId(),
                        validator.validate(request).getErrors().get(0));
            }
            DataSnapshot snapshot = new DataSnapshot();
            snapshot.setDataSnapshotId("transient-550e8400-e29b-41d4-a716-446655440000");
            snapshot.setDataSource("test-read-model");
            StringBuilder hash = new StringBuilder("sha256:");
            for (int index = 0; index < 64; index++) {
                hash.append('a');
            }
            snapshot.setQueryHash(hash.toString());
            snapshot.setRowCount(1L);
            snapshot.setGeneratedAt(Instant.parse("2026-01-01T00:00:00Z"));
            snapshot.setSnapshotPersistence("transient");
            QualitySummary quality = new QualitySummary();
            quality.setSubjectLinkStatus("unverified");
            return AgentToolResponse.completed(request.getToolCallId(), request.getRunId(),
                    "test-read-model", 1L, snapshot, quality, "typed-test-data");
        }
    }
}
