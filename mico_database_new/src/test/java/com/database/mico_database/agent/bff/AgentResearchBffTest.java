package com.database.mico_database.agent.bff;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.http.HttpMethod;
import org.springframework.http.MediaType;
import org.springframework.http.converter.json.MappingJackson2HttpMessageConverter;
import org.springframework.http.converter.StringHttpMessageConverter;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.test.web.client.MockRestServiceServer;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;
import org.springframework.web.client.RestTemplate;

import java.util.Collections;

import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.not;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.header;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.method;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.requestTo;
import static org.springframework.test.web.client.response.MockRestResponseCreators.withSuccess;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.content;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

class AgentResearchBffTest {

    private static final String BASE_URL = "http://runtime.test";
    private static final String TOKEN = "runtime-token";
    private ObjectMapper objectMapper;
    private RestTemplate restTemplate;
    private MockRestServiceServer server;
    private MockMvc mockMvc;

    @BeforeEach
    void setUp() {
        objectMapper = new ObjectMapper().findAndRegisterModules();
        restTemplate = new RestTemplate();
        server = MockRestServiceServer.createServer(restTemplate);
        AgentResearchBffProperties properties = new AgentResearchBffProperties(true, BASE_URL, TOKEN);
        AgentResearchRuntimeClient client = new AgentResearchRuntimeClient(restTemplate, objectMapper, properties);
        mockMvc = MockMvcBuilders.standaloneSetup(
                        new AgentResearchBffController(client, properties, objectMapper))
                .setMessageConverters(new StringHttpMessageConverter(),
                        new MappingJackson2HttpMessageConverter(objectMapper))
                .build();
        SecurityContextHolder.getContext().setAuthentication(
                new UsernamePasswordAuthenticationToken("researcher", "N/A", Collections.emptyList()));
    }

    @AfterEach
    void tearDown() {
        SecurityContextHolder.clearContext();
    }

    @Test
    void naturalLanguageQuestionUsesClosedRuntimeContract() throws Exception {
        server.expect(requestTo(BASE_URL + "/internal/runtime/intent-runs"))
                .andExpect(method(HttpMethod.POST))
                .andExpect(header("Authorization", "Bearer " + TOKEN))
                .andExpect(org.springframework.test.web.client.match.MockRestRequestMatchers
                        .jsonPath("$.allowedWorkflows[0]").value("dynamic_read_query"))
                .andExpect(org.springframework.test.web.client.match.MockRestRequestMatchers
                        .jsonPath("$.requestedScopes[0]").value("mico:query:read"))
                .andExpect(org.springframework.test.web.client.match.MockRestRequestMatchers
                        .jsonPath("$.question").value("比较两组微生物丰度"))
                .andExpect(org.springframework.test.web.client.match.MockRestRequestMatchers
                        .jsonPath("$.sql").doesNotExist())
                .andRespond(withSuccess(safeRuntimeResponse(), MediaType.APPLICATION_JSON));

        mockMvc.perform(post(AgentResearchBffController.RESEARCH_PATH)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"question\":\"比较两组微生物丰度\"}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.runId").isString())
                .andExpect(jsonPath("$.workflow").value("dynamic_read_query"))
                .andExpect(jsonPath("$.report.rowCount").value(2))
                .andExpect(content().string(not(containsString("sourceSampleId"))))
                .andExpect(content().string(not(containsString("patient_data_manager"))));
        server.verify();
    }

    @Test
    void literatureQuestionCanReturnFullTextEvidenceRoute() throws Exception {
        server.expect(requestTo(BASE_URL + "/internal/runtime/intent-runs"))
                .andExpect(method(HttpMethod.POST))
                .andExpect(org.springframework.test.web.client.match.MockRestRequestMatchers
                        .jsonPath("$.allowedWorkflows[1]").value("knowledge_retrieval"))
                .andExpect(org.springframework.test.web.client.match.MockRestRequestMatchers
                        .jsonPath("$.requestedScopes[1]").value("mico:evidence:read"))
                .andRespond(withSuccess(
                        "{\"status\":\"COMPLETED\",\"workflow\":\"knowledge_retrieval\","
                                + "\"report\":{\"status\":\"COMPLETED\",\"topic\":\"microbiome\","
                                + "\"references\":[{\"title\":\"Full-text evidence\","
                                + "\"evidenceTier\":\"fulltext\",\"retrievalRoute\":\"hybrid\"}]}}",
                        MediaType.APPLICATION_JSON));

        mockMvc.perform(post(AgentResearchBffController.RESEARCH_PATH)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"question\":\"查找微生物组相关文献证据\"}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.workflow").value("knowledge_retrieval"))
                .andExpect(jsonPath("$.report.references[0].evidenceTier").value("fulltext"));
        server.verify();
    }

    @Test
    void browserCannotForgeRuntimeFieldsOrSql() throws Exception {
        mockMvc.perform(post(AgentResearchBffController.RESEARCH_PATH)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"question\":\"比较差异\",\"sql\":\"SELECT 1\"}"))
                .andExpect(status().isBadRequest())
                .andExpect(content().string(not(containsString("SELECT 1"))));
        server.verify();
    }

    @Test
    void missingLoginIsRejectedBeforeRuntimeCall() throws Exception {
        SecurityContextHolder.clearContext();
        mockMvc.perform(post(AgentResearchBffController.RESEARCH_PATH)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"question\":\"比较差异\"}"))
                .andExpect(status().isUnauthorized())
                .andExpect(jsonPath("$.code").value("UNAUTHORIZED"));
        server.verify();
    }

    @Test
    void disabledBffFailsClosed() throws Exception {
        AgentResearchBffProperties disabled = new AgentResearchBffProperties(false, BASE_URL, TOKEN);
        AgentResearchRuntimeClient client = new AgentResearchRuntimeClient(restTemplate, objectMapper, disabled);
        MockMvc disabledMvc = MockMvcBuilders.standaloneSetup(
                        new AgentResearchBffController(client, disabled, objectMapper))
                .setMessageConverters(new StringHttpMessageConverter(),
                        new MappingJackson2HttpMessageConverter(objectMapper))
                .build();
        disabledMvc.perform(post(AgentResearchBffController.RESEARCH_PATH)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"question\":\"比较差异\"}"))
                .andExpect(status().isServiceUnavailable())
                .andExpect(jsonPath("$.code").value("AGENT_BFF_DISABLED"));
        server.verify();
    }

    @Test
    void unsafeRuntimeResponseIsRejectedWithoutForwardingPayload() throws Exception {
        server.expect(requestTo(BASE_URL + "/internal/runtime/intent-runs"))
                .andRespond(withSuccess(
                        "{\"status\":\"COMPLETED\",\"workflow\":\"dynamic_read_query\","
                                + "\"report\":{\"sourceSampleId\":\"must-not-escape\"}}",
                        MediaType.APPLICATION_JSON));
        mockMvc.perform(post(AgentResearchBffController.RESEARCH_PATH)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"question\":\"比较差异\"}"))
                .andExpect(status().isBadGateway())
                .andExpect(jsonPath("$.code").value("AGENT_RUNTIME_RESPONSE_INVALID"))
                .andExpect(content().string(not(containsString("must-not-escape"))));
        server.verify();
    }

    @Test
    void runtimeAuthorizationFailureIsSanitized() throws Exception {
        server.expect(requestTo(BASE_URL + "/internal/runtime/intent-runs"))
                .andRespond(org.springframework.test.web.client.response.MockRestResponseCreators
                        .withStatus(org.springframework.http.HttpStatus.UNAUTHORIZED));
        mockMvc.perform(post(AgentResearchBffController.RESEARCH_PATH)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"question\":\"比较差异\"}"))
                .andExpect(status().isBadGateway())
                .andExpect(jsonPath("$.code").value("AGENT_RUNTIME_UNAUTHORIZED"))
                .andExpect(content().string(not(containsString(TOKEN))));
        server.verify();
    }

    private String safeRuntimeResponse() {
        return "{\"status\":\"COMPLETED\",\"workflow\":\"dynamic_read_query\","
                + "\"report\":{\"status\":\"COMPLETED\",\"source\":\"java_agent_read_model\","
                + "\"rowCount\":2,\"snapshotPersistence\":\"transient\","
                + "\"generatedAnalysis\":{\"metrics\":{\"row_count\":2}}}}";
    }
}
