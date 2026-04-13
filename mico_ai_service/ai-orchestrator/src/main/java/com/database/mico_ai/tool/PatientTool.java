package com.database.mico_ai.tool;

import com.database.mico_ai.client.BusinessApiClient;
import dev.langchain4j.agent.tool.P;
import dev.langchain4j.agent.tool.Tool;
import org.springframework.stereotype.Component;

import java.util.Map;

@Component
public class PatientTool {

    private final BusinessApiClient businessApiClient;

    public PatientTool(BusinessApiClient businessApiClient) {
        this.businessApiClient = businessApiClient;
    }

    @Tool("Search patients by id, name, or disease")
    public Map<String, Object> searchPatients(@P("query type") String queryType,
                                              @P("query value") String queryValue,
                                              @P("page number") int page,
                                              @P("page size") int size) {
        return businessApiClient.searchPatients(queryType, queryValue, page, size);
    }

    @Tool("Get patient detail by patient id")
    public Map<String, Object> getPatient(@P("patient id") String patientId) {
        return businessApiClient.getPatient(patientId);
    }

    @Tool("List sample summaries for a patient")
    public Map<String, Object> listSamples(@P("patient id") String patientId) {
        return businessApiClient.listPatientSamples(patientId);
    }

    @Tool("Get top microbial features for a patient sample")
    public Map<String, Object> getTopFeatures(@P("patient id") String patientId,
                                              @P("sample id") String sampleId,
                                              @P("feature limit") int limit) {
        return businessApiClient.getSampleTopFeatures(patientId, sampleId, limit);
    }

    @Tool("Get the healthy cohort reference for a patient sample")
    public Map<String, Object> getHealthyReference(@P("patient id") String patientId,
                                                   @P("sample id") String sampleId,
                                                   @P("feature limit") int limit) {
        return businessApiClient.getHealthyReference(patientId, sampleId, limit);
    }

    @Tool("Get the full prediction payload for a patient sample")
    public Map<String, Object> getPredictionPayload(@P("patient id") String patientId,
                                                    @P("sample id") String sampleId) {
        return businessApiClient.getPredictionPayload(patientId, sampleId);
    }
}
