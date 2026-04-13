package com.database.mico_database.controller;

import com.database.mico_database.model.MicrobeAbundance;
import com.database.mico_database.model.Patient;
import com.database.mico_database.model.PredictionRequest;
import com.database.mico_database.service.DashboardService;
import com.database.mico_database.service.PatientService;
import com.github.pagehelper.PageInfo;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api")
public class AgentApiController {

    private final DashboardService dashboardService;
    private final PatientService patientService;

    public AgentApiController(DashboardService dashboardService, PatientService patientService) {
        this.dashboardService = dashboardService;
        this.patientService = patientService;
    }

    @GetMapping("/dashboard/summary")
    public Map<String, Object> dashboardSummary() {
        Map<String, Object> response = success();
        response.put("data", dashboardService.getDashboardDataDirect());
        return response;
    }

    @GetMapping("/patients/search")
    public Map<String, Object> searchPatients(@RequestParam("queryType") String queryType,
                                              @RequestParam("queryValue") String queryValue,
                                              @RequestParam(value = "page", defaultValue = "1") int page,
                                              @RequestParam(value = "size", defaultValue = "10") int size) {
        PageInfo<Patient> pageInfo = patientService.searchPatientsLite(queryType, queryValue, page, size);

        Map<String, Object> data = new LinkedHashMap<>();
        data.put("queryType", queryType);
        data.put("queryValue", queryValue);
        data.put("page", pageInfo.getPageNum());
        data.put("size", pageInfo.getPageSize());
        data.put("totalPages", pageInfo.getPages());
        data.put("totalRecords", pageInfo.getTotal());
        data.put("items", pageInfo.getList());

        Map<String, Object> response = success();
        response.put("data", data);
        return response;
    }

    @GetMapping("/patients/{patientId}")
    public ResponseEntity<Map<String, Object>> getPatient(@PathVariable("patientId") String patientId) {
        Patient patient = patientService.getPatientDetail(patientId);
        if (patient == null) {
            return notFound("Patient not found", "patientId", patientId);
        }

        Map<String, Object> data = new LinkedHashMap<>();
        data.put("patientId", patient.getPatientId());
        data.put("patientName", patient.getPatientName());
        data.put("group", patient.getGroup());
        data.put("disease", patient.getDisease());
        data.put("age", patient.getAge());
        data.put("gender", patient.getGender());
        data.put("country", patient.getCountry());
        data.put("bmi", patient.getBmi());
        data.put("bodySite", patient.getBodySite());
        data.put("sequencingPlatform", patient.getSequencingPlatform());
        data.put("sampleCount", patientService.listPatientSamples(patientId).size());
        data.put("cytokineCount", patient.getCytokineData() == null ? 0 : patient.getCytokineData().size());

        Map<String, Object> response = success();
        response.put("data", data);
        return ResponseEntity.ok(response);
    }

    @GetMapping("/patients/{patientId}/samples")
    public ResponseEntity<Map<String, Object>> listPatientSamples(@PathVariable("patientId") String patientId) {
        Patient patient = patientService.getPatientDetail(patientId);
        if (patient == null) {
            return notFound("Patient not found", "patientId", patientId);
        }

        Map<String, Object> data = new LinkedHashMap<>();
        data.put("patientId", patientId);
        data.put("items", patientService.listPatientSamples(patientId));

        Map<String, Object> response = success();
        response.put("data", data);
        return ResponseEntity.ok(response);
    }

    @GetMapping("/patients/{patientId}/samples/{sampleId}")
    public ResponseEntity<Map<String, Object>> getPatientSample(@PathVariable("patientId") String patientId,
                                                                @PathVariable("sampleId") String sampleId) {
        Map<String, Object> sample = patientService.getPatientSampleDetail(patientId, sampleId);
        if (sample == null) {
            return notFound("Sample not found", "sampleId", sampleId);
        }

        Map<String, Object> response = success();
        response.put("data", sample);
        return ResponseEntity.ok(response);
    }

    @GetMapping("/patients/{patientId}/samples/{sampleId}/top-features")
    public ResponseEntity<Map<String, Object>> getTopFeatures(@PathVariable("patientId") String patientId,
                                                              @PathVariable("sampleId") String sampleId,
                                                              @RequestParam(value = "limit", defaultValue = "20") int limit) {
        Map<String, Object> sample = patientService.getPatientSampleDetail(patientId, sampleId);
        if (sample == null) {
            return notFound("Sample not found", "sampleId", sampleId);
        }

        List<MicrobeAbundance> features = patientService.getTopSampleFeatures(patientId, sampleId, limit);
        Map<String, Object> data = new LinkedHashMap<>();
        data.put("patientId", patientId);
        data.put("sampleId", sampleId);
        data.put("limit", Math.max(1, Math.min(limit, 100)));
        data.put("items", features);

        Map<String, Object> response = success();
        response.put("data", data);
        return ResponseEntity.ok(response);
    }

    @GetMapping("/patients/{patientId}/samples/{sampleId}/prediction-payload")
    public ResponseEntity<Map<String, Object>> getPredictionPayload(@PathVariable("patientId") String patientId,
                                                                    @PathVariable("sampleId") String sampleId) {
        PredictionRequest payload = patientService.buildSamplePredictionRequest(patientId, sampleId);
        if (payload == null) {
            return notFound("Sample not found", "sampleId", sampleId);
        }

        Map<String, Object> response = success();
        response.put("data", payload);
        return ResponseEntity.ok(response);
    }

    @GetMapping("/patients/{patientId}/samples/{sampleId}/healthy-reference")
    public ResponseEntity<Map<String, Object>> getHealthyReference(@PathVariable("patientId") String patientId,
                                                                   @PathVariable("sampleId") String sampleId,
                                                                   @RequestParam(value = "limit", defaultValue = "8") int limit) {
        Map<String, Object> reference = patientService.buildHealthyReferenceComparison(patientId, sampleId, limit);
        if (reference == null) {
            return notFound("Sample not found", "sampleId", sampleId);
        }

        Map<String, Object> response = success();
        response.put("data", reference);
        return ResponseEntity.ok(response);
    }

    private Map<String, Object> success() {
        Map<String, Object> response = new LinkedHashMap<>();
        response.put("success", true);
        return response;
    }

    private ResponseEntity<Map<String, Object>> notFound(String error, String field, String value) {
        Map<String, Object> response = new LinkedHashMap<>();
        response.put("success", false);
        response.put("error", error);
        response.put(field, value);
        return ResponseEntity.status(HttpStatus.NOT_FOUND).body(response);
    }
}
