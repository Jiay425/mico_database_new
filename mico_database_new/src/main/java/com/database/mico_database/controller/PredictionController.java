package com.database.mico_database.controller;

import com.database.mico_database.model.PredictionRequest;
import com.database.mico_database.service.ModelGatewayService;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

import java.util.LinkedHashMap;
import java.util.Map;

@RestController
public class PredictionController {

    private final ModelGatewayService modelGatewayService;

    public PredictionController(ModelGatewayService modelGatewayService) {
        this.modelGatewayService = modelGatewayService;
    }

    @PostMapping("/predict_api")
    public Map<String, Object> predict(@RequestBody PredictionRequest request) {
        if (request == null || !request.hasPayload()) {
            return badRequest("Sample payload is required before prediction.");
        }
        return modelGatewayService.predict(request);
    }

    @PostMapping("/api/predict/run")
    public Map<String, Object> runPrediction(@RequestBody PredictionRequest request) {
        return predict(request);
    }

    @PostMapping("/api/abundance_curves")
    public Map<String, Object> abundanceCurves(@RequestBody PredictionRequest request) {
        if (request == null || !request.hasPayload()) {
            return badRequest("Sample payload is required for abundance curves.");
        }
        return modelGatewayService.abundanceCurves(request);
    }

    @PostMapping("/api/pcoa_data")
    public Map<String, Object> pcoaData(@RequestBody PredictionRequest request) {
        if (request == null || !request.hasPayload()) {
            return badRequest("Sample payload is required for PCoA data.");
        }
        return modelGatewayService.pcoaData(request);
    }

    private Map<String, Object> badRequest(String error) {
        Map<String, Object> response = new LinkedHashMap<>();
        response.put("success", false);
        response.put("model_ready", false);
        response.put("error", error);
        return response;
    }
}
