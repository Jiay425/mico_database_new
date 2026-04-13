package com.database.mico_database.service;

import com.database.mico_database.config.RabbitConfig;
import com.database.mico_database.mapper.PatientMapper;
import com.database.mico_database.model.Patient;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.amqp.rabbit.annotation.RabbitListener;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.Map;

@Service
@Slf4j
@ConditionalOnProperty(name = "app.rabbitmq.enabled", havingValue = "true")
public class AnalysisConsumer {

    private final RestTemplate restTemplate = new RestTemplate();

    @Autowired
    private ObjectMapper objectMapper;

    @Autowired
    private PatientMapper patientMapper;

    @Autowired
    private PatientService patientService;

    @Autowired
    private RabbitTemplate rabbitTemplate;

    @RabbitListener(queues = RabbitConfig.ANALYSIS_QUEUE_NAME)
    public void handleAnalysisTask(String patientId) {
        log.info("[consumer] received analysis task, id: {}", patientId);

        try {
            String pythonServiceUrl = "http://localhost:5001/predict";
            String jsonPayload = "{\"patient_id\": \"" + patientId + "\"}";

            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            HttpEntity<String> request = new HttpEntity<String>(jsonPayload, headers);

            log.info("[consumer] calling python AI service...");

            String responseJson = restTemplate.postForObject(pythonServiceUrl, request, String.class);
            log.info("[consumer] python response: {}", responseJson);

            Map result = objectMapper.readValue(responseJson, Map.class);

            Patient patientUpdate = new Patient();
            patientUpdate.setPatientName(patientId);

            if (result.get("risk_score") != null) {
                patientUpdate.setRiskScore(Double.valueOf(result.get("risk_score").toString()));
            }
            if (result.get("risk_level") != null) {
                patientUpdate.setRiskLevel((String) result.get("risk_level"));
            }
            if (result.get("suggestion") != null) {
                patientUpdate.setAiSuggestion((String) result.get("suggestion"));
            }

            patientMapper.updatePatientRisk(patientUpdate);

            String routingKey = "patient.update";
            rabbitTemplate.convertAndSend("patient_exchange", routingKey, patientId);
            log.info("[mq] sent cache invalidation notice, id: {}", patientId);

        } catch (Exception e) {
            log.error("[consumer] failed to process id: " + patientId, e);
        }
    }
}
