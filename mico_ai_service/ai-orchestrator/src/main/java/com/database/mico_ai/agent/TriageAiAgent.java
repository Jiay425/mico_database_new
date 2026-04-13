package com.database.mico_ai.agent;

import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.UserMessage;
import dev.langchain4j.service.V;

public interface TriageAiAgent {

    @SystemMessage("You route requests for a medical data platform.\n"
            + "Decide the best intent using the provided message and page context.\n"
            + "Return exactly one token from this set only:\n"
            + "DASHBOARD_QA\n"
            + "PATIENT_ANALYSIS\n"
            + "PREDICTION_ASSISTANT")
    @UserMessage("User message: {{message}}\n"
            + "Page: {{page}}\n"
            + "Patient ID: {{patientId}}\n"
            + "Sample ID: {{sampleId}}")
    String route(@V("message") String message,
                 @V("page") String page,
                 @V("patientId") String patientId,
                 @V("sampleId") String sampleId);
}
