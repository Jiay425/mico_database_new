package com.database.mico_ai.agent;

public final class AgentPromptCatalog {

    public static final String TRIAGE = "Decide whether the user wants dashboard QA, patient analysis, or prediction assistance.";
    public static final String DASHBOARD = "Answer only with facts grounded in live dashboard data from tools.";
    public static final String PATIENT = "Summarize patient and sample data clearly and avoid inventing clinical claims.";
    public static final String PREDICTION = "Explain prediction results with disease labels, top probabilities, and feature evidence.";

    private AgentPromptCatalog() {
    }
}
