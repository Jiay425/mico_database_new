package com.database.mico_ai.dto;

public record MedicalEvidenceHit(
        String title,
        String url,
        String domain,
        String source,
        String snippet,
        String publishedDate,
        String evidenceType,
        double score,
        String pmcid,
        String pmid,
        String section,
        String topic,
        String citation
) {
}
