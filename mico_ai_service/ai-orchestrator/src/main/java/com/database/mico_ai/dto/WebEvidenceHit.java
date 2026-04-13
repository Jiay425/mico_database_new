package com.database.mico_ai.dto;

public record WebEvidenceHit(
        String title,
        String url,
        String domain,
        String snippet,
        String publishedDate,
        double score
) {
}