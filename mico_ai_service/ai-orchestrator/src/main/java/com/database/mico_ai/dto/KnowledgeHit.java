package com.database.mico_ai.dto;

public record KnowledgeHit(
        String library,
        String libraryLabel,
        String title,
        String sourcePath,
        String snippet,
        double score
) {
}
