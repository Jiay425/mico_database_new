package com.database.mico_ai.tool;

import com.database.mico_ai.config.KnowledgeBaseProperties;
import com.database.mico_ai.dto.KnowledgeHit;
import com.database.mico_ai.service.KnowledgeBaseService;
import org.springframework.stereotype.Service;

import java.util.Collection;
import java.util.List;
import java.util.Map;

@Service
public class KnowledgeTool {

    private final KnowledgeBaseService knowledgeBaseService;
    private final KnowledgeBaseProperties properties;

    public KnowledgeTool(KnowledgeBaseService knowledgeBaseService,
                         KnowledgeBaseProperties properties) {
        this.knowledgeBaseService = knowledgeBaseService;
        this.properties = properties;
    }

    public List<KnowledgeHit> searchKnowledge(String query,
                                              Collection<String> preferredLibraries,
                                              Integer maxResults) {
        int limit = maxResults == null || maxResults <= 0 ? properties.getMaxResults() : maxResults;
        return knowledgeBaseService.search(query, preferredLibraries, limit);
    }

    public Map<String, Object> knowledgeStatus() {
        return knowledgeBaseService.status();
    }
}
