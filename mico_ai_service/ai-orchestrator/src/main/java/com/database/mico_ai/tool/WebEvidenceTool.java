package com.database.mico_ai.tool;

import com.database.mico_ai.dto.WebEvidenceHit;
import com.database.mico_ai.service.WebSearchService;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Map;

@Service
public class WebEvidenceTool {

    private final WebSearchService webSearchService;

    public WebEvidenceTool(WebSearchService webSearchService) {
        this.webSearchService = webSearchService;
    }

    public List<WebEvidenceHit> searchWebEvidence(String query, Integer maxResults) {
        return webSearchService.search(query, maxResults);
    }

    public Map<String, Object> webSearchStatus() {
        return webSearchService.status();
    }
}