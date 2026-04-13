package com.database.mico_ai.tool;

import com.database.mico_ai.dto.MedicalEvidenceHit;
import com.database.mico_ai.service.MedicalEvidenceService;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Map;

@Service
public class MedicalEvidenceTool {

    private final MedicalEvidenceService medicalEvidenceService;

    public MedicalEvidenceTool(MedicalEvidenceService medicalEvidenceService) {
        this.medicalEvidenceService = medicalEvidenceService;
    }

    public List<MedicalEvidenceHit> searchMedicalEvidence(String query, Integer maxResults) {
        return medicalEvidenceService.searchMedicalEvidence(query, maxResults);
    }

    public List<Map<String, Object>> listMedicalSources() {
        return medicalEvidenceService.listMedicalSources();
    }

    public Map<String, Object> medicalEvidenceStatus() {
        return medicalEvidenceService.medicalEvidenceStatus();
    }
}