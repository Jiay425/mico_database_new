package com.database.mico_database.agent.contract;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Quality boundaries returned alongside a tool result. */
public class QualitySummary {

    private String subjectLinkStatus = AgentContractConstants.SUBJECT_LINK_STATUS_UNVERIFIED;
    private Map<String, Long> missingCounts = Collections.emptyMap();
    /** Null means the current payload did not provide a defensible duplicate count. */
    private Long duplicateCount;
    /** Null means the current payload did not provide a defensible orphan count. */
    private Long orphanCount;
    private List<String> warnings = Collections.emptyList();

    public String getSubjectLinkStatus() {
        return subjectLinkStatus;
    }

    public void setSubjectLinkStatus(String subjectLinkStatus) {
        this.subjectLinkStatus = subjectLinkStatus;
    }

    public Map<String, Long> getMissingCounts() {
        return missingCounts;
    }

    public void setMissingCounts(Map<String, Long> missingCounts) {
        if (missingCounts == null) {
            this.missingCounts = Collections.emptyMap();
        } else {
            this.missingCounts = Collections.unmodifiableMap(new LinkedHashMap<>(missingCounts));
        }
    }

    public Long getDuplicateCount() {
        return duplicateCount;
    }

    public void setDuplicateCount(Long duplicateCount) {
        this.duplicateCount = duplicateCount;
    }

    public Long getOrphanCount() {
        return orphanCount;
    }

    public void setOrphanCount(Long orphanCount) {
        this.orphanCount = orphanCount;
    }

    public List<String> getWarnings() {
        return warnings;
    }

    public void setWarnings(List<String> warnings) {
        if (warnings == null) {
            this.warnings = Collections.emptyList();
        } else {
            this.warnings = Collections.unmodifiableList(new ArrayList<>(warnings));
        }
    }
}
