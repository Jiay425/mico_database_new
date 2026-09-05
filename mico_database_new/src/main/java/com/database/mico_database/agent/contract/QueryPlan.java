package com.database.mico_database.agent.contract;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.ArrayList;
import java.util.List;

/**
 * Semantic query plan emitted by the Action Materializer.
 *
 * The plan contains stable catalog IDs only. It deliberately has no SQL,
 * physical table names, join keys, or join predicates; those belong to the
 * Java catalog/compiler boundary.
 */
@JsonIgnoreProperties(ignoreUnknown = false)
public final class QueryPlan {

    @JsonProperty("schemaVersion")
    private String schemaVersion = "query-plan-v1";
    @JsonProperty("root_entity")
    private String rootEntity;
    @JsonProperty("relation_path")
    private List<String> relationPath = new ArrayList<>();
    @JsonProperty("select_fields")
    private List<String> selectFields = new ArrayList<>();
    @JsonProperty("aggregations")
    private List<QueryAggregation> aggregations = new ArrayList<>();
    @JsonProperty("filters")
    private List<QueryFilter> filters = new ArrayList<>();
    @JsonProperty("group_by")
    private List<String> groupBy = new ArrayList<>();
    @JsonProperty("limit")
    private Integer limit;
    /** Runtime-owned technical cap; never selected by the scientific Policy. */
    @JsonProperty("sample_limit_per_group")
    private Integer sampleLimitPerGroup;
    /** Root semantic dimension used by the Runtime-owned sample cap. */
    @JsonProperty("sample_limit_group_field")
    private String sampleLimitGroupField;

    public String getSchemaVersion() { return schemaVersion; }
    public void setSchemaVersion(String schemaVersion) { this.schemaVersion = schemaVersion; }
    public String getRootEntity() { return rootEntity; }
    public void setRootEntity(String rootEntity) { this.rootEntity = rootEntity; }
    public List<String> getRelationPath() { return relationPath; }
    public void setRelationPath(List<String> relationPath) { this.relationPath = relationPath; }
    public List<String> getSelectFields() { return selectFields; }
    public void setSelectFields(List<String> selectFields) { this.selectFields = selectFields; }
    public List<QueryAggregation> getAggregations() { return aggregations; }
    public void setAggregations(List<QueryAggregation> aggregations) { this.aggregations = aggregations; }
    public List<QueryFilter> getFilters() { return filters; }
    public void setFilters(List<QueryFilter> filters) { this.filters = filters; }
    public List<String> getGroupBy() { return groupBy; }
    public void setGroupBy(List<String> groupBy) { this.groupBy = groupBy; }
    public Integer getLimit() { return limit; }
    public void setLimit(Integer limit) { this.limit = limit; }
    public Integer getSampleLimitPerGroup() { return sampleLimitPerGroup; }
    public void setSampleLimitPerGroup(Integer sampleLimitPerGroup) {
        this.sampleLimitPerGroup = sampleLimitPerGroup;
    }
    public String getSampleLimitGroupField() { return sampleLimitGroupField; }
    public void setSampleLimitGroupField(String sampleLimitGroupField) {
        this.sampleLimitGroupField = sampleLimitGroupField;
    }
}
