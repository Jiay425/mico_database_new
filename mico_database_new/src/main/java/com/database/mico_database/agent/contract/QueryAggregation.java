package com.database.mico_database.agent.contract;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

/** One closed aggregation in a QueryPlan. */
@JsonIgnoreProperties(ignoreUnknown = false)
public final class QueryAggregation {

    @JsonProperty("field")
    private String field;
    @JsonProperty("op")
    private String op;

    public String getField() { return field; }
    public void setField(String field) { this.field = field; }
    public String getOp() { return op; }
    public void setOp(String op) { this.op = op; }
}
