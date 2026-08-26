package com.database.mico_database.agent.readmodel;

import com.fasterxml.jackson.annotation.JsonIgnore;

import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Versioned metadata-only description of the Java read surface.
 *
 * This object deliberately contains no sample values, disease labels,
 * locators, payload rows, credentials or database URL. It tells a planner
 * which semantic fields may be proposed; Java still validates every query.
 */
public final class SchemaSemanticCatalogReadModel {

    public static final String SOURCE = "java_schema_contract";
    public static final String SCHEMA_VERSION = "schema-catalog-v1";

    private String schemaVersion;
    private String source;
    private Instant generatedAt;
    private List<Entity> entities;
    private List<Join> joins;
    private List<String> queryRules;

    public static SchemaSemanticCatalogReadModel current() {
        SchemaSemanticCatalogReadModel model = new SchemaSemanticCatalogReadModel();
        model.schemaVersion = SCHEMA_VERSION;
        model.source = SOURCE;
        model.generatedAt = Instant.now();
        model.entities = Arrays.asList(
                entity("patient_record", "patients", "patient_id",
                        field("patient_id", "integer", false, true, false, false, true,
                                "Internal Mico record key; not a Subject ID"),
                        field("disease", "string", true, true, false, false, false,
                                "Raw disease label; normalization is not implied"),
                        field("age", "integer", true, true, true, true, false,
                                "Age field when present"),
                        field("gender", "string", true, true, true, true, false,
                                "Gender field when present"),
                        field("country", "string", true, true, true, true, false,
                                "Country or region field when present"),
                        field("body_site", "string", true, true, false, true, false,
                                "Body-site field when present")),
                entity("sample_metadata", "meta2db_sample_metadata", null,
                        field("patient_id", "integer", false, true, false, false, true,
                                "Internal record association"),
                        field("sample_id", "string", false, true, false, false, true,
                                "Source sample namespace; not a global Subject ID"),
                        field("project_name", "string", true, true, true, false, false,
                                "Study or project name when available"),
                        field("profile_sample", "string", true, false, false, false, false,
                                "Profile sample metadata when available"),
                        field("raw_metadata", "json", true, false, false, false, true,
                                "Raw metadata; payload values are not returned by this catalog")),
                entity("standard_abundance", "microbe_abundance_standard", null,
                        field("patient_id", "integer", false, true, false, false, true,
                                "Internal record association"),
                        field("sample_id", "string", false, true, false, false, true,
                                "Source sample namespace"),
                        field("microbe_name_standard", "string", true, false, true, false, false,
                                "Standardized feature name"),
                        field("abundance_value", "number", true, false, false, true, false,
                                "Stored abundance value"),
                        field("feature_version", "string", true, true, true, false, false,
                                "Feature definition version"),
                        field("source_batch", "string", true, true, true, false, false,
                                "Source/import batch token")),
                entity("disease_dictionary", "diseases", "disease_id",
                        field("disease_id", "integer", false, true, false, false, true,
                                "Business disease dictionary key"),
                        field("disease_name", "string", true, true, true, false, false,
                                "Disease dictionary display name")),
                entity("record_disease_link", "patient_diseases", null,
                        field("patient_id", "integer", false, true, false, false, true,
                                "Internal record association"),
                        field("disease_id", "integer", false, true, false, false, false,
                                "Disease dictionary association")));
        model.joins = Arrays.asList(
                join("patient_record", "patient_id", "sample_metadata", "patient_id",
                        "Internal record association; not Subject identity"),
                join("patient_record", "patient_id", "standard_abundance", "patient_id",
                        "Internal record association for stored abundance"),
                join("patient_record", "patient_id", "record_disease_link", "patient_id",
                        "Internal record association"),
                join("disease_dictionary", "disease_id", "record_disease_link", "disease_id",
                        "Business disease dictionary association"));
        model.queryRules = Arrays.asList(
                "select_or_with_only",
                "explicit_columns_only",
                "no_cross_database_reference",
                "bounded_limit_required",
                "java_final_validation");
        return model;
    }

    private static Entity entity(String name, String table, String primaryKey, Field... fields) {
        Entity entity = new Entity();
        entity.entityName = name;
        entity.sourceTable = table;
        entity.fields = Arrays.asList(fields);
        entity.primaryKeyFields = primaryKey == null
                ? Collections.<String>emptyList()
                : Collections.singletonList(primaryKey);
        return entity;
    }

    private static Field field(String name, String dataType, boolean nullable,
                               boolean filterable, boolean groupable, boolean aggregatable,
                               boolean sensitive, String description) {
        Field field = new Field();
        field.name = name;
        field.dataType = dataType;
        field.nullable = nullable;
        field.semanticStatus = "verified";
        field.filterable = filterable;
        field.groupable = groupable;
        field.aggregatable = aggregatable;
        field.displayable = !sensitive;
        field.sensitive = sensitive;
        field.description = description;
        return field;
    }

    private static Join join(String leftEntity, String leftField,
                             String rightEntity, String rightField, String description) {
        Join join = new Join();
        join.leftEntity = leftEntity;
        join.leftField = leftField;
        join.rightEntity = rightEntity;
        join.rightField = rightField;
        join.relationshipStatus = "verified";
        join.description = description;
        return join;
    }

    @JsonIgnore
    public int getEntityCount() {
        return entities == null ? 0 : entities.size();
    }

    public String getSchemaVersion() { return schemaVersion; }
    public String getSource() { return source; }
    public Instant getGeneratedAt() { return generatedAt; }
    public List<Entity> getEntities() { return entities == null ? Collections.emptyList() : entities; }
    public List<Join> getJoins() { return joins == null ? Collections.emptyList() : joins; }
    public List<String> getQueryRules() { return queryRules == null ? Collections.emptyList() : queryRules; }

    public static final class Entity {
        private String entityName;
        private String sourceTable;
        private List<Field> fields = new ArrayList<>();
        private List<String> primaryKeyFields = new ArrayList<>();
        public String getEntityName() { return entityName; }
        public String getSourceTable() { return sourceTable; }
        public List<Field> getFields() { return fields; }
        public List<String> getPrimaryKeyFields() { return primaryKeyFields; }
    }

    public static final class Field {
        private String name;
        private String dataType;
        private boolean nullable;
        private String semanticStatus;
        private boolean filterable;
        private boolean groupable;
        private boolean aggregatable;
        private boolean displayable;
        private boolean sensitive;
        private String description;
        public String getName() { return name; }
        public String getDataType() { return dataType; }
        public boolean isNullable() { return nullable; }
        public String getSemanticStatus() { return semanticStatus; }
        public boolean isFilterable() { return filterable; }
        public boolean isGroupable() { return groupable; }
        public boolean isAggregatable() { return aggregatable; }
        public boolean isDisplayable() { return displayable; }
        public boolean isSensitive() { return sensitive; }
        public String getDescription() { return description; }
    }

    public static final class Join {
        private String leftEntity;
        private String leftField;
        private String rightEntity;
        private String rightField;
        private String relationshipStatus;
        private String description;
        public String getLeftEntity() { return leftEntity; }
        public String getLeftField() { return leftField; }
        public String getRightEntity() { return rightEntity; }
        public String getRightField() { return rightField; }
        public String getRelationshipStatus() { return relationshipStatus; }
        public String getDescription() { return description; }
    }
}
