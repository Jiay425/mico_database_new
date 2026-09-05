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
    private List<String> internalAnalysisFields;
    private List<String> queryRules;

    public static SchemaSemanticCatalogReadModel current() {
        SchemaSemanticCatalogReadModel model = new SchemaSemanticCatalogReadModel();
        model.schemaVersion = SCHEMA_VERSION;
        model.source = SOURCE;
        model.generatedAt = Instant.now();
        model.entities = Arrays.asList(
                entity("sample", "patient_record", "patients", "patient_id",
                        field("patient_id", "integer", false, true, false, false, true,
                                "Internal Mico record key; not a Subject ID"),
                        field("disease", "string", true, true, true, false, false,
                                "Raw disease label; normalization is not implied"),
                        field("age", "integer", true, true, true, true, false,
                                "Age field when present"),
                        field("gender", "string", true, true, true, true, false,
                                "Gender field when present"),
                        field("country", "string", true, true, true, true, false,
                                "Country or region field when present"),
                        field("body_site", "string", true, true, false, true, false,
                                "Body-site field when present")),
                entity("metadata", "sample_metadata", "meta2db_sample_metadata", null,
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
                entity("abundance", "standard_abundance", "microbe_abundance_standard", null,
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
                entity("disease", "disease_dictionary", "diseases", "disease_id",
                        field("disease_id", "integer", false, true, false, false, true,
                                "Business disease dictionary key"),
                        field("disease_name", "string", true, true, true, false, false,
                                "Disease dictionary display name")),
                entity("sample_disease", "record_disease_link", "patient_diseases", null,
                        field("patient_id", "integer", false, true, false, false, true,
                                "Internal record association"),
                        field("disease_id", "integer", false, true, false, false, false,
                                "Disease dictionary association")));
        model.joins = Arrays.asList(
                join("sample_to_metadata", "patient_record", "patient_id", "sample_metadata", "patient_id",
                        "Internal record association; not Subject identity", "one_to_many"),
                join("sample_to_abundance", "patient_record", "patient_id", "standard_abundance", "patient_id",
                        "Internal record association for stored abundance", "one_to_many"),
                join("sample_to_disease", "patient_record", "patient_id", "record_disease_link", "patient_id",
                        "Internal record association", "one_to_many"),
                join("disease_to_sample_link", "disease_dictionary", "disease_id", "record_disease_link", "disease_id",
                        "Business disease dictionary association", "one_to_many"));
        // This is a Runtime capability, not a selectable business field. Java
        // appends the corresponding opaque token only when requested by the
        // internal analysis boundary; raw sample_id/patient_id remain hidden.
        model.internalAnalysisFields = Collections.singletonList("analysis.sample_key");
        model.queryRules = Arrays.asList(
                "select_or_with_only",
                "explicit_columns_only",
                "no_cross_database_reference",
                "bounded_limit_required",
                "java_final_validation");
        return model;
    }

    private static Entity entity(String entityId, String name, String table, String primaryKey, Field... fields) {
        Entity entity = new Entity();
        entity.entityId = entityId;
        entity.entityName = name;
        entity.sourceTable = table;
        entity.fields = Arrays.asList(fields);
        for (Field field : entity.fields) {
            field.fieldId = entityId + "." + semanticFieldId(entityId, field.name);
            field.scientificCapabilities = scientificCapabilities(entityId, field.name);
        }
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

    private static Join join(String relationId, String leftEntity, String leftField,
                             String rightEntity, String rightField, String description,
                             String cardinality) {
        Join join = new Join();
        join.relationId = relationId;
        join.leftEntity = leftEntity;
        join.leftField = leftField;
        join.rightEntity = rightEntity;
        join.rightField = rightField;
        join.relationshipStatus = "verified";
        join.cardinality = cardinality;
        join.description = description;
        return join;
    }

    private static String semanticFieldId(String entityId, String physicalField) {
        if ("sample".equals(entityId)) {
            return physicalField;
        }
        if ("metadata".equals(entityId)) {
            return "project_name".equals(physicalField) ? "project" : physicalField;
        }
        if ("abundance".equals(entityId)) {
            if ("abundance_value".equals(physicalField)) return "value";
            if ("microbe_name_standard".equals(physicalField)) return "feature";
        }
        if ("disease".equals(entityId) && "disease_name".equals(physicalField)) {
            return "name";
        }
        return physicalField;
    }

    /**
     * Scientific meaning is intentionally separate from SQL capabilities.
     * In particular, an aggregatable numeric field is not automatically an
     * outcome: age is aggregatable for profiling but remains a covariate.
     */
    private static List<String> scientificCapabilities(String entityId, String physicalField) {
        if ("sample".equals(entityId)) {
            if ("disease".equals(physicalField)) {
                return Arrays.asList("dimension", "stratifier");
            }
            if ("age".equals(physicalField)) {
                return Arrays.asList("covariate", "stratifier");
            }
            if ("gender".equals(physicalField) || "country".equals(physicalField)) {
                return Arrays.asList("covariate", "dimension", "stratifier");
            }
            return Collections.emptyList();
        }
        if ("metadata".equals(entityId) && "project_name".equals(physicalField)) {
            return Collections.singletonList("dimension");
        }
        if ("abundance".equals(entityId)) {
            if ("abundance_value".equals(physicalField)) {
                return Collections.singletonList("outcome");
            }
            if ("microbe_name_standard".equals(physicalField)
                    || "feature_version".equals(physicalField)
                    || "source_batch".equals(physicalField)) {
                return Arrays.asList("dimension", "stratifier");
            }
            return Collections.emptyList();
        }
        if ("disease".equals(entityId) && "disease_name".equals(physicalField)) {
            return Arrays.asList("dimension", "stratifier");
        }
        return Collections.emptyList();
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
    public List<String> getInternalAnalysisFields() {
        return internalAnalysisFields == null ? Collections.emptyList() : internalAnalysisFields;
    }
    public List<String> getQueryRules() { return queryRules == null ? Collections.emptyList() : queryRules; }

    public static final class Entity {
        private String entityId;
        private String entityName;
        private String sourceTable;
        private List<Field> fields = new ArrayList<>();
        private List<String> primaryKeyFields = new ArrayList<>();
        public String getEntityId() { return entityId; }
        public String getEntityName() { return entityName; }
        public String getSourceTable() { return sourceTable; }
        public List<Field> getFields() { return fields; }
        public List<String> getPrimaryKeyFields() { return primaryKeyFields; }
    }

    public static final class Field {
        private String fieldId;
        private String name;
        private String dataType;
        private boolean nullable;
        private String semanticStatus;
        private boolean filterable;
        private boolean groupable;
        private boolean aggregatable;
        private boolean displayable;
        private boolean sensitive;
        private List<String> scientificCapabilities = new ArrayList<>();
        private String description;
        public String getFieldId() { return fieldId; }
        public String getName() { return name; }
        public String getDataType() { return dataType; }
        public boolean isNullable() { return nullable; }
        public String getSemanticStatus() { return semanticStatus; }
        public boolean isFilterable() { return filterable; }
        public boolean isGroupable() { return groupable; }
        public boolean isAggregatable() { return aggregatable; }
        public boolean isDisplayable() { return displayable; }
        public boolean isSensitive() { return sensitive; }
        public List<String> getScientificCapabilities() {
            return scientificCapabilities == null
                    ? Collections.emptyList()
                    : scientificCapabilities;
        }
        public String getDescription() { return description; }
    }

    public static final class Join {
        private String relationId;
        private String leftEntity;
        private String leftField;
        private String rightEntity;
        private String rightField;
        private String relationshipStatus;
        private String cardinality;
        private String description;
        public String getRelationId() { return relationId; }
        public String getLeftEntity() { return leftEntity; }
        public String getLeftField() { return leftField; }
        public String getRightEntity() { return rightEntity; }
        public String getRightField() { return rightField; }
        public String getRelationshipStatus() { return relationshipStatus; }
        public String getCardinality() { return cardinality; }
        public String getDescription() { return description; }
    }
}
