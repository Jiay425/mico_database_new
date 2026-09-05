package com.database.mico_database.agent.contract;

import com.database.mico_database.agent.readmodel.SchemaSemanticCatalogReadModel;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.StringJoiner;

/**
 * Compiles a semantic QueryPlan into parameterized SQL using the Java-owned
 * catalog. The model cannot provide physical identifiers, join predicates, or
 * SQL fragments.
 */
public final class QueryPlanCompiler {

    private static final int MAX_HOPS = 2;
    /** Typed plans may return a bounded sample x feature projection. */
    private static final int MAX_LIMIT = 20_000;
    /** Runtime-owned technical cap; a model cannot raise this value. */
    public static final int MAX_SAMPLE_LIMIT_PER_GROUP = 50;
    /** Internal result alias; the raw sample identifier is never returned. */
    public static final String OPAQUE_SAMPLE_KEY_SOURCE_ALIAS = "__mico_analysis_sample_source";
    public static final String OPAQUE_SAMPLE_KEY_RESULT_ALIAS = "a_analysis_sample_key";

    private final SchemaSemanticCatalogReadModel catalog;
    private final Map<String, SchemaSemanticCatalogReadModel.Entity> entitiesById = new HashMap<>();
    private final Map<String, SchemaSemanticCatalogReadModel.Entity> entitiesByName = new HashMap<>();
    private final Map<String, SchemaSemanticCatalogReadModel.Join> joinsById = new HashMap<>();

    public QueryPlanCompiler() {
        this(SchemaSemanticCatalogReadModel.current());
    }

    public QueryPlanCompiler(SchemaSemanticCatalogReadModel catalog) {
        if (catalog == null) {
            throw new IllegalArgumentException("catalog is required");
        }
        this.catalog = catalog;
        for (SchemaSemanticCatalogReadModel.Entity entity : catalog.getEntities()) {
            if (entity.getEntityId() == null || entity.getEntityId().trim().isEmpty()) {
                throw new IllegalArgumentException("catalog entity semantic ID is missing");
            }
            entitiesById.put(entity.getEntityId(), entity);
            entitiesByName.put(entity.getEntityName(), entity);
        }
        for (SchemaSemanticCatalogReadModel.Join join : catalog.getJoins()) {
            if (join.getRelationId() == null || join.getRelationId().trim().isEmpty()) {
                throw new IllegalArgumentException("catalog relation semantic ID is missing");
            }
            joinsById.put(join.getRelationId(), join);
        }
    }

    public CompiledQuery compile(QueryPlan plan) {
        return compile(plan, false);
    }

    /**
     * Compile a normal semantic plan, optionally appending the Runtime-owned
     * sample identity source needed for sample-level abundance analysis.
     *
     * The flag is deliberately outside QueryPlan: it is technical execution
     * metadata, not a model-selected scientific field.  Java still refuses it
     * for grouped/aggregated plans because one opaque key cannot represent a
     * grouped result row.
     */
    public CompiledQuery compile(QueryPlan plan, boolean includeOpaqueSampleKey) {
        validatePlanShape(plan);
        if (includeOpaqueSampleKey && (!plan.getAggregations().isEmpty()
                || !plan.getGroupBy().isEmpty())) {
            throw new IllegalArgumentException("opaque sample key requires a raw, ungrouped projection");
        }
        SchemaSemanticCatalogReadModel.Entity root = requiredEntity(plan.getRootEntity());
        Map<String, SchemaSemanticCatalogReadModel.Entity> active = new HashMap<>();
        active.put(root.getEntityName(), root);
        List<JoinBinding> selectedJoins = new ArrayList<>();
        Set<String> seenRelations = new HashSet<>();
        for (String relationId : plan.getRelationPath()) {
            if (relationId == null || relationId.trim().isEmpty()) {
                throw new IllegalArgumentException("query plan relation path contains a blank relation");
            }
            if (!seenRelations.add(relationId)) {
                throw new IllegalArgumentException("query plan relation path contains a duplicate");
            }
            SchemaSemanticCatalogReadModel.Join join = joinsById.get(relationId);
            if (join == null || !"verified".equals(join.getRelationshipStatus())) {
                throw new IllegalArgumentException("query plan relation is not enabled");
            }
            SchemaSemanticCatalogReadModel.Entity left = entitiesByName.get(join.getLeftEntity());
            SchemaSemanticCatalogReadModel.Entity right = entitiesByName.get(join.getRightEntity());
            if (left == null || right == null) {
                throw new IllegalArgumentException("catalog relation endpoint is unknown");
            }
            boolean leftActive = active.containsKey(left.getEntityName());
            boolean rightActive = active.containsKey(right.getEntityName());
            if (leftActive == rightActive) {
                throw new IllegalArgumentException("query plan relation path is ambiguous or disconnected");
            }
            if ("many_to_many".equals(join.getCardinality())
                    || "unknown".equals(join.getCardinality())) {
                throw new IllegalArgumentException("query plan relation cardinality is not safe");
            }
            SchemaSemanticCatalogReadModel.Entity source = leftActive ? left : right;
            SchemaSemanticCatalogReadModel.Entity target = leftActive ? right : left;
            active.put(target.getEntityName(), target);
            selectedJoins.add(new JoinBinding(join, source, target));
        }

        List<FieldRef> selectedFields = new ArrayList<>();
        for (String fieldId : plan.getSelectFields()) {
            FieldRef ref = field(fieldId, active);
            if (!ref.field.isDisplayable() || ref.field.isSensitive()) {
                throw new IllegalArgumentException("query plan select field is not displayable");
            }
            selectedFields.add(ref);
        }
        List<AggregationRef> aggregations = new ArrayList<>();
        for (QueryAggregation aggregation : plan.getAggregations()) {
            if (aggregation == null) {
                throw new IllegalArgumentException("query plan aggregation is missing");
            }
            FieldRef ref = field(aggregation.getField(), active);
            String op = aggregation.getOp();
            if (!ref.field.isAggregatable() || ref.field.isSensitive()
                    || !isAggregation(op)) {
                throw new IllegalArgumentException("query plan aggregation is not allowed");
            }
            aggregations.add(new AggregationRef(ref, op));
        }
        for (String groupField : plan.getGroupBy()) {
            if (groupField == null || groupField.trim().isEmpty()) {
                throw new IllegalArgumentException("query plan group field is missing");
            }
            FieldRef ref = field(groupField, active);
            if (!ref.field.isGroupable() || !containsField(plan.getSelectFields(), groupField)) {
                throw new IllegalArgumentException("query plan group field is not allowed");
            }
        }

        FieldRef sampleLimitGroup = null;
        FieldRef rootPrimaryKey = null;
        if (plan.getSampleLimitPerGroup() != null) {
            if (!plan.getAggregations().isEmpty() || !plan.getGroupBy().isEmpty()) {
                throw new IllegalArgumentException("sample limit requires a raw, ungrouped projection");
            }
            if (plan.getSampleLimitPerGroup() < 1
                    || plan.getSampleLimitPerGroup() > MAX_SAMPLE_LIMIT_PER_GROUP
                    || plan.getSampleLimitGroupField() == null) {
                throw new IllegalArgumentException("sample limit per group is outside the approved range");
            }
            sampleLimitGroup = field(plan.getSampleLimitGroupField(), active);
            if (!sampleLimitGroup.entity.getEntityName().equals(root.getEntityName())
                    || !sampleLimitGroup.field.isGroupable()
                    || sampleLimitGroup.field.isSensitive()
                    || !containsField(plan.getSelectFields(), plan.getSampleLimitGroupField())) {
                throw new IllegalArgumentException("sample limit group field must be a selected root dimension");
            }
            if (plan.getFilters().stream().anyMatch(filter -> filter != null
                    && !filter.getField().startsWith(root.getEntityId() + "."))) {
                throw new IllegalArgumentException("sample-bounded reads allow only root filters");
            }
            List<String> primaryKeys = root.getPrimaryKeyFields();
            if (primaryKeys == null || primaryKeys.size() != 1) {
                throw new IllegalArgumentException("sample-bounded root requires one primary key");
            }
            String primaryKeyName = primaryKeys.get(0);
            for (SchemaSemanticCatalogReadModel.Field candidate : root.getFields()) {
                if (primaryKeyName.equals(candidate.getName())) {
                    rootPrimaryKey = new FieldRef(root, candidate,
                            root.getEntityId() + "." + primaryKeyName);
                    break;
                }
            }
            if (rootPrimaryKey == null || rootPrimaryKey.field.isSensitive() == false) {
                throw new IllegalArgumentException("sample-bounded root primary key is unavailable");
            }
        } else if (plan.getSampleLimitGroupField() != null) {
            throw new IllegalArgumentException("sample limit group field requires a sample limit");
        }

        FieldRef abundanceSampleIdentity = null;
        if (includeOpaqueSampleKey) {
            SchemaSemanticCatalogReadModel.Entity abundance = active.get("standard_abundance");
            if (abundance == null) {
                throw new IllegalArgumentException("opaque sample key requires the abundance relation");
            }
            for (SchemaSemanticCatalogReadModel.Field candidate : abundance.getFields()) {
                if ("sample_id".equals(candidate.getName()) && candidate.isSensitive()) {
                    abundanceSampleIdentity = new FieldRef(abundance, candidate,
                            "abundance.sample_id");
                    break;
                }
            }
            if (abundanceSampleIdentity == null) {
                throw new IllegalArgumentException("opaque sample key source is unavailable");
            }
        }

        List<QueryParameter> parameters = new ArrayList<>();
        StringBuilder sql = new StringBuilder("SELECT ");
        StringJoiner projection = new StringJoiner(", ");
        for (FieldRef ref : selectedFields) {
            projection.add(ref.qualifiedName() + " AS " + alias(ref.fieldId));
        }
        for (AggregationRef aggregation : aggregations) {
            projection.add(aggregation.expression() + " AS " + alias(aggregation.ref.fieldId + "_" + aggregation.op));
        }
        if (abundanceSampleIdentity != null) {
            projection.add(abundanceSampleIdentity.qualifiedName()
                    + " AS " + OPAQUE_SAMPLE_KEY_SOURCE_ALIAS);
        }
        sql.append(projection).append(" FROM ")
                .append(root.getSourceTable()).append(" ").append(alias(root.getEntityId()));
        for (JoinBinding binding : selectedJoins) {
            SchemaSemanticCatalogReadModel.Join join = binding.join;
            sql.append(" JOIN ").append(binding.target.getSourceTable()).append(" ")
                    .append(alias(binding.target.getEntityId())).append(" ON ")
                    .append(alias(entityByName(join.getLeftEntity()).getEntityId())).append(".")
                    .append(join.getLeftField()).append(" = ")
                    .append(alias(entityByName(join.getRightEntity()).getEntityId())).append(".")
                    .append(join.getRightField());
        }
        appendFilters(sql, plan.getFilters(), active, parameters);
        if (sampleLimitGroup != null && rootPrimaryKey != null) {
            appendSampleLimitPredicate(sql, plan, root, rootPrimaryKey, sampleLimitGroup, parameters);
        }
        if (!plan.getGroupBy().isEmpty()) {
            sql.append(" GROUP BY ");
            StringJoiner grouping = new StringJoiner(", ");
            for (String groupField : plan.getGroupBy()) {
                grouping.add(field(groupField, active).qualifiedName());
            }
            sql.append(grouping);
        }
        sql.append(" LIMIT ?");
        parameters.add(new QueryParameter(plan.getLimit()));
        return new CompiledQuery(sql.toString(), parameters, catalog.getSchemaVersion(),
                abundanceSampleIdentity != null);
    }

    /**
     * Restrict a raw one-to-many read to N unique root records per selected
     * dimension before the abundance join is materialized.  The subquery is
     * assembled entirely from catalog-owned identifiers; values remain JDBC
     * parameters and the outer row LIMIT still protects result size.
     */
    private void appendSampleLimitPredicate(
            StringBuilder sql,
            QueryPlan plan,
            SchemaSemanticCatalogReadModel.Entity root,
            FieldRef rootPrimaryKey,
            FieldRef sampleLimitGroup,
            List<QueryParameter> parameters) {
        String boundedAlias = "b_" + root.getEntityId();
        // ``appendFilters`` owns the outer WHERE clause when root filters are
        // present.  A broad sample-bounded projection has no filters, so it
        // needs to start that clause itself rather than producing ``FROM …
        // AND …``.  The inner bounded subquery gets its own WHERE below.
        sql.append(plan.getFilters() == null || plan.getFilters().isEmpty() ? " WHERE " : " AND ")
                .append(rootPrimaryKey.qualifiedName())
                .append(" IN (SELECT bounded.__mico_sample_pk FROM (SELECT ")
                .append(boundedAlias).append(".").append(rootPrimaryKey.field.getName())
                .append(" AS __mico_sample_pk, ROW_NUMBER() OVER (PARTITION BY ")
                .append(boundedAlias).append(".").append(sampleLimitGroup.field.getName())
                .append(" ORDER BY ").append(boundedAlias).append(".")
                .append(rootPrimaryKey.field.getName()).append(") AS __mico_sample_rank FROM ")
                .append(root.getSourceTable()).append(" ").append(boundedAlias);
        appendFiltersForAlias(sql, plan.getFilters(), root, boundedAlias, parameters);
        sql.append(") bounded WHERE bounded.__mico_sample_rank <= ?)");
        parameters.add(new QueryParameter(plan.getSampleLimitPerGroup()));
    }

    private void appendFiltersForAlias(
            StringBuilder sql,
            List<QueryFilter> filters,
            SchemaSemanticCatalogReadModel.Entity root,
            String alias,
            List<QueryParameter> parameters) {
        if (filters == null || filters.isEmpty()) return;
        sql.append(" WHERE ");
        StringJoiner clauses = new StringJoiner(" AND ");
        for (QueryFilter filter : filters) {
            if (filter == null || !filter.getField().startsWith(root.getEntityId() + ".")) {
                throw new IllegalArgumentException("sample-bounded reads allow only root filters");
            }
            FieldRef ref = field(filter.getField(),
                    Collections.singletonMap(root.getEntityName(), root));
            if (!ref.field.isFilterable() || ref.field.isSensitive()) {
                throw new IllegalArgumentException("query plan filter field is not allowed");
            }
            String qualified = alias + "." + ref.field.getName();
            String operator = filter.getOperator();
            if ("is_null".equals(operator) || "is_not_null".equals(operator)) {
                clauses.add(qualified + ("is_null".equals(operator) ? " IS NULL" : " IS NOT NULL"));
            } else if ("eq".equals(operator) || "neq".equals(operator)) {
                if (filter.getValue() == null) {
                    throw new IllegalArgumentException("query plan equality filter requires a value");
                }
                clauses.add(qualified + ("eq".equals(operator) ? " = ?" : " <> ?"));
                parameters.add(new QueryParameter(filter.getValue()));
            } else if ("in".equals(operator) && filter.getValue() instanceof List) {
                List<?> values = (List<?>) filter.getValue();
                if (values.isEmpty() || values.size() > 20) {
                    throw new IllegalArgumentException("query plan IN filter is out of bounds");
                }
                StringJoiner placeholders = new StringJoiner(", ", "(", ")");
                for (Object value : values) {
                    if (value == null) {
                        throw new IllegalArgumentException("query plan IN filter requires non-null values");
                    }
                    placeholders.add("?");
                    parameters.add(new QueryParameter(value));
                }
                clauses.add(qualified + " IN " + placeholders);
            } else {
                throw new IllegalArgumentException("query plan filter operator is not allowed");
            }
        }
        sql.append(clauses);
    }

    private void appendFilters(StringBuilder sql, List<QueryFilter> filters,
                               Map<String, SchemaSemanticCatalogReadModel.Entity> active,
                               List<QueryParameter> parameters) {
        if (filters == null || filters.isEmpty()) return;
        sql.append(" WHERE ");
        StringJoiner clauses = new StringJoiner(" AND ");
        for (QueryFilter filter : filters) {
            FieldRef ref = field(filter.getField(), active);
            if (!ref.field.isFilterable() || ref.field.isSensitive()) {
                throw new IllegalArgumentException("query plan filter field is not allowed");
            }
            String operator = filter.getOperator();
            if ("is_null".equals(operator) || "is_not_null".equals(operator)) {
                clauses.add(ref.qualifiedName() + ("is_null".equals(operator) ? " IS NULL" : " IS NOT NULL"));
            } else if ("eq".equals(operator) || "neq".equals(operator)) {
                if (filter.getValue() == null) {
                    throw new IllegalArgumentException("query plan equality filter requires a value");
                }
                clauses.add(ref.qualifiedName() + ("eq".equals(operator) ? " = ?" : " <> ?"));
                parameters.add(new QueryParameter(filter.getValue()));
            } else if ("in".equals(operator) && filter.getValue() instanceof List) {
                List<?> values = (List<?>) filter.getValue();
                if (values.isEmpty() || values.size() > 20) {
                    throw new IllegalArgumentException("query plan IN filter is out of bounds");
                }
                StringJoiner placeholders = new StringJoiner(", ", "(", ")");
                for (Object value : values) {
                    if (value == null) {
                        throw new IllegalArgumentException("query plan IN filter requires non-null values");
                    }
                    placeholders.add("?");
                    parameters.add(new QueryParameter(value));
                }
                clauses.add(ref.qualifiedName() + " IN " + placeholders);
            } else {
                throw new IllegalArgumentException("query plan filter operator is not allowed");
            }
        }
        sql.append(clauses);
    }

    private void validatePlanShape(QueryPlan plan) {
        if (plan == null || !"query-plan-v1".equals(plan.getSchemaVersion())
                || plan.getRootEntity() == null || plan.getSelectFields() == null
                || plan.getSelectFields().isEmpty() || plan.getSelectFields().size() > 16
                || plan.getRelationPath() == null || plan.getRelationPath().size() > MAX_HOPS
                || plan.getAggregations() == null || plan.getAggregations().size() > 8
                || plan.getFilters() == null || plan.getFilters().size() > 16
                || plan.getGroupBy() == null || plan.getGroupBy().size() > 8
                || plan.getLimit() == null || plan.getLimit() < 1 || plan.getLimit() > MAX_LIMIT
                || (plan.getSampleLimitPerGroup() != null
                    && (plan.getSampleLimitPerGroup() < 1
                        || plan.getSampleLimitPerGroup() > MAX_SAMPLE_LIMIT_PER_GROUP))) {
            throw new IllegalArgumentException("query plan shape is invalid");
        }
        if (new HashSet<>(plan.getSelectFields()).size() != plan.getSelectFields().size()
                || new HashSet<>(plan.getGroupBy()).size() != plan.getGroupBy().size()) {
            throw new IllegalArgumentException("query plan field list contains duplicates");
        }
        if (plan.getSelectFields().stream().anyMatch(value -> value == null || value.trim().isEmpty())
                || plan.getGroupBy().stream().anyMatch(value -> value == null || value.trim().isEmpty())
                || plan.getRelationPath().stream().anyMatch(value -> value == null || value.trim().isEmpty())) {
            throw new IllegalArgumentException("query plan contains a blank identifier");
        }
        if (plan.getAggregations().stream().anyMatch(value -> value == null
                || value.getField() == null || value.getField().trim().isEmpty()
                || value.getOp() == null || value.getOp().trim().isEmpty())) {
            throw new IllegalArgumentException("query plan contains an incomplete aggregation");
        }
        if (plan.getFilters().stream().anyMatch(value -> value == null
                || value.getField() == null || value.getField().trim().isEmpty()
                || value.getOperator() == null || value.getOperator().trim().isEmpty())) {
            throw new IllegalArgumentException("query plan contains an incomplete filter");
        }
        if ((!plan.getGroupBy().isEmpty() && plan.getAggregations().isEmpty())
                || !new HashSet<>(plan.getSelectFields()).containsAll(plan.getGroupBy())
                || (!plan.getAggregations().isEmpty()
                    && !new HashSet<>(plan.getSelectFields()).equals(new HashSet<>(plan.getGroupBy())))) {
            throw new IllegalArgumentException("query plan grouping shape is invalid");
        }
    }

    private FieldRef field(String fieldId, Map<String, SchemaSemanticCatalogReadModel.Entity> active) {
        if (fieldId == null) throw new IllegalArgumentException("query plan field is missing");
        int separator = fieldId.indexOf('.');
        if (separator <= 0 || separator == fieldId.length() - 1) {
            throw new IllegalArgumentException("query plan field ID is invalid");
        }
        String entityId = fieldId.substring(0, separator);
        SchemaSemanticCatalogReadModel.Entity entity = entitiesById.get(entityId);
        if (entity == null || !active.containsKey(entity.getEntityName())) {
            throw new IllegalArgumentException("query plan field entity is not in relation path");
        }
        String stableFieldId = fieldId;
        for (SchemaSemanticCatalogReadModel.Field candidate : entity.getFields()) {
            if (stableFieldId.equals(candidate.getFieldId())) {
                return new FieldRef(entity, candidate, stableFieldId);
            }
        }
        throw new IllegalArgumentException("query plan field is not in catalog");
    }

    private SchemaSemanticCatalogReadModel.Entity requiredEntity(String id) {
        SchemaSemanticCatalogReadModel.Entity entity = entitiesById.get(id);
        if (entity == null) throw new IllegalArgumentException("query plan root entity is not in catalog");
        return entity;
    }

    private SchemaSemanticCatalogReadModel.Entity entityByName(String name) {
        SchemaSemanticCatalogReadModel.Entity entity = entitiesByName.get(name);
        if (entity == null) throw new IllegalArgumentException("catalog entity is missing");
        return entity;
    }

    private static boolean containsField(List<String> fields, String target) {
        return fields != null && fields.contains(target);
    }

    private static boolean isAggregation(String op) {
        return "count".equals(op) || "mean".equals(op) || "min".equals(op)
                || "max".equals(op) || "sum".equals(op);
    }

    private static String alias(String value) {
        return "a_" + value.replace('.', '_');
    }

    public static final class CompiledQuery {
        private final String sql;
        private final List<QueryParameter> parameters;
        private final String catalogVersion;
        private final boolean includesOpaqueSampleKey;

        private CompiledQuery(String sql, List<QueryParameter> parameters, String catalogVersion,
                              boolean includesOpaqueSampleKey) {
            this.sql = sql;
            this.parameters = parameters;
            this.catalogVersion = catalogVersion;
            this.includesOpaqueSampleKey = includesOpaqueSampleKey;
        }

        public String getSql() { return sql; }
        public List<QueryParameter> getParameters() { return parameters; }
        public String getCatalogVersion() { return catalogVersion; }
        public boolean includesOpaqueSampleKey() { return includesOpaqueSampleKey; }
    }

    public static final class QueryParameter {
        private final Object value;
        private QueryParameter(Object value) { this.value = value; }
        public Object getValue() { return value; }
    }

    private static final class FieldRef {
        private final SchemaSemanticCatalogReadModel.Entity entity;
        private final SchemaSemanticCatalogReadModel.Field field;
        private final String fieldId;
        private FieldRef(SchemaSemanticCatalogReadModel.Entity entity,
                         SchemaSemanticCatalogReadModel.Field field, String fieldId) {
            this.entity = entity;
            this.field = field;
            this.fieldId = fieldId;
        }
        private String qualifiedName() {
            return alias(entity.getEntityId()) + "." + field.getName();
        }
    }

    private static final class AggregationRef {
        private final FieldRef ref;
        private final String op;
        private AggregationRef(FieldRef ref, String op) { this.ref = ref; this.op = op; }
        private String expression() {
            if ("count".equals(op)) return "COUNT(" + ref.qualifiedName() + ")";
            if ("mean".equals(op)) return "AVG(" + ref.qualifiedName() + ")";
            return op.toUpperCase() + "(" + ref.qualifiedName() + ")";
        }
    }

    private static final class JoinBinding {
        private final SchemaSemanticCatalogReadModel.Join join;
        private final SchemaSemanticCatalogReadModel.Entity source;
        private final SchemaSemanticCatalogReadModel.Entity target;

        private JoinBinding(SchemaSemanticCatalogReadModel.Join join,
                            SchemaSemanticCatalogReadModel.Entity source,
                            SchemaSemanticCatalogReadModel.Entity target) {
            this.join = join;
            this.source = source;
            this.target = target;
        }
    }
}
