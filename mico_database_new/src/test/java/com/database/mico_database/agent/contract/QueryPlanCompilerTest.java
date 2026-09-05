package com.database.mico_database.agent.contract;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Collections;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertThrows;

class QueryPlanCompilerTest {

    @Test
    void compilesOrderedCatalogRelationsAndBindsFilterAndLimit() {
        QueryPlan plan = new QueryPlan();
        plan.setRootEntity("sample");
        plan.setRelationPath(Arrays.asList("sample_to_abundance", "sample_to_metadata"));
        plan.setSelectFields(Collections.singletonList("metadata.project"));
        QueryAggregation aggregation = new QueryAggregation();
        aggregation.setField("abundance.value");
        aggregation.setOp("mean");
        plan.setAggregations(Collections.singletonList(aggregation));
        QueryFilter filter = new QueryFilter();
        filter.setField("sample.disease");
        filter.setOperator("eq");
        filter.setValue("T2D");
        plan.setFilters(Collections.singletonList(filter));
        plan.setGroupBy(Collections.singletonList("metadata.project"));
        plan.setLimit(100);

        QueryPlanCompiler.CompiledQuery compiled = new QueryPlanCompiler().compile(plan);

        assertTrue(compiled.getSql().contains("JOIN microbe_abundance_standard"));
        assertTrue(compiled.getSql().contains("JOIN meta2db_sample_metadata"));
        assertTrue(compiled.getSql().contains("AVG("));
        assertTrue(compiled.getSql().contains(" = ?"));
        assertTrue(compiled.getSql().endsWith("LIMIT ?"));
        assertEquals(2, compiled.getParameters().size());
        assertEquals("T2D", compiled.getParameters().get(0).getValue());
        assertEquals(100, compiled.getParameters().get(1).getValue());
    }

    @Test
    void rejectsUnknownRelationAndPhysicalField() {
        QueryPlan unknownRelation = new QueryPlan();
        unknownRelation.setRootEntity("sample");
        unknownRelation.setRelationPath(Collections.singletonList("sample_to_unknown"));
        unknownRelation.setSelectFields(Collections.singletonList("sample.disease"));
        unknownRelation.setLimit(10);
        assertThrows(IllegalArgumentException.class,
                () -> new QueryPlanCompiler().compile(unknownRelation));

        QueryPlan physicalField = new QueryPlan();
        physicalField.setRootEntity("sample");
        physicalField.setSelectFields(Collections.singletonList("patients.disease"));
        physicalField.setLimit(10);
        assertThrows(IllegalArgumentException.class,
                () -> new QueryPlanCompiler().compile(physicalField));
    }

    @Test
    void rejectsIncompletePlanItemsBeforeCatalogLookup() {
        QueryPlan blankRelation = new QueryPlan();
        blankRelation.setRootEntity("sample");
        blankRelation.setRelationPath(Arrays.asList((String) null));
        blankRelation.setSelectFields(Collections.singletonList("sample.disease"));
        blankRelation.setLimit(10);
        assertThrows(IllegalArgumentException.class,
                () -> new QueryPlanCompiler().compile(blankRelation));

        QueryPlan nullAggregation = new QueryPlan();
        nullAggregation.setRootEntity("sample");
        nullAggregation.setSelectFields(Collections.singletonList("sample.disease"));
        nullAggregation.setAggregations(Arrays.asList((QueryAggregation) null));
        nullAggregation.setLimit(10);
        assertThrows(IllegalArgumentException.class,
                () -> new QueryPlanCompiler().compile(nullAggregation));

        QueryPlan nullFilter = new QueryPlan();
        nullFilter.setRootEntity("sample");
        nullFilter.setSelectFields(Collections.singletonList("sample.disease"));
        nullFilter.setFilters(Arrays.asList((QueryFilter) null));
        nullFilter.setLimit(10);
        assertThrows(IllegalArgumentException.class,
                () -> new QueryPlanCompiler().compile(nullFilter));
    }

    @Test
    void appendsOnlyAnOpaqueSampleSourceForRawAbundanceProjection() {
        QueryPlan plan = new QueryPlan();
        plan.setRootEntity("sample");
        plan.setRelationPath(Collections.singletonList("sample_to_abundance"));
        plan.setSelectFields(Arrays.asList(
                "sample.disease", "abundance.feature", "abundance.value"));
        plan.setLimit(20);

        QueryPlanCompiler.CompiledQuery compiled = new QueryPlanCompiler().compile(plan, true);

        assertTrue(compiled.includesOpaqueSampleKey());
        assertTrue(compiled.getSql().contains("a_abundance.sample_id AS "
                + QueryPlanCompiler.OPAQUE_SAMPLE_KEY_SOURCE_ALIAS));
        assertTrue(compiled.getSql().contains("LIMIT ?"));
    }

    @Test
    void rejectsOpaqueSampleKeyForGroupedResults() {
        QueryPlan plan = new QueryPlan();
        plan.setRootEntity("sample");
        plan.setRelationPath(Collections.singletonList("sample_to_abundance"));
        plan.setSelectFields(Collections.singletonList("sample.disease"));
        QueryAggregation aggregation = new QueryAggregation();
        aggregation.setField("abundance.value");
        aggregation.setOp("mean");
        plan.setAggregations(Collections.singletonList(aggregation));
        plan.setGroupBy(Collections.singletonList("sample.disease"));
        plan.setLimit(20);

        assertThrows(IllegalArgumentException.class,
                () -> new QueryPlanCompiler().compile(plan, true));
    }

    @Test
    void compilesRuntimeOwnedSampleBoundBeforeAbundanceFanout() {
        QueryPlan plan = new QueryPlan();
        plan.setRootEntity("sample");
        plan.setRelationPath(Collections.singletonList("sample_to_abundance"));
        plan.setSelectFields(Arrays.asList(
                "sample.disease", "sample.age", "abundance.feature", "abundance.value"));
        QueryFilter filter = new QueryFilter();
        filter.setField("sample.disease");
        filter.setOperator("in");
        filter.setValue(Arrays.asList("T2D", "healthy"));
        plan.setFilters(Collections.singletonList(filter));
        plan.setLimit(20_000);
        plan.setSampleLimitPerGroup(50);
        plan.setSampleLimitGroupField("sample.disease");

        QueryPlanCompiler.CompiledQuery compiled = new QueryPlanCompiler().compile(plan, true);

        assertTrue(compiled.getSql().contains("ROW_NUMBER() OVER (PARTITION BY"));
        assertTrue(compiled.getSql().contains("__mico_sample_rank <= ?"));
        assertTrue(compiled.getSql().contains("IN (SELECT bounded.__mico_sample_pk"));
        assertEquals(6, compiled.getParameters().size());
        assertEquals("T2D", compiled.getParameters().get(0).getValue());
        assertEquals("healthy", compiled.getParameters().get(1).getValue());
        assertEquals("T2D", compiled.getParameters().get(2).getValue());
        assertEquals("healthy", compiled.getParameters().get(3).getValue());
        assertEquals(50, compiled.getParameters().get(4).getValue());
        assertEquals(20_000, compiled.getParameters().get(5).getValue());
        assertTrue(compiled.getSql().endsWith("LIMIT ?"));
    }

    @Test
    void rejectsSampleBoundWhenGroupDimensionIsNotRoot() {
        QueryPlan plan = new QueryPlan();
        plan.setRootEntity("sample");
        plan.setRelationPath(Arrays.asList("sample_to_abundance", "sample_to_metadata"));
        plan.setSelectFields(Arrays.asList(
                "metadata.project", "abundance.feature", "abundance.value"));
        plan.setLimit(100);
        plan.setSampleLimitPerGroup(10);
        plan.setSampleLimitGroupField("metadata.project");

        assertThrows(IllegalArgumentException.class,
                () -> new QueryPlanCompiler().compile(plan, true));
    }

    @Test
    void compilesSampleBoundForBroadProjectionWithoutRootFilters() {
        QueryPlan plan = new QueryPlan();
        plan.setRootEntity("sample");
        plan.setRelationPath(Collections.singletonList("sample_to_abundance"));
        plan.setSelectFields(Arrays.asList(
                "sample.disease", "abundance.feature", "abundance.value"));
        plan.setLimit(20_000);
        plan.setSampleLimitPerGroup(50);
        plan.setSampleLimitGroupField("sample.disease");

        QueryPlanCompiler.CompiledQuery compiled = new QueryPlanCompiler().compile(plan, true);

        assertTrue(compiled.getSql().contains(" WHERE a_sample.patient_id IN "));
        assertEquals(2, compiled.getParameters().size());
        assertEquals(50, compiled.getParameters().get(0).getValue());
        assertEquals(20_000, compiled.getParameters().get(1).getValue());
    }
}
