package com.database.mico_database.agent.contract;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DynamicReadQueryPolicyTest {

    @Test
    void acceptsDynamicReadShapeWithoutAStaticTableAllowlist() {
        String sql = "SELECT p.disease, m.project_name, COUNT(*) AS sample_count "
                + "FROM patients p JOIN meta2db_sample_metadata m ON m.patient_id = p.patient_id "
                + "WHERE p.disease IS NOT NULL GROUP BY p.disease, m.project_name "
                + "ORDER BY sample_count DESC LIMIT 20 OFFSET 10";

        assertTrue(DynamicReadQueryPolicy.validate(sql).isValid());
    }

    @Test
    void acceptsUnionAndLimitCommaSyntaxButBoundsBothResultAndOffset() {
        assertTrue(DynamicReadQueryPolicy.validate(
                "SELECT disease FROM patients LIMIT 10,20").isValid());
        assertTrue(DynamicReadQueryPolicy.validate(
                "SELECT disease FROM patients LIMIT 20 OFFSET 1000000").isValid());
        assertFalse(DynamicReadQueryPolicy.validate(
                "SELECT disease FROM patients LIMIT 20 OFFSET 1000001").isValid());
        assertTrue(DynamicReadQueryPolicy.validate(
                "SELECT disease FROM patients UNION ALL SELECT disease FROM diseases LIMIT 20").isValid());
    }

    @Test
    void rejectsWritesAndCrossDatabaseOrAdministrativeReferences() {
        String[] rejected = {
                "INSERT INTO patients(patient_id) VALUES (1) LIMIT 1",
                "UPDATE patients SET disease='x' LIMIT 1",
                "DELETE FROM patients LIMIT 1",
                "DROP TABLE patients LIMIT 1",
                "SELECT * FROM patient_data_manager.patients LIMIT 1",
                "SELECT * FROM information_schema.tables LIMIT 1",
                "SELECT * FROM patients FOR UPDATE LIMIT 1",
                "SELECT * FROM patients INTO OUTFILE 'x' LIMIT 1"
        };
        for (String sql : rejected) {
            assertFalse(DynamicReadQueryPolicy.validate(sql).isValid(), sql);
        }
    }

    @Test
    void permitsDomainTextContainingSqlLookingCharactersOnlyInsideLiterals() {
        assertTrue(DynamicReadQueryPolicy.validate(
                "SELECT disease FROM patients WHERE disease = 'T2D;fatty_liver -- domain text' LIMIT 20").isValid());
        assertTrue(DynamicReadQueryPolicy.validate(
                "SELECT disease FROM patients WHERE disease = 'research@example.org ?' LIMIT 20").isValid());
    }

    @Test
    void permitsSafeScalarReplaceFunctionButNotReplaceInto() {
        assertTrue(DynamicReadQueryPolicy.validate(
                "SELECT REPLACE(disease, '_', ' ') AS normalized_disease FROM patients LIMIT 20").isValid());
        assertFalse(DynamicReadQueryPolicy.validate(
                "REPLACE INTO patients(patient_id) VALUES (1) LIMIT 1").isValid());
    }
}
