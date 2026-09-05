package com.database.mico_database.agent.readmodel;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class OpaqueSampleKeyEncoderTest {

    @Test
    void tokenIsStableWithinRunButNotAcrossRuns() {
        OpaqueSampleKeyEncoder encoder = new OpaqueSampleKeyEncoder();

        String first = encoder.encode("run-a", "sample-123");
        String same = encoder.encode("run-a", "sample-123");
        String otherRun = encoder.encode("run-b", "sample-123");

        assertEquals(first, same);
        assertNotEquals(first, otherRun);
        assertTrue(first.matches("s_[0-9a-f]{48}"));
        assertTrue(!first.contains("sample-123"));
    }
}
