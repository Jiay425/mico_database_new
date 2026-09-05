package com.database.mico_database.agent.readmodel;

import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import java.nio.charset.StandardCharsets;
import java.security.GeneralSecurityException;
import java.security.SecureRandom;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Produces an opaque sample identity for one Agent Run.
 *
 * The raw source sample identifier never leaves Java.  Each run receives a
 * fresh random HMAC key, so equal samples are joinable inside that run while
 * tokens cannot be correlated across runs or reversed with a dictionary attack
 * against the source identifier.
 */
public final class OpaqueSampleKeyEncoder {

    private static final String ALGORITHM = "HmacSHA256";
    private static final String PREFIX = "s_";
    private static final int KEY_BYTES = 32;
    private static final int TOKEN_HEX_BYTES = 24;

    private final SecureRandom random;
    private final Map<String, byte[]> runKeys = new ConcurrentHashMap<>();

    public OpaqueSampleKeyEncoder() {
        this(new SecureRandom());
    }

    OpaqueSampleKeyEncoder(SecureRandom random) {
        if (random == null) {
            throw new IllegalArgumentException("random is required");
        }
        this.random = random;
    }

    public String encode(String runId, Object rawSampleId) {
        if (runId == null || runId.trim().isEmpty()) {
            throw new IllegalArgumentException("runId is required for an opaque sample key");
        }
        if (rawSampleId == null) {
            return null;
        }
        byte[] key = runKeys.computeIfAbsent(runId, ignored -> newKey());
        try {
            Mac mac = Mac.getInstance(ALGORITHM);
            mac.init(new SecretKeySpec(key, ALGORITHM));
            byte[] digest = mac.doFinal(String.valueOf(rawSampleId)
                    .getBytes(StandardCharsets.UTF_8));
            StringBuilder token = new StringBuilder(PREFIX.length() + TOKEN_HEX_BYTES * 2);
            token.append(PREFIX);
            for (int index = 0; index < TOKEN_HEX_BYTES; index++) {
                token.append(String.format("%02x", digest[index] & 0xff));
            }
            return token.toString();
        } catch (GeneralSecurityException exception) {
            throw new IllegalStateException("opaque sample key unavailable", exception);
        }
    }

    private byte[] newKey() {
        byte[] key = new byte[KEY_BYTES];
        random.nextBytes(key);
        return key;
    }
}
