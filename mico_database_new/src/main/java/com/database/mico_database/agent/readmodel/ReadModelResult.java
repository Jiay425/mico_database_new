package com.database.mico_database.agent.readmodel;

/** Typed read result paired with a transient read receipt. */
public final class ReadModelResult<T> {

    private final T data;
    private final ReadReceipt readReceipt;

    public ReadModelResult(T data, ReadReceipt readReceipt) {
        this.data = data;
        this.readReceipt = readReceipt;
    }

    public T getData() {
        return data;
    }

    public ReadReceipt getReadReceipt() {
        return readReceipt;
    }
}
