from __future__ import annotations

import base64

import pytest

from mico_agent_runtime.storage.crypto import (
    RuntimeStateCipher,
    RuntimeStateConfigurationError,
    RuntimeStateDecryptError,
    StateBindingContext,
)
from tests.storage_fixtures import RUN_ID, TASK_ID, TRACE_ID


def key_text() -> str:
    return base64.urlsafe_b64encode(bytes(range(32))).rstrip(b"=").decode("ascii")


def cipher_env(key_id: str = "key-runtime-v1") -> dict[str, str]:
    return {
        "MICO_RUNTIME_STATE_ENCRYPTION_KEY": key_text(),
        "MICO_RUNTIME_STATE_ENCRYPTION_KEY_ID": key_id,
    }


def context() -> StateBindingContext:
    return StateBindingContext(
        runId=RUN_ID,
        taskId=TASK_ID,
        traceId=TRACE_ID,
        dataContractVersion="v1",
    )


def test_aes_gcm_roundtrip_does_not_expose_plaintext() -> None:
    cipher = RuntimeStateCipher.from_environment(cipher_env())
    state = {
        "task": "structured-state",
        "sourceSampleId": "complete-source-sample",
        "internalRecordId": 18423,
        "javaPayload": {"fake": "payload"},
    }
    envelope = cipher.encrypt(state, context())
    serialized = cipher.serialize(envelope)
    assert "complete-source-sample" not in serialized
    assert "18423" not in serialized
    assert "fake" not in serialized
    assert cipher.decrypt(envelope, context()) == state


@pytest.mark.parametrize("field", ["runId", "taskId", "traceId", "dataContractVersion"])
def test_aad_context_mismatch_cannot_decrypt(field: str) -> None:
    cipher = RuntimeStateCipher.from_environment(cipher_env())
    envelope = cipher.encrypt({"state": "safe"}, context())
    changed = {
        "runId": "run-" + "e" * 32,
        "taskId": "task-" + "e" * 32,
        "traceId": "trace-" + "e" * 32,
        "dataContractVersion": "v2",
    }[field]
    if field == "dataContractVersion":
        wrong_context = StateBindingContext.model_construct(
            runId=RUN_ID,
            taskId=TASK_ID,
            traceId=TRACE_ID,
            dataContractVersion=changed,
        )
    else:
        wrong_context = context().model_copy(update={field: changed})
    with pytest.raises(RuntimeStateDecryptError) as error:
        cipher.decrypt(envelope, wrong_context)
    assert error.value.code == "RUNTIME_STATE_DECRYPT_FAILED"
    assert "other" not in str(error.value)


@pytest.mark.parametrize(
    "encoded",
    [None, "not-base64!", base64.urlsafe_b64encode(b"short").rstrip(b"=").decode("ascii")],
)
def test_missing_or_invalid_key_fails_closed(encoded: str | None) -> None:
    env = {} if encoded is None else {
        "MICO_RUNTIME_STATE_ENCRYPTION_KEY": encoded,
        "MICO_RUNTIME_STATE_ENCRYPTION_KEY_ID": "key-runtime-v1",
    }
    with pytest.raises(RuntimeStateConfigurationError) as error:
        RuntimeStateCipher.from_environment(env)
    assert error.value.code == "RUNTIME_STATE_ENCRYPTION_KEY_INVALID"
    assert "short" not in str(error.value)


def test_tampered_envelope_returns_fixed_error() -> None:
    cipher = RuntimeStateCipher.from_environment(cipher_env())
    envelope = cipher.encrypt({"state": "safe"}, context())
    ciphertext = bytearray(base64.urlsafe_b64decode(envelope.ciphertext + "==="))
    ciphertext[0] ^= 0x01
    tampered_ciphertext = base64.urlsafe_b64encode(bytes(ciphertext)).rstrip(b"=").decode("ascii")
    assert tampered_ciphertext != envelope.ciphertext
    tampered = envelope.model_copy(update={"ciphertext": tampered_ciphertext})
    with pytest.raises(RuntimeStateDecryptError) as error:
        cipher.decrypt(tampered, context())
    assert error.value.code == "RUNTIME_STATE_DECRYPT_FAILED"


def test_tampered_nonce_returns_fixed_error() -> None:
    cipher = RuntimeStateCipher.from_environment(cipher_env())
    envelope = cipher.encrypt({"state": "safe"}, context())
    nonce = bytearray(base64.urlsafe_b64decode(envelope.nonce + "==="))
    nonce[-1] ^= 0x01
    tampered_nonce = base64.urlsafe_b64encode(bytes(nonce)).rstrip(b"=").decode("ascii")
    assert tampered_nonce != envelope.nonce
    with pytest.raises(RuntimeStateDecryptError) as error:
        cipher.decrypt(envelope.model_copy(update={"nonce": tampered_nonce}), context())
    assert error.value.code == "RUNTIME_STATE_DECRYPT_FAILED"


@pytest.mark.parametrize("encoded", ["not-base64!", "abc=", "abc$", "A"])
def test_noncanonical_or_invalid_base64url_is_rejected_safely(encoded: str) -> None:
    cipher = RuntimeStateCipher.from_environment(cipher_env())
    envelope = cipher.encrypt({"state": "safe"}, context())
    with pytest.raises(RuntimeStateDecryptError) as error:
        cipher.decrypt(envelope.model_copy(update={"ciphertext": encoded}), context())
    assert error.value.code == "RUNTIME_STATE_DECRYPT_FAILED"
    if len(encoded) > 1:
        assert encoded not in str(error.value)


def test_noncanonical_base64url_with_equivalent_decoded_bytes_is_rejected() -> None:
    cipher = RuntimeStateCipher.from_environment(cipher_env())
    envelope = cipher.encrypt({"s": "x"}, context())
    assert len(envelope.ciphertext) % 4 == 2
    last = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_".index(envelope.ciphertext[-1])
    noncanonical_last = (last & 0b110000) | ((last & 0b001111) | 0b0001)
    noncanonical = envelope.ciphertext[:-1] + "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"[noncanonical_last]
    assert noncanonical != envelope.ciphertext
    with pytest.raises(RuntimeStateDecryptError) as error:
        cipher.decrypt(envelope.model_copy(update={"ciphertext": noncanonical}), context())
    assert error.value.code == "RUNTIME_STATE_DECRYPT_FAILED"


def test_nonce_wrong_length_is_rejected_before_decryption() -> None:
    cipher = RuntimeStateCipher.from_environment(cipher_env())
    envelope = cipher.encrypt({"state": "safe"}, context())
    with pytest.raises(RuntimeStateDecryptError) as error:
        cipher.decrypt(envelope.model_copy(update={"nonce": "AQ"}), context())
    assert error.value.code == "RUNTIME_STATE_DECRYPT_FAILED"


def test_ciphertext_shorter_than_gcm_tag_is_rejected_before_decryption() -> None:
    cipher = RuntimeStateCipher.from_environment(cipher_env())
    envelope = cipher.encrypt({"state": "safe"}, context())
    with pytest.raises(RuntimeStateDecryptError) as error:
        cipher.decrypt(envelope.model_copy(update={"ciphertext": "AA"}), context())
    assert error.value.code == "RUNTIME_STATE_DECRYPT_FAILED"


def test_decrypt_error_does_not_expose_underlying_details() -> None:
    cipher = RuntimeStateCipher.from_environment(cipher_env())
    envelope = cipher.encrypt({"state": "safe"}, context())
    tampered = envelope.model_copy(update={"ciphertext": "not-base64!"})
    with pytest.raises(RuntimeStateDecryptError) as error:
        cipher.decrypt(tampered, context())
    assert str(error.value) == "RUNTIME_STATE_DECRYPT_FAILED"
    assert "InvalidTag" not in str(error.value)
    assert "not-base64!" not in str(error.value)


@pytest.mark.parametrize("key_id", [None, "not a valid key id", "", "x" * 129])
def test_missing_or_invalid_explicit_key_id_fails_closed(key_id: str | None) -> None:
    env = {"MICO_RUNTIME_STATE_ENCRYPTION_KEY": key_text()}
    if key_id is not None:
        env["MICO_RUNTIME_STATE_ENCRYPTION_KEY_ID"] = key_id
    with pytest.raises(RuntimeStateConfigurationError) as error:
        RuntimeStateCipher.from_environment(env)
    assert error.value.code == "RUNTIME_STATE_ENCRYPTION_KEY_ID_INVALID"
    if key_id:
        assert key_id not in str(error.value)


def test_key_id_is_explicit_and_not_derived_from_key() -> None:
    cipher = RuntimeStateCipher.from_environment(cipher_env("runtime-state-key-v2"))
    assert cipher.key_id == "runtime-state-key-v2"
