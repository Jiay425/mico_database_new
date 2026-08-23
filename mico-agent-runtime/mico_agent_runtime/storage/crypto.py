from __future__ import annotations

import base64
import binascii
import json
import os
import re
from typing import Any, Literal, Mapping

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pydantic import Field, TypeAdapter

from mico_agent_runtime.contracts.base import ClosedModel

from .types import KeyIdentifier, RunId, TaskId, TraceId


class RuntimeStateConfigurationError(RuntimeError):
    """Safe configuration failure with no key or plaintext details."""

    def __init__(self, code: str = "RUNTIME_STATE_ENCRYPTION_KEY_INVALID") -> None:
        super().__init__(code)
        self.code = code


class RuntimeStateEncryptionError(RuntimeError):
    """Safe encryption failure with no plaintext or ciphertext details."""

    def __init__(self, code: str = "RUNTIME_STATE_ENCRYPT_FAILED") -> None:
        super().__init__(code)
        self.code = code


class RuntimeStateDecryptError(RuntimeError):
    """Safe decryption failure with no key, AAD, or payload details."""

    def __init__(self, code: str = "RUNTIME_STATE_DECRYPT_FAILED") -> None:
        super().__init__(code)
        self.code = code


class StateBindingContext(ClosedModel):
    runId: RunId
    taskId: TaskId
    traceId: TraceId
    dataContractVersion: Literal["v1"]


class EncryptedStateEnvelope(ClosedModel):
    algorithm: str = Field(pattern=r"^AES-256-GCM$")
    keyId: KeyIdentifier
    nonce: str = Field(min_length=1, max_length=64)
    ciphertext: str = Field(min_length=1, max_length=131072)


_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _decode(value: str) -> bytes:
    if not value or not _KEY_PATTERN.fullmatch(value):
        raise ValueError("invalid base64url")
    padding = "=" * (-len(value) % 4)
    if len(value) % 4 == 1:
        raise ValueError("invalid base64url")
    try:
        decoded = base64.urlsafe_b64decode(value + padding)
    except (binascii.Error, ValueError, TypeError) as exc:
        raise ValueError("invalid base64url") from exc
    if _encode(decoded) != value:
        raise ValueError("invalid base64url")
    return decoded


def _aad_bytes(context: StateBindingContext) -> bytes:
    return json.dumps(
        context.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


class RuntimeStateCipher:
    """AES-256-GCM codec with environment-only key loading and bound AAD."""

    algorithm = "AES-256-GCM"

    def __init__(self, key: bytes, key_id: str) -> None:
        if len(key) != 32:
            raise RuntimeStateConfigurationError()
        try:
            validated_key_id = TypeAdapter(KeyIdentifier).validate_python(key_id)
        except (TypeError, ValueError):
            raise RuntimeStateConfigurationError("RUNTIME_STATE_ENCRYPTION_KEY_ID_INVALID") from None
        self._key = bytes(key)
        self.key_id = validated_key_id
        self._aes = AESGCM(self._key)

    @classmethod
    def from_environment(cls, env: Mapping[str, str] | None = None) -> "RuntimeStateCipher":
        source = os.environ if env is None else env
        encoded = source.get("MICO_RUNTIME_STATE_ENCRYPTION_KEY")
        if not encoded or not _KEY_PATTERN.fullmatch(encoded):
            raise RuntimeStateConfigurationError()
        key_id = source.get("MICO_RUNTIME_STATE_ENCRYPTION_KEY_ID")
        if not key_id:
            raise RuntimeStateConfigurationError("RUNTIME_STATE_ENCRYPTION_KEY_ID_INVALID")
        try:
            TypeAdapter(KeyIdentifier).validate_python(key_id)
        except (TypeError, ValueError):
            raise RuntimeStateConfigurationError("RUNTIME_STATE_ENCRYPTION_KEY_ID_INVALID") from None
        try:
            key = _decode(encoded)
        except (ValueError, binascii.Error):
            raise RuntimeStateConfigurationError() from None
        return cls(key, key_id)

    def encrypt(self, state: Mapping[str, Any], context: StateBindingContext) -> EncryptedStateEnvelope:
        try:
            plaintext = json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            nonce = os.urandom(12)
            ciphertext = self._aes.encrypt(nonce, plaintext, _aad_bytes(context))
            return EncryptedStateEnvelope(
                algorithm=self.algorithm,
                keyId=self.key_id,
                nonce=_encode(nonce),
                ciphertext=_encode(ciphertext),
            )
        except RuntimeStateEncryptionError:
            raise
        except (TypeError, ValueError, OSError) as exc:
            raise RuntimeStateEncryptionError() from exc

    def decrypt(self, envelope: EncryptedStateEnvelope, context: StateBindingContext) -> dict[str, Any]:
        try:
            if envelope.algorithm != self.algorithm or envelope.keyId != self.key_id:
                raise RuntimeStateDecryptError()
            nonce = _decode(envelope.nonce)
            ciphertext = _decode(envelope.ciphertext)
            if len(nonce) != 12 or len(ciphertext) < 16:
                raise ValueError("invalid encrypted envelope size")
            plaintext = self._aes.decrypt(nonce, ciphertext, _aad_bytes(context))
            value = json.loads(plaintext.decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("state payload must be an object")
            return value
        except RuntimeStateDecryptError:
            raise
        except (AttributeError, InvalidTag, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError, binascii.Error):
            raise RuntimeStateDecryptError() from None

    @staticmethod
    def serialize(envelope: EncryptedStateEnvelope) -> str:
        """Serialize only the encrypted envelope; never serialize plaintext state."""

        return json.dumps(envelope.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    @staticmethod
    def deserialize(value: str) -> EncryptedStateEnvelope:
        try:
            parsed = json.loads(value)
            return EncryptedStateEnvelope.model_validate(parsed)
        except (TypeError, ValueError, json.JSONDecodeError):
            raise RuntimeStateDecryptError() from None
