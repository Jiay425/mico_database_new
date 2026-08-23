from __future__ import annotations

import math
import os
import json
import shutil
import ssl
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


class EmbeddingConfigurationError(ValueError):
    """The Gemini embedding provider is not safely configured."""


class EmbeddingRequestError(RuntimeError):
    """A provider call failed without exposing provider details."""


def _windows_system_ssl_context() -> ssl.SSLContext | None:
    """Build a verified context from Windows' system ROOT store when available."""
    enum_certificates = getattr(ssl, "enum_certificates", None)
    der_to_pem = getattr(ssl, "DER_cert_to_PEM_cert", None)
    if not callable(enum_certificates) or not callable(der_to_pem):
        return None
    try:
        context = ssl.create_default_context()
        loaded = 0
        for certificate, encoding, _trust in enum_certificates("ROOT"):
            if encoding != "x509_asn":
                continue
            try:
                context.load_verify_locations(cadata=der_to_pem(certificate))
                loaded += 1
            except (ssl.SSLError, ValueError):
                continue
        return context if loaded else None
    except (OSError, ssl.SSLError):
        return None


class EmbeddingPort(Protocol):
    modelName: str

    def embed_query(self, query: str) -> list[float]:
        ...

    def embed_document(self, title: str | None, text: str) -> list[float]:
        ...


@dataclass(frozen=True)
class GeminiEmbeddingConfiguration:
    modelName: str = "gemini-embedding-2"
    apiKey: str = field(repr=False, default="")
    taskType: str = "search result"
    transport: str = "sdk"

    @classmethod
    def from_environment(cls, env: Mapping[str, str] | None = None) -> "GeminiEmbeddingConfiguration":
        source = os.environ if env is None else env
        if source.get("MICO_GEMINI_EMBEDDING_ENABLED", "").strip().lower() != "true":
            raise EmbeddingConfigurationError("GEMINI_EMBEDDING_DISABLED")
        api_key = (
            source.get("MICO_GEMINI_API_KEY", "").strip()
            or source.get("GEMINI_API_KEY", "").strip()
            or source.get("GOOGLE_API_KEY", "").strip()
        )
        if not api_key:
            raise EmbeddingConfigurationError("GEMINI_EMBEDDING_KEY_MISSING")
        model = source.get("MICO_GEMINI_EMBEDDING_MODEL", "gemini-embedding-2").strip()
        if model != "gemini-embedding-2":
            raise EmbeddingConfigurationError("GEMINI_EMBEDDING_MODEL_UNSUPPORTED")
        transport = source.get("MICO_GEMINI_EMBEDDING_TRANSPORT", "sdk").strip().lower()
        if transport not in {"sdk", "curl"}:
            raise EmbeddingConfigurationError("GEMINI_EMBEDDING_TRANSPORT_UNSUPPORTED")
        return cls(modelName=model, apiKey=api_key, transport=transport)


def prepare_query(value: str) -> str:
    return f"task: search result | query: {value}"


def prepare_document(title: str | None, text: str) -> str:
    return f"title: {title or 'none'} | text: {text}"


def _as_values(value: Any) -> list[float] | None:
    if isinstance(value, Mapping):
        value = value.get("values") or value.get("embedding")
    else:
        value = getattr(value, "values", None) or getattr(value, "embedding", None)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    try:
        values = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if not values or not all(math.isfinite(item) for item in values):
        return None
    norm = math.sqrt(sum(item * item for item in values))
    if norm <= 0:
        return None
    return [item / norm for item in values]


def extract_embedding(response: Any) -> list[float]:
    """Extract one normalized vector from the google-genai response shape."""
    candidates = []
    embeddings = getattr(response, "embeddings", None)
    if embeddings is None and isinstance(response, Mapping):
        embeddings = response.get("embeddings")
    if isinstance(embeddings, Sequence) and not isinstance(embeddings, (str, bytes, bytearray)):
        candidates.extend(embeddings)
    single = getattr(response, "embedding", None)
    if single is None and isinstance(response, Mapping):
        single = response.get("embedding")
    if single is not None:
        candidates.append(single)
    if not candidates:
        candidates.append(response)
    values = _as_values(candidates[0])
    if values is None:
        raise EmbeddingRequestError("GEMINI_EMBEDDING_RESPONSE_INVALID")
    return values


class GeminiEmbeddingPort:
    """Explicit Gemini Embedding 2 adapter; it never logs credentials or text."""

    def __init__(self, configuration: GeminiEmbeddingConfiguration, client: Any | None = None) -> None:
        self.modelName = configuration.modelName
        self._api_key = configuration.apiKey
        self._transport = configuration.transport
        self._client = client
        self._http_client = None
        self._curl_path = shutil.which("curl.exe") or shutil.which("curl")
        if self._transport == "curl":
            if not self._curl_path:
                raise EmbeddingConfigurationError("GEMINI_EMBEDDING_CURL_UNAVAILABLE")
            return
        if self._client is None:
            try:
                from google import genai
                client_kwargs: dict[str, Any] = {"api_key": configuration.apiKey}
                # Windows system trust contains enterprise/interception roots
                # that are not present in certifi.  This keeps certificate
                # verification enabled while making the provider usable in
                # the project environment.
                try:
                    import httpx
                    from google.genai import types

                    verify_context = _windows_system_ssl_context()
                    if verify_context is None:
                        raise ImportError("system trust store unavailable")
                    self._http_client = httpx.Client(
                        verify=verify_context,
                        timeout=60.0,
                    )
                    client_kwargs["http_options"] = types.HttpOptions(
                        httpxClient=self._http_client,
                    )
                except (ImportError, OSError, ssl.SSLError):
                    pass
                self._client = genai.Client(**client_kwargs)
            except Exception as exc:
                raise EmbeddingConfigurationError("GEMINI_EMBEDDING_SDK_UNAVAILABLE") from exc

    @classmethod
    def from_environment(
        cls,
        env: Mapping[str, str] | None = None,
        client: Any | None = None,
    ) -> "GeminiEmbeddingPort":
        return cls(GeminiEmbeddingConfiguration.from_environment(env), client=client)

    def _embed(self, content: str) -> list[float]:
        if self._transport == "curl":
            return self._curl_embed(content)
        try:
            response = self._client.models.embed_content(
                model=self.modelName,
                contents=content,
            )
            return extract_embedding(response)
        except EmbeddingRequestError:
            raise
        except Exception as exc:
            raise EmbeddingRequestError("GEMINI_EMBEDDING_REQUEST_FAILED") from exc

    @staticmethod
    def _curl_quote(value: str) -> str:
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("\r", "\\r").replace("\n", "\\n") + '"'

    def _curl_embed(self, content: str) -> list[float]:
        payload = json.dumps(
            {"content": {"parts": [{"text": content}]}},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        body_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", suffix=".json", delete=False
            ) as body:
                body.write(payload)
                body_path = body.name
            config = "\n".join([
                "--silent",
                "--show-error",
                "--fail-with-body",
                "--max-time 90",
                "--request POST",
                "--url " + self._curl_quote(
                    f"https://generativelanguage.googleapis.com/v1beta/models/"
                    f"{self.modelName}:embedContent"
                ),
                "--header \"Content-Type: application/json\"",
                f"--header {self._curl_quote('x-goog-api-key: ' + self._api_key)}",
                f"--data-binary {self._curl_quote('@' + body_path)}",
                "",
            ])
            completed = subprocess.run(
                [self._curl_path, "--config", "-"],
                input=config,
                text=True,
                capture_output=True,
                timeout=120,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise EmbeddingRequestError("GEMINI_EMBEDDING_REQUEST_FAILED") from exc
        finally:
            if body_path:
                try:
                    os.unlink(body_path)
                except OSError:
                    pass
        if completed.returncode != 0:
            raise EmbeddingRequestError("GEMINI_EMBEDDING_REQUEST_FAILED")
        try:
            return extract_embedding(json.loads(completed.stdout))
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise EmbeddingRequestError("GEMINI_EMBEDDING_RESPONSE_INVALID") from exc

    def embed_query(self, query: str) -> list[float]:
        return self._embed(prepare_query(query))

    def embed_document(self, title: str | None, text: str) -> list[float]:
        return self._embed(prepare_document(title, text))

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()
        if self._http_client is not None:
            self._http_client.close()
