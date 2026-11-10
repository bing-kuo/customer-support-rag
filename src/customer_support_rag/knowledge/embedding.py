"""Synchronous local embedding for existing knowledge chunks.

Importing this module does not open a connection or contact a model. The
adapter owns its HTTPX client. Results stay in memory.
"""

import math
from collections.abc import Sequence

import httpx
from pydantic import BaseModel, ConfigDict

from customer_support_rag.knowledge.models import KnowledgeChunk

EMBEDDING_MODEL = "qwen3-embedding:0.6b"
EMBEDDING_DIMENSIONS = 1024
MAX_CHUNKS_PER_REQUEST = 8
MAX_EMBED_TEXT_CHARS = 1200
OLLAMA_EMBED_URL = "http://127.0.0.1:11434/api/embed"

_TIMEOUT_MESSAGE = "Embedding request timed out."
_REQUEST_MESSAGE = "Embedding request failed."
_INVALID_JSON = "Invalid embedding response: malformed JSON."
_INVALID_SHAPE = "Invalid embedding response: unexpected shape."
_INVALID_MODEL = "Invalid embedding response: model mismatch."
_INVALID_NUMBER = "Invalid embedding response: non-numeric value."
_NON_FINITE = "Invalid embedding response: non-finite value."
_ZERO_VECTOR = "Invalid embedding response: zero vector."


class EmbeddingError(Exception):
    """Raised when the embedding request or response cannot be used.

    Messages name a safe category. They do not include document text, response
    bodies, tokens, or local paths.
    """


class EmbeddedChunk(BaseModel):
    """One original chunk and the immutable vector produced for its text."""

    model_config = ConfigDict(frozen=True)

    chunk: KnowledgeChunk
    vector: tuple[float, ...]


class OllamaEmbeddingAdapter:
    """Embed chunk text with the fixed local Ollama model.

    Only ``chunk.text`` is sent. Blank text is skipped and the original chunks
    are left unchanged. The call rejects oversized batches instead of splitting
    or truncating them, and it returns no partial result after a failed response.

    The adapter builds the client with public HTTPX arguments: ``trust_env``
    is off, there is no auth, and redirects are disabled. Production uses
    ``HTTPTransport()``, whose default connection retry count is zero. A test
    may pass an exact ``MockTransport``; there is no caller-supplied client.
    Use :meth:`close` or the context manager when finished.
    """

    def __init__(self, transport: httpx.MockTransport | None = None) -> None:
        if transport is not None and type(transport) is not httpx.MockTransport:
            raise TypeError("transport must be an httpx.MockTransport")
        self._client = httpx.Client(
            transport=transport if transport is not None else httpx.HTTPTransport(),
            trust_env=False,
            auth=None,
            params=None,
            headers=None,
            cookies=None,
            follow_redirects=False,
            event_hooks=None,
        )

    def close(self) -> None:
        """Close the owned client."""
        self._client.close()

    def __enter__(self) -> "OllamaEmbeddingAdapter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def embed_chunks(self, chunks: Sequence[KnowledgeChunk]) -> list[EmbeddedChunk]:
        """Return embeddings for non-blank chunks, in input order."""
        originals = _validated_chunks(chunks)
        eligible = [chunk for chunk in originals if chunk.text.strip()]
        if not eligible:
            return []

        response = self._post([chunk.text for chunk in eligible])
        vectors = _vectors(response, expected_count=len(eligible))
        return [
            EmbeddedChunk(chunk=chunk, vector=vector)
            for chunk, vector in zip(eligible, vectors, strict=True)
        ]

    def _post(self, texts: list[str]) -> httpx.Response:
        try:
            response = self._client.post(
                OLLAMA_EMBED_URL,
                json={
                    "model": EMBEDDING_MODEL,
                    "input": texts,
                    "truncate": False,
                    "keep_alive": 0,
                },
                follow_redirects=False,
                timeout=httpx.Timeout(connect=5, read=60, write=5, pool=5),
            )
        except httpx.TimeoutException:
            raise EmbeddingError(_TIMEOUT_MESSAGE) from None
        except httpx.RequestError:
            raise EmbeddingError(_REQUEST_MESSAGE) from None
        if response.status_code < 200 or response.status_code > 299:
            raise EmbeddingError(
                f"Embedding request returned HTTP {response.status_code}."
            )
        return response


def _validated_chunks(chunks: Sequence[KnowledgeChunk]) -> tuple[KnowledgeChunk, ...]:
    if isinstance(chunks, (str, bytes)) or not isinstance(chunks, Sequence):
        raise ValueError("chunks must be a sequence of knowledge chunks")
    if len(chunks) > MAX_CHUNKS_PER_REQUEST:
        raise ValueError("at most 8 chunks can be embedded at once")
    validated: list[KnowledgeChunk] = []
    for chunk in chunks:
        if not isinstance(chunk, KnowledgeChunk):
            raise ValueError("chunks must be a sequence of knowledge chunks")
        if len(chunk.text) > MAX_EMBED_TEXT_CHARS:
            raise ValueError("chunk text must be at most 1200 characters")
        validated.append(chunk)
    return tuple(validated)


def _vectors(
    response: httpx.Response, *, expected_count: int
) -> tuple[tuple[float, ...], ...]:
    payload = _json_object(response)
    if payload.get("model") != EMBEDDING_MODEL:
        raise EmbeddingError(_INVALID_MODEL)
    embeddings = payload.get("embeddings")
    if not isinstance(embeddings, list) or len(embeddings) != expected_count:
        raise EmbeddingError(_INVALID_SHAPE)
    return tuple(_vector(item) for item in embeddings)


def _json_object(response: httpx.Response) -> dict[str, object]:
    try:
        payload = response.json()
    except RecursionError:
        raise EmbeddingError(_INVALID_JSON) from None
    except ValueError:
        raise EmbeddingError(_INVALID_JSON) from None
    if not isinstance(payload, dict):
        raise EmbeddingError(_INVALID_SHAPE)
    return payload


def _vector(value: object) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != EMBEDDING_DIMENSIONS:
        raise EmbeddingError(_INVALID_SHAPE)
    numbers: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise EmbeddingError(_INVALID_NUMBER)
        try:
            number = float(item)
        except OverflowError:
            raise EmbeddingError(_NON_FINITE) from None
        if not math.isfinite(number):
            raise EmbeddingError(_NON_FINITE)
        numbers.append(number)
    try:
        squared = math.fsum(number * number for number in numbers)
    except OverflowError:
        raise EmbeddingError(_NON_FINITE) from None
    if not math.isfinite(squared) or squared <= 0:
        raise EmbeddingError(_ZERO_VECTOR if squared == 0 else _NON_FINITE)
    return tuple(numbers)
