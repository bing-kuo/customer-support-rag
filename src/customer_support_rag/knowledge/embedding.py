"""Synchronous local embedding for existing knowledge chunks.

Importing this module does not open a client or contact a model. The caller
owns the HTTPX client and closes it. Results stay in memory.
"""

import math
from collections.abc import Sequence

import httpcore
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
    """

    def __init__(self, client: httpx.Client) -> None:
        _require_direct_client(client)
        self._client = client

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


def _require_direct_client(client: httpx.Client) -> None:
    """Allow a non-retrying HTTPX client or an exact mock transport.

    A subclass, proxy mount, event hook, or connection retry could send the
    document text more than once or to another destination.
    """
    if type(client) is not httpx.Client:
        raise ValueError("client must be an httpx.Client")
    hooks = client.event_hooks
    if hooks.get("request") or hooks.get("response"):
        raise ValueError("client must not define event hooks")
    if any(mounted is not None for mounted in client._mounts.values()):
        raise ValueError("client must not route requests through another transport")

    transport = client._transport
    if type(transport) is httpx.MockTransport:
        return
    if type(transport) is not httpx.HTTPTransport:
        raise ValueError(
            "client transport must be the default HTTP transport or a mock"
        )
    pool = transport._pool
    if type(pool) is not httpcore.ConnectionPool:
        raise ValueError("client transport must not use a proxy")
    if type(pool._retries) is not int or pool._retries != 0:
        raise ValueError("client transport must not retry")
    if pool._proxy is not None or pool._uds is not None:
        raise ValueError("client transport must not use a proxy or local socket")


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
        number = float(item)
        if not math.isfinite(number):
            raise EmbeddingError(_NON_FINITE)
        numbers.append(number)
    squared = math.fsum(number * number for number in numbers)
    if not math.isfinite(squared) or squared <= 0:
        raise EmbeddingError(_ZERO_VECTOR if squared == 0 else _NON_FINITE)
    return tuple(numbers)
