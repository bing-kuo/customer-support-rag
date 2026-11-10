"""Synchronous local embedding for existing knowledge chunks.

Importing this module does not open a connection or contact a model. The
adapter owns its HTTPX client. Results stay in memory.
"""

import math
import re
from collections.abc import Callable, Sequence

import httpx
from pydantic import BaseModel, ConfigDict

from customer_support_rag.knowledge.models import KnowledgeChunk

EMBEDDING_MODEL = "qwen3-embedding:0.6b"
EMBEDDING_DIMENSIONS = 1024
MAX_CHUNKS_PER_REQUEST = 8
MAX_EMBED_TEXT_CHARS = 1200
MAX_QUERY_CHARS = 1000
QUERY_PROFILE = "qwen3-retrieval-query-v1"
QUERY_INPUT_PREFIX = (
    "Instruct: Given a customer support question, retrieve relevant passages "
    "that answer the question\nQuery:"
)
OLLAMA_EMBED_URL = "http://127.0.0.1:11434/api/embed"
OLLAMA_TAGS_URL = "http://127.0.0.1:11434/api/tags"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUEST_TIMEOUT = httpx.Timeout(connect=5, read=60, write=5, pool=5)

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


class EmbeddedQuery(BaseModel):
    """One question vector produced with the fixed retrieval query profile.

    The vector is the adapter's float tuple. It is not stored and it is not a
    probability.
    """

    model_config = ConfigDict(frozen=True)

    vector: tuple[float, ...]
    model: str
    model_digest: str
    dimensions: int
    query_profile: str


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

    def embed_query(
        self, question: str, *, expected_model_digest: str
    ) -> EmbeddedQuery:
        """Embed one question after the local model digest matches.

        The same owned client reads ``/api/tags`` before and after the single
        embed request. A failed check raises ``EmbeddingError`` and returns no
        vector. This does not write a database or embed document chunks.
        """
        text = query_embedding_input(question)
        digest = _expected_digest(expected_model_digest)
        self._require_tag_digest(digest)
        response = self._post([text])
        self._require_tag_digest(digest)
        vector = _vectors(response, expected_count=1)[0]
        return EmbeddedQuery(
            vector=vector,
            model=EMBEDDING_MODEL,
            model_digest=digest,
            dimensions=EMBEDDING_DIMENSIONS,
            query_profile=QUERY_PROFILE,
        )

    def _post(self, texts: list[str]) -> httpx.Response:
        return self._exchange(
            lambda: self._client.post(
                OLLAMA_EMBED_URL,
                json={
                    "model": EMBEDDING_MODEL,
                    "input": texts,
                    "truncate": False,
                    "keep_alive": 0,
                },
                follow_redirects=False,
                timeout=_REQUEST_TIMEOUT,
            )
        )

    def _get_tags(self) -> httpx.Response:
        return self._exchange(
            lambda: self._client.get(
                OLLAMA_TAGS_URL,
                follow_redirects=False,
                timeout=_REQUEST_TIMEOUT,
            )
        )

    def _exchange(self, send: Callable[[], httpx.Response]) -> httpx.Response:
        try:
            response = send()
        except httpx.TimeoutException:
            raise EmbeddingError(_TIMEOUT_MESSAGE) from None
        except httpx.RequestError:
            raise EmbeddingError(_REQUEST_MESSAGE) from None
        if response.status_code < 200 or response.status_code > 299:
            raise EmbeddingError(
                f"Embedding request returned HTTP {response.status_code}."
            )
        return response

    def _require_tag_digest(self, expected: str) -> None:
        digest = _tag_digest(_json_object(self._get_tags()))
        if digest != expected:
            raise EmbeddingError("Embedding model digest does not match.")


def query_embedding_input(question: str) -> str:
    """Return the exact ``qwen3-retrieval-query-v1`` model input.

    The question is kept as supplied, including its own spaces and line breaks.
    A blank, non-string, or overlong question is rejected before any request.
    """
    if not isinstance(question, str):
        raise ValueError("question must be a string")
    if not question.strip():
        raise ValueError("question must not be blank")
    if len(question) > MAX_QUERY_CHARS:
        raise ValueError("question must be at most 1000 characters")
    text = QUERY_INPUT_PREFIX + question
    if len(text) > MAX_EMBED_TEXT_CHARS:
        raise ValueError("query input must be at most 1200 characters")
    return text


def _expected_digest(value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError("expected model digest must be 64 lowercase hex characters")
    return value


def _tag_digest(payload: dict[str, object]) -> str:
    """Return the one digest whose model name is the fixed embedding tag."""
    models = payload.get("models")
    if not isinstance(models, list):
        raise EmbeddingError("Invalid embedding tags response: unexpected shape.")
    found: list[str] = []
    for item in models:
        if not isinstance(item, dict):
            raise EmbeddingError("Invalid embedding tags response: unexpected shape.")
        if item.get("name") != EMBEDDING_MODEL:
            continue
        digest = item.get("digest")
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise EmbeddingError("Invalid embedding tags response: digest format.")
        found.append(digest)
    if not found:
        raise EmbeddingError("Embedding model was not found.")
    if len(found) != 1:
        raise EmbeddingError("Embedding model was duplicated.")
    return found[0]


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
