"""Deterministic mock embeddings. This does not contact Ollama."""

import hashlib
import json

import httpx

from customer_support_rag.knowledge.chunker import chunk_document
from customer_support_rag.knowledge.embedding import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    OllamaEmbeddingAdapter,
)
from customer_support_rag.knowledge.models import KnowledgeDocument

MOCK_EMBED_DIGEST = hashlib.sha256(b"customer-support-rag-mock-embed-v1").hexdigest()


def mock_vector(text: str) -> list[float]:
    """Map character bigrams into the fixed 1024-dimension shape.

    The numbers are a test double. They are not Qwen vectors and they are not
    a confidence score.
    """
    vector = [0.0] * EMBEDDING_DIMENSIONS
    characters = [character for character in text if not character.isspace()]
    if len(characters) < 2:
        vector[0] = 1.0
        return vector
    for left, right in zip(characters, characters[1:], strict=False):
        bucket = (ord(left) * 131 + ord(right)) % EMBEDDING_DIMENSIONS
        vector[bucket] += 1.0
    if not any(vector):
        vector[0] = 1.0
    return vector


def mock_transport(digest: str = MOCK_EMBED_DIGEST) -> httpx.MockTransport:
    """Return a MockTransport for ``/api/embed`` and ``/api/tags`` only."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={"models": [{"name": EMBEDDING_MODEL, "digest": digest}]},
            )
        if request.url.path != "/api/embed":
            return httpx.Response(404)
        try:
            payload = json.loads(request.content.decode("utf-8"))
            texts = payload["input"]
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError):
            return httpx.Response(400)
        if not isinstance(texts, list) or not all(
            isinstance(item, str) for item in texts
        ):
            return httpx.Response(400)
        return httpx.Response(
            200,
            json={
                "model": EMBEDDING_MODEL,
                "embeddings": [mock_vector(text) for text in texts],
            },
        )

    return httpx.MockTransport(handler)


class MockEmbedder:
    """Embed one document through the owned mock transport."""

    def __init__(self) -> None:
        self.calls = 0
        self.transport = mock_transport()

    def embed_document(self, document: KnowledgeDocument) -> list[object]:
        self.calls += 1
        with OllamaEmbeddingAdapter(self.transport) as adapter:
            return adapter.embed_chunks(chunk_document(document))
