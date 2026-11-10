"""Exact cosine ranking over caller-selected stored embeddings.

Importing this module does not open a database or contact a model. An index
reads the caller's chosen keys once, then ranks those vectors in memory.
Scores are raw cosine values, not probabilities, and there is no cutoff.
"""

import math
import re
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from pydantic import BaseModel, ConfigDict

from customer_support_rag.knowledge.embedding import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    QUERY_PROFILE,
    EmbeddedQuery,
)
from customer_support_rag.knowledge.models import KnowledgeChunk
from customer_support_rag.knowledge.storage import (
    CHUNKING_PROFILE,
    INPUT_PROFILE,
    EmbeddingStore,
    StoredEmbeddingDocument,
)

_MAX_KEYS = 1000
_MAX_ELIGIBLE_CHUNKS = 1000
_MAX_TOP_K = 10
_PRODUCTS = frozenset({"hp-a", "hp-b"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class RetrievalError(Exception):
    """Raised when selected embeddings cannot be ranked safely.

    Messages name a safe category. They do not include question text, document
    text, payload JSON, database paths, or SQL.
    """


class RetrievalHit(BaseModel):
    """One ranked chunk and its unrounded cosine score."""

    model_config = ConfigDict(frozen=True)

    artifact_key: str
    chunk: KnowledgeChunk
    raw_score: float


class _Row(NamedTuple):
    artifact_key: str
    chunk: KnowledgeChunk


class ExactCosineIndex:
    """Rank stored chunk vectors with float64 exact cosine.

    ``from_store`` loads only the caller's keys. ``search`` does not read the
    store, call a model, or treat a low score as a refusal.
    """

    def __init__(self, digest: str, rows: tuple[_Row, ...], matrix: np.ndarray) -> None:
        self._digest = digest
        self._rows = rows
        self._matrix = matrix

    @classmethod
    def from_store(
        cls,
        store: EmbeddingStore,
        *,
        artifact_keys: Sequence[str],
        expected_model_digest: str,
    ) -> "ExactCosineIndex":
        """Load the named documents and prepare normalized vector copies."""
        keys = _require_keys(artifact_keys)
        digest = _require_digest(expected_model_digest)
        documents: list[StoredEmbeddingDocument] = []
        for key in keys:
            loaded = store.get_document(key)
            if loaded is None:
                raise RetrievalError("Required embedding document is missing.")
            _require_compatible(loaded, digest)
            documents.append(loaded)
        rows, vectors = _eligible(documents)
        matrix = (
            _normalized(vectors)
            if vectors
            else np.zeros((0, EMBEDDING_DIMENSIONS), dtype=np.float64)
        )
        return cls(digest, rows, matrix)

    def search(
        self, query: EmbeddedQuery, *, product_id: str, top_k: int = 5
    ) -> list[RetrievalHit]:
        """Return the top product chunks ordered by raw cosine, then source."""
        product = _require_product(product_id)
        limit = _require_top_k(top_k)
        checked = _require_query(query, self._digest)
        query_row = _normalized([checked.vector])[0]
        selected = [
            index
            for index, row in enumerate(self._rows)
            if row.chunk.metadata.product_id == product
        ]
        if not selected:
            return []
        scores = _scores(self._matrix[np.asarray(selected, dtype=np.intp)], query_row)
        order = sorted(
            range(len(selected)),
            key=lambda position: (
                -scores[position],
                self._rows[selected[position]].artifact_key,
                self._rows[selected[position]].chunk.index,
            ),
        )
        return [
            RetrievalHit(
                artifact_key=self._rows[selected[position]].artifact_key,
                chunk=self._rows[selected[position]].chunk,
                raw_score=scores[position],
            )
            for position in order[:limit]
        ]


def _require_keys(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(
            "artifact keys must be a sequence of 1 to 1000 unique sha256 keys"
        )
    if not 1 <= len(value) <= _MAX_KEYS:
        raise ValueError(
            "artifact keys must be a sequence of 1 to 1000 unique sha256 keys"
        )
    keys: list[str] = []
    for item in value:
        if not isinstance(item, str) or _SHA256.fullmatch(item) is None:
            raise ValueError(
                "artifact keys must be a sequence of 1 to 1000 unique sha256 keys"
            )
        keys.append(item)
    if len(set(keys)) != len(keys):
        raise ValueError(
            "artifact keys must be a sequence of 1 to 1000 unique sha256 keys"
        )
    return tuple(keys)


def _require_digest(value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError("expected model digest must be 64 lowercase hex characters")
    return value


def _require_product(value: object) -> str:
    if not isinstance(value, str) or value not in _PRODUCTS:
        raise ValueError("product_id must be hp-a or hp-b")
    return value


def _require_top_k(value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= _MAX_TOP_K
    ):
        raise ValueError("top_k must be an integer from 1 to 10")
    return value


def _require_compatible(document: StoredEmbeddingDocument, digest: str) -> None:
    if (
        document.embedding_model != EMBEDDING_MODEL
        or document.model_digest != digest
        or isinstance(document.dimensions, bool)
        or document.dimensions != EMBEDDING_DIMENSIONS
        or document.input_profile != INPUT_PROFILE
        or document.chunking_profile != CHUNKING_PROFILE
    ):
        raise RetrievalError("Embedding document is not compatible with this index.")


def _require_query(query: object, digest: str) -> EmbeddedQuery:
    if not isinstance(query, EmbeddedQuery):
        raise ValueError("query must be an embedded query")
    if (
        query.model != EMBEDDING_MODEL
        or query.model_digest != digest
        or isinstance(query.dimensions, bool)
        or query.dimensions != EMBEDDING_DIMENSIONS
        or query.query_profile != QUERY_PROFILE
    ):
        raise RetrievalError("Query embedding is not compatible with this index.")
    return query


def _eligible(
    documents: Sequence[StoredEmbeddingDocument],
) -> tuple[tuple[_Row, ...], tuple[tuple[float, ...], ...]]:
    rows: list[_Row] = []
    vectors: list[tuple[float, ...]] = []
    for document in documents:
        for embedded in document.embedded_chunks:
            if not embedded.chunk.text.strip():
                continue
            rows.append(_Row(document.artifact_key, embedded.chunk))
            vectors.append(embedded.vector)
    if len(rows) > _MAX_ELIGIBLE_CHUNKS:
        raise RetrievalError("Too many eligible chunks.")
    return tuple(rows), tuple(vectors)


def _finite_numbers(vector: object) -> list[float]:
    if not isinstance(vector, tuple) or len(vector) != EMBEDDING_DIMENSIONS:
        raise RetrievalError("Retrieval vector has an invalid shape.")
    numbers: list[float] = []
    for item in vector:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise RetrievalError("Retrieval vector is not numeric.")
        try:
            number = float(item)
        except OverflowError:
            raise RetrievalError("Retrieval vector is not finite.") from None
        if not math.isfinite(number):
            raise RetrievalError("Retrieval vector is not finite.")
        numbers.append(number)
    return numbers


def _normalized(vectors: Sequence[tuple[float, ...]]) -> np.ndarray:
    matrix = np.asarray(
        [_finite_numbers(vector) for vector in vectors], dtype=np.float64
    )
    if matrix.shape != (len(vectors), EMBEDDING_DIMENSIONS):
        raise RetrievalError("Retrieval vector has an invalid shape.")
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            norms = np.linalg.norm(matrix, axis=1)
            if norms.shape != (len(vectors),) or not bool(np.all(np.isfinite(norms))):
                raise RetrievalError(
                    "Retrieval vector norm is not finite and positive."
                )
            if bool(np.any(norms <= 0.0)):
                raise RetrievalError(
                    "Retrieval vector norm is not finite and positive."
                )
            normalized = matrix / norms[:, np.newaxis]
    except FloatingPointError:
        raise RetrievalError(
            "Retrieval vector norm is not finite and positive."
        ) from None
    if not bool(np.all(np.isfinite(normalized))):
        raise RetrievalError("Retrieval vector norm is not finite and positive.")
    return normalized


def _scores(documents: np.ndarray, query: np.ndarray) -> list[float]:
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            scores = documents @ query
    except FloatingPointError:
        raise RetrievalError("Retrieval score is not finite.") from None
    if scores.shape != (documents.shape[0],) or not bool(np.all(np.isfinite(scores))):
        raise RetrievalError("Retrieval score is not finite.")
    return [float(score) for score in scores.tolist()]
