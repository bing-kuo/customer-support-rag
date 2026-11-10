"""SQLite persistence for one synthetic document and its chunk embeddings.

Importing this module does not open a database or contact a model. The store
never calls the embedding adapter, scans a directory, or recomputes vectors.
The caller supplies an existing parent directory and the model digest.
"""

import hashlib
import json
import math
import re
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

from pydantic import BaseModel, ConfigDict, ValidationError

from customer_support_rag.knowledge.chunker import chunk_document
from customer_support_rag.knowledge.embedding import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    EmbeddedChunk,
)
from customer_support_rag.knowledge.models import (
    KnowledgeChunk,
    KnowledgeDocument,
    KnowledgeMetadata,
)

FORMAT_VERSION = 1
INPUT_PROFILE = "chunk-text-v1"
CHUNKING_PROFILE = "heading-char-v1"
_LOCK_TIMEOUT_SECONDS = 5.0
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PAYLOAD_KEYS = {"format_version", "source", "embedding", "chunking", "chunks"}
_SOURCE_KEYS = {"metadata", "body", "body_sha256"}
_METADATA_KEYS = {"document_id", "product_id", "title", "version", "synthetic"}
_EMBEDDING_KEYS = {"model", "model_digest", "dimensions", "input_profile"}
_CHUNKING_KEYS = {"profile", "max_chars"}
_CHUNK_KEYS = {"index", "char_start", "char_end", "heading_path", "vector"}


class _Rejected(Exception):
    """A stored or caller-supplied value failed a specific contract check."""


class StorageError(Exception):
    """Raised when a stored embedding cannot be read or written safely.

    Messages name a safe category. They do not include document text, payload
    JSON, database paths, or SQL parameters.
    """


class StorageConflictError(StorageError):
    """Raised when an existing artifact key has a different payload."""


class StoredEmbeddingDocument(BaseModel):
    """One saved document, its settings, and the chunks read back from it."""

    model_config = ConfigDict(frozen=True)

    artifact_key: str
    format_version: int
    document: KnowledgeDocument
    embedding_model: str
    model_digest: str
    dimensions: int
    input_profile: str
    chunking_profile: str
    max_chars: int
    chunks: tuple[KnowledgeChunk, ...]
    embedded_chunks: tuple[EmbeddedChunk, ...]


class _Prepared(NamedTuple):
    key: str
    payload: str


def document_embedding_key(
    document: KnowledgeDocument, *, model_digest: str, max_chars: int = 1200
) -> str:
    """Return the identity key for a document without reading a database.

    The digest is caller-supplied. The key covers metadata, the body hash, and
    the embedding and chunking settings. It does not cover vectors.
    """
    checked = _require_document(document)
    digest = _require_sha256(
        model_digest, "model digest must be 64 lowercase hex characters"
    )
    limit = _require_max_chars(max_chars)
    _require_source(checked)
    return _key_for(checked, digest, limit)


class EmbeddingStore:
    """Save and reload one document payload per artifact key.

    ``isolation_level=None`` makes the transaction explicit on Python 3.12 and
    3.13. Writes use ``BEGIN IMMEDIATE`` and do not retry a locked database.
    """

    def __init__(self, path: str | Path) -> None:
        database = _database_path(path)
        try:
            self._connection = sqlite3.connect(
                database,
                isolation_level=None,
                timeout=_LOCK_TIMEOUT_SECONDS,
            )
            self._connection.execute("PRAGMA busy_timeout = 5000")
            self._prepare_schema()
        except StorageError:
            self._close_failed_open()
            raise
        except sqlite3.Error:
            self._close_failed_open()
            raise StorageError("Embedding store could not open the database.") from None

    def close(self) -> None:
        """Close the owned database connection."""
        self._connection.close()

    def _close_failed_open(self) -> None:
        connection = getattr(self, "_connection", None)
        if connection is not None:
            connection.close()

    def __enter__(self) -> "EmbeddingStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def put_document(
        self,
        document: KnowledgeDocument,
        embeddings: Sequence[EmbeddedChunk],
        *,
        model_digest: str,
        max_chars: int = 1200,
    ) -> str:
        """Validate and atomically insert one document payload.

        An identical canonical payload for the same key is a no-op. A different
        payload for that key raises :class:`StorageConflictError` and keeps the
        stored row.
        """
        prepared = _prepare_put(document, embeddings, model_digest, max_chars)
        self._write(prepared.key, prepared.payload)
        return prepared.key

    def get_document(self, artifact_key: str) -> StoredEmbeddingDocument | None:
        """Return the saved document, or None when the key is absent.

        A present row that does not match the storage contract raises
        :class:`StorageError`. It is not treated as a missing document and it
        is not rewritten.
        """
        key = _require_sha256(
            artifact_key, "artifact key must be 64 lowercase hex characters"
        )
        try:
            row = self._connection.execute(
                "SELECT payload_json FROM embedding_documents WHERE artifact_key = ?",
                (key,),
            ).fetchone()
        except sqlite3.Error:
            raise StorageError("Embedding store read failed.") from None
        if row is None:
            return None
        payload = row[0]
        if not isinstance(payload, str):
            raise StorageError("Invalid stored embedding payload.")
        return _decode(key, payload)

    def _prepare_schema(self) -> None:
        try:
            self._connection.execute("BEGIN IMMEDIATE")
        except sqlite3.Error:
            raise StorageError("Embedding store could not open the database.") from None
        try:
            version = _user_version(self._connection)
            objects = _schema_objects(self._connection)
            if version == 0 and not objects:
                self._connection.execute(
                    """
                    CREATE TABLE embedding_documents (
                        artifact_key TEXT PRIMARY KEY,
                        payload_json TEXT NOT NULL
                    )
                    """
                )
                self._connection.execute("PRAGMA user_version = 1")
                self._connection.execute("COMMIT")
                return
            if (
                version == FORMAT_VERSION
                and objects == [("table", "embedding_documents")]
                and _table_matches(self._connection)
            ):
                self._connection.execute("ROLLBACK")
                return
            self._connection.execute("ROLLBACK")
        except _Rejected:
            _rollback(self._connection)
            raise StorageError("Unsupported database schema.") from None
        except sqlite3.Error:
            _rollback(self._connection)
            raise StorageError("Embedding store could not open the database.") from None
        raise StorageError("Unsupported database schema.")

    def _write(self, key: str, payload: str) -> None:
        try:
            self._connection.execute("BEGIN IMMEDIATE")
        except sqlite3.Error:
            raise StorageError("Embedding store write failed.") from None
        try:
            row = self._connection.execute(
                "SELECT payload_json FROM embedding_documents WHERE artifact_key = ?",
                (key,),
            ).fetchone()
            if row is not None:
                if row[0] == payload:
                    self._connection.execute("COMMIT")
                    return
                self._connection.execute("ROLLBACK")
                raise StorageConflictError(
                    "Stored embedding payload conflicts with the new value."
                )
            self._connection.execute(
                """
                INSERT INTO embedding_documents (artifact_key, payload_json)
                VALUES (?, ?)
                """,
                (key, payload),
            )
            self._connection.execute("COMMIT")
        except StorageConflictError:
            raise
        except sqlite3.Error:
            _rollback(self._connection)
            raise StorageError("Embedding store write failed.") from None


def _database_path(path: object) -> Path:
    if not isinstance(path, (str, Path)):
        raise ValueError("database path must be a filesystem path")
    database = Path(path)
    if database.is_dir():
        raise ValueError("database path must not be a directory")
    parent = database.parent
    if parent != Path("") and not parent.is_dir():
        raise ValueError("database parent directory must exist")
    return database


def _prepare_put(
    document: object,
    embeddings: object,
    model_digest: object,
    max_chars: object,
) -> _Prepared:
    checked = _require_document(document)
    digest = _require_sha256(
        model_digest, "model digest must be 64 lowercase hex characters"
    )
    limit = _require_max_chars(max_chars)
    _require_source(checked)
    chunks = tuple(chunk_document(checked, max_chars=limit))
    if not any(chunk.text.strip() for chunk in chunks):
        raise ValueError("document body must contain non-blank text")
    body_hash = _body_sha256(checked.body)
    if any(chunk.body_sha256 != body_hash for chunk in chunks):
        raise ValueError("embeddings must match the eligible chunks")
    supplied = _require_embeddings(embeddings)
    eligible = tuple(chunk for chunk in chunks if chunk.text.strip())
    if len(supplied) != len(eligible):
        raise ValueError("embeddings must match the eligible chunks")
    vectors: dict[int, tuple[float, ...]] = {}
    for item, chunk in zip(supplied, eligible, strict=True):
        if not _same_chunk(item.chunk, chunk):
            raise ValueError("embeddings must match the eligible chunks")
        try:
            vectors[chunk.index] = _vector(item.vector)
        except _Rejected:
            raise ValueError(
                "embedding vector must contain 1024 finite non-zero numbers"
            ) from None
    payload = _canonical_json(
        _payload(checked, chunks, vectors, digest, limit, body_hash)
    )
    return _Prepared(_key_for(checked, digest, limit, body_hash), payload)


def _decode(artifact_key: str, payload_json: str) -> StoredEmbeddingDocument:
    try:
        payload = json.loads(payload_json)
        return _stored_from_payload(artifact_key, payload)
    except (
        json.JSONDecodeError,
        RecursionError,
        ValueError,
        UnicodeEncodeError,
        _Rejected,
        ValidationError,
    ):
        raise StorageError("Invalid stored embedding payload.") from None


def _stored_from_payload(artifact_key: str, payload: object) -> StoredEmbeddingDocument:
    record = _object(payload, _PAYLOAD_KEYS)
    if not _exact_int(record.get("format_version"), FORMAT_VERSION):
        raise _Rejected
    source = _object(record.get("source"), _SOURCE_KEYS)
    metadata_record = _object(source.get("metadata"), _METADATA_KEYS)
    embedding = _object(record.get("embedding"), _EMBEDDING_KEYS)
    chunking = _object(record.get("chunking"), _CHUNKING_KEYS)
    body = source.get("body")
    if not isinstance(body, str):
        raise _Rejected
    body_hash = source.get("body_sha256")
    if body_hash != _body_sha256(body):
        raise _Rejected
    digest = embedding.get("model_digest")
    if (
        embedding.get("model") != EMBEDDING_MODEL
        or not _exact_int(embedding.get("dimensions"), EMBEDDING_DIMENSIONS)
        or embedding.get("input_profile") != INPUT_PROFILE
        or not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
    ):
        raise _Rejected
    if chunking.get("profile") != CHUNKING_PROFILE:
        raise _Rejected
    limit = chunking.get("max_chars")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 2 <= limit <= 1200:
        raise _Rejected
    metadata = KnowledgeMetadata(
        document_id=_text(metadata_record.get("document_id")),
        product_id=_text(metadata_record.get("product_id")),
        title=_text(metadata_record.get("title")),
        version=_text(metadata_record.get("version")),
        synthetic=_true(metadata_record.get("synthetic")),
    )
    _reject_invalid_source(metadata, body)
    document = KnowledgeDocument(metadata=metadata, body=body)
    expected = tuple(chunk_document(document, max_chars=limit))
    raw_chunks = record.get("chunks")
    if not isinstance(raw_chunks, list) or len(raw_chunks) != len(expected):
        raise _Rejected
    chunks: list[KnowledgeChunk] = []
    embedded: list[EmbeddedChunk] = []
    for index, (raw_chunk, chunk) in enumerate(zip(raw_chunks, expected, strict=True)):
        saved = _object(raw_chunk, _CHUNK_KEYS)
        heading = _heading_path(saved.get("heading_path"))
        if (
            not _exact_int(saved.get("index"), index)
            or not _exact_int(saved.get("char_start"), chunk.char_start)
            or not _exact_int(saved.get("char_end"), chunk.char_end)
            or heading != chunk.heading_path
            or chunk.index != index
            or chunk.body_sha256 != body_hash
        ):
            raise _Rejected
        vector = saved.get("vector")
        if chunk.text.strip():
            numbers = _vector(vector)
            embedded.append(EmbeddedChunk(chunk=chunk, vector=numbers))
        elif vector is not None:
            raise _Rejected
        chunks.append(chunk)
    if _key_for(document, digest, limit, body_hash) != artifact_key:
        raise _Rejected
    return StoredEmbeddingDocument(
        artifact_key=artifact_key,
        format_version=FORMAT_VERSION,
        document=document,
        embedding_model=EMBEDDING_MODEL,
        model_digest=digest,
        dimensions=EMBEDDING_DIMENSIONS,
        input_profile=INPUT_PROFILE,
        chunking_profile=CHUNKING_PROFILE,
        max_chars=limit,
        chunks=tuple(chunks),
        embedded_chunks=tuple(embedded),
    )


def _payload(
    document: KnowledgeDocument,
    chunks: Sequence[KnowledgeChunk],
    vectors: Mapping[int, tuple[float, ...]],
    model_digest: str,
    max_chars: int,
    body_hash: str,
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for chunk in chunks:
        vector = vectors.get(chunk.index)
        if chunk.text.strip():
            if vector is None:
                raise _Rejected
            stored_vector: list[float] | None = list(vector)
        else:
            stored_vector = None
        records.append(
            {
                "index": chunk.index,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "heading_path": list(chunk.heading_path),
                "vector": stored_vector,
            }
        )
    return {
        "format_version": FORMAT_VERSION,
        "source": {
            "metadata": _metadata_record(document.metadata),
            "body": document.body,
            "body_sha256": body_hash,
        },
        "embedding": _embedding_record(model_digest),
        "chunking": _chunking_record(max_chars),
        "chunks": records,
    }


def _key_for(
    document: KnowledgeDocument,
    model_digest: str,
    max_chars: int,
    body_hash: str | None = None,
) -> str:
    identity = {
        "format_version": FORMAT_VERSION,
        "metadata": _metadata_record(document.metadata),
        "body_sha256": body_hash or _body_sha256(document.body),
        "embedding": _embedding_record(model_digest),
        "chunking": _chunking_record(max_chars),
    }
    return hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()


def _metadata_record(metadata: KnowledgeMetadata) -> dict[str, object]:
    return {
        "document_id": metadata.document_id,
        "product_id": metadata.product_id,
        "title": metadata.title,
        "version": metadata.version,
        "synthetic": metadata.synthetic,
    }


def _embedding_record(model_digest: str) -> dict[str, object]:
    return {
        "model": EMBEDDING_MODEL,
        "model_digest": model_digest,
        "dimensions": EMBEDDING_DIMENSIONS,
        "input_profile": INPUT_PROFILE,
    }


def _chunking_record(max_chars: int) -> dict[str, object]:
    return {"profile": CHUNKING_PROFILE, "max_chars": max_chars}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _require_document(document: object) -> KnowledgeDocument:
    if not isinstance(document, KnowledgeDocument):
        raise ValueError("document must be a knowledge document")
    return document


def _reject_invalid_source(metadata: object, body: object) -> None:
    """Re-check a source that model_copy can build without field validators."""
    if not isinstance(body, str) or not body.strip():
        raise _Rejected("document body must contain non-blank text")
    if not isinstance(metadata, KnowledgeMetadata):
        raise _Rejected("document metadata must be synthetic hp-a or hp-b")
    try:
        KnowledgeMetadata(
            document_id=metadata.document_id,
            product_id=metadata.product_id,
            title=metadata.title,
            version=metadata.version,
            synthetic=metadata.synthetic,
        )
    except ValidationError:
        raise _Rejected("document metadata must be synthetic hp-a or hp-b") from None


def _require_source(document: KnowledgeDocument) -> None:
    try:
        _reject_invalid_source(document.metadata, document.body)
    except _Rejected as rejected:
        message = rejected.args[0] if rejected.args else "invalid document source"
        raise ValueError(message) from None


def _require_embeddings(embeddings: object) -> tuple[EmbeddedChunk, ...]:
    if isinstance(embeddings, (str, bytes)) or not isinstance(embeddings, Sequence):
        raise ValueError("embeddings must be a sequence of embedded chunks")
    items: list[EmbeddedChunk] = []
    for item in embeddings:
        if not isinstance(item, EmbeddedChunk):
            raise ValueError("embeddings must be a sequence of embedded chunks")
        items.append(item)
    return tuple(items)


def _require_sha256(value: object, message: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(message)
    return value


def _require_max_chars(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 2 <= value <= 1200:
        raise ValueError("max_chars must be an integer from 2 to 1200")
    return value


def _same_chunk(left: KnowledgeChunk, right: KnowledgeChunk) -> bool:
    return (
        isinstance(left, KnowledgeChunk)
        and left.metadata == right.metadata
        and left.body_sha256 == right.body_sha256
        and left.index == right.index
        and left.char_start == right.char_start
        and left.char_end == right.char_end
        and left.heading_path == right.heading_path
        and left.text == right.text
    )


def _vector(value: object) -> tuple[float, ...]:
    # Caller and stored vectors are checked again here. The adapter's private
    # parser is not reused, and a constructed model is not trusted.
    if not isinstance(value, (list, tuple)) or isinstance(value, (str, bytes)):
        raise _Rejected
    if len(value) != EMBEDDING_DIMENSIONS:
        raise _Rejected
    numbers: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise _Rejected
        try:
            number = float(item)
        except OverflowError:
            raise _Rejected from None
        if not math.isfinite(number):
            raise _Rejected
        numbers.append(number)
    try:
        squared = math.fsum(number * number for number in numbers)
    except OverflowError:
        raise _Rejected from None
    if not math.isfinite(squared) or squared <= 0:
        raise _Rejected
    return tuple(numbers)


def _object(value: object, keys: set[str]) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise _Rejected
    return value


def _heading_path(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise _Rejected
    return tuple(value)


def _exact_int(value: object, expected: int) -> bool:
    return type(value) is int and value == expected


def _text(value: object) -> str:
    if not isinstance(value, str):
        raise _Rejected
    return value


def _true(value: object) -> bool:
    if value is not True:
        raise _Rejected
    return True


def _body_sha256(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _user_version(connection: sqlite3.Connection) -> int:
    row = connection.execute("PRAGMA user_version").fetchone()
    if row is None or isinstance(row[0], bool) or not isinstance(row[0], int):
        raise _Rejected
    return row[0]


def _schema_objects(connection: sqlite3.Connection) -> list[tuple[str, str]]:
    rows = connection.execute(
        """
        SELECT type, name FROM sqlite_master
        WHERE name NOT LIKE 'sqlite_%'
        ORDER BY type, name
        """
    ).fetchall()
    return [(str(row[0]), str(row[1])) for row in rows]


def _table_matches(connection: sqlite3.Connection) -> bool:
    rows = connection.execute("PRAGMA table_info(embedding_documents)").fetchall()
    if len(rows) != 2:
        return False
    key, payload = rows
    return (
        key[1] == "artifact_key"
        and str(key[2]).upper() == "TEXT"
        and key[5] == 1
        and payload[1] == "payload_json"
        and str(payload[2]).upper() == "TEXT"
        and payload[3] == 1
        and payload[5] == 0
    )


def _rollback(connection: sqlite3.Connection) -> None:
    try:
        connection.execute("ROLLBACK")
    except sqlite3.Error:
        return
