"""Immutable candidate releases and one active pointer.

A candidate is a new directory. Build never writes the active pointer.
Activation replaces ``active.json`` only after the previous pointer matches.
A failure before that replace leaves the previous pointer. If the replace
succeeds and the pointer directory fsync then fails, activation returns
``activated_unconfirmed``: the pointer already names the new release.
The writer lock is ``flock`` in this process; it is not a distributed lock.

Same-bytes rebuild returns the existing directory and does not embed again.
Artifact keys do not cover vectors. Copying them requires the same transport,
the full embedding identity, the source manifest, the database file hash, and
the document identity. A mock candidate also requires the mock digest and a
mock vector for every chunk. An Ollama candidate requires the published digest
and is never filled from a mock candidate. Snapshot, no-op, reuse, evaluation,
and activation reject a manifest whose model, digest, dimensions, document
profile, chunking profile, or query profile differs from that identity or from
the stored payload. A regression must repeat that same identity.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from customer_support_rag.knowledge.embedding import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    PUBLISHED_MODEL_DIGEST,
    QUERY_PROFILE,
)
from customer_support_rag.knowledge.parser import parse_document
from customer_support_rag.knowledge.storage import (
    CHUNKING_PROFILE,
    INPUT_PROFILE,
    EmbeddingStore,
    StorageError,
    document_embedding_key,
)
from customer_support_rag.maintenance.approval import document_approval_for
from customer_support_rag.maintenance.errors import MaintenanceError
from customer_support_rag.maintenance.hashes import (
    DocumentDigest,
    describe_document,
    sha256_file,
)
from customer_support_rag.maintenance.json_text import (
    canonical_json,
    human_json,
    load_object,
)
from customer_support_rag.maintenance.mock_embed import (
    MOCK_EMBED_DIGEST,
    MockEmbedder,
    OllamaEmbedder,
    mock_vector,
)
from customer_support_rag.maintenance.paths import (
    ensure_real_directory,
    release_directory,
    rename_within,
    require_directory,
)

_MAX_CHARS = 1200
_POINTER_NAME = "active.json"


@dataclass(frozen=True)
class BuildResult:
    release_id: str
    noop: bool
    embedded: int
    copied: int


@dataclass(frozen=True)
class ActiveSnapshot:
    """One verified release captured at the start of a request."""

    release_id: str
    database: Path
    artifact_keys: tuple[str, ...]
    model_digest: str
    manifest_sha256: str
    database_sha256: str
    transport: str


@contextmanager
def workspace_lock(workspace: Path) -> Iterator[Path]:
    """Serialize build and activation inside one workspace."""
    root = require_directory(workspace)
    handle_path = root / ".writer.lock"
    if handle_path.is_symlink():
        raise MaintenanceError("symlink is not allowed")
    handle = handle_path.open("a+b")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield root
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def build_candidate(
    workspace: Path,
    knowledge_root: Path,
    sources: tuple[str, ...] | list[str],
    embedder: object | None = None,
) -> BuildResult:
    """Build one complete candidate, or return the identical release.

    The default embedder is the offline mock. An ``OllamaEmbedder`` selects
    the published digest and does not reuse mock vectors.
    """
    with workspace_lock(workspace) as root:
        described = _approved_sources(root, knowledge_root, sources)
        selected = MockEmbedder() if embedder is None else embedder
        transport, model_digest = _profile(selected)
        identity, release_id = _identity(described, transport, model_digest)
        releases = ensure_real_directory(root, PurePosixPath("releases"))
        final = releases / release_id
        if final.is_symlink():
            raise MaintenanceError("symlink is not allowed")
        if final.exists():
            if _matches_existing(final, release_id, identity):
                return BuildResult(release_id, True, 0, 0)
            raise MaintenanceError("release already exists")
        partial_name = f".partial-{uuid.uuid4().hex}"
        partial = ensure_real_directory(root, PurePosixPath("releases") / partial_name)
        try:
            embedded, copied = _write_database(
                root, partial, described, selected, transport, model_digest
            )
            database_sha = sha256_file(partial / "knowledge.sqlite")
            manifest = _manifest(
                release_id, described, database_sha, transport, model_digest
            )
            _durable_bytes(
                partial / "manifest.json", human_json(manifest).encode("utf-8")
            )
            _fsync_directory(partial)
            if releases.is_symlink() or final.is_symlink():
                raise MaintenanceError("symlink is not allowed")
            rename_within(releases, partial_name, release_id)
            try:
                _fsync_directory(releases)
            except MaintenanceError:
                pass
        except MaintenanceError:
            _discard_partial(root, partial)
            raise
        except Exception:  # noqa: BLE001 - remove a partial candidate after any failure
            _discard_partial(root, partial)
            raise MaintenanceError("candidate could not be written") from None
        return BuildResult(release_id, False, embedded, copied)


def load_active_snapshot(workspace: Path) -> ActiveSnapshot:
    """Read the pointer once and verify the manifest and database bytes."""
    root = require_directory(workspace)
    pointer = root / _POINTER_NAME
    payload, pointer_bytes = _read_pointer(pointer)
    return _snapshot_from_pointer(root, payload, len(pointer_bytes))


def activate(
    workspace: Path,
    release_id: str,
    *,
    expected_previous: str | None,
) -> str:
    """Replace the active pointer, or return ``noop`` when it already matches.

    ``expected_previous`` is the SHA-256 of the current pointer bytes. ``None``
    means the pointer must be absent. A mismatch does not write. The return is
    ``activated`` when the pointer and its directory fsync both finish, or
    ``activated_unconfirmed`` when the pointer is already replaced and only the
    directory fsync failed. ``noop`` means the pointer already matched.
    """
    with workspace_lock(workspace) as root:
        approval = _activation_approval(root, release_id)
        directory = release_directory(root, release_id)
        manifest_path = _regular_file(directory / "manifest.json")
        database_path = _regular_file(directory / "knowledge.sqlite")
        manifest_sha = sha256_file(manifest_path)
        database_sha = sha256_file(database_path)
        if (
            approval["manifest_sha256"] != manifest_sha
            or approval["database_sha256"] != database_sha
        ):
            raise MaintenanceError("release approval does not match the candidate")
        snapshot = _verified_directory(
            root, directory, release_id, manifest_sha, database_sha
        )
        _require_regression(
            directory,
            approval,
            release_id=release_id,
            manifest_sha256=manifest_sha,
            database_sha256=database_sha,
            transport=snapshot.transport,
            model_digest=snapshot.model_digest,
        )
        pointer = root / _POINTER_NAME
        current = _pointer_sha(pointer)
        target = _pointer_digest(release_id, manifest_sha, database_sha)
        if current == target:
            if expected_previous != current:
                raise MaintenanceError("active pointer does not match")
            _open_keys(snapshot)
            return "noop"
        if expected_previous != current:
            raise MaintenanceError("active pointer does not match")
        _open_keys(snapshot)
        return _install_pointer(pointer, snapshot)


def candidate_snapshot(workspace: Path, release_id: str) -> ActiveSnapshot:
    """Verify one candidate directory without reading the active pointer."""
    root = require_directory(workspace)
    directory = release_directory(root, release_id)
    manifest_path = _regular_file(directory / "manifest.json")
    database_path = _regular_file(directory / "knowledge.sqlite")
    return _verified_directory(
        root,
        directory,
        release_id,
        sha256_file(manifest_path),
        sha256_file(database_path),
    )


def release_hashes(workspace: Path, release_id: str) -> dict[str, str]:
    """Return the candidate hashes a release approval must repeat."""
    root = require_directory(workspace)
    directory = release_directory(root, release_id)
    regression_path = _regular_file(directory / "regression.json")
    try:
        report = load_object(regression_path, regression_path.read_bytes())
    except OSError:
        raise MaintenanceError("regression is unavailable") from None
    snapshot = candidate_snapshot(workspace, release_id)
    manifest_sha256 = snapshot.manifest_sha256
    database_sha256 = snapshot.database_sha256
    from customer_support_rag.maintenance.evaluate import regression_matches_candidate

    if not regression_matches_candidate(
        report,
        release_id=release_id,
        manifest_sha256=manifest_sha256,
        database_sha256=database_sha256,
        transport=snapshot.transport,
        model_digest=snapshot.model_digest,
    ):
        raise MaintenanceError("regression does not match the candidate")
    code_sha = report.get("code_sha")
    suite_version = report.get("suite_version")
    if not isinstance(code_sha, str) or not isinstance(suite_version, str):
        raise MaintenanceError("regression is invalid")
    return {
        "manifest_sha256": manifest_sha256,
        "database_sha256": database_sha256,
        "regression_sha256": sha256_file(regression_path),
        "code_sha": code_sha,
        "suite_version": suite_version,
    }


def ensure_unique_documents(
    described: tuple[DocumentDigest, ...] | list[DocumentDigest],
) -> None:
    """Reject a source list that repeats a document id."""
    seen: set[str] = set()
    for item in described:
        if item.document_id in seen:
            raise MaintenanceError("document version conflict")
        seen.add(item.document_id)


def _approved_sources(
    workspace: Path,
    knowledge_root: Path,
    sources: tuple[str, ...] | list[str],
) -> tuple[DocumentDigest, ...]:
    if isinstance(sources, (str, bytes)) or not isinstance(sources, (tuple, list)):
        raise MaintenanceError("source list is invalid")
    if not sources or len(sources) != len(set(sources)):
        raise MaintenanceError("source list is invalid")
    described = tuple(describe_document(knowledge_root, source) for source in sources)
    ensure_unique_documents(described)
    for item in described:
        approval = document_approval_for(workspace, item.file_sha256)
        compared = (
            "document_id",
            "product_id",
            "version",
            "relative_path",
            "body_sha256",
        )
        for field in compared:
            if approval.get(field) != getattr(item, field):
                raise MaintenanceError("document approval does not match the file")
    return tuple(sorted(described, key=lambda item: item.relative_path))


def _profile(embedder: object) -> tuple[str, str]:
    """Return the transport name and digest selected by this embedder."""
    if isinstance(embedder, OllamaEmbedder):
        return "ollama", PUBLISHED_MODEL_DIGEST
    return "mock", MOCK_EMBED_DIGEST


def _identity(
    described: tuple[DocumentDigest, ...], transport: str, model_digest: str
) -> tuple[dict[str, object], str]:
    documents = []
    for item in described:
        parsed = parse_document(item.text)
        key = document_embedding_key(
            parsed, model_digest=model_digest, max_chars=_MAX_CHARS
        )
        documents.append(
            {
                "relative_path": item.relative_path,
                "file_sha256": item.file_sha256,
                "body_sha256": item.body_sha256,
                "document_id": item.document_id,
                "product_id": item.product_id,
                "version": item.version,
                "artifact_key": key,
            }
        )
    identity: dict[str, object] = {
        "schema": "knowledge-release-identity",
        "schema_version": 1,
        "transport": transport,
        "embedding": _embedding_block(model_digest),
        "sources": documents,
    }
    digest = hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()
    return identity, digest


def _write_database(
    workspace: Path,
    partial: Path,
    described: tuple[DocumentDigest, ...],
    embedder: object,
    transport: str,
    model_digest: str,
) -> tuple[int, int]:
    database = partial / "knowledge.sqlite"
    embedded = 0
    copied = 0
    store = EmbeddingStore(database)
    try:
        for item in described:
            document = parse_document(item.text)
            key = document_embedding_key(document, model_digest=model_digest)
            reusable = _reusable_embeddings(
                workspace, key, document.body, transport, model_digest
            )
            if reusable is not None:
                try:
                    store.put_document(document, reusable, model_digest=model_digest)
                    copied += 1
                    continue
                except ValueError:
                    pass
            try:
                vectors = embedder.embed_document(document)  # type: ignore[attr-defined]
            except MaintenanceError:
                raise
            except Exception:  # noqa: BLE001 - embedder failures stay category-only
                raise MaintenanceError("embedding failed") from None
            store.put_document(document, vectors, model_digest=model_digest)
            embedded += 1
    finally:
        store.close()
    for suffix in ("-journal", "-wal", "-shm"):
        leftover = Path(str(database) + suffix)
        if leftover.exists():
            raise MaintenanceError("candidate could not be written")
    _durable_bytes(database, database.read_bytes())
    return embedded, copied


def _reusable_embeddings(
    workspace: Path,
    artifact_key: str,
    body: str,
    transport: str,
    model_digest: str,
) -> object | None:
    releases = workspace / "releases"
    if releases.is_symlink() or not releases.is_dir():
        return None
    try:
        releases.resolve().relative_to(workspace)
    except (OSError, ValueError):
        return None
    for child in sorted(releases.iterdir()):
        if child.is_symlink() or not child.is_dir() or child.name.startswith("."):
            continue
        reusable = _embeddings_from_release(
            child, artifact_key, body, transport, model_digest
        )
        if reusable is not None:
            return reusable
    return None


def _embeddings_from_release(
    directory: Path,
    artifact_key: str,
    body: str,
    transport: str,
    model_digest: str,
) -> object | None:
    database = directory / "knowledge.sqlite"
    manifest_path = directory / "manifest.json"
    if (
        database.is_symlink()
        or manifest_path.is_symlink()
        or not database.is_file()
        or not manifest_path.is_file()
    ):
        return None
    try:
        manifest = load_object(manifest_path, manifest_path.read_bytes())
    except (OSError, MaintenanceError):
        return None
    embedding = manifest.get("embedding")
    database_record = manifest.get("database")
    documents = manifest.get("documents")
    if (
        manifest.get("transport") != transport
        or not isinstance(embedding, dict)
        or embedding != _embedding_block(model_digest)
        or not _digest_matches_transport(transport, model_digest)
        or not isinstance(database_record, dict)
        or database_record.get("sha256") != sha256_file(database)
        or not isinstance(documents, list)
    ):
        return None
    identity = next(
        (
            item
            for item in documents
            if isinstance(item, dict) and item.get("artifact_key") == artifact_key
        ),
        None,
    )
    if not isinstance(identity, dict):
        return None
    if identity.get("body_sha256") != hashlib.sha256(body.encode("utf-8")).hexdigest():
        return None
    try:
        store = EmbeddingStore.open_readonly(database)
    except (OSError, StorageError):
        return None
    try:
        loaded = store.get_document(artifact_key)
    except (OSError, StorageError):
        return None
    finally:
        store.close()
    if loaded is None or loaded.document.body != body or not loaded.embedded_chunks:
        return None
    if not _payload_matches_identity(loaded, identity, _embedding_block(model_digest)):
        return None
    if transport == "mock":
        for item in loaded.embedded_chunks:
            if item.vector != tuple(mock_vector(item.chunk.text)):
                return None
    return loaded.embedded_chunks


def _manifest(
    release_id: str,
    described: tuple[DocumentDigest, ...],
    database_sha256: str,
    transport: str,
    model_digest: str,
) -> dict[str, object]:
    documents = []
    for item in described:
        parsed = parse_document(item.text)
        documents.append(
            {
                "relative_path": item.relative_path,
                "document_id": item.document_id,
                "product_id": item.product_id,
                "version": item.version,
                "title": item.title,
                "file_sha256": item.file_sha256,
                "body_sha256": item.body_sha256,
                "artifact_key": document_embedding_key(
                    parsed, model_digest=model_digest
                ),
            }
        )
    return {
        "schema": "knowledge-release-manifest",
        "schema_version": 1,
        "release_id": release_id,
        "transport": transport,
        "embedding": _embedding_block(model_digest),
        "documents": documents,
        "database": {"relative_path": "knowledge.sqlite", "sha256": database_sha256},
    }


def embedding_identity(model_digest: str) -> dict[str, object]:
    """Return the only embedding identity a candidate may record."""
    return _embedding_block(model_digest)


def _embedding_block(model_digest: str) -> dict[str, object]:
    return {
        "model": EMBEDDING_MODEL,
        "model_digest": model_digest,
        "dimensions": EMBEDDING_DIMENSIONS,
        "input_profile": INPUT_PROFILE,
        "chunking_profile": CHUNKING_PROFILE,
        "max_chars": _MAX_CHARS,
        "query_profile": QUERY_PROFILE,
    }


def _digest_matches_transport(transport: str, model_digest: str) -> bool:
    if transport == "mock":
        return model_digest == MOCK_EMBED_DIGEST
    if transport == "ollama":
        return model_digest == PUBLISHED_MODEL_DIGEST
    return False


def _matches_existing(
    directory: Path, release_id: str, identity: dict[str, object]
) -> bool:
    if directory.is_symlink() or not directory.is_dir():
        return False
    manifest_path = directory / "manifest.json"
    database_path = directory / "knowledge.sqlite"
    if manifest_path.is_symlink() or database_path.is_symlink():
        return False
    if not manifest_path.is_file() or not database_path.is_file():
        return False
    try:
        manifest = load_object(manifest_path, manifest_path.read_bytes())
    except (OSError, MaintenanceError):
        return False
    sources = identity["sources"]
    if not isinstance(sources, list) or manifest.get("release_id") != release_id:
        return False
    documents = manifest.get("documents")
    database = manifest.get("database")
    if not isinstance(documents, list) or not isinstance(database, dict):
        return False
    if database.get("sha256") != sha256_file(database_path):
        return False
    actual = {
        (item.get("relative_path"), item.get("file_sha256"), item.get("artifact_key"))
        for item in documents
        if isinstance(item, dict)
    }
    expected = {
        (item["relative_path"], item["file_sha256"], item["artifact_key"])
        for item in sources
        if isinstance(item, dict)
    }
    embedding = identity.get("embedding")
    if not (
        actual == expected
        and manifest.get("transport") == identity.get("transport")
        and isinstance(embedding, dict)
        and manifest.get("embedding") == embedding
        and _digest_matches_transport(
            str(identity.get("transport")), str(embedding.get("model_digest"))
        )
        and all(isinstance(item, dict) for item in documents)
    ):
        return False
    digest = embedding.get("model_digest")
    if not isinstance(digest, str):
        return False
    try:
        _require_stored_identity(database_path, documents, digest)
    except MaintenanceError:
        return False
    return True


def _read_pointer(pointer: Path) -> tuple[dict[str, object], bytes]:
    if pointer.is_symlink() or not pointer.is_file():
        raise MaintenanceError("active release is unavailable")
    try:
        raw = pointer.read_bytes()
    except OSError:
        raise MaintenanceError("active release is unavailable") from None
    payload = load_object(pointer, raw)
    return payload, raw


def _snapshot_from_pointer(
    root: Path, payload: dict[str, object], _size: int
) -> ActiveSnapshot:
    if (
        payload.get("schema") != "knowledge-active-pointer"
        or payload.get("schema_version") != 1
    ):
        raise MaintenanceError("active release is unavailable")
    release_id = payload.get("release_id")
    manifest_sha = payload.get("manifest_sha256")
    database_sha = payload.get("database_sha256")
    if not all(
        isinstance(item, str) for item in (release_id, manifest_sha, database_sha)
    ):
        raise MaintenanceError("active release is unavailable")
    assert isinstance(release_id, str)
    assert isinstance(manifest_sha, str)
    assert isinstance(database_sha, str)
    try:
        directory = release_directory(root, release_id)
    except MaintenanceError:
        raise MaintenanceError("active release is unavailable") from None
    return _verified_directory(root, directory, release_id, manifest_sha, database_sha)


def _verified_directory(
    root: Path,
    directory: Path,
    release_id: str,
    manifest_sha: str,
    database_sha: str,
) -> ActiveSnapshot:
    del root
    manifest_path = _regular_file(directory / "manifest.json")
    database_path = _regular_file(directory / "knowledge.sqlite")
    if (
        sha256_file(manifest_path) != manifest_sha
        or sha256_file(database_path) != database_sha
    ):
        raise MaintenanceError("active release is unavailable")
    try:
        manifest = load_object(manifest_path, manifest_path.read_bytes())
    except OSError:
        raise MaintenanceError("active release is unavailable") from None
    transport = manifest.get("transport")
    database = manifest.get("database")
    documents = manifest.get("documents")
    embedding = manifest.get("embedding")
    digest = embedding.get("model_digest") if isinstance(embedding, dict) else None
    if (
        manifest.get("release_id") != release_id
        or not isinstance(transport, str)
        or not isinstance(digest, str)
        or not _digest_matches_transport(transport, digest)
        or not isinstance(database, dict)
        or database.get("relative_path") != "knowledge.sqlite"
        or database.get("sha256") != database_sha
        or not isinstance(documents, list)
        or not documents
        or embedding != _embedding_block(digest)
    ):
        raise MaintenanceError("active release is unavailable")
    keys: list[str] = []
    stored_documents: list[dict[str, object]] = []
    for item in documents:
        if not isinstance(item, dict) or not isinstance(item.get("artifact_key"), str):
            raise MaintenanceError("active release is unavailable")
        keys.append(item["artifact_key"])
        stored_documents.append(item)
    if len(keys) != len(set(keys)):
        raise MaintenanceError("active release is unavailable")
    _require_stored_identity(database_path, stored_documents, digest)
    return ActiveSnapshot(
        release_id=release_id,
        database=database_path,
        artifact_keys=tuple(keys),
        model_digest=digest,
        manifest_sha256=manifest_sha,
        database_sha256=database_sha,
        transport=transport,
    )


def _require_stored_identity(
    database_path: Path,
    documents: list[dict[str, object]],
    model_digest: str,
) -> None:
    """Reject a manifest that does not match the stored payload identity."""
    expected = _embedding_block(model_digest)
    try:
        store = EmbeddingStore.open_readonly(database_path)
    except (OSError, StorageError):
        raise MaintenanceError("active release is unavailable") from None
    try:
        for item in documents:
            key = item.get("artifact_key")
            if not isinstance(key, str):
                raise MaintenanceError("active release is unavailable")
            try:
                loaded = store.get_document(key)
            except (OSError, StorageError):
                raise MaintenanceError("active release is unavailable") from None
            if loaded is None or not _payload_matches_identity(loaded, item, expected):
                raise MaintenanceError("active release is unavailable")
    finally:
        store.close()


def _payload_matches_identity(
    loaded: object,
    item: dict[str, object],
    expected: dict[str, object],
) -> bool:
    from customer_support_rag.knowledge.storage import StoredEmbeddingDocument

    if not isinstance(loaded, StoredEmbeddingDocument):
        return False
    metadata = loaded.document.metadata
    body_hash = hashlib.sha256(loaded.document.body.encode("utf-8")).hexdigest()
    return (
        loaded.embedding_model == expected["model"]
        and loaded.model_digest == expected["model_digest"]
        and loaded.dimensions == expected["dimensions"]
        and loaded.input_profile == expected["input_profile"]
        and loaded.chunking_profile == expected["chunking_profile"]
        and loaded.max_chars == expected["max_chars"]
        and item.get("document_id") == metadata.document_id
        and item.get("product_id") == metadata.product_id
        and item.get("version") == metadata.version
        and item.get("title") == metadata.title
        and item.get("body_sha256") == body_hash
    )


def _open_keys(snapshot: ActiveSnapshot) -> None:
    store = EmbeddingStore.open_readonly(snapshot.database)
    try:
        for key in snapshot.artifact_keys:
            if store.get_document(key) is None:
                raise MaintenanceError("active release is unavailable")
    finally:
        store.close()


def _activation_approval(workspace: Path, release_id: str) -> dict[str, object]:
    from customer_support_rag.maintenance.approval import release_approval_for

    try:
        return release_approval_for(workspace, release_id)
    except MaintenanceError:
        raise MaintenanceError("release approval is unavailable") from None


def _pointer_sha(pointer: Path) -> str | None:
    if not pointer.exists():
        return None
    if pointer.is_symlink() or not pointer.is_file():
        raise MaintenanceError("active release is unavailable")
    return sha256_file(pointer)


def _pointer_digest(release_id: str, manifest_sha: str, database_sha: str) -> str:
    raw = _pointer_bytes(release_id, manifest_sha, database_sha)
    return hashlib.sha256(raw).hexdigest()


def _pointer_bytes(release_id: str, manifest_sha: str, database_sha: str) -> bytes:
    payload = {
        "schema": "knowledge-active-pointer",
        "schema_version": 1,
        "release_id": release_id,
        "manifest_sha256": manifest_sha,
        "database_sha256": database_sha,
    }
    return human_json(payload).encode("utf-8")


def _require_regression(
    directory: Path,
    approval: dict[str, object],
    *,
    release_id: str,
    manifest_sha256: str,
    database_sha256: str,
    transport: str,
    model_digest: str,
) -> None:
    from customer_support_rag.maintenance.evaluate import (
        regression_matches_candidate,
        require_mock_regression,
    )

    path = _regular_file(directory / "regression.json")
    try:
        report = load_object(path, path.read_bytes())
    except OSError:
        raise MaintenanceError("regression is unavailable") from None
    if not regression_matches_candidate(
        report,
        release_id=release_id,
        manifest_sha256=manifest_sha256,
        database_sha256=database_sha256,
        transport=transport,
        model_digest=model_digest,
    ):
        raise MaintenanceError("regression does not match the candidate")
    if sha256_file(path) != approval.get("regression_sha256"):
        raise MaintenanceError("release approval does not match the candidate")
    if report.get("code_sha") != approval.get("code_sha"):
        raise MaintenanceError("release approval does not match the candidate")
    require_mock_regression(report)


def _install_pointer(pointer: Path, snapshot: ActiveSnapshot) -> str:
    if pointer.is_symlink():
        raise MaintenanceError("symlink is not allowed")
    payload = _pointer_bytes(
        snapshot.release_id, snapshot.manifest_sha256, snapshot.database_sha256
    )
    _durable_bytes(pointer, payload)
    try:
        _fsync_directory(pointer.parent)
    except MaintenanceError:
        return "activated_unconfirmed"
    return "activated"


def _discard_partial(root: Path, partial: Path) -> None:
    if partial.is_symlink() or not partial.exists() or not partial.is_dir():
        return
    if not partial.name.startswith(".partial-"):
        return
    try:
        partial.resolve().relative_to(root)
    except (OSError, ValueError):
        return
    shutil.rmtree(partial)


def _regular_file(path: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise MaintenanceError("release is unavailable")
    return path


def _durable_bytes(path: Path, payload: bytes) -> None:
    """Write bytes, fsync them, then replace the destination.

    The destination is unchanged until ``os.replace``. A crash before replace
    leaves the previous file, or no file if this is the first write.
    """
    if path.is_symlink():
        raise MaintenanceError("symlink is not allowed")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    except OSError:
        raise MaintenanceError("candidate could not be written") from None
    finally:
        os.close(descriptor)
    try:
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise MaintenanceError("candidate could not be written") from None


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    except OSError:
        raise MaintenanceError("candidate could not be written") from None
    finally:
        os.close(descriptor)
