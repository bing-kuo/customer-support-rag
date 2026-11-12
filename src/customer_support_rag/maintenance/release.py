"""Immutable candidate releases and one active pointer.

A candidate is a new directory. Build never writes the active pointer.
Activation fsyncs the candidate first, then replaces ``active.json`` on the
same filesystem. A crash before that replace leaves the previous pointer.
The writer lock is ``flock`` in this process; it is not a distributed lock.

Same-bytes rebuild returns the existing directory and does not embed again.
Artifact keys do not cover vectors, so activation also checks the database
file hash stored in the manifest.
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
from pathlib import Path

from customer_support_rag.knowledge.embedding import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
)
from customer_support_rag.knowledge.parser import parse_document
from customer_support_rag.knowledge.storage import (
    CHUNKING_PROFILE,
    INPUT_PROFILE,
    EmbeddingStore,
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
from customer_support_rag.maintenance.mock_embed import MOCK_EMBED_DIGEST, MockEmbedder
from customer_support_rag.maintenance.paths import release_directory, require_directory

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
    embedder: MockEmbedder | None = None,
) -> BuildResult:
    """Build one complete mock candidate, or return the identical release."""
    with workspace_lock(workspace) as root:
        described = _approved_sources(root, knowledge_root, sources)
        identity, release_id = _identity(described)
        final = root / "releases" / release_id
        if final.exists():
            if _matches_existing(final, release_id, identity):
                return BuildResult(release_id, True, 0, 0)
            raise MaintenanceError("release already exists")
        selected = MockEmbedder() if embedder is None else embedder
        partial = root / "releases" / f".partial-{uuid.uuid4().hex}"
        partial.mkdir(parents=True)
        try:
            embedded, copied = _write_database(root, partial, described, selected)
            database_sha = sha256_file(partial / "knowledge.sqlite")
            manifest = _manifest(release_id, described, database_sha)
            _durable_bytes(
                partial / "manifest.json", human_json(manifest).encode("utf-8")
            )
            _fsync_directory(partial)
            final.parent.mkdir(parents=True, exist_ok=True)
            os.rename(partial, final)
            try:
                _fsync_directory(final.parent)
            except MaintenanceError:
                pass
        except MaintenanceError:
            shutil.rmtree(partial, ignore_errors=True)
            raise
        except Exception:  # noqa: BLE001 - remove a partial candidate after any failure
            shutil.rmtree(partial, ignore_errors=True)
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
    means the pointer must be absent. A mismatch does not write.
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
        _require_regression(directory, approval)
        snapshot = _verified_directory(
            root, directory, release_id, manifest_sha, database_sha
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
        _install_pointer(pointer, snapshot)
        return "activated"


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
    manifest_path = _regular_file(directory / "manifest.json")
    database_path = _regular_file(directory / "knowledge.sqlite")
    regression_path = _regular_file(directory / "regression.json")
    try:
        report = load_object(regression_path, regression_path.read_bytes())
    except OSError:
        raise MaintenanceError("regression is unavailable") from None
    code_sha = report.get("code_sha")
    suite_version = report.get("suite_version")
    if not isinstance(code_sha, str) or not isinstance(suite_version, str):
        raise MaintenanceError("regression is invalid")
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "database_sha256": sha256_file(database_path),
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


def _identity(described: tuple[DocumentDigest, ...]) -> tuple[dict[str, object], str]:
    documents = []
    for item in described:
        parsed = parse_document(item.text)
        key = document_embedding_key(
            parsed, model_digest=MOCK_EMBED_DIGEST, max_chars=_MAX_CHARS
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
        "transport": "mock",
        "embedding": _embedding_block(),
        "sources": documents,
    }
    digest = hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()
    return identity, digest


def _write_database(
    workspace: Path,
    partial: Path,
    described: tuple[DocumentDigest, ...],
    embedder: MockEmbedder,
) -> tuple[int, int]:
    database = partial / "knowledge.sqlite"
    embedded = 0
    copied = 0
    store = EmbeddingStore(database)
    try:
        for item in described:
            document = parse_document(item.text)
            key = document_embedding_key(document, model_digest=MOCK_EMBED_DIGEST)
            reusable = _reusable_embeddings(workspace, key, document.body)
            if reusable is not None:
                try:
                    store.put_document(
                        document, reusable, model_digest=MOCK_EMBED_DIGEST
                    )
                    copied += 1
                    continue
                except ValueError:
                    pass
            try:
                vectors = embedder.embed_document(document)
            except MaintenanceError:
                raise
            except Exception:  # noqa: BLE001 - embedder failures stay category-only
                raise MaintenanceError("embedding failed") from None
            store.put_document(document, vectors, model_digest=MOCK_EMBED_DIGEST)
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
    workspace: Path, artifact_key: str, body: str
) -> object | None:
    releases = workspace / "releases"
    if not releases.is_dir():
        return None
    for child in sorted(releases.iterdir()):
        if not child.is_dir() or child.is_symlink() or child.name.startswith("."):
            continue
        database = child / "knowledge.sqlite"
        manifest = child / "manifest.json"
        if (
            database.is_symlink()
            or manifest.is_symlink()
            or not database.is_file()
            or not manifest.is_file()
        ):
            continue
        store = EmbeddingStore.open_readonly(database)
        try:
            loaded = store.get_document(artifact_key)
        finally:
            store.close()
        if loaded is None or loaded.document.body != body:
            continue
        if loaded.model_digest != MOCK_EMBED_DIGEST:
            continue
        return loaded.embedded_chunks
    return None


def _manifest(
    release_id: str, described: tuple[DocumentDigest, ...], database_sha256: str
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
                    parsed, model_digest=MOCK_EMBED_DIGEST
                ),
            }
        )
    return {
        "schema": "knowledge-release-manifest",
        "schema_version": 1,
        "release_id": release_id,
        "transport": "mock",
        "embedding": _embedding_block(),
        "documents": documents,
        "database": {"relative_path": "knowledge.sqlite", "sha256": database_sha256},
    }


def _embedding_block() -> dict[str, object]:
    return {
        "model": EMBEDDING_MODEL,
        "model_digest": MOCK_EMBED_DIGEST,
        "dimensions": EMBEDDING_DIMENSIONS,
        "input_profile": INPUT_PROFILE,
        "chunking_profile": CHUNKING_PROFILE,
        "max_chars": _MAX_CHARS,
    }


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
    return actual == expected and manifest.get("transport") == "mock"


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
    if manifest.get("release_id") != release_id or manifest.get("transport") != "mock":
        raise MaintenanceError("active release is unavailable")
    database = manifest.get("database")
    documents = manifest.get("documents")
    embedding = manifest.get("embedding")
    if (
        not isinstance(database, dict)
        or database.get("relative_path") != "knowledge.sqlite"
        or database.get("sha256") != database_sha
        or not isinstance(documents, list)
        or not documents
        or not isinstance(embedding, dict)
        or embedding.get("model_digest") != MOCK_EMBED_DIGEST
    ):
        raise MaintenanceError("active release is unavailable")
    keys: list[str] = []
    for item in documents:
        if not isinstance(item, dict) or not isinstance(item.get("artifact_key"), str):
            raise MaintenanceError("active release is unavailable")
        keys.append(item["artifact_key"])
    if len(keys) != len(set(keys)):
        raise MaintenanceError("active release is unavailable")
    return ActiveSnapshot(
        release_id=release_id,
        database=database_path,
        artifact_keys=tuple(keys),
        model_digest=MOCK_EMBED_DIGEST,
        manifest_sha256=manifest_sha,
        database_sha256=database_sha,
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


def _require_regression(directory: Path, approval: dict[str, object]) -> None:
    from customer_support_rag.maintenance.evaluate import require_mock_regression

    path = _regular_file(directory / "regression.json")
    try:
        report = load_object(path, path.read_bytes())
    except OSError:
        raise MaintenanceError("regression is unavailable") from None
    if sha256_file(path) != approval.get("regression_sha256"):
        raise MaintenanceError("release approval does not match the candidate")
    if report.get("code_sha") != approval.get("code_sha"):
        raise MaintenanceError("release approval does not match the candidate")
    require_mock_regression(report)


def _install_pointer(pointer: Path, snapshot: ActiveSnapshot) -> None:
    if pointer.is_symlink():
        raise MaintenanceError("symlink is not allowed")
    payload = _pointer_bytes(
        snapshot.release_id, snapshot.manifest_sha256, snapshot.database_sha256
    )
    _durable_bytes(pointer, payload)
    _fsync_directory(pointer.parent)


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
