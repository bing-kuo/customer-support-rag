"""Consume Marcus's approval records. This module cannot create one.

The check is process isolation on one OS account: the record must already
exist, name reviewer ``marcus``, and match the file bytes. It is not a
cryptographic or OS-enforced signature. Test fixtures must stay in a temporary
workspace and are not demo approvals.
"""

import hashlib
from pathlib import Path

from customer_support_rag.maintenance.errors import MaintenanceError
from customer_support_rag.maintenance.hashes import describe_document, sha256_file
from customer_support_rag.maintenance.json_text import human_json, load_object
from customer_support_rag.maintenance.paths import require_directory

DOCUMENT_SCHEMA = "synthetic-document-approval"
RELEASE_SCHEMA = "synthetic-release-approval"
_SCHEMA_VERSION = 1
_REVIEWER = "marcus"
_DOCUMENT_FIELDS = (
    "schema",
    "schema_version",
    "document_id",
    "product_id",
    "version",
    "relative_path",
    "file_sha256",
    "body_sha256",
    "reviewer",
    "evidence",
)
_RELEASE_FIELDS = (
    "schema",
    "schema_version",
    "release_id",
    "manifest_sha256",
    "database_sha256",
    "regression_sha256",
    "code_sha",
    "suite_version",
    "reviewer",
    "evidence",
)


def consume_document_approval(
    workspace: Path, knowledge_root: Path, approval_path: Path
) -> Path:
    """Copy a matching document approval into the workspace control directory."""
    root = require_directory(workspace)
    record = _document_record(knowledge_root, approval_path)
    destination = _control_file(root, "document-approvals", record["file_sha256"])
    return _store(destination, record)


def consume_release_approval(workspace: Path, approval_path: Path) -> Path:
    """Copy a release approval whose hashes match the candidate files."""
    root = require_directory(workspace)
    record = _release_record(root, approval_path)
    destination = _control_file(root, "release-approvals", record["release_id"])
    return _store(destination, record)


def document_approval_for(workspace: Path, file_sha256: str) -> dict[str, object]:
    """Return the consumed document approval for the current file bytes."""
    path = _existing_control(workspace, "document-approvals", file_sha256)
    payload = _read_record(path)
    if set(payload) != set(_DOCUMENT_FIELDS) | {"approval_sha256"}:
        raise MaintenanceError("document approval is invalid")
    if (
        payload.get("file_sha256") != file_sha256
        or payload.get("reviewer") != _REVIEWER
    ):
        raise MaintenanceError("document approval is invalid")
    return payload


def release_approval_for(workspace: Path, release_id: str) -> dict[str, object]:
    """Return the consumed release approval."""
    path = _existing_control(workspace, "release-approvals", release_id)
    payload = _read_record(path)
    if set(payload) != set(_RELEASE_FIELDS) | {"approval_sha256"}:
        raise MaintenanceError("release approval is invalid")
    if payload.get("release_id") != release_id or payload.get("reviewer") != _REVIEWER:
        raise MaintenanceError("release approval is invalid")
    return payload


def _document_record(knowledge_root: Path, approval_path: Path) -> dict[str, object]:
    raw, payload = _approval_file(approval_path)
    _exact_fields(payload, _DOCUMENT_FIELDS)
    if payload.get("schema") != DOCUMENT_SCHEMA:
        raise MaintenanceError("document approval is invalid")
    _common_claims(payload)
    relative = payload.get("relative_path")
    if not isinstance(relative, str):
        raise MaintenanceError("document approval is invalid")
    described = describe_document(knowledge_root, relative)
    expected = {
        "document_id": described.document_id,
        "product_id": described.product_id,
        "version": described.version,
        "relative_path": described.relative_path,
        "file_sha256": described.file_sha256,
        "body_sha256": described.body_sha256,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise MaintenanceError("document approval does not match the file")
    record = {key: payload[key] for key in _DOCUMENT_FIELDS}
    record["approval_sha256"] = hashlib.sha256(raw).hexdigest()
    return record


def _release_record(workspace: Path, approval_path: Path) -> dict[str, object]:
    raw, payload = _approval_file(approval_path)
    _exact_fields(payload, _RELEASE_FIELDS)
    if payload.get("schema") != RELEASE_SCHEMA:
        raise MaintenanceError("release approval is invalid")
    _common_claims(payload)
    release_id = payload.get("release_id")
    if not isinstance(release_id, str):
        raise MaintenanceError("release approval is invalid")
    from customer_support_rag.maintenance.release import release_hashes

    hashes = release_hashes(workspace, release_id)
    compared = {
        "manifest_sha256": hashes["manifest_sha256"],
        "database_sha256": hashes["database_sha256"],
        "regression_sha256": hashes["regression_sha256"],
        "code_sha": hashes["code_sha"],
        "suite_version": hashes["suite_version"],
    }
    for key, value in compared.items():
        if payload.get(key) != value:
            raise MaintenanceError("release approval does not match the candidate")
    record = {key: payload[key] for key in _RELEASE_FIELDS}
    record["approval_sha256"] = hashlib.sha256(raw).hexdigest()
    return record


def _approval_file(path: Path) -> tuple[bytes, dict[str, object]]:
    if not isinstance(path, Path) or path.is_symlink() or not path.is_file():
        raise MaintenanceError("approval is unavailable")
    try:
        raw = path.read_bytes()
    except OSError:
        raise MaintenanceError("approval is unavailable") from None
    return raw, load_object(path, raw)


def _exact_fields(payload: dict[str, object], fields: tuple[str, ...]) -> None:
    if set(payload) != set(fields):
        raise MaintenanceError("approval is invalid")


def _common_claims(payload: dict[str, object]) -> None:
    if payload.get("schema_version") is not _SCHEMA_VERSION:
        raise MaintenanceError("approval is invalid")
    if payload.get("reviewer") != _REVIEWER:
        raise MaintenanceError("approval reviewer is not marcus")
    evidence = payload.get("evidence")
    if not isinstance(evidence, str) or not evidence.strip() or len(evidence) > 500:
        raise MaintenanceError("approval is invalid")
    code_sha = payload.get("code_sha")
    if code_sha is not None and (
        not isinstance(code_sha, str)
        or len(code_sha) != 40
        or any(character not in "0123456789abcdef" for character in code_sha)
    ):
        raise MaintenanceError("approval is invalid")


def _control_file(workspace: Path, kind: str, name: str) -> Path:
    if not isinstance(name, str) or len(name) not in {40, 64}:
        raise MaintenanceError("approval is invalid")
    if any(character not in "0123456789abcdef" for character in name):
        raise MaintenanceError("approval is invalid")
    directory = workspace / "control" / kind
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"
    if path.is_symlink():
        raise MaintenanceError("symlink is not allowed")
    return path


def _existing_control(workspace: Path, kind: str, name: str) -> Path:
    root = require_directory(workspace)
    path = root / "control" / kind / f"{name}.json"
    if path.is_symlink() or not path.is_file():
        raise MaintenanceError("approval is unavailable")
    try:
        path.resolve().relative_to(root)
    except (OSError, ValueError):
        raise MaintenanceError("path escapes the root") from None
    return path


def _store(destination: Path, record: dict[str, object]) -> Path:
    encoded = human_json(record).encode("utf-8")
    if destination.is_file():
        if destination.read_bytes() == encoded:
            return destination
        raise MaintenanceError("approval already exists")
    _write_durable(destination, encoded)
    return destination


def _read_record(path: Path) -> dict[str, object]:
    try:
        raw = path.read_bytes()
    except OSError:
        raise MaintenanceError("approval is unavailable") from None
    return load_object(path, raw)


def _write_durable(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.is_symlink() or path.is_symlink():
        raise MaintenanceError("symlink is not allowed")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            import os

            os.fsync(handle.fileno())
        temporary.replace(path)
    except OSError:
        raise MaintenanceError("approval could not be stored") from None
    finally:
        if temporary.exists() and temporary.is_file() and not temporary.is_symlink():
            temporary.unlink(missing_ok=True)


def stored_file_sha256(path: Path) -> str:
    """Hash one confined approval file."""
    return sha256_file(path)
