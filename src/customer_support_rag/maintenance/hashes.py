"""File and body hashes for synthetic knowledge documents."""

import hashlib
from pathlib import Path

from customer_support_rag.knowledge.models import KnowledgeValidationError
from customer_support_rag.knowledge.parser import parse_document
from customer_support_rag.maintenance.errors import MaintenanceError
from customer_support_rag.maintenance.paths import markdown_file, require_directory

CURRENT_SOURCES = (
    "hp-a/headphone-care.md",
    "hp-b/connection-troubleshooting.md",
)
_MAX_DOCUMENT_BYTES = 64 * 1024


class DocumentDigest:
    """Hashes and identity read from one Markdown file."""

    def __init__(
        self,
        relative_path: str,
        file_sha256: str,
        body_sha256: str,
        document_id: str,
        product_id: str,
        version: str,
        title: str,
        text: str,
    ) -> None:
        self.relative_path = relative_path
        self.file_sha256 = file_sha256
        self.body_sha256 = body_sha256
        self.document_id = document_id
        self.product_id = product_id
        self.version = version
        self.title = title
        self.text = text

    def as_dict(self) -> dict[str, str]:
        return {
            "relative_path": self.relative_path,
            "file_sha256": self.file_sha256,
            "body_sha256": self.body_sha256,
            "document_id": self.document_id,
            "product_id": self.product_id,
            "version": self.version,
            "title": self.title,
        }


def describe_document(root: Path, relative_path: str) -> DocumentDigest:
    """Hash the raw file and the parsed body. This does not approve it."""
    directory = require_directory(root)
    path = markdown_file(directory, relative_path)
    raw = _read_limited(path)
    try:
        text = raw.decode("utf-8")
        document = parse_document(text)
    except (UnicodeDecodeError, KnowledgeValidationError):
        raise MaintenanceError("document is invalid") from None
    metadata = document.metadata
    return DocumentDigest(
        relative_path=relative_path,
        file_sha256=hashlib.sha256(raw).hexdigest(),
        body_sha256=hashlib.sha256(document.body.encode("utf-8")).hexdigest(),
        document_id=metadata.document_id,
        product_id=metadata.product_id,
        version=metadata.version,
        title=metadata.title,
        text=text,
    )


def scan_current_sources(root: Path) -> tuple[DocumentDigest, ...]:
    """Describe the fixed current sources. Drafts are not included."""
    return tuple(describe_document(root, path) for path in CURRENT_SOURCES)


def sha256_file(path: Path) -> str:
    """Hash a regular file that has already been confined."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_limited(path: Path) -> bytes:
    try:
        size = path.stat().st_size
    except OSError:
        raise MaintenanceError("document is unavailable") from None
    if size > _MAX_DOCUMENT_BYTES:
        raise MaintenanceError("document is invalid")
    try:
        return path.read_bytes()
    except OSError:
        raise MaintenanceError("document is unavailable") from None
