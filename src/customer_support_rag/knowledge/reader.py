"""Root-bounded reader for local Markdown knowledge documents."""

from pathlib import Path

from customer_support_rag.knowledge.models import KnowledgeDocument
from customer_support_rag.knowledge.parser import parse_document


def read_document(root: str | Path, relative_path: str | Path) -> KnowledgeDocument:
    """Read one Markdown file contained by root and parse it as plain data."""
    root_path = Path(root).resolve(strict=True)
    candidate = Path(relative_path)
    if candidate.is_absolute() or candidate.as_posix().startswith("/"):
        raise ValueError("Document path must be relative to the knowledge root.")
    if (
        candidate.suffix.lower() != ".md"
        or ".." in candidate.parts
        or candidate.as_posix().startswith("http")
    ):
        raise ValueError(
            "Document path must identify a local Markdown file under the root."
        )

    resolved = (root_path / candidate).resolve(strict=True)
    try:
        resolved.relative_to(root_path)
    except ValueError as exc:
        raise ValueError("Document path escapes the knowledge root.") from exc
    if not resolved.is_file():
        raise ValueError("Document path must identify a regular file.")
    with resolved.open("r", encoding="utf-8", newline="") as document_file:
        text = document_file.read()
    return parse_document(text)
