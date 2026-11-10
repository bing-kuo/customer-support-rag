"""Pure parser for TOML-front-matter Markdown knowledge documents."""

import tomllib

from pydantic import ValidationError

from customer_support_rag.knowledge.models import (
    KnowledgeDocument,
    KnowledgeMetadata,
    KnowledgeValidationError,
)


def parse_document(text: str) -> KnowledgeDocument:
    """Parse and validate a document without performing I/O or rendering it."""
    if not isinstance(text, str):
        raise KnowledgeValidationError("Document content must be text.")
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\r\n") != "+++":
        raise KnowledgeValidationError(
            "Document must start with a +++ front matter line."
        )

    closing_index = next(
        (
            index
            for index, line in enumerate(lines[1:], start=1)
            if line.rstrip("\r\n") == "+++"
        ),
        None,
    )
    if closing_index is None:
        raise KnowledgeValidationError("Document front matter is not closed.")

    try:
        raw_metadata = tomllib.loads("".join(lines[1:closing_index]))
    except tomllib.TOMLDecodeError as exc:
        raise KnowledgeValidationError(f"Invalid TOML front matter: {exc}") from exc

    body = "".join(lines[closing_index + 1 :])
    if not body.strip():
        raise KnowledgeValidationError("Document body must not be empty.")
    try:
        metadata = KnowledgeMetadata.model_validate(raw_metadata)
    except ValidationError as exc:
        raise KnowledgeValidationError(f"Invalid document metadata: {exc}") from exc
    return KnowledgeDocument(metadata=metadata, body=body)
