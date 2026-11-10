"""Validated data models for synthetic knowledge documents."""

from pydantic import BaseModel, ConfigDict, StrictBool, StrictStr, field_validator


class KnowledgeValidationError(ValueError):
    """Raised when knowledge metadata or document content is invalid."""


class KnowledgeMetadata(BaseModel):
    """Metadata required to identify a knowledge document."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    document_id: StrictStr
    product_id: StrictStr
    title: StrictStr
    version: StrictStr
    synthetic: StrictBool

    @field_validator("document_id", "product_id", "title", "version")
    @classmethod
    def require_non_empty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value

    @field_validator("product_id")
    @classmethod
    def require_supported_product(cls, value: str) -> str:
        if value not in {"hp-a", "hp-b"}:
            raise ValueError("must be one of: hp-a, hp-b")
        return value

    @field_validator("synthetic")
    @classmethod
    def require_synthetic_data(cls, value: bool) -> bool:
        if value is not True:
            raise ValueError("must be true")
        return value


class KnowledgeDocument(BaseModel):
    """A validated metadata record and its unmodified Markdown body."""

    model_config = ConfigDict(frozen=True)

    metadata: KnowledgeMetadata
    body: str
