"""Knowledge document parsing, chunking, local embedding, and storage."""

from customer_support_rag.knowledge.chunker import chunk_document
from customer_support_rag.knowledge.embedding import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    MAX_CHUNKS_PER_REQUEST,
    MAX_EMBED_TEXT_CHARS,
    OLLAMA_EMBED_URL,
    EmbeddedChunk,
    EmbeddingError,
    OllamaEmbeddingAdapter,
)
from customer_support_rag.knowledge.models import (
    KnowledgeChunk,
    KnowledgeDocument,
    KnowledgeMetadata,
)
from customer_support_rag.knowledge.parser import parse_document
from customer_support_rag.knowledge.reader import read_document
from customer_support_rag.knowledge.storage import (
    EmbeddingStore,
    StorageConflictError,
    StorageError,
    StoredEmbeddingDocument,
    document_embedding_key,
)

__all__ = [
    "EMBEDDING_DIMENSIONS",
    "EMBEDDING_MODEL",
    "MAX_CHUNKS_PER_REQUEST",
    "MAX_EMBED_TEXT_CHARS",
    "OLLAMA_EMBED_URL",
    "EmbeddedChunk",
    "EmbeddingError",
    "EmbeddingStore",
    "KnowledgeChunk",
    "KnowledgeDocument",
    "KnowledgeMetadata",
    "OllamaEmbeddingAdapter",
    "StorageConflictError",
    "StorageError",
    "StoredEmbeddingDocument",
    "chunk_document",
    "document_embedding_key",
    "parse_document",
    "read_document",
]
