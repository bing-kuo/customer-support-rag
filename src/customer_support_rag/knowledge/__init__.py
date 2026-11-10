"""Knowledge document parsing, chunking, local embedding, storage, and ranking."""

from customer_support_rag.knowledge.chunker import chunk_document
from customer_support_rag.knowledge.embedding import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    MAX_CHUNKS_PER_REQUEST,
    MAX_EMBED_TEXT_CHARS,
    OLLAMA_EMBED_URL,
    OLLAMA_TAGS_URL,
    QUERY_PROFILE,
    EmbeddedChunk,
    EmbeddedQuery,
    EmbeddingError,
    OllamaEmbeddingAdapter,
    query_embedding_input,
)
from customer_support_rag.knowledge.models import (
    KnowledgeChunk,
    KnowledgeDocument,
    KnowledgeMetadata,
)
from customer_support_rag.knowledge.parser import parse_document
from customer_support_rag.knowledge.reader import read_document
from customer_support_rag.knowledge.retrieval import (
    ExactCosineIndex,
    RetrievalError,
    RetrievalHit,
)
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
    "OLLAMA_TAGS_URL",
    "QUERY_PROFILE",
    "EmbeddedChunk",
    "EmbeddedQuery",
    "EmbeddingError",
    "EmbeddingStore",
    "ExactCosineIndex",
    "KnowledgeChunk",
    "KnowledgeDocument",
    "KnowledgeMetadata",
    "OllamaEmbeddingAdapter",
    "RetrievalError",
    "RetrievalHit",
    "StorageConflictError",
    "StorageError",
    "StoredEmbeddingDocument",
    "chunk_document",
    "document_embedding_key",
    "parse_document",
    "query_embedding_input",
    "read_document",
]
