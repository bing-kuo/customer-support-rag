"""One stateless question over stored embeddings and one Luna call.

Importing this module does not open a database, read a key, or contact a
model. The caller supplies an existing database, explicit artifact keys, the
fixed digest, and the project root that holds ``.env``.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict

from customer_support_rag.knowledge.answering import (
    MAX_EVIDENCE_CHARS,
    MAX_HITS,
    AnswerPreparationError,
    AnswerRequest,
    GroundedDraft,
    ResolvedSource,
    _require_one_version,
    handoff_without_evidence,
    prepare_answer,
)
from customer_support_rag.knowledge.embedding import (
    PUBLISHED_MODEL_DIGEST,
    EmbeddingError,
    OllamaEmbeddingAdapter,
)
from customer_support_rag.knowledge.retrieval import (
    ExactCosineIndex,
    RetrievalError,
    RetrievalHit,
)
from customer_support_rag.knowledge.storage import EmbeddingStore, StorageError
from customer_support_rag.luna import LunaError, Usage
from customer_support_rag.luna_live import post_prepared_answer
from customer_support_rag.maintenance.errors import MaintenanceError
from customer_support_rag.maintenance.release import load_active_snapshot

PUBLISHED_DIGEST = PUBLISHED_MODEL_DIGEST
HP_A_CARE_ARTIFACT = "9a2283a73584261d31460d1b46855145d384c756834c8984c1539e96647cb4ad"
HP_B_CONNECTION_ARTIFACT = (
    "18812b328a8e3614c4c83a849c2bfcd108e695453c8eaf9d57cf87200cefafae"
)
EXAMPLE_ARTIFACT_KEYS = (HP_A_CARE_ARTIFACT, HP_B_CONNECTION_ARTIFACT)
DATABASE_ENV = "CUSTOMER_SUPPORT_RAG_DATABASE"
KEY_ROOT_ENV = "CUSTOMER_SUPPORT_RAG_KEY_ROOT"
_TOP_K = 4

_STATUS = {
    "ask_not_configured": 503,
    "dependency_unavailable": 503,
    "evidence_unavailable": 503,
    "paid_blocked": 503,
    "wire_body_exceeded": 503,
    "policy_rejected": 503,
    "runtime_unavailable": 503,
    "embedding_unavailable": 502,
    "insufficient_quota": 429,
    "rate_limit": 429,
    "provider_capacity": 429,
    "logical_timeout": 504,
    "provider_transport_error": 504,
    "provider_authentication_or_access_unavailable": 502,
    "provider_model_or_endpoint_unavailable": 502,
    "provider_service_unavailable": 502,
    "invalid_response": 502,
    "invalid_model_schema": 502,
    "unknown_citation": 502,
    "output_refusal": 502,
    "unverified_returned_model": 502,
    "unverified_service_tier": 502,
    "unknown_response_terminal_state": 502,
    "contradictory_response": 502,
    "provider_incomplete": 502,
    "provider_failed": 502,
    "provider_cancelled": 502,
    "provider_unavailable": 502,
}


class AskFailure(Exception):
    """A safe API failure. ``code`` is a fixed English token, not a message."""

    def __init__(self, status_code: int, code: str) -> None:
        super().__init__(code)
        self.status_code = status_code
        self.code = code


class UsageCounts(BaseModel):
    """Known token counts. A null field was omitted; it is not zero."""

    model_config = ConfigDict(extra="forbid")

    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_tokens: int | None
    cache_write_tokens: int | None
    ttl_30m_tokens: int | None
    ttl_12h_tokens: int | None
    reasoning_tokens: int | None


class CitedSource(BaseModel):
    """One cited local chunk. The text and offsets come from the stored hit."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    artifact_key: str
    document_id: str
    product_id: Literal["hp-a", "hp-b"]
    version: str
    title: str
    chunk_index: int
    heading: str
    char_start: int
    char_end: int
    text: str


class AskResponse(BaseModel):
    """One answer or handoff. ``usage`` is null when the counts were not known.

    ``usage_code`` is also null when no model call was made. An unknown billing
    shape keeps the answer, leaves ``usage`` null, and sets ``usage_code``.
    """

    model_config = ConfigDict(extra="forbid")

    request_id: str
    status: Literal["answer", "handoff"]
    answer: str | None
    reason: Literal["no_evidence", "conflicting_evidence", "refund_request"] | None
    sources: list[CitedSource]
    usage: UsageCounts | None
    usage_code: str | None
    usage_fields: list[str]


@dataclass(frozen=True)
class AskConfig:
    """Trusted process settings. A request body cannot replace these fields."""

    database: Path
    artifact_keys: tuple[str, ...]
    model_digest: str
    key_root: Path
    embedding_transport: httpx.MockTransport | None = None
    release_root: Path | None = None


@dataclass(frozen=True)
class _Prepared:
    draft: GroundedDraft | None = None
    request: AnswerRequest | None = None


def config_from_environment() -> AskConfig | None:
    """Read process paths. This does not open a database or a release."""
    database = os.environ.get(DATABASE_ENV)
    key_root = os.environ.get(KEY_ROOT_ENV)
    release = os.environ.get("CUSTOMER_SUPPORT_RAG_RELEASE_ROOT")
    if release:
        if not key_root:
            return None
        return AskConfig(
            database=Path(database or release),
            artifact_keys=(),
            model_digest=PUBLISHED_DIGEST,
            key_root=Path(key_root),
            release_root=Path(release),
        )
    if not database or not key_root:
        return None
    return AskConfig(
        database=Path(database),
        artifact_keys=EXAMPLE_ARTIFACT_KEYS,
        model_digest=PUBLISHED_DIGEST,
        key_root=Path(key_root),
    )


async def answer_ask(
    config: AskConfig | None,
    product_id: str,
    question: str,
    request_id: str,
) -> AskResponse:
    """Retrieve, then send at most one Luna request.

    An empty product candidate list returns ``no_evidence`` and does not read
    the key. Dependency failures are not reported as no evidence.
    """
    if config is None:
        raise AskFailure(503, "ask_not_configured")
    try:
        prepared = await asyncio.to_thread(_prepare, config, product_id, question)
    except AskFailure:
        raise
    except (StorageError, RetrievalError, OSError):
        raise AskFailure(503, "dependency_unavailable") from None
    except EmbeddingError:
        raise AskFailure(502, "embedding_unavailable") from None
    except (AnswerPreparationError, ValueError):
        raise AskFailure(503, "evidence_unavailable") from None
    if prepared.draft is not None:
        return _response(request_id, prepared.draft, None, None, ())
    if not isinstance(prepared.request, AnswerRequest):
        raise AskFailure(503, "evidence_unavailable")
    try:
        live = await post_prepared_answer(
            config.key_root,
            prepared.request,
            request_id=request_id,
        )
    except LunaError as exc:
        raise _provider_failure(exc.code) from None
    if live.stopped is not None or live.draft is None:
        raise _provider_failure(live.stopped or "provider_unavailable")
    return _response(
        request_id,
        live.draft,
        live.usage,
        live.usage_code,
        live.usage_fields,
    )


def _prepare(config: AskConfig, product_id: str, question: str) -> _Prepared:
    if config.release_root is not None:
        try:
            snapshot = load_active_snapshot(config.release_root)
        except MaintenanceError:
            raise AskFailure(503, "release_unavailable") from None
        database = snapshot.database
        artifact_keys = snapshot.artifact_keys
        model_digest = snapshot.model_digest
    else:
        database = config.database
        artifact_keys = config.artifact_keys
        model_digest = config.model_digest
    store = EmbeddingStore.open_readonly(database)
    try:
        try:
            index = ExactCosineIndex.from_store(
                store,
                artifact_keys=artifact_keys,
                expected_model_digest=model_digest,
            )
        except ValueError:
            raise AskFailure(503, "dependency_unavailable") from None
        adapter = OllamaEmbeddingAdapter(config.embedding_transport)
        try:
            query = adapter.embed_query(question, expected_model_digest=model_digest)
            hits = index.search(query, product_id=product_id, top_k=_TOP_K)
        finally:
            adapter.close()
    finally:
        store.close()
    if not hits:
        empty = prepare_answer(question, product_id=product_id, hits=[])
        return _Prepared(draft=handoff_without_evidence(empty))
    _require_one_version(_candidate_sources(hits))
    request = prepare_answer(
        question, product_id=product_id, hits=_select_whole_chunks(hits)
    )
    return _Prepared(request=request)


def _candidate_sources(hits: list[RetrievalHit]) -> tuple[ResolvedSource, ...]:
    """Map every search hit before any length prefix is dropped."""
    return tuple(
        ResolvedSource(
            source_id=f"ev:{hit.artifact_key}:{hit.chunk.index}",
            hit=hit,
        )
        for hit in hits
    )


def _select_whole_chunks(hits: list[RetrievalHit]) -> tuple[RetrievalHit, ...]:
    """Keep a ranked prefix of complete chunks inside the evidence budget.

    Conflicting versions are rejected on the full candidate list before this
    runs. A chunk is never truncated. One chunk longer than the budget stops
    the request. A later chunk that would overflow the total is left out,
    together with every chunk after it.
    """
    selected: list[RetrievalHit] = []
    total = 0
    for hit in hits:
        size = len(hit.chunk.text)
        if size > MAX_EVIDENCE_CHARS:
            raise AskFailure(503, "evidence_unavailable")
        if total + size > MAX_EVIDENCE_CHARS or len(selected) >= MAX_HITS:
            break
        selected.append(hit)
        total += size
    if not selected:
        raise AskFailure(503, "evidence_unavailable")
    return tuple(selected)


def _provider_failure(code: str) -> AskFailure:
    if code not in _STATUS:
        return AskFailure(502, "provider_unavailable")
    return AskFailure(_STATUS[code], code)


def _response(
    request_id: str,
    draft: GroundedDraft,
    usage: Usage | None,
    usage_code: str | None,
    usage_fields: tuple[str, ...],
) -> AskResponse:
    return AskResponse(
        request_id=request_id,
        status=draft.status,
        answer=draft.answer,
        reason=draft.reason,
        sources=[_cited(source) for source in draft.sources],
        usage=None if usage is None else _counts(usage),
        usage_code=usage_code,
        usage_fields=list(usage_fields),
    )


def _cited(source: ResolvedSource) -> CitedSource:
    chunk = source.hit.chunk
    metadata = chunk.metadata
    return CitedSource(
        source_id=source.source_id,
        artifact_key=source.hit.artifact_key,
        document_id=metadata.document_id,
        product_id=metadata.product_id,
        version=metadata.version,
        title=metadata.title,
        chunk_index=chunk.index,
        heading=" / ".join(chunk.heading_path),
        char_start=chunk.char_start,
        char_end=chunk.char_end,
        text=chunk.text,
    )


def _counts(usage: Usage) -> UsageCounts:
    return UsageCounts(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        cached_tokens=usage.cached_tokens,
        cache_write_tokens=usage.cache_write_tokens,
        ttl_30m_tokens=usage.ttl_30m_tokens,
        ttl_12h_tokens=usage.ttl_12h_tokens,
        reasoning_tokens=usage.reasoning_tokens,
    )
