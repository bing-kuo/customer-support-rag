"""One HP-A live call. Key-only. No local budget admission.

Importing this module does not read a key, open a ledger, or make a network
call. ``hp_a_check`` reports the key and the wire bytes. It does not authorize
a send. ``hp_a_once`` is the only live entry: it reads the key again, posts
at most once, and does not open the historical ledger or a policy file.
A stopped call does not write an unknown row. An unrecognized billing field
does not discard a parsed answer and does not invent token counts. This
module does not record a live answer.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path

from customer_support_rag.knowledge.answering import (
    AnswerRequest,
    GroundedDraft,
    prepare_answer,
)
from customer_support_rag.knowledge.chunker import chunk_document
from customer_support_rag.knowledge.reader import read_document
from customer_support_rag.knowledge.retrieval import RetrievalHit
from customer_support_rag.knowledge.storage import document_embedding_key
from customer_support_rag.luna import (
    DEFAULT_DEADLINE_SECONDS,
    MAX_WIRE_BYTES,
    LunaError,
    Usage,
    WireParts,
    _post_live_once,
    build_luna_body,
    network_transport,
    wire_bytes,
    wire_composition,
)

QUESTION = "HP-A 耳機可以用水洗嗎？"
REQUEST_ID = "hp-a-luna-natural-answer-one-shot-v1"
CARE_PATH = "hp-a/headphone-care.md"
PUBLISHED_DIGEST = "ac6da0dfba84a81fdbfbaf330198c33cd77c4cdfc53e8bc50eb581914a15621d"
_CARE_ARTIFACT = "9a2283a73584261d31460d1b46855145d384c756834c8984c1539e96647cb4ad"

__all__ = [
    "QUESTION",
    "REQUEST_ID",
    "LiveCall",
    "LiveReadiness",
    "hp_a_check",
    "hp_a_once",
    "post_prepared_answer",
    "prepare_hp_a_request",
]


@dataclass(frozen=True)
class LiveCall:
    """One finished live response. ``stopped`` is a safe code, or None.

    ``usage`` is present only when the known billing shape matched. ``None``
    means the counts were not identified, not that they were zero.
    ``usage_code`` and ``usage_fields`` name that gap. Fields are safe names,
    never values. This object is not a ledger row.
    """

    draft: GroundedDraft | None
    usage: Usage | None
    stopped: str | None
    usage_code: str | None = None
    usage_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (self.stopped is None) == (self.draft is None):
            raise LunaError("invalid_response")
        if (self.usage is None) == (self.usage_code is None):
            raise LunaError("invalid_response")
        if self.usage_code is None and self.usage_fields:
            raise LunaError("invalid_response")


@dataclass(frozen=True)
class LiveReadiness:
    """Key and wire report. It is not a send authorization."""

    key_present: bool
    failures: tuple[str, ...]
    wire: WireParts | None = None


def prepare_hp_a_request(knowledge: Path) -> AnswerRequest:
    """Build the one care-chunk request. This does not embed or rank."""
    try:
        document = read_document(knowledge, CARE_PATH)
        chunks = chunk_document(document)
    except (OSError, ValueError):
        raise LunaError("hp_a_evidence_unavailable") from None
    care = [chunk for chunk in chunks if chunk.index == 1]
    if len(care) != 1 or (care[0].char_start, care[0].char_end) != (1, 71):
        raise LunaError("hp_a_evidence_unavailable")
    try:
        key = document_embedding_key(document, model_digest=PUBLISHED_DIGEST)
    except (OSError, ValueError):
        raise LunaError("hp_a_evidence_unavailable") from None
    if key != _CARE_ARTIFACT:
        raise LunaError("hp_a_evidence_unavailable")
    try:
        return prepare_answer(
            QUESTION,
            product_id="hp-a",
            hits=[RetrievalHit(artifact_key=key, chunk=care[0], raw_score=0.0)],
        )
    except (OSError, ValueError):
        raise LunaError("hp_a_evidence_unavailable") from None


def hp_a_check(root: Path, *, knowledge: Path | None = None) -> LiveReadiness:
    """Read the key file and the wire. This never opens a ledger or a socket."""
    failures: list[str] = []
    wire: WireParts | None = None
    try:
        request = prepare_hp_a_request(_knowledge(knowledge))
        wire = wire_composition(request)
    except LunaError as exc:
        failures.append(exc.code)
    key_present, key_failures = _key_status(root / ".env")
    failures.extend(key_failures)
    return LiveReadiness(
        key_present=key_present,
        failures=_unique(failures),
        wire=wire,
    )


async def post_prepared_answer(
    root: Path,
    request: AnswerRequest,
    *,
    request_id: str,
    deadline: float = DEFAULT_DEADLINE_SECONDS,
) -> LiveCall:
    """Send one prepared request once. The key is the only local admission.

    ``hp_a_once`` and the single-question API share this sender, transport, and
    parser. Policy, rates, token-cost proof, and the historical ledger are not
    read. A 429, timeout, or other provider error stops this call and is not
    retried. A billing-shape miss keeps a parsed answer and leaves usage
    unidentified. The request id is the caller's; this function does not reuse
    a previous id or claim exactly-once delivery.
    """
    if not isinstance(request, AnswerRequest):
        raise LunaError("policy_rejected")
    if (
        not isinstance(request_id, str)
        or not request_id.strip()
        or len(request_id) > 200
    ):
        raise LunaError("policy_rejected")
    key, key_failures = _read_key(root / ".env")
    if key is None or key_failures:
        raise LunaError("paid_blocked")
    encoded = wire_bytes(build_luna_body(request))
    if len(encoded) > MAX_WIRE_BYTES:
        raise LunaError("wire_body_exceeded")
    transport = network_transport()
    try:
        draft, note, stopped = await _post_live_once(
            encoded,
            request,
            api_key=key,
            transport=transport,
            request_id=request_id,
            deadline=deadline,
        )
    finally:
        await transport.aclose()
    return LiveCall(
        draft=draft,
        usage=note.usage,
        stopped=stopped,
        usage_code=note.code,
        usage_fields=note.fields,
    )


async def hp_a_once(
    root: Path,
    *,
    knowledge: Path | None = None,
    deadline: float = DEFAULT_DEADLINE_SECONDS,
) -> LiveCall:
    """Send the one prepared HP-A request once.

    The key is checked before evidence is prepared, then
    :func:`post_prepared_answer` reads it again and posts at most once.
    """
    key, key_failures = _read_key(root / ".env")
    if key is None or key_failures:
        raise LunaError("paid_blocked")
    request = prepare_hp_a_request(_knowledge(knowledge))
    return await post_prepared_answer(
        root,
        request,
        request_id=REQUEST_ID,
        deadline=deadline,
    )


def _knowledge(knowledge: Path | None) -> Path:
    if knowledge is None:
        return Path(__file__).resolve().parents[2] / "knowledge"
    return knowledge


def _key_status(path: Path) -> tuple[bool, tuple[str, ...]]:
    key, failures = _read_key(path)
    return key is not None and not failures, failures


def _read_key(path: Path) -> tuple[str | None, tuple[str, ...]]:
    code = _owner_file_code(path)
    if code == "missing":
        return None, ("key_missing",)
    if code == "not_owner_only":
        return None, ("key_not_owner_only",)
    if code is not None:
        return None, ("key_unavailable",)
    try:
        text = _read_owner_text(path)
        key = _parse_env(text)
    except LunaError as exc:
        return None, (exc.code,)
    except OSError:
        return None, ("key_unavailable",)
    if not _key_shape(key):
        return None, ("key_invalid",)
    return key, ()


def _parse_env(text: str) -> str:
    found: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        name, separator, value = line.partition("=")
        if separator != "=" or name != "OPENAI_API_KEY" or name.strip() != name:
            raise LunaError("env_unexpected_assignment")
        if found is not None:
            raise LunaError("key_invalid")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        found = value
    if found is None:
        raise LunaError("key_missing")
    return found


def _key_shape(value: str) -> bool:
    return 8 <= len(value) <= 512 and all(
        not character.isspace() and 33 <= ord(character) <= 126 for character in value
    )


def _owner_file_code(path: Path) -> str | None:
    try:
        info = os.lstat(path)
    except OSError:
        return "missing" if not path.exists() else "unavailable"
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        return "unavailable"
    if info.st_size > 16384:
        return "unavailable"
    if info.st_mode & 0o077:
        return "not_owner_only"
    return None


def _read_owner_text(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_size > 16384
            or info.st_mode & 0o077
        ):
            raise LunaError("key_unavailable")
        with os.fdopen(descriptor, "r", encoding="utf-8", closefd=False) as handle:
            return handle.read()
    finally:
        os.close(descriptor)


def _unique(items: list[str]) -> tuple[str, ...]:
    seen: list[str] = []
    for item in items:
        if item not in seen:
            seen.append(item)
    return tuple(seen)
