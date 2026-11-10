"""One HP-A readiness check and one explicit Luna send.

Importing this module does not read a key, open a ledger, or make a network
call. ``hp_a_check`` is read-only. ``hp_a_once`` is the only send, and only
after that check allows it. This module does not record a live answer.
"""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from customer_support_rag.budget import (
    BudgetError,
    LedgerContinuity,
    inspect_ledger,
    load_policy,
    load_runtime,
    open_ledger,
    rates_cover_documented_ceiling,
)
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
    LunaError,
    WireParts,
    build_luna_body,
    generate_draft,
    network_transport,
    permit_live_transport,
    wire_bytes,
    wire_composition,
)

QUESTION = "HP-A 耳機可以用水洗嗎？"
REQUEST_ID = "hp-a-luna-natural-answer-one-shot-v1"
SCOPE = "developer_smoke"
CARE_PATH = "hp-a/headphone-care.md"
PUBLISHED_DIGEST = "ac6da0dfba84a81fdbfbaf330198c33cd77c4cdfc53e8bc50eb581914a15621d"
_CARE_ARTIFACT = "9a2283a73584261d31460d1b46855145d384c756834c8984c1539e96647cb4ad"
_ALWAYS_NOTES = (
    "structural_allowance_1024_is_not_a_published_tokenizer_bound",
    "read_only_snapshot_rechecked_inside_reserve",
    "ultrafast_not_listed_for_gpt_6_luna",
    "long_context_not_applied_below_272k",
    "fedramp_endpoint_not_used",
)

__all__ = [
    "QUESTION",
    "REQUEST_ID",
    "LiveReadiness",
    "hp_a_check",
    "hp_a_once",
    "prepare_hp_a_request",
]


@dataclass(frozen=True)
class LiveReadiness:
    """Safe readiness. Codes and wire lengths only; no key, hash, or amount."""

    key_present: bool
    history_continuous: bool
    original_debit_counted: bool
    replay_blocked: bool
    cap_unchanged: bool
    no_pending: bool
    one_request_fits: bool
    pricing_bound_verified: bool
    failures: tuple[str, ...]
    notes: tuple[str, ...]
    wire: WireParts | None = None

    @property
    def network_allowed(self) -> bool:
        return (
            self.key_present
            and self.history_continuous
            and self.original_debit_counted
            and self.replay_blocked
            and self.cap_unchanged
            and self.no_pending
            and self.one_request_fits
            and self.pricing_bound_verified
            and not self.failures
        )


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
    """Read the caller's root and return readiness. This never opens a socket."""
    knowledge_root = _knowledge(knowledge)
    failures: list[str] = []
    notes: list[str] = list(_ALWAYS_NOTES)
    wire: WireParts | None = None
    body_hash: str | None = None
    reserved: int | None = None
    try:
        request = prepare_hp_a_request(knowledge_root)
        encoded = wire_bytes(build_luna_body(request))
        wire = wire_composition(request)
        body_hash = hashlib.sha256(encoded).hexdigest()
        reserved = wire.reserved_input_tokens
    except LunaError as exc:
        failures.append(exc.code)
    key_present, key_failures = _key_status(root / ".env")
    failures.extend(key_failures)
    policy, policy_failures = _load_checked_policy(
        root / "runtime" / "cloud-policy.json"
    )
    failures.extend(policy_failures)
    if policy is not None and SCOPE not in policy.scopes:
        failures.append("scope_not_enabled")
    priced = policy is not None and rates_cover_documented_ceiling(policy)
    if policy is not None and not priced:
        failures.append("documented_rate_ceiling_uncovered")
    ledger_path = root / "runtime" / "cloud-budget.sqlite3"
    access_failures, access_notes = _ledger_access(ledger_path)
    failures.extend(access_failures)
    notes.extend(access_notes)
    if policy is None or access_failures:
        continuity = _false_continuity()
    else:
        try:
            continuity = inspect_ledger(
                ledger_path,
                policy,
                request_id=REQUEST_ID,
                body_hash=body_hash,
                input_tokens=reserved,
            )
        except BudgetError as exc:
            continuity = _false_continuity(exc.code)
    if wire is None:
        continuity = LedgerContinuity(
            history_continuous=continuity.history_continuous,
            original_debit_counted=continuity.original_debit_counted,
            replay_blocked=continuity.replay_blocked,
            cap_unchanged=continuity.cap_unchanged,
            no_pending=continuity.no_pending,
            one_request_fits=False,
            failures=continuity.failures,
            notes=continuity.notes,
        )
    failures.extend(continuity.failures)
    notes.extend(continuity.notes)
    return LiveReadiness(
        key_present=key_present,
        history_continuous=continuity.history_continuous,
        original_debit_counted=continuity.original_debit_counted,
        replay_blocked=continuity.replay_blocked,
        cap_unchanged=continuity.cap_unchanged,
        no_pending=continuity.no_pending,
        one_request_fits=continuity.one_request_fits,
        pricing_bound_verified=priced,
        failures=_unique(failures),
        notes=_unique(notes),
        wire=wire,
    )


async def hp_a_once(root: Path, *, knowledge: Path | None = None) -> GroundedDraft:
    """Send the one prepared request after a second read-only check.

    A false check raises ``paid_blocked`` before the factory is built.
    """
    if not hp_a_check(root, knowledge=knowledge).network_allowed:
        raise LunaError("paid_blocked")
    if not hp_a_check(root, knowledge=knowledge).network_allowed:
        raise LunaError("paid_blocked")
    env_path, policy_path, ledger_path = _private_paths(root)
    key, key_failures = _read_key(env_path)
    if key is None or key_failures:
        raise LunaError("paid_blocked")
    try:
        runtime = load_runtime(policy_path, key)
    except BudgetError:
        raise LunaError("paid_blocked") from None
    request = prepare_hp_a_request(_knowledge(knowledge))
    transport = network_transport()
    try:
        ledger = open_ledger(ledger_path)
        return await generate_draft(
            request,
            request_id=REQUEST_ID,
            runtime=runtime,
            ledger=ledger,
            transport=transport,
            scope=SCOPE,
            admission=permit_live_transport(),
        )
    finally:
        await transport.aclose()


def _knowledge(knowledge: Path | None) -> Path:
    if knowledge is None:
        return Path(__file__).resolve().parents[2] / "knowledge"
    return knowledge


def _private_paths(root: Path) -> tuple[Path, Path, Path]:
    return (
        root / ".env",
        root / "runtime" / "cloud-policy.json",
        root / "runtime" / "cloud-budget.sqlite3",
    )


def _load_checked_policy(path: Path):
    code = _owner_file_code(path)
    if code == "not_owner_only":
        return None, ("policy_not_owner_only",)
    if code is not None:
        return None, ("policy_unavailable",)
    try:
        return load_policy(path), ()
    except BudgetError as exc:
        return None, (exc.code,)


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


def _ledger_access(path: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    try:
        info = os.lstat(path)
        parent = os.lstat(path.parent)
    except OSError:
        return ("ledger_unavailable",), ()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        return ("ledger_unavailable",), ()
    file_loose = bool(info.st_mode & 0o077)
    parent_loose = bool(parent.st_mode & 0o077)
    if file_loose and parent_loose:
        return ("ledger_not_owner_only",), ()
    if file_loose:
        return (), ("ledger_mode_loose_inside_owner_only_directory",)
    return (), ()


def _false_continuity(*codes: str) -> LedgerContinuity:
    return LedgerContinuity(
        history_continuous=False,
        original_debit_counted=False,
        replay_blocked=False,
        cap_unchanged=False,
        no_pending=False,
        one_request_fits=False,
        failures=codes,
        notes=(),
    )


def _unique(items: list[str]) -> tuple[str, ...]:
    seen: list[str] = []
    for item in items:
        if item not in seen:
            seen.append(item)
    return tuple(seen)
