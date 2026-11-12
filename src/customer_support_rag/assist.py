"""Server-owned slots for at most two field clarifications and one answer.

The client does not get to send an attempt count, a role, or a history.
Filled text is unverified user data. A random token only locates memory in
this process; it is not an authentication system.
"""

from __future__ import annotations

import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

MAX_CLARIFICATIONS = 2
MAX_FIELD_CHARS = 1000
MAX_TOTAL_CHARS = 2000
TTL_SECONDS = 900
MAX_SESSIONS = 32
_PRODUCTS = frozenset({"hp-a", "hp-b"})
_MISSING = object()


class AssistError(Exception):
    """A slot request was rejected before any model call."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class AssistView:
    """The slot outcome before or after the single allowed model call."""

    status: str
    token: str
    attempt: int
    missing: tuple[str, ...]
    product_id: str | None
    question: str | None
    steps_tried: str | None
    reason: str | None
    summary: str | None
    needs_model: bool


@dataclass
class _Slot:
    token: str
    created: float
    clarifications: int = 0
    product_id: str | None = None
    question: str | None = None
    steps_tried: str | None = None
    model_used: bool = False
    exhausted: bool = False
    terminal_reason: str | None = None


class SlotCoordinator:
    """Keep ephemeral slots. Expired tokens are rejected, not renewed."""

    def __init__(self, *, now: Callable[[], float] | None = None) -> None:
        self._now = time.time if now is None else now
        self._slots: dict[str, _Slot] = {}
        self._lock = threading.Lock()

    def start(
        self,
        *,
        product_id: str | None = None,
        question: str | None = None,
        steps_tried: str | None = None,
    ) -> AssistView:
        with self._lock:
            self._purge()
            _validate_field(product_id, product=True)
            _validate_field(question, product=False)
            _validate_field(steps_tried, product=False)
            _validate_total(question, steps_tried)
            if len(self._slots) >= MAX_SESSIONS:
                raise AssistError("slot_capacity")
            token = secrets.token_hex(16)
            slot = _Slot(token=token, created=self._now())
            self._slots[token] = slot
            try:
                return self._apply(
                    slot,
                    product_id=product_id,
                    question=question,
                    steps_tried=steps_tried,
                )
            except AssistError:
                self._slots.pop(token, None)
                raise

    def reset(self, token: str) -> AssistView:
        """Drop the named slot and open an empty one. This is explicit.

        The response stays ``clarify``, and the new slot already counts as
        clarification 1. Two empty continuations are then clarify 2 and handoff.
        """
        with self._lock:
            self._purge()
            slot = self._live(token)
            del self._slots[slot.token]
            if len(self._slots) >= MAX_SESSIONS:
                raise AssistError("slot_capacity")
            fresh = secrets.token_hex(16)
            opened = _Slot(token=fresh, created=self._now(), clarifications=1)
            self._slots[fresh] = opened
            return self._view(
                opened,
                status="clarify",
                missing=("product_id", "question", "steps_tried"),
            )

    def continue_slot(
        self,
        token: str,
        *,
        product_id: object = _MISSING,
        question: object = _MISSING,
        steps_tried: object = _MISSING,
    ) -> AssistView:
        """Apply one continue step.

        An omitted argument keeps the stored field. ``None`` clears that field
        and must not leave the previous text for a later model call. Clearing a
        product that was already selected also clears its question and steps.
        """
        with self._lock:
            self._purge()
            slot = self._live(token)
            return self._apply(
                slot,
                product_id=product_id,
                question=question,
                steps_tried=steps_tried,
            )

    def _live(self, token: str) -> _Slot:
        if not isinstance(token, str) or len(token) != 32:
            raise AssistError("invalid_slot")
        slot = self._slots.get(token)
        if slot is None or self._now() - slot.created > TTL_SECONDS:
            self._slots.pop(token, None)
            raise AssistError("invalid_slot")
        return slot

    def _purge(self) -> None:
        expired = [
            token
            for token, slot in self._slots.items()
            if self._now() - slot.created > TTL_SECONDS
        ]
        for token in expired:
            del self._slots[token]

    def _apply(
        self,
        slot: _Slot,
        *,
        product_id: object,
        question: object,
        steps_tried: object,
    ) -> AssistView:
        if product_id is not _MISSING:
            _validate_field(_text_or_none(product_id), product=True)
        if question is not _MISSING:
            _validate_field(_text_or_none(question), product=False)
        if steps_tried is not _MISSING:
            _validate_field(_text_or_none(steps_tried), product=False)
        switched = isinstance(product_id, str) and slot.product_id not in {
            None,
            product_id,
        }
        cleared_product = product_id is None and slot.product_id is not None
        drop_dependents = switched or cleared_product
        if slot.model_used or slot.exhausted:
            if switched:
                slot.product_id = product_id
                slot.question = None
                slot.steps_tried = None
                slot.terminal_reason = "reset_required"
            return self._closed_view(slot)
        prospective_question = None if drop_dependents else slot.question
        prospective_steps = None if drop_dependents else slot.steps_tried
        if question is not _MISSING:
            prospective_question = _text_or_none(question)
        if steps_tried is not _MISSING:
            prospective_steps = _text_or_none(steps_tried)
        _validate_total(prospective_question, prospective_steps)
        if drop_dependents:
            slot.question = None
            slot.steps_tried = None
        if product_id is not _MISSING:
            slot.product_id = _text_or_none(product_id)
        if question is not _MISSING:
            slot.question = _text_or_none(question)
        if steps_tried is not _MISSING:
            slot.steps_tried = _text_or_none(steps_tried)
        missing = _missing(slot)
        if not missing:
            slot.model_used = True
            return self._view(slot, status="ready", missing=(), needs_model=True)
        if slot.clarifications >= MAX_CLARIFICATIONS:
            slot.exhausted = True
            slot.terminal_reason = "insufficient_information"
            return self._closed_view(slot)
        slot.clarifications += 1
        return self._view(slot, status="clarify", missing=missing)

    def _closed_view(self, slot: _Slot) -> AssistView:
        reason = slot.terminal_reason or (
            "reset_required" if slot.model_used else "insufficient_information"
        )
        slot.terminal_reason = reason
        return self._view(slot, status="handoff", missing=_missing(slot), reason=reason)

    def _view(
        self,
        slot: _Slot,
        *,
        status: str,
        missing: tuple[str, ...],
        reason: str | None = None,
        needs_model: bool = False,
    ) -> AssistView:
        summary = None
        if status == "handoff":
            summary = handoff_summary(
                product_id=slot.product_id,
                question=slot.question,
                steps_tried=slot.steps_tried,
                missing=missing,
                reason=reason or "insufficient_information",
            )
        return AssistView(
            status=status,
            token=slot.token,
            attempt=slot.clarifications,
            missing=missing,
            product_id=slot.product_id,
            question=slot.question,
            steps_tried=slot.steps_tried,
            reason=reason,
            summary=summary,
            needs_model=needs_model,
        )


def handoff_summary(
    *,
    product_id: str | None,
    question: str | None,
    steps_tried: str | None,
    missing: tuple[str, ...],
    reason: str,
) -> str:
    """Copy text for a person. The user fields are not verified facts."""
    missing_text = "、".join(missing) if missing else "無"
    return "\n".join(
        (
            "使用者陳述未經驗證。",
            f"商品：{product_id or '未提供'}",
            f"確認問題：{question or '未提供'}",
            f"已試步驟：{steps_tried or '未提供'}",
            f"缺少：{missing_text}",
            f"原因：{reason}",
        )
    )


def _text_or_none(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise AssistError("invalid_request")


def _missing(slot: _Slot) -> tuple[str, ...]:
    fields = (
        ("product_id", slot.product_id),
        ("question", slot.question),
        ("steps_tried", slot.steps_tried),
    )
    return tuple(name for name, value in fields if value is None)


def _validate_field(value: str | None, *, product: bool) -> None:
    if value is None:
        return
    if product:
        if value not in _PRODUCTS:
            raise AssistError("invalid_request")
        return
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_FIELD_CHARS:
        raise AssistError("invalid_request")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise AssistError("invalid_request") from None


def _validate_total(question: str | None, steps: str | None) -> None:
    if len(question or "") + len(steps or "") > MAX_TOTAL_CHARS:
        raise AssistError("invalid_request")
