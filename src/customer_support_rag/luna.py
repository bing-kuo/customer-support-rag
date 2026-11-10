"""Fixed gpt-6-luna adapter. Tests inject a mock transport and a tmp ledger.

Importing this module does not read a key, open a database, or make a network
call. A caller-built network transport is refused. The one-shot runner may
pass the factory transport only after its own readiness check. ``store=false``
does not mean zero data retention.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass

import httpx

from customer_support_rag.budget import (
    OUTPUT_RESERVATION,
    TOTAL_RESERVATION,
    BudgetError,
    Ledger,
    LunaPolicy,
    LunaRuntime,
)
from customer_support_rag.knowledge.answering import (
    GROUNDED_POLICY,
    PROMPT_REVISION,
    AnswerRequest,
    DraftValidationError,
    GroundedDraft,
    handoff_without_evidence,
    resolve_draft,
)

MODEL = "gpt-6-luna"
ORIGIN = "https://api.openai.com"
RESPONSES_PATH = "/v1/responses"
MAX_WIRE_BYTES = 16 * 1024
MAX_INPUT_TOKENS = TOTAL_RESERVATION - OUTPUT_RESERVATION
# Not a published tokenizer constant. One byte of the serialized body counts as
# one token, and this allowance is reserved for structural tokens the official
# token-counting guide says may be absent from that text. The count API is not
# called. A body that cannot fit under the old 4096 total is refused.
WIRE_FUDGE = 1024
MAX_RESPONSE_BYTES = 64 * 1024
CONNECT_TIMEOUT_SECONDS = 2.0
DEFAULT_DEADLINE_SECONDS = 60.0
_REQUEST_ID = re.compile(r"req_[A-Za-z0-9_-]{1,150}")
_RESPONSE_ID = re.compile(r"resp_[A-Za-z0-9_-]{1,150}")
_TERMINAL = frozenset({"completed", "incomplete", "failed", "cancelled"})
_USAGE_TOP = frozenset(
    {
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "input_tokens_details",
        "output_tokens_details",
    }
)
_INPUT_DETAILS = frozenset(
    {"cached_tokens", "cache_write_tokens", "cache_write_tokens_details"}
)
_TTL = frozenset({"ttl_30m_tokens", "ttl_12h_tokens"})
_TOOL_FAMILIES = frozenset({"image_gen", "web_search"})
_IMAGE_GEN_COUNTS = frozenset(
    {"input_tokens", "output_tokens", "total_tokens", "cached_tokens"}
)
_IMAGE_GEN_OPTIONAL = frozenset({"cached_tokens_details", "output_tokens_details"})
DRAFT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "status": {"type": "string", "enum": ["answer", "handoff"]},
        "answer": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "source_ids": {"type": "array", "items": {"type": "string"}},
        "reason": {
            "anyOf": [
                {"type": "null"},
                {
                    "type": "string",
                    "enum": [
                        "no_evidence",
                        "conflicting_evidence",
                        "refund_request",
                    ],
                },
            ]
        },
    },
    "required": ["status", "answer", "source_ids", "reason"],
}

__all__ = [
    "DRAFT_SCHEMA",
    "MODEL",
    "LunaError",
    "Usage",
    "WireParts",
    "build_luna_body",
    "generate_draft",
    "network_transport",
    "parse_luna_output",
    "parse_usage",
    "responses_client",
    "wire_composition",
    "wire_bytes",
]


class LunaError(Exception):
    """A safe provider failure. The message is a fixed English code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


_FACTORY = object()


@dataclass(frozen=True)
class LiveAdmission:
    """Permit created only after a readiness check. A runtime flag is not one."""

    _marker: object


@dataclass(frozen=True)
class WireParts:
    """Byte lengths of one serialized body. Lengths are not token certificates."""

    policy_bytes: int
    data_bytes: int
    schema_bytes: int
    framing_bytes: int
    total_bytes: int
    reserved_input_tokens: int
    ceiling_headroom_tokens: int


@dataclass(frozen=True)
class Usage:
    """Observed token counts. ``None`` means the field was omitted, not zero."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_tokens: int | None
    cache_write_tokens: int | None
    ttl_30m_tokens: int | None
    ttl_12h_tokens: int | None
    reasoning_tokens: int | None


def network_transport() -> httpx.AsyncHTTPTransport:
    """Build the one TLS transport. Tests may replace this; callers do not."""
    transport = httpx.AsyncHTTPTransport(verify=True, trust_env=False, retries=0)
    pool = transport._pool
    if pool._retries != 0 or type(pool).__name__ != "AsyncConnectionPool":
        raise LunaError("provider_transport_error")
    transport._luna_factory = _FACTORY
    return transport


def responses_client(
    transport: httpx.AsyncBaseTransport,
    *,
    admission: LiveAdmission | None = None,
) -> httpx.AsyncClient:
    """Build the fixed client around a mock or an admitted factory transport.

    A caller-built network transport is refused. ``pricing_bound_verified`` is
    not consulted here.
    """
    if not _transport_permitted(transport, admission):
        raise LunaError("paid_blocked")
    return httpx.AsyncClient(
        base_url=ORIGIN,
        verify=True,
        trust_env=False,
        follow_redirects=False,
        timeout=httpx.Timeout(None, connect=CONNECT_TIMEOUT_SECONDS),
        transport=transport,
    )


def build_luna_body(request: AnswerRequest) -> dict[str, object]:
    """Build one Responses body from the merged draft contract.

    The fixed policy is the only instruction. Question and evidence stay in
    one user data block. This does not reserve budget or call a transport.
    """
    _require_request(request)
    data = request.model_data_json()
    return {
        "model": MODEL,
        "service_tier": "default",
        "reasoning": {"effort": "none"},
        "instructions": GROUNDED_POLICY,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": data}]}],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "grounded_draft",
                "strict": True,
                "schema": DRAFT_SCHEMA,
            }
        },
        "max_output_tokens": OUTPUT_RESERVATION,
        "store": False,
        "background": False,
        "stream": False,
        "tools": [],
        "tool_choice": "none",
        "truncation": "disabled",
    }


def wire_bytes(body: Mapping[str, object]) -> bytes:
    """Return the exact JSON bytes that will be posted and hashed."""
    try:
        return json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError):
        raise LunaError("wire_body_exceeded") from None


def input_reservation(encoded: bytes) -> int:
    """Reserve one token per serialized byte, plus the structural allowance.

    The allowance is not a published tokenizer bound. A body that cannot fit
    under the existing 4096 total is refused instead of truncated.
    """
    if len(encoded) > MAX_WIRE_BYTES:
        raise LunaError("wire_body_exceeded")
    candidate = len(encoded) + WIRE_FUDGE
    if candidate > MAX_INPUT_TOKENS:
        raise LunaError("input_bound_exceeded")
    return candidate


def wire_composition(request: AnswerRequest) -> WireParts:
    """Split one body into policy, model data, schema, and framing.

    ``data_bytes`` is the model-visible JSON before it is placed in a JSON
    string. ``framing_bytes`` is the rest of the wire, including object syntax
    and the extra bytes that string escaping adds. The token reservation uses
    the whole wire plus the structural allowance, not the visible pieces alone.
    """
    _require_request(request)
    encoded = wire_bytes(build_luna_body(request))
    policy = GROUNDED_POLICY.encode()
    data = request.model_data_json().encode()
    schema = json.dumps(
        DRAFT_SCHEMA,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    escaped = json.dumps(request.model_data_json(), ensure_ascii=False).encode()
    spans = (_span(encoded, policy), _span(encoded, schema), _span(encoded, escaped))
    if len(escaped) < len(data) + 2 or _overlaps(spans):
        raise LunaError("wire_composition_invalid")
    framing = len(encoded) - (len(policy) + len(data) + len(schema))
    if framing < 0:
        raise LunaError("wire_composition_invalid")
    reserved = input_reservation(encoded)
    return WireParts(
        policy_bytes=len(policy),
        data_bytes=len(data),
        schema_bytes=len(schema),
        framing_bytes=framing,
        total_bytes=len(encoded),
        reserved_input_tokens=reserved,
        ceiling_headroom_tokens=MAX_INPUT_TOKENS - reserved,
    )


def parse_usage(raw: Mapping[str, object]) -> Usage:
    """Read a terminal usage object. Unknown fee fields are not ignored."""
    usage = raw.get("usage")
    if usage is None:
        raise LunaError("usage_missing")
    if not isinstance(usage, dict) or set(usage) - _USAGE_TOP:
        raise LunaError(
            "usage_shape_invalid"
            if not isinstance(usage, dict)
            else "unknown_billing_component"
        )
    input_tokens = _count(usage.get("input_tokens"))
    output_tokens = _count(usage.get("output_tokens"))
    total_tokens = _count(usage.get("total_tokens"))
    if total_tokens != input_tokens + output_tokens:
        raise LunaError("usage_totals_invalid")
    incoming = _details(usage.get("input_tokens_details"), _INPUT_DETAILS)
    outgoing = _details(
        usage.get("output_tokens_details"), frozenset({"reasoning_tokens"})
    )
    cached = _optional(incoming, "cached_tokens")
    written = _optional(incoming, "cache_write_tokens")
    reasoning = _optional(outgoing, "reasoning_tokens")
    if (cached or 0) + (written or 0) > input_tokens or (
        reasoning or 0
    ) > output_tokens:
        raise LunaError("known_usage_extension_invalid")
    short, long = _ttl(incoming, written)
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_tokens=cached,
        cache_write_tokens=written,
        ttl_30m_tokens=short,
        ttl_12h_tokens=long,
        reasoning_tokens=reasoning,
    )


def parse_luna_output(
    raw: Mapping[str, object], request: AnswerRequest
) -> GroundedDraft:
    """Resolve the single assistant JSON object. This does not settle usage."""
    _require_request(request)
    texts = _output_texts(raw)
    if len(texts) != 1:
        raise LunaError("invalid_model_schema")
    try:
        payload = _json_object(texts[0])
    except LunaError as exc:
        if exc.code == "invalid_response":
            raise LunaError("invalid_model_schema") from None
        raise
    try:
        return resolve_draft(request, payload)
    except DraftValidationError as exc:
        if str(exc) == "draft cites a source that was not provided":
            raise LunaError("unknown_citation") from None
        raise LunaError("invalid_model_schema") from None


async def generate_draft(
    request: AnswerRequest,
    *,
    request_id: str,
    runtime: LunaRuntime,
    ledger: Ledger,
    transport: httpx.AsyncBaseTransport,
    scope: str,
    deadline: float = DEFAULT_DEADLINE_SECONDS,
    admission: LiveAdmission | None = None,
) -> GroundedDraft:
    """Reserve, send once through the injected transport, then resolve.

    Empty evidence returns the fixed handoff and does not reserve. Failures
    after a durable send keep the reservation. Nothing is retried. A network
    transport also needs the permit from the one-shot runner.
    """
    if not _transport_permitted(transport, admission):
        raise LunaError("paid_blocked")
    if not isinstance(runtime, LunaRuntime) or not isinstance(ledger, Ledger):
        raise LunaError("runtime_unavailable")
    if (
        type(deadline) not in (int, float)
        or not math.isfinite(deadline)
        or deadline <= 0
    ):
        raise LunaError("runtime_unavailable")
    if not isinstance(request, AnswerRequest):
        raise LunaError("policy_rejected")
    if not request.evidence:
        return handoff_without_evidence(request)
    encoded = wire_bytes(build_luna_body(request))
    reserved_input = input_reservation(encoded)
    body_hash = hashlib.sha256(encoded).hexdigest()
    ledger.reserve(
        runtime.policy,
        request_id,
        body_hash,
        reserved_input,
        OUTPUT_RESERVATION,
        scope,
        runtime.api_key,
    )
    try:
        ledger.require_admission(runtime.policy)
        ledger.mark_sent(request_id)
    except BudgetError:
        ledger.release(request_id)
        raise
    started = time.monotonic()
    try:
        payload, response_id, request_header = await asyncio.wait_for(
            _post(encoded, request_id, runtime, transport, admission), deadline
        )
    except TimeoutError:
        ledger.retain_unknown(request_id)
        raise LunaError("logical_timeout") from None
    except asyncio.CancelledError:
        ledger.retain_unknown(request_id)
        raise
    except LunaError:
        ledger.retain_unknown(request_id)
        raise
    return _account(
        payload,
        request,
        runtime.policy,
        ledger,
        request_id,
        response_id,
        request_header,
        time.monotonic() - started,
    )


def _span(haystack: bytes, needle: bytes) -> tuple[int, int]:
    start = haystack.find(needle)
    if start < 0 or haystack.find(needle, start + 1) != -1:
        raise LunaError("wire_composition_invalid")
    return start, start + len(needle)


def _overlaps(spans: tuple[tuple[int, int], ...]) -> bool:
    ordered = sorted(spans)
    return any(
        left[1] > right[0] for left, right in zip(ordered, ordered[1:], strict=False)
    )


def _factory_transport(transport: object) -> bool:
    return (
        type(transport) is httpx.AsyncHTTPTransport
        and getattr(transport, "_luna_factory", None) is _FACTORY
    )


def _transport_permitted(transport: object, admission: LiveAdmission | None) -> bool:
    if type(transport) is httpx.MockTransport:
        return True
    return _factory_transport(transport) and (
        isinstance(admission, LiveAdmission) and admission._marker is _FACTORY
    )


def permit_live_transport() -> LiveAdmission:
    """Issue the transport permit. The one-shot runner calls this after checks."""
    return LiveAdmission(_FACTORY)


def _require_request(request: AnswerRequest) -> None:
    if not isinstance(request, AnswerRequest):
        raise LunaError("policy_rejected")
    if request.prompt_revision != PROMPT_REVISION or request.policy != GROUNDED_POLICY:
        raise LunaError("policy_rejected")


def _count(value: object) -> int:
    if type(value) is not int or value < 0 or value > 2**31:
        raise LunaError("usage_shape_invalid")
    return value


def _details(value: object, allowed: frozenset[str]) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise LunaError("usage_shape_invalid")
    if set(value) - allowed:
        raise LunaError("unknown_billing_component")
    return value


def _optional(mapping: Mapping[str, object], key: str) -> int | None:
    if key not in mapping:
        return None
    return _count(mapping[key])


def _ttl(
    incoming: Mapping[str, object], written: int | None
) -> tuple[int | None, int | None]:
    details = incoming.get("cache_write_tokens_details")
    if details is None:
        return None, None
    if not isinstance(details, dict):
        raise LunaError("known_usage_extension_invalid")
    if set(details) - _TTL:
        raise LunaError("unknown_billing_component")
    if set(details) != _TTL or written is None:
        raise LunaError("known_usage_extension_invalid")
    short = _count(details["ttl_30m_tokens"])
    long = _count(details["ttl_12h_tokens"])
    if short + long != written:
        raise LunaError("known_usage_extension_invalid")
    return short, long


def _output_texts(raw: Mapping[str, object]) -> list[str]:
    items = raw.get("output")
    if not isinstance(items, list):
        raise LunaError("invalid_model_schema")
    texts: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            raise LunaError("invalid_model_schema")
        kind = item.get("type")
        if kind == "reasoning":
            continue
        if kind != "message":
            raise LunaError("invalid_model_schema")
        if item.get("role") != "assistant" or item.get("status") != "completed":
            raise LunaError("invalid_model_schema")
        content = item.get("content")
        if not isinstance(content, list):
            raise LunaError("invalid_model_schema")
        for part in content:
            if not isinstance(part, dict):
                raise LunaError("invalid_model_schema")
            if part.get("type") == "refusal":
                raise LunaError("output_refusal")
            if part.get("type") != "output_text" or not isinstance(
                part.get("text"), str
            ):
                raise LunaError("invalid_model_schema")
            texts.append(part["text"])
    return texts


def _json_object(text: str) -> dict[str, object]:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        keys = [key for key, _value in items]
        if len(keys) != len(set(keys)):
            raise LunaError("invalid_response")
        return dict(items)

    def reject_constant(_name: str) -> None:
        raise LunaError("invalid_response")

    try:
        value = json.loads(
            text, object_pairs_hook=pairs, parse_constant=reject_constant
        )
    except LunaError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError):
        raise LunaError("invalid_response") from None
    if not isinstance(value, dict):
        raise LunaError("invalid_model_schema")
    return value


async def _post(
    encoded: bytes,
    request_id: str,
    runtime: LunaRuntime,
    transport: httpx.AsyncBaseTransport,
    admission: LiveAdmission | None,
) -> tuple[Mapping[str, object], str | None, str | None]:
    headers = {
        "Authorization": f"Bearer {runtime.api_key}",
        "Content-Type": "application/json",
        "X-Client-Request-Id": request_id,
    }
    try:
        async with responses_client(transport, admission=admission) as client:
            # stream=True bounds the HTTP read. The JSON body still has stream false.
            async with client.stream(
                "POST", RESPONSES_PATH, content=encoded, headers=headers
            ) as response:
                body = await _read_bounded(response)
                status_code = response.status_code
                request_header = _safe_id(
                    response.headers.get("x-request-id"), _REQUEST_ID, runtime
                )
    except LunaError:
        raise
    except httpx.TimeoutException:
        raise LunaError("provider_transport_error") from None
    except httpx.TransportError:
        raise LunaError("provider_transport_error") from None
    if status_code in (401, 403):
        raise LunaError("provider_authentication_or_access_unavailable")
    if status_code == 404:
        raise LunaError("provider_model_or_endpoint_unavailable")
    if status_code == 429:
        raise LunaError("provider_capacity")
    if status_code != 200:
        raise LunaError("provider_service_unavailable")
    try:
        payload = _json_object(body.decode("utf-8"))
    except UnicodeError:
        raise LunaError("invalid_response") from None
    return (
        payload,
        _safe_id(_text(payload.get("id")), _RESPONSE_ID, runtime),
        request_header,
    )


async def _read_bounded(response: httpx.Response) -> bytes:
    """Stop pulling once the body exceeds 64KiB. The caller must not pre-read it."""
    declared = response.headers.get("content-length")
    if declared is not None and (
        not declared.isdigit() or int(declared) > MAX_RESPONSE_BYTES
    ):
        raise LunaError("unbounded_response")
    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > MAX_RESPONSE_BYTES:
            raise LunaError("unbounded_response")
        chunks.append(chunk)
    return b"".join(chunks)


def _reject_unpriced_tool_fee(payload: Mapping[str, object]) -> None:
    """Accept only omitted, null, or documented zero tool usage.

    ``tool_usage`` is separate from base ``usage``. A zero count is not a new
    fee. A nonzero, unknown, or badly shaped tool fee keeps the reservation.
    """
    if "tool_usage" not in payload or payload["tool_usage"] is None:
        return
    raw = payload["tool_usage"]
    if not isinstance(raw, dict) or set(raw) - _TOOL_FAMILIES:
        raise LunaError("unknown_billing_component")
    for name, value in raw.items():
        if name == "web_search":
            _zero_web_search(value)
        else:
            _zero_image_gen(value)


def _exact_zero(value: object) -> None:
    if type(value) is not int or value != 0:
        raise LunaError("unknown_billing_component")


def _zero_web_search(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {"num_requests"}:
        raise LunaError("unknown_billing_component")
    _exact_zero(value["num_requests"])


def _zero_image_gen(value: object) -> None:
    if not isinstance(value, dict):
        raise LunaError("unknown_billing_component")
    required = _IMAGE_GEN_COUNTS | {"input_tokens_details"}
    if set(value) - (required | _IMAGE_GEN_OPTIONAL) or not required <= set(value):
        raise LunaError("unknown_billing_component")
    for key in _IMAGE_GEN_COUNTS:
        _exact_zero(value[key])
    details = value["input_tokens_details"]
    if not isinstance(details, dict) or set(details) != {"image_tokens", "text_tokens"}:
        raise LunaError("unknown_billing_component")
    _exact_zero(details["image_tokens"])
    _exact_zero(details["text_tokens"])
    cached = value.get("cached_tokens_details")
    if cached is not None:
        if not isinstance(cached, dict) or set(cached) != {
            "cached_image_tokens",
            "cached_text_tokens",
        }:
            raise LunaError("unknown_billing_component")
        _exact_zero(cached["cached_image_tokens"])
        _exact_zero(cached["cached_text_tokens"])
    outgoing = value.get("output_tokens_details")
    if outgoing is not None:
        if not isinstance(outgoing, dict) or set(outgoing) != {
            "image_tokens",
            "text_tokens",
        }:
            raise LunaError("unknown_billing_component")
        _exact_zero(outgoing["image_tokens"])
        _exact_zero(outgoing["text_tokens"])


def _contradicts_terminal(status: str, payload: Mapping[str, object]) -> bool:
    """Reject a completed body that also carries an error or incomplete details."""
    error = payload.get("error", None)
    details = payload.get("incomplete_details", None)
    if status == "completed":
        return error is not None or details is not None
    if error is not None and status not in {"failed", "cancelled"}:
        return True
    return details is not None and status != "incomplete"


def _account(
    payload: Mapping[str, object],
    request: AnswerRequest,
    policy: LunaPolicy,
    ledger: Ledger,
    request_id: str,
    response_id: str | None,
    request_header: str | None,
    latency: float,
) -> GroundedDraft:
    if payload.get("model") != MODEL:
        ledger.retain_unknown(request_id)
        raise LunaError("unverified_returned_model")
    if payload.get("service_tier") != "default":
        ledger.retain_unknown(request_id)
        raise LunaError("unverified_service_tier")
    status = payload.get("status")
    if status not in _TERMINAL:
        ledger.retain_unknown(request_id)
        raise LunaError("unknown_response_terminal_state")
    try:
        usage = parse_usage(payload)
        _reject_unpriced_tool_fee(payload)
    except LunaError:
        ledger.retain_unknown(request_id)
        raise
    try:
        ledger.settle(
            policy,
            request_id,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_tokens=usage.cached_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            returned_model=MODEL,
            provider_request_id=request_header,
            provider_response_id=response_id,
            latency_seconds=latency,
        )
    except BudgetError:
        _retain_quietly(ledger, request_id)
        raise
    if _contradicts_terminal(status, payload):
        raise LunaError("contradictory_response")
    if status != "completed":
        raise LunaError(f"provider_{status}")
    return parse_luna_output(payload, request)


def _retain_quietly(ledger: Ledger, request_id: str) -> None:
    try:
        ledger.retain_unknown(request_id)
    except BudgetError:
        return


def _safe_id(
    value: object, pattern: re.Pattern[str], runtime: LunaRuntime
) -> str | None:
    if (
        not isinstance(value, str)
        or runtime.api_key in value
        or pattern.fullmatch(value) is None
    ):
        return None
    return value


def _text(value: object) -> str | None:
    return value if isinstance(value, str) else None
