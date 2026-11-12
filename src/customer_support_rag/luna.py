"""Fixed gpt-6-luna adapter. Tests inject a mock transport and a tmp ledger.

Importing this module does not read a key, open a database, or make a network
call. ``generate_draft`` accepts only a mock transport and is the offline
ledger path. The one-shot runner is the only live entry. ``store=false`` does
not mean zero data retention.
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
from pathlib import Path

import httpx

from customer_support_rag.budget import (
    OUTPUT_RESERVATION,
    TOTAL_RESERVATION,
    BudgetError,
    Ledger,
    LunaPolicy,
    LunaRuntime,
)
from customer_support_rag.eval_diagnostic import (
    DiagnosticCaptureError,
    EvalDiagnostic,
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
# Offline ledger reservation candidate only. It is not a live token-cost gate
# and the count API is not called. The live entry does not consult it.
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
_QUOTA_CODES = frozenset(
    {
        "insufficient_quota",
        "credit_balance_exhausted",
        "organization_spend_limit_exceeded",
        "project_spend_limit_exceeded",
        "organization_usage_limit_exceeded",
    }
)
_RATE_CODES = frozenset({"rate_limit_exceeded", "rate_limit", "slow_down"})


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


def responses_client(transport: httpx.AsyncBaseTransport) -> httpx.AsyncClient:
    """Build the fixed client for a mock transport.

    A factory or caller-built network transport is refused before any request.
    """
    if type(transport) is not httpx.MockTransport:
        raise LunaError("paid_blocked")
    return _open_client(transport)


def _open_client(transport: httpx.AsyncBaseTransport) -> httpx.AsyncClient:
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

    The allowance is only the offline candidate above. A body that cannot fit
    under the historical total is refused instead of truncated.
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
) -> GroundedDraft:
    """Reserve on the offline ledger, send once through a mock, then resolve.

    Empty evidence returns the fixed handoff and does not reserve. Failures
    after a durable send keep the reservation. Nothing is retried. A network
    transport is refused before reserve. The live entry does not call this.
    """
    if type(transport) is not httpx.MockTransport:
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
            _post(encoded, request_id, runtime, transport), deadline
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
) -> tuple[Mapping[str, object], str | None, str | None]:
    body, status_code, request_header = await _exchange(
        encoded, request_id, runtime.api_key, transport
    )
    if status_code != 200:
        raise LunaError(_provider_status_code(status_code, body))
    return _completed_payload(body, runtime.api_key, request_header)


async def _post_live_once(
    encoded: bytes,
    request: AnswerRequest,
    *,
    api_key: str,
    transport: httpx.AsyncBaseTransport,
    request_id: str,
    deadline: float,
    diagnostic: EvalDiagnostic | None = None,
) -> tuple[GroundedDraft | None, _UsageNote, str | None]:
    """Post one live body. This does not reserve or write a ledger."""
    if not _factory_transport(transport):
        raise LunaError("paid_blocked")
    _require_deadline(deadline)
    try:
        body, status_code, _request_header = await asyncio.wait_for(
            _exchange(encoded, request_id, api_key, transport), deadline
        )
    except TimeoutError:
        raise LunaError("logical_timeout") from None
    record = None
    if diagnostic is not None:
        try:
            record = diagnostic.capture(body, http_status=status_code, api_key=api_key)
        except DiagnosticCaptureError as exc:
            raise LunaError(exc.args[0]) from None
    if status_code != 200:
        code = _provider_status_code(status_code, body)
        if diagnostic is not None and record is not None:
            _finish_diagnostic(
                diagnostic,
                record,
                parser_stage="http_status",
                safe_code=(
                    code
                    if code in {"provider_service_unavailable", "provider_capacity"}
                    else "provider_service_unavailable"
                ),
            )
        raise LunaError(code)
    try:
        payload, _response_id, _request_header = _completed_payload(
            body, api_key, _request_header
        )
        outcome = _live_outcome(payload, request)
    except LunaError as exc:
        if diagnostic is not None and record is not None:
            stage = (
                "output_parser"
                if exc.code == "invalid_model_schema"
                else "provider_envelope"
            )
            safe = (
                exc.code
                if exc.code
                in {
                    "invalid_model_schema",
                    "invalid_response",
                    "unknown_citation",
                    "output_refusal",
                }
                else "invalid_response"
            )
            _finish_diagnostic(diagnostic, record, parser_stage=stage, safe_code=safe)
        raise
    if diagnostic is not None and record is not None:
        stopped = outcome[2]
        if stopped is None:
            stage, code = "parsed", "parsed"
        elif stopped == "output_refusal":
            stage, code = "refusal", stopped
        elif stopped in {
            "invalid_model_schema",
            "invalid_response",
            "unknown_citation",
        }:
            stage, code = "output_parser", stopped
        else:
            stage = "provider_envelope"
            code = (
                stopped
                if stopped
                in {
                    "provider_incomplete",
                    "provider_failed",
                    "provider_cancelled",
                    "contradictory_response",
                }
                else "invalid_response"
            )
        _finish_diagnostic(diagnostic, record, parser_stage=stage, safe_code=code)
    return outcome


def _finish_diagnostic(
    diagnostic: EvalDiagnostic,
    record: Path,
    *,
    parser_stage: str,
    safe_code: str | None,
) -> None:
    """Keep diagnostic persistence failures inside Luna's safe error contract."""
    try:
        diagnostic.finish(record, parser_stage=parser_stage, safe_code=safe_code)
    except DiagnosticCaptureError as exc:
        code = exc.args[0] if exc.args else "diagnostic_storage_unavailable"
        if code not in {
            "diagnostic_outcome_invalid",
            "diagnostic_record_missing",
            "diagnostic_storage_unavailable",
        }:
            code = "diagnostic_storage_unavailable"
        raise LunaError(code) from None
    except Exception:
        raise LunaError("diagnostic_storage_unavailable") from None


def _require_deadline(deadline: float) -> None:
    if (
        type(deadline) not in (int, float)
        or not math.isfinite(deadline)
        or deadline <= 0
    ):
        raise LunaError("runtime_unavailable")


async def _exchange(
    encoded: bytes,
    request_id: str,
    api_key: str,
    transport: httpx.AsyncBaseTransport,
) -> tuple[bytes, int, str | None]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Client-Request-Id": request_id,
    }
    try:
        async with _open_client(transport) as client:
            # stream=True bounds the HTTP read. The JSON body still has stream false.
            async with client.stream(
                "POST", RESPONSES_PATH, content=encoded, headers=headers
            ) as response:
                body = await _read_bounded(response)
                status_code = response.status_code
                request_header = _public_id(
                    response.headers.get("x-request-id"), _REQUEST_ID, api_key
                )
    except LunaError:
        raise
    except httpx.TimeoutException:
        raise LunaError("provider_transport_error") from None
    except httpx.TransportError:
        raise LunaError("provider_transport_error") from None
    return body, status_code, request_header


def _completed_payload(
    body: bytes, api_key: str, request_header: str | None
) -> tuple[Mapping[str, object], str | None, str | None]:
    try:
        payload = _json_object(body.decode("utf-8"))
    except UnicodeError:
        raise LunaError("invalid_response") from None
    return (
        payload,
        _public_id(_text(payload.get("id")), _RESPONSE_ID, api_key),
        request_header,
    )


def _provider_status_code(status_code: int, body: bytes) -> str:
    if status_code in (401, 403):
        return "provider_authentication_or_access_unavailable"
    if status_code == 404:
        return "provider_model_or_endpoint_unavailable"
    if status_code == 429:
        return _limit_code(body)
    return "provider_service_unavailable"


def _limit_code(body: bytes) -> str:
    """Map a 429 body to a safe code. An unrecognized 429 is not quota."""
    error = _error_object(body)
    if error is None:
        return "provider_capacity"
    code = error.get("code")
    kind = error.get("type")
    if isinstance(code, str) and code in _QUOTA_CODES:
        return "insufficient_quota"
    if isinstance(code, str) and code in _RATE_CODES:
        return "rate_limit"
    if kind == "insufficient_quota":
        return "insufficient_quota"
    if kind == "rate_limit_error":
        return "rate_limit"
    return "provider_capacity"


def _error_object(body: bytes) -> Mapping[str, object] | None:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if not isinstance(error, dict):
        return None
    return error


_USAGE_OBSERVATION_CODES = frozenset(
    {
        "unknown_billing_component",
        "usage_missing",
        "usage_shape_invalid",
        "usage_totals_invalid",
        "known_usage_extension_invalid",
    }
)
_SAFE_FIELD = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,64}")


@dataclass(frozen=True)
class _UsageNote:
    """Recognized counts, or a safe incomplete record.

    ``usage is None`` means the counts were not identified. It is not zero.
    ``fields`` are dotted names only. Values are never copied. A key that is
    not a short identifier becomes ``unlisted_field``.
    """

    usage: Usage | None
    code: str | None
    fields: tuple[str, ...]

    def __post_init__(self) -> None:
        if (self.usage is None) == (self.code is None):
            raise LunaError("invalid_response")
        if self.code is None and self.fields:
            raise LunaError("invalid_response")


def _safe_field(prefix: str, key: object) -> str:
    if isinstance(key, str) and _SAFE_FIELD.fullmatch(key):
        return f"{prefix}.{key}"
    return f"{prefix}.unlisted_field"


def _extra_fields(value: object, allowed: frozenset[str], prefix: str) -> list[str]:
    if not isinstance(value, dict):
        return []
    return [_safe_field(prefix, key) for key in value if key not in allowed]


def _billing_gaps(payload: Mapping[str, object]) -> tuple[str, ...]:
    """Name billing keys outside the pinned sets. This stores no values."""
    found: list[str] = []
    usage = payload.get("usage")
    found.extend(_extra_fields(usage, _USAGE_TOP, "usage"))
    if isinstance(usage, dict):
        incoming = usage.get("input_tokens_details")
        found.extend(
            _extra_fields(incoming, _INPUT_DETAILS, "usage.input_tokens_details")
        )
        if isinstance(incoming, dict):
            found.extend(
                _extra_fields(
                    incoming.get("cache_write_tokens_details"),
                    _TTL,
                    "usage.input_tokens_details.cache_write_tokens_details",
                )
            )
        found.extend(
            _extra_fields(
                usage.get("output_tokens_details"),
                frozenset({"reasoning_tokens"}),
                "usage.output_tokens_details",
            )
        )
    tool = payload.get("tool_usage")
    if "tool_usage" in payload and tool is not None and not isinstance(tool, dict):
        found.append("tool_usage")
    if isinstance(tool, dict):
        found.extend(_extra_fields(tool, _TOOL_FAMILIES, "tool_usage"))
        web = tool.get("web_search")
        if isinstance(web, dict):
            found.extend(
                _extra_fields(web, frozenset({"num_requests"}), "tool_usage.web_search")
            )
            if web.get("num_requests") not in (None, 0):
                found.append("tool_usage.web_search.num_requests")
        image = tool.get("image_gen")
        if isinstance(image, dict):
            allowed = _IMAGE_GEN_COUNTS | _IMAGE_GEN_OPTIONAL
            found.extend(_extra_fields(image, allowed, "tool_usage.image_gen"))
            if not _IMAGE_GEN_COUNTS <= set(image):
                found.append("tool_usage.image_gen")
            for key in _IMAGE_GEN_COUNTS:
                if image.get(key) not in (None, 0):
                    found.append(f"tool_usage.image_gen.{key}")
    return tuple(sorted(dict.fromkeys(found)))


def _observe_live_usage(payload: Mapping[str, object]) -> _UsageNote:
    """Read usage for a live answer. A billing-shape miss does not raise.

    The offline ledger path still calls ``parse_usage`` directly and keeps
    its strict rejection. This note never fills missing counts with zero.
    """
    try:
        usage = parse_usage(payload)
        _reject_unpriced_tool_fee(payload)
    except LunaError as exc:
        if exc.code not in _USAGE_OBSERVATION_CODES:
            raise
        fields = (
            _billing_gaps(payload) if exc.code == "unknown_billing_component" else ()
        )
        return _UsageNote(None, exc.code, fields)
    return _UsageNote(usage, None, ())


def _live_outcome(
    payload: Mapping[str, object], request: AnswerRequest
) -> tuple[GroundedDraft | None, _UsageNote, str | None]:
    if payload.get("model") != MODEL:
        raise LunaError("unverified_returned_model")
    if payload.get("service_tier") != "default":
        raise LunaError("unverified_service_tier")
    status = payload.get("status")
    if status not in _TERMINAL:
        raise LunaError("unknown_response_terminal_state")
    note = _observe_live_usage(payload)
    if _contradicts_terminal(str(status), payload):
        return None, note, "contradictory_response"
    if status != "completed":
        return None, note, f"provider_{status}"
    try:
        draft = parse_luna_output(payload, request)
    except LunaError as exc:
        return None, note, exc.code
    return draft, note, None


def _public_id(value: object, pattern: re.Pattern[str], secret: str) -> str | None:
    if (
        not isinstance(value, str)
        or secret in value
        or pattern.fullmatch(value) is None
    ):
        return None
    return value


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


def _text(value: object) -> str | None:
    return value if isinstance(value, str) else None
