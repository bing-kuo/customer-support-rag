"""Live entry tests. No real provider call and no historical ledger write."""

from __future__ import annotations

import asyncio
import hashlib
import json
import socket
import ssl
from pathlib import Path

import httpx
import pytest

from customer_support_rag import luna
from customer_support_rag.budget import (
    LunaRuntime,
    init_synthetic_ledger,
    load_policy,
    open_ledger,
)
from customer_support_rag.luna import (
    MAX_WIRE_BYTES,
    MODEL,
    LunaError,
    build_luna_body,
    generate_draft,
    network_transport,
    responses_client,
    wire_bytes,
    wire_composition,
)
from customer_support_rag.luna_live import (
    QUESTION,
    REQUEST_ID,
    hp_a_check,
    hp_a_once,
    prepare_hp_a_request,
)

KNOWLEDGE = Path(__file__).parents[1] / "knowledge"
KEY = "synthetic-key-not-live"
BODY_HASH = "ac5a515cdc5406c15906b913788cd895190564128b48c8117bb9fe97847fe36f"
ANSWER = "使用後以乾燥柔軟的布擦拭耳罩。請勿將耳機浸入水中，也不要使用溶劑清潔。"


@pytest.fixture(autouse=True)
def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    real_socket = socket.socket

    def guarded(family: int = socket.AF_INET, *args: object, **kwargs: object):
        if family != socket.AF_UNIX:
            raise AssertionError("unexpected socket")
        return real_socket(family, *args, **kwargs)

    monkeypatch.setattr(socket, "socket", guarded)
    monkeypatch.setattr(
        socket, "create_connection", lambda *_args, **_kwargs: guarded(-1)
    )


def test_hp_a_wire_is_one_fixed_body() -> None:
    request = prepare_hp_a_request(KNOWLEDGE)
    parts = wire_composition(request)
    body = build_luna_body(request)
    encoded = wire_bytes(body)
    assert request.question == QUESTION
    assert (
        request.sources[0].hit.chunk.char_start,
        request.sources[0].hit.chunk.char_end,
    ) == (1, 71)
    assert parts.policy_bytes == 623
    assert parts.data_bytes == 415
    assert parts.schema_bytes == 677
    assert parts.framing_bytes == 409
    assert parts.total_bytes == 2124
    assert parts.total_bytes <= MAX_WIRE_BYTES
    assert body["max_output_tokens"] == 512
    assert body["model"] == MODEL
    assert body["store"] is False
    assert body["tools"] == []
    assert hashlib.sha256(encoded).hexdigest() == BODY_HASH


def test_check_reports_the_key_without_opening_a_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_budget(monkeypatch)
    root = _key_only(tmp_path)
    report = hp_a_check(root, knowledge=KNOWLEDGE)
    assert report.key_present is True
    assert report.failures == ()
    assert report.wire is not None
    assert report.wire.total_bytes == 2124
    missing = hp_a_check(tmp_path / "empty", knowledge=KNOWLEDGE)
    assert missing.key_present is False
    assert "key_missing" in missing.failures


def test_missing_key_does_not_build_the_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = {"factory": 0}

    def factory() -> None:
        seen["factory"] += 1
        raise AssertionError("factory")

    monkeypatch.setattr("customer_support_rag.luna_live.network_transport", factory)
    with pytest.raises(LunaError) as blocked:
        asyncio.run(hp_a_once(tmp_path, knowledge=KNOWLEDGE))
    assert blocked.value.code == "paid_blocked"
    assert seen["factory"] == 0
    assert not (tmp_path / "runtime" / "cloud-budget.sqlite3").exists()


def test_live_call_does_not_read_policy_or_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_budget(monkeypatch)
    root = _key_only(tmp_path)
    (root / "runtime").mkdir()
    (root / "runtime" / "cloud-policy.json").write_text("{", encoding="utf-8")
    (root / "runtime" / "cloud-budget.sqlite3").write_bytes(b"not a database")
    calls: list[httpx.Request] = []

    async def handler(_self: object, request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=_envelope(_source_text(_source_id())))

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handler)
    result = asyncio.run(hp_a_once(root, knowledge=KNOWLEDGE))
    assert result.stopped is None
    assert result.draft is not None
    assert result.draft.answer == ANSWER
    assert result.usage is not None
    assert result.usage_code is None
    assert result.usage_fields == ()
    assert (result.usage.input_tokens, result.usage.output_tokens) == (12, 8)
    assert len(calls) == 1
    assert calls[0].method == "POST"
    assert calls[0].url.path == "/v1/responses"
    assert calls[0].url.host == "api.openai.com"
    assert calls[0].headers["x-client-request-id"] == REQUEST_ID
    posted = json.loads(calls[0].content.decode())
    assert posted["model"] == MODEL
    assert posted["store"] is False
    assert posted["tools"] == []
    assert posted["max_output_tokens"] == 512
    assert (root / "runtime" / "cloud-budget.sqlite3").read_bytes() == b"not a database"


def test_public_factory_does_not_reserve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _offline_ledger(tmp_path)
    policy = load_policy(root / "runtime" / "cloud-policy.json")
    runtime = LunaRuntime(policy, KEY, pricing_bound_verified=True)
    ledger = open_ledger(root / "runtime" / "cloud-budget.sqlite3")
    before = (root / "runtime" / "cloud-budget.sqlite3").read_bytes()
    called = {"handler": 0}

    async def handler(_self: object, _request: httpx.Request) -> httpx.Response:
        called["handler"] += 1
        raise AssertionError("handler")

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handler)
    transport = network_transport()
    try:
        with pytest.raises(LunaError) as blocked:
            asyncio.run(
                generate_draft(
                    prepare_hp_a_request(KNOWLEDGE),
                    request_id="bypass",
                    runtime=runtime,
                    ledger=ledger,
                    transport=transport,
                    scope="developer_smoke",
                )
            )
        assert blocked.value.code == "paid_blocked"
        with pytest.raises(LunaError) as client:
            responses_client(transport)
        assert client.value.code == "paid_blocked"
    finally:
        asyncio.run(transport.aclose())
    assert called["handler"] == 0
    assert (root / "runtime" / "cloud-budget.sqlite3").read_bytes() == before
    assert not hasattr(luna, "permit_live_transport")


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        (
            {"error": {"code": "insufficient_quota", "type": "insufficient_quota"}},
            "insufficient_quota",
        ),
        (
            {
                "error": {
                    "code": "credit_balance_exhausted",
                    "type": "insufficient_quota",
                    "message": "do-not-leak",
                }
            },
            "insufficient_quota",
        ),
        (
            {
                "error": {
                    "code": "project_spend_limit_exceeded",
                    "type": "insufficient_quota",
                    "message": "do-not-leak",
                }
            },
            "insufficient_quota",
        ),
        (
            {"error": {"code": "rate_limit_exceeded", "type": "rate_limit_error"}},
            "rate_limit",
        ),
        (
            {
                "error": {
                    "code": "slow_down",
                    "type": "rate_limit_error",
                    "message": "do-not-leak",
                }
            },
            "rate_limit",
        ),
        ({"error": {"type": "insufficient_quota"}}, "insufficient_quota"),
        ({"error": {"code": "server_busy", "type": "tokens"}}, "provider_capacity"),
        (None, "provider_capacity"),
    ],
)
def test_429_stops_once_and_does_not_write_a_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: object, code: str
) -> None:
    _forbid_budget(monkeypatch)
    root = _key_only(tmp_path)
    calls = {"n": 0}

    async def handler(_self: object, _request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if payload is None:
            return httpx.Response(429, headers={"retry-after": "30"})
        return httpx.Response(429, json=payload, headers={"retry-after": "30"})

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handler)
    with pytest.raises(LunaError) as stopped:
        asyncio.run(hp_a_once(root, knowledge=KNOWLEDGE))
    assert stopped.value.code == code
    assert "do-not-leak" not in str(stopped.value)
    assert calls["n"] == 1
    assert not (root / "runtime").exists()


def test_timeout_does_not_block_a_later_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_budget(monkeypatch)
    root = _key_only(tmp_path)
    calls = {"n": 0}

    async def handler(_self: object, _request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            await asyncio.sleep(30)
        return httpx.Response(200, json=_envelope(_source_text(_source_id())))

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handler)
    with pytest.raises(LunaError) as first:
        asyncio.run(hp_a_once(root, knowledge=KNOWLEDGE, deadline=0.05))
    assert first.value.code == "logical_timeout"
    result = asyncio.run(hp_a_once(root, knowledge=KNOWLEDGE))
    assert result.stopped is None
    assert result.draft is not None
    assert result.usage is not None
    assert result.usage_code is None
    assert calls["n"] == 2
    assert not (root / "runtime").exists()


def test_quality_failure_keeps_usage_without_a_ledger_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_budget(monkeypatch)
    root = _key_only(tmp_path)

    async def handler(_self: object, _request: httpx.Request) -> httpx.Response:
        text = json.dumps(
            {
                "status": "answer",
                "answer": ANSWER,
                "reason": None,
                "claims": [
                    {
                        "answer_start": 0,
                        "answer_end": len(ANSWER),
                        "source_id": "missing-source",
                        "source_start": 0,
                        "source_end": 1,
                    }
                ],
            }
        )
        return httpx.Response(200, json=_envelope(text))

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handler)
    result = asyncio.run(hp_a_once(root, knowledge=KNOWLEDGE))
    assert result.draft is None
    assert result.stopped == "unknown_citation"
    assert result.usage is not None
    assert result.usage_code is None
    assert result.usage.total_tokens == 20
    assert not (root / "runtime").exists()


def test_unknown_billing_field_keeps_the_answer_and_hides_the_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_budget(monkeypatch)
    root = _key_only(tmp_path)
    calls = {"n": 0}

    async def handler(_self: object, _request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        payload = _envelope(_source_text(_source_id()))
        usage = dict(payload["usage"])
        usage["priority_processing"] = "do-not-leak"
        usage["sk-secret"] = "do-not-leak"
        payload["usage"] = usage
        payload["tool_usage"] = {"audio_gen": {"seconds": 3}}
        return httpx.Response(200, json=payload)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handler)
    result = asyncio.run(hp_a_once(root, knowledge=KNOWLEDGE))
    assert calls["n"] == 1
    assert result.stopped is None
    assert result.draft is not None
    assert result.draft.answer == ANSWER
    source = result.draft.sources[0].hit.chunk
    assert source.index == 1
    assert (source.char_start, source.char_end) == (1, 71)
    assert "乾燥柔軟的布" in source.text
    assert "浸入水中" in source.text
    assert "溶劑" in source.text
    assert result.usage is None
    assert result.usage_code == "unknown_billing_component"
    assert result.usage_fields == (
        "tool_usage.audio_gen",
        "usage.priority_processing",
        "usage.unlisted_field",
    )
    visible = repr(result)
    assert "do-not-leak" not in visible
    assert "sk-secret" not in visible
    assert not (root / "runtime").exists()


def test_unknown_billing_field_still_rejects_a_bad_citation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_budget(monkeypatch)
    root = _key_only(tmp_path)
    calls = {"n": 0}

    async def handler(_self: object, _request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        text = json.dumps(
            {
                "status": "answer",
                "answer": ANSWER,
                "reason": None,
                "claims": [
                    {
                        "answer_start": 0,
                        "answer_end": len(ANSWER),
                        "source_id": "missing-source",
                        "source_start": 0,
                        "source_end": 1,
                    }
                ],
            }
        )
        payload = _envelope(text)
        usage = dict(payload["usage"])
        usage["priority_processing"] = "do-not-leak"
        payload["usage"] = usage
        return httpx.Response(200, json=payload)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handler)
    result = asyncio.run(hp_a_once(root, knowledge=KNOWLEDGE))
    assert calls["n"] == 1
    assert result.draft is None
    assert result.stopped == "unknown_citation"
    assert result.usage is None
    assert result.usage_code == "unknown_billing_component"
    assert result.usage_fields == ("usage.priority_processing",)
    assert "do-not-leak" not in repr(result)
    assert not (root / "runtime").exists()


def test_429_answer_body_is_not_returned_or_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_budget(monkeypatch)
    root = _key_only(tmp_path)
    calls = {"n": 0}

    async def handler(_self: object, _request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(
            429,
            json={
                "error": {
                    "code": "rate_limit_exceeded",
                    "type": "rate_limit_error",
                    "message": "do-not-leak",
                },
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": ANSWER}],
                    }
                ],
            },
            headers={"retry-after": "30"},
        )

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handler)
    with pytest.raises(LunaError) as stopped:
        asyncio.run(hp_a_once(root, knowledge=KNOWLEDGE))
    assert stopped.value.code == "rate_limit"
    assert calls["n"] == 1
    assert "do-not-leak" not in str(stopped.value)
    assert ANSWER not in str(stopped.value)


def test_refusal_still_stops_without_returning_the_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_budget(monkeypatch)
    root = _key_only(tmp_path)
    calls = {"n": 0}

    async def handler(_self: object, _request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        payload = _envelope("")
        payload["output"] = [
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "refusal", "refusal": "no"}],
            }
        ]
        return httpx.Response(200, json=payload)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handler)
    result = asyncio.run(hp_a_once(root, knowledge=KNOWLEDGE))
    assert calls["n"] == 1
    assert result.draft is None
    assert result.stopped == "output_refusal"
    assert result.usage is not None
    assert result.usage_code is None
    assert result.usage.total_tokens == 20


def test_factory_uses_tls_without_proxy_retries_or_redirects() -> None:
    transport = network_transport()
    try:
        pool = transport._pool
        assert pool._retries == 0
        assert type(pool).__name__ == "AsyncConnectionPool"
        context = pool._ssl_context
        assert isinstance(context, ssl.SSLContext)
        assert context.verify_mode == ssl.CERT_REQUIRED
        assert context.check_hostname is True
        client = responses_client(
            httpx.MockTransport(lambda _request: httpx.Response(204))
        )
        assert client.trust_env is False
        assert client.follow_redirects is False
        asyncio.run(client.aclose())
    finally:
        asyncio.run(transport.aclose())


def _forbid_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("budget")

    monkeypatch.setattr("customer_support_rag.budget.open_ledger", boom)
    monkeypatch.setattr("customer_support_rag.budget.load_policy", boom)
    monkeypatch.setattr("customer_support_rag.budget.load_runtime", boom)


def _key_only(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    env = root / ".env"
    env.write_text(f'# local\nexport OPENAI_API_KEY="{KEY}"\n', encoding="utf-8")
    env.chmod(0o600)
    return root


def _offline_ledger(root: Path) -> Path:
    _key_only(root)
    payload = {
        "version": 1,
        "model": "gpt-6-luna",
        "budget_id": "synthetic-budget",
        "cap_units": 20,
        "input_rate_units": 2,
        "output_rate_units": 2,
        "extra_units": 0,
        "pricing_revision": "gpt-6-luna-standard-2026-11-08-reviewed",
        "pricing_expires_at": "2099-01-02T03:04:05Z",
        "enabled_scopes": ["developer_smoke"],
        "enabled": True,
    }
    runtime = root / "runtime"
    runtime.mkdir()
    policy = runtime / "cloud-policy.json"
    policy.write_text(json.dumps(payload), encoding="utf-8")
    policy.chmod(0o600)
    init_synthetic_ledger(runtime / "cloud-budget.sqlite3", load_policy(policy))
    return root


def _source_id() -> str:
    request = prepare_hp_a_request(KNOWLEDGE)
    return request.evidence[0].source_id


def _envelope(text: str) -> dict[str, object]:
    return {
        "id": "resp_fixture",
        "model": MODEL,
        "status": "completed",
        "service_tier": "default",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
        "usage": {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
    }


def _source_text(source_id: str) -> str:
    quote = "請勿將耳機浸入水中，也不要使用溶劑清潔。"
    return json.dumps(
        {
            "status": "answer",
            "answer": ANSWER,
            "reason": None,
            "claims": [
                {
                    "answer_start": 0,
                    "answer_end": len(ANSWER),
                    "source_id": source_id,
                    "source_start": 23,
                    "source_end": 23 + len(quote),
                }
            ],
        },
        ensure_ascii=False,
    )
