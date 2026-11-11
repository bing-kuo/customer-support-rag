"""Host and Origin checks. Rejections must not read a key or call a model."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from customer_support_rag.api import create_app
from customer_support_rag.local_http import host_is_trusted, origin_is_same

ASK = {"product_id": "hp-a", "question": "HP-A 耳機可以用水洗嗎？"}


def test_untrusted_host_is_rejected_before_key_or_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_work(monkeypatch)
    client = TestClient(create_app(None), base_url="http://evil.example")
    response = client.post("/ask", json=ASK)
    assert response.status_code == 422
    payload = response.json()
    assert payload["code"] == "invalid_request"
    assert payload["request_id"].startswith("ask-")
    assert response.headers["content-type"].startswith("application/json")
    assert "text/html" not in response.headers["content-type"]
    health = client.get("/health")
    assert health.status_code == 422
    assert health.json()["code"] == "invalid_request"


def test_testserver_host_needs_explicit_configuration() -> None:
    production = TestClient(create_app(None), base_url="http://testserver")
    denied = production.get("/health")
    assert denied.status_code == 422
    assert denied.json()["code"] == "invalid_request"
    allowed = TestClient(
        create_app(None, extra_hosts=("testserver",)),
        base_url="http://testserver",
    )
    ok = allowed.get("/health")
    assert ok.status_code == 200
    assert ok.json()["status"] == "ok"


def test_forwarded_host_does_not_widen_trust(tmp_path: Path) -> None:
    root = _web_tree(tmp_path)
    local = TestClient(create_app(None, web_root=root), base_url="http://127.0.0.1")
    spoofed = local.get("/", headers={"x-forwarded-host": "evil.example"})
    assert spoofed.status_code == 422
    assert spoofed.json()["code"] == "invalid_request"
    assert "<!doctype html>" not in spoofed.text
    foreign = TestClient(
        create_app(None, web_root=root), base_url="http://evil.example"
    )
    rebound = foreign.get("/", headers={"x-forwarded-host": "127.0.0.1"})
    assert rebound.status_code == 422
    allowed = local.get("/")
    assert allowed.status_code == 200
    assert allowed.text == "<!doctype html><title>fixture</title>"


def test_malformed_ipv6_origin_is_invalid_request_not_server_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_work(monkeypatch)
    client = TestClient(
        create_app(None),
        base_url="http://127.0.0.1",
        raise_server_exceptions=False,
    )
    for origin in ("http://[", "http://[bad]"):
        response = client.post("/ask", json=ASK, headers={"origin": origin})
        assert response.status_code == 422
        payload = response.json()
        assert payload["code"] == "invalid_request"
        assert payload["request_id"].startswith("ask-")
        assert response.headers["content-type"].startswith("application/json")
        assert "Internal Server Error" not in response.text
        assert origin not in response.text


def test_unrelated_origin_parser_error_stays_a_server_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_work(monkeypatch)

    def broken(_value: str, *_args: object, **_kwargs: object) -> object:
        raise RuntimeError("parser bug")

    monkeypatch.setattr("customer_support_rag.local_http.urlparse", broken)
    client = TestClient(
        create_app(None),
        base_url="http://127.0.0.1",
        raise_server_exceptions=False,
    )
    response = client.post("/ask", json=ASK, headers={"origin": "http://127.0.0.1"})
    assert response.status_code == 500
    assert "invalid_request" not in response.text
    assert "parser bug" not in response.text


def test_null_and_cross_origin_ask_are_rejected_without_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_work(monkeypatch)
    client = TestClient(create_app(None), base_url="http://127.0.0.1")
    for origin in ("null", "http://evil.example", "http://127.0.0.1:8000"):
        response = client.post("/ask", json=ASK, headers={"origin": origin})
        assert response.status_code == 422
        payload = response.json()
        assert payload["code"] == "invalid_request"
        assert payload["request_id"].startswith("ask-")
        assert origin not in response.text


def test_cli_without_origin_reaches_ask_without_opening_a_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_work(monkeypatch)
    client = TestClient(create_app(None), base_url="http://127.0.0.1")
    response = client.post("/ask", json=ASK)
    assert response.status_code == 503
    assert response.json()["code"] == "ask_not_configured"


def test_matching_origin_reaches_ask_without_wildcard_cors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_work(monkeypatch)
    client = TestClient(create_app(None), base_url="http://127.0.0.1")
    response = client.post("/ask", json=ASK, headers={"origin": "http://127.0.0.1"})
    assert response.status_code == 503
    assert response.json()["code"] == "ask_not_configured"
    names = {name.lower() for name in response.headers}
    assert "access-control-allow-origin" not in names


def test_localhost_host_is_accepted_for_health() -> None:
    client = TestClient(create_app(None), base_url="http://localhost")
    assert client.get("/health").status_code == 200


def test_matching_ipv6_loopback_origin_is_trusted() -> None:
    assert host_is_trusted("[::1]")
    assert origin_is_same("http://[::1]", "http", "[::1]")
    assert not origin_is_same("http://[", "http", "127.0.0.1")
    assert not origin_is_same("http://[bad]", "http", "127.0.0.1")


def test_request_cannot_choose_the_web_root(tmp_path: Path) -> None:
    root = _web_tree(tmp_path)
    client = TestClient(create_app(None, web_root=root), base_url="http://127.0.0.1")
    leaked = client.get("/", headers={"x-web-root": str(tmp_path)})
    assert leaked.status_code == 200
    assert leaked.text == "<!doctype html><title>fixture</title>"
    query = client.get("/?web_root=/etc")
    assert query.status_code == 200
    assert query.text == "<!doctype html><title>fixture</title>"


def _forbid_work(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("work")

    monkeypatch.setattr("customer_support_rag.ask.EmbeddingStore.open_readonly", boom)
    monkeypatch.setattr("customer_support_rag.ask.OllamaEmbeddingAdapter", boom)
    monkeypatch.setattr("customer_support_rag.luna_live._read_key", boom)
    monkeypatch.setattr("customer_support_rag.luna_live.network_transport", boom)


def _web_tree(tmp_path: Path) -> Path:
    root = tmp_path / "web"
    assets = root / "assets"
    assets.mkdir(parents=True)
    (root / "index.html").write_text(
        "<!doctype html><title>fixture</title>", encoding="utf-8"
    )
    (assets / "style.css").write_text("body{color:#111}", encoding="utf-8")
    (assets / "app.js").write_text("window.__fixture=true;", encoding="utf-8")
    return root
