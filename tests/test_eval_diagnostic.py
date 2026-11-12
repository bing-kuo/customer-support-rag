"""Offline tests for the explicit, owner-only parser diagnostic capture."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from customer_support_rag import luna
from customer_support_rag.eval_diagnostic import (
    MAX_DIAGNOSTIC_BODY,
    DiagnosticCaptureError,
    EvalDiagnostic,
    replay_parser_body,
)
from customer_support_rag.knowledge.answering import prepare_answer
from customer_support_rag.luna import LunaError

REPO = Path(__file__).parents[1]
SHA = "a" * 64


def diagnostic(path: Path) -> EvalDiagnostic:
    return EvalDiagnostic.opt_in(
        path,
        candidate="baseline",
        case_id="faq-absent-baseline",
        code_sha=SHA,
        suite_version="dev-suite-v1",
        suite_sha256=SHA,
        rubric_version="semantic-rubric-v1",
        rubric_sha256=SHA,
    )


def test_capture_is_explicit_bounded_owner_only_and_header_free() -> None:
    root = REPO / ".data" / "test-eval-diagnostic"
    if root.exists():
        for child in root.iterdir():
            child.unlink()
        root.rmdir()
    recorder = diagnostic(root)
    body = b'{"output":[],"status":"completed"}'
    meta_path = recorder.capture(body, http_status=200, api_key="synthetic-secret")
    recorder.finish(
        meta_path, parser_stage="output_parser", safe_code="invalid_model_schema"
    )

    response_path = meta_path.with_name(
        meta_path.name.replace(".meta.json", ".response.json")
    )
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    assert response_path.read_bytes() == body
    assert metadata["parser_stage"] == "output_parser"
    assert metadata["safe_code"] == "invalid_model_schema"
    assert metadata["headers_saved"] is False
    assert metadata["authorization_saved"] is False
    assert os.stat(root).st_mode & 0o777 == 0o700
    assert os.stat(meta_path).st_mode & 0o777 == 0o600
    assert os.stat(response_path).st_mode & 0o777 == 0o600
    response_path.unlink()
    meta_path.unlink()
    root.rmdir()


@pytest.mark.parametrize(
    ("body", "key", "code"),
    [
        (
            b"x" * (MAX_DIAGNOSTIC_BODY + 1),
            "synthetic-secret",
            "diagnostic_body_too_large",
        ),
        (
            b'{"error":"synthetic-secret"}',
            "synthetic-secret",
            "diagnostic_secret_in_body",
        ),
    ],
)
def test_rejected_capture_leaves_no_record(body: bytes, key: str, code: str) -> None:
    root = REPO / ".data" / "test-eval-diagnostic-rejected"
    if root.exists():
        for child in root.iterdir():
            child.unlink()
        root.rmdir()
    recorder = diagnostic(root)
    with pytest.raises(DiagnosticCaptureError, match=code):
        recorder.capture(body, http_status=200, api_key=key)
    assert list(root.iterdir()) == []
    root.rmdir()


def test_default_path_does_not_auto_capture_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = REPO / ".data" / "must-not-auto-capture"
    monkeypatch.setenv("CUSTOMER_SUPPORT_RAG_EVAL_DIAGNOSTIC_DIR", str(root))
    assert not root.exists()
    # Opt-in requires constructing EvalDiagnostic and passing it to the live API.
    assert not root.exists()


@pytest.mark.parametrize(
    ("part", "expected"),
    [
        (
            {
                "type": "output_text",
                "text": (
                    '{"status":"handoff","answer":null,"source_ids":[],'
                    '"reason":"no_evidence"}'
                ),
            },
            "handoff",
        ),
        ({"type": "refusal", "refusal": "synthetic refusal"}, "output_refusal"),
        ({"type": "output_text", "text": "not-json"}, "invalid_model_schema"),
    ],
)
def test_saved_response_can_replay_through_original_parser(
    part: dict[str, str], expected: str
) -> None:
    request = prepare_answer("synthetic no-evidence query", product_id="hp-a", hits=[])
    body = json.dumps(
        {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [part],
                }
            ]
        }
    ).encode()
    if expected == "handoff":
        assert replay_parser_body(body, request).status == "handoff"
    else:
        with pytest.raises(LunaError, match=expected):
            replay_parser_body(body, request)


def test_capture_failure_is_a_safe_stop_and_cannot_be_recorded_as_success() -> None:
    recorder = diagnostic(REPO / ".data" / "test-eval-diagnostic-failure")
    with pytest.raises(DiagnosticCaptureError, match="diagnostic_secret_in_body"):
        recorder.capture(
            b"synthetic-secret", http_status=200, api_key="synthetic-secret"
        )
    with pytest.raises(DiagnosticCaptureError, match="diagnostic_record_missing"):
        recorder.finish(
            recorder.root / "missing.meta.json",
            parser_stage="parsed",
            safe_code="parsed",
        )
    recorder.root.rmdir()


def test_live_path_capture_failure_stops_after_single_mock_exchange(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = diagnostic(REPO / ".data" / "test-eval-diagnostic-live-failure")
    calls = 0

    def fail_write(_path: Path, _data: bytes) -> None:
        raise OSError("synthetic disk failure")

    monkeypatch.setattr(EvalDiagnostic, "_write_new", staticmethod(fail_write))

    async def mock_exchange(*_args: object, **_kwargs: object):
        nonlocal calls
        calls += 1
        return b'{"error":"synthetic-secret"}', 200, None

    monkeypatch.setattr(luna, "_factory_transport", lambda _transport: True)
    monkeypatch.setattr(luna, "_exchange", mock_exchange)
    request = prepare_answer("synthetic query", product_id="hp-a", hits=[])

    async def run():
        return await luna._post_live_once(
            b"{}",
            request,
            api_key="synthetic-key",
            transport=object(),  # type: ignore[arg-type]
            request_id="synthetic-request",
            deadline=1,
            diagnostic=recorder,
        )

    with pytest.raises(LunaError, match="diagnostic_storage_unavailable"):
        asyncio.run(run())
    assert calls == 1
    assert list(recorder.root.iterdir()) == []
    recorder.root.rmdir()
