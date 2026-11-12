"""Offline tests for the explicit, owner-only parser diagnostic capture."""

from __future__ import annotations

import asyncio
import json
import os
import socket
from pathlib import Path

import pytest

from customer_support_rag import eval_diagnostic as eval_diagnostic_module
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
GIT_SHA = "a" * 40


@pytest.fixture(autouse=True)
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject_connect(*_args: object, **_kwargs: object) -> None:
        pytest.fail("diagnostic tests must remain offline")

    monkeypatch.setattr(socket.socket, "connect", reject_connect)


def diagnostic(path: Path) -> EvalDiagnostic:
    return EvalDiagnostic.opt_in(
        path,
        candidate="baseline",
        case_id="faq-absent-baseline",
        code_sha=GIT_SHA,
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


def test_opt_in_rejects_symlink_ancestor_without_touching_target(
    tmp_path: Path,
) -> None:
    link = REPO / ".data" / "diagnostic-symlink-ancestor"
    outside = tmp_path / "outside"
    outside.mkdir()
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(
        DiagnosticCaptureError, match="diagnostic_root_not_gitignored_evidence"
    ):
        diagnostic(link / "capture")
    assert list(outside.iterdir()) == []
    link.unlink()


def test_opt_in_rejects_insecure_existing_mode_without_chmod(tmp_path: Path) -> None:
    root = REPO / ".data" / "diagnostic-insecure-existing"
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o755)
    with pytest.raises(DiagnosticCaptureError, match="diagnostic_storage_unavailable"):
        diagnostic(root)
    assert root.stat().st_mode & 0o777 == 0o755
    root.chmod(0o700)
    root.rmdir()


def test_opt_in_rejects_group_or_world_writable_ancestor_without_chmod() -> None:
    parent = REPO / ".data" / "diagnostic-writable-ancestor"
    parent.mkdir(parents=True, exist_ok=True)
    parent.chmod(0o777)
    with pytest.raises(DiagnosticCaptureError, match="diagnostic_storage_unavailable"):
        diagnostic(parent / "capture")
    assert parent.stat().st_mode & 0o777 == 0o777
    parent.chmod(0o700)
    parent.rmdir()


def test_capture_rejects_root_replaced_by_symlink(tmp_path: Path) -> None:
    root = REPO / ".data" / "diagnostic-replaced-root"
    if root.is_symlink():
        root.unlink()
    elif root.exists():
        root.rmdir()
    recorder = diagnostic(root)
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    root.rmdir()
    root.symlink_to(outside, target_is_directory=True)
    with pytest.raises(DiagnosticCaptureError, match="diagnostic_storage_unavailable"):
        recorder.capture(b"{}", http_status=200, api_key="synthetic-secret")
    assert list(outside.iterdir()) == []
    root.unlink()


@pytest.mark.parametrize(
    "body",
    [
        b'{"error":"synthetic-secret"}',
        b'{"error":"synthetic-\\u0073ecret"}',
        b'{"error":"synthetic-\\\\u0073ecret"}',
        b'{"nested":{"error":"synthetic-\\u0073ecret"}}',
        b'"{\\"error\\":\\"synthetic-\\\\u0073ecret\\"}"',
    ],
)
def test_capture_rejects_json_escape_key_shapes(body: bytes) -> None:
    root = REPO / ".data" / "test-eval-diagnostic-key-shape"
    if root.exists():
        for child in root.iterdir():
            child.unlink()
        root.rmdir()
    root.mkdir(parents=True, mode=0o700)
    recorder = diagnostic(root)
    with pytest.raises(DiagnosticCaptureError, match="diagnostic_secret_in_body"):
        recorder.capture(body, http_status=200, api_key="synthetic-secret")
    assert list(root.iterdir()) == []
    root.rmdir()


@pytest.mark.parametrize(
    ("field", "value", "accepted"),
    [
        ("code_sha", "a" * 40, True),
        ("code_sha", "a" * 64, False),
        ("suite_sha256", "a" * 64, True),
        ("suite_sha256", "a" * 40, False),
        ("rubric_sha256", "b" * 64, True),
        ("rubric_sha256", "b" * 40, False),
    ],
)
def test_diagnostic_hash_fields_use_their_own_lengths(
    field: str, value: str, accepted: bool
) -> None:
    root = REPO / ".data" / "test-eval-diagnostic-hash-length"
    if root.exists():
        root.rmdir()
    identity = {
        "candidate": "baseline",
        "case_id": "faq-absent-baseline",
        "code_sha": GIT_SHA,
        "suite_version": "dev-suite-v1",
        "suite_sha256": SHA,
        "rubric_version": "semantic-rubric-v1",
        "rubric_sha256": SHA,
    }
    identity[field] = value
    if accepted:
        recorder = EvalDiagnostic.opt_in(root, **identity)  # type: ignore[arg-type]
        assert recorder.root == root
        root.rmdir()
    else:
        with pytest.raises(DiagnosticCaptureError, match="diagnostic_identity_invalid"):
            EvalDiagnostic.opt_in(root, **identity)  # type: ignore[arg-type]
        assert not root.exists()


def test_finish_failure_is_a_safe_luna_error_after_one_exchange(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = REPO / ".data" / "test-eval-diagnostic-finish-failure"
    if root.exists():
        for child in root.iterdir():
            child.unlink()
        root.rmdir()
    recorder = diagnostic(root)
    calls = 0

    original_write_at = eval_diagnostic_module._write_new_at
    writes = 0

    def fail_finish_write(root_fd: int, name: str, data: bytes) -> None:
        nonlocal writes
        writes += 1
        if writes == 3:
            raise OSError("synthetic raw persistence exception")
        original_write_at(root_fd, name, data)

    async def mock_exchange(*_args: object, **_kwargs: object):
        nonlocal calls
        calls += 1
        return (
            b'{"output":[{"type":"message","role":"assistant","status":"completed",'
            b'"content":[{"type":"output_text","text":"{\\"status\\":\\"handoff\\",'
            b'\\"answer\\":null,\\"source_ids\\":[],\\"reason\\":\\"no_evidence\\"}"}]}]}',
            200,
            None,
        )

    monkeypatch.setattr(eval_diagnostic_module, "_write_new_at", fail_finish_write)
    monkeypatch.setattr(luna, "_factory_transport", lambda _transport: True)
    monkeypatch.setattr(luna, "_exchange", mock_exchange)
    request = prepare_answer("synthetic query", product_id="hp-a", hits=[])

    with pytest.raises(LunaError, match="diagnostic_storage_unavailable") as exc:
        asyncio.run(
            luna._post_live_once(
                b"{}",
                request,
                api_key="synthetic-key",
                transport=object(),  # type: ignore[arg-type]
                request_id="synthetic-request",
                deadline=1,
                diagnostic=recorder,
            )
        )
    assert str(exc.value) == "diagnostic_storage_unavailable"
    assert calls == 1
    for child in root.iterdir():
        child.unlink()
    root.rmdir()
