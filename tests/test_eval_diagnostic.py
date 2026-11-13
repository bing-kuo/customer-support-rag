"""Offline tests for the explicit, owner-only parser diagnostic capture."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
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
from customer_support_rag.luna_live import _key_shape

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
                    '{"status":"handoff","answer":null,"reason":"no_evidence",'
                    '"claims":[]}'
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

    def fail_write(_root_fd: int, _name: str, _data: bytes) -> None:
        raise OSError("synthetic disk failure")

    monkeypatch.setattr(eval_diagnostic_module, "_write_new_at", fail_write)

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


def test_opt_in_normalizes_dotdot_before_gitignored_boundary_check() -> None:
    escaped = REPO / ".data" / ".." / ".venv" / "diagnostic-dotdot-fixture"
    with pytest.raises(
        DiagnosticCaptureError, match="diagnostic_root_not_gitignored_evidence"
    ):
        diagnostic(escaped)
    assert not escaped.exists()


def test_ancestor_replacement_cannot_redirect_capture_or_finish(
    tmp_path: Path,
) -> None:
    capture_parent = REPO / ".data" / "diagnostic-ancestor-capture"
    capture_root = capture_parent / "capture"
    capture_pinned = capture_parent.with_name("diagnostic-ancestor-capture-pinned")
    outside_capture = tmp_path / "outside-capture"
    outside_capture.mkdir(mode=0o700)
    if capture_parent.exists():
        shutil.rmtree(capture_parent)
    if capture_pinned.exists():
        shutil.rmtree(capture_pinned)
    recorder = diagnostic(capture_root)
    capture_parent.rename(capture_pinned)
    (outside_capture / "capture").mkdir(mode=0o700)
    capture_parent.symlink_to(outside_capture, target_is_directory=True)
    try:
        with pytest.raises(
            DiagnosticCaptureError, match="diagnostic_storage_unavailable"
        ):
            recorder.capture(b"{}", http_status=200, api_key="synthetic-secret")
        assert list((outside_capture / "capture").iterdir()) == []
    finally:
        recorder.close()
        capture_parent.unlink()
        shutil.rmtree(capture_pinned)

    finish_parent = REPO / ".data" / "diagnostic-ancestor-finish"
    finish_root = finish_parent / "capture"
    finish_pinned = finish_parent.with_name("diagnostic-ancestor-finish-pinned")
    outside_finish = tmp_path / "outside-finish"
    outside_finish.mkdir(mode=0o700)
    if finish_parent.exists():
        shutil.rmtree(finish_parent)
    if finish_pinned.exists():
        shutil.rmtree(finish_pinned)
    recorder = diagnostic(finish_root)
    metadata_path = recorder.capture(b"{}", http_status=200, api_key="synthetic-secret")
    finish_parent.rename(finish_pinned)
    (outside_finish / "capture").mkdir(mode=0o700)
    finish_parent.symlink_to(outside_finish, target_is_directory=True)
    try:
        with pytest.raises(
            DiagnosticCaptureError, match="diagnostic_storage_unavailable"
        ):
            recorder.finish(metadata_path, parser_stage="parsed", safe_code="parsed")
        assert list((outside_finish / "capture").iterdir()) == []
        pinned_metadata = finish_pinned / "capture" / metadata_path.name
        metadata = json.loads(pinned_metadata.read_text(encoding="utf-8"))
        assert metadata["parser_stage"] == "pending"
    finally:
        recorder.close()
        finish_parent.unlink()
        shutil.rmtree(finish_pinned)


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
    ("key", "encoded"),
    [
        ("synthetic/secret", '"synthetic/secret"'),
        ("synthetic/secret", r'"synthetic\/secret"'),
        ("synthetic/secret", r'"synthetic\u002fsecret"'),
        (
            "synthetic/secret",
            r'"\u0073\u0079\u006e\u0074\u0068\u0065\u0074\u0069\u0063\u002f\u0073\u0065\u0063\u0072\u0065\u0074"',
        ),
        ("synthetic/secret", r'"synthetic\u002fse\u0063ret"'),
        ("synthetic\\secret", json.dumps("synthetic\\secret")),
        ('synthetic"secret', json.dumps('synthetic"secret')),
    ],
)
def test_capture_rejects_json_escape_encodings_of_accepted_keys(
    key: str, encoded: str
) -> None:
    root = REPO / ".data" / "test-eval-diagnostic-slash-key"
    if root.exists():
        shutil.rmtree(root)
    recorder = diagnostic(root)
    assert _key_shape(key)
    with pytest.raises(DiagnosticCaptureError, match="diagnostic_secret_in_body"):
        recorder.capture(
            ('{"token":' + encoded + "}").encode(), http_status=200, api_key=key
        )
    assert list(root.iterdir()) == []
    recorder.close()
    root.rmdir()


def test_capture_rejects_key_wrapped_inside_json_string() -> None:
    root = REPO / ".data" / "test-eval-diagnostic-wrapped-key"
    if root.exists():
        shutil.rmtree(root)
    recorder = diagnostic(root)
    body = json.dumps({"wrapped": json.dumps({"token": "synthetic/secret"})}).encode()
    with pytest.raises(DiagnosticCaptureError, match="diagnostic_secret_in_body"):
        recorder.capture(body, http_status=200, api_key="synthetic/secret")
    assert list(root.iterdir()) == []
    recorder.close()
    root.rmdir()


@pytest.mark.parametrize(
    "nested",
    [
        "prefix junk " + json.dumps({"token": "synthetic/secret"}).replace("/", r"\/"),
        json.dumps({"token": "synthetic/secret"}).replace("/", r"\/") + ' trailing "',
        json.dumps({"token": "synthetic/secret"}).replace("/", r"\/")
        + " trailing "
        + chr(92),
        json.dumps({"note": "safe"})
        + " middle junk "
        + json.dumps({"token": "synthetic/secret"}).replace("/", r"\/"),
        '{"note":"safe"} trailing "synthetic\\/secret',
        '{"note":"safe"} trailing "bad\\q synthetic\\/secret"',
    ],
)
def test_capture_keeps_scanning_completed_tokens_across_malformed_fragments(
    nested: str,
) -> None:
    root = REPO / ".data" / "test-eval-diagnostic-malformed-fragments"
    if root.exists():
        shutil.rmtree(root)
    recorder = diagnostic(root)
    body = json.dumps({"wrapped": nested}).encode("utf-8")
    with pytest.raises(DiagnosticCaptureError, match="diagnostic_secret_in_body"):
        recorder.capture(body, http_status=200, api_key="synthetic/secret")
    assert list(root.iterdir()) == []
    recorder.close()
    root.rmdir()


@pytest.mark.parametrize(
    ("depth", "expected"),
    [
        (7, "diagnostic_secret_in_body"),
        (8, "diagnostic_secret_in_body"),
        (9, "diagnostic_body_uninspectable"),
        (10, "diagnostic_body_uninspectable"),
        (11, "diagnostic_body_uninspectable"),
    ],
)
def test_partial_escape_depth_limit_never_treats_unknown_as_safe(
    depth: int, expected: str
) -> None:
    root = REPO / ".data" / f"test-eval-diagnostic-partial-depth-{depth}"
    if root.exists():
        shutil.rmtree(root)
    fragment = r"synthetic\/secret"
    for _ in range(depth):
        fragment = json.dumps(fragment)[1:-1]
    body = json.dumps({"wrapped": '"' + fragment}).encode("utf-8")
    recorder = diagnostic(root)
    with pytest.raises(DiagnosticCaptureError, match=expected):
        recorder.capture(body, http_status=200, api_key="synthetic/secret")
    assert list(root.iterdir()) == []
    recorder.close()
    root.rmdir()


def test_unstable_partial_escape_without_key_fails_closed() -> None:
    root = REPO / ".data" / "test-eval-diagnostic-partial-depth-safe-looking"
    if root.exists():
        shutil.rmtree(root)
    fragment = r"ordinary\/suffix"
    for _ in range(9):
        fragment = json.dumps(fragment)[1:-1]
    body = json.dumps({"wrapped": '"' + fragment}).encode("utf-8")
    recorder = diagnostic(root)
    with pytest.raises(DiagnosticCaptureError, match="diagnostic_body_uninspectable"):
        recorder.capture(body, http_status=200, api_key="synthetic/secret")
    assert list(root.iterdir()) == []
    recorder.close()
    root.rmdir()


@pytest.mark.parametrize(
    ("leading", "trailing"),
    [
        (leading, trailing)
        for leading in ("", " ", "\t", "\n", "\r", " \r\n\t")
        for trailing in ("", " ", "\t", "\n", "\r", " \r\n\t")
    ],
)
def test_nested_key_scan_accepts_only_complete_json_whitespace(
    leading: str, trailing: str
) -> None:
    key = "synthetic/secret"
    encoded_inner = json.dumps({"token": key}).replace("/", r"\/")
    nested = leading + encoded_inner + trailing
    assert key in eval_diagnostic_module._complete_json_string_tokens(nested)
    body = json.dumps({"wrapped": nested}).encode("utf-8")
    assert eval_diagnostic_module._contains_secret_shape(body, key)


@pytest.mark.parametrize(
    ("leading", "trailing"),
    [("\u00a0", ""), ("", "\u00a0"), ("\u2003", ""), ("", "\u2003")],
)
def test_non_json_whitespace_does_not_hide_nested_key(
    leading: str, trailing: str
) -> None:
    root = REPO / ".data" / "test-eval-diagnostic-non-json-whitespace"
    if root.exists():
        shutil.rmtree(root)
    key = "synthetic/secret"
    nested = leading + json.dumps({"token": key}).replace("/", r"\/") + trailing
    assert eval_diagnostic_module._complete_json_string_tokens(nested) is None
    recorder = diagnostic(root)
    with pytest.raises(DiagnosticCaptureError, match="diagnostic_secret_in_body"):
        recorder.capture(
            json.dumps({"wrapped": nested}).encode("utf-8"),
            http_status=200,
            api_key=key,
        )
    assert list(root.iterdir()) == []
    recorder.close()
    root.rmdir()


def test_non_json_trailing_text_does_not_hide_nested_key() -> None:
    key = "synthetic/secret"
    nested = json.dumps({"token": key}).replace("/", r"\/") + " trailing text"
    assert eval_diagnostic_module._complete_json_string_tokens(nested) is None
    assert eval_diagnostic_module._contains_secret_shape(
        json.dumps({"wrapped": nested}).encode(), key
    )


def test_deep_nested_document_honors_trailing_boundary_and_scans_keys() -> None:
    key = "synthetic/secret"
    nested = (
        " \t\r\n"
        + "[" * 1500
        + json.dumps({"token": key}).replace("/", r"\/")
        + "]" * 1500
        + "\r\n\t "
    )
    assert key in eval_diagnostic_module._complete_json_string_tokens(nested)
    assert eval_diagnostic_module._contains_secret_shape(
        json.dumps({"wrapped": nested}).encode(), key
    )
    assert eval_diagnostic_module._complete_json_string_tokens(nested + " junk") is None


@pytest.mark.parametrize(
    "number",
    ["0", "-12", "9" * 5000, "-" + "8" * 5000, "1.25", "-3.5e+12", "1e10000"],
)
def test_nested_json_numeric_metadata_is_scanned_without_integer_conversion(
    number: str,
) -> None:
    key = "synthetic/secret"
    nested = " \r\n" + '{"number":' + number + ',"token":"synthetic\\/secret"}\t '
    assert key in eval_diagnostic_module._complete_json_string_tokens(nested)
    assert eval_diagnostic_module._contains_secret_shape(
        json.dumps({"wrapped": nested}).encode(), key
    )


def test_nested_big_integer_metadata_without_secrets_remains_inspectable() -> None:
    nested = '{"metadata_number":' + "7" * 5000 + "}"
    assert eval_diagnostic_module._complete_json_string_tokens(nested) == [
        "metadata_number"
    ]
    assert not eval_diagnostic_module._contains_secret_shape(
        json.dumps({"wrapped": nested}).encode(), "synthetic-absent-key"
    )


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (b"\xffsynthetic-secret", "diagnostic_secret_in_body"),
        (b"\xffsafe-body", "diagnostic_body_encoding_invalid"),
    ],
)
def test_non_utf8_capture_fails_closed_without_partial_records(
    body: bytes, expected: str
) -> None:
    root = REPO / ".data" / "test-eval-diagnostic-nonutf8"
    if root.exists():
        shutil.rmtree(root)
    recorder = diagnostic(root)
    with pytest.raises(DiagnosticCaptureError, match=expected):
        recorder.capture(body, http_status=200, api_key="synthetic-secret")
    assert list(root.iterdir()) == []
    recorder.close()
    root.rmdir()


def test_secret_scan_escape_depth_limit_fails_closed_without_records() -> None:
    root = REPO / ".data" / "test-eval-diagnostic-escape-depth"
    if root.exists():
        shutil.rmtree(root)
    recorder = diagnostic(root)
    nested = r"\\u005c" + "u005c" * 10 + r"\u0061"
    body = json.dumps({"note": nested}).encode("utf-8")
    with pytest.raises(DiagnosticCaptureError, match="diagnostic_body_uninspectable"):
        recorder.capture(body, http_status=200, api_key="synthetic-secret")
    assert list(root.iterdir()) == []
    recorder.close()
    root.rmdir()


@pytest.mark.parametrize(
    "metadata_note",
    [
        pytest.param("[" * 12000 + "0" + "]" * 12000, id="deep-bracket-text"),
        pytest.param('The display is 6" wide.', id="inch-quote-prose"),
        pytest.param(
            'The note says "measure 6\\"" and keeps C:\\manuals\\front as text.',
            id="quotes-and-backslashes-prose",
        ),
    ],
)
def test_secret_scan_is_bounded_and_accepts_metadata_text(
    monkeypatch: pytest.MonkeyPatch, metadata_note: str
) -> None:
    root = REPO / ".data" / "test-eval-diagnostic-deep-envelope"
    if root.exists():
        shutil.rmtree(root)
    recorder = diagnostic(root)
    request = prepare_answer("synthetic deep envelope", product_id="hp-a", hits=[])
    envelope = {
        "id": "resp_fixture",
        "model": "gpt-6-luna",
        "status": "completed",
        "service_tier": "default",
        "metadata": {"invented_note": metadata_note},
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": json.dumps(
                            {
                                "status": "handoff",
                                "answer": None,
                                "reason": "no_evidence",
                                "claims": [],
                            }
                        ),
                    }
                ],
            }
        ],
        "usage": {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
    }
    body = json.dumps(envelope).encode("utf-8")
    assert len(body) <= MAX_DIAGNOSTIC_BODY
    calls = 0

    async def exchange(*_args: object, **_kwargs: object):
        nonlocal calls
        calls += 1
        return body, 200, None

    monkeypatch.setattr(luna, "_factory_transport", lambda _transport: True)
    monkeypatch.setattr(luna, "_exchange", exchange)

    async def run():
        return await luna._post_live_once(
            b"{}",
            request,
            api_key="synthetic-key",
            transport=object(),  # type: ignore[arg-type]
            request_id="synthetic-deep-envelope",
            deadline=1,
            diagnostic=recorder,
        )

    outcome, _usage, stopped = asyncio.run(run())
    assert calls == 1
    assert stopped is None
    assert outcome is not None and outcome.status == "handoff"
    response_path = next(root.glob("*.response.json"))
    assert response_path.read_bytes() == body
    recorder.close()
    shutil.rmtree(root)


def test_deep_response_parser_recursion_maps_to_safe_luna_error_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = REPO / ".data" / "test-eval-diagnostic-deep-array"
    if root.exists():
        shutil.rmtree(root)
    recorder = diagnostic(root)
    body = b"[" * 1500 + b"0" + b"]" * 1500
    calls = 0

    async def exchange(*_args: object, **_kwargs: object):
        nonlocal calls
        calls += 1
        return body, 200, None

    monkeypatch.setattr(luna, "_factory_transport", lambda _transport: True)
    monkeypatch.setattr(luna, "_exchange", exchange)
    request = prepare_answer("synthetic deep array", product_id="hp-a", hits=[])

    async def run():
        return await luna._post_live_once(
            b"{}",
            request,
            api_key="synthetic-key",
            transport=object(),  # type: ignore[arg-type]
            request_id="synthetic-deep-array",
            deadline=1,
            diagnostic=recorder,
        )

    with pytest.raises(LunaError) as caught:
        asyncio.run(run())
    assert str(caught.value) in {"invalid_response", "invalid_model_schema"}
    assert calls == 1
    metadata_path = next(root.glob("*.meta.json"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["safe_code"] in {"invalid_response", "invalid_model_schema"}
    assert list(root.glob("*.response.json"))[0].read_bytes() == body
    recorder.close()
    shutil.rmtree(root)


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
        response = json.dumps(
            {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps(
                                    {
                                        "status": "handoff",
                                        "answer": None,
                                        "reason": "no_evidence",
                                        "claims": [],
                                    }
                                ),
                            }
                        ],
                    }
                ]
            }
        ).encode()
        return response, 200, None

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
