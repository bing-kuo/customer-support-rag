"""Read-only readiness and a socket-blocked one-shot. No live provider call."""

from __future__ import annotations

import asyncio
import hashlib
import json
import socket
import sqlite3
from pathlib import Path

import httpx
import pytest

from customer_support_rag.budget import (
    OUTPUT_RESERVATION,
    LunaPolicy,
    LunaRuntime,
    init_synthetic_ledger,
    load_policy,
    rates_cover_documented_ceiling,
)
from customer_support_rag.luna import (
    WIRE_FUDGE,
    LunaError,
    build_luna_body,
    generate_draft,
    network_transport,
    permit_live_transport,
    responses_client,
    wire_bytes,
    wire_composition,
)
from customer_support_rag.luna_live import (
    QUESTION,
    REQUEST_ID,
    SCOPE,
    hp_a_check,
    hp_a_once,
    prepare_hp_a_request,
)

KNOWLEDGE = Path(__file__).parents[1] / "knowledge"
KEY = "synthetic-key-not-live"
BODY_HASH = "e861e88a0dd3ed17a554d9de8091af8c3b680b5abd9db3a03c58e1aebbe502fe"
FLOOR_INPUT = 275_000_000_000
FLOOR_OUTPUT = 550_000_000_000


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


def test_hp_a_wire_is_inside_the_old_ceiling() -> None:
    request = prepare_hp_a_request(KNOWLEDGE)
    parts = wire_composition(request)
    encoded = wire_bytes(build_luna_body(request))
    assert request.question == QUESTION
    assert request.evidence[0].text
    assert (
        request.sources[0].hit.chunk.char_start,
        request.sources[0].hit.chunk.char_end,
    ) == (1, 71)
    assert parts.policy_bytes == 607
    assert parts.data_bytes == 415
    assert parts.schema_bytes == 398
    assert parts.framing_bytes == 409
    assert parts.total_bytes == 1829
    assert parts.reserved_input_tokens == parts.total_bytes + WIRE_FUDGE == 2853
    assert parts.ceiling_headroom_tokens == 731
    assert parts.reserved_input_tokens + OUTPUT_RESERVATION <= 4096
    assert hashlib.sha256(encoded).hexdigest() == BODY_HASH


def test_floor_rates_cover_and_one_unit_less_does_not(tmp_path: Path) -> None:
    covered = _policy(
        tmp_path / "floor", input_rate_units=FLOOR_INPUT, output_rate_units=FLOOR_OUTPUT
    )
    assert rates_cover_documented_ceiling(covered) is True
    short_input = _policy(
        tmp_path / "short-input",
        input_rate_units=FLOOR_INPUT - 1,
        output_rate_units=FLOOR_OUTPUT,
    )
    short_output = _policy(
        tmp_path / "short-output",
        input_rate_units=FLOOR_INPUT,
        output_rate_units=FLOOR_OUTPUT - 1,
    )
    extra_only = _policy(
        tmp_path / "extra",
        input_rate_units=1000,
        output_rate_units=2000,
        extra_units=2**59,
    )
    assert rates_cover_documented_ceiling(short_input) is False
    assert rates_cover_documented_ceiling(short_output) is False
    assert rates_cover_documented_ceiling(extra_only) is False
    high = _policy(
        tmp_path / "high",
        input_rate_units=10**15,
        output_rate_units=10**15,
        cap_units=2**59,
    )
    assert rates_cover_documented_ceiling(high) is True


def test_missing_key_and_bad_schema_do_not_open_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_send(monkeypatch)
    missing = tmp_path / "missing"
    report = hp_a_check(missing, knowledge=KNOWLEDGE)
    assert report.key_present is False
    assert report.network_allowed is False
    assert "key_missing" in report.failures
    root = _ready(tmp_path / "schema")
    connection = sqlite3.connect(root / "runtime" / "cloud-budget.sqlite3")
    connection.execute("CREATE TABLE extra(id INTEGER)")
    connection.commit()
    connection.close()
    broken = hp_a_check(root, knowledge=KNOWLEDGE)
    assert broken.network_allowed is False
    assert broken.history_continuous is False
    assert "ledger_schema_incompatible" in broken.failures
    with pytest.raises(LunaError) as blocked:
        asyncio.run(hp_a_once(root, knowledge=KNOWLEDGE))
    assert blocked.value.code == "paid_blocked"


def test_check_keeps_history_and_does_not_open_the_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _ready(tmp_path)
    before = _snapshot(root)
    monkeypatch.setattr(
        "customer_support_rag.luna_live.open_ledger",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("rw ledger")),
    )
    monkeypatch.setattr(
        "customer_support_rag.luna_live.network_transport",
        lambda: (_ for _ in ()).throw(AssertionError("network")),
    )
    report = hp_a_check(root, knowledge=KNOWLEDGE)
    assert report.network_allowed is True
    assert report.pricing_bound_verified is True
    assert report.key_present is True
    assert report.history_continuous is True
    assert report.original_debit_counted is True
    assert report.replay_blocked is True
    assert report.cap_unchanged is True
    assert report.no_pending is True
    assert report.one_request_fits is True
    assert (
        "structural_allowance_1024_is_not_a_published_tokenizer_bound" in report.notes
    )
    assert (
        "original_unknown_pricing_hash_differs_from_current_policy" not in report.notes
    )
    assert _snapshot(root) == before
    states = _states(root)
    assert [row["state"] for row in states] == ["settled", "unknown"]
    assert states[1]["used_cost"] is None


def test_original_pricing_hash_difference_is_a_note_only(tmp_path: Path) -> None:
    root = _ready(tmp_path, unknown_hash="ab" * 32)
    report = hp_a_check(root, knowledge=KNOWLEDGE)
    assert report.history_continuous is True
    assert report.original_debit_counted is True
    assert "original_unknown_pricing_hash_differs_from_current_policy" in report.notes
    assert _states(root)[1]["pricing_hash"] == "ab" * 32


def test_wrong_binding_pending_and_existing_id_block_send(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_send(monkeypatch)
    bound = _ready(tmp_path / "bound")
    _execute(
        bound,
        "UPDATE budgets SET binding=?",
        ("ab" * 32,),
    )
    binding = hp_a_check(bound, knowledge=KNOWLEDGE)
    assert binding.history_continuous is False
    assert "budget_binding_mismatch" in binding.failures
    assert _states(bound)[1]["state"] == "unknown"

    pending = _ready(tmp_path / "pending")
    _insert_request(
        pending,
        id="still-sent",
        budget="synthetic-budget",
        body_hash="4" * 64,
        owner="historical",
        scope=SCOPE,
        state="sent",
        res_input=10,
        res_output=OUTPUT_RESERVATION,
        res_cost=1,
        pricing_hash=_policy_at(pending).pricing_hash(),
        requested_model="gpt-6-luna",
        created=3.0,
    )
    waiting = hp_a_check(pending, knowledge=KNOWLEDGE)
    assert waiting.no_pending is False
    assert "pending_reservation" in waiting.failures

    taken = _ready(tmp_path / "taken")
    _insert_request(
        taken,
        id=REQUEST_ID,
        budget="synthetic-budget",
        body_hash="5" * 64,
        owner="historical",
        scope=SCOPE,
        state="reserved",
        res_input=10,
        res_output=OUTPUT_RESERVATION,
        res_cost=1,
        pricing_hash=_policy_at(taken).pricing_hash(),
        requested_model="gpt-6-luna",
        created=4.0,
    )
    exists = hp_a_check(taken, knowledge=KNOWLEDGE)
    assert "request_id_exists" in exists.failures
    assert exists.network_allowed is False
    with pytest.raises(LunaError) as blocked:
        asyncio.run(hp_a_once(taken, knowledge=KNOWLEDGE))
    assert blocked.value.code == "paid_blocked"
    assert any(
        row["id"] == REQUEST_ID and row["state"] == "reserved" for row in _states(taken)
    )


def test_replayed_body_stays_blocked_without_a_new_id(tmp_path: Path) -> None:
    root = _ready(tmp_path)
    policy = _policy_at(root)
    _insert_request(
        root,
        id="same-body",
        budget=policy.budget_id,
        body_hash=BODY_HASH,
        owner="historical",
        scope=SCOPE,
        state="settled",
        res_input=10,
        res_output=OUTPUT_RESERVATION,
        res_cost=1,
        used_input=1,
        used_output=1,
        used_cost=1,
        pricing_hash=policy.pricing_hash(),
        requested_model="gpt-6-luna",
        created=5.0,
    )
    _insert_debit(root, "same-body", policy)
    report = hp_a_check(root, knowledge=KNOWLEDGE)
    assert report.replay_blocked is True
    assert report.one_request_fits is False
    assert "new_body_matches_debited_hash" in report.failures
    assert report.network_allowed is False


def test_low_rates_and_loose_permissions(tmp_path: Path) -> None:
    low = _ready(
        tmp_path / "low",
        input_rate_units=1000,
        output_rate_units=2000,
        cap_units=10**9,
    )
    report = hp_a_check(low, knowledge=KNOWLEDGE)
    assert report.pricing_bound_verified is False
    assert "documented_rate_ceiling_uncovered" in report.failures
    assert report.network_allowed is False

    noted = _ready(tmp_path / "noted")
    ledger = noted / "runtime" / "cloud-budget.sqlite3"
    ledger.chmod(0o644)
    noted.joinpath("runtime").chmod(0o700)
    loose_note = hp_a_check(noted, knowledge=KNOWLEDGE)
    assert loose_note.network_allowed is True
    assert "ledger_mode_loose_inside_owner_only_directory" in loose_note.notes

    exposed = _ready(tmp_path / "exposed")
    exposed_ledger = exposed / "runtime" / "cloud-budget.sqlite3"
    exposed_ledger.chmod(0o644)
    exposed.joinpath("runtime").chmod(0o755)
    blocked = hp_a_check(exposed, knowledge=KNOWLEDGE)
    assert blocked.network_allowed is False
    assert "ledger_not_owner_only" in blocked.failures


def test_runtime_flag_and_raw_transport_cannot_pass(tmp_path: Path) -> None:
    root = _ready(tmp_path)
    policy = _policy_at(root)
    runtime = LunaRuntime(policy, KEY, pricing_bound_verified=True)
    from customer_support_rag.budget import open_ledger

    ledger = open_ledger(root / "runtime" / "cloud-budget.sqlite3")
    request = prepare_hp_a_request(KNOWLEDGE)
    raw = httpx.AsyncHTTPTransport(verify=True, trust_env=False, retries=0)
    try:
        with pytest.raises(LunaError) as blocked:
            asyncio.run(
                generate_draft(
                    request,
                    request_id="flag",
                    runtime=runtime,
                    ledger=ledger,
                    transport=raw,
                    scope=SCOPE,
                    admission=permit_live_transport(),
                )
            )
        assert blocked.value.code == "paid_blocked"
        factory = network_transport()
        factory._luna_factory = object()
        with pytest.raises(LunaError) as forged:
            responses_client(factory, admission=permit_live_transport())
        assert forged.value.code == "paid_blocked"
        asyncio.run(factory.aclose())
    finally:
        asyncio.run(raw.aclose())
    assert [row["state"] for row in _states(root)] == ["settled", "unknown"]


def test_ready_once_keeps_one_unknown_and_does_not_post(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _ready(tmp_path)
    seen = {"inet": 0, "factory": 0}
    real_socket = socket.socket

    def guarded(family: int = socket.AF_INET, *args: object, **kwargs: object):
        if family == socket.AF_UNIX:
            return real_socket(family, *args, **kwargs)
        seen["inet"] += 1
        raise OSError("blocked")

    def one_address(*_args: object, **_kwargs: object):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))]

    monkeypatch.setattr(socket, "socket", guarded)
    monkeypatch.setattr(socket, "getaddrinfo", one_address)
    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("blocked")),
    )
    real_factory = network_transport

    def factory():
        seen["factory"] += 1
        return real_factory()

    monkeypatch.setattr("customer_support_rag.luna_live.network_transport", factory)
    with pytest.raises(LunaError) as caught:
        asyncio.run(hp_a_once(root, knowledge=KNOWLEDGE))
    assert caught.value.code == "provider_transport_error"
    assert str(caught.value) == "provider_transport_error"
    assert seen["factory"] == 1
    assert seen["inet"] == 1
    rows = _states(root)
    original = next(row for row in rows if row["id"] == "original-unknown")
    live = next(row for row in rows if row["id"] == REQUEST_ID)
    assert original["state"] == "unknown"
    assert original["used_cost"] is None
    assert live["state"] == "unknown"
    assert live["used_input"] is None
    assert _debit_ids(root) == ["original-unknown"]


def _forbid_send(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("send")

    monkeypatch.setattr("customer_support_rag.luna_live.network_transport", boom)
    monkeypatch.setattr("customer_support_rag.luna_live.open_ledger", boom)


def _policy(
    directory: Path, *, name: str = "policy.json", **overrides: object
) -> LunaPolicy:
    payload: dict[str, object] = {
        "version": 1,
        "model": "gpt-6-luna",
        "budget_id": "synthetic-budget",
        "cap_units": 2**59,
        "input_rate_units": FLOOR_INPUT,
        "output_rate_units": FLOOR_OUTPUT,
        "extra_units": 0,
        "pricing_revision": "gpt-6-luna-standard-2026-11-08-reviewed",
        "pricing_expires_at": "2099-01-02T03:04:05Z",
        "enabled_scopes": [SCOPE],
        "enabled": True,
    }
    payload.update(overrides)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o600)
    return load_policy(path)


def _ready(root: Path, *, unknown_hash: str | None = None, **overrides: object) -> Path:
    policy = _policy(root / "runtime", name="cloud-policy.json", **overrides)
    if unknown_hash is None:
        unknown_hash = policy.pricing_hash()
    env = root / ".env"
    env.write_text(f'# local\nexport OPENAI_API_KEY="{KEY}"\n', encoding="utf-8")
    env.chmod(0o600)
    ledger = root / "runtime" / "cloud-budget.sqlite3"
    init_synthetic_ledger(ledger, policy)
    root.joinpath("runtime").chmod(0o700)
    _insert_request(
        root,
        id="historical-settled",
        budget=policy.budget_id,
        body_hash="1" * 64,
        owner="historical",
        scope=SCOPE,
        state="settled",
        res_input=20,
        res_output=OUTPUT_RESERVATION,
        res_cost=2,
        used_input=4,
        used_output=2,
        used_cost=1,
        pricing_hash=policy.pricing_hash(),
        requested_model="gpt-6-luna",
        created=1.0,
    )
    _insert_request(
        root,
        id="original-unknown",
        budget=policy.budget_id,
        body_hash="2" * 64,
        owner="historical",
        scope=SCOPE,
        state="unknown",
        res_input=30,
        res_output=OUTPUT_RESERVATION,
        res_cost=2,
        pricing_hash=unknown_hash,
        requested_model="gpt-6-luna",
        created=2.0,
    )
    _insert_debit(root, "original-unknown", policy, pricing_hash=unknown_hash)
    return root


def _insert_request(root: Path, **fields: object) -> None:
    columns = ", ".join(fields)
    marks = ", ".join("?" for _ in fields)
    connection = sqlite3.connect(root / "runtime" / "cloud-budget.sqlite3")
    try:
        connection.execute(
            f"INSERT INTO requests ({columns}) VALUES ({marks})", tuple(fields.values())
        )
        connection.commit()
    finally:
        connection.close()


def _insert_debit(
    root: Path, request_id: str, policy: LunaPolicy, *, pricing_hash: str | None = None
) -> None:
    row = next(item for item in _states(root) if item["id"] == request_id)
    connection = sqlite3.connect(root / "runtime" / "cloud-budget.sqlite3")
    try:
        connection.execute(
            "INSERT INTO conservative_debits VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row["id"],
                row["budget"],
                row["body_hash"],
                row["pricing_hash"] if pricing_hash is None else pricing_hash,
                row["res_cost"],
                row["res_input"],
                row["res_output"],
                "a" * 64,
                "b" * 64,
                policy.cap_fingerprint(),
                1.0,
                "{}",
            ),
        )
        connection.commit()
    finally:
        connection.close()


def _states(root: Path) -> list[sqlite3.Row]:
    connection = sqlite3.connect(
        (root / "runtime" / "cloud-budget.sqlite3").absolute().as_uri() + "?mode=ro",
        uri=True,
    )
    connection.row_factory = sqlite3.Row
    try:
        return list(connection.execute("SELECT * FROM requests ORDER BY id"))
    finally:
        connection.close()


def _debit_ids(root: Path) -> list[str]:
    connection = sqlite3.connect(
        (root / "runtime" / "cloud-budget.sqlite3").absolute().as_uri() + "?mode=ro",
        uri=True,
    )
    try:
        return [
            row[0]
            for row in connection.execute(
                "SELECT id FROM conservative_debits ORDER BY id"
            )
        ]
    finally:
        connection.close()


def _policy_at(root: Path) -> LunaPolicy:
    return load_policy(root / "runtime" / "cloud-policy.json")


def _snapshot(root: Path) -> tuple[bytes, bytes]:
    return (
        (root / ".env").read_bytes(),
        (root / "runtime" / "cloud-budget.sqlite3").read_bytes(),
    )


def _execute(root: Path, sql: str, params: tuple[object, ...]) -> None:
    connection = sqlite3.connect(root / "runtime" / "cloud-budget.sqlite3")
    try:
        connection.execute(sql, params)
        connection.commit()
    finally:
        connection.close()
