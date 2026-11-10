"""Synthetic old-ledger continuity. These tests never open a real budget file."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest

from customer_support_rag.budget import (
    OUTPUT_RESERVATION,
    BudgetError,
    LunaPolicy,
    fingerprint,
    init_synthetic_ledger,
    load_policy,
    load_runtime,
    open_ledger,
)

KEY = "synthetic-key-not-live"
REVISION = "gpt-6-luna-standard-2026-11-08-reviewed"


def document(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "version": 1,
        "model": "gpt-6-luna",
        "budget_id": "synthetic-budget",
        "cap_units": 123456,
        "input_rate_units": 1000,
        "output_rate_units": 2000,
        "extra_units": 3,
        "pricing_revision": REVISION,
        "pricing_expires_at": "2099-01-02T03:04:05Z",
        "enabled_scopes": ["developer_smoke"],
        "enabled": True,
    }
    payload.update(overrides)
    return payload


def write_policy(directory: Path, **overrides: object) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "policy.json"
    path.write_text(json.dumps(document(**overrides)), encoding="utf-8")
    path.chmod(0o600)
    return path


def policy_at(directory: Path, **overrides: object) -> LunaPolicy:
    return load_policy(write_policy(directory, **overrides))


def request_rows(path: Path) -> list[sqlite3.Row]:
    connection = sqlite3.connect(path.absolute().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        return list(connection.execute("SELECT * FROM requests ORDER BY id"))
    finally:
        connection.close()


def insert_request(path: Path, **fields: object) -> None:
    columns = ", ".join(fields)
    marks = ", ".join("?" for _ in fields)
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            f"INSERT INTO requests ({columns}) VALUES ({marks})", tuple(fields.values())
        )
        connection.commit()
    finally:
        connection.close()


def insert_matching_debit(path: Path, policy: LunaPolicy, request_id: str) -> None:
    row = next(item for item in request_rows(path) if item["id"] == request_id)
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "INSERT INTO conservative_debits VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row["id"],
                row["budget"],
                row["body_hash"],
                row["pricing_hash"],
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


def test_binding_matches_the_historical_ledger_algorithm(tmp_path: Path) -> None:
    policy = policy_at(tmp_path)
    assert policy.binding() == (
        "757df411ca54c955eff8e503213ab28e7ed5d2ac29aeb253b9dbba9ca60d2f7e"
    )
    assert policy.pricing_hash() == (
        "b2ad0a950bc6401edcfafef21232b59d2bfde531509d06eabb99dce660ae45b0"
    )
    assert policy.cap_fingerprint() == fingerprint({"ceiling": 123456})
    assert policy.reservation_cost(100, OUTPUT_RESERVATION) == 10
    runtime = load_runtime(tmp_path / "policy.json", KEY)
    assert runtime.pricing_bound_verified is False
    assert runtime.api_key == KEY


def test_open_refuses_to_create_a_ledger(tmp_path: Path) -> None:
    missing = tmp_path / "missing.sqlite"
    with pytest.raises(BudgetError) as caught:
        open_ledger(missing)
    assert caught.value.code == "ledger_unavailable"
    assert not missing.exists()
    policy = policy_at(tmp_path)
    path = tmp_path / "ledger.sqlite"
    init_synthetic_ledger(path, policy)
    with pytest.raises(BudgetError) as again:
        init_synthetic_ledger(path, policy)
    assert again.value.code == "ledger_unavailable"
    open_ledger(path)


def test_history_and_original_replay_rules(tmp_path: Path) -> None:
    policy = policy_at(tmp_path, cap_units=19)
    path = tmp_path / "ledger.sqlite"
    init_synthetic_ledger(path, policy)
    insert_request(
        path,
        id="historical-settled",
        budget=policy.budget_id,
        body_hash="1" * 64,
        owner="historical",
        scope="developer_smoke",
        state="settled",
        res_input=100,
        res_output=OUTPUT_RESERVATION,
        res_cost=10,
        used_input=10,
        used_output=4,
        used_cost=4,
        pricing_hash=policy.pricing_hash(),
        requested_model="gpt-6-luna",
        created=1.0,
    )
    insert_request(
        path,
        id="original-unknown",
        budget=policy.budget_id,
        body_hash="2" * 64,
        owner="historical",
        scope="developer_smoke",
        state="unknown",
        res_input=80,
        res_output=OUTPUT_RESERVATION,
        res_cost=5,
        pricing_hash=policy.pricing_hash(),
        requested_model="gpt-6-luna",
        created=2.0,
    )
    insert_matching_debit(path, policy, "original-unknown")
    ledger = open_ledger(path)
    with pytest.raises(BudgetError) as exhausted:
        ledger.reserve(
            policy,
            "new-request",
            "3" * 64,
            100,
            OUTPUT_RESERVATION,
            "developer_smoke",
            KEY,
        )
    assert exhausted.value.code == "budget_exhausted"

    room = policy_at(tmp_path / "room", cap_units=21)
    room_path = tmp_path / "room.sqlite"
    init_synthetic_ledger(room_path, room)
    for row in request_rows(path):
        insert_request(room_path, **dict(row))
    insert_matching_debit(room_path, room, "original-unknown")
    admitted = open_ledger(room_path)
    admitted.reserve(
        room, "new-request", "3" * 64, 100, OUTPUT_RESERVATION, "developer_smoke", KEY
    )
    with pytest.raises(BudgetError) as duplicate:
        admitted.reserve(
            room,
            "original-unknown",
            "4" * 64,
            100,
            OUTPUT_RESERVATION,
            "developer_smoke",
            KEY,
        )
    assert duplicate.value.code == "duplicate_or_conflicting_request"
    with pytest.raises(BudgetError) as replay:
        admitted.reserve(
            room, "other-id", "2" * 64, 100, OUTPUT_RESERVATION, "developer_smoke", KEY
        )
    assert replay.value.code == "permanent_original_body_replay"


def test_wrong_match_and_binding_changes_block(tmp_path: Path) -> None:
    policy = policy_at(tmp_path, cap_units=10**9)
    path = tmp_path / "ledger.sqlite"
    init_synthetic_ledger(path, policy)
    insert_request(
        path,
        id="original-unknown",
        budget=policy.budget_id,
        body_hash="2" * 64,
        owner="historical",
        scope="developer_smoke",
        state="unknown",
        res_input=80,
        res_output=OUTPUT_RESERVATION,
        res_cost=5,
        pricing_hash=policy.pricing_hash(),
        requested_model="gpt-6-luna",
        created=2.0,
    )
    connection = sqlite3.connect(path)
    connection.execute(
        "INSERT INTO conservative_debits VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "original-unknown",
            policy.budget_id,
            "2" * 64,
            policy.pricing_hash(),
            4,
            80,
            OUTPUT_RESERVATION,
            "a" * 64,
            "b" * 64,
            policy.cap_fingerprint(),
            1.0,
            "{}",
        ),
    )
    connection.commit()
    connection.close()
    ledger = open_ledger(path)
    with pytest.raises(BudgetError) as blocked:
        ledger.reserve(
            policy,
            "new-request",
            "3" * 64,
            100,
            OUTPUT_RESERVATION,
            "developer_smoke",
            KEY,
        )
    assert blocked.value.code == "unreconciled_or_foreign_reservation"

    changed_cap = policy_at(tmp_path / "cap", cap_units=policy.cap_units + 1)
    fresh = tmp_path / "fresh.sqlite"
    init_synthetic_ledger(fresh, policy)
    with pytest.raises(BudgetError) as cap_change:
        open_ledger(fresh).reserve(
            changed_cap,
            "new-request",
            "3" * 64,
            100,
            OUTPUT_RESERVATION,
            "developer_smoke",
            KEY,
        )
    assert cap_change.value.code == "cap_change_requires_authorization"

    changed_budget = policy_at(
        tmp_path / "budget",
        budget_id="other-budget",
        cap_units=policy.cap_units,
    )
    with pytest.raises(BudgetError) as identity:
        open_ledger(fresh).reserve(
            changed_budget,
            "new-request",
            "3" * 64,
            100,
            OUTPUT_RESERVATION,
            "developer_smoke",
            KEY,
        )
    assert identity.value.code == "budget_identity_change_requires_reconciliation"

    changed_rate = policy_at(
        tmp_path / "rate",
        input_rate_units=1001,
        cap_units=policy.cap_units,
    )
    with pytest.raises(BudgetError) as binding:
        open_ledger(fresh).reserve(
            changed_rate,
            "new-request",
            "3" * 64,
            100,
            OUTPUT_RESERVATION,
            "developer_smoke",
            KEY,
        )
    assert binding.value.code == "budget_configuration_changed"


def test_two_connections_race_for_the_last_reservation(tmp_path: Path) -> None:
    policy = policy_at(tmp_path, cap_units=11)
    path = tmp_path / "ledger.sqlite"
    init_synthetic_ledger(path, policy)
    ledger = open_ledger(path)
    barrier = threading.Barrier(2)
    results: list[str] = []

    def attempt(request_id: str, body: str) -> None:
        barrier.wait()
        try:
            ledger.reserve(
                policy,
                request_id,
                body,
                100,
                OUTPUT_RESERVATION,
                "developer_smoke",
                KEY,
            )
        except BudgetError as exc:
            results.append(exc.code)
        else:
            results.append("reserved")

    threads = [
        threading.Thread(target=attempt, args=("one", "a" * 64)),
        threading.Thread(target=attempt, args=("two", "b" * 64)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(results) == ["budget_exhausted", "reserved"]
    assert [row["state"] for row in request_rows(path)] == ["reserved"]

    with pytest.raises(BudgetError) as reopened:
        open_ledger(path).reserve(
            policy, "three", "c" * 64, 100, OUTPUT_RESERVATION, "developer_smoke", KEY
        )
    assert reopened.value.code == "unreconciled_or_foreign_reservation"
    assert [row["state"] for row in request_rows(path)] == ["reserved"]


def test_same_run_can_settle_unknown_and_a_new_process_cannot(tmp_path: Path) -> None:
    policy = policy_at(tmp_path, cap_units=10**9)
    path = tmp_path / "ledger.sqlite"
    init_synthetic_ledger(path, policy)
    ledger = open_ledger(path)
    ledger.reserve(
        policy, "late", "d" * 64, 100, OUTPUT_RESERVATION, "developer_smoke", KEY
    )
    ledger.mark_sent("late")
    ledger.retain_unknown("late")
    with pytest.raises(BudgetError) as other:
        open_ledger(path).settle(
            policy,
            "late",
            input_tokens=5,
            output_tokens=5,
            cached_tokens=None,
            reasoning_tokens=None,
            returned_model="gpt-6-luna",
            provider_request_id=None,
            provider_response_id=None,
            latency_seconds=0.0,
        )
    assert other.value.code == "reservation_not_settleable"
    assert request_rows(path)[0]["state"] == "unknown"
    ledger.settle(
        policy,
        "late",
        input_tokens=5,
        output_tokens=5,
        cached_tokens=None,
        reasoning_tokens=0,
        returned_model="gpt-6-luna",
        provider_request_id="req_fixture",
        provider_response_id="resp_fixture",
        latency_seconds=0.1,
    )
    settled = request_rows(path)[0]
    assert settled["state"] == "settled"
    assert settled["used_input"] == 5
    assert settled["cached_tokens"] is None
    assert settled["reasoning_tokens"] == 0


def test_unsent_release_drops_the_reservation(tmp_path: Path) -> None:
    policy = policy_at(tmp_path, cap_units=11)
    path = tmp_path / "ledger.sqlite"
    init_synthetic_ledger(path, policy)
    ledger = open_ledger(path)
    ledger.reserve(
        policy, "once", "e" * 64, 100, OUTPUT_RESERVATION, "developer_smoke", KEY
    )
    ledger.release("once")
    assert request_rows(path)[0]["state"] == "released"
    ledger.reserve(
        policy, "twice", "f" * 64, 100, OUTPUT_RESERVATION, "developer_smoke", KEY
    )
    assert {row["id"]: row["state"] for row in request_rows(path)}[
        "twice"
    ] == "reserved"


def test_expiry_blocks_a_new_reservation_but_not_settle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = policy_at(tmp_path, cap_units=10**9)
    path = tmp_path / "ledger.sqlite"
    init_synthetic_ledger(path, policy)
    ledger = open_ledger(path)
    ledger.reserve(
        policy,
        "sent-before-expiry",
        "a" * 64,
        100,
        OUTPUT_RESERVATION,
        "developer_smoke",
        KEY,
    )
    ledger.mark_sent("sent-before-expiry")
    monkeypatch.setattr(
        "customer_support_rag.budget._now",
        lambda: datetime(2100, 1, 1, tzinfo=UTC),
    )
    ledger.settle(
        policy,
        "sent-before-expiry",
        input_tokens=4,
        output_tokens=4,
        cached_tokens=None,
        reasoning_tokens=None,
        returned_model="gpt-6-luna",
        provider_request_id=None,
        provider_response_id=None,
        latency_seconds=0.0,
    )
    assert request_rows(path)[0]["state"] == "settled"
    with pytest.raises(BudgetError) as expired:
        ledger.reserve(
            policy,
            "after-expiry",
            "b" * 64,
            100,
            OUTPUT_RESERVATION,
            "developer_smoke",
            KEY,
        )
    assert expired.value.code == "policy_invalid"
    assert [row["id"] for row in request_rows(path)] == ["sent-before-expiry"]
