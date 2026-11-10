"""Minimal ledger compatible with the existing cumulative budget file.

Opening a ledger never creates or migrates a database. Synthetic fixtures use
``init_synthetic_ledger``. Importing this module does not read the environment,
a key, or a database. The reservation formula is the historical local unit
formula already stored in that ledger. It is not a verified current price.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

MODEL = "gpt-6-luna"
HISTORICAL_PRICING_REVISION = "gpt-6-luna-standard-2026-11-08-reviewed"
HISTORICAL_BOUND_REVISION = "luna-short-text-all-documented-tiers-v1"
MAX_REQUESTS = 200
MAX_INPUT_TOKENS = 200 * 3584
MAX_OUTPUT_TOKENS = 200 * 512
MAX_TOTAL_TOKENS = 200 * 4096
OUTPUT_RESERVATION = 512
TOTAL_RESERVATION = 4096
TIER_MULTIPLIER = 2
SCOPES = frozenset({"developer_smoke", "public_eval", "holdout", "http_ui", "serve"})
_POLICY_FIELDS = frozenset(
    {
        "version",
        "model",
        "budget_id",
        "cap_units",
        "input_rate_units",
        "output_rate_units",
        "extra_units",
        "pricing_revision",
        "pricing_expires_at",
        "enabled_scopes",
        "enabled",
    }
)
_REQUEST_COLUMNS = frozenset(
    {
        "id",
        "budget",
        "body_hash",
        "owner",
        "scope",
        "state",
        "res_input",
        "res_output",
        "res_cost",
        "used_input",
        "used_output",
        "used_cost",
        "pricing_hash",
        "requested_model",
        "returned_model",
        "provider_request_id",
        "provider_response_id",
        "cached_tokens",
        "reasoning_tokens",
        "created",
        "latency",
    }
)
_DEBIT_COLUMNS = frozenset(
    {
        "id",
        "budget",
        "body_hash",
        "pricing_hash",
        "res_cost",
        "res_input",
        "res_output",
        "decision_hash",
        "review_hash",
        "ceiling_fingerprint",
        "at",
        "decision_metadata",
    }
)
_DEBIT_MATCH = (
    "EXISTS (SELECT 1 FROM conservative_debits c WHERE c.id=requests.id "
    "AND c.budget=requests.budget AND c.body_hash=requests.body_hash "
    "AND c.pricing_hash=requests.pricing_hash AND c.res_cost=requests.res_cost "
    "AND c.res_input=requests.res_input AND c.res_output=requests.res_output)"
)
_REQUEST_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")
_BODY_HASH = re.compile(r"[a-f0-9]{64}")
_BUDGET_ID = re.compile(r"[A-Za-z0-9_-]{1,80}")
_SCHEMA = """
CREATE TABLE policy_guard(
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    cap_fingerprint TEXT NOT NULL
);
CREATE TABLE budgets(
    id TEXT PRIMARY KEY,
    binding TEXT NOT NULL
);
CREATE TABLE requests(
    id TEXT PRIMARY KEY,
    budget TEXT,
    body_hash TEXT,
    owner TEXT,
    scope TEXT,
    state TEXT,
    res_input INTEGER,
    res_output INTEGER,
    res_cost INTEGER,
    used_input INTEGER,
    used_output INTEGER,
    used_cost INTEGER,
    pricing_hash TEXT,
    requested_model TEXT,
    returned_model TEXT,
    provider_request_id TEXT,
    provider_response_id TEXT,
    cached_tokens INTEGER,
    reasoning_tokens INTEGER,
    created REAL,
    latency REAL
);
CREATE TABLE conservative_debits(
    id TEXT PRIMARY KEY,
    budget TEXT,
    body_hash TEXT,
    pricing_hash TEXT,
    res_cost INTEGER,
    res_input INTEGER,
    res_output INTEGER,
    decision_hash TEXT,
    review_hash TEXT,
    ceiling_fingerprint TEXT,
    at REAL,
    decision_metadata TEXT
);
CREATE TABLE diagnostics(
    id TEXT,
    metadata TEXT NOT NULL,
    at REAL
);
CREATE TABLE events(
    id TEXT,
    event TEXT,
    at REAL
);
"""


class BudgetError(Exception):
    """A safe accounting failure. The message is a fixed English code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def fingerprint(value: object) -> str:
    """Hash the canonical JSON used by the existing ledger binding."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _exact_int(value: object, *, minimum: int, maximum: int = 2**60) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise BudgetError("policy_invalid")
    return value


def _aware(value: object) -> datetime:
    if not isinstance(value, str):
        raise BudgetError("policy_invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise BudgetError("policy_invalid") from None
    if parsed.tzinfo is None:
        raise BudgetError("policy_invalid")
    return parsed


@dataclass(frozen=True)
class LunaPolicy:
    """One explicit policy. Amount fields stay out of ``repr``."""

    budget_id: str
    scopes: tuple[str, ...]
    pricing_expires_at: datetime
    pricing_revision: str
    cap_units: int = field(repr=False)
    input_rate_units: int = field(repr=False)
    output_rate_units: int = field(repr=False)
    extra_units: int = field(repr=False)

    def reservation_cost(self, input_tokens: int, output_tokens: int) -> int:
        """Return the historical local reservation. This is not an invoice."""
        raw = (
            input_tokens * self.input_rate_units
            + output_tokens * self.output_rate_units
            + 999_999
        ) // 1_000_000 + self.extra_units
        cost = TIER_MULTIPLIER * raw
        if cost > 2**60:
            raise BudgetError("budget_unavailable")
        return cost

    def cap_fingerprint(self) -> str:
        return fingerprint({"ceiling": self.cap_units})

    def binding(self) -> str:
        """Match the existing ledger binding, which excludes only the key."""
        expires_at = str(datetime.max.replace(tzinfo=UTC))
        payload = {
            "budget_id": self.budget_id,
            "scopes": self.scopes,
            "expires_at": expires_at,
            "pricing_expires_at": str(self.pricing_expires_at),
            "max_requests": MAX_REQUESTS,
            "max_input_tokens": MAX_INPUT_TOKENS,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "max_total_tokens": MAX_TOTAL_TOKENS,
            "cap_units": self.cap_units,
            "input_rate_units": self.input_rate_units,
            "output_rate_units": self.output_rate_units,
            "extra_units": self.extra_units,
            "pricing_revision": self.pricing_revision,
        }
        return fingerprint(payload)

    def pricing_hash(self) -> str:
        return fingerprint(
            {
                "bound_revision": HISTORICAL_BOUND_REVISION,
                "revision": self.pricing_revision,
                "input": self.input_rate_units,
                "output": self.output_rate_units,
                "extra": self.extra_units,
                "expires": str(self.pricing_expires_at),
                "model": MODEL,
            }
        )


@dataclass(frozen=True)
class LunaRuntime:
    """Caller-supplied key and policy. The price bound is not verified."""

    policy: LunaPolicy
    api_key: str = field(repr=False)
    pricing_bound_verified: bool = False


def load_policy(path: Path) -> LunaPolicy:
    """Read one explicit policy file. This does not open a ledger or ``.env``."""
    raw = _read_private_json(path)
    if not isinstance(raw, dict) or set(raw) != _POLICY_FIELDS:
        raise BudgetError("policy_invalid")
    if (
        raw["version"] != 1
        or raw["model"] != MODEL
        or raw["enabled"] is not True
        or raw["pricing_revision"] != HISTORICAL_PRICING_REVISION
    ):
        raise BudgetError("policy_invalid")
    budget_id = raw["budget_id"]
    if not isinstance(budget_id, str) or _BUDGET_ID.fullmatch(budget_id) is None:
        raise BudgetError("policy_invalid")
    scopes = raw["enabled_scopes"]
    if (
        not isinstance(scopes, list)
        or not scopes
        or any(not isinstance(item, str) or item not in SCOPES for item in scopes)
        or len(set(scopes)) != len(scopes)
    ):
        raise BudgetError("policy_invalid")
    expires = _aware(raw["pricing_expires_at"])
    if datetime.now(UTC) >= expires:
        raise BudgetError("policy_invalid")
    return LunaPolicy(
        budget_id=budget_id,
        scopes=tuple(scopes),
        pricing_expires_at=expires,
        pricing_revision=HISTORICAL_PRICING_REVISION,
        cap_units=_exact_int(raw["cap_units"], minimum=1),
        input_rate_units=_exact_int(raw["input_rate_units"], minimum=1),
        output_rate_units=_exact_int(raw["output_rate_units"], minimum=1),
        extra_units=_exact_int(raw["extra_units"], minimum=0),
    )


def load_runtime(path: Path, api_key: str) -> LunaRuntime:
    """Build a runtime from an explicit policy path and key argument.

    The environment is not read. ``pricing_bound_verified`` stays false: the
    historical formula is not a same-day fee ceiling.
    """
    if (
        not isinstance(api_key, str)
        or not 8 <= len(api_key) <= 512
        or any(
            character.isspace() or ord(character) < 33 or ord(character) > 126
            for character in api_key
        )
    ):
        raise BudgetError("policy_invalid")
    policy = load_policy(path)
    if api_key in policy.budget_id or any(api_key in scope for scope in policy.scopes):
        raise BudgetError("policy_invalid")
    return LunaRuntime(policy=policy, api_key=api_key, pricing_bound_verified=False)


def init_synthetic_ledger(path: Path, policy: LunaPolicy) -> None:
    """Create a fixture database. Refuses an existing file and does not migrate."""
    if not isinstance(policy, LunaPolicy):
        raise BudgetError("ledger_unavailable")
    if path.exists() or path.is_symlink() or not path.parent.is_dir():
        raise BudgetError("ledger_unavailable")
    connection = sqlite3.connect(path)
    try:
        connection.executescript(_SCHEMA)
        connection.execute(
            "INSERT INTO policy_guard VALUES (1, ?)", (policy.cap_fingerprint(),)
        )
        connection.execute(
            "INSERT INTO budgets VALUES (?, ?)", (policy.budget_id, policy.binding())
        )
        connection.commit()
    except sqlite3.DatabaseError:
        raise BudgetError("ledger_schema_or_storage_unavailable") from None
    finally:
        connection.close()
    path.chmod(0o600)


def open_ledger(path: Path) -> Ledger:
    """Open an existing compatible ledger. Missing files are not created."""
    if path.is_symlink() or not path.is_file():
        raise BudgetError("ledger_unavailable")
    uri = path.absolute().as_uri() + "?mode=rw"
    try:
        connection = sqlite3.connect(uri, uri=True, timeout=2)
    except sqlite3.DatabaseError:
        raise BudgetError("ledger_unavailable") from None
    try:
        _require_schema(connection)
    finally:
        connection.close()
    return Ledger(path)


class Ledger:
    """Reserve, mark sent, release, retain, or settle inside one transaction."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.owner = os.urandom(16).hex()

    def reserve(
        self,
        policy: LunaPolicy,
        request_id: str,
        body_hash: str,
        input_tokens: int,
        output_tokens: int,
        scope: str,
        api_key: str,
    ) -> None:
        """Atomically admit one new reservation or leave the file unchanged."""
        self._check_reserve_shape(
            policy, request_id, body_hash, input_tokens, output_tokens, scope, api_key
        )
        cost = policy.reservation_cost(input_tokens, output_tokens)
        with self._transaction() as db:
            guard = db.execute(
                "SELECT cap_fingerprint FROM policy_guard WHERE singleton=1"
            ).fetchone()
            if guard is None:
                raise BudgetError("cap_binding_missing")
            if guard[0] != policy.cap_fingerprint():
                raise BudgetError("cap_change_requires_authorization")
            old = db.execute(
                "SELECT body_hash, state FROM requests WHERE id=?", (request_id,)
            ).fetchone()
            if old and (old[0] != body_hash or old[1] != "released"):
                raise BudgetError("duplicate_or_conflicting_request")
            if db.execute(
                "SELECT 1 FROM conservative_debits WHERE body_hash=?", (body_hash,)
            ).fetchone():
                raise BudgetError("permanent_original_body_replay")
            if db.execute(
                "SELECT 1 FROM requests WHERE (state='unknown' AND NOT "
                + _DEBIT_MATCH
                + ") OR state='sent' OR (state='reserved' AND owner!=?)",
                (self.owner,),
            ).fetchone():
                raise BudgetError("unreconciled_or_foreign_reservation")
            if db.execute(
                "SELECT 1 FROM budgets WHERE id!=?", (policy.budget_id,)
            ).fetchone():
                raise BudgetError("budget_identity_change_requires_reconciliation")
            binding = db.execute(
                "SELECT binding FROM budgets WHERE id=?", (policy.budget_id,)
            ).fetchone()
            if binding is None:
                raise BudgetError("budget_binding_missing")
            if binding[0] != policy.binding():
                raise BudgetError("budget_configuration_changed")
            used = db.execute(
                "SELECT count(*), "
                "coalesce(sum(coalesce(used_input, res_input)), 0), "
                "coalesce(sum(coalesce(used_output, res_output)), 0), "
                "coalesce(sum(coalesce(used_cost, res_cost)), 0) "
                "FROM requests WHERE budget=? AND state!='released'",
                (policy.budget_id,),
            ).fetchone()
            if (
                used[0] + 1 > MAX_REQUESTS
                or used[1] + input_tokens > MAX_INPUT_TOKENS
                or used[2] + output_tokens > MAX_OUTPUT_TOKENS
                or used[1] + used[2] + input_tokens + output_tokens > MAX_TOTAL_TOKENS
                or used[3] + cost >= policy.cap_units
            ):
                raise BudgetError("budget_exhausted")
            if (
                db.execute(
                    "SELECT count(*) FROM requests WHERE state IN ('reserved', 'sent')"
                ).fetchone()[0]
                >= 3
            ):
                raise BudgetError("reservation_capacity")
            if old:
                db.execute("DELETE FROM requests WHERE id=?", (request_id,))
            db.execute(
                "INSERT INTO requests("
                "id, budget, body_hash, owner, scope, state, res_input, res_output, "
                "res_cost, pricing_hash, requested_model, created"
                ") VALUES (?, ?, ?, ?, ?, 'reserved', ?, ?, ?, ?, ?, ?)",
                (
                    request_id,
                    policy.budget_id,
                    body_hash,
                    self.owner,
                    scope,
                    input_tokens,
                    output_tokens,
                    cost,
                    policy.pricing_hash(),
                    MODEL,
                    time.time(),
                ),
            )
            _event(db, request_id, "reserved")

    def mark_sent(self, request_id: str) -> None:
        """Durably mark one owned reservation sent, before any transport call."""
        with self._transaction() as db:
            if db.execute(
                "SELECT 1 FROM requests WHERE state='sent' OR (state='unknown' AND NOT "
                + _DEBIT_MATCH
                + ")"
            ).fetchone():
                raise BudgetError("unreconciled_or_foreign_reservation")
            if (
                db.execute(
                    "UPDATE requests SET state='sent' WHERE id=? AND owner=? "
                    "AND state='reserved'",
                    (request_id, self.owner),
                ).rowcount
                != 1
            ):
                raise BudgetError("reservation_not_marked")
            _event(db, request_id, "sent")

    def release(self, request_id: str) -> None:
        """Release one owned reservation that was never marked sent."""
        with self._transaction() as db:
            if db.execute(
                "UPDATE requests SET state='released' WHERE id=? AND owner=? "
                "AND state='reserved'",
                (request_id, self.owner),
            ).rowcount:
                _event(db, request_id, "released_before_send")

    def retain_unknown(self, request_id: str) -> None:
        """Keep the full reservation of one request this process marked sent."""
        with self._transaction() as db:
            if db.execute(
                "UPDATE requests SET state='unknown' WHERE id=? AND owner=? "
                "AND state='sent'",
                (request_id, self.owner),
            ).rowcount:
                _event(db, request_id, "unknown_cost_retained")

    def settle(
        self,
        policy: LunaPolicy,
        request_id: str,
        *,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int | None,
        reasoning_tokens: int | None,
        returned_model: str,
        provider_request_id: str | None,
        provider_response_id: str | None,
        latency_seconds: float,
    ) -> None:
        """Record a local usage ceiling. This does not claim the provider settled."""
        if (
            type(input_tokens) is not int
            or type(output_tokens) is not int
            or input_tokens < 0
            or output_tokens < 0
            or returned_model != MODEL
        ):
            raise BudgetError("usage_exceeds_reservation")
        cost = policy.reservation_cost(input_tokens, output_tokens)
        with self._transaction() as db:
            if db.execute(
                "SELECT 1 FROM conservative_debits WHERE id=?", (request_id,)
            ).fetchone():
                raise BudgetError("permanent_conservative_debit")
            row = db.execute(
                "SELECT res_input, res_output, res_cost, pricing_hash, state "
                "FROM requests WHERE id=? AND owner=?",
                (request_id, self.owner),
            ).fetchone()
            if row is None or row[4] not in ("sent", "unknown"):
                raise BudgetError("reservation_not_settleable")
            if (
                row[3] != policy.pricing_hash()
                or input_tokens > row[0]
                or output_tokens > row[1]
                or cost > row[2]
            ):
                raise BudgetError("usage_exceeds_reservation")
            db.execute(
                "UPDATE requests SET state='settled', used_input=?, used_output=?, "
                "used_cost=?, cached_tokens=?, reasoning_tokens=?, returned_model=?, "
                "provider_request_id=?, provider_response_id=?, latency=? WHERE id=?",
                (
                    input_tokens,
                    output_tokens,
                    cost,
                    cached_tokens,
                    reasoning_tokens,
                    returned_model,
                    provider_request_id,
                    provider_response_id,
                    latency_seconds,
                    request_id,
                ),
            )
            _event(db, request_id, "settled_usage_upper_bound")

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        uri = self.path.absolute().as_uri() + "?mode=rw"
        connection = sqlite3.connect(uri, uri=True, timeout=2, isolation_level=None)
        try:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except sqlite3.DatabaseError:
            connection.rollback()
            raise BudgetError("ledger_schema_or_storage_unavailable") from None
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _check_reserve_shape(
        self,
        policy: LunaPolicy,
        request_id: str,
        body_hash: str,
        input_tokens: int,
        output_tokens: int,
        scope: str,
        api_key: str,
    ) -> None:
        if not isinstance(policy, LunaPolicy) or not isinstance(api_key, str):
            raise BudgetError("budget_unavailable")
        if (
            not isinstance(request_id, str)
            or _REQUEST_ID.fullmatch(request_id) is None
            or not isinstance(body_hash, str)
            or _BODY_HASH.fullmatch(body_hash) is None
            or api_key in request_id
            or api_key in body_hash
        ):
            raise BudgetError("budget_unavailable")
        if scope not in policy.scopes or scope not in SCOPES:
            raise BudgetError("budget_unavailable")
        if (
            type(input_tokens) is not int
            or type(output_tokens) is not int
            or input_tokens < 1
            or output_tokens != OUTPUT_RESERVATION
            or input_tokens + output_tokens > TOTAL_RESERVATION
        ):
            raise BudgetError("budget_unavailable")


def _event(connection: sqlite3.Connection, request_id: str, name: str) -> None:
    connection.execute(
        "INSERT INTO events VALUES (?, ?, ?)", (request_id, name, time.time())
    )


def _read_private_json(path: Path) -> object:
    if not isinstance(path, Path) or path.is_symlink():
        raise BudgetError("policy_unavailable")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise BudgetError("policy_unavailable") from None
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_size > 16384
            or info.st_mode & 0o077
        ):
            raise BudgetError("policy_unavailable")
        with os.fdopen(descriptor, "r", encoding="utf-8", closefd=False) as handle:
            try:
                return json.load(handle)
            except json.JSONDecodeError:
                raise BudgetError("policy_invalid") from None
    finally:
        os.close(descriptor)


def _require_schema(connection: sqlite3.Connection) -> None:
    required = {
        "policy_guard": {"singleton", "cap_fingerprint"},
        "budgets": {"id", "binding"},
        "requests": _REQUEST_COLUMNS,
        "conservative_debits": _DEBIT_COLUMNS,
        "diagnostics": {"id", "metadata", "at"},
        "events": {"id", "event", "at"},
    }
    try:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if not set(required) <= tables:
            raise BudgetError("ledger_schema_incompatible")
        for table, columns in required.items():
            found = {
                row[1] for row in connection.execute(f"PRAGMA table_info({table})")
            }
            if not columns <= found:
                raise BudgetError("ledger_schema_incompatible")
    except sqlite3.DatabaseError:
        raise BudgetError("ledger_schema_incompatible") from None
