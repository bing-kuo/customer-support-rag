"""JSON encodings used by maintenance records.

Human files keep insertion order and a trailing newline. Canonical JSON is
only an in-memory byte string for hashes.
"""

import json

from customer_support_rag.maintenance.errors import MaintenanceError

_MAX_RECORD_BYTES = 256 * 1024


def human_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def load_object(path: object, raw: bytes) -> dict[str, object]:
    """Parse one JSON object. ``path`` is unused and keeps callers from logging it."""
    del path
    if len(raw) > _MAX_RECORD_BYTES:
        raise MaintenanceError("record is invalid")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise MaintenanceError("record is invalid") from None
    if not isinstance(payload, dict):
        raise MaintenanceError("record is invalid")
    return payload
