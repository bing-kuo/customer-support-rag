"""Loopback Host and same-origin checks for the local demo.

The request cannot widen the trusted hosts. ``X-Forwarded-Host`` is never a
reason to accept a call. These helpers do not read a key or open a database.
"""

from __future__ import annotations

from urllib.parse import urlparse

LOOPBACK_HOSTNAMES = frozenset({"127.0.0.1", "localhost", "::1"})


def hostname_from_host_header(value: str | None) -> str | None:
    """Return the hostname from a Host header, or None when the header is unsafe."""
    if value is None or not value.isascii() or not value.strip():
        return None
    if value != value.strip() or any(ch in value for ch in " \t\r\n,"):
        return None
    if len(value) > 253:
        return None
    if value.startswith("["):
        close = value.find("]")
        if close < 2:
            return None
        name = value[1:close]
        rest = value[close + 1 :]
        if rest and not _port_suffix(rest):
            return None
        return name.lower()
    if ":" in value:
        name, port = value.rsplit(":", 1)
        if not name or not port.isdigit() or int(port) > 65535 or len(port) > 5:
            return None
        return name.lower()
    return value.lower()


def host_is_trusted(host_header: str | None, extra_hosts: tuple[str, ...] = ()) -> bool:
    """True when Host is loopback or an explicit extra hostname."""
    hostname = hostname_from_host_header(host_header)
    if hostname is None:
        return False
    allowed = {item.lower() for item in extra_hosts}
    allowed.update(LOOPBACK_HOSTNAMES)
    return hostname in allowed


def forwarded_host_present(headers: object) -> bool:
    """True when a proxy host header is present and must not be trusted."""
    value = headers.get("x-forwarded-host")
    return value is not None and value != ""


def origin_is_same(origin: str, scheme: str, host_header: str) -> bool:
    """True when Origin matches this request's scheme, host, and port.

    ``null`` and cross-origin values are False. A missing Origin is handled by
    the caller so a CLI request without the header can proceed.
    """
    if origin.lower() == "null":
        return False
    parsed = urlparse(origin)
    if parsed.scheme.lower() not in {"http", "https"}:
        return False
    if parsed.username is not None or parsed.password is not None:
        return False
    if parsed.path or parsed.params or parsed.query or parsed.fragment:
        return False
    if not parsed.netloc or parsed.netloc.lower() != host_header.lower():
        return False
    return parsed.scheme.lower() == scheme.lower()


def _port_suffix(value: str) -> bool:
    if not value.startswith(":") or not value[1:].isdigit():
        return False
    port = value[1:]
    return 1 <= len(port) <= 5 and int(port) <= 65535
