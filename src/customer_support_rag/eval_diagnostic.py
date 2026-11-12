"""Explicit, owner-only capture for bounded synthetic eval parser diagnosis.

Normal serving never constructs this recorder. It stores only the bounded
provider response body (without request/response headers) and a finite metadata
record; callers must opt in and provide the reviewed eval identity.
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

MAX_DIAGNOSTIC_BODY = 64 * 1024
MAX_SECRET_UNESCAPE_LAYERS = 8
MAX_API_KEY_BYTES = 4096
_CANDIDATES = frozenset({"baseline", "faq", "policy-v1", "policy-v2"})
_CASES = frozenset(
    {
        "refund-generation",
        "critical-generation",
        "faq-absent-baseline",
        "faq-present",
        "policy-v1-window",
        "policy-v2-window",
    }
)
_GIT_SHA40 = re.compile(r"[a-f0-9]{40}")
_HEX64 = re.compile(r"[a-f0-9]{64}")


class DiagnosticCaptureError(Exception):
    """A bounded safe capture failed; the response must not be parsed onward."""


@dataclass
class EvalDiagnostic:
    root: Path
    candidate: str
    case_id: str
    code_sha: str
    suite_version: str
    suite_sha256: str
    rubric_version: str
    rubric_sha256: str
    _root_fd: int = field(repr=False)
    _root_identity: tuple[int, int] = field(repr=False)

    @classmethod
    def opt_in(
        cls,
        root: Path,
        *,
        candidate: str,
        case_id: str,
        code_sha: str,
        suite_version: str,
        suite_sha256: str,
        rubric_version: str,
        rubric_sha256: str,
    ) -> EvalDiagnostic:
        """Explicitly opt in after validating evidence identity and storage."""
        repo = Path(__file__).resolve().parents[2]
        requested = Path(os.path.abspath(root.expanduser()))
        evidence_root = Path(os.path.abspath(repo / ".data"))
        if not _is_within_without_symlinks(requested, evidence_root):
            raise DiagnosticCaptureError("diagnostic_root_not_gitignored_evidence")
        try:
            subprocess.run(
                ["git", "check-ignore", "-q", str(requested)],
                cwd=repo,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.CalledProcessError):
            raise DiagnosticCaptureError(
                "diagnostic_root_not_gitignored_evidence"
            ) from None
        if (
            candidate not in _CANDIDATES
            or case_id not in _CASES
            or not _GIT_SHA40.fullmatch(code_sha)
            or not _HEX64.fullmatch(suite_sha256)
            or not _HEX64.fullmatch(rubric_sha256)
            or suite_version != "dev-suite-v1"
            or rubric_version != "semantic-rubric-v1"
        ):
            raise DiagnosticCaptureError("diagnostic_identity_invalid")
        try:
            target = requested
            root_fd = _create_private_root(target, evidence_root)
            identity = _directory_identity(root_fd)
        except OSError:
            raise DiagnosticCaptureError("diagnostic_storage_unavailable") from None
        return cls(
            target,
            candidate,
            case_id,
            code_sha,
            suite_version,
            suite_sha256,
            rubric_version,
            rubric_sha256,
            root_fd,
            identity,
        )

    def close(self) -> None:
        """Release the pinned diagnostic directory descriptor."""
        root_fd = getattr(self, "_root_fd", -1)
        if root_fd >= 0:
            self._root_fd = -1
            os.close(root_fd)

    def __enter__(self) -> EvalDiagnostic:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except OSError:
            pass

    def capture(self, body: bytes, *, http_status: int, api_key: str) -> Path:
        """Persist one already bounded body, never headers or credentials."""
        if len(body) > MAX_DIAGNOSTIC_BODY:
            raise DiagnosticCaptureError("diagnostic_body_too_large")
        if api_key and _contains_secret_shape(body, api_key):
            raise DiagnosticCaptureError("diagnostic_secret_in_body")
        if type(http_status) is not int or not 100 <= http_status <= 599:
            raise DiagnosticCaptureError("diagnostic_status_invalid")
        stamp = f"{time.time_ns()}-{os.getpid()}"
        raw_path = self.root / f"{stamp}.response.json"
        meta_path = self.root / f"{stamp}.meta.json"
        meta = {
            "schema": "synthetic-eval-parser-diagnostic-v1",
            "candidate": self.candidate,
            "case_id": self.case_id,
            "code_sha": self.code_sha,
            "suite_version": self.suite_version,
            "suite_sha256": self.suite_sha256,
            "rubric_version": self.rubric_version,
            "rubric_sha256": self.rubric_sha256,
            "http_status": http_status,
            "stage": "provider_body_pre_parse",
            "parser_stage": "pending",
            "safe_code": None,
            "response_bytes": len(body),
            "headers_saved": False,
            "authorization_saved": False,
        }
        root_fd = -1
        try:
            root_fd = os.dup(self._root_fd)
            _require_pinned_root(self.root, root_fd, self._root_identity)
            _write_new_at(root_fd, raw_path.name, body)
            _write_new_at(
                root_fd, meta_path.name, json.dumps(meta, sort_keys=True).encode()
            )
            _require_pinned_root(self.root, root_fd, self._root_identity)
        except OSError:
            if root_fd >= 0:
                _unlink_private_at(root_fd, raw_path.name)
                _unlink_private_at(root_fd, meta_path.name)
            raise DiagnosticCaptureError("diagnostic_storage_unavailable") from None
        finally:
            if root_fd >= 0:
                os.close(root_fd)
        return meta_path

    def finish(
        self, metadata_path: Path, *, parser_stage: str, safe_code: str | None
    ) -> None:
        """Attach finite parser outcome to the latest record for this case."""
        allowed_stages = {
            "http_status",
            "provider_envelope",
            "output_parser",
            "parsed",
            "refusal",
        }
        allowed_codes = {
            None,
            "invalid_model_schema",
            "invalid_response",
            "unknown_citation",
            "output_refusal",
            "provider_service_unavailable",
            "provider_capacity",
            "provider_incomplete",
            "provider_failed",
            "provider_cancelled",
            "contradictory_response",
            "unverified_returned_model",
            "unverified_service_tier",
            "unknown_response_terminal_state",
            "parsed",
        }
        if parser_stage not in allowed_stages or safe_code not in allowed_codes:
            raise DiagnosticCaptureError("diagnostic_outcome_invalid")
        path = metadata_path
        if path.parent != self.root or path.suffixes[-2:] != [".meta", ".json"]:
            raise DiagnosticCaptureError("diagnostic_record_missing")
        root_fd = -1
        try:
            root_fd = os.dup(self._root_fd)
            _require_pinned_root(self.root, root_fd, self._root_identity)
            try:
                record_fd = os.open(
                    path.name,
                    os.O_RDONLY | os.O_NOFOLLOW,
                    dir_fd=root_fd,
                )
            except FileNotFoundError:
                raise DiagnosticCaptureError("diagnostic_record_missing") from None
            with os.fdopen(record_fd, "rb") as stream:
                record = json.loads(stream.read().decode("utf-8"))
            record["parser_stage"] = parser_stage
            record["safe_code"] = safe_code
            tmp = path.with_suffix(".tmp")
            _write_new_at(
                root_fd, tmp.name, json.dumps(record, sort_keys=True).encode()
            )
            os.replace(tmp.name, path.name, src_dir_fd=root_fd, dst_dir_fd=root_fd)
            os.chmod(path.name, 0o600, dir_fd=root_fd, follow_symlinks=False)
            _require_pinned_root(self.root, root_fd, self._root_identity)
        except DiagnosticCaptureError:
            if root_fd >= 0:
                try:
                    os.unlink(path.with_suffix(".tmp").name, dir_fd=root_fd)
                except OSError:
                    pass
            raise
        except (OSError, UnicodeError, ValueError, TypeError, RecursionError):
            if root_fd >= 0:
                try:
                    os.unlink(path.with_suffix(".tmp").name, dir_fd=root_fd)
                except OSError:
                    pass
            raise DiagnosticCaptureError("diagnostic_storage_unavailable") from None
        finally:
            if root_fd >= 0:
                os.close(root_fd)


def _write_new_at(root_fd: int, name: str, data: bytes) -> None:
    fd = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=root_fd,
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            os.unlink(name, dir_fd=root_fd)
        except OSError:
            pass
        raise


def _open_verified_root(root: Path) -> int:
    """Open each absolute path component with dir_fd and O_NOFOLLOW."""
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise OSError("no no-follow directory opens")
    root = Path(os.path.abspath(root))
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | nofollow
    fd = os.open(root.anchor, flags)
    try:
        for component in root.parts[1:]:
            next_fd = os.open(component, flags, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        opened = os.fstat(fd)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or opened.st_uid != os.getuid()
            or stat.S_IMODE(opened.st_mode) != 0o700
        ):
            raise OSError("unsafe diagnostic directory")
        return fd
    except BaseException:
        os.close(fd)
        raise


def _directory_identity(fd: int) -> tuple[int, int]:
    info = os.fstat(fd)
    return info.st_dev, info.st_ino


def _require_pinned_root(root: Path, root_fd: int, identity: tuple[int, int]) -> None:
    info = os.fstat(root_fd)
    if (info.st_dev, info.st_ino) != identity:
        raise OSError("diagnostic directory changed")
    check_fd = _open_verified_root(root)
    try:
        if _directory_identity(check_fd) != identity:
            raise OSError("diagnostic path changed")
    finally:
        os.close(check_fd)


def _unlink_private_at(root_fd: int, name: str) -> None:
    try:
        os.unlink(name, dir_fd=root_fd)
    except OSError:
        pass


def _is_within_without_symlinks(path: Path, parent: Path) -> bool:
    """Check lexical containment and reject every existing symlink component."""
    path = Path(os.path.abspath(path))
    parent = Path(os.path.abspath(parent))
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    current = Path(path.anchor)
    try:
        for component in path.parts[1:]:
            candidate = current / component
            try:
                info = candidate.lstat()
            except FileNotFoundError:
                current = candidate
                continue
            if stat.S_ISLNK(info.st_mode):
                return False
            current = candidate
    except OSError:
        return False
    return True


def _create_private_root(path: Path, parent: Path) -> int:
    """Open/create private descendants using only descriptor-relative steps."""
    path = Path(os.path.abspath(path))
    parent = Path(os.path.abspath(parent))
    relative = path.relative_to(parent)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise OSError("no no-follow directory opens")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | nofollow
    fd = os.open(path.anchor, flags)
    parent_components = parent.parts[1:]
    target_components = relative.parts
    try:
        for index, component in enumerate((*parent_components, *target_components)):
            is_private = index >= len(parent_components)
            is_evidence_root = index == len(parent_components) - 1
            if is_private or is_evidence_root:
                try:
                    os.mkdir(component, mode=0o700, dir_fd=fd)
                except FileExistsError:
                    pass
            next_fd = os.open(component, flags, dir_fd=fd)
            os.close(fd)
            fd = next_fd
            if is_private or is_evidence_root:
                info = os.fstat(fd)
                if (
                    not stat.S_ISDIR(info.st_mode)
                    or info.st_uid != os.getuid()
                    or (
                        index == len(parent_components) + len(target_components) - 1
                        and stat.S_IMODE(info.st_mode) != 0o700
                    )
                    or (
                        index != len(parent_components) + len(target_components) - 1
                        and stat.S_IMODE(info.st_mode) & 0o022
                    )
                ):
                    raise OSError("unsafe diagnostic directory")
        return fd
    except BaseException:
        os.close(fd)
        raise


def _contains_secret_shape(body: bytes, api_key: str) -> bool:
    """Search bounded response bytes and JSON string encodings for the key.

    This is a string-token scan rather than a recursive JSON parse so valid
    provider metadata can be arbitrarily deep without exhausting Python's
    parser stack. Decoded string values are rescanned a bounded number of
    times to cover complete JSON documents wrapped inside JSON strings, while
    treating ordinary decoded metadata prose as data.
    """
    try:
        key_bytes = api_key.encode("ascii")
    except UnicodeEncodeError:
        raise DiagnosticCaptureError("diagnostic_body_uninspectable") from None
    if len(key_bytes) > MAX_API_KEY_BYTES:
        raise DiagnosticCaptureError("diagnostic_body_uninspectable")
    if key_bytes in body:
        return True
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        raise DiagnosticCaptureError("diagnostic_body_encoding_invalid") from None
    pending = [(text, 0, False)]
    inspected: set[tuple[str, bool]] = set()
    escaped_unicode = re.compile(r"\\+u([0-9a-fA-F]{4})")
    while pending:
        value, layer, is_json_document = pending.pop()
        if (value, is_json_document) in inspected:
            continue
        inspected.add((value, is_json_document))
        if api_key in value:
            return True
        unescaped = value
        for _ in range(MAX_SECRET_UNESCAPE_LAYERS):
            unescaped = escaped_unicode.sub(
                lambda match: chr(int(match.group(1), 16)), unescaped
            )
            if api_key in unescaped:
                return True
        if escaped_unicode.search(unescaped):
            raise DiagnosticCaptureError("diagnostic_body_uninspectable")
        if is_json_document:
            tokens = _complete_json_string_tokens(value)
            if tokens is None:
                raise DiagnosticCaptureError("diagnostic_body_uninspectable")
            fragments: list[str] = []
        else:
            tokens, fragments = _scan_json_string_tokens(value)
        for decoded in tokens:
            if api_key in decoded:
                return True
            nested_tokens = _complete_json_string_tokens(decoded)
            if nested_tokens is not None:
                if layer >= MAX_SECRET_UNESCAPE_LAYERS:
                    raise DiagnosticCaptureError("diagnostic_body_uninspectable")
                pending.append((decoded, layer + 1, True))
                continue
            # Invalid or prose text is not reinterpreted as a JSON document,
            # but individually complete JSON string tokens still get decoded
            # so a key cannot hide behind a non-JSON prefix such as NBSP.
            decoded_tokens, decoded_fragments = _scan_json_string_tokens(decoded)
            for fragment in decoded_tokens:
                if api_key in fragment:
                    return True
                if _complete_json_string_tokens(fragment) is not None:
                    if layer >= MAX_SECRET_UNESCAPE_LAYERS:
                        raise DiagnosticCaptureError("diagnostic_body_uninspectable")
                    pending.append((fragment, layer + 1, True))
            for fragment in decoded_fragments:
                if _fragment_contains_key(fragment, api_key):
                    return True
        for fragment in fragments:
            if _fragment_contains_key(fragment, api_key):
                return True
    return False


def _json_string_tokens(text: str) -> list[str] | None:
    """Decode each complete JSON string token without descending into objects."""
    tokens, fragments = _scan_json_string_tokens(text)
    return tokens if not fragments else None


def _scan_json_string_tokens(text: str) -> tuple[list[str], list[str]]:
    """Keep complete string tokens even when later text is malformed."""
    values: list[str] = []
    fragments: list[str] = []
    index = 0
    while index < len(text):
        if text[index] != '"':
            index += 1
            continue
        start = index
        index += 1
        while index < len(text):
            character = text[index]
            if character == "\\":
                if index + 1 >= len(text):
                    fragments.append(text[start + 1 :])
                    return values, fragments
                index += 2
            elif character == '"':
                index += 1
                try:
                    decoded = json.loads(text[start:index])
                except (json.JSONDecodeError, RecursionError):
                    fragments.append(text[start + 1 : index - 1])
                else:
                    if isinstance(decoded, str):
                        values.append(decoded)
                    else:
                        fragments.append(text[start + 1 : index - 1])
                break
            else:
                index += 1
        else:
            fragments.append(text[start + 1 :])
            return values, fragments
    return values, fragments


def _decode_json_escapes(text: str) -> str:
    """Decode standard JSON escapes in a fragment, preserving invalid pairs."""
    escapes = {
        '"': '"',
        "\\": "\\",
        "/": "/",
        "b": "\b",
        "f": "\f",
        "n": "\n",
        "r": "\r",
        "t": "\t",
    }
    output: list[str] = []
    index = 0
    while index < len(text):
        if text[index] != "\\" or index + 1 >= len(text):
            output.append(text[index])
            index += 1
            continue
        escape = text[index + 1]
        if escape in escapes:
            output.append(escapes[escape])
            index += 2
        elif escape == "u" and index + 6 <= len(text):
            digits = text[index + 2 : index + 6]
            if re.fullmatch(r"[0-9a-fA-F]{4}", digits):
                output.append(chr(int(digits, 16)))
                index += 6
            else:
                output.append(text[index : index + 2])
                index += 2
        else:
            output.append(text[index : index + 2])
            index += 2
    return "".join(output)


def _fragment_contains_key(fragment: str, api_key: str) -> bool:
    """Search an incomplete token through the bounded JSON escape layers."""
    for _ in range(MAX_SECRET_UNESCAPE_LAYERS + 1):
        if api_key in fragment:
            return True
        decoded = _decode_json_escapes(fragment)
        if decoded == fragment:
            return False
        fragment = decoded
    return api_key in fragment


def _complete_json_string_tokens(text: str) -> list[str] | None:
    """Return string values only when text is one complete JSON document.

    Ordinary decoded prose containing quote marks is ignored. If deeply nested
    JSON exceeds the library parser's stack, scan its bounded string tokens
    without recursively parsing the document.
    """
    start = 0
    while start < len(text) and text[start] in " \t\r\n":
        start += 1
    try:
        decoder = json.JSONDecoder(parse_int=str, parse_float=str)
        _value, end = decoder.raw_decode(text, start)
    except json.JSONDecodeError:
        return None
    except RecursionError:
        return _deep_json_string_tokens(text, start)
    if any(character not in " \t\r\n" for character in text[end:]):
        return None
    return _json_string_tokens(text[start:end])


def _deep_json_string_tokens(text: str, start: int) -> list[str] | None:
    """Bound deeply nested JSON string scanning and honor its exact end."""
    if start >= len(text) or text[start] not in "[{":
        return None
    expected: list[str] = []
    strings: list[str] = []
    index = start
    while index < len(text):
        character = text[index]
        if character == '"':
            token_start = index
            index += 1
            while index < len(text):
                if text[index] == "\\":
                    index += 2
                elif text[index] == '"':
                    index += 1
                    break
                else:
                    index += 1
            else:
                return None
            try:
                value = json.loads(text[token_start:index])
            except (json.JSONDecodeError, RecursionError):
                return None
            if not isinstance(value, str):
                return None
            strings.append(value)
            continue
        if character == "[":
            expected.append("]")
        elif character == "{":
            expected.append("}")
        elif character in "]}":
            if not expected or expected.pop() != character:
                return None
            if not expected:
                if any(char not in " \t\r\n" for char in text[index + 1 :]):
                    return None
                return strings
        index += 1
    return None


def replay_parser_body(body: bytes, request: object):
    """Replay a captured provider body through the unchanged Luna output parser.

    The caller reconstructs the original synthetic ``AnswerRequest`` from its
    candidate snapshot and suite case; this module never stores question or
    evidence text in the diagnostic metadata.
    """
    from customer_support_rag.luna import LunaError, _json_object, parse_luna_output

    if len(body) > MAX_DIAGNOSTIC_BODY:
        raise LunaError("unbounded_response")
    try:
        payload = _json_object(body.decode("utf-8"))
    except UnicodeError:
        raise LunaError("invalid_response") from None
    return parse_luna_output(payload, request)  # type: ignore[arg-type]
