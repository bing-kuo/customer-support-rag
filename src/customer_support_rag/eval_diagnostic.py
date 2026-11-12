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
from dataclasses import dataclass
from pathlib import Path

MAX_DIAGNOSTIC_BODY = 64 * 1024
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


@dataclass(frozen=True)
class EvalDiagnostic:
    root: Path
    candidate: str
    case_id: str
    code_sha: str
    suite_version: str
    suite_sha256: str
    rubric_version: str
    rubric_sha256: str

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
        requested = root.expanduser()
        evidence_root = repo / ".data"
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
            _create_private_root(requested, evidence_root)
            target = requested.absolute()
            info = target.lstat()
            if (
                not stat.S_ISDIR(info.st_mode)
                or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o700
            ):
                raise OSError
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
        )

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
        try:
            _require_private_root(self.root)
            self._write_new(raw_path, body)
            self._write_new(meta_path, json.dumps(meta, sort_keys=True).encode())
        except OSError:
            _unlink_private_record(self.root, raw_path.name)
            _unlink_private_record(self.root, meta_path.name)
            raise DiagnosticCaptureError("diagnostic_storage_unavailable") from None
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
        if not path.is_file():
            raise DiagnosticCaptureError("diagnostic_record_missing")
        root_fd = -1
        try:
            root_fd = _open_verified_root(self.root)
            record_fd = os.open(
                path.name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=root_fd
            )
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
        except Exception:
            if root_fd >= 0:
                try:
                    os.unlink(path.with_suffix(".tmp").name, dir_fd=root_fd)
                except OSError:
                    pass
            raise DiagnosticCaptureError("diagnostic_storage_unavailable") from None
        finally:
            if root_fd >= 0:
                os.close(root_fd)

    @staticmethod
    def _write_new(path: Path, data: bytes) -> None:
        root_fd = _open_verified_root(path.parent)
        try:
            _write_new_at(root_fd, path.name, data)
        finally:
            os.close(root_fd)


def _write_new_at(root_fd: int, name: str, data: bytes) -> None:
    fd = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
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
    _require_private_root(root)
    fd = os.open(
        root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        opened = os.fstat(fd)
        current = root.lstat()
        if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
            raise OSError("diagnostic root changed")
        if not stat.S_ISDIR(opened.st_mode) or stat.S_IMODE(opened.st_mode) != 0o700:
            raise OSError("unsafe diagnostic directory")
        return fd
    except BaseException:
        os.close(fd)
        raise


def _unlink_private_record(root: Path, name: str) -> None:
    try:
        root_fd = _open_verified_root(root)
    except OSError:
        return
    try:
        os.unlink(name, dir_fd=root_fd)
    except OSError:
        pass
    finally:
        os.close(root_fd)


def _is_within_without_symlinks(path: Path, parent: Path) -> bool:
    """Check lexical containment and reject every existing symlink component."""
    path = path.absolute()
    parent = parent.absolute()
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    current = Path(path.anchor)
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
    return True


def _create_private_root(path: Path, parent: Path) -> None:
    """Create missing descendants without changing existing directory modes."""
    relative = path.absolute().relative_to(parent.absolute())
    current = parent.absolute()
    try:
        current.mkdir(mode=0o700)
    except FileExistsError:
        info = current.lstat()
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise OSError("unsafe diagnostic directory") from None
    for component in relative.parts:
        current = current / component
        try:
            current.mkdir(mode=0o700)
        except FileExistsError:
            info = current.lstat()
            if (
                not stat.S_ISDIR(info.st_mode)
                or stat.S_ISLNK(info.st_mode)
                or info.st_uid != os.getuid()
                or (current == path.absolute() and stat.S_IMODE(info.st_mode) != 0o700)
            ):
                raise OSError("unsafe diagnostic directory") from None


def _require_private_root(root: Path) -> None:
    info = root.lstat()
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise OSError("unsafe diagnostic directory")


def _contains_secret_shape(body: bytes, api_key: str) -> bool:
    """Detect literal and JSON-escaped key text, including nested JSON strings."""
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        return False
    pending = [text]
    seen: set[str] = set()
    while pending:
        value = pending.pop()
        if value in seen:
            continue
        seen.add(value)
        if api_key in value:
            return True
        unescaped = re.sub(
            r"\\u([0-9a-fA-F]{4})",
            lambda match: chr(int(match.group(1), 16)),
            value,
        )
        if unescaped != value:
            pending.append(unescaped)
        try:
            decoded = json.loads(value)
        except (ValueError, TypeError):
            continue
        if isinstance(decoded, str):
            pending.append(decoded)
        elif isinstance(decoded, list):
            pending.extend(_string_values(decoded))
        elif isinstance(decoded, dict):
            pending.extend(_string_values(decoded))
    return False


def _string_values(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [item for pair in value.items() for item in _string_values(pair)]
    if isinstance(value, (list, tuple)):
        return [item for child in value for item in _string_values(child)]
    return []


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
