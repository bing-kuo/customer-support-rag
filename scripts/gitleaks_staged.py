"""Scan the exact Git index snapshot with the project's pinned Gitleaks image."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unicodedata
from collections.abc import Callable
from pathlib import Path, PurePosixPath

GITLEAKS_IMAGE = (
    "ghcr.io/gitleaks/gitleaks@"
    "sha256:e1b35e12a8c6fa8901f060459cfb6b2fc4c484d3afbe3b029733a3bbfab07055"
)


class ScanError(Exception):
    """Raised when the staged snapshot cannot be scanned safely."""


def parse_index_entries(index_output: bytes) -> dict[str, tuple[str, str]]:
    entries: dict[str, tuple[str, str]] = {}
    normalized_paths: set[str] = set()

    for record in index_output.split(b"\0"):
        if not record:
            continue

        metadata, separator, path_bytes = record.partition(b"\t")
        if not separator:
            raise ScanError("Git returned a malformed staged index entry")

        try:
            mode, object_id, stage = metadata.decode("ascii").split()
            path = path_bytes.decode("utf-8")
        except (UnicodeDecodeError, ValueError):
            raise ScanError("Staged paths must be valid UTF-8") from None

        path_parts = path.split("/")
        pure_path = PurePosixPath(path)
        if (
            not path
            or "\\" in path
            or pure_path.is_absolute()
            or any(part in {"", ".", ".."} for part in path_parts)
            or path_parts[0].casefold() == ".git"
        ):
            raise ScanError("Refusing an unsafe staged path")
        if mode not in {"100644", "100755"} or stage != "0":
            raise ScanError("Only resolved regular files can be scanned")

        normalized_path = unicodedata.normalize(
            "NFD", unicodedata.normalize("NFD", path).casefold()
        )
        if normalized_path in normalized_paths:
            raise ScanError(
                "Refusing staged paths that collide after Unicode normalization "
                "and case folding"
            )
        normalized_paths.add(normalized_path)
        entries[path] = (mode, object_id)

    return entries


def read_staged_entries(repository: Path) -> dict[str, tuple[str, str]]:
    result = subprocess.run(
        ["git", "-C", str(repository), "ls-files", "--stage", "-z"],
        check=True,
        capture_output=True,
    )
    return parse_index_entries(result.stdout)


def export_staged_snapshot(repository: Path, snapshot: Path) -> None:
    for path, (mode, object_id) in read_staged_entries(repository).items():
        destination = snapshot.joinpath(*PurePosixPath(path).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        blob = subprocess.run(
            ["git", "-C", str(repository), "cat-file", "blob", object_id],
            check=True,
            capture_output=True,
        ).stdout
        destination.write_bytes(blob)
        destination.chmod(0o755 if mode == "100755" else 0o644)


def run_gitleaks(
    snapshot: Path, runner: Callable[..., subprocess.CompletedProcess] | None = None
) -> None:
    command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{snapshot}:/scan:ro",
        GITLEAKS_IMAGE,
        "dir",
        "/scan",
        "--redact",
        "--no-banner",
    ]
    try:
        result = (runner or subprocess.run)(command, check=False)
    except OSError:
        raise ScanError("Could not start the pinned Gitleaks scanner") from None
    if result.returncode != 0:
        raise ScanError(f"Gitleaks exited with status {result.returncode}")


def repository_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip())


def main() -> int:
    try:
        repository = repository_root()
        with tempfile.TemporaryDirectory(
            prefix=".gitleaks-staged-", dir=repository
        ) as temporary_directory:
            snapshot = Path(temporary_directory) / "snapshot"
            snapshot.mkdir()
            export_staged_snapshot(repository, snapshot)
            run_gitleaks(snapshot)
    except (OSError, subprocess.SubprocessError, ScanError) as error:
        print(f"Staged secret scan failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
