"""Root confinement for maintenance files.

Ancestor symlinks of the chosen root are resolved once. Every path beneath
that root must be a real file or directory, not a symlink.
"""

from pathlib import Path, PurePosixPath

from customer_support_rag.maintenance.errors import MaintenanceError


def require_directory(path: Path) -> Path:
    """Return the real directory named by ``path``."""
    if not isinstance(path, Path):
        raise MaintenanceError("directory is unavailable")
    if path.is_symlink():
        raise MaintenanceError("symlink is not allowed")
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        raise MaintenanceError("directory is unavailable") from None
    if not resolved.is_dir() or resolved.is_symlink():
        raise MaintenanceError("directory is unavailable")
    return resolved


def markdown_file(root: Path, relative: str) -> Path:
    """Return one Markdown file inside ``root``."""
    return _descend(root, _relative(relative, markdown=True), file=True)


def release_directory(root: Path, release_id: str) -> Path:
    """Return the immutable directory of one release."""
    if not isinstance(release_id, str) or len(release_id) != 64:
        raise MaintenanceError("release is unavailable")
    if any(character not in "0123456789abcdef" for character in release_id):
        raise MaintenanceError("release is unavailable")
    return _descend(root, PurePosixPath("releases") / release_id, file=False)


def _relative(value: object, *, markdown: bool) -> PurePosixPath:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or value.startswith("/")
    ):
        raise MaintenanceError("path is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts:
        raise MaintenanceError("path is invalid")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise MaintenanceError("path is invalid")
    if markdown and path.suffix.lower() != ".md":
        raise MaintenanceError("path is invalid")
    return path


def _descend(root: Path, relative: PurePosixPath, *, file: bool) -> Path:
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise MaintenanceError("symlink is not allowed")
    if file:
        if not current.is_file():
            raise MaintenanceError("document is unavailable")
    elif not current.is_dir():
        raise MaintenanceError("release is unavailable")
    try:
        current.resolve().relative_to(root)
    except (OSError, ValueError):
        raise MaintenanceError("path escapes the root") from None
    return current
