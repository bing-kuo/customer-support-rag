"""Root confinement for maintenance files.

Ancestor symlinks of the chosen root are resolved once. Every path beneath
that root must be a real file or directory, not a symlink. Writers create
missing directories only through a parent opened without following links.
"""

import os
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
    _must_stay_inside(root, current)
    return current


def ensure_real_directory(root: Path, relative: PurePosixPath) -> Path:
    """Return a real directory under ``root``, creating missing components.

    Each existing component must be a directory, not a symlink, and must stay
    inside ``root``. A missing component is created in its parent with
    ``O_NOFOLLOW``, so a symlink parent or leaf cannot receive the new
    directory.
    """
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise MaintenanceError("path is invalid")
    current = root
    for part in relative.parts:
        nxt = current / part
        if nxt.is_symlink():
            raise MaintenanceError("symlink is not allowed")
        if nxt.exists():
            if not nxt.is_dir():
                raise MaintenanceError("directory is unavailable")
            _must_stay_inside(root, nxt)
            current = nxt
            continue
        _must_stay_inside(root, current)
        try:
            _mkdir_nofollow(current, part)
        except FileExistsError:
            if nxt.is_symlink() or not nxt.is_dir():
                raise MaintenanceError("symlink is not allowed") from None
        if nxt.is_symlink() or not nxt.is_dir():
            raise MaintenanceError("symlink is not allowed")
        _must_stay_inside(root, nxt)
        current = nxt
    return current


def rename_within(parent: Path, source: str, destination: str) -> None:
    """Rename two entries inside one real directory.

    The parent is opened with ``O_NOFOLLOW``. A symlink parent raises before
    the rename, so the directory cannot move outside the workspace.
    """
    if parent.is_symlink() or not parent.is_dir():
        raise MaintenanceError("symlink is not allowed")
    descriptor = _open_directory_nofollow(parent)
    try:
        os.rename(source, destination, src_dir_fd=descriptor, dst_dir_fd=descriptor)
    except OSError:
        raise MaintenanceError("candidate could not be written") from None
    finally:
        os.close(descriptor)


def _must_stay_inside(root: Path, path: Path) -> None:
    try:
        path.resolve().relative_to(root)
    except (OSError, ValueError):
        raise MaintenanceError("path escapes the root") from None


def _mkdir_nofollow(parent: Path, name: str) -> None:
    descriptor = _open_directory_nofollow(parent)
    try:
        os.mkdir(name, 0o755, dir_fd=descriptor)
    except FileExistsError:
        raise
    except OSError:
        raise MaintenanceError("directory is unavailable") from None
    finally:
        os.close(descriptor)


def _open_directory_nofollow(path: Path) -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        return os.open(path, flags)
    except OSError:
        raise MaintenanceError("symlink is not allowed") from None
