"""Fixed same-origin files from a repository or test web root.

Only ``index.html`` at the web root and the named assets are readable. The
request cannot choose the root. Missing files are absent, not a generated UI.
This module does not read a key or open a database.
"""

from __future__ import annotations

from pathlib import Path

INDEX_NAME = "index.html"
INDEX_CONTENT_TYPE = "text/html; charset=utf-8"
ASSET_CONTENT_TYPES = {
    "style.css": "text/css; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
}
NOSNIFF = {"X-Content-Type-Options": "nosniff"}


def default_web_root() -> Path:
    """Locate ``web/`` next to ``src/`` in this repository checkout."""
    return Path(__file__).resolve().parents[2] / "web"


def index_file(web_root: Path) -> Path | None:
    """Return the homepage file, or None when it is missing or unsafe."""
    return _regular_child(web_root, INDEX_NAME)


def asset_file(web_root: Path, name: str) -> tuple[Path, str] | None:
    """Return a named asset and its content type, or None when it is not served."""
    content_type = ASSET_CONTENT_TYPES.get(name)
    if content_type is None:
        return None
    path = _regular_child(web_root / "assets", name)
    if path is None:
        return None
    return path, content_type


def _regular_child(directory: Path, name: str) -> Path | None:
    if name != Path(name).name or name.startswith(".") or "/" in name or "\\" in name:
        return None
    try:
        if directory.is_symlink() or not directory.is_dir():
            return None
        root = directory.resolve()
    except OSError:
        return None
    candidate = directory / name
    if candidate.is_symlink():
        return None
    try:
        resolved = candidate.resolve()
    except OSError:
        return None
    if resolved.is_symlink() or not resolved.is_file():
        return None
    if not resolved.is_relative_to(root):
        return None
    return resolved
