"""Explicitly enable this clone's project-local Git hooks when safe."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HOOKS_PATH = ".githooks"


class SetupError(Exception):
    """Raised when enabling project hooks could override user configuration."""


def git(
    repository: Path, *arguments: str, check: bool = True
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=check,
        capture_output=True,
        text=True,
    )


def repository_root() -> Path:
    return Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )


def git_directory(repository: Path, option: str) -> Path:
    output = git(repository, "rev-parse", option).stdout.strip()
    path = Path(output)
    return (repository / path).resolve() if not path.is_absolute() else path.resolve()


def configured_hook_paths(repository: Path) -> list[tuple[str, str]]:
    result = git(
        repository,
        "config",
        "--show-origin",
        "--get-all",
        "core.hooksPath",
        check=False,
    )
    if result.returncode == 1:
        return []
    if result.returncode != 0:
        raise SetupError("Could not inspect existing core.hooksPath settings")

    settings: list[tuple[str, str]] = []
    for line in result.stdout.splitlines():
        origin, separator, value = line.partition("\t")
        if not separator:
            raise SetupError("Could not parse existing core.hooksPath settings")
        settings.append((origin, value))
    return settings


def ensure_no_existing_hooks(repository: Path, git_dir: Path) -> None:
    settings = configured_hook_paths(repository)
    if settings:
        if all(value == HOOKS_PATH for _, value in settings):
            return
        descriptions = ", ".join(f"{origin}: {value}" for origin, value in settings)
        raise SetupError(
            "Existing core.hooksPath setting was preserved ("
            + descriptions
            + "). Review it and chain the project hook explicitly if desired."
        )

    hooks_directory = git_dir / "hooks"
    if hooks_directory.exists():
        custom_hooks = sorted(
            path.name
            for path in hooks_directory.iterdir()
            if path.is_file() and not path.name.endswith(".sample")
        )
        if custom_hooks:
            raise SetupError(
                "Existing Git hooks were preserved ("
                + ", ".join(custom_hooks)
                + "). Review or chain them before enabling project hooks."
            )


def ensure_standalone_clone(repository: Path, git_dir: Path, common_dir: Path) -> None:
    if git_dir != common_dir:
        raise SetupError(
            "Refusing to change shared hook configuration from a linked worktree. "
            "Run this installer in a standalone clone before creating worktrees."
        )

    worktree_output = git(repository, "worktree", "list", "--porcelain").stdout
    worktrees = [
        line.removeprefix("worktree ")
        for line in worktree_output.splitlines()
        if line.startswith("worktree ")
    ]
    if len(worktrees) != 1:
        raise SetupError(
            "Refusing to change shared hook configuration while linked worktrees "
            "exist. Run this installer in a standalone clone before creating worktrees."
        )


def install_hooks(repository: Path) -> str:
    git_dir = git_directory(repository, "--absolute-git-dir")
    common_dir = git_directory(repository, "--git-common-dir")
    hooks_path = repository / HOOKS_PATH / "pre-commit"
    if (
        hooks_path.is_symlink()
        or not hooks_path.is_file()
        or not hooks_path.stat().st_mode & 0o111
    ):
        raise SetupError("The project pre-commit hook is missing or not executable")

    current_settings = configured_hook_paths(repository)
    ensure_no_existing_hooks(repository, git_dir)
    if current_settings:
        return "Project hooks are already enabled; no Git configuration changed."

    ensure_standalone_clone(repository, git_dir, common_dir)
    git(repository, "config", "--local", "core.hooksPath", HOOKS_PATH)
    installed_settings = git(
        repository, "config", "--local", "--get-all", "core.hooksPath"
    )
    if installed_settings.stdout.splitlines() != [HOOKS_PATH]:
        raise SetupError("Could not verify the project-local hook setting")
    return "Enabled project hooks in this standalone clone's local Git configuration."


def main() -> int:
    try:
        print(install_hooks(repository_root()))
    except (OSError, subprocess.SubprocessError, SetupError) as error:
        print(f"Git hook setup failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
