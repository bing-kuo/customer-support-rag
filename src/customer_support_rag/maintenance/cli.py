"""Command line for mock draft hashes, candidates, and explicit activation.

There is no approve command. Reviewer identity is read from an existing record.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from customer_support_rag.maintenance.approval import (
    consume_document_approval,
    consume_release_approval,
)
from customer_support_rag.maintenance.errors import MaintenanceError
from customer_support_rag.maintenance.evaluate import evaluate_release
from customer_support_rag.maintenance.hashes import (
    describe_document,
    scan_current_sources,
)
from customer_support_rag.maintenance.json_text import human_json
from customer_support_rag.maintenance.release import (
    activate,
    build_candidate,
    load_active_snapshot,
)


def main(argv: list[str] | None = None) -> int:
    """Run one maintenance command. Returns a process status code."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except MaintenanceError as exc:
        print(str(exc), file=sys.stderr)
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="knowledge-maintenance")
    commands = parser.add_subparsers(dest="command", required=True)
    hashes = commands.add_parser("hashes")
    hashes.add_argument("--root", type=Path, required=True)
    hashes.add_argument("--path", required=True)
    hashes.set_defaults(func=_hashes)
    scan = commands.add_parser("scan")
    scan.add_argument("--root", type=Path, required=True)
    scan.set_defaults(func=_scan)
    consume_document = commands.add_parser("consume-document")
    _workspace(consume_document)
    consume_document.add_argument("--root", type=Path, required=True)
    consume_document.add_argument("--approval", type=Path, required=True)
    consume_document.set_defaults(func=_consume_document)
    build = commands.add_parser("build")
    _workspace(build)
    build.add_argument("--root", type=Path, required=True)
    build.add_argument("--source", action="append", required=True)
    build.set_defaults(func=_build)
    evaluate = commands.add_parser("evaluate")
    _workspace(evaluate)
    evaluate.add_argument("--root", type=Path, required=True)
    evaluate.add_argument("--release", required=True)
    evaluate.add_argument("--suite", type=Path, required=True)
    evaluate.add_argument("--code-sha", required=True)
    evaluate.set_defaults(func=_evaluate)
    consume_release = commands.add_parser("consume-release")
    _workspace(consume_release)
    consume_release.add_argument("--approval", type=Path, required=True)
    consume_release.set_defaults(func=_consume_release)
    activated = commands.add_parser("activate")
    _workspace(activated)
    activated.add_argument("--release", required=True)
    group = activated.add_mutually_exclusive_group(required=True)
    group.add_argument("--expect")
    group.add_argument("--expect-none", action="store_true")
    activated.set_defaults(func=_activate)
    status = commands.add_parser("status")
    _workspace(status)
    status.set_defaults(func=_status)
    return parser


def _workspace(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", type=Path, required=True)


def _hashes(args: argparse.Namespace) -> int:
    _print(describe_document(args.root, args.path).as_dict())
    return 0


def _scan(args: argparse.Namespace) -> int:
    _print([item.as_dict() for item in scan_current_sources(args.root)])
    return 0


def _consume_document(args: argparse.Namespace) -> int:
    path = consume_document_approval(args.workspace, args.root, args.approval)
    _print({"consumed": path.name})
    return 0


def _build(args: argparse.Namespace) -> int:
    result = build_candidate(args.workspace, args.root, tuple(args.source))
    _print(
        {
            "release_id": result.release_id,
            "noop": result.noop,
            "embedded": result.embedded,
            "copied": result.copied,
        }
    )
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    path = evaluate_release(
        args.workspace,
        args.root,
        args.release,
        args.suite,
        code_sha=args.code_sha,
    )
    _print({"regression": path.name})
    return 0


def _consume_release(args: argparse.Namespace) -> int:
    path = consume_release_approval(args.workspace, args.approval)
    _print({"consumed": path.name})
    return 0


def _activate(args: argparse.Namespace) -> int:
    expected = None if args.expect_none else args.expect
    outcome = activate(args.workspace, args.release, expected_previous=expected)
    _print({"outcome": outcome})
    return 0


def _status(args: argparse.Namespace) -> int:
    try:
        snapshot = load_active_snapshot(args.workspace)
    except MaintenanceError as exc:
        _print({"active": None, "error": str(exc)})
        return 1
    _print(
        {
            "active": snapshot.release_id,
            "manifest_sha256": snapshot.manifest_sha256,
            "database_sha256": snapshot.database_sha256,
        }
    )
    return 0


def git_head(cwd: Path) -> str:
    """Return the current commit SHA without changing the repository."""
    output = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=cwd, text=True
    ).strip()
    return output


def _print(payload: object) -> None:
    sys.stdout.write(human_json(payload))
