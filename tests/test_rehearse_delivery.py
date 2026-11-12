"""CLI exit behavior for the offline delivery rehearsal."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).parents[1]
SCRIPT = REPO / "scripts" / "rehearse_delivery.py"


def test_help_exits_without_running_the_rehearsal() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "mock-only knowledge lifecycle rehearsal" in result.stdout


def test_unknown_argument_exits_with_usage_error() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--unexpected-option"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "unrecognized arguments" in result.stderr


def test_faq_manual_build_uses_rehearsal_source_path() -> None:
    docs = (REPO / "docs" / "reproducible-delivery.md").read_text(encoding="utf-8")

    assert (REPO / "knowledge" / "drafts/hp-a/charging-faq.md").is_file()
    assert "--source drafts/hp-a/charging-faq.md" in docs
    assert "--source hp-a/charging-faq.md" not in docs
