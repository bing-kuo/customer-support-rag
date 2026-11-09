import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).parents[1] / "scripts" / "gitleaks_staged.py"
SPEC = importlib.util.spec_from_file_location("gitleaks_staged", SCRIPT)
gitleaks_staged = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gitleaks_staged)

SYNTHETIC_MARKER = "SYNTHETIC_" + "PRIVATE_MARKER"


class StagedScanTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name) / "repository"
        self.worktree = Path(self.temporary_directory.name) / "linked-worktree"
        self.root.mkdir()
        self.run_git(self.root, "init")
        self.run_git(self.root, "config", "user.name", "Synthetic Test")
        self.run_git(self.root, "config", "user.email", "test@example.com")
        (self.root / "README.md").write_text("Synthetic repository\n")
        self.run_git(self.root, "add", "README.md")
        self.run_git(self.root, "commit", "-m", "test base")
        self.run_git(
            self.root,
            "worktree",
            "add",
            "-b",
            "test-linked-worktree",
            str(self.worktree),
            "HEAD",
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def run_git(self, repository: Path, *arguments: str) -> None:
        subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
        )

    def make_snapshot(self) -> Path:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        snapshot = Path(temporary_directory.name) / "snapshot"
        snapshot.mkdir()
        gitleaks_staged.export_staged_snapshot(self.worktree, snapshot)
        return snapshot

    def fake_scanner(
        self, command: list[str], check: bool = False
    ) -> subprocess.CompletedProcess:
        self.assertIn(gitleaks_staged.GITLEAKS_IMAGE, command)
        self.assertIn("dir", command)
        mount = command[command.index("-v") + 1]
        snapshot = Path(mount.split(":", maxsplit=1)[0])
        content = (snapshot / "credential.txt").read_text()
        return subprocess.CompletedProcess(command, int(SYNTHETIC_MARKER in content))

    def stage_credential(self, content: str) -> None:
        (self.worktree / "credential.txt").write_text(content)
        self.run_git(self.worktree, "add", "credential.txt")

    def store_blob(self, content: str) -> str:
        blob = (
            subprocess.run(
                ["git", "-C", str(self.worktree), "hash-object", "-w", "--stdin"],
                input=content.encode(),
                check=True,
                capture_output=True,
            )
            .stdout.decode()
            .strip()
        )
        return blob

    def index_entry(self, path: str, object_id: str) -> bytes:
        return f"100644 {object_id} 0\t{path}\0".encode()

    def export_with_index_output(self, index_output: bytes, snapshot: Path) -> None:
        real_run = subprocess.run
        commands = []

        def run(command, **kwargs):
            commands.append(command)
            if command[-3:] == ["ls-files", "--stage", "-z"]:
                return subprocess.CompletedProcess(command, 0, index_output, b"")
            return real_run(command, **kwargs)

        try:
            with patch.object(gitleaks_staged.subprocess, "run", side_effect=run):
                gitleaks_staged.export_staged_snapshot(self.worktree, snapshot)
        finally:
            self.export_commands = commands

    def test_clean_staged_file_is_accepted_in_linked_worktree(self):
        self.stage_credential("synthetic clean value\n")
        snapshot = self.make_snapshot()

        gitleaks_staged.run_gitleaks(snapshot, self.fake_scanner)

    def test_staged_synthetic_secret_is_rejected(self):
        self.stage_credential(SYNTHETIC_MARKER + "\n")
        snapshot = self.make_snapshot()

        with self.assertRaisesRegex(gitleaks_staged.ScanError, "status 1"):
            gitleaks_staged.run_gitleaks(snapshot, self.fake_scanner)

    def test_unstaged_content_does_not_replace_staged_blob(self):
        self.stage_credential("staged clean value\n")
        (self.worktree / "credential.txt").write_text(SYNTHETIC_MARKER + "\n")
        snapshot = self.make_snapshot()

        self.assertEqual(
            (snapshot / "credential.txt").read_text(), "staged clean value\n"
        )
        gitleaks_staged.run_gitleaks(snapshot, self.fake_scanner)

    def test_parser_rejects_path_traversal(self):
        index_output = b"100644 " + b"0" * 40 + b" 0\t../outside.txt\0"

        with self.assertRaisesRegex(gitleaks_staged.ScanError, "unsafe staged path"):
            gitleaks_staged.parse_index_entries(index_output)

    def test_canonical_path_collision_fails_before_export(self):
        composed_path = "caf\u00e9.txt"
        decomposed_path = "cafe\u0301.txt"
        secret_blob = self.store_blob(SYNTHETIC_MARKER + "\n")
        clean_blob = self.store_blob("clean content\n")
        self.assertNotEqual(secret_blob, clean_blob)
        index_output = self.index_entry(composed_path, secret_blob) + self.index_entry(
            decomposed_path, clean_blob
        )
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        snapshot = Path(temporary_directory.name) / "snapshot"
        snapshot.mkdir()

        with self.assertRaisesRegex(gitleaks_staged.ScanError, "Unicode normalization"):
            self.export_with_index_output(index_output, snapshot)

        self.assertEqual(list(snapshot.iterdir()), [])
        self.assertEqual(len(self.export_commands), 1)

    def test_case_collision_fails_before_export(self):
        secret_blob = self.store_blob(SYNTHETIC_MARKER + "\n")
        clean_blob = self.store_blob("clean content\n")
        self.assertNotEqual(secret_blob, clean_blob)
        index_output = self.index_entry("Report.txt", secret_blob) + self.index_entry(
            "report.txt", clean_blob
        )
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        snapshot = Path(temporary_directory.name) / "snapshot"
        snapshot.mkdir()

        with self.assertRaisesRegex(gitleaks_staged.ScanError, "case folding"):
            self.export_with_index_output(index_output, snapshot)

        self.assertEqual(list(snapshot.iterdir()), [])
        self.assertEqual(len(self.export_commands), 1)

    def test_export_rejects_staged_symlinks(self):
        (self.worktree / "link.txt").symlink_to("target.txt")
        self.run_git(self.worktree, "add", "link.txt")
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        snapshot = Path(temporary_directory.name) / "snapshot"
        snapshot.mkdir()

        with self.assertRaisesRegex(gitleaks_staged.ScanError, "regular files"):
            gitleaks_staged.export_staged_snapshot(self.worktree, snapshot)


if __name__ == "__main__":
    unittest.main()
