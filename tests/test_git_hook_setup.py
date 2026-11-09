import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
HOOK = ROOT / ".githooks" / "pre-commit"
SCANNER = ROOT / "scripts" / "gitleaks_staged.py"
INSTALLER = ROOT / "scripts" / "install_git_hooks.py"
SYNTHETIC_MARKER = "SYNTHETIC_" + "PRIVATE_MARKER"


class GitHookSetupTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.repository = self.root / "repository"
        self.repository.mkdir()
        self.global_config = self.root / "global-config"
        self.global_config.write_text("")
        self.environment = os.environ.copy()
        self.environment.update(
            GIT_CONFIG_GLOBAL=str(self.global_config),
            GIT_CONFIG_NOSYSTEM="1",
        )
        self.run_git(self.repository, "init")
        self.run_git(self.repository, "config", "user.name", "Synthetic Test")
        self.run_git(self.repository, "config", "user.email", "test@example.com")
        (self.repository / ".githooks").mkdir()
        (self.repository / "scripts").mkdir()
        shutil.copy2(HOOK, self.repository / ".githooks" / "pre-commit")
        shutil.copy2(SCANNER, self.repository / "scripts" / "gitleaks_staged.py")
        shutil.copy2(INSTALLER, self.repository / "scripts" / "install_git_hooks.py")
        (self.repository / "README.md").write_text("Synthetic clone\n")
        self.run_git(self.repository, "add", ".")
        self.run_git(self.repository, "commit", "-m", "synthetic base")

    def tearDown(self):
        self.temporary_directory.cleanup()

    def run_git(self, repository: Path, *arguments: str, env=None, check=True):
        return subprocess.run(
            ["git", "-C", str(repository), *arguments],
            env=env or self.environment,
            check=check,
            capture_output=True,
            text=True,
        )

    def run_installer(self, repository: Path, env=None):
        return subprocess.run(
            [sys.executable, "scripts/install_git_hooks.py"],
            cwd=repository,
            env=env or self.environment,
            capture_output=True,
            text=True,
        )

    def make_fake_docker(self) -> tuple[Path, Path]:
        binary_directory = self.root / "fake-bin"
        binary_directory.mkdir()
        docker = binary_directory / "docker"
        docker.write_text(
            "#!/usr/bin/env python3\n"
            "import os\n"
            "from pathlib import Path\n"
            "import sys\n"
            "arguments = sys.argv[1:]\n"
            "snapshot = Path(arguments[arguments.index('-v') + 1].split(':', 1)[0])\n"
            "content = b''.join(\n"
            "    path.read_bytes()\n"
            "    for path in snapshot.rglob('*')\n"
            "    if path.is_file()\n"
            ")\n"
            "with Path(os.environ['SYNTHETIC_SCAN_LOG']).open('a') as log:\n"
            "    log.write('scan\\n')\n"
            "marker = os.environ['SYNTHETIC_SCAN_MARKER'].encode()\n"
            "sys.exit(1 if marker in content else 0)\n"
        )
        docker.chmod(0o755)
        return binary_directory, self.root / "scan.log"

    def test_setup_enables_normal_commit_and_fails_closed(self):
        setup = self.run_installer(self.repository)
        self.assertEqual(setup.returncode, 0, setup.stderr)
        self.assertIn("local Git configuration", setup.stdout)

        repeated_setup = self.run_installer(self.repository)
        self.assertEqual(repeated_setup.returncode, 0, repeated_setup.stderr)
        configured_path = self.run_git(
            self.repository, "config", "--local", "--get-all", "core.hooksPath"
        ).stdout.splitlines()
        self.assertEqual(configured_path, [".githooks"])

        binary_directory, scan_log = self.make_fake_docker()
        environment = self.environment.copy()
        environment.update(
            PATH=str(binary_directory) + os.pathsep + environment.get("PATH", ""),
            SYNTHETIC_SCAN_LOG=str(scan_log),
            SYNTHETIC_SCAN_MARKER=SYNTHETIC_MARKER,
        )

        credential = self.repository / "credential.txt"
        credential.write_text("clean staged content\n")
        self.run_git(self.repository, "add", "credential.txt")
        clean_commit = self.run_git(
            self.repository, "commit", "-m", "clean staged content", env=environment
        )
        self.assertEqual(clean_commit.returncode, 0)

        previous_head = self.run_git(
            self.repository, "rev-parse", "HEAD"
        ).stdout.strip()
        credential.write_text(SYNTHETIC_MARKER + "\n")
        self.run_git(self.repository, "add", "credential.txt")
        rejected_commit = self.run_git(
            self.repository,
            "commit",
            "-m",
            "synthetic staged secret",
            env=environment,
            check=False,
        )
        self.assertNotEqual(rejected_commit.returncode, 0)
        self.assertEqual(
            self.run_git(self.repository, "rev-parse", "HEAD").stdout.strip(),
            previous_head,
        )
        self.assertEqual(scan_log.read_text().splitlines(), ["scan", "scan"])

    def test_setup_preserves_conflicting_hooks_path(self):
        self.run_git(
            self.repository,
            "config",
            "--local",
            "core.hooksPath",
            ".custom-hooks",
        )

        setup = self.run_installer(self.repository)

        self.assertNotEqual(setup.returncode, 0)
        self.assertIn("was preserved", setup.stderr)
        self.assertEqual(
            self.run_git(
                self.repository, "config", "--local", "--get", "core.hooksPath"
            ).stdout.strip(),
            ".custom-hooks",
        )

    def test_setup_preserves_default_custom_hook(self):
        custom_hook = self.repository / ".git" / "hooks" / "pre-commit"
        custom_hook.write_text("#!/bin/sh\nexit 0\n")
        custom_hook.chmod(0o755)

        setup = self.run_installer(self.repository)

        self.assertNotEqual(setup.returncode, 0)
        self.assertIn("Existing Git hooks were preserved", setup.stderr)
        self.assertEqual(custom_hook.read_text(), "#!/bin/sh\nexit 0\n")
        self.assertEqual(
            self.run_git(
                self.repository,
                "config",
                "--local",
                "--get",
                "core.hooksPath",
                check=False,
            ).returncode,
            1,
        )

    def test_setup_refuses_to_change_shared_config_from_linked_worktree(self):
        linked_worktree = self.root / "linked-worktree"
        self.run_git(
            self.repository,
            "worktree",
            "add",
            "-b",
            "synthetic-linked-worktree",
            str(linked_worktree),
            "HEAD",
        )

        setup = self.run_installer(linked_worktree)

        self.assertNotEqual(setup.returncode, 0)
        self.assertIn("linked worktree", setup.stderr)
        self.assertEqual(
            self.run_git(
                self.repository,
                "config",
                "--local",
                "--get",
                "core.hooksPath",
                check=False,
            ).returncode,
            1,
        )

    def test_setup_preserves_global_hook_path(self):
        self.global_config.write_text("[core]\n\thooksPath = /custom/global-hooks\n")

        setup = self.run_installer(self.repository)

        self.assertNotEqual(setup.returncode, 0)
        self.assertIn("Existing core.hooksPath setting was preserved", setup.stderr)
        self.assertIn("/custom/global-hooks", self.global_config.read_text())
        self.assertEqual(
            self.run_git(
                self.repository,
                "config",
                "--local",
                "--get",
                "core.hooksPath",
                check=False,
            ).returncode,
            1,
        )


if __name__ == "__main__":
    unittest.main()
