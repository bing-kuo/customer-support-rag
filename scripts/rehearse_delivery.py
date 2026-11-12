"""Run a temporary, mock-only knowledge lifecycle rehearsal."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from customer_support_rag import ask
from customer_support_rag.api import create_app
from customer_support_rag.knowledge.answering import GroundedDraft
from customer_support_rag.knowledge.embedding import EMBEDDING_MODEL
from customer_support_rag.luna_live import LiveCall
from customer_support_rag.maintenance import cli as maintenance_cli
from customer_support_rag.maintenance.hashes import describe_document
from customer_support_rag.maintenance.json_text import human_json
from customer_support_rag.maintenance.mock_embed import MOCK_EMBED_DIGEST, mock_vector
from customer_support_rag.maintenance.release import release_hashes

REPO = Path(__file__).resolve().parents[1]
BASELINE = ("hp-a/headphone-care.md", "hp-b/connection-troubleshooting.md")
SOURCE_SETS = {
    "baseline": BASELINE,
    "faq": ("drafts/hp-a/charging-faq.md", *BASELINE),
    "policy-v1": ("drafts/hp-a/return-policy-v1.md", *BASELINE),
    "policy-v2": ("drafts/hp-a/return-policy-v2.md", *BASELINE),
}
FAKE_EVIDENCE = "OFFLINE REHEARSAL FIXTURE ONLY; NOT A REVIEW OR RELEASE APPROVAL"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a temporary, mock-only knowledge lifecycle rehearsal."
    )
    parser.parse_args(argv)
    _block_network()
    with tempfile.TemporaryDirectory(prefix="customer-support-rag-rehearsal-") as tmp:
        root = Path(tmp)
        knowledge = root / "knowledge"
        workspace = root / "workspace"
        approvals = root / "fake-approvals"
        key_root = root / "fake-key-root"
        guard_root = root / "network-guard"
        shutil.copytree(REPO / "knowledge", knowledge)
        workspace.mkdir(mode=0o700)
        approvals.mkdir(mode=0o700)
        key_root.mkdir(mode=0o700)
        fake_env = key_root / ".env"
        fake_env.write_text("OPENAI_API_KEY=sk-fake-rehearsal-only\n", encoding="utf-8")
        fake_env.chmod(0o600)
        _write_child_guard(guard_root)

        _require_clean_head()
        _assert_local_startup()
        initial = _run("status", "--workspace", str(workspace), expected=1)
        _require(
            json.loads(initial.stdout)["active"] is None, "fresh workspace is empty"
        )
        serving = {
            "before_activation": _assert_release_serving(
                workspace, key_root, expected_release=None
            )
        }

        releases = _build_all(workspace, knowledge, approvals)
        _assert_case(workspace, releases["baseline"], "faq-absent-baseline")
        _assert_case(workspace, releases["faq"], "faq-present")
        _assert_case(workspace, releases["policy-v1"], "policy-v1-window")
        _assert_case(workspace, releases["policy-v2"], "policy-v2-window")
        for release_id in releases.values():
            _assert_generation_unverified(workspace, release_id)

        _consume_release_approval(workspace, approvals, releases["baseline"])
        _run(
            "activate",
            "--workspace",
            str(workspace),
            "--release",
            releases["baseline"],
            "--expect-none",
        )
        baseline_pointer = (workspace / "active.json").read_bytes()
        _require(
            _status(workspace)["active"] == releases["baseline"],
            "baseline should be active before candidate switching",
        )
        serving["baseline"] = _assert_release_serving(
            workspace, key_root, expected_release=releases["baseline"]
        )

        _consume_release_approval(workspace, approvals, releases["faq"])
        _activate_from_pointer(workspace, releases["faq"], baseline_pointer)
        _require(
            (workspace / "releases" / releases["baseline"]).is_dir(),
            "old baseline snapshot must remain after FAQ activation",
        )
        _require(
            _restart_status(guard_root, workspace)["active"] == releases["faq"],
            "a fresh CLI process must load the active FAQ snapshot",
        )
        serving["faq"] = _assert_release_serving(
            workspace, key_root, expected_release=releases["faq"]
        )

        _consume_release_approval(workspace, approvals, releases["policy-v1"])
        current_pointer = (workspace / "active.json").read_bytes()
        failure = _run(
            "activate",
            "--workspace",
            str(workspace),
            "--release",
            releases["policy-v1"],
            "--expect",
            "0" * 64,
            expected=1,
        )
        _require(
            bool(failure.stderr), "stale pointer compare should report a safe error"
        )
        _require(
            (workspace / "active.json").read_bytes() == current_pointer,
            "failed pointer compare must preserve the active FAQ snapshot",
        )
        current = _status(workspace)
        _require(
            current["active"] == releases["faq"], "status must confirm old snapshot"
        )
        _activate_from_pointer(workspace, releases["policy-v1"], current_pointer)
        _require(
            _status(workspace)["active"] == releases["policy-v1"],
            "status-confirmed pointer should allow the recovery switch",
        )
        serving["policy-v1"] = _assert_release_serving(
            workspace, key_root, expected_release=releases["policy-v1"]
        )
        _require(
            (workspace / "releases" / releases["faq"]).is_dir(),
            "previous FAQ snapshot must remain after recovery",
        )

        print(
            json.dumps(
                {
                    "result": "passed",
                    "scope": "temporary mock-only rehearsal",
                    "approvals": "fake fixtures only; not real approvals",
                    "releases": releases,
                    "outbound_connections": 0,
                    "ollama_embedding_calls": 0,
                    "mock_query_embedding_calls": sum(
                        item["mock_query_embedding_calls"] for item in serving.values()
                    ),
                    "luna_calls": 0,
                    "mock_generation_calls": sum(
                        item["mock_generation_calls"] for item in serving.values()
                    ),
                    "serving_release_checks": serving,
                    "key": "fake-only; never read by rehearsal",
                    "usage": None,
                    "active_after_recovery": releases["policy-v1"],
                    "previous_snapshots_retained": True,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0


def _build_all(
    workspace: Path,
    knowledge: Path,
    approvals: Path,
) -> dict[str, str]:
    code_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    releases: dict[str, str] = {}
    for name, sources in SOURCE_SETS.items():
        for source in sources:
            _consume_document_approval(workspace, knowledge, approvals, source)
        build_args = (
            "build",
            "--workspace",
            str(workspace),
            "--root",
            str(knowledge),
            *_source_args(sources),
            "--embed",
            "mock",
        )
        built = _json(_run(*build_args))
        release_id = str(built["release_id"])
        releases[name] = release_id
        noop = _json(_run(*build_args))
        _require(noop["noop"] is True and noop["embedded"] == 0, "build must reuse")
        _run(
            "evaluate",
            "--workspace",
            str(workspace),
            "--root",
            str(knowledge),
            "--release",
            release_id,
            "--suite",
            str(REPO / "eval" / "dev-suite.json"),
            "--code-sha",
            code_sha,
            "--embed",
            "mock",
        )
    return releases


def _consume_document_approval(
    workspace: Path, knowledge: Path, approvals: Path, source: str
) -> None:
    described = describe_document(knowledge, source)
    path = approvals / f"document-{described.file_sha256}.json"
    if not path.exists():
        path.write_text(
            human_json(
                {
                    "schema": "synthetic-document-approval",
                    "schema_version": 1,
                    "document_id": described.document_id,
                    "product_id": described.product_id,
                    "version": described.version,
                    "relative_path": described.relative_path,
                    "file_sha256": described.file_sha256,
                    "body_sha256": described.body_sha256,
                    "reviewer": "marcus",
                    "evidence": FAKE_EVIDENCE,
                }
            ),
            encoding="utf-8",
        )
    _run(
        "consume-document",
        "--workspace",
        str(workspace),
        "--root",
        str(knowledge),
        "--approval",
        str(path),
    )


def _consume_release_approval(
    workspace: Path, approvals: Path, release_id: str
) -> None:
    hashes = release_hashes(workspace, release_id)
    path = approvals / f"release-{release_id}.json"
    path.write_text(
        human_json(
            {
                "schema": "synthetic-release-approval",
                "schema_version": 1,
                "release_id": release_id,
                **hashes,
                "reviewer": "marcus",
                "evidence": FAKE_EVIDENCE,
            }
        ),
        encoding="utf-8",
    )
    _run("consume-release", "--workspace", str(workspace), "--approval", str(path))


def _evaluate_case(workspace: Path, release_id: str, case_id: str) -> dict[str, object]:
    report = json.loads(
        (workspace / "releases" / release_id / "regression.json").read_text(
            encoding="utf-8"
        )
    )
    return next(case for case in report["cases"] if case["id"] == case_id)


def _assert_case(workspace: Path, release_id: str, case_id: str) -> None:
    _require(
        _evaluate_case(workspace, release_id, case_id)["result"] == "pass",
        f"mock case failed: {case_id}",
    )


def _assert_generation_unverified(workspace: Path, release_id: str) -> None:
    report = json.loads(
        (workspace / "releases" / release_id / "regression.json").read_text(
            encoding="utf-8"
        )
    )
    generation = [item for item in report["cases"] if item["type"] == "generation"]
    _require(bool(generation), "evaluation must include generation cases")
    for case in generation:
        _require(case["result"] == "unverified", "generation stays unverified")
        _require(case["model_called"] is False, "generation must not call a model")
        _require(case.get("usage") is None, "unknown usage stays null")


def _activate_from_pointer(workspace: Path, release_id: str, pointer: bytes) -> None:
    _run(
        "activate",
        "--workspace",
        str(workspace),
        "--release",
        release_id,
        "--expect",
        hashlib.sha256(pointer).hexdigest(),
    )


def _status(workspace: Path) -> dict[str, object]:
    return _json(_run("status", "--workspace", str(workspace)))


def _restart_status(guard_root: Path, workspace: Path) -> dict[str, object]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        str(guard_root) + os.pathsep + environment.get("PYTHONPATH", "")
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "customer_support_rag.maintenance",
            "status",
            "--workspace",
            str(workspace),
        ],
        cwd=REPO,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    _require(result.returncode == 0, "fresh process status failed")
    return json.loads(result.stdout)


def _assert_local_startup() -> None:
    # The API deliberately rejects TestClient's default "testserver" Host.
    # Use a trusted loopback host so this exercises the same local boundary as
    # the documented localhost startup without opening a listening socket.
    with TestClient(create_app(None), base_url="http://localhost") as client:
        _require(client.get("/health").status_code == 200, "health endpoint failed")
        _require(client.get("/docs").status_code == 200, "API docs failed")
        _require(client.get("/").status_code == 200, "local page failed")


def _assert_release_serving(
    workspace: Path, key_root: Path, *, expected_release: str | None
) -> dict[str, object]:
    names = (
        "CUSTOMER_SUPPORT_RAG_RELEASE_ROOT",
        "CUSTOMER_SUPPORT_RAG_KEY_ROOT",
        "CUSTOMER_SUPPORT_RAG_DATABASE",
    )
    previous = {name: os.environ.get(name) for name in names}
    try:
        os.environ["CUSTOMER_SUPPORT_RAG_RELEASE_ROOT"] = str(workspace)
        os.environ["CUSTOMER_SUPPORT_RAG_KEY_ROOT"] = str(key_root)
        os.environ.pop("CUSTOMER_SUPPORT_RAG_DATABASE", None)
        config = ask.config_from_environment()
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    _require(config is not None, "release-root environment did not configure serving")
    _require(config.release_root == workspace, "serving did not select release root")
    status = _run(
        "status",
        "--workspace",
        str(workspace),
        expected=1 if expected_release is None else 0,
    )
    active = _json(status)["active"]
    _require(
        active == expected_release,
        "maintenance status and expected serving release differ",
    )
    query_calls = 0
    generation_calls = 0

    def embedding_fixture(request: httpx.Request) -> httpx.Response:
        nonlocal query_calls
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={
                    "models": [{"name": EMBEDDING_MODEL, "digest": MOCK_EMBED_DIGEST}]
                },
            )
        if request.url.path != "/api/embed":
            return httpx.Response(404)
        payload = json.loads(request.content.decode("utf-8"))
        texts = payload["input"]
        query_calls += len(texts)
        return httpx.Response(
            200,
            json={
                "model": EMBEDDING_MODEL,
                "embeddings": [mock_vector(text) for text in texts],
            },
        )

    config = replace(config, embedding_transport=httpx.MockTransport(embedding_fixture))

    async def fake_generation(_root: Path, request: object, **_kwargs: object):
        nonlocal generation_calls
        generation_calls += 1
        if not getattr(request, "sources", ()):
            raise RuntimeError("serving fixture expected an active candidate source")
        return LiveCall(
            draft=GroundedDraft(
                status="answer",
                answer="OFFLINE REHEARSAL MOCK ANSWER; NOT MODEL OUTPUT",
                reason=None,
                sources=(request.sources[0],),
            ),
            usage=None,
            stopped=None,
            usage_code="offline_rehearsal_mock",
        )

    with patch.object(ask, "post_prepared_answer", fake_generation):
        with TestClient(create_app(config), base_url="http://localhost") as client:
            response = client.post(
                "/ask",
                json={
                    "product_id": "hp-a",
                    "question": "Synthetic serving release-root check",
                },
            )
            assist_start = client.post(
                "/assist",
                json={
                    "action": "start",
                    "question": "Synthetic assist release-root check",
                },
            )
            assist_view = assist_start.json()
            _require(
                assist_start.status_code == 200
                and assist_view.get("status") == "clarify"
                and assist_view.get("missing") == ["product_id", "steps_tried"],
                "assist should clarify required fields before retrieval",
            )
            _require(
                query_calls == (0 if expected_release is None else 1)
                and generation_calls == (0 if expected_release is None else 1),
                "incomplete assist must not call serving adapters",
            )
            assist_answer = client.post(
                "/assist",
                json={
                    "action": "continue",
                    "slot_token": assist_view["slot_token"],
                    "product_id": "hp-a",
                    "steps_tried": "Synthetic troubleshooting step",
                },
            )
    if expected_release is None:
        _require(
            response.status_code == 503
            and response.json().get("code") == "release_unavailable",
            "serving should report an unconfigured active release before activation",
        )
        _require(
            query_calls == 0 and generation_calls == 0, "inactive root called adapters"
        )
        _require(
            assist_answer.status_code == 503
            and assist_answer.json().get("code") == "release_unavailable",
            "completed assist should report an inactive release",
        )
    else:
        _require(
            response.status_code == 200, "active release did not serve the mock query"
        )
        payload = response.json()
        _require(
            payload.get("answer") == "OFFLINE REHEARSAL MOCK ANSWER; NOT MODEL OUTPUT"
            and bool(payload.get("sources")),
            "serving did not return evidence from the active snapshot",
        )
        _require(
            assist_answer.status_code == 200
            and assist_answer.json().get("answer")
            == "OFFLINE REHEARSAL MOCK ANSWER; NOT MODEL OUTPUT"
            and bool(assist_answer.json().get("sources")),
            "assist did not continue to active release serving",
        )
        _require(
            query_calls == 2 and generation_calls == 2,
            "ask/assist mock serving call count changed",
        )
    return {
        "active_release": expected_release,
        "mock_query_embedding_calls": query_calls,
        "mock_generation_calls": generation_calls,
    }


def _require_clean_head() -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO, capture_output=True, check=True
    )
    _require(not status.stdout, "rehearsal requires a clean committed worktree")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    _require(re.fullmatch(r"[0-9a-f]{40}", head) is not None, "invalid Git HEAD")


def _run(*arguments: str, expected: int = 0) -> subprocess.CompletedProcess[str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        result_code = maintenance_cli.main(list(arguments))
    result = subprocess.CompletedProcess(
        arguments, result_code, stdout.getvalue(), stderr.getvalue()
    )
    _require(result_code == expected, f"maintenance command failed: {arguments[0]}")
    return result


def _json(result: subprocess.CompletedProcess[str]) -> dict[str, object]:
    payload = json.loads(result.stdout)
    _require(isinstance(payload, dict), "maintenance CLI must return a JSON object")
    return payload


def _source_args(sources: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(item for source in sources for item in ("--source", source))


def _write_child_guard(directory: Path) -> None:
    directory.mkdir(mode=0o700)
    (directory / "sitecustomize.py").write_text(
        "import socket\n"
        "def reject(*args, **kwargs):\n"
        "    raise RuntimeError('offline rehearsal forbids outbound connections')\n"
        "socket.socket.connect = reject\n"
        "socket.socket.connect_ex = reject\n"
        "socket.create_connection = reject\n",
        encoding="utf-8",
    )


def _block_network() -> None:
    def reject(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("offline rehearsal forbids outbound connections")

    socket.socket.connect = reject  # type: ignore[method-assign]
    socket.socket.connect_ex = reject  # type: ignore[method-assign]
    socket.create_connection = reject  # type: ignore[assignment]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


if __name__ == "__main__":
    raise SystemExit(main())
