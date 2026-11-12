"""Mock maintenance lifecycle. Approvals here are temporary fixtures only."""

from __future__ import annotations

import asyncio
import json
import socket
from pathlib import Path

import httpx
import pytest

from customer_support_rag.api import create_app
from customer_support_rag.ask import (
    PUBLISHED_DIGEST,
    AskConfig,
    AskFailure,
    config_from_environment,
)
from customer_support_rag.knowledge import KnowledgeChunk, KnowledgeMetadata
from customer_support_rag.knowledge.storage import EmbeddingStore
from customer_support_rag.luna import MODEL
from customer_support_rag.maintenance.approval import consume_document_approval
from customer_support_rag.maintenance.baseline import baseline_order, bigram_overlap
from customer_support_rag.maintenance.cli import main
from customer_support_rag.maintenance.errors import MaintenanceError
from customer_support_rag.maintenance.evaluate import (
    evaluate_release,
    require_mock_regression,
)
from customer_support_rag.maintenance.hashes import (
    describe_document,
    scan_current_sources,
    sha256_file,
)
from customer_support_rag.maintenance.json_text import human_json, load_object
from customer_support_rag.maintenance.mock_embed import MOCK_EMBED_DIGEST, MockEmbedder
from customer_support_rag.maintenance.release import (
    activate,
    build_candidate,
    candidate_snapshot,
    release_hashes,
)

ROOT = Path(__file__).parents[1]
KNOWLEDGE = ROOT / "knowledge"
SUITE = ROOT / "eval" / "dev-suite.json"
CODE = "ab" * 20
CARE_BODY = "223243a3ba9c40840746b1b2ee66e40b970a8fefd1324bb586930a381f9f4e59"
BASELINE = (
    "hp-a/headphone-care.md",
    "hp-b/connection-troubleshooting.md",
)
FAQ = ("drafts/hp-a/charging-faq.md", *BASELINE)
POLICY_V1 = ("drafts/hp-a/return-policy-v1.md", *BASELINE)
POLICY_V2 = ("drafts/hp-a/return-policy-v2.md", *BASELINE)
KEY = "synthetic-key-not-live"


@pytest.fixture(autouse=True)
def _block_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    real_socket = socket.socket

    def guarded(family: int = socket.AF_INET, *args: object, **kwargs: object):
        if family != socket.AF_UNIX:
            raise AssertionError("unexpected socket")
        return real_socket(family, *args, **kwargs)

    monkeypatch.setattr(socket, "socket", guarded)
    monkeypatch.setattr(
        socket, "create_connection", lambda *_args, **_kwargs: guarded(-1)
    )

    def budget(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("budget")

    monkeypatch.setattr("customer_support_rag.budget.LunaPolicy", budget)
    monkeypatch.setattr("customer_support_rag.budget.Ledger", budget)
    monkeypatch.setattr("customer_support_rag.budget.open_ledger", budget)
    monkeypatch.setattr("customer_support_rag.budget.load_policy", budget)
    monkeypatch.setattr("customer_support_rag.budget.load_runtime", budget)
    monkeypatch.setattr("customer_support_rag.budget.init_synthetic_ledger", budget)
    monkeypatch.setattr("customer_support_rag.luna.generate_draft", budget)


def test_file_hash_differs_from_body_and_scan_skips_drafts() -> None:
    care = describe_document(KNOWLEDGE, "hp-a/headphone-care.md")
    assert care.file_sha256 != care.body_sha256
    assert care.body_sha256 == CARE_BODY
    faq = describe_document(KNOWLEDGE, "drafts/hp-a/charging-faq.md")
    policy_v1 = describe_document(KNOWLEDGE, "drafts/hp-a/return-policy-v1.md")
    policy_v2 = describe_document(KNOWLEDGE, "drafts/hp-a/return-policy-v2.md")
    fixture = describe_document(KNOWLEDGE, "fixtures/injection-note.md")
    assert len({faq.file_sha256, policy_v1.file_sha256, policy_v2.file_sha256}) == 3
    assert policy_v1.document_id == policy_v2.document_id == "hp-a-return-policy"
    assert policy_v1.version == "1"
    assert policy_v2.version == "2"
    assert fixture.document_id == "fixture-injection-note"
    scanned = scan_current_sources(KNOWLEDGE)
    assert [item.relative_path for item in scanned] == list(BASELINE)
    assert faq.relative_path not in {item.relative_path for item in scanned}


def test_consume_rejects_changed_bytes_wrong_reviewer_and_extra_fields(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    approval = tmp_path / "care.json"
    _write_document_approval(KNOWLEDGE, BASELINE[0], approval)
    stored = consume_document_approval(workspace, KNOWLEDGE, approval)
    assert consume_document_approval(workspace, KNOWLEDGE, approval) == stored
    changed = tmp_path / "changed.json"
    payload = load_object(approval, approval.read_bytes())
    payload["body_sha256"] = "cd" * 32
    changed.write_text(human_json(payload), encoding="utf-8")
    with pytest.raises(MaintenanceError, match="does not match"):
        consume_document_approval(workspace, KNOWLEDGE, changed)
    reviewer = tmp_path / "ethan.json"
    payload = load_object(approval, approval.read_bytes())
    payload["reviewer"] = "ethan"
    reviewer.write_text(human_json(payload), encoding="utf-8")
    with pytest.raises(MaintenanceError, match="not marcus"):
        consume_document_approval(workspace, KNOWLEDGE, reviewer)
    extra = tmp_path / "extra.json"
    payload = load_object(approval, approval.read_bytes())
    payload["note"] = "extra"
    extra.write_text(human_json(payload), encoding="utf-8")
    with pytest.raises(MaintenanceError, match="invalid"):
        consume_document_approval(workspace, KNOWLEDGE, extra)
    other = tmp_path / "other-evidence.json"
    _write_document_approval(KNOWLEDGE, BASELINE[0], other, evidence="另一份夾具")
    with pytest.raises(MaintenanceError, match="already exists"):
        consume_document_approval(workspace, KNOWLEDGE, other)


def test_cli_has_no_approve_command(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["approve"])
    assert caught.value.code == 2
    assert "approve" in capsys.readouterr().err
    listed = main(["hashes", "--root", str(KNOWLEDGE), "--path", BASELINE[0]])
    assert listed == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["file_sha256"] != printed["body_sha256"]
    assert main(["status", "--workspace", str(KNOWLEDGE)]) == 1


def test_build_does_not_activate_and_rebuild_embeds_nothing(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    _approve(workspace, BASELINE)
    first = build_candidate(workspace, KNOWLEDGE, BASELINE)
    assert first.noop is False
    assert first.embedded == 2
    assert first.copied == 0
    assert not (workspace / "active.json").exists()
    embedder = MockEmbedder()
    again = build_candidate(workspace, KNOWLEDGE, BASELINE, embedder)
    assert again.noop is True
    assert again.release_id == first.release_id
    assert again.embedded == 0
    assert embedder.calls == 0
    reversed_sources = tuple(reversed(BASELINE))
    same = build_candidate(workspace, KNOWLEDGE, reversed_sources, embedder)
    assert same.release_id == first.release_id
    assert embedder.calls == 0


def test_faq_build_copies_baseline_documents(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    _approve(workspace, FAQ)
    baseline = build_candidate(workspace, KNOWLEDGE, BASELINE)
    embedder = MockEmbedder()
    faq = build_candidate(workspace, KNOWLEDGE, FAQ, embedder)
    assert faq.release_id != baseline.release_id
    assert faq.copied == 2
    assert faq.embedded == 1
    assert embedder.calls == 1
    documents = _documents(workspace, faq.release_id)
    assert {item["document_id"] for item in documents} == {
        "hp-a-care",
        "hp-b-connection",
        "hp-a-charging-faq",
    }
    assert all(len(item["artifact_key"]) == 64 for item in documents)


def test_conflicting_policy_versions_are_rejected(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    _approve(workspace, (*POLICY_V1, POLICY_V2[0]))
    with pytest.raises(MaintenanceError, match="document version conflict"):
        build_candidate(
            workspace,
            KNOWLEDGE,
            ("drafts/hp-a/return-policy-v1.md", "drafts/hp-a/return-policy-v2.md"),
        )
    assert not (workspace / "active.json").exists()
    assert list((workspace / "releases").glob(".partial-*")) == []
    first = build_candidate(workspace, KNOWLEDGE, POLICY_V1)
    second_workspace = _workspace(tmp_path / "v2")
    _approve(second_workspace, POLICY_V2)
    second = build_candidate(second_workspace, KNOWLEDGE, POLICY_V2)
    v1 = _policy_version(workspace, first.release_id)
    v2 = _policy_version(second_workspace, second.release_id)
    assert v1 == "1"
    assert v2 == "2"


def test_traversal_and_symlink_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(MaintenanceError, match="path is invalid"):
        describe_document(KNOWLEDGE, "../README.md")
    with pytest.raises(MaintenanceError, match="path is invalid"):
        describe_document(KNOWLEDGE, "/etc/passwd")
    root = tmp_path / "tree"
    nested = root / "hp-a"
    nested.mkdir(parents=True)
    outside = tmp_path / "outside.md"
    outside.write_text("outside\n", encoding="utf-8")
    (nested / "linked.md").symlink_to(outside)
    with pytest.raises(MaintenanceError, match="symlink"):
        describe_document(root, "hp-a/linked.md")


def test_failed_embed_removes_the_partial_directory(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    _approve(workspace, BASELINE)

    class Boom:
        def embed_document(self, _document: object) -> list[object]:
            raise RuntimeError("boom")

    with pytest.raises(MaintenanceError, match="embedding failed"):
        build_candidate(workspace, KNOWLEDGE, BASELINE, Boom())
    assert list((workspace / "releases").glob(".partial-*")) == []
    assert not (workspace / "active.json").exists()


def test_tamper_and_failed_regression_do_not_activate(tmp_path: Path) -> None:
    missing = _workspace(tmp_path / "missing")
    release_id = _build_evaluated(missing, BASELINE)
    with pytest.raises(MaintenanceError, match="unavailable"):
        activate(missing, release_id, expected_previous=None)
    assert _pointer_bytes(missing) is None

    tampered = _workspace(tmp_path / "tampered")
    tampered_id = _build_evaluated(tampered, BASELINE)
    _approve_release(tmp_path / "ok", tampered, tampered_id)
    database = tampered / "releases" / tampered_id / "knowledge.sqlite"
    original = database.read_bytes()
    database.write_bytes(original + b" ")
    with pytest.raises(MaintenanceError, match="does not match"):
        activate(tampered, tampered_id, expected_previous=None)
    assert _pointer_bytes(tampered) is None

    generated = _workspace(tmp_path / "generated")
    generated_id = _build_evaluated(generated, BASELINE)
    report_path = generated / "releases" / generated_id / "regression.json"
    report = load_object(report_path, report_path.read_bytes())
    cases = report["cases"]
    assert isinstance(cases, list)
    generation = next(item for item in cases if item["type"] == "generation")
    generation["result"] = "pass"
    report_path.write_text(human_json(report), encoding="utf-8")
    _approve_release(tmp_path / "bad", generated, generated_id)
    with pytest.raises(MaintenanceError, match="unverified"):
        activate(generated, generated_id, expected_previous=None)
    assert _pointer_bytes(generated) is None

    manifest = missing / "releases" / release_id / "manifest.json"
    payload = load_object(manifest, manifest.read_bytes())
    database_record = payload["database"]
    assert isinstance(database_record, dict)
    database_record["sha256"] = "ab" * 32
    manifest.write_text(human_json(payload), encoding="utf-8")
    with pytest.raises(MaintenanceError, match="unavailable"):
        candidate_snapshot(missing, release_id)


def test_expected_previous_mismatch_keeps_the_pointer(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    baseline = _activate_sources(tmp_path / "base", workspace, BASELINE)
    current = sha256_file(workspace / "active.json")
    faq = _build_evaluated(workspace, FAQ)
    _approve_release(tmp_path / "faq", workspace, faq)
    with pytest.raises(MaintenanceError, match="does not match"):
        activate(workspace, faq, expected_previous="0" * 64)
    assert sha256_file(workspace / "active.json") == current
    assert load_active(workspace) == baseline
    assert activate(workspace, faq, expected_previous=current) == "activated"
    moved = sha256_file(workspace / "active.json")
    assert activate(workspace, faq, expected_previous=moved) == "noop"
    with pytest.raises(MaintenanceError, match="does not match"):
        activate(workspace, faq, expected_previous=None)
    assert sha256_file(workspace / "active.json") == moved


def test_evaluate_writes_cases_and_refuses_an_empty_run(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    release_id = _build_evaluated(workspace, BASELINE)
    report_path = workspace / "releases" / release_id / "regression.json"
    report = load_object(report_path, report_path.read_bytes())
    cases = report["cases"]
    assert isinstance(cases, list)
    assert cases
    assert {item["result"] for item in cases if item["type"] == "generation"} == {
        "unverified"
    }
    assert all(item["model_called"] is False for item in cases)
    assert report["mock_is_not_generation_evidence"] is True
    with pytest.raises(MaintenanceError, match="no cases"):
        require_mock_regression({**report, "cases": []})
    suite = json.loads(SUITE.read_text(encoding="utf-8"))
    for case in suite["cases"]:
        if case["id"] == "normal-care":
            case["expect_document_id"] = "missing-doc"
    broken = tmp_path / "broken-suite.json"
    broken.write_text(json.dumps(suite), encoding="utf-8")
    other = _workspace(tmp_path / "other")
    _approve(other, BASELINE)
    built = build_candidate(other, KNOWLEDGE, BASELINE)
    with pytest.raises(MaintenanceError, match="failing case"):
        evaluate_release(other, KNOWLEDGE, built.release_id, broken, code_sha=CODE)
    assert not (other / "releases" / built.release_id / "regression.json").exists()


def test_policy_releases_record_their_own_version(tmp_path: Path) -> None:
    for sources, version in ((POLICY_V1, "1"), (POLICY_V2, "2")):
        workspace = _workspace(tmp_path / version)
        release_id = _build_evaluated(workspace, sources)
        assert _policy_version(workspace, release_id) == version
        report = workspace / "releases" / release_id / "regression.json"
        assert report.is_file()


def test_bad_release_root_does_not_open_the_legacy_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path)
    legacy = tmp_path / "legacy.sqlite"
    legacy.write_bytes(b"legacy")
    opened: list[object] = []

    def open_readonly(path: object) -> None:
        opened.append(path)
        raise AssertionError("opened")

    monkeypatch.setattr(EmbeddingStore, "open_readonly", open_readonly)
    config = AskConfig(
        database=legacy,
        artifact_keys=(PUBLISHED_DIGEST,),
        model_digest=PUBLISHED_DIGEST,
        key_root=tmp_path,
        release_root=workspace,
    )
    with pytest.raises(AskFailure) as caught:
        asyncio.run(answer(config))
    assert caught.value.status_code == 503
    assert caught.value.code == "release_unavailable"
    assert opened == []


def test_release_root_env_does_not_require_the_legacy_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CUSTOMER_SUPPORT_RAG_DATABASE", raising=False)
    monkeypatch.delenv("CUSTOMER_SUPPORT_RAG_KEY_ROOT", raising=False)
    monkeypatch.delenv("CUSTOMER_SUPPORT_RAG_RELEASE_ROOT", raising=False)
    monkeypatch.setenv("CUSTOMER_SUPPORT_RAG_RELEASE_ROOT", str(tmp_path))
    assert config_from_environment() is None
    monkeypatch.setenv("CUSTOMER_SUPPORT_RAG_KEY_ROOT", str(tmp_path))
    monkeypatch.setenv("CUSTOMER_SUPPORT_RAG_DATABASE", str(tmp_path / "legacy.sqlite"))
    config = config_from_environment()
    assert config is not None
    assert config.release_root == tmp_path
    assert config.artifact_keys == ()
    assert config.model_digest == PUBLISHED_DIGEST


def test_bigram_baseline_breaks_ties_by_key_then_index() -> None:
    assert bigram_overlap("abcd", "ab") == 0.5
    assert bigram_overlap("a b", "ab") == 1.0
    metadata = KnowledgeMetadata(
        document_id="hp-a-care",
        product_id="hp-a",
        title="保養",
        version="1",
        synthetic=True,
    )

    def chunk(index: int) -> KnowledgeChunk:
        return KnowledgeChunk(
            metadata=metadata,
            body_sha256="cd" * 32,
            index=index,
            char_start=0,
            char_end=2,
            text="充電",
            heading_path=(),
        )

    low = "a" * 64
    high = "b" * 64
    ordered = baseline_order(
        "充電",
        [(high, chunk(0)), (low, chunk(1)), (low, chunk(0))],
        top_k=4,
    )
    assert [(key, item.index) for key, item, _score in ordered] == [
        (low, 0),
        (low, 1),
        (high, 0),
    ]


def test_vertical_mock_ask_cites_faq_only_after_that_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path)
    question = "指示燈閃爍時還能繼續充電嗎？"
    evidence: list[list[dict[str, object]]] = []

    async def luna(_self: object, request: httpx.Request) -> httpx.Response:
        data = _model_data(request)
        rows = data["evidence"]
        assert isinstance(rows, list)
        evidence.append(rows)
        source_ids = [item["source_id"] for item in rows]
        return httpx.Response(200, json=_envelope(_draft("合成草稿。", source_ids)))

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", luna)
    baseline = _activate_sources(tmp_path / "base-approval", workspace, BASELINE)
    client = _ask_client(tmp_path, workspace)
    before = client.post("/ask", json={"product_id": "hp-a", "question": question})
    assert before.status_code == 200
    assert "hp-a-charging-faq" not in {
        item["document_id"] for item in before.json()["sources"]
    }
    faq = _build_evaluated(workspace, FAQ)
    _approve_release(tmp_path / "faq-approval", workspace, faq)
    assert (
        activate(workspace, faq, expected_previous=_pointer_sha(workspace))
        == "activated"
    )
    assert load_active(workspace) != baseline
    after = client.post("/ask", json={"product_id": "hp-a", "question": question})
    assert after.status_code == 200
    body = after.json()
    assert body["status"] == "answer"
    assert "hp-a-charging-faq" in {item["document_id"] for item in body["sources"]}
    assert any("停止充電" in str(item.get("text")) for item in evidence[-1])
    assert MOCK_EMBED_DIGEST != PUBLISHED_DIGEST


def _workspace(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _approve(workspace: Path, sources: tuple[str, ...]) -> None:
    for source in sources:
        approval = workspace / f".approval-{source.replace('/', '-')}"
        _write_document_approval(KNOWLEDGE, source, approval)
        consume_document_approval(workspace, KNOWLEDGE, approval)


def _write_document_approval(
    root: Path, relative: str, destination: Path, *, evidence: str = "暫存夾具"
) -> None:
    described = describe_document(root, relative)
    payload = {
        "schema": "synthetic-document-approval",
        "schema_version": 1,
        "document_id": described.document_id,
        "product_id": described.product_id,
        "version": described.version,
        "relative_path": described.relative_path,
        "file_sha256": described.file_sha256,
        "body_sha256": described.body_sha256,
        "reviewer": "marcus",
        "evidence": evidence,
    }
    destination.write_text(human_json(payload), encoding="utf-8")


def _build_evaluated(workspace: Path, sources: tuple[str, ...]) -> str:
    _approve(workspace, sources)
    built = build_candidate(workspace, KNOWLEDGE, sources)
    evaluate_release(workspace, KNOWLEDGE, built.release_id, SUITE, code_sha=CODE)
    return built.release_id


def _activate_sources(
    approval_root: Path, workspace: Path, sources: tuple[str, ...]
) -> str:
    release_id = _build_evaluated(workspace, sources)
    _approve_release(approval_root, workspace, release_id)
    assert activate(workspace, release_id, expected_previous=None) == "activated"
    return release_id


def _approve_release(
    approval_root: Path, workspace: Path, release_id: str, *, name: str = "release.json"
) -> None:
    approval_root.mkdir(parents=True, exist_ok=True)
    hashes = release_hashes(workspace, release_id)
    payload = {
        "schema": "synthetic-release-approval",
        "schema_version": 1,
        "release_id": release_id,
        "manifest_sha256": hashes["manifest_sha256"],
        "database_sha256": hashes["database_sha256"],
        "regression_sha256": hashes["regression_sha256"],
        "code_sha": hashes["code_sha"],
        "suite_version": hashes["suite_version"],
        "reviewer": "marcus",
        "evidence": "暫存夾具，不是核准",
    }
    path = approval_root / name
    path.write_text(human_json(payload), encoding="utf-8")
    from customer_support_rag.maintenance.approval import consume_release_approval

    consume_release_approval(workspace, path)


def _documents(workspace: Path, release_id: str) -> list[dict[str, object]]:
    manifest = workspace / "releases" / release_id / "manifest.json"
    payload = load_object(manifest, manifest.read_bytes())
    documents = payload["documents"]
    assert isinstance(documents, list)
    return documents


def _policy_version(workspace: Path, release_id: str) -> str:
    matched = [
        item["version"]
        for item in _documents(workspace, release_id)
        if item["document_id"] == "hp-a-return-policy"
    ]
    assert matched == ["1"] or matched == ["2"]
    return str(matched[0])


def _pointer_bytes(workspace: Path) -> bytes | None:
    path = workspace / "active.json"
    if not path.exists():
        return None
    return path.read_bytes()


def _pointer_sha(workspace: Path) -> str:
    path = workspace / "active.json"
    assert path.is_file()
    return sha256_file(path)


def load_active(workspace: Path) -> str:
    from customer_support_rag.maintenance.release import load_active_snapshot

    return load_active_snapshot(workspace).release_id


async def answer(config: AskConfig):
    from customer_support_rag.ask import answer_ask

    return await answer_ask(config, "hp-a", "可以用水洗嗎？", "req-test")


def _ask_client(tmp_path: Path, workspace: Path):
    from fastapi.testclient import TestClient

    root = tmp_path / "key-root"
    root.mkdir(parents=True, exist_ok=True)
    env = root / ".env"
    env.write_text(f'OPENAI_API_KEY="{KEY}"\n', encoding="utf-8")
    env.chmod(0o600)
    from customer_support_rag.maintenance.mock_embed import mock_transport

    config = AskConfig(
        database=tmp_path / "unused.sqlite",
        artifact_keys=(),
        model_digest=PUBLISHED_DIGEST,
        key_root=root,
        embedding_transport=mock_transport(),
        release_root=workspace,
    )
    return TestClient(create_app(config), base_url="http://127.0.0.1")


def _model_data(request: httpx.Request) -> dict[str, object]:
    posted = json.loads(request.content.decode())
    text = posted["input"][0]["content"][0]["text"]
    data = json.loads(text)
    assert isinstance(data, dict)
    assert posted["model"] == MODEL
    return data


def _draft(answer_text: str, source_ids: list[object]) -> str:
    return json.dumps(
        {
            "status": "answer",
            "answer": answer_text,
            "source_ids": source_ids,
            "reason": None,
        },
        ensure_ascii=False,
    )


def _envelope(text: str) -> dict[str, object]:
    return {
        "id": "resp_fixture",
        "model": MODEL,
        "status": "completed",
        "service_tier": "default",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
        "usage": {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
    }
