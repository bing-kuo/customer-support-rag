"""Mock maintenance lifecycle. Approvals here are temporary fixtures only."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sqlite3
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
from customer_support_rag.knowledge.embedding import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    MAX_EMBED_TEXT_CHARS,
    OLLAMA_EMBED_URL,
    PUBLISHED_MODEL_DIGEST,
)
from customer_support_rag.knowledge.storage import (
    CHUNKING_PROFILE,
    INPUT_PROFILE,
    EmbeddingStore,
)
from customer_support_rag.luna import MODEL
from customer_support_rag.maintenance.approval import consume_document_approval
from customer_support_rag.maintenance.baseline import baseline_order, bigram_overlap
from customer_support_rag.maintenance.cli import main
from customer_support_rag.maintenance.errors import MaintenanceError
from customer_support_rag.maintenance.evaluate import (
    _select_query_transport,
    evaluate_release,
    regression_matches_candidate,
    require_mock_regression,
)
from customer_support_rag.maintenance.hashes import (
    describe_document,
    scan_current_sources,
    sha256_file,
)
from customer_support_rag.maintenance.json_text import human_json, load_object
from customer_support_rag.maintenance.mock_embed import (
    MOCK_EMBED_DIGEST,
    MockEmbedder,
    OllamaEmbedder,
    mock_vector,
)
from customer_support_rag.maintenance.release import (
    BuildResult,
    activate,
    build_candidate,
    candidate_snapshot,
    embedding_identity,
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


def test_symlinks_write_zero_bytes_outside(tmp_path: Path) -> None:
    control = _workspace(tmp_path / "symlink-control")
    outside_control = _workspace(tmp_path / "outside-control")
    (control / "control").symlink_to(outside_control, target_is_directory=True)
    source = tmp_path / "doc-approval.json"
    _write_document_approval(KNOWLEDGE, BASELINE[0], source)
    with pytest.raises(MaintenanceError, match="symlink"):
        consume_document_approval(control, KNOWLEDGE, source)
    assert _regular_file_bytes(outside_control) == 0

    releases = _workspace(tmp_path / "symlink-releases")
    _approve(releases, BASELINE)
    outside_releases = _workspace(tmp_path / "outside-releases")
    (releases / "releases").symlink_to(outside_releases, target_is_directory=True)
    with pytest.raises(MaintenanceError, match="symlink"):
        build_candidate(releases, KNOWLEDGE, BASELINE)
    assert _regular_file_bytes(outside_releases) == 0
    assert (releases / "releases").is_symlink()

    nested = _workspace(tmp_path / "nested-parent")
    (nested / "control").mkdir()
    outside_nested = _workspace(tmp_path / "outside-nested")
    (nested / "control" / "document-approvals").symlink_to(
        outside_nested, target_is_directory=True
    )
    with pytest.raises(MaintenanceError, match="symlink"):
        consume_document_approval(nested, KNOWLEDGE, source)
    assert _regular_file_bytes(outside_nested) == 0


def test_transplanted_regression_cannot_authorize_another_candidate(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path / "transplant")
    baseline = _activate_sources(tmp_path / "base-review", workspace, BASELINE)
    before = _pointer_sha(workspace)
    faq = _build_evaluated(workspace, FAQ)
    old_report = workspace / "releases" / baseline / "regression.json"
    new_report = workspace / "releases" / faq / "regression.json"
    new_report.write_bytes(old_report.read_bytes())
    with pytest.raises(
        MaintenanceError, match="regression does not match the candidate"
    ):
        _approve_release(tmp_path / "new-review", workspace, faq)
    assert _pointer_sha(workspace) == before
    assert load_active(workspace) == baseline

    swapped = _workspace(tmp_path / "swapped")
    base = _activate_sources(tmp_path / "swap-base", swapped, BASELINE)
    faq_id = _build_evaluated(swapped, FAQ)
    _approve_release(tmp_path / "swap-faq", swapped, faq_id)
    current = _pointer_sha(swapped)
    (swapped / "releases" / faq_id / "regression.json").write_bytes(
        (swapped / "releases" / base / "regression.json").read_bytes()
    )
    with pytest.raises(
        MaintenanceError, match="regression does not match the candidate"
    ):
        activate(swapped, faq_id, expected_previous=current)
    assert _pointer_sha(swapped) == current
    assert load_active(swapped) == base


def test_damaged_vectors_are_embedded_again(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "damaged")
    _approve(workspace, FAQ)
    baseline = build_candidate(workspace, KNOWLEDGE, BASELINE)
    selected = _tamper_vectors(workspace / "releases" / baseline.release_id)
    with pytest.raises(MaintenanceError):
        candidate_snapshot(workspace, baseline.release_id)
    built = build_candidate(workspace, KNOWLEDGE, FAQ)
    assert built.copied == 0
    assert built.embedded == 3
    _assert_mock_vectors(workspace, built.release_id)
    assert not _vector_is_tampered(workspace, built.release_id, selected)

    laundered = _workspace(tmp_path / "laundered")
    _approve(laundered, FAQ)
    source = build_candidate(laundered, KNOWLEDGE, BASELINE)
    _tamper_vectors(laundered / "releases" / source.release_id)
    directory = laundered / "releases" / source.release_id
    manifest_path = directory / "manifest.json"
    manifest = load_object(manifest_path, manifest_path.read_bytes())
    database_record = manifest["database"]
    assert isinstance(database_record, dict)
    database_record["sha256"] = sha256_file(directory / "knowledge.sqlite")
    manifest_path.write_text(human_json(manifest), encoding="utf-8")
    candidate_snapshot(laundered, source.release_id)
    copied = build_candidate(laundered, KNOWLEDGE, FAQ)
    assert copied.copied == 0
    assert copied.embedded == 3
    _assert_mock_vectors(laundered, copied.release_id)


def test_pointer_fsync_failure_stays_unconfirmed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace = _workspace(tmp_path)
    baseline = _activate_sources(tmp_path / "base", workspace, BASELINE)
    faq = _build_evaluated(workspace, FAQ)
    _approve_release(tmp_path / "faq", workspace, faq)
    before = _pointer_sha(workspace)

    def boom(_path: Path) -> None:
        raise MaintenanceError("injected directory fsync failure")

    monkeypatch.setattr(
        "customer_support_rag.maintenance.release._fsync_directory", boom
    )
    code = main(
        [
            "activate",
            "--workspace",
            str(workspace),
            "--release",
            faq,
            "--expect",
            before,
        ]
    )
    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["outcome"] == "activated_unconfirmed"
    assert printed["pointer_sha256"] == _pointer_sha(workspace)
    assert printed["pointer_sha256"] != before
    assert "noop" in str(printed["recovery"])
    assert load_active(workspace) == faq
    assert load_active(workspace) != baseline
    with pytest.raises(MaintenanceError, match="active pointer does not match"):
        activate(workspace, faq, expected_previous=before)
    assert _pointer_sha(workspace) == printed["pointer_sha256"]
    assert (
        activate(workspace, faq, expected_previous=printed["pointer_sha256"]) == "noop"
    )


def test_replace_failure_keeps_the_old_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path)
    baseline = _activate_sources(tmp_path / "base", workspace, BASELINE)
    faq = _build_evaluated(workspace, FAQ)
    _approve_release(tmp_path / "faq", workspace, faq)
    before = _pointer_sha(workspace)
    real_replace = os.replace

    def fail(src: str, dst: str, *args: object, **kwargs: object) -> None:
        if Path(dst).name == "active.json":
            raise OSError("injected replace failure")
        real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr("customer_support_rag.maintenance.release.os.replace", fail)
    with pytest.raises(MaintenanceError, match="could not be written"):
        activate(workspace, faq, expected_previous=before)
    assert _pointer_sha(workspace) == before
    assert load_active(workspace) == baseline


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


def test_vertical_mock_ask_hands_off_for_the_faqs_human_action(
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
        if not any("停止充電" in str(item.get("text")) for item in rows):
            return httpx.Response(200, json=_envelope(_handoff("no_evidence")))
        return httpx.Response(200, json=_envelope(_draft("合成草稿。", data)))

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", luna)
    baseline = _activate_sources(tmp_path / "base-approval", workspace, BASELINE)
    client = _ask_client(tmp_path, workspace)
    before = client.post("/ask", json={"product_id": "hp-a", "question": question})
    assert before.status_code == 200
    assert before.json()["status"] == "handoff"
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
    assert body["status"] == "handoff"
    assert body["answer"] is None
    faq_snapshot = candidate_snapshot(workspace, faq)
    with EmbeddingStore.open_readonly(faq_snapshot.database) as store:
        assert any(
            (stored := store.get_document(key)) is not None
            and stored.document.metadata.document_id == "hp-a-charging-faq"
            for key in faq_snapshot.artifact_keys
        )
    assert len(evidence) == 1
    assert MOCK_EMBED_DIGEST != PUBLISHED_DIGEST


def test_default_cli_embed_stays_offline_and_ollama_is_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured: dict[str, object] = {}

    def fake_build(*args: object, **kwargs: object):
        captured["embedder"] = kwargs.get(
            "embedder", args[3] if len(args) > 3 else None
        )
        return BuildResult("ab" * 32, False, 0, 0)

    monkeypatch.setattr(
        "customer_support_rag.maintenance.cli.build_candidate", fake_build
    )
    assert (
        main(
            [
                "build",
                "--workspace",
                str(tmp_path),
                "--root",
                str(KNOWLEDGE),
                "--source",
                BASELINE[0],
            ]
        )
        == 0
    )
    assert captured["embedder"] is None
    assert (
        main(
            [
                "build",
                "--workspace",
                str(tmp_path),
                "--root",
                str(KNOWLEDGE),
                "--source",
                BASELINE[0],
                "--embed",
                "ollama",
            ]
        )
        == 0
    )
    live = captured["embedder"]
    assert isinstance(live, OllamaEmbedder)
    assert live.transport is None
    assert live.calls == 0
    capsys.readouterr()

    def fake_evaluate(*_args: object, **kwargs: object) -> Path:
        captured["live_query"] = kwargs.get("live_query")
        return tmp_path / "regression.json"

    monkeypatch.setattr(
        "customer_support_rag.maintenance.cli.evaluate_release", fake_evaluate
    )
    assert (
        main(
            [
                "evaluate",
                "--workspace",
                str(tmp_path),
                "--root",
                str(KNOWLEDGE),
                "--release",
                "ab" * 32,
                "--suite",
                str(SUITE),
                "--code-sha",
                CODE,
            ]
        )
        == 0
    )
    assert captured["live_query"] is False
    assert (
        main(
            [
                "evaluate",
                "--workspace",
                str(tmp_path),
                "--root",
                str(KNOWLEDGE),
                "--release",
                "ab" * 32,
                "--suite",
                str(SUITE),
                "--code-sha",
                CODE,
                "--embed",
                "ollama",
            ]
        )
        == 0
    )
    assert captured["live_query"] is True


def test_ollama_identity_does_not_reuse_mock_vectors(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    _approve(workspace, BASELINE)
    mocked = build_candidate(workspace, KNOWLEDGE, BASELINE)
    requests: list[httpx.Request] = []
    embedder = OllamaEmbedder(_published_transport(requests))
    built = build_candidate(workspace, KNOWLEDGE, BASELINE, embedder)
    assert built.release_id != mocked.release_id
    assert built.copied == 0
    assert built.embedded == 2
    assert embedder.calls == 2
    mock_manifest = _manifest_payload(workspace, mocked.release_id)
    real_manifest = _manifest_payload(workspace, built.release_id)
    assert mock_manifest["transport"] == "mock"
    assert real_manifest["transport"] == "ollama"
    assert _embedding_digest(mock_manifest) == MOCK_EMBED_DIGEST
    assert _embedding_digest(real_manifest) == PUBLISHED_MODEL_DIGEST
    posts = [item for item in requests if item.url.path == "/api/embed"]
    assert len(posts) == 2
    assert all(str(item.url) == OLLAMA_EMBED_URL for item in posts)
    body = json.loads(posts[0].content)
    assert body["model"] == EMBEDDING_MODEL
    assert body["truncate"] is False
    assert body["keep_alive"] == 0
    again = build_candidate(workspace, KNOWLEDGE, BASELINE, embedder)
    assert again.noop is True
    assert again.release_id == built.release_id
    assert embedder.calls == 2
    assert len([item for item in requests if item.url.path == "/api/embed"]) == 2


def test_ollama_faq_reuses_only_the_published_documents(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    _approve(workspace, FAQ)
    baseline = build_candidate(
        workspace, KNOWLEDGE, BASELINE, OllamaEmbedder(_published_transport([]))
    )
    requests: list[httpx.Request] = []
    embedder = OllamaEmbedder(_published_transport(requests))
    faq = build_candidate(workspace, KNOWLEDGE, FAQ, embedder)
    assert faq.release_id != baseline.release_id
    assert faq.copied == 2
    assert faq.embedded == 1
    assert embedder.calls == 1
    assert len([item for item in requests if item.url.path == "/api/embed"]) == 1


def test_ollama_digest_timeout_and_partial_stop_without_retry(tmp_path: Path) -> None:
    mismatch = _workspace(tmp_path / "mismatch")
    _approve(mismatch, BASELINE)
    mismatch_requests: list[httpx.Request] = []
    mismatch_embedder = OllamaEmbedder(
        _published_transport(mismatch_requests, fail="mismatch")
    )
    with pytest.raises(MaintenanceError, match="embedding failed"):
        build_candidate(mismatch, KNOWLEDGE, BASELINE, mismatch_embedder)
    assert mismatch_embedder.calls == 0
    assert [item.url.path for item in mismatch_requests] == ["/api/tags"]
    assert list((mismatch / "releases").glob(".partial-*")) == []

    timed_out = _workspace(tmp_path / "timeout")
    _approve(timed_out, BASELINE)
    timeout_requests: list[httpx.Request] = []
    timeout_embedder = OllamaEmbedder(
        _published_transport(timeout_requests, fail="timeout")
    )
    with pytest.raises(MaintenanceError, match="embedding failed"):
        build_candidate(timed_out, KNOWLEDGE, BASELINE, timeout_embedder)
    assert timeout_embedder.calls == 0
    assert [item.url.path for item in timeout_requests] == ["/api/tags", "/api/embed"]
    assert list((timed_out / "releases").glob(".partial-*")) == []

    stopped = _workspace(tmp_path / "second")
    _approve(stopped, BASELINE)
    stopped_requests: list[httpx.Request] = []
    stopped_embedder = OllamaEmbedder(
        _published_transport(stopped_requests, fail="second")
    )
    with pytest.raises(MaintenanceError, match="embedding failed"):
        build_candidate(stopped, KNOWLEDGE, BASELINE, stopped_embedder)
    assert stopped_embedder.calls == 1
    assert (
        len([item for item in stopped_requests if item.url.path == "/api/embed"]) == 2
    )
    assert list(stopped.rglob("knowledge.sqlite")) == []

    discarded = _workspace(tmp_path / "after")
    _approve(discarded, BASELINE)
    after_requests: list[httpx.Request] = []
    after_embedder = OllamaEmbedder(_published_transport(after_requests, fail="after"))
    with pytest.raises(MaintenanceError, match="embedding failed"):
        build_candidate(discarded, KNOWLEDGE, BASELINE, after_embedder)
    assert after_embedder.calls == 0
    assert [item.url.path for item in after_requests] == [
        "/api/tags",
        "/api/embed",
        "/api/tags",
    ]
    assert list(discarded.rglob("knowledge.sqlite")) == []

    conflict = _workspace(tmp_path / "conflict")
    _approve(conflict, (*POLICY_V1, POLICY_V2[0]))
    conflict_embedder = OllamaEmbedder(_published_transport([]))
    with pytest.raises(MaintenanceError, match="document version conflict"):
        build_candidate(
            conflict,
            KNOWLEDGE,
            ("drafts/hp-a/return-policy-v1.md", "drafts/hp-a/return-policy-v2.md"),
            conflict_embedder,
        )
    assert conflict_embedder.calls == 0
    assert not (conflict / "active.json").exists()


def test_ollama_evaluate_stays_offline_and_generation_stays_unverified(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    _approve(workspace, BASELINE)
    mocked = build_candidate(workspace, KNOWLEDGE, BASELINE)
    built = build_candidate(
        workspace, KNOWLEDGE, BASELINE, OllamaEmbedder(_published_transport([]))
    )
    with pytest.raises(MaintenanceError, match="offline"):
        evaluate_release(workspace, KNOWLEDGE, built.release_id, SUITE, code_sha=CODE)
    assert not (workspace / "releases" / built.release_id / "regression.json").exists()
    report_path = evaluate_release(
        workspace,
        KNOWLEDGE,
        built.release_id,
        SUITE,
        code_sha=CODE,
        query_transport=_published_transport([]),
    )
    report = load_object(report_path, report_path.read_bytes())
    assert report["transport"] == "ollama"
    assert report["model_digest"] == PUBLISHED_MODEL_DIGEST
    assert report["embedding"] == embedding_identity(PUBLISHED_MODEL_DIGEST)
    assert report["query_transport"] == "mock"
    assert report["mock_is_not_generation_evidence"] is True
    cases = report["cases"]
    assert isinstance(cases, list)
    assert {item["result"] for item in cases if item["type"] == "generation"} == {
        "unverified"
    }
    assert all(item["model_called"] is False for item in cases)
    normal = next(item for item in cases if item["id"] == "normal-care")
    assert normal["result"] == "pass"
    selected, label = _select_query_transport(
        candidate_snapshot(workspace, built.release_id), None, True
    )
    assert selected is None
    assert label == "ollama"
    with pytest.raises(MaintenanceError, match="another embedder"):
        _select_query_transport(
            candidate_snapshot(workspace, mocked.release_id), None, True
        )


def test_rewritten_mock_manifest_cannot_claim_the_published_digest(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    _approve(workspace, BASELINE)
    mocked = build_candidate(workspace, KNOWLEDGE, BASELINE)
    directory = workspace / "releases" / mocked.release_id
    manifest_path = directory / "manifest.json"
    manifest = load_object(manifest_path, manifest_path.read_bytes())
    manifest["transport"] = "ollama"
    embedding = manifest["embedding"]
    assert isinstance(embedding, dict)
    embedding["model_digest"] = PUBLISHED_MODEL_DIGEST
    database = manifest["database"]
    assert isinstance(database, dict)
    database["sha256"] = sha256_file(directory / "knowledge.sqlite")
    manifest_path.write_text(human_json(manifest), encoding="utf-8")
    with pytest.raises(MaintenanceError, match="unavailable"):
        candidate_snapshot(workspace, mocked.release_id)


def test_manifest_identity_tamper_is_rejected_at_every_entry(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    previous = _activate_sources(tmp_path / "mock-review", workspace, BASELINE)
    before = _pointer_sha(workspace)
    mock_again = build_candidate(workspace, KNOWLEDGE, BASELINE)
    assert mock_again.noop is True
    assert mock_again.release_id == previous
    assert mock_again.embedded == 0
    assert mock_again.copied == 0

    requests: list[httpx.Request] = []
    embedder = OllamaEmbedder(_published_transport(requests))
    built = build_candidate(workspace, KNOWLEDGE, BASELINE, embedder)
    assert built.copied == 0
    assert built.embedded == 2
    manifest_path = workspace / "releases" / built.release_id / "manifest.json"
    original = manifest_path.read_bytes()
    fields: dict[str, object] = {
        "model": "other-model",
        "dimensions": 7,
        "input_profile": "other-input",
        "chunking_profile": "other-chunks",
        "max_chars": 17,
        "query_profile": "other-query",
    }
    for name, value in fields.items():
        _rewrite_embedding(manifest_path, **{name: value})
        _assert_manifest_identity_rejected(workspace, built.release_id, embedder)
        manifest_path.write_bytes(original)

    payload = load_object(manifest_path, original)
    payload["transport"] = "mock"
    embedding = payload["embedding"]
    assert isinstance(embedding, dict)
    embedding["model_digest"] = MOCK_EMBED_DIGEST
    manifest_path.write_text(human_json(payload), encoding="utf-8")
    _assert_manifest_identity_rejected(workspace, built.release_id, embedder)
    manifest_path.write_bytes(original)
    assert embedder.calls == 2
    assert len([item for item in requests if item.url.path == "/api/embed"]) == 2
    again = build_candidate(workspace, KNOWLEDGE, BASELINE, embedder)
    assert again.noop is True
    assert again.embedded == 0
    assert len([item for item in requests if item.url.path == "/api/embed"]) == 2

    marcus = {
        "model": "other-model",
        "dimensions": 7,
        "input_profile": "other-input",
        "chunking_profile": "other-chunks",
        "max_chars": 17,
    }
    _rewrite_embedding(manifest_path, **marcus)
    rewritten = load_object(manifest_path, manifest_path.read_bytes())
    rewritten_embedding = rewritten["embedding"]
    assert isinstance(rewritten_embedding, dict)
    assert rewritten["transport"] == "ollama"
    assert rewritten_embedding["model_digest"] == PUBLISHED_MODEL_DIGEST
    _assert_stored_payload_unchanged(workspace, built.release_id)
    _assert_manifest_identity_rejected(workspace, built.release_id, embedder)
    assert embedder.calls == 2
    _approve(workspace, FAQ)
    reused = build_candidate(
        workspace, KNOWLEDGE, FAQ, OllamaEmbedder(_published_transport([]))
    )
    assert reused.release_id != built.release_id
    assert reused.copied == 0
    assert reused.embedded == 3
    manifest_path.write_bytes(original)

    evaluate_release(
        workspace,
        KNOWLEDGE,
        built.release_id,
        SUITE,
        code_sha=CODE,
        query_transport=_published_transport([]),
    )
    combined = dict(marcus)
    _rewrite_embedding(manifest_path, **combined)
    with pytest.raises(MaintenanceError, match="unavailable"):
        _approve_release(tmp_path / "blocked", workspace, built.release_id)
    manifest_path.write_bytes(original)
    _approve_release(tmp_path / "ollama-review", workspace, built.release_id)
    _rewrite_embedding(manifest_path, **combined)
    _patch_consumed_approval(
        workspace,
        built.release_id,
        manifest_sha256=sha256_file(manifest_path),
    )
    with pytest.raises(MaintenanceError, match="unavailable"):
        activate(workspace, built.release_id, expected_previous=before)
    assert _pointer_sha(workspace) == before
    assert load_active(workspace) == previous
    assert list((workspace / "releases").glob(".partial-*")) == []


def test_regression_identity_cannot_swap_mock_and_ollama(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    previous = _activate_sources(tmp_path / "mock-review", workspace, BASELINE)
    requests: list[httpx.Request] = []
    embedder = OllamaEmbedder(_published_transport(requests))
    built = build_candidate(workspace, KNOWLEDGE, BASELINE, embedder)
    report_path = evaluate_release(
        workspace,
        KNOWLEDGE,
        built.release_id,
        SUITE,
        code_sha=CODE,
        query_transport=_published_transport([]),
    )
    original = report_path.read_bytes()
    report = load_object(report_path, original)
    snapshot = candidate_snapshot(workspace, built.release_id)
    assert report["query_transport"] == "mock"
    assert report["embedding"] == embedding_identity(PUBLISHED_MODEL_DIGEST)
    assert regression_matches_candidate(
        report,
        release_id=built.release_id,
        manifest_sha256=snapshot.manifest_sha256,
        database_sha256=snapshot.database_sha256,
        transport=snapshot.transport,
        model_digest=snapshot.model_digest,
    )
    _reject_swapped_report(workspace, built.release_id, report_path, original, False)
    swapped = load_object(report_path, report_path.read_bytes())
    swapped_embedding = swapped["embedding"]
    assert swapped["transport"] == "mock"
    assert swapped["model_digest"] == MOCK_EMBED_DIGEST
    assert swapped["query_transport"] == "mock"
    assert isinstance(swapped_embedding, dict)
    assert swapped_embedding["model_digest"] == PUBLISHED_MODEL_DIGEST
    stayed = candidate_snapshot(workspace, built.release_id)
    assert stayed.transport == "ollama"
    assert stayed.model_digest == PUBLISHED_MODEL_DIGEST
    _reject_swapped_report(workspace, built.release_id, report_path, original, True)
    _reject_report_embedding_lie(workspace, built.release_id, report_path, original)
    report_path.write_bytes(original)
    _approve_release(tmp_path / "ollama-review", workspace, built.release_id)
    current = _pointer_sha(workspace)
    assert activate(workspace, built.release_id, expected_previous=current) == (
        "activated"
    )
    moved = _pointer_sha(workspace)
    assert load_active(workspace) == built.release_id
    again = build_candidate(workspace, KNOWLEDGE, BASELINE, embedder)
    assert again.noop is True
    assert again.embedded == 0
    assert again.copied == 0
    assert len([item for item in requests if item.url.path == "/api/embed"]) == 2

    _reject_swapped_report(workspace, built.release_id, report_path, original, False)
    _patch_consumed_approval(
        workspace,
        built.release_id,
        regression_sha256=sha256_file(report_path),
    )
    with pytest.raises(
        MaintenanceError, match="regression does not match the candidate"
    ):
        activate(workspace, built.release_id, expected_previous=moved)
    assert _pointer_sha(workspace) == moved
    assert load_active(workspace) == built.release_id
    stayed_after = candidate_snapshot(workspace, built.release_id)
    assert stayed_after.transport == "ollama"
    assert stayed_after.model_digest == PUBLISHED_MODEL_DIGEST

    mock_path = workspace / "releases" / previous / "regression.json"
    mock_report = load_object(mock_path, mock_path.read_bytes())
    mock_report["transport"] = "ollama"
    mock_report["model_digest"] = PUBLISHED_MODEL_DIGEST
    mock_report["query_transport"] = "mock"
    mock_report["embedding"] = embedding_identity(PUBLISHED_MODEL_DIGEST)
    mock_path.write_text(human_json(mock_report), encoding="utf-8")
    require_mock_regression(load_object(mock_path, mock_path.read_bytes()))
    with pytest.raises(
        MaintenanceError, match="regression does not match the candidate"
    ):
        release_hashes(workspace, previous)
    _patch_consumed_approval(
        workspace,
        previous,
        regression_sha256=sha256_file(mock_path),
    )
    with pytest.raises(
        MaintenanceError, match="regression does not match the candidate"
    ):
        activate(workspace, previous, expected_previous=moved)
    assert _pointer_sha(workspace) == moved
    assert load_active(workspace) == built.release_id


def test_failed_embed_keeps_the_old_pointer(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    previous = _activate_sources(tmp_path / "mock-review", workspace, BASELINE)
    before = _pointer_sha(workspace)
    _approve(workspace, FAQ)
    requests: list[httpx.Request] = []
    embedder = OllamaEmbedder(_published_transport(requests, fail="timeout"))
    with pytest.raises(MaintenanceError, match="embedding failed"):
        build_candidate(workspace, KNOWLEDGE, FAQ, embedder)
    assert embedder.calls == 0
    assert [item.url.path for item in requests] == ["/api/tags", "/api/embed"]
    assert _pointer_sha(workspace) == before
    assert load_active(workspace) == previous
    assert list((workspace / "releases").glob(".partial-*")) == []
    databases = [path for path in workspace.rglob("knowledge.sqlite") if path.is_file()]
    assert [path.parent.name for path in databases] == [previous]


def test_ollama_release_keeps_the_published_digest_at_ask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path)
    calls = {"luna": 0}

    async def luna(_self: object, request: httpx.Request) -> httpx.Response:
        calls["luna"] += 1
        data = _model_data(request)
        rows = data["evidence"]
        assert isinstance(rows, list)
        return httpx.Response(200, json=_envelope(_draft("合成草稿。", data)))

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", luna)
    release_id = _build_evaluated_ollama(workspace, BASELINE)
    _approve_release(tmp_path / "approval", workspace, release_id)
    assert activate(workspace, release_id, expected_previous=None) == "activated"
    assert candidate_snapshot(workspace, release_id).model_digest == PUBLISHED_DIGEST
    client = _ask_client(tmp_path, workspace, _published_transport([]))
    answered = client.post(
        "/ask",
        json={"product_id": "hp-a", "question": "耳機可以用水洗嗎？"},
    )
    assert answered.status_code == 200
    assert "hp-a-care" in {item["document_id"] for item in answered.json()["sources"]}
    assert calls["luna"] == 1
    rejected = _ask_client(tmp_path, workspace).post(
        "/ask",
        json={"product_id": "hp-a", "question": "耳機可以用水洗嗎？"},
    )
    assert rejected.status_code == 502
    assert rejected.json()["code"] == "embedding_unavailable"
    assert calls["luna"] == 1


def _rewrite_embedding(path: Path, **fields: object) -> None:
    payload = load_object(path, path.read_bytes())
    embedding = payload["embedding"]
    assert isinstance(embedding, dict)
    embedding.update(fields)
    path.write_text(human_json(payload), encoding="utf-8")


def _patch_consumed_approval(
    workspace: Path, release_id: str, **fields: object
) -> None:
    path = workspace / "control" / "release-approvals" / f"{release_id}.json"
    payload = load_object(path, path.read_bytes())
    payload.update(fields)
    path.write_text(human_json(payload), encoding="utf-8")


def _assert_stored_payload_unchanged(workspace: Path, release_id: str) -> None:
    directory = workspace / "releases" / release_id
    manifest = load_object(
        directory / "manifest.json", (directory / "manifest.json").read_bytes()
    )
    documents = manifest["documents"]
    assert isinstance(documents, list)
    store = EmbeddingStore.open_readonly(directory / "knowledge.sqlite")
    try:
        for item in documents:
            assert isinstance(item, dict)
            loaded = store.get_document(str(item["artifact_key"]))
            assert loaded is not None
            assert loaded.embedding_model == EMBEDDING_MODEL
            assert loaded.model_digest == PUBLISHED_MODEL_DIGEST
            assert loaded.dimensions == EMBEDDING_DIMENSIONS
            assert loaded.input_profile == INPUT_PROFILE
            assert loaded.chunking_profile == CHUNKING_PROFILE
            assert loaded.max_chars == MAX_EMBED_TEXT_CHARS
    finally:
        store.close()


def _assert_manifest_identity_rejected(
    workspace: Path, release_id: str, embedder: OllamaEmbedder
) -> None:
    with pytest.raises(MaintenanceError, match="unavailable"):
        candidate_snapshot(workspace, release_id)
    with pytest.raises(MaintenanceError, match="release already exists"):
        build_candidate(workspace, KNOWLEDGE, BASELINE, embedder)
    with pytest.raises(MaintenanceError, match="unavailable"):
        evaluate_release(
            workspace,
            KNOWLEDGE,
            release_id,
            SUITE,
            code_sha=CODE,
            query_transport=_published_transport([]),
        )


def _reject_swapped_report(
    workspace: Path,
    release_id: str,
    path: Path,
    original: bytes,
    include_embedding: bool,
) -> None:
    report = load_object(path, original)
    report["transport"] = "mock"
    report["model_digest"] = MOCK_EMBED_DIGEST
    report["query_transport"] = "mock"
    if include_embedding:
        report["embedding"] = embedding_identity(MOCK_EMBED_DIGEST)
    path.write_text(human_json(report), encoding="utf-8")
    _assert_report_does_not_match(workspace, release_id, path)


def _reject_report_embedding_lie(
    workspace: Path, release_id: str, path: Path, original: bytes
) -> None:
    report = load_object(path, original)
    embedding = report["embedding"]
    assert isinstance(embedding, dict)
    embedding["model"] = "other-model"
    path.write_text(human_json(report), encoding="utf-8")
    _assert_report_does_not_match(workspace, release_id, path)
    missing = load_object(path, original)
    missing.pop("embedding")
    path.write_text(human_json(missing), encoding="utf-8")
    _assert_report_does_not_match(workspace, release_id, path)


def _assert_report_does_not_match(workspace: Path, release_id: str, path: Path) -> None:
    loaded = load_object(path, path.read_bytes())
    require_mock_regression(loaded)
    snapshot = candidate_snapshot(workspace, release_id)
    assert (
        regression_matches_candidate(
            loaded,
            release_id=release_id,
            manifest_sha256=snapshot.manifest_sha256,
            database_sha256=snapshot.database_sha256,
            transport=snapshot.transport,
            model_digest=snapshot.model_digest,
        )
        is False
    )
    with pytest.raises(
        MaintenanceError, match="regression does not match the candidate"
    ):
        release_hashes(workspace, release_id)


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


def _build_evaluated_ollama(workspace: Path, sources: tuple[str, ...]) -> str:
    _approve(workspace, sources)
    built = build_candidate(
        workspace, KNOWLEDGE, sources, OllamaEmbedder(_published_transport([]))
    )
    evaluate_release(
        workspace,
        KNOWLEDGE,
        built.release_id,
        SUITE,
        code_sha=CODE,
        query_transport=_published_transport([]),
    )
    return built.release_id


def _published_transport(
    requests: list[httpx.Request], *, fail: str | None = None
) -> httpx.MockTransport:
    seen = {"tags": 0, "posts": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/tags":
            seen["tags"] += 1
            digest = PUBLISHED_MODEL_DIGEST
            if fail == "mismatch" or (fail == "after" and seen["tags"] > 1):
                digest = "ab" * 32
            return httpx.Response(
                200,
                json={"models": [{"name": EMBEDDING_MODEL, "digest": digest}]},
            )
        if request.url.path != "/api/embed":
            return httpx.Response(404)
        seen["posts"] += 1
        if fail == "timeout" or (fail == "second" and seen["posts"] >= 2):
            raise httpx.ReadTimeout("slow")
        try:
            payload = json.loads(request.content.decode("utf-8"))
            texts = payload["input"]
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError):
            return httpx.Response(400)
        if not isinstance(texts, list) or not all(
            isinstance(item, str) for item in texts
        ):
            return httpx.Response(400)
        return httpx.Response(
            200,
            json={
                "model": EMBEDDING_MODEL,
                "embeddings": [mock_vector(text) for text in texts],
            },
        )

    return httpx.MockTransport(handler)


def _manifest_payload(workspace: Path, release_id: str) -> dict[str, object]:
    path = workspace / "releases" / release_id / "manifest.json"
    payload = load_object(path, path.read_bytes())
    assert isinstance(payload, dict)
    return payload


def _embedding_digest(manifest: dict[str, object]) -> str:
    embedding = manifest["embedding"]
    assert isinstance(embedding, dict)
    digest = embedding["model_digest"]
    assert isinstance(digest, str)
    return digest


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


def _regular_file_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        total += path.stat().st_size
    return total


def _tamper_vectors(directory: Path) -> str:
    database = directory / "knowledge.sqlite"
    connection = sqlite3.connect(database)
    rows = connection.execute(
        "select artifact_key, payload_json from embedding_documents"
    ).fetchall()
    selected = None
    for key, raw in rows:
        payload = json.loads(raw)
        for chunk in payload["chunks"]:
            if chunk["vector"] is not None:
                chunk["vector"] = [1.0] + [0.0] * 1023
        connection.execute(
            "update embedding_documents set payload_json=? where artifact_key=?",
            (
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                key,
            ),
        )
        selected = key
    connection.commit()
    connection.close()
    assert isinstance(selected, str)
    return selected


def _assert_mock_vectors(workspace: Path, release_id: str) -> None:
    directory = workspace / "releases" / release_id
    manifest = load_object(
        directory / "manifest.json", (directory / "manifest.json").read_bytes()
    )
    documents = manifest["documents"]
    assert isinstance(documents, list)
    store = EmbeddingStore.open_readonly(directory / "knowledge.sqlite")
    try:
        for item in documents:
            assert isinstance(item, dict)
            loaded = store.get_document(str(item["artifact_key"]))
            assert loaded is not None
            assert loaded.embedded_chunks
            for embedded in loaded.embedded_chunks:
                assert embedded.vector == tuple(mock_vector(embedded.chunk.text))
    finally:
        store.close()


def _vector_is_tampered(workspace: Path, release_id: str, artifact_key: str) -> bool:
    database = workspace / "releases" / release_id / "knowledge.sqlite"
    store = EmbeddingStore.open_readonly(database)
    try:
        loaded = store.get_document(artifact_key)
        assert loaded is not None
        return loaded.embedded_chunks[0].vector == (1.0,) + (0.0,) * 1023
    finally:
        store.close()


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


def _ask_client(
    tmp_path: Path, workspace: Path, transport: httpx.MockTransport | None = None
):
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
        embedding_transport=transport if transport is not None else mock_transport(),
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


def _handoff(reason: str) -> str:
    return json.dumps(
        {"status": "handoff", "answer": None, "reason": reason, "claims": []},
        ensure_ascii=False,
    )


def _draft(answer_text: str, data: dict[str, object]) -> str:
    rows = data["evidence"]
    assert isinstance(rows, list)
    answer = answer_text
    claims = []
    for item in rows:
        assert isinstance(item, dict)
        text = item["text"]
        source_id = item["source_id"]
        assert isinstance(text, str) and isinstance(source_id, str)
        claims.append(
            {
                "answer_start": 0,
                "answer_end": len(answer),
                "source_id": source_id,
                "source_start": 0,
                "source_end": len(text),
            }
        )
    return json.dumps(
        {
            "status": "answer",
            "answer": answer,
            "reason": None,
            "claims": claims,
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
