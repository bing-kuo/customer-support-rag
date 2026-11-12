"""Mock regression for one candidate. Generation cases stay unverified.

Running this module does not call Luna or Ollama. A boolean with no cases is
not a result.
"""

from __future__ import annotations

from pathlib import Path

from customer_support_rag.assist import SlotCoordinator
from customer_support_rag.knowledge.embedding import OllamaEmbeddingAdapter
from customer_support_rag.knowledge.retrieval import ExactCosineIndex
from customer_support_rag.knowledge.storage import EmbeddingStore
from customer_support_rag.maintenance.baseline import baseline_order
from customer_support_rag.maintenance.errors import MaintenanceError
from customer_support_rag.maintenance.hashes import describe_document, sha256_file
from customer_support_rag.maintenance.json_text import human_json, load_object
from customer_support_rag.maintenance.mock_embed import mock_transport
from customer_support_rag.maintenance.paths import require_directory
from customer_support_rag.maintenance.release import (
    ActiveSnapshot,
    candidate_snapshot,
    ensure_unique_documents,
    workspace_lock,
)

REQUIRED_CATEGORIES = frozenset(
    {
        "normal",
        "wrong_product",
        "version_conflict",
        "no_evidence",
        "refund_handoff",
        "insufficient_information",
        "user_prompt_injection",
        "knowledge_prompt_injection",
        "critical_safety",
    }
)
_CODE_SHA = set("0123456789abcdef")


def evaluate_release(
    workspace: Path,
    knowledge_root: Path,
    release_id: str,
    suite_path: Path,
    *,
    code_sha: str,
) -> Path:
    """Write ``regression.json`` beside the candidate database."""
    _code_sha(code_sha)
    suite = _load_suite(suite_path)
    with workspace_lock(workspace):
        snapshot = candidate_snapshot(workspace, release_id)
        cases = [
            _run_case(case, snapshot, suite, knowledge_root) for case in _cases(suite)
        ]
        report = {
            "schema": "knowledge-release-regression",
            "schema_version": 1,
            "release_id": snapshot.release_id,
            "manifest_sha256": snapshot.manifest_sha256,
            "database_sha256": snapshot.database_sha256,
            "code_sha": code_sha,
            "suite_version": suite["suite_version"],
            "rubric_version": suite["rubric_version"],
            "transport": "mock",
            "mock_is_not_generation_evidence": True,
            "cases": cases,
        }
        require_mock_regression(report)
        root = require_directory(workspace)
        from customer_support_rag.maintenance.paths import release_directory

        destination = release_directory(root, snapshot.release_id) / "regression.json"
        if destination.is_symlink():
            raise MaintenanceError("symlink is not allowed")
        encoded = human_json(report).encode("utf-8")
        if destination.is_file() and destination.read_bytes() == encoded:
            return destination
        from customer_support_rag.maintenance.release import _durable_bytes

        _durable_bytes(destination, encoded)
        return destination


def require_mock_regression(report: dict[str, object]) -> None:
    """Reject an empty run, a failing control case, or a mock generation pass."""
    cases = report.get("cases")
    if not isinstance(cases, list) or not cases:
        raise MaintenanceError("regression has no cases")
    categories: set[str] = set()
    saw_generation = False
    for case in cases:
        if not isinstance(case, dict):
            raise MaintenanceError("regression is invalid")
        category = case.get("category")
        result = case.get("result")
        kind = case.get("type")
        if not isinstance(category, str) or not isinstance(result, str):
            raise MaintenanceError("regression is invalid")
        categories.add(category)
        if result == "fail":
            raise MaintenanceError(f"regression has a failing case: {case.get('id')}")
        if kind == "generation":
            saw_generation = True
            if result != "unverified" or case.get("model_called") is not False:
                raise MaintenanceError("generation evidence is not unverified")
    if not saw_generation or not REQUIRED_CATEGORIES <= categories:
        raise MaintenanceError("regression is missing a required category")
    if report.get("mock_is_not_generation_evidence") is not True:
        raise MaintenanceError("regression is invalid")


def _run_case(
    case: dict[str, object],
    snapshot: ActiveSnapshot,
    suite: dict[str, object],
    knowledge_root: Path,
) -> dict[str, object]:
    kind = case["type"]
    base = {
        "id": case["id"],
        "category": case["category"],
        "type": kind,
    }
    if kind == "generation":
        return {
            **base,
            "result": "unverified",
            "model_called": False,
            "rubric": case["rubric"],
            "question": case["question"],
            "product_id": case["product_id"],
        }
    if kind == "build":
        return {**base, **_build_case(case, knowledge_root)}
    if kind == "assist":
        return {**base, **_assist_case(case)}
    if kind == "absent_document":
        return {**base, **_absent_case(case, snapshot)}
    if kind == "retrieval":
        return {**base, **_retrieval_case(case, snapshot, suite)}
    raise MaintenanceError("suite is invalid")


def _build_case(case: dict[str, object], knowledge_root: Path) -> dict[str, object]:
    sources = case["sources"]
    if not isinstance(sources, list):
        raise MaintenanceError("suite is invalid")
    try:
        described = tuple(
            describe_document(knowledge_root, source)
            for source in sources
            if isinstance(source, str)
        )
        if len(described) != len(sources):
            raise MaintenanceError("suite is invalid")
        ensure_unique_documents(described)
    except MaintenanceError as exc:
        if str(exc) == "document version conflict":
            return {"result": "pass", "rejected": True, "model_called": False}
        raise
    return {"result": "fail", "rejected": False, "model_called": False}


def _assist_case(case: dict[str, object]) -> dict[str, object]:
    steps = case["steps"]
    if not isinstance(steps, list) or not steps:
        raise MaintenanceError("suite is invalid")
    coordinator = SlotCoordinator(now=lambda: 0.0)
    view = None
    token: str | None = None
    for step in steps:
        if not isinstance(step, dict) or not isinstance(step.get("action"), str):
            raise MaintenanceError("suite is invalid")
        fields = {
            "product_id": step.get("product_id")
            if isinstance(step.get("product_id"), str)
            else None,
            "question": step.get("question")
            if isinstance(step.get("question"), str)
            else None,
            "steps_tried": step.get("steps_tried")
            if isinstance(step.get("steps_tried"), str)
            else None,
        }
        action = step["action"]
        if action == "start":
            view = coordinator.start(**fields)
            token = view.token
        elif action == "reset":
            if token is None:
                raise MaintenanceError("suite is invalid")
            view = coordinator.reset(token)
            token = view.token
        elif action == "continue":
            if token is None:
                raise MaintenanceError("suite is invalid")
            view = coordinator.continue_slot(token, **fields)
        else:
            raise MaintenanceError("suite is invalid")
    if view is None:
        raise MaintenanceError("suite is invalid")
    matches = view.status == case.get("expect_status") and (
        case.get("expect_reason") is None or view.reason == case.get("expect_reason")
    )
    if case.get("expect_question") is not None:
        matches = matches and view.question == case.get("expect_question")
    return {
        "result": "pass" if matches else "fail",
        "status": view.status,
        "reason": view.reason,
        "question": view.question,
        "model_called": False,
        "needs_model": view.needs_model,
    }


def _absent_case(
    case: dict[str, object], snapshot: ActiveSnapshot
) -> dict[str, object]:
    document_id = case.get("document_id")
    present = _document_ids(snapshot)
    missing = isinstance(document_id, str) and document_id not in present
    return {
        "result": "pass" if missing else "fail",
        "document_id": document_id,
        "model_called": False,
    }


def _retrieval_case(
    case: dict[str, object], snapshot: ActiveSnapshot, suite: dict[str, object]
) -> dict[str, object]:
    source_sets = suite["source_sets"]
    name = case.get("source_set")
    if not isinstance(source_sets, dict) or not isinstance(name, str):
        raise MaintenanceError("suite is invalid")
    expected_paths = source_sets.get(name)
    if not isinstance(expected_paths, list):
        raise MaintenanceError("suite is invalid")
    actual = _source_paths(snapshot)
    if actual != set(expected_paths):
        return {"result": "not_applicable", "model_called": False, "source_set": name}
    question = case.get("question")
    product = case.get("product_id")
    if not isinstance(question, str) or not isinstance(product, str):
        raise MaintenanceError("suite is invalid")
    top_k = suite.get("top_k")
    if not isinstance(top_k, int):
        raise MaintenanceError("suite is invalid")
    cosine, baseline = _rank(snapshot, question, product, top_k)
    detail = {
        "result": "pass",
        "model_called": False,
        "source_set": name,
        "cosine": cosine,
        "baseline": baseline,
    }
    expect_id = case.get("expect_document_id")
    absent_id = case.get("expect_absent_document_id")
    if isinstance(expect_id, str):
        hit = _find(cosine, expect_id, case.get("expect_version"))
        if hit is None or hit["product_id"] != case.get("expect_product_id"):
            detail["result"] = "fail"
    if isinstance(absent_id, str) and _find(cosine, absent_id, None) is not None:
        detail["result"] = "fail"
    if case.get("expect_product_id") and not isinstance(expect_id, str):
        if any(item["product_id"] != case.get("expect_product_id") for item in cosine):
            detail["result"] = "fail"
    return detail


def _rank(
    snapshot: ActiveSnapshot, question: str, product: str, top_k: int
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    store = EmbeddingStore.open_readonly(snapshot.database)
    try:
        index = ExactCosineIndex.from_store(
            store,
            artifact_keys=snapshot.artifact_keys,
            expected_model_digest=snapshot.model_digest,
        )
        rows = []
        for key in snapshot.artifact_keys:
            loaded = store.get_document(key)
            if loaded is None:
                raise MaintenanceError("candidate database is incomplete")
            for chunk in loaded.chunks:
                if chunk.metadata.product_id == product and chunk.text.strip():
                    rows.append((key, chunk))
        adapter = OllamaEmbeddingAdapter(mock_transport(snapshot.model_digest))
        try:
            query = adapter.embed_query(
                question, expected_model_digest=snapshot.model_digest
            )
            hits = index.search(query, product_id=product, top_k=top_k)
        finally:
            adapter.close()
    finally:
        store.close()
    cosine = [
        {
            "rank": index_number,
            "artifact_key": hit.artifact_key,
            "document_id": hit.chunk.metadata.document_id,
            "product_id": hit.chunk.metadata.product_id,
            "version": hit.chunk.metadata.version,
            "chunk_index": hit.chunk.index,
        }
        for index_number, hit in enumerate(hits, start=1)
    ]
    lexical = [
        {
            "rank": index_number,
            "artifact_key": key,
            "document_id": chunk.metadata.document_id,
            "product_id": chunk.metadata.product_id,
            "version": chunk.metadata.version,
            "chunk_index": chunk.index,
            "overlap": score,
        }
        for index_number, (key, chunk, score) in enumerate(
            baseline_order(question, rows, top_k=top_k), start=1
        )
    ]
    return cosine, lexical


def _find(
    hits: list[dict[str, object]], document_id: str, version: object
) -> dict[str, object] | None:
    for hit in hits:
        if hit["document_id"] != document_id:
            continue
        if version is not None and hit["version"] != version:
            continue
        return hit
    return None


def _document_ids(snapshot: ActiveSnapshot) -> set[str]:
    return {item["document_id"] for item in _manifest_documents(snapshot)}


def _source_paths(snapshot: ActiveSnapshot) -> set[str]:
    return {item["relative_path"] for item in _manifest_documents(snapshot)}


def _manifest_documents(snapshot: ActiveSnapshot) -> list[dict[str, object]]:
    manifest_path = snapshot.database.parent / "manifest.json"
    try:
        payload = load_object(manifest_path, manifest_path.read_bytes())
    except OSError:
        raise MaintenanceError("release is unavailable") from None
    if sha256_file(manifest_path) != snapshot.manifest_sha256:
        raise MaintenanceError("release is unavailable")
    documents = payload.get("documents")
    if not isinstance(documents, list):
        raise MaintenanceError("release is unavailable")
    return [item for item in documents if isinstance(item, dict)]


def _load_suite(path: Path) -> dict[str, object]:
    if not isinstance(path, Path) or path.is_symlink() or not path.is_file():
        raise MaintenanceError("suite is unavailable")
    try:
        payload = load_object(path, path.read_bytes())
    except OSError:
        raise MaintenanceError("suite is unavailable") from None
    if (
        payload.get("schema") != "knowledge-dev-suite"
        or payload.get("schema_version") != 1
        or payload.get("suite_version") != "dev-suite-v1"
        or payload.get("rubric_version") != "semantic-rubric-v1"
        or payload.get("baseline") != "character-bigram-v1"
        or payload.get("top_k") != 4
    ):
        raise MaintenanceError("suite is invalid")
    cases = payload.get("cases")
    source_sets = payload.get("source_sets")
    if not isinstance(cases, list) or not cases or not isinstance(source_sets, dict):
        raise MaintenanceError("suite is invalid")
    categories = {case.get("category") for case in cases if isinstance(case, dict)}
    if not REQUIRED_CATEGORIES <= {
        item for item in categories if isinstance(item, str)
    }:
        raise MaintenanceError("suite is invalid")
    return payload


def _cases(suite: dict[str, object]) -> list[dict[str, object]]:
    cases = suite["cases"]
    if not isinstance(cases, list):
        raise MaintenanceError("suite is invalid")
    checked: list[dict[str, object]] = []
    seen: set[str] = set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str):
            raise MaintenanceError("suite is invalid")
        if case["id"] in seen or case.get("type") not in {
            "retrieval",
            "build",
            "assist",
            "generation",
            "absent_document",
        }:
            raise MaintenanceError("suite is invalid")
        seen.add(case["id"])
        checked.append(case)
    return checked


def _code_sha(value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in _CODE_SHA for character in value)
    ):
        raise MaintenanceError("code sha is invalid")
